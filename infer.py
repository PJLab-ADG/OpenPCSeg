import argparse
import copy
import datetime
import glob
import os
from pathlib import Path
from prettytable import PrettyTable
import time
import tqdm
import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from pcseg.data import build_dataloader
from pcseg.model import build_network, load_data_to_gpu
from pcseg.optim import build_optimizer, build_scheduler
from tools.utils.common import common_utils, commu_utils
from tools.utils.train.config import cfgs, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from tools.utils.train_utils import model_state_to_cpu


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-9)


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]

    return hist


def parse_config():
    parser = argparse.ArgumentParser(description='OpenPCSeg training script version 0.1')

    # == general configs ==
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/voxel/minkunet_mk18_cr10.yaml',
                        help='specify the config for training')
    parser.add_argument('--extra_tag', type=str, default='default',
                        help='extra tag for this experiment.')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--fix_random_seed', action='store_true', default=False,
                        help='whether to fix random seed.')
    # == training configs ==
    parser.add_argument('--batch_size', type=int, default=1, required=False,
                        help='batch size for model training.')
    parser.add_argument('--epochs', type=int, default=None, required=False,
                        help='number of epochs for model training.')
    parser.add_argument('--sync_bn', action='store_true', default=False,
                        help='whether to use sync bn.')
    parser.add_argument('--ckp', type=str, default=None,
                        help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='pretrained_model')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='whether to use mixture precision training.')
    parser.add_argument('--ckp_save_interval', type=int, default=1,
                        help='number of training epochs')
    parser.add_argument('--max_ckp_save_num', type=int, default=30,
                        help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False,
                        help='')
    # == evaluation configs ==
    parser.add_argument('--eval', action='store_true', default=True,
                        help='only perform evaluate')
    parser.add_argument('--eval_interval', type=int, default=50,
                        help='number of training epochs')
    # == device configs ==
    parser.add_argument('--workers', type=int, default=5,  
                        help='number of workers for dataloader') 
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none',
                        help='')
    parser.add_argument('--tcp_port', type=int, default=18888,
                        help='tcp port for distrbuted training')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfgs)
    cfgs.TAG = Path(args.cfg_file).stem
    cfgs.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[2:-1])

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfgs)

    return args, cfgs


class Trainer:

    def __init__(self, args, cfgs):
        # set init
        log_dir, ckp_dir, logger, logger_tb, if_dist_train, total_gpus, cfgs = \
            self.init(args, cfgs)
        self.args = args
        self.cfgs = cfgs

        # set save path
        self.log_dir = log_dir
        self.ckp_dir = ckp_dir
        
        # set logger
        self.logger = logger
        self.logger_tb = logger_tb

        # set device
        self.if_amp = args.amp
        self.total_gpus = total_gpus
        self.rank = cfgs.LOCAL_RANK

        # set train config
        self.total_epoch = args.epochs
        self.if_dist_train = if_dist_train
        self.eval_interval = args.eval_interval
        self.ckp_save_interval = args.ckp_save_interval
    
        # set dataloader
        dataset, loader, sampler = build_dataloader(
            data_cfgs=cfgs.DATA,
            modality=cfgs.MODALITY,
            batch_size=cfgs.OPTIM.BATCH_SIZE_PER_GPU,
            dist=self.if_dist_train,
            root_path=cfgs.DATA.DATA_PATH,
            workers=args.workers,
            logger=logger,
            training=True,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            total_epochs=self.total_epoch,
        )
        self.train_set = dataset
        self.loader = loader
        self.sampler = sampler

        if cfgs.DATA.DATASET == 'nuscenes':
            num_class = 17
        elif cfgs.DATA.DATASET == 'semantickitti' or cfgs.DATA.DATASET == 'scribblekitti':
            num_class = 20
        elif cfgs.DATA.DATASET == 'waymo':
            num_class = 23
        
        # set model
        model = build_network(
            model_cfgs=cfgs.MODEL,
            num_class=num_class,
        )
        if args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda()
        if args.pretrained_model is not None:
            model.load_params_from_file(
                filename=args.pretrained_model,
                to_cpu=if_dist_train,
                logger=logger
            )

        # set optimizer
        self.optimizer = build_optimizer(
            model=model,
            optim_cfg=cfgs.OPTIM,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            total_iters_each_epoch=len(loader),
            total_epochs=args.epochs,
            optim_cfg=cfgs.OPTIM,
        )
        self.scaler = amp.GradScaler(enabled=self.if_amp)
        self.grad_norm_clip = cfgs.OPTIM.GRAD_NORM_CLIP

        start_epoch = it = 0
        self.it = it
        self.start_epoch = start_epoch
        self.cur_epoch = start_epoch
        self.model = model

        # -----------------------resume---------------------------
        if cfgs.LOCAL_RANK == 0:
            print('resuming...')
        if args.ckp is not None:
            self.resume(args.ckp)
        else:
            ckp_list = glob.glob(str(ckp_dir / '*checkpoint_epoch_*.pth'))
            if cfgs.LOCAL_RANK == 0:
                print('found checkpoint list:', ckp_list)
            if len(ckp_list) > 0:
                ckp_list.sort(key=os.path.getmtime)
                if cfgs.LOCAL_RANK == 0:
                    print('loading ckpt:', ckp_list[-1])
                self.resume(ckp_list[-1])

        if if_dist_train:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[cfgs.LOCAL_RANK % torch.cuda.device_count()],
            )
        self.model.train()
        
        logger.info(self.model)
        logger.info("Model parameters: {:.3f} M".format(get_n_params(self.model)/1e6))

        if cfgs.DATA.DATASET == 'nuscenes':
            self.unique_label = np.array(list(range(16)))  # 0 is ignore
        elif cfgs.DATA.DATASET == 'semantickitti' or cfgs.DATA.DATASET == 'scribblekitti':
            self.unique_label = np.array(list(range(19)))  # 0 is ignore
        elif cfgs.DATA.DATASET == 'waymo':
            self.unique_label = np.array(list(range(22)))  # 0 is ignore
        else:
            raise NotImplementedError
    
    @staticmethod
    def init(args, cfgs):
        if args.launcher == 'none':
            if_dist_train = False
            total_gpus = 1
        else:
            total_gpus, cfgs.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
                args.tcp_port, args.local_rank, backend='nccl'
            )
            if_dist_train = True

        if args.batch_size is None:
            args.batch_size = cfgs.OPTIM.BATCH_SIZE_PER_GPU
        else:
            assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
            args.batch_size = args.batch_size // total_gpus
            cfgs.OPTIM.BATCH_SIZE_PER_GPU = args.batch_size
        cfgs.OPTIM.LR = total_gpus * cfgs.OPTIM.BATCH_SIZE_PER_GPU * cfgs.OPTIM.LR_PER_SAMPLE
        args.epochs = cfgs.OPTIM.NUM_EPOCHS if args.epochs is None else args.epochs

        if args.fix_random_seed:
            common_utils.set_random_seed(42)

        log_dir = cfgs.ROOT_DIR / 'logs' / cfgs.EXP_GROUP_PATH / cfgs.TAG / args.extra_tag
        ckp_dir = log_dir / 'ckp'
        log_dir.mkdir(parents=True, exist_ok=True)
        ckp_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        logger = common_utils.create_logger(log_file, rank=cfgs.LOCAL_RANK)

        # log to file
        logger.info('**********************Start logging**********************')
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
        logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

        if if_dist_train:
            logger.info('total_batch_size: %d' % (total_gpus * cfgs.OPTIM.BATCH_SIZE_PER_GPU))
            logger.info('total_lr: %f' % cfgs.OPTIM.LR)
        
        for key, val in vars(args).items():
            logger.info('{:16} {}'.format(key, val))
        
        log_config_to_file(cfgs, logger=logger)
        if cfgs.LOCAL_RANK == 0:
            os.system('cp %s %s' % (args.cfg_file, log_dir))

        logger_tb = SummaryWriter(log_dir=str(log_dir / 'tensorboard')) if cfgs.LOCAL_RANK == 0 else None

        return log_dir, ckp_dir, logger, logger_tb, if_dist_train, total_gpus, cfgs

    def save_checkpoint(self):
        trained_epoch = self.cur_epoch + 1
        ckp_name = self.ckp_dir / ('checkpoint_epoch_%d' % trained_epoch)
        checkpoint_state = {}
        checkpoint_state['epoch'] = trained_epoch
        checkpoint_state['it'] = self.it
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(self.model.module.state_dict())
        else:
            model_state = model_state_to_cpu(self.model.state_dict())

        checkpoint_state['model_state'] = model_state
        checkpoint_state['optimizer_state'] = self.optimizer.state_dict()
        checkpoint_state['scaler_state'] = self.scaler.state_dict()
        checkpoint_state['scheduler_state'] = self.scheduler.state_dict()

        torch.save(checkpoint_state, f"{ckp_name}.pth")

    def resume(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        self.logger.info(f"==> Loading parameters from checkpoint {filename}")
        checkpoint = torch.load(filename, map_location='cpu')
        self.cur_epoch = checkpoint['epoch']
        self.start_epoch = checkpoint['epoch']
        if cfgs.LOCAL_RANK == 0:
            print('checkpoint["epoch"]:', checkpoint['epoch'])
        self.it = checkpoint['it']
        self.model.load_params(checkpoint['model_state'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        try:
            self.scaler.load_state_dict(checkpoint['scaler_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        except Exception as e:
            print('Ignore error:', e)
        
        self.logger.info('==> Done')
        return

    def train_one_epoch(self, tbar, data_cfg):
        self.model.train()
        total_it_each_epoch = len(self.loader)
        dataloader_iter = iter(self.loader)

        if self.sampler is not None:
            self.sampler.set_epoch(self.cur_epoch)

        if self.rank == 0:
            pbar = tqdm.tqdm(
                total=total_it_each_epoch,
                leave=self.cur_epoch + 1 == self.total_epoch,
                desc='train',
                dynamic_ncols=True,
            )

            data_time = common_utils.AverageMeter()
            batch_time = common_utils.AverageMeter()
            forward_time = common_utils.AverageMeter()
        
        for cur_it in range(total_it_each_epoch):

            end = time.time()
            batch = next(dataloader_iter)
            data_timer = time.time()
            cur_data_time = data_timer - end

            try:
                cur_lr = float(self.optimizer.lr)
            except:
                cur_lr = self.optimizer.param_groups[0]['lr']

            if self.logger_tb is not None:
                self.logger_tb.add_scalar('meta_data/learning_rate', cur_lr, self.it)

            self.model.train()
            self.optimizer.zero_grad()

            load_data_to_gpu(batch)

            with amp.autocast(enabled=self.if_amp):
                ret_dict, tb_dict, disp_dict = self.model(batch)
                loss = ret_dict['loss'].mean()
            
            forward_timer = time.time()
            cur_forward_time = forward_timer - data_timer

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.it += 1
            cur_batch_time = time.time() - end

            # average reduce
            avg_data_time = commu_utils.average_reduce_value(cur_data_time)
            avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
            avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

            if self.rank == 0:
                data_time.update(avg_data_time)
                forward_time.update(avg_forward_time)
                batch_time.update(avg_batch_time)
                disp_dict.update({
                    'loss': loss.item(),
                    'lr': cur_lr, 
                    'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                    'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})',
                    'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})',
                })

                pbar.update()
                pbar.set_postfix(dict(total_it=self.it))
                tbar.set_postfix(disp_dict)
                tbar.refresh()
                if self.logger_tb is not None:
                    self.logger_tb.add_scalar('train/loss', loss, self.it)
                    self.logger_tb.add_scalar('meta_data/learning_rate', cur_lr, self.it)
                    for key, val in tb_dict.items():
                        self.logger_tb.add_scalar('train/' + key, val, self.it)

        if 'Range' not in data_cfg.DATASET:
            self.loader.dataset.point_cloud_dataset.resample()  
        if self.rank == 0:
            pbar.close()

    def evaluate(self, dataloader, prefix):
        result_dir = self.log_dir / 'eval' / ('epoch_%s' % (self.cur_epoch+1))
        result_dir.mkdir(parents=True, exist_ok=True)
        dataset = dataloader.dataset

        class_names = dataset.class_names

        self.logger.info(f"*************** TRAINED EPOCH {self.cur_epoch+1} {prefix} EVALUATION *****************")
        if self.rank == 0:
            progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
        metric = {}
        metric['hist_list'] = []
        save_dir = self.cfgs.DATA.OUTPUT_DIR
        os.makedirs(save_dir, exist_ok=True) 

        for i, batch_dict in enumerate(dataloader):
            load_data_to_gpu(batch_dict)

            with torch.no_grad():
                ret_dict = self.model(batch_dict)
            
            point_predict = ret_dict['point_predict']
            point_labels = ret_dict['point_labels']
            
            save_name = (10-len(str(i))) * '0' + str(i)
            save_path = save_dir + save_name + '.npy'
            np.save(save_path, ret_dict['point_predict'][0])

            if isinstance(point_predict, torch.Tensor):
                if point_predict.size() != point_labels.size():
                    point_predict = nn.functional.softmax(point_predict, dim=1).argmax(dim=1)
                    point_predict = point_predict.detach().cpu().numpy()
                    point_labels = point_labels.detach().cpu().numpy()

            for pred, label in zip(point_predict, point_labels):
                metric['hist_list'].append(fast_hist_crop(pred, label, self.unique_label))
            
            if self.rank == 0:
                progress_bar.update()
        
        if self.rank == 0:
            progress_bar.close()

        return {}


    def train(self):

        with tqdm.trange(
            self.start_epoch, self.total_epoch, desc='epochs', dynamic_ncols=True, leave=(self.rank==0),
        ) as tbar:

            for cur_epoch in tbar:
                self.cur_epoch = cur_epoch
                self.train_one_epoch(tbar, self.cfgs.DATA)
                trained_epoch = cur_epoch + 1
                if trained_epoch % self.ckp_save_interval == 0 and self.rank == 0:
                    self.save_checkpoint()
                
                if (cur_epoch+1) % self.eval_interval == 0 or cur_epoch == self.total_epoch-1:
                    self.model.eval()
                    data_config = copy.deepcopy(self.cfgs.DATA)
                    _, test_loader, _ = build_dataloader(
                        data_cfgs=data_config,
                        modality=self.cfgs.MODALITY,
                        batch_size=self.cfgs.OPTIM.BATCH_SIZE_PER_GPU,
                        dist=self.if_dist_train,
                        workers=self.args.workers,
                        logger=self.logger,
                        training=False
                    )
                    self.evaluate(test_loader, "val")
                    if self.if_dist_train:
                        torch.distributed.barrier()
                    
                    time.sleep(1)
            
            if len(tbar) == 0:
                self.model.eval()
                data_config = copy.deepcopy(self.cfgs.DATA)
                _, test_loader, _ = build_dataloader(
                    data_cfgs=data_config,
                    batch_size=self.cfgs.OPTIM.BATCH_SIZE_PER_GPU,
                    dist=self.if_dist_train,
                    workers=self.args.workers,
                    logger=self.logger,
                    training=False
                )
                self.evaluate(test_loader, "val")

                if self.if_dist_train:
                    torch.distributed.barrier()
                
                time.sleep(1)
            

def main():
    args, cfgs = parse_config()
    trainer = Trainer(args, cfgs)

    if args.eval:
        trainer.cur_epoch -= 1
        trainer.model.eval()
        data_config = copy.deepcopy(cfgs.DATA)
        _, test_loader, _ = build_dataloader(
            data_cfgs=data_config,
            modality=cfgs.MODALITY,
            batch_size=cfgs.OPTIM.BATCH_SIZE_PER_GPU,
            dist=trainer.if_dist_train,
            workers=args.workers,
            logger=trainer.logger,
            training=False,
        )

        trainer.evaluate(test_loader, "val")
        if trainer.if_dist_train:
            torch.distributed.barrier()
        time.sleep(1)
    else:
        trainer.train()


if __name__ == '__main__':
    main()
