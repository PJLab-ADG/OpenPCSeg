'''
Implementation for Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation

Reference:
    [1] https://github.com/xinge008/Cylinder3D
'''


import torch
from torch import nn
import torch.nn.functional
import torch_scatter
import torchsparse
import torchsparse.nn.functional
import torchsparse.nn as spnn
from torchsparse import PointTensor
from pcseg.model.segmentor.base_segmentors import BaseSegmentor
from pcseg.loss import Losses
from tools.utils.common.seg_utils import voxelize

__all__ = ['Cylinder_TS']


def initial_voxelize_max(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1,
    )
    pc_hash = torchsparse.nn.functional.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = torchsparse.nn.functional.sphashquery(pc_hash, sparse_hash)
    counts = torchsparse.nn.functional.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = torchsparse.nn.functional.spvoxelize(torch.floor(new_float_coord), idx_query, counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = torch_scatter.scatter_max(z.F, idx_query, dim=0)[0]

    new_tensor = torchsparse.SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, indice_key: str = None):
    return spnn.Conv3d(
        in_planes, out_planes,
        kernel_size=1, stride=stride, bias=False,
    )

def conv1x3(in_planes: int, out_planes: int, stride: int = 1, indice_key: str = None):
    return spnn.Conv3d(
        in_planes, out_planes,
        kernel_size=(1, 3, 3), stride=stride, bias=False,
    )

def conv3x1(in_planes: int, out_planes: int, stride: int = 1, indice_key: str = None):
    return spnn.Conv3d(
        in_planes, out_planes,
        kernel_size=(3, 1, 3), stride=stride, bias=False,
    )

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, indice_key: str = None):
    return spnn.Conv3d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, bias=False,
    )

def conv1x1x3(in_planes: int, out_planes: int, stride: int = 1, indice_key: str = None):
    return spnn.Conv3d(
        in_planes, out_planes,
        kernel_size=(1, 1, 3), stride=stride, bias=False,
    )

def conv1x3x1(in_planes: int, out_planes: int, stride: int = 1, indice_key: str = None):
    return spnn.Conv3d(
        in_planes, out_planes,
        kernel_size=(1, 3, 1), stride=stride, bias=False,
    )

def conv3x1x1(in_planes: int, out_planes: int, stride: int = 1, indice_key: str = None):
    return spnn.Conv3d(
        in_planes, out_planes,
        kernel_size=(3, 1, 1), stride=stride, bias=False,
    )


class ResContextBlock(nn.Module):
    def __init__(self,
        in_filters: int,
        out_filters: int,
        kernel_size: tuple = (3, 3, 3),
        stride: int = 1,
        indice_key: str = None,
        if_dist: bool = False,
    ):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv3x1(
            out_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(
            out_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.F = self.act1(shortcut.F)
        shortcut.F = self.bn0(shortcut.F)

        shortcut = self.conv1_2(shortcut)
        shortcut.F = self.act1_2(shortcut.F)
        shortcut.F = self.bn0_2(shortcut.F)

        resA = self.conv2(x)
        resA.F = self.act2(resA.F)
        resA.F = self.bn1(resA.F)

        resA = self.conv3(resA)
        resA.F = self.act3(resA.F)
        resA.F = self.bn2(resA.F)
        resA.F = resA.F + shortcut.F

        return resA


class ResBlock(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        dropout_rate: float,
        kernel_size: tuple = (3, 3, 3),
        stride: int = 1,
        pooling: bool = True,
        drop_out: bool = True,
        height_pooling: bool = False,
        indice_key: str = None,
        if_dist: bool = False,
    ):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(
            out_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(
            out_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spnn.Conv3d(
                    out_filters, out_filters,
                    kernel_size=3, stride=2, bias=False,
                )
            else:
                self.pool = spnn.Conv3d(
                    out_filters, out_filters,
                    kernel_size=3, stride=(2, 2, 1), bias=False,
                )
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.F = self.act1(shortcut.F)
        shortcut.F = self.bn0(shortcut.F)

        shortcut = self.conv1_2(shortcut)
        shortcut.F = self.act1_2(shortcut.F)
        shortcut.F = self.bn0_2(shortcut.F)

        resA = self.conv2(x)
        resA.F = self.act2(resA.F)
        resA.F = self.bn1(resA.F)

        resA = self.conv3(resA)
        resA.F = self.act3(resA.F)
        resA.F = self.bn2(resA.F)

        resA.F = resA.F + shortcut.F

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self,
        in_filters: int,
        out_filters: int,
        kernel_size: tuple = (3, 3, 3),
        indice_key: str = None,
        up_key: str = None,
        height_pooling: bool = False,
        if_dist: bool = False,
    ):
        super(UpBlock, self).__init__()
        self.trans_dilao = conv3x3(
            in_filters, out_filters,
            indice_key=indice_key + "new_up",
        )
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(
            out_filters, out_filters,
            indice_key=indice_key,
        )
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(
            out_filters, out_filters,
            indice_key=indice_key,
        )
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(
            out_filters, out_filters,
            indice_key=indice_key,
        )
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)

        if height_pooling:
            self.up_subm = spnn.Conv3d(
                out_filters, out_filters,
                kernel_size=3, stride=2, bias=False, transposed=True,
            )
        else:
            self.up_subm = spnn.Conv3d(
                out_filters, out_filters,
                kernel_size=3, stride=(2, 2, 1), bias=False, transposed=True,
            )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA.F = self.trans_act(upA.F)
        upA.F = self.trans_bn(upA.F)

        upA = self.up_subm(upA)
        upA.F = upA.F + skip.F

        upE = self.conv1(upA)
        upE.F = self.act1(upE.F)
        upE.F = self.bn1(upE.F)

        upE = self.conv2(upE)
        upE.F = self.act2(upE.F)
        upE.F = self.bn2(upE.F)

        upE = self.conv3(upE)
        upE.F = self.act3(upE.F)
        upE.F = self.bn3(upE.F)

        return upE


class ReconBlock(nn.Module):
    def __init__(self,
        in_filters: int,
        out_filters: int,
        kernel_size: tuple = (3, 3, 3),
        stride: int = 1,
        indice_key: str = None,
        if_dist: bool = False,
    ):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.bn0 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.bn0_2 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(
            in_filters, out_filters,
            indice_key=indice_key + "bef",
        )
        self.bn0_3 = nn.SyncBatchNorm(out_filters) if if_dist else nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.F = self.bn0(shortcut.F)
        shortcut.F = self.act1(shortcut.F)

        shortcut2 = self.conv1_2(x)
        shortcut2.F = self.bn0_2(shortcut2.F)
        shortcut2.F = self.act1_2(shortcut2.F)

        shortcut3 = self.conv1_3(x)
        shortcut3.F = self.bn0_3(shortcut3.F)
        shortcut3.F = self.act1_3(shortcut3.F)
        shortcut.F = shortcut.F + shortcut2.F + shortcut3.F

        shortcut.F = shortcut.F * x.F

        return shortcut


class Cylinder_TS(BaseSegmentor):

    def __init__(
        self,
        model_cfgs,
        num_class: int = 20,
    ):
        super().__init__(model_cfgs, num_class)
        self.name = "RPV_Cylinder"
        self.in_feature_dim = model_cfgs.IN_FEATURE_DIM
        self.ignore_label = model_cfgs.IGNORE_LABEL
        self.init_size = model_cfgs.get('INIT_SIZE', 32)
        if_dist = model_cfgs.IF_DIST

        self.PPmodel = nn.Sequential(
            nn.SyncBatchNorm(self.in_feature_dim) if if_dist else nn.BatchNorm1d(self.in_feature_dim),  # fea_dim, sync
            nn.Linear(self.in_feature_dim, 64),
            nn.SyncBatchNorm(64) if if_dist else nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.SyncBatchNorm(128) if if_dist else nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.SyncBatchNorm(256) if if_dist else nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.fea_compression = nn.Sequential(
            nn.Linear(256, 16),
            nn.ReLU()
        )

        self.downCntx = ResContextBlock(16, self.init_size, indice_key="pre", if_dist=if_dist)

        self.resBlock2 = ResBlock(
            self.init_size, 2 * self.init_size, 0.2,
            height_pooling=True, indice_key="down2", if_dist=if_dist
        )
        self.resBlock3 = ResBlock(
            2 * self.init_size, 4 * self.init_size, 0.2,
            height_pooling=True, indice_key="down3", if_dist=if_dist
        )
        self.resBlock4 = ResBlock(
            4 * self.init_size, 8 * self.init_size, 0.2,
            pooling=True, height_pooling=False, indice_key="down4", if_dist=if_dist
        )
        self.resBlock5 = ResBlock(
            8 * self.init_size, 16 * self.init_size, 0.2,
            pooling=True, height_pooling=False, indice_key="down5", if_dist=if_dist
        )

        self.upBlock0 = UpBlock(
            16 * self.init_size, 16 * self.init_size,
            indice_key="up0", up_key="down5", height_pooling=False, if_dist=if_dist
        )
        self.upBlock1 = UpBlock(
            16 * self.init_size, 8 * self.init_size,
            indice_key="up1", up_key="down4", height_pooling=False, if_dist=if_dist
        )
        self.upBlock2 = UpBlock(
            8 * self.init_size, 4 * self.init_size,
            indice_key="up2", up_key="down3", height_pooling=True, if_dist=if_dist
        )
        self.upBlock3 = UpBlock(
            4 * self.init_size, 2 * self.init_size,
            indice_key="up3", up_key="down2", height_pooling=True, if_dist=if_dist
        )

        self.ReconNet = ReconBlock(
            2 * self.init_size, 2 * self.init_size,
            indice_key="recon", if_dist=if_dist
        )

        self.logits = spnn.Conv3d(
            4 * self.init_size, self.num_class,
            kernel_size=3, stride=1, bias=True,
        )

        label_smoothing = model_cfgs.get('LABEL_SMOOTHING', 0)
        self.in_channel_num = int(self.init_size * 64 / 16)
        self.point_refinement = model_cfgs.get('POINT_REFINEMENT', True)
        if self.point_refinement:
            self.change_dim = torch.nn.Sequential(
                torch.nn.Linear(self.in_channel_num, 256),  
                nn.SyncBatchNorm(256) if if_dist else nn.BatchNorm1d(256),
                nn.LeakyReLU()
            )
            self.point_logits = torch.nn.Linear(256, self.num_class)

        # loss
        default_loss_cfgs = {
            'LOSS_TYPES': ['CELoss', 'LovLoss'],
            'LOSS_WEIGHTS': [1.0, 1.0],
            'KNN': 10,
        }
        self.loss_cfgs = self.model_cfgs.get('LOSS_CONFIG', default_loss_cfgs)

        loss_types = self.loss_cfgs.get('LOSS_TYPES', default_loss_cfgs['LOSS_TYPES'])
        loss_weights = self.loss_cfgs.get('LOSS_WEIGHTS', default_loss_cfgs['LOSS_WEIGHTS'])
        k_nearest_neighbors = self.loss_cfgs.get('KNN', default_loss_cfgs['KNN'])

        self.criterion_losses = Losses(
            loss_types=loss_types, 
            loss_weights=loss_weights, 
            ignore_index=model_cfgs.IGNORE_LABEL,
            knn=k_nearest_neighbors,
            label_smoothing=label_smoothing,
        ) 

        self.loss_funs = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_label, label_smoothing=label_smoothing)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_training_loss(self, outputs, voxel_label, batch_dict):
        coords_xyz = batch_dict['voxel_coord'][:, :3].float()
        offset = batch_dict['offset']            
        loss = self.criterion_losses(outputs, voxel_label, xyz=coords_xyz, offset=offset)
        return loss

    def forward(self, batch_dict, ensemble=False):
        point_feature = batch_dict['point_feature']
        point_feature = self.PPmodel(point_feature)
        z = PointTensor(point_feature, batch_dict['point_coord'].float())
        ret = voxelize(z)
        ret.F = self.fea_compression(ret.F)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e.F = torch.cat((up0e.F, up1e.F), 1)

        logits = self.logits(up0e).F

        # point refinement
        if self.point_refinement:
            point_hash = torchsparse.nn.functional.sphash(batch_dict['point_coord'].to(logits).int())
            voxel_hash = torchsparse.nn.functional.sphash(batch_dict['voxel_coord'].to(logits).int())
            idx_query = torchsparse.nn.functional.sphashquery(point_hash, voxel_hash)
            point_feature_from_voxel = up0e.F[idx_query]
            point_feature_from_voxel = self.change_dim(point_feature_from_voxel)
            point_feature_cat = point_feature + point_feature_from_voxel
            point_logits = self.point_logits(point_feature_cat)
            loss_point = self.loss_funs(point_logits, batch_dict['point_label'])

        if self.training:
            target = batch_dict['voxel_label']
            logits_hash = torchsparse.nn.functional.sphash(up0e.C)
            voxel_hash = torchsparse.nn.functional.sphash(batch_dict['voxel_coord'].to(up0e.C))
            idx_query = torchsparse.nn.functional.sphashquery(logits_hash, voxel_hash)
            target = target[idx_query]
            loss = self.get_training_loss(logits, target, batch_dict)
            if self.point_refinement:
                ret_dict = {'loss': loss + loss_point}
                disp_dict = {'loss': loss.item(), 'loss_point': loss_point.item()}
                tb_dict = {'loss': loss.item(), 'loss_point': loss_point.item()}
            else:
                ret_dict = {'loss': loss}
                disp_dict = {'loss': loss.item()}
                tb_dict = {'loss': loss.item()}
            return ret_dict, tb_dict, disp_dict
        else:
            inverse_map = batch_dict['inverse_map']
            all_labels = batch_dict['point_label']

            point_predict = []
            point_labels = []
            point_predict_logits = []
            label_offset = 0
            for idx in range(int(batch_dict['point_coord'][:, -1].max() + 1)):
                mask_point = batch_dict['point_coord'][:, -1] == idx
                mask_logits = up0e.C[:, -1] == idx

                if ensemble:
                    out_batch_i = logits[mask_logits]
                else:
                    out_batch_i = logits[mask_logits].argmax(1)
                    out_batch_i_logits = logits[mask_logits]
                logits_hash = torchsparse.nn.functional.sphash(up0e.C[mask_logits])
                point_hash = torchsparse.nn.functional.sphash(batch_dict['point_coord'][mask_point].to(up0e.C))
                idx_query = torchsparse.nn.functional.sphashquery(point_hash, logits_hash)
                point_predict.append(out_batch_i[idx_query][:batch_dict['num_points'][idx]].cpu().numpy())
                point_labels.append(all_labels[mask_point][:batch_dict['num_points'][idx]].cpu().numpy())
                point_predict_logits.append(out_batch_i_logits[idx_query][:batch_dict['num_points'][idx]].cpu().numpy())
            
            return {'point_predict': point_predict, 'point_labels': point_labels, 'name': batch_dict['name'],'point_predict_logits': point_predict_logits}

    def forward_ensemble(self, batch_dict):
        return self.forward(batch_dict, ensemble=True)
