import glob
import math
import numpy as np
import os
import yaml
import sys

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torchvision.transforms import functional as FF
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse

from pcseg.data.dataset.semantickitti.laserscan import LaserScan


inv_label_dict = {0: 0, 1: 10, 2: 11, 3: 15, 4: 18, 5: 20, 6: 30, 7: 31, 8: 32, 9: 40, 10: 44, 11: 48, 12: 49, 13: 50, 14: 51, 15: 70, 16: 71, 17: 72, 18: 80, 19: 81}


def validate_semkitti(logger, args, model, device):
    lidar_list = glob.glob(args.root + 'sequences/' + '08' + '/velodyne/*.bin')

    scale_x = np.expand_dims(
        np.ones([args.H, args.W])*50.0, axis=-1).astype(np.float32)
    scale_y = np.expand_dims(
        np.ones([args.H, args.W])*50.0, axis=-1).astype(np.float32)
    scale_z = np.expand_dims(
        np.ones([args.H, args.W])*3.0, axis=-1).astype(np.float32)
    scale_matrx = np.concatenate([scale_x, scale_y, scale_z], axis=2)

    A = LaserScan(project=True, flip_sign=False, H=args.H, W=args.W, fov_up=3.0, fov_down=-25.0)
    knn = KNN({'knn': 5, 'search': 5, 'sigma': 1.0, 'cutoff': 1.0}, 20)

    for i in range(len(lidar_list)):

        if not(os.path.exists(args.save_path_val)):
            os.makedirs(args.save_path_val)

        path_list = lidar_list[i].split('/')
        label_file = os.path.join(args.save_path_val, path_list[-1][:len(path_list[-1])-3]) + "label"
        if os.path.exists(label_file):
            os.remove(label_file)

        A.open_scan(lidar_list[i])

        xyz       = torch.unsqueeze(FF.to_tensor(A.proj_xyz/scale_matrx), axis=0)
        intensity = torch.unsqueeze(FF.to_tensor(A.proj_remission), axis=0)
        depth     = torch.unsqueeze(FF.to_tensor(A.proj_range/80.0), axis=0)
        mask      = torch.unsqueeze(FF.to_tensor(A.proj_mask), axis=0)

        scan = torch.cat([xyz, intensity, depth, mask], axis=1).to(device)

        with torch.no_grad():
            logits = model(scan, return_loss=True)  # [cls, 32, 512]
            logits = F.interpolate(logits, size=(args.H, args.W), mode='bilinear', align_corners=True)  # [bs, cls, 64, 2048]
            logits = logits.squeeze(0)  # [cls, 64, 2048]
            argmax = torch.argmax(logits, dim=0, keepdim=True)

        if args.postprocess == 'knn':
            label = postprocess_knn(A, argmax, depth, knn, device)
        elif args.postprocess == 'fid':
            label = postprocess_fid(A, argmax, depth, device)

        label = np.asarray(label)
        label = label.astype(np.uint32)
        label.tofile(label_file)

        if i % 500 == 0:
            logger.info("'{}' have evaluated ...".format(i))


    # print summary
    logger.info("*" * 60)
    logger.info("INTERFACE:")
    logger.info("Data: {}".format(args.root))
    logger.info("Predictions: {}".format(args.save_path_val))
    logger.info("Backend: {}".format(args.backend))
    logger.info("Split: {}".format(args.split))
    logger.info("Config: {}".format(args.yaml))
    logger.info("Limit: {}".format(args.limit))
    logger.info("Codalab: {}".format(args.codalab))
    logger.info("*" * 60)

    DATA = yaml.safe_load(open(args.yaml, 'r'))

    class_strings = DATA["labels"]
    class_remap = DATA["learning_map"]
    class_inv_remap = DATA["learning_map_inv"]
    class_ignore = DATA["learning_ignore"]
    nr_classes = len(class_inv_remap)

    maxkey = max(class_remap.keys())

    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(class_remap.keys())] = list(class_remap.values())

    ignore = []
    for cl, ign in class_ignore.items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)
            logger.info("Ignoring xentropy class '{}' in IoU evaluation".format(x_cl))

    if args.backend == "torch":
        from .torch_ioueval import iouEval
        evaluator = iouEval(nr_classes, ignore)
    elif args.backend == "numpy":
        from .np_ioueval import iouEval
        evaluator = iouEval(nr_classes, ignore)

    evaluator.reset()

    test_sequences = DATA["split"][args.split]

    label_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        label_paths = os.path.join(args.root, "sequences", str(sequence), "labels")

        seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_paths)) for f in fn if ".label" in f]
        seq_label_names.sort()
        label_names.extend(seq_label_names)

    pred_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        # pred_paths = os.path.join(args.save_path_val, "sequences", sequence, "predictions")
        pred_paths = args.save_path_val
        seq_pred_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(pred_paths)) for f in fn if ".label" in f]
        seq_pred_names.sort()
        pred_names.extend(seq_pred_names)

    assert(len(label_names) == len(pred_names))

    progress = 10
    count = 0
    logger.info("Evaluating sequences: ")

    for label_file, pred_file in zip(label_names, pred_names):

        count += 1
        if 100 * count / len(label_names) > progress:
            logger.info("{:d}% ".format(progress))
            progress += 10

        label = np.fromfile(label_file, dtype=np.int32)
        label = label.reshape((-1))     # reshape to vector
        label = label & 0xFFFF          # get lower half for semantics
        if args.limit is not None:
            label = label[:args.limit]  # limit to desired length
        label = remap_lut[label]        # remap to xentropy format

        pred = np.fromfile(pred_file, dtype=np.int32)
        pred = pred.reshape((-1))       # reshape to vector
        pred = pred & 0xFFFF            # get lower half for semantics
        if args.limit is not None:
            pred = pred[:args.limit]    # limit to desired length
        pred = remap_lut[pred]          # remap to xentropy format

        # add single scan to evaluation
        evaluator.addBatch(pred, label)

    m_accuracy = evaluator.getacc()
    m_jaccard, class_jaccard = evaluator.getIoU()

    logger.info('Validation set:\n'
        'Acc avg {m_accuracy:.3f}\n'
        'IoU avg {m_jaccard:.3f}'.format(m_accuracy=m_accuracy, m_jaccard=m_jaccard))

    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            logger.info('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
                i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc)
            )

    logger.info("*" * 60)
    for i, jacc in enumerate(class_jaccard):
        if i not in ignore:
            sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
            sys.stdout.write(",")
    sys.stdout.write('{jacc:.3f}'.format(jacc=m_jaccard.item()))
    sys.stdout.write(",")
    sys.stdout.write('{acc:.3f}'.format(acc=m_accuracy.item()))
    sys.stdout.write('\n')
    sys.stdout.flush()

    # if codalab is necessary, then do it
    if args.codalab is not None:
        results = {}
        results["accuracy_mean"] = float(m_accuracy)
        results["iou_mean"] = float(m_jaccard)
        for i, jacc in enumerate(class_jaccard):
            if i not in ignore:
                results["iou_"+class_strings[class_inv_remap[i]]] = float(jacc)
        output_filename = os.path.join(args.codalab, 'scores.txt')
        with open(output_filename, 'w') as yaml_file:
            yaml.dump(results, yaml_file, default_flow_style=False)


def postprocess_knn(A, pred, depth, knn, device):
    t_1 = torch.squeeze(depth*80.0).detach().to(device)
    t_2 = torch.squeeze(FF.to_tensor(np.reshape(A.unproj_range, (1, -1)))).detach().to(device)
    t_3 = torch.squeeze(pred).detach().to(device)
    t_4 = torch.squeeze(FF.to_tensor(np.reshape(A.proj_x, (1, -1)))).to(dtype=torch.long).detach().to(device)
    t_5 = torch.squeeze(FF.to_tensor(np.reshape(A.proj_y, (1, -1)))).to(dtype=torch.long).detach().to(device)

    unproj_argmax = knn(t_1, t_2, t_3, t_4, t_5)
    unproj_argmax = unproj_argmax.detach().cpu().numpy()

    label = []
    for i in unproj_argmax:
        upper_half = 0
        lower_half = inv_label_dict[i.item()]
        label_each = (upper_half << 16) + lower_half
        label.append(label_each)
    
    return label


def postprocess_fid(A, pred, depth, device):
    t_1 = torch.squeeze(depth*80.0).detach().to(device)
    t_3 = torch.squeeze(pred).detach().to(device)

    proj_unfold_range, proj_unfold_pre = NN_filter(t_1, t_3)
    proj_unfold_range = proj_unfold_range.cpu().numpy()
    proj_unfold_pre = proj_unfold_pre.cpu().numpy()

    label = []
    for jj in range(len(A.proj_x)):
        y_range, x_range = A.proj_y[jj], A.proj_x[jj]
        upper_half = 0
        if A.unproj_range[jj] == A.proj_range[y_range, x_range]:
            lower_half = inv_label_dict[pred[y_range, x_range]]
        else:
            potential_label = proj_unfold_pre[0, :, y_range, x_range]
            potential_range = proj_unfold_range[0, :, y_range, x_range]
            min_arg = np.argmin(abs(potential_range-A.unproj_range[jj]))
            lower_half = inv_label_dict[potential_label[min_arg]]
        label_each = (upper_half << 16) + lower_half
        label.append(label_each)

    return label


def NN_filter(depth, pred, k_size=5):
    pred = pred.double()
    H, W = np.shape(depth)
    
    proj_range_expand = torch.unsqueeze(depth, axis=0)
    proj_range_expand = torch.unsqueeze(proj_range_expand, axis=0)
    
    semantic_pred_expand = torch.unsqueeze(pred, axis=0)
    semantic_pred_expand = torch.unsqueeze(semantic_pred_expand, axis=0)
    
    pad = int((k_size - 1) / 2)

    proj_unfold_range = F.unfold(proj_range_expand,kernel_size=(k_size, k_size), padding=(pad, pad))
    proj_unfold_range = proj_unfold_range.reshape(-1, k_size*k_size, H, W)
        
    proj_unfold_pre = F.unfold(semantic_pred_expand,kernel_size=(k_size, k_size), padding=(pad, pad))
    proj_unfold_pre = proj_unfold_pre.reshape(-1, k_size*k_size, H, W)
    
    return proj_unfold_range, proj_unfold_pre


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(kernel_size, kernel_size)

    return gaussian_kernel


class KNN(torch.nn.Module):
    def __init__(self, params={'knn': 5, 'search': 5, 'sigma': 1.0, 'cutoff': 1.0}, nclasses=20):
        super().__init__()
        self.knn = params["knn"]
        self.search = params["search"]
        self.sigma = params["sigma"]
        self.cutoff = params["cutoff"]
        self.nclasses = nclasses

    def forward(self, proj_range, unproj_range, proj_argmax, px, py):
        if proj_range.is_cuda: device = torch.device("cuda")
        else: device = torch.device("cpu")

        H, W = proj_range.shape
        P = unproj_range.shape

        if (self.search % 2 == 0): raise ValueError("Nearest neighbor kernel must be odd number")

        pad = int((self.search - 1) / 2)
        proj_unfold_k_rang = F.unfold(proj_range[None, None, ...], kernel_size=(self.search, self.search), padding=(pad, pad))

        idx_list = py * W + px

        unproj_unfold_k_rang = proj_unfold_k_rang[:, :, idx_list]
        unproj_unfold_k_rang[unproj_unfold_k_rang < 0] = float("inf")
        center = int(((self.search * self.search) - 1) / 2)
        unproj_unfold_k_rang[:, center, :] = unproj_range

        k2_distances = torch.abs(unproj_unfold_k_rang - unproj_range)
        inv_gauss_k = (1 - get_gaussian_kernel(self.search, self.sigma, 1)).view(1, -1, 1)
        inv_gauss_k = inv_gauss_k.to(device).type(proj_range.type())
        k2_distances = k2_distances * inv_gauss_k

        _, knn_idx = k2_distances.topk(self.knn, dim=1, largest=False, sorted=False)

        proj_unfold_1_argmax = F.unfold(proj_argmax[None, None, ...].float(), kernel_size=(self.search, self.search), padding=(pad, pad)).long()
        unproj_unfold_1_argmax = proj_unfold_1_argmax[:, :, idx_list]
        knn_argmax = torch.gather(input=unproj_unfold_1_argmax, dim=1, index=knn_idx)

        if self.cutoff > 0:
          knn_distances = torch.gather(input=k2_distances, dim=1, index=knn_idx)
          knn_invalid_idx = knn_distances > self.cutoff
          knn_argmax[knn_invalid_idx] = self.nclasses

        knn_argmax_onehot = torch.zeros((1, self.nclasses + 1, P[0]), device=device).type(proj_range.type())
        ones = torch.ones_like(knn_argmax).type(proj_range.type())
        knn_argmax_onehot = knn_argmax_onehot.scatter_add_(1, knn_argmax, ones)
        knn_argmax_out = knn_argmax_onehot[:, 1:-1].argmax(dim=1) + 1
        knn_argmax_out = knn_argmax_out.view(P)

        return knn_argmax_out


class ClassWeightSemikitti:
	@staticmethod
	def get_weight():
		return [0.0,
                1.0/(0.040818519255974316+0.001789309418528068+0.001),
                1.0/(0.00016609538710764618+0.001),
                1.0/(0.00039838616015114444+0.001),
                1.0/(0.0020633612104619787+0.00010157861367183268+0.001),
                1.0/(2.7879693665067774e-05+0.0016218197275284021+0.00011351574470342043+4.3840131989471124e-05+0.001),
                1.0/(0.00017698551338515307+0.00016059776092534436+0.001),
                1.0/(1.1065903904919655e-08+0.00012709999297008662+0.001),
                1.0/(5.532951952459828e-09+3.745553104802113e-05+0.001),
                1.0/(0.1987493871255525+4.7084144280367186e-05+0.001),
                1.0/(0.014717169549888214+0.001),
                1.0/(0.14392298360372+0.001),
                1.0/(0.0039048553037472045+0.001),
                1.0/(0.1326861944777486+0.001),
                1.0/(0.0723592229456223+0.001),
                1.0/(0.26681502148037506+0.001),
                1.0/(0.006035012012626033+0.001),
                1.0/(0.07814222006271769+0.001),
                1.0/(0.002855498193863172+0.001),
                1.0/(0.0006155958086189918+0.001)
            ]
	@staticmethod
	def get_bin_weight(bin_num):
		weight_list=[]
		for i in range(bin_num+1):
			weight_list.append(abs(i/float(bin_num)-0.5)*2+0.2)
		return weight_list


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def get_semantic_segmentation(sem):
	# map semantic output to labels
	if sem.size(0) != 1:
		raise ValueError('Only supports inference for batch size = 1')
	sem = sem.squeeze(0)
	predict_pre = torch.argmax(sem, dim=0, keepdim=True)
	'''
	sem_prob=F.softmax(sem,dim=0)
	change_mask_motorcyclist=torch.logical_and(predict_pre==7,sem_prob[8:9,:,:]>0.1)
	predict_pre[change_mask_motorcyclist]=8
	'''
	return predict_pre


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    #vprobas = probas[valid.nonzero().squeeze()]
    vprobas = probas[torch.squeeze(torch.nonzero(valid))]
    vlabels = labels[valid]
    return vprobas, vlabels


class Lovasz_softmax(torch.nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(Lovasz_softmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, self.classes, self.per_image, self.ignore)


class DiceLoss(_WeightedLoss):
    """
    This criterion is based on Dice coefficients.

    Modified version of: https://github.com/ai-med/nn-common-modules/blob/master/nn_common_modules/losses.py (MIT)
    Arxiv paper: https://arxiv.org/pdf/1606.04797.pdf
    """

    def __init__(
        self, 
        weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = 255,
        binary: bool = False,
        reduction: str = 'mean'):
        """
        :param weight:  <torch.FloatTensor: n_class>. Optional scalar weight for each class.
        :param ignore_index: Label id to ignore when calculating loss.
        :param binary: Whether we are only doing binary segmentation.
        :param reduction: Specifies the reduction to apply to the output. Can be 'none', 'mean' or 'sum'.
        """
        super().__init__(weight=weight, reduction=reduction)
        self.ignore_index = ignore_index
        self.binary = binary

    def forward(self, predictions: torch.FloatTensor, targets: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward pass.
        :param predictions: <torch.FloatTensor: n_samples, C, H, W>. Predicted scores.
        :param targets: <torch.LongTensor: n_samples, H, W>. Target labels.
        :return: <torch.FloatTensor: 1>. Scalar loss output.
        """
        self._check_dimensions(predictions=predictions, targets=targets)
        predictions = F.softmax(predictions, dim=1)
        if self.binary:
            return self._dice_loss_binary(predictions, targets)
        return self._dice_loss_multichannel(predictions, targets, self.weight, self.ignore_index)

    @staticmethod
    def _dice_loss_binary(predictions: torch.FloatTensor, targets: torch.LongTensor) -> torch.FloatTensor:
        """
        Dice loss for one channel binarized input.
        :param predictions: <torch.FloatTensor: n_samples, 1, H, W>. Predicted scores.
        :param targets: <torch.LongTensor: n_samples, H, W>. Target labels.
        :return: <torch.FloatTensor: 1>. Scalar loss output.
        """
        eps = 0.0001

        assert predictions.size(1) == 1, 'predictions should have a class size of 1 when doing binary dice loss.'

        intersection = predictions * targets

        # Summed over batch, height and width.
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = predictions + targets
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = 1 - (numerator / denominator)

        # Averaged by classes.
        return loss_per_channel.sum() / predictions.size(1)

    @staticmethod
    def _dice_loss_multichannel(predictions: torch.FloatTensor,
                                targets: torch.LongTensor,
                                weight: Optional[torch.FloatTensor] = None,
                                ignore_index: int = -100) -> torch.FloatTensor:
        """
        Calculate the loss for multichannel predictions.
        :param predictions: <torch.FloatTensor: n_samples, n_class, H, W>. Predicted scores.
        :param targets: <torch.LongTensor: n_samples, H, W>. Target labels.
        :param weight:  <torch.FloatTensor: n_class>. Optional scalar weight for each class.
        :param ignore_index: Label id to ignore when calculating loss.
        :return: <torch.FloatTensor: 1>. Scalar loss output.
        """
        eps = 0.0001
        encoded_target = predictions.detach() * 0

        mask = targets == ignore_index
        targets = targets.clone()
        targets[mask] = 0
        encoded_target.scatter_(1, targets.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0

        intersection = predictions * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = predictions + encoded_target

        denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1)
        if denominator.sum() == 0:
            # Means only void gradients. Summing the denominator would lead to loss of 0.
            return denominator.sum()
        denominator = denominator + eps

        if weight is None:
            weight = 1
        else:
            # We need to ensure that the weights and the loss terms resides in the same device id.
            # Especially crucial when we are using DataParallel/DistributedDataParallel.
            weight = weight / weight.mean()

        loss_per_channel = weight * (1 - (numerator / denominator))

        # Averaged by classes.
        return loss_per_channel.sum() / predictions.size(1)

    def _check_dimensions(self, predictions: torch.FloatTensor, targets: torch.LongTensor) -> None:
        error_message = ""
        if predictions.size(0) != targets.size(0):
            error_message += f'Predictions and targets should have the same batch size, but predictions have batch ' f'size {predictions.size(0)} and targets have batch size {targets.size(0)}.\n'
        if self.weight is not None and self.weight.size(0) != predictions.size(1):
            error_message += f'Weights and the second dimension of predictions should have the same dimensions ' f'equal to the number of classes, but weights has dimension {self.weight.size()} and ' f'targets has dimension {targets.size()}.\n'
        if self.binary and predictions.size(1) != 1:
            error_message += f'Binary class should have one channel representing the number of classes along the ' f'second dimension of the predictions, but the actual dimensions of the predictions ' f'is {predictions.size()}\n'
        if not self.binary and predictions.size(1) == 1:
            error_message += f'Predictions has dimension {predictions.size()}. The 2nd dimension equal to 1 ' f'indicates that this is binary, but binary was set to {self.binary} by construction\n'
        if error_message:
            raise ValueError(error_message)


class CrossEntropyDiceLoss(nn.Module):
    """ This is the combination of Cross Entropy and Dice Loss. """

    def __init__(self, reduction: str = 'mean', ignore_index: int = -100, weight: torch.Tensor = None):
        """
        :param reduction: Specifies the reduction to apply to the output. Can be 'none', 'mean' or 'sum'.
        :param ignore_index: Label id to ignore when calculating loss.
        :param weight:  <torch.FloatTensor: n_class>. Optional scalar weight for each class.
        """
        super(CrossEntropyDiceLoss, self).__init__()
        self.dice = DiceLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Calculate the loss.
        :param predictions: <torch.FloatTensor: n_samples, n_class, H, W>. Predicted scores.
        :param targets: <torch.LongTensor: n_samples, H, W>. Target labels.
        :return: <torch.FloatTensor: 1>. Scalar loss output.
        """
        ce_loss = self.cross_entropy(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return ce_loss + dice_loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss proposed in:
        Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
        https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax), shape (N, C, H, W)
            - gt: ground truth map, shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """
        n, c, _, _ = pred.shape

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt


        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    divce = label.device
    one_hot_label = torch.eye(
        n_classes, device=divce, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label


class MixTeacherNusc:
    
    def __init__(self, strategy,):
        super(MixTeacherNusc, self).__init__()

        if strategy == 'mixture':
            self.strategies = [
                'col1row2', 'col1row3', 'col2row1', 'col3row1', 'col2row2', 'col1row4', 'col2row4'
            ]
        elif strategy == 'mixtureV2':
            self.strategies = [
                'col1row3', 'col1row4', 'col1row5', 'col1row6', 
                'col2row3', 'col2row4', 'col2row5', 'col2row6', 
                'col3row3', 'col3row4', 'col3row5', 'col3row6', 
                'col4row3', 'col4row4', 'col4row5', 'col4row6', 
                'col6row4',
            ]

    def forward(self, image, label, image_aux, label_aux):
        """
        Arguments:
            - strategy: RangeAug strategies.
            - image: original image, size: [bs, 6, h, w].
            - label: original label, size: [bs, h, w].
            - image_aux: auxiliary image, size: [bs, 6, h, w].
            - label_aux: auxiliary label, size: [bs, h, w].
        Return:
            (2x) Augmented images and labels.
        """
        
        bs, bs_aux = image.size()[0], image_aux.size()[0]
        
        if bs != bs_aux:
            if bs > bs_aux:
                image, label = image[:bs_aux, :, :, :], label[:bs_aux, :, :]
            else:
                image_aux, label_aux = image_aux[:bs, :, :, :], label_aux[:bs, :, :]

        strategy = np.random.choice(self.strategies, size=1)[0]

        if strategy == 'col1row2':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col1row2(image, label, image_aux, label_aux)

        elif strategy == 'col1row3':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col1row3(image, label, image_aux, label_aux)

        elif strategy == 'col1row4':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col1row4(image, label, image_aux, label_aux)

        elif strategy == 'col1row5':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col1row5(image, label, image_aux, label_aux)

        elif strategy == 'col1row6':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col1row6(image, label, image_aux, label_aux)

        elif strategy == 'col1row8':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col1row8(image, label, image_aux, label_aux)

        elif strategy == 'col1row16':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col1row16(image, label, image_aux, label_aux)

        elif strategy == 'col2row1':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col2row1(image, label, image_aux, label_aux)

        elif strategy == 'col2row2':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col2row2(image, label, image_aux, label_aux)

        elif strategy == 'col2row3':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col2row3(image, label, image_aux, label_aux)

        elif strategy == 'col2row4':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col2row4(image, label, image_aux, label_aux)

        elif strategy == 'col2row5':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col2row5(image, label, image_aux, label_aux)

        elif strategy == 'col2row6':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col2row6(image, label, image_aux, label_aux)

        elif strategy == 'col3row1':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col3row1(image, label, image_aux, label_aux)

        elif strategy == 'col3row2':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col3row2(image, label, image_aux, label_aux)
        
        elif strategy == 'col3row3':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col3row3(image, label, image_aux, label_aux)

        elif strategy == 'col3row4':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col3row4(image, label, image_aux, label_aux)

        elif strategy == 'col3row5':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col3row5(image, label, image_aux, label_aux)

        elif strategy == 'col3row6':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col3row6(image, label, image_aux, label_aux)

        elif strategy == 'col4row1':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col4row1(image, label, image_aux, label_aux)

        elif strategy == 'col4row2':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col4row2(image, label, image_aux, label_aux)

        elif strategy == 'col4row3':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col4row3(image, label, image_aux, label_aux)

        elif strategy == 'col4row4':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col4row4(image, label, image_aux, label_aux)

        elif strategy == 'col4row5':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col4row5(image, label, image_aux, label_aux)

        elif strategy == 'col4row6':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col4row6(image, label, image_aux, label_aux)

        elif strategy == 'col6row4':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col6row4(image, label, image_aux, label_aux)

        elif strategy == 'col16row1':
            img_aux1, lbl_aux1, img_aux2, lbl_aux2 = self.col16row1(image, label, image_aux, label_aux)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2, strategy


    def col1row3(self, img, lbl, img_aux, lbl_aux):
        h1 = int(img.size()[-2] / 3)  # 32/3 = 10
        h2 = 2 * h1                   # 10*2 = 20

        imgA_1, imgA_2, imgA_3 = img[:, :h1, :], img[:, h1:h2, :], img[:, h2:, :]  # upper1, middle1, lower1
        lblA_1, lblA_2, lblA_3 = lbl[   :h1, :], lbl[   h1:h2, :], lbl[   h2:, :]  # upper1, middle1, lower1

        imgB_1, imgB_2, imgB_3 = img_aux[:, :h1, :], img_aux[:, h1:h2, :], img_aux[:, h2:, :]  # upper2, middle2, lower2
        lblB_1, lblB_2, lblB_3 = lbl_aux[   :h1, :], lbl_aux[   h1:h2, :], lbl_aux[   h2:, :]  # upper2, middle2, lower2

        img_aux1 = torch.cat((imgA_1, imgB_2, imgA_3), dim=-2)  # upper1, middle2, lower1
        lbl_aux1 = torch.cat((lblA_1, lblB_2, lblA_3), dim=-2)  # upper1, middle2, lower1

        img_aux2 = torch.cat((imgB_1, imgA_2, imgB_3), dim=-2)  # upper2, middle1, lower2
        lbl_aux2 = torch.cat((lblB_1, lblA_2, lblB_3), dim=-2)  # upper2, middle1, lower2
        
        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col1row4(self, img, lbl, img_aux, lbl_aux):
        h1 = int(img.size()[-2] / 4)     # 32/4 = 8
        mid_h = int(img.size()[-2] / 2)  # 32/2 = 16
        h3 = 3 * h1                      # 8*3  = 24

        imgA_1, imgA_2, imgA_3, imgA_4 = img[:, :h1, :], img[:, h1:mid_h, :], img[:, mid_h:h3, :], img[:, h3:, :]
        lblA_1, lblA_2, lblA_3, lblA_4 = lbl[   :h1, :], lbl[   h1:mid_h, :], lbl[   mid_h:h3, :], lbl[   h3:, :]

        imgB_1, imgB_2, imgB_3, imgB_4 = img_aux[:, :h1, :], img_aux[:, h1:mid_h, :], img_aux[:, mid_h:h3, :], img_aux[:, h3:, :]
        lblB_1, lblB_2, lblB_3, lblB_4 = lbl_aux[   :h1, :], lbl_aux[   h1:mid_h, :], lbl_aux[   mid_h:h3, :], lbl_aux[   h3:, :]

        img_aux1 = torch.cat((imgA_1, imgB_2, imgA_3, imgB_4), dim=-2)
        lbl_aux1 = torch.cat((lblA_1, lblB_2, lblA_3, lblB_4), dim=-2)

        img_aux2 = torch.cat((imgB_1, imgA_2, imgB_3, imgA_4), dim=-2)
        lbl_aux2 = torch.cat((lblB_1, lblA_2, lblB_3, lblA_4), dim=-2)
        
        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col1row5(self, img, lbl, img_aux, lbl_aux):
        h1 = int(img.size()[-2] / 5)     # 32/5 = 6
        h2 = 2 * h1                      # 2*6  = 12
        h3 = 3 * h1                      # 3*6  = 18
        h4 = 4 * h1                      # 4*6  = 24

        imgA_1, imgA_2, imgA_3, imgA_4, imgA_5 = img[:, :h1, :], img[:, h1:h2, :], img[:, h2:h3, :], img[:, h3:h4, :], img[:, h4:, :]
        lblA_1, lblA_2, lblA_3, lblA_4, lblA_5 = lbl[   :h1, :], lbl[   h1:h2, :], lbl[   h2:h3, :], lbl[   h3:h4, :], lbl[   h4:, :]

        imgB_1, imgB_2, imgB_3, imgB_4, imgB_5 = img_aux[:, :h1, :], img_aux[:, h1:h2, :], img_aux[:, h2:h3, :], img_aux[:, h3:h4, :], img_aux[:, h4:, :]
        lblB_1, lblB_2, lblB_3, lblB_4, lblB_5 = lbl_aux[   :h1, :], lbl_aux[   h1:h2, :], lbl_aux[   h2:h3, :], lbl_aux[   h3:h4, :], lbl_aux[   h4:, :]

        img_aux1 = torch.cat((imgA_1, imgB_2, imgA_3, imgB_4, imgA_5), dim=-2)
        lbl_aux1 = torch.cat((lblA_1, lblB_2, lblA_3, lblB_4, lblA_5), dim=-2)

        img_aux2 = torch.cat((imgB_1, imgA_2, imgB_3, imgA_4, imgB_5), dim=-2)
        lbl_aux2 = torch.cat((lblB_1, lblA_2, lblB_3, lblA_4, lblB_5), dim=-2)
        
        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col1row6(self, img, lbl, img_aux, lbl_aux):
        h1 = int(img.size()[-2] / 6)     # 32/6 = 5
        h2 = 2 * h1                      # 2*5  = 10
        h3 = int(img.size()[-2] / 2)     # 32/2 = 16
        h4 = 4 * h1                      # 4*5  = 20
        h5 = 5 * h1                      # 5*5  = 25

        imgA_1, imgA_2, imgA_3, imgA_4, imgA_5, imgA_6 = img[:, :h1, :], img[:, h1:h2, :], img[:, h2:h3, :], img[:, h3:h4, :], img[:, h4:h5, :], img[:, h5:, :]
        lblA_1, lblA_2, lblA_3, lblA_4, lblA_5, lblA_6 = lbl[   :h1, :], lbl[   h1:h2, :], lbl[   h2:h3, :], lbl[   h3:h4, :], lbl[   h4:h5, :], lbl[   h5:, :]

        imgB_1, imgB_2, imgB_3, imgB_4, imgB_5, imgB_6 = img_aux[:, :h1, :], img_aux[:, h1:h2, :], img_aux[:, h2:h3, :], img_aux[:, h3:h4, :], img_aux[:, h4:h5, :], img_aux[:, h5:, :]
        lblB_1, lblB_2, lblB_3, lblB_4, lblB_5, lblB_6 = lbl_aux[   :h1, :], lbl_aux[   h1:h2, :], lbl_aux[   h2:h3, :], lbl_aux[   h3:h4, :], lbl_aux[   h4:h5, :], lbl_aux[   h5:, :]

        img_aux1 = torch.cat((imgA_1, imgB_2, imgA_3, imgB_4, imgA_5, imgB_6), dim=-2)
        lbl_aux1 = torch.cat((lblA_1, lblB_2, lblA_3, lblB_4, lblA_5, lblB_6), dim=-2)

        img_aux2 = torch.cat((imgB_1, imgA_2, imgB_3, imgA_4, imgB_5, imgA_6), dim=-2)
        lbl_aux2 = torch.cat((lblB_1, lblA_2, lblB_3, lblA_4, lblB_5, lblA_6), dim=-2)
        
        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col1row8(self, img, lbl, img_aux, lbl_aux):
        h1 = int(img.size()[-2] / 8)     # 32/8 = 4
        h2 = int(img.size()[-2] / 4)     # 32/4 = 8
        h3 = 3 * h1                      # 3*4  = 12
        h4 = int(img.size()[-2] / 2)     # 32/2 = 16
        h5 = 5 * h1                      # 5*4  = 20
        h6 = 6 * h1                      # 6*4  = 24
        h7 = 7 * h1                      # 7*4  = 28

        imgA_1, imgA_2, imgA_3, imgA_4, imgA_5, imgA_6, imgA_7, imgA_8 = img[:, :h1, :], img[:, h1:h2, :], img[:, h2:h3, :], img[:, h3:h4, :], img[:, h4:h5, :], img[:, h5:h6, :], img[:, h6:h7, :], img[:, h7:, :]
        lblA_1, lblA_2, lblA_3, lblA_4, lblA_5, lblA_6, lblA_7, lblA_8 = lbl[   :h1, :], lbl[   h1:h2, :], lbl[   h2:h3, :], lbl[   h3:h4, :], lbl[   h4:h5, :], lbl[   h5:h6, :], lbl[   h6:h7, :], lbl[   h7:, :]

        imgB_1, imgB_2, imgB_3, imgB_4, imgB_5, imgB_6, imgB_7, imgB_8 = img_aux[:, :h1, :], img_aux[:, h1:h2, :], img_aux[:, h2:h3, :], img_aux[:, h3:h4, :], img_aux[:, h4:h5, :], img_aux[:, h5:h6, :], img_aux[:, h6:h7, :], img_aux[:, h7:, :]
        lblB_1, lblB_2, lblB_3, lblB_4, lblB_5, lblB_6, lblB_7, lblB_8 = lbl_aux[   :h1, :], lbl_aux[   h1:h2, :], lbl_aux[   h2:h3, :], lbl_aux[   h3:h4, :], lbl_aux[   h4:h5, :], lbl_aux[   h5:h6, :], lbl_aux[   h6:h7, :], lbl_aux[   h7:, :]

        img_aux1 = torch.cat((imgA_1, imgB_2, imgA_3, imgB_4, imgA_5, imgB_6, imgA_7, imgB_8), dim=-2)
        lbl_aux1 = torch.cat((lblA_1, lblB_2, lblA_3, lblB_4, lblA_5, lblB_6, lblA_7, lblB_8), dim=-2)

        img_aux2 = torch.cat((imgB_1, imgA_2, imgB_3, imgA_4, imgB_5, imgA_6, imgB_7, imgA_8), dim=-2)
        lbl_aux2 = torch.cat((lblB_1, lblA_2, lblB_3, lblA_4, lblB_5, lblA_6, lblB_7, lblA_8), dim=-2)
        
        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col1row16(self, img, lbl, img_aux, lbl_aux):
        h1 = int(img.size()[-2] / 16)    # 32/16 = 2
        h2 = int(img.size()[-2] / 8)     # 32/8  = 4
        h3 = 3 * h1                      # 3*2   = 6
        h4 = int(img.size()[-2] / 4)     # 32/4  = 8
        h5 = 5 * h1                      # 5*2   = 10
        h6 = 6 * h1                      # 6*2   = 12
        h7 = 7 * h1                      # 7*2   = 14
        h8 = int(img.size()[-2] / 2)     # 32/2  = 16
        h9 = 9 * h1                      # 9*2   = 18
        h10 = 10 * h1                    # 10*2  = 20
        h11 = 11 * h1                    # 11*2  = 22
        h12 = 12 * h1                    # 12*2  = 24
        h13 = 13 * h1                    # 13*2  = 26
        h14 = 14 * h1                    # 14*2  = 28
        h15 = 15 * h1                    # 15*2  = 30

        imgA_1, imgA_2,  imgA_3,  imgA_4,  imgA_5,  imgA_6,  imgA_7,  imgA_8  = img[:,   :h1, :], img[:, h1:h2,  :], img[:,  h2:h3,  :], img[:,  h3:h4,  :], img[:,  h4:h5,  :], img[:,  h5:h6,  :], img[:,  h6:h7,  :], img[:,  h7:h8, :]
        imgA_9, imgA_10, imgA_11, imgA_12, imgA_13, imgA_14, imgA_15, imgA_16 = img[:, h8:h9, :], img[:, h9:h10, :], img[:, h10:h11, :], img[:, h11:h12, :], img[:, h12:h13, :], img[:, h13:h14, :], img[:, h14:h15, :], img[:, h15:,   :]
        
        lblA_1, lblA_2,  lblA_3,  lblA_4,  lblA_5,  lblA_6,  lblA_7,  lblA_8  = lbl[     :h1, :], lbl[   h1:h2,  :], lbl[    h2:h3,  :], lbl[    h3:h4,  :], lbl[    h4:h5,  :], lbl[    h5:h6,  :], lbl[    h6:h7,  :], lbl[    h7:h8, :]
        lblA_9, lblA_10, lblA_11, lblA_12, lblA_13, lblA_14, lblA_15, lblA_16 = lbl[   h8:h9, :], lbl[   h9:h10, :], lbl[   h10:h11, :], lbl[   h11:h12, :], lbl[   h12:h13, :], lbl[   h13:h14, :], lbl[   h14:h15, :], lbl[   h15:,   :]

        imgB_1, imgB_2,  imgB_3,  imgB_4,  imgB_5,  imgB_6,  imgB_7,  imgB_8  = img_aux[:,   :h1, :], img_aux[:, h1:h2,  :], img_aux[:,  h2:h3,  :], img_aux[:,  h3:h4,  :], img_aux[:,  h4:h5,  :], img_aux[:,  h5:h6,  :], img_aux[:,  h6:h7,  :], img_aux[:,  h7:h8, :]
        imgB_9, imgB_10, imgB_11, imgB_12, imgB_13, imgB_14, imgB_15, imgB_16 = img_aux[:, h8:h9, :], img_aux[:, h9:h10, :], img_aux[:, h10:h11, :], img_aux[:, h11:h12, :], img_aux[:, h12:h13, :], img_aux[:, h13:h14, :], img_aux[:, h14:h15, :], img_aux[:, h15:,   :]

        lblB_1, lblB_2,  lblB_3,  lblB_4,  lblB_5,  lblB_6,  lblB_7,  lblB_8  = lbl_aux[     :h1, :], lbl_aux[   h1:h2,  :], lbl_aux[    h2:h3,  :], lbl_aux[    h3:h4,  :], lbl_aux[    h4:h5,  :], lbl_aux[    h5:h6,  :], lbl_aux[    h6:h7,  :], lbl_aux[    h7:h8, :]
        lblB_9, lblB_10, lblB_11, lblB_12, lblB_13, lblB_14, lblB_15, lblB_16 = lbl_aux[   h8:h9, :], lbl_aux[   h9:h10, :], lbl_aux[   h10:h11, :], lbl_aux[   h11:h12, :], lbl_aux[   h12:h13, :], lbl_aux[   h13:h14, :], lbl_aux[   h14:h15, :], lbl_aux[   h15:,   :]

        img_aux1 = torch.cat((imgA_1, imgB_2, imgA_3, imgB_4, imgA_5, imgB_6, imgA_7, imgB_8, imgA_9, imgB_10, imgA_11, imgB_12, imgA_13, imgB_14, imgA_15, imgB_16), dim=-2)
        lbl_aux1 = torch.cat((lblA_1, lblB_2, lblA_3, lblB_4, lblA_5, lblB_6, lblA_7, lblB_8, lblA_9, lblB_10, lblA_11, lblB_12, lblA_13, lblB_14, lblA_15, lblB_16), dim=-2)

        img_aux2 = torch.cat((imgB_1, imgA_2, imgB_3, imgA_4, imgB_5, imgA_6, imgB_7, imgA_8, imgB_9, imgA_10, imgB_11, imgA_12, imgB_13, imgA_14, imgB_15, imgA_16), dim=-2)
        lbl_aux2 = torch.cat((lblB_1, lblA_2, lblB_3, lblA_4, lblB_5, lblA_6, lblB_7, lblA_8, lblB_9, lblA_10, lblB_11, lblA_12, lblB_13, lblA_14, lblB_15, lblA_16), dim=-2)
        
        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col2row1(self, img, lbl, img_aux, lbl_aux):
        mid_w = int(img.size()[-1] / 2)  # 1920/2 = 960
        
        imgA_1, imgA_2 = img[:, :, :mid_w], img[:, :, mid_w:]  # left-half1, right-half1
        lblA_1, lblA_2 = lbl[   :, :mid_w], lbl[   :, mid_w:]  # left-half1, right-half1
        
        imgB_1, imgB_2 = img_aux[:, :, :mid_w], img_aux[:, :, mid_w:]  # left-half2, right-half2
        lblB_1, lblB_2 = lbl_aux[   :, :mid_w], lbl_aux[   :, mid_w:]  # left-half2, right-half2
        
        img_aux1 = torch.cat((imgA_1, imgB_2), dim=-1)  # left-half1, right-half2
        lbl_aux1 = torch.cat((lblA_1, lblB_2), dim=-1)  # left-half1, right-half2

        img_aux2 = torch.cat((imgB_1, imgA_2), dim=-1)  # left-half2, right-half1
        lbl_aux2 = torch.cat((lblB_1, lblA_2), dim=-1)  # left-half2, right-half1
        
        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col2row2(self, img, lbl, img_aux, lbl_aux):
        mid_h = int(img.size()[-2] / 2)  # 32/2 = 16
        mid_w = int(img.size()[-1] / 2)  # 1920/2 = 960
        
        imgA_11, imgA_12, imgA_21, imgA_22 = img[:, :mid_h, :mid_w], img[:, :mid_h, mid_w:], img[:, mid_h:, :mid_w], img[:, mid_h:, mid_w:]
        lblA_11, lblA_12, lblA_21, lblA_22 = lbl[   :mid_h, :mid_w], lbl[   :mid_h, mid_w:], lbl[   mid_h:, :mid_w], lbl[   mid_h:, mid_w:]
        
        imgB_11, imgB_12, imgB_21, imgB_22 = img_aux[:, :mid_h, :mid_w], img_aux[:, :mid_h, mid_w:], img_aux[:, mid_h:, :mid_w], img_aux[:, mid_h:, mid_w:]
        lblB_11, lblB_12, lblB_21, lblB_22 = lbl_aux[   :mid_h, :mid_w], lbl_aux[   :mid_h, mid_w:], lbl_aux[   mid_h:, :mid_w], lbl_aux[   mid_h:, mid_w:]

        concat_img_aux1_top = torch.cat((imgA_11, imgB_12), dim=-1)
        concat_img_aux1_bot = torch.cat((imgB_21, imgA_22), dim=-1)
        img_aux1 = torch.cat((concat_img_aux1_top, concat_img_aux1_bot), dim=-2)
        
        concat_lbl_aux1_top = torch.cat((lblA_11, lblB_12), dim=-1)
        concat_lbl_aux1_bot = torch.cat((lblB_21, lblA_22), dim=-1)
        lbl_aux1 = torch.cat((concat_lbl_aux1_top, concat_lbl_aux1_bot), dim=-2)
        
        concat_img_aux2_top = torch.cat((imgB_11, imgA_12), dim=-1)
        concat_img_aux2_bot = torch.cat((imgA_21, imgB_22), dim=-1)
        img_aux2 = torch.cat((concat_img_aux2_top, concat_img_aux2_bot), dim=-2)
        
        concat_lbl_aux2_top = torch.cat((lblB_11, lblA_12), dim=-1)
        concat_lbl_aux2_bot = torch.cat((lblA_21, lblB_22), dim=-1)
        lbl_aux2 = torch.cat((concat_lbl_aux2_top, concat_lbl_aux2_bot), dim=-2)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2
    
    
    def col2row3(self, img, lbl, img_aux, lbl_aux):  # [bs, 6, 32, 1920]
        h1 = int(img.size()[-2] / 3)  # 32/3 = 10
        h2 = 2 * h1  # 2*10 = 20
        mid_w = int(img.size()[-1] / 2)  # 1920/2 = 960
        
        imgA_11, imgA_12, imgA_21, imgA_22, imgA_31, imgA_32 = img[:, :h1, :mid_w], img[:, :h1, mid_w:], img[:, h1:h2, :mid_w], img[:, h1:h2, mid_w:], img[:, h2:, :mid_w], img[:, h2:, mid_w:]
        lblA_11, lblA_12, lblA_21, lblA_22, lblA_31, lblA_32 = lbl[   :h1, :mid_w], lbl[   :h1, mid_w:], lbl[   h1:h2, :mid_w], lbl[   h1:h2, mid_w:], lbl[   h2:, :mid_w], lbl[   h2:, mid_w:]
        
        imgB_11, imgB_12, imgB_21, imgB_22, imgB_31, imgB_32 = img_aux[:, :h1, :mid_w], img_aux[:, :h1, mid_w:], img_aux[:, h1:h2, :mid_w], img_aux[:, h1:h2, mid_w:], img_aux[:, h2:, :mid_w], img_aux[:, h2:, mid_w:]
        lblB_11, lblB_12, lblB_21, lblB_22, lblB_31, lblB_32 = lbl_aux[   :h1, :mid_w], lbl_aux[   :h1, mid_w:], lbl_aux[   h1:h2, :mid_w], lbl_aux[   h1:h2, mid_w:], lbl_aux[   h2:, :mid_w], lbl_aux[   h2:, mid_w:]

        concat_img_aux1_top = torch.cat((imgA_11, imgB_12), dim=-1)
        concat_img_aux1_mid = torch.cat((imgB_21, imgA_22), dim=-1)
        concat_img_aux1_bot = torch.cat((imgA_31, imgB_32), dim=-1)
        img_aux1 = torch.cat((concat_img_aux1_top, concat_img_aux1_mid, concat_img_aux1_bot), dim=-2)
        
        concat_lbl_aux1_top = torch.cat((lblA_11, lblB_12), dim=-1)
        concat_lbl_aux1_mid = torch.cat((lblB_21, lblA_22), dim=-1)
        concat_lbl_aux1_bot = torch.cat((lblA_31, lblB_32), dim=-1)
        lbl_aux1 = torch.cat((concat_lbl_aux1_top, concat_lbl_aux1_mid, concat_lbl_aux1_bot), dim=-2)
        
        concat_img_aux2_top = torch.cat((imgB_11, imgA_12), dim=-1)
        concat_img_aux2_mid = torch.cat((imgA_21, imgB_22), dim=-1)
        concat_img_aux2_bot = torch.cat((imgB_31, imgA_32), dim=-1)
        img_aux2 = torch.cat((concat_img_aux2_top, concat_img_aux2_mid, concat_img_aux2_bot), dim=-2)
        
        concat_lbl_aux2_top = torch.cat((lblB_11, lblA_12), dim=-1)
        concat_lbl_aux2_mid = torch.cat((lblA_21, lblB_22), dim=-1)
        concat_lbl_aux2_bot = torch.cat((lblB_31, lblA_32), dim=-1)
        lbl_aux2 = torch.cat((concat_lbl_aux2_top, concat_lbl_aux2_mid, concat_lbl_aux2_bot), dim=-2)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col2row4(self, img, lbl, img_aux, lbl_aux):
        h1 = int(img.size()[-2] / 4)     # 32/4 = 8
        mid_h = int(img.size()[-2] / 2)  # 32/2 = 16
        h3 = 3 * h1                      # 8*3  = 24
        mid_w = int(img.size()[-1] / 2)  # 1920/2 = 960

        imgA_11, imgA_21, imgA_31, imgA_41 = img[:, :h1, :mid_w], img[:, h1:mid_h, :mid_w], img[:, mid_h:h3, :mid_w], img[:, h3:, :mid_w]
        imgA_12, imgA_22, imgA_32, imgA_42 = img[:, :h1, mid_w:], img[:, h1:mid_h, mid_w:], img[:, mid_h:h3, mid_w:], img[:, h3:, mid_w:]
        lblA_11, lblA_21, lblA_31, lblA_41 = lbl[   :h1, :mid_w], lbl[   h1:mid_h, :mid_w], lbl[   mid_h:h3, :mid_w], lbl[   h3:, :mid_w]
        lblA_12, lblA_22, lblA_32, lblA_42 = lbl[   :h1, mid_w:], lbl[   h1:mid_h, mid_w:], lbl[   mid_h:h3, mid_w:], lbl[   h3:, mid_w:]

        imgB_11, imgB_21, imgB_31, imgB_41 = img_aux[:, :h1, :mid_w], img_aux[:, h1:mid_h, :mid_w], img_aux[:, mid_h:h3, :mid_w], img_aux[:, h3:, :mid_w]
        imgB_12, imgB_22, imgB_32, imgB_42 = img_aux[:, :h1, mid_w:], img_aux[:, h1:mid_h, mid_w:], img_aux[:, mid_h:h3, mid_w:], img_aux[:, h3:, mid_w:]
        lblB_11, lblB_21, lblB_31, lblB_41 = lbl_aux[   :h1, :mid_w], lbl_aux[   h1:mid_h, :mid_w], lbl_aux[   mid_h:h3, :mid_w], lbl_aux[   h3:, :mid_w]
        lblB_12, lblB_22, lblB_32, lblB_42 = lbl_aux[   :h1, mid_w:], lbl_aux[   h1:mid_h, mid_w:], lbl_aux[   mid_h:h3, mid_w:], lbl_aux[   h3:, mid_w:]

        img_aux1_l, img_aux1_r = torch.cat((imgA_11, imgB_21, imgA_31, imgB_41), dim=-2), torch.cat((imgB_12, imgA_22, imgB_32, imgA_42), dim=-2)
        img_aux1 = torch.cat((img_aux1_l, img_aux1_r), dim=-1)
        lbl_aux1_l, lbl_aux1_r = torch.cat((lblA_11, lblB_21, lblA_31, lblB_41), dim=-2), torch.cat((lblB_12, lblA_22, lblB_32, lblA_42), dim=-2)
        lbl_aux1 = torch.cat((lbl_aux1_l, lbl_aux1_r), dim=-1)

        img_aux2_l, img_aux2_r = torch.cat((imgB_11, imgA_21, imgB_31, imgA_41), dim=-2), torch.cat((imgA_12, imgB_22, imgA_32, imgB_42), dim=-2)
        img_aux2 = torch.cat((img_aux2_l, img_aux2_r), dim=-1)
        lbl_aux2_l, lbl_aux2_r = torch.cat((lblB_11, lblA_21, lblB_31, lblA_41), dim=-2), torch.cat((lblA_12, lblB_22, lblA_32, lblB_42), dim=-2)
        lbl_aux2 = torch.cat((lbl_aux2_l, lbl_aux2_r), dim=-1)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col2row5(self, img, lbl, img_aux, lbl_aux):
        h1 = int(img.size()[-2] / 5)     # 32/5 = 6
        h2 = 2 * h1                      # 2*6  = 12
        h3 = 3 * h1                      # 3*6  = 18
        h4 = 4 * h1                      # 4*6  = 24
        mid_w = int(img.size()[-1] / 2)  # 1920/2 = 960

        imgA_11, imgA_21, imgA_31, imgA_41, imgA_51 = img[:, :h1, :mid_w], img[:, h1:h2, :mid_w], img[:, h2:h3, :mid_w], img[:, h3:h4, :mid_w], img[:, h4:, :mid_w]
        imgA_12, imgA_22, imgA_32, imgA_42, imgA_52 = img[:, :h1, mid_w:], img[:, h1:h2, mid_w:], img[:, h2:h3, mid_w:], img[:, h3:h4, mid_w:], img[:, h4:, mid_w:]
        lblA_11, lblA_21, lblA_31, lblA_41, lblA_51 = lbl[   :h1, :mid_w], lbl[   h1:h2, :mid_w], lbl[   h2:h3, :mid_w], lbl[   h3:h4, :mid_w], lbl[   h4:, :mid_w]
        lblA_12, lblA_22, lblA_32, lblA_42, lblA_52 = lbl[   :h1, mid_w:], lbl[   h1:h2, mid_w:], lbl[   h2:h3, mid_w:], lbl[   h3:h4, mid_w:], lbl[   h4:, mid_w:]

        imgB_11, imgB_21, imgB_31, imgB_41, imgB_51 = img_aux[:, :h1, :mid_w], img_aux[:, h1:h2, :mid_w], img_aux[:, h2:h3, :mid_w], img_aux[:, h3:h4, :mid_w], img_aux[:, h4:, :mid_w]
        imgB_12, imgB_22, imgB_32, imgB_42, imgB_52 = img_aux[:, :h1, mid_w:], img_aux[:, h1:h2, mid_w:], img_aux[:, h2:h3, mid_w:], img_aux[:, h3:h4, mid_w:], img_aux[:, h4:, mid_w:]
        lblB_11, lblB_21, lblB_31, lblB_41, lblB_51 = lbl_aux[   :h1, :mid_w], lbl_aux[   h1:h2, :mid_w], lbl_aux[   h2:h3, :mid_w], lbl_aux[   h3:h4, :mid_w], lbl_aux[   h4:, :mid_w]
        lblB_12, lblB_22, lblB_32, lblB_42, lblB_52 = lbl_aux[   :h1, mid_w:], lbl_aux[   h1:h2, mid_w:], lbl_aux[   h2:h3, mid_w:], lbl_aux[   h3:h4, mid_w:], lbl_aux[   h4:, mid_w:]

        img_aux1_l, img_aux1_r = torch.cat((imgA_11, imgB_21, imgA_31, imgB_41, imgA_51), dim=-2), torch.cat((imgB_12, imgA_22, imgB_32, imgA_42, imgB_52), dim=-2)
        img_aux1 = torch.cat((img_aux1_l, img_aux1_r), dim=-1)
        lbl_aux1_l, lbl_aux1_r = torch.cat((lblA_11, lblB_21, lblA_31, lblB_41, lblA_51), dim=-2), torch.cat((lblB_12, lblA_22, lblB_32, lblA_42, lblB_52), dim=-2)
        lbl_aux1 = torch.cat((lbl_aux1_l, lbl_aux1_r), dim=-1)

        img_aux2_l, img_aux2_r = torch.cat((imgB_11, imgA_21, imgB_31, imgA_41, imgB_51), dim=-2), torch.cat((imgA_12, imgB_22, imgA_32, imgB_42, imgA_52), dim=-2)
        img_aux2 = torch.cat((img_aux2_l, img_aux2_r), dim=-1)
        lbl_aux2_l, lbl_aux2_r = torch.cat((lblB_11, lblA_21, lblB_31, lblA_41, lblB_51), dim=-2), torch.cat((lblA_12, lblB_22, lblA_32, lblB_42, lblA_52), dim=-2)
        lbl_aux2 = torch.cat((lbl_aux2_l, lbl_aux2_r), dim=-1)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col2row6(self, img, lbl, img_aux, lbl_aux):
        h1 = int(img.size()[-2] / 6)     # 32/6 = 5
        h2 = 2 * h1                      # 2*5  = 10
        h3 = 3 * h1                      # 3*5  = 15
        h4 = 4 * h1                      # 4*5  = 20
        h5 = 5 * h1                      # 5*5  = 25
        mid_w = int(img.size()[-1] / 2)  # 1920/2 = 960

        imgA_11, imgA_21, imgA_31, imgA_41, imgA_51, imgA_61 = img[:, :h1, :mid_w], img[:, h1:h2, :mid_w], img[:, h2:h3, :mid_w], img[:, h3:h4, :mid_w], img[:, h4:h5, :mid_w], img[:, h5:, :mid_w]
        imgA_12, imgA_22, imgA_32, imgA_42, imgA_52, imgA_62 = img[:, :h1, mid_w:], img[:, h1:h2, mid_w:], img[:, h2:h3, mid_w:], img[:, h3:h4, mid_w:], img[:, h4:h5, mid_w:], img[:, h5:, mid_w:]
        lblA_11, lblA_21, lblA_31, lblA_41, lblA_51, lblA_61 = lbl[   :h1, :mid_w], lbl[   h1:h2, :mid_w], lbl[   h2:h3, :mid_w], lbl[   h3:h4, :mid_w], lbl[   h4:h5, :mid_w], lbl[   h5:, :mid_w]
        lblA_12, lblA_22, lblA_32, lblA_42, lblA_52, lblA_62 = lbl[   :h1, mid_w:], lbl[   h1:h2, mid_w:], lbl[   h2:h3, mid_w:], lbl[   h3:h4, mid_w:], lbl[   h4:h5, mid_w:], lbl[   h5:, mid_w:]

        imgB_11, imgB_21, imgB_31, imgB_41, imgB_51, imgB_61 = img_aux[:, :h1, :mid_w], img_aux[:, h1:h2, :mid_w], img_aux[:, h2:h3, :mid_w], img_aux[:, h3:h4, :mid_w], img_aux[:, h4:h5, :mid_w], img_aux[:, h5:, :mid_w]
        imgB_12, imgB_22, imgB_32, imgB_42, imgB_52, imgB_62 = img_aux[:, :h1, mid_w:], img_aux[:, h1:h2, mid_w:], img_aux[:, h2:h3, mid_w:], img_aux[:, h3:h4, mid_w:], img_aux[:, h4:h5, mid_w:], img_aux[:, h5:, mid_w:]
        lblB_11, lblB_21, lblB_31, lblB_41, lblB_51, lblB_61 = lbl_aux[   :h1, :mid_w], lbl_aux[   h1:h2, :mid_w], lbl_aux[   h2:h3, :mid_w], lbl_aux[   h3:h4, :mid_w], lbl_aux[   h4:h5, :mid_w], lbl_aux[   h5:, :mid_w]
        lblB_12, lblB_22, lblB_32, lblB_42, lblB_52, lblB_62 = lbl_aux[   :h1, mid_w:], lbl_aux[   h1:h2, mid_w:], lbl_aux[   h2:h3, mid_w:], lbl_aux[   h3:h4, mid_w:], lbl_aux[   h4:h5, mid_w:], lbl_aux[   h5:, mid_w:]

        img_aux1_l, img_aux1_r = torch.cat((imgA_11, imgB_21, imgA_31, imgB_41, imgA_51, imgB_61), dim=-2), torch.cat((imgB_12, imgA_22, imgB_32, imgA_42, imgB_52, imgA_62), dim=-2)
        img_aux1 = torch.cat((img_aux1_l, img_aux1_r), dim=-1)
        lbl_aux1_l, lbl_aux1_r = torch.cat((lblA_11, lblB_21, lblA_31, lblB_41, lblA_51, lblB_61), dim=-2), torch.cat((lblB_12, lblA_22, lblB_32, lblA_42, lblB_52, lblA_62), dim=-2)
        lbl_aux1 = torch.cat((lbl_aux1_l, lbl_aux1_r), dim=-1)

        img_aux2_l, img_aux2_r = torch.cat((imgB_11, imgA_21, imgB_31, imgA_41, imgB_51, imgA_61), dim=-2), torch.cat((imgA_12, imgB_22, imgA_32, imgB_42, imgA_52, imgB_62), dim=-2)
        img_aux2 = torch.cat((img_aux2_l, img_aux2_r), dim=-1)
        lbl_aux2_l, lbl_aux2_r = torch.cat((lblB_11, lblA_21, lblB_31, lblA_41, lblB_51, lblA_61), dim=-2), torch.cat((lblA_12, lblB_22, lblA_32, lblB_42, lblA_52, lblB_62), dim=-2)
        lbl_aux2 = torch.cat((lbl_aux2_l, lbl_aux2_r), dim=-1)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col3row1(self, img, lbl, img_aux, lbl_aux):
        w1 = int(img.size()[-1] / 3)  # 1920/3 = 640
        w2 = 2 * w1                   # 2 * 640 = 1280
        
        imgA_1, imgA_2, imgA_3 = img[:, :, :w1], img[:, :, w1:w2], img[:, :, w2:]  # left1, middle1, right1
        lblA_1, lblA_2, lblA_3 = lbl[   :, :w1], lbl[   :, w1:w2], lbl[   :, w2:]  # left1, middle1, right1
        
        imgB_1, imgB_2, imgB_3 = img_aux[:, :, :w1], img_aux[:, :, w1:w2], img_aux[:, :, w2:]  # left2, middle2, right2
        lblB_1, lblB_2, lblB_3 = lbl_aux[   :, :w1], lbl_aux[   :, w1:w2], lbl_aux[   :, w2:]  # left2, middle2, right2
        
        img_aux1 = torch.cat((imgA_1, imgB_2, imgA_3), dim=-1)  # left1, middle2, right1
        lbl_aux1 = torch.cat((lblA_1, lblB_2, lblA_3), dim=-1)  # left1, middle2, right1

        img_aux2 = torch.cat((imgB_1, imgA_2, imgB_3), dim=-1)  # left2, middle1, right1
        lbl_aux2 = torch.cat((lblB_1, lblA_2, lblB_3), dim=-1)  # left2, middle1, right1
        
        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col3row2(self, img, lbl, img_aux, lbl_aux):
        w1 = int(img.size()[-1] / 3)  # 1920/3 = 640
        w2 = 2 * w1                   # 2 * 640 = 1280
        mid_h = int(img.size()[-2] / 2)  # 32/2 = 16

        imgA_11, imgA_12, imgA_13, imgA_21, imgA_22, imgA_23 = img[:, :mid_h, :w1], img[:, :mid_h, w1:w2], img[:, :mid_h, w2:], img[:, mid_h:, :w1], img[:, mid_h:, w1:w2], img[:, mid_h:, w2:]  # left1, middle1, right1
        lblA_11, lblA_12, lblA_13, lblA_21, lblA_22, lblA_23 = lbl[   :mid_h, :w1], lbl[   :mid_h, w1:w2], lbl[   :mid_h, w2:], lbl[   mid_h:, :w1], lbl[   mid_h:, w1:w2], lbl[   mid_h:, w2:]  # left1, middle1, right1

        imgB_11, imgB_12, imgB_13, imgB_21, imgB_22, imgB_23 = img_aux[:, :mid_h, :w1], img_aux[:, :mid_h, w1:w2], img_aux[:, :mid_h, w2:], img_aux[:, mid_h:, :w1], img_aux[:, mid_h:, w1:w2], img_aux[:, mid_h:, w2:]  # left2, middle2, right2
        lblB_11, lblB_12, lblB_13, lblB_21, lblB_22, lblB_23 = lbl_aux[   :mid_h, :w1], lbl_aux[   :mid_h, w1:w2], lbl_aux[   :mid_h, w2:], lbl_aux[   mid_h:, :w1], lbl_aux[   mid_h:, w1:w2], lbl_aux[   mid_h:, w2:]  # left2, middle2, right2

        img_aux1_top = torch.cat((imgA_11, imgB_12, imgA_13), dim=-1)
        img_aux1_bot = torch.cat((imgB_21, imgA_22, imgB_23), dim=-1)
        img_aux1 = torch.cat((img_aux1_top, img_aux1_bot), dim=-2)

        lbl_aux1_top = torch.cat((lblA_11, lblB_12, lblA_13), dim=-1)
        lbl_aux1_bot = torch.cat((lblB_21, lblA_22, lblB_23), dim=-1)
        lbl_aux1 = torch.cat((lbl_aux1_top, lbl_aux1_bot), dim=-2)

        img_aux2_top = torch.cat((imgB_11, imgA_12, imgB_13), dim=-1)
        img_aux2_bot = torch.cat((imgA_21, imgB_22, imgA_23), dim=-1)
        img_aux2 = torch.cat((img_aux2_top, img_aux2_bot), dim=-2)

        lbl_aux2_top = torch.cat((lblB_11, lblA_12, lblB_13), dim=-1)
        lbl_aux2_bot = torch.cat((lblA_21, lblB_22, lblA_23), dim=-1)
        lbl_aux2 = torch.cat((lbl_aux2_top, lbl_aux2_bot), dim=-2)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col3row3(self, img, lbl, img_aux, lbl_aux):  ##
        w1 = int(img.size()[-1] / 3)  # 1920/3 = 640
        w2 = 2 * w1                   # 2 * 640 = 1280
        h1 = int(img.size()[-2] / 3)  # 32/3 = 10
        h2 = 2 * h1                   # 2 * 10 = 20

        imgA_11, imgA_12, imgA_13 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:]
        imgA_21, imgA_22, imgA_23 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:]
        imgA_31, imgA_32, imgA_33 = img[:, h2:  , :w1], img[:, h2:  , w1:w2], img[:, h2:  , w2:]
        
        lblA_11, lblA_12, lblA_13 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:]
        lblA_21, lblA_22, lblA_23 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:]
        lblA_31, lblA_32, lblA_33 = lbl[   h2:  , :w1], lbl[   h2:  , w1:w2], lbl[   h2:  , w2:]

        imgB_11, imgB_12, imgB_13 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:]
        imgB_21, imgB_22, imgB_23 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:]
        imgB_31, imgB_32, imgB_33 = img_aux[:, h2:  , :w1], img_aux[:, h2:  , w1:w2], img_aux[:, h2:  , w2:]

        lblB_11, lblB_12, lblB_13 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:]
        lblB_21, lblB_22, lblB_23 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:]
        lblB_31, lblB_32, lblB_33 = lbl_aux[   h2:  , :w1], lbl_aux[   h2:  , w1:w2], lbl_aux[   h2:  , w2:]

        img_aux1_top = torch.cat((imgA_11, imgB_12, imgA_13), dim=-1)
        img_aux1_mid = torch.cat((imgB_21, imgA_22, imgB_23), dim=-1)
        img_aux1_bot = torch.cat((imgA_31, imgB_32, imgA_33), dim=-1)
        img_aux1 = torch.cat((img_aux1_top, img_aux1_mid, img_aux1_bot), dim=-2)

        lbl_aux1_top = torch.cat((lblA_11, lblB_12, lblA_13), dim=-1)
        lbl_aux1_mid = torch.cat((lblB_21, lblA_22, lblB_23), dim=-1)
        lbl_aux1_bot = torch.cat((lblA_31, lblB_32, lblA_33), dim=-1)
        lbl_aux1 = torch.cat((lbl_aux1_top, lbl_aux1_mid, lbl_aux1_bot), dim=-2)

        img_aux2_top = torch.cat((imgB_11, imgA_12, imgB_13), dim=-1)
        img_aux2_mid = torch.cat((imgA_21, imgB_22, imgA_23), dim=-1)
        img_aux2_bot = torch.cat((imgB_31, imgA_32, imgB_33), dim=-1)
        img_aux2 = torch.cat((img_aux2_top, img_aux2_mid, img_aux2_bot), dim=-2)

        lbl_aux2_top = torch.cat((lblB_11, lblA_12, lblB_13), dim=-1)
        lbl_aux2_mid = torch.cat((lblA_21, lblB_22, lblA_23), dim=-1)
        lbl_aux2_bot = torch.cat((lblB_31, lblA_32, lblB_33), dim=-1)
        lbl_aux2 = torch.cat((lbl_aux2_top, lbl_aux2_mid, lbl_aux2_bot), dim=-2)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col3row4(self, img, lbl, img_aux, lbl_aux):
        w1 = int(img.size()[-1] / 3)  # 1920/3 = 640
        w2 = 2 * w1                   # 2 * 640 = 1280
        h1 = int(img.size()[-2] / 4)  # 32/4 = 8
        h2 = int(img.size()[-2] / 2)  # 32/2 = 16
        h3 = 3 * h1                   # 3 * 24 = 24

        imgA_11, imgA_12, imgA_13 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:]
        imgA_21, imgA_22, imgA_23 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:]
        imgA_31, imgA_32, imgA_33 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:]
        imgA_41, imgA_42, imgA_43 = img[:, h3:  , :w1], img[:, h3:  , w1:w2], img[:, h3:  , w2:]
        
        lblA_11, lblA_12, lblA_13 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:]
        lblA_21, lblA_22, lblA_23 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:]
        lblA_31, lblA_32, lblA_33 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:]
        lblA_41, lblA_42, lblA_43 = lbl[   h3:  , :w1], lbl[   h3:  , w1:w2], lbl[   h3:  , w2:]

        imgB_11, imgB_12, imgB_13 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:]
        imgB_21, imgB_22, imgB_23 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:]
        imgB_31, imgB_32, imgB_33 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:]
        imgB_41, imgB_42, imgB_43 = img_aux[:, h3:  , :w1], img_aux[:, h3:  , w1:w2], img_aux[:, h3:  , w2:]

        lblB_11, lblB_12, lblB_13 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:]
        lblB_21, lblB_22, lblB_23 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:]
        lblB_31, lblB_32, lblB_33 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:]
        lblB_41, lblB_42, lblB_43 = lbl_aux[   h3:  , :w1], lbl_aux[   h3:  , w1:w2], lbl_aux[   h3:  , w2:]

        img_aux1_ro1 = torch.cat((imgA_11, imgB_12, imgA_13), dim=-1)
        img_aux1_ro2 = torch.cat((imgB_21, imgA_22, imgB_23), dim=-1)
        img_aux1_ro3 = torch.cat((imgA_31, imgB_32, imgA_33), dim=-1)
        img_aux1_ro4 = torch.cat((imgB_41, imgA_42, imgB_43), dim=-1)
        img_aux1 = torch.cat((img_aux1_ro1, img_aux1_ro2, img_aux1_ro3, img_aux1_ro4), dim=-2)

        lbl_aux1_ro1 = torch.cat((lblA_11, lblB_12, lblA_13), dim=-1)
        lbl_aux1_ro2 = torch.cat((lblB_21, lblA_22, lblB_23), dim=-1)
        lbl_aux1_ro3 = torch.cat((lblA_31, lblB_32, lblA_33), dim=-1)
        lbl_aux1_ro4 = torch.cat((lblB_41, lblA_42, lblB_43), dim=-1)
        lbl_aux1 = torch.cat((lbl_aux1_ro1, lbl_aux1_ro2, lbl_aux1_ro3, lbl_aux1_ro4), dim=-2)

        img_aux2_ro1 = torch.cat((imgB_11, imgA_12, imgB_13), dim=-1)
        img_aux2_ro2 = torch.cat((imgA_21, imgB_22, imgA_23), dim=-1)
        img_aux2_ro3 = torch.cat((imgB_31, imgA_32, imgB_33), dim=-1)
        img_aux2_ro4 = torch.cat((imgA_41, imgB_42, imgA_43), dim=-1)
        img_aux2 = torch.cat((img_aux2_ro1, img_aux2_ro2, img_aux2_ro3, img_aux2_ro4), dim=-2)

        lbl_aux2_ro1 = torch.cat((lblB_11, lblA_12, lblB_13), dim=-1)
        lbl_aux2_ro2 = torch.cat((lblA_21, lblB_22, lblA_23), dim=-1)
        lbl_aux2_ro3 = torch.cat((lblB_31, lblA_32, lblB_33), dim=-1)
        lbl_aux2_ro4 = torch.cat((lblA_41, lblB_42, lblA_43), dim=-1)
        lbl_aux2 = torch.cat((lbl_aux2_ro1, lbl_aux2_ro2, lbl_aux2_ro3, lbl_aux2_ro4), dim=-2)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col3row5(self, img, lbl, img_aux, lbl_aux):
        w1 = int(img.size()[-1] / 3)  # 1920/3 = 640
        w2 = 2 * w1                   # 2 * 640 = 1280
        h1 = int(img.size()[-2] / 5)  # 32/5  = 6
        h2 = 2 * h1                   # 2 * 6 = 12
        h3 = 3 * h1                   # 3 * 6 = 18
        h4 = 4 * h1                   # 4 * 6 = 24

        imgA_11, imgA_12, imgA_13 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:]
        imgA_21, imgA_22, imgA_23 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:]
        imgA_31, imgA_32, imgA_33 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:]
        imgA_41, imgA_42, imgA_43 = img[:, h3:h4, :w1], img[:, h3:h4, w1:w2], img[:, h3:h4, w2:]
        imgA_51, imgA_52, imgA_53 = img[:, h4:  , :w1], img[:, h4:  , w1:w2], img[:, h4:  , w2:]
        
        lblA_11, lblA_12, lblA_13 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:]
        lblA_21, lblA_22, lblA_23 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:]
        lblA_31, lblA_32, lblA_33 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:]
        lblA_41, lblA_42, lblA_43 = lbl[   h3:h4, :w1], lbl[   h3:h4, w1:w2], lbl[   h3:h4, w2:]
        lblA_51, lblA_52, lblA_53 = lbl[   h4:  , :w1], lbl[   h4:  , w1:w2], lbl[   h4:  , w2:]

        imgB_11, imgB_12, imgB_13 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:]
        imgB_21, imgB_22, imgB_23 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:]
        imgB_31, imgB_32, imgB_33 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:]
        imgB_41, imgB_42, imgB_43 = img_aux[:, h3:h4, :w1], img_aux[:, h3:h4, w1:w2], img_aux[:, h3:h4, w2:]
        imgB_51, imgB_52, imgB_53 = img_aux[:, h4:  , :w1], img_aux[:, h4:  , w1:w2], img_aux[:, h4:  , w2:]

        lblB_11, lblB_12, lblB_13 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:]
        lblB_21, lblB_22, lblB_23 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:]
        lblB_31, lblB_32, lblB_33 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:]
        lblB_41, lblB_42, lblB_43 = lbl_aux[   h3:h4, :w1], lbl_aux[   h3:h4, w1:w2], lbl_aux[   h3:h4, w2:]
        lblB_51, lblB_52, lblB_53 = lbl_aux[   h4:  , :w1], lbl_aux[   h4:  , w1:w2], lbl_aux[   h4:  , w2:]

        img_aux1_ro1 = torch.cat((imgA_11, imgB_12, imgA_13), dim=-1)
        img_aux1_ro2 = torch.cat((imgB_21, imgA_22, imgB_23), dim=-1)
        img_aux1_ro3 = torch.cat((imgA_31, imgB_32, imgA_33), dim=-1)
        img_aux1_ro4 = torch.cat((imgB_41, imgA_42, imgB_43), dim=-1)
        img_aux1_ro5 = torch.cat((imgA_51, imgB_52, imgA_53), dim=-1)
        img_aux1 = torch.cat((img_aux1_ro1, img_aux1_ro2, img_aux1_ro3, img_aux1_ro4, img_aux1_ro5), dim=-2)

        lbl_aux1_ro1 = torch.cat((lblA_11, lblB_12, lblA_13), dim=-1)
        lbl_aux1_ro2 = torch.cat((lblB_21, lblA_22, lblB_23), dim=-1)
        lbl_aux1_ro3 = torch.cat((lblA_31, lblB_32, lblA_33), dim=-1)
        lbl_aux1_ro4 = torch.cat((lblB_41, lblA_42, lblB_43), dim=-1)
        lbl_aux1_ro5 = torch.cat((lblA_51, lblB_52, lblA_53), dim=-1)
        lbl_aux1 = torch.cat((lbl_aux1_ro1, lbl_aux1_ro2, lbl_aux1_ro3, lbl_aux1_ro4, lbl_aux1_ro5), dim=-2)

        img_aux2_ro1 = torch.cat((imgB_11, imgA_12, imgB_13), dim=-1)
        img_aux2_ro2 = torch.cat((imgA_21, imgB_22, imgA_23), dim=-1)
        img_aux2_ro3 = torch.cat((imgB_31, imgA_32, imgB_33), dim=-1)
        img_aux2_ro4 = torch.cat((imgA_41, imgB_42, imgA_43), dim=-1)
        img_aux2_ro5 = torch.cat((imgB_51, imgA_52, imgB_53), dim=-1)
        img_aux2 = torch.cat((img_aux2_ro1, img_aux2_ro2, img_aux2_ro3, img_aux2_ro4, img_aux2_ro5), dim=-2)

        lbl_aux2_ro1 = torch.cat((lblB_11, lblA_12, lblB_13), dim=-1)
        lbl_aux2_ro2 = torch.cat((lblA_21, lblB_22, lblA_23), dim=-1)
        lbl_aux2_ro3 = torch.cat((lblB_31, lblA_32, lblB_33), dim=-1)
        lbl_aux2_ro4 = torch.cat((lblA_41, lblB_42, lblA_43), dim=-1)
        lbl_aux2_ro5 = torch.cat((lblB_51, lblA_52, lblB_53), dim=-1)
        lbl_aux2 = torch.cat((lbl_aux2_ro1, lbl_aux2_ro2, lbl_aux2_ro3, lbl_aux2_ro4, lbl_aux2_ro5), dim=-2)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col3row6(self, img, lbl, img_aux, lbl_aux):
        w1 = int(img.size()[-1] / 3)  # 1920/3 = 640
        w2 = 2 * w1                   # 2 * 640 = 1280
        h1 = int(img.size()[-2] / 6)  # 32/6  = 5
        h2 = 2 * h1                   # 2 * 5 = 10
        h3 = int(img.size()[-2] / 2)  # 32/2  = 16
        h4 = 4 * h1                   # 4 * 5 = 20
        h5 = 5 * h1                   # 5 * 5 = 25

        imgA_11, imgA_12, imgA_13 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:]
        imgA_21, imgA_22, imgA_23 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:]
        imgA_31, imgA_32, imgA_33 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:]
        imgA_41, imgA_42, imgA_43 = img[:, h3:h4, :w1], img[:, h3:h4, w1:w2], img[:, h3:h4, w2:]
        imgA_51, imgA_52, imgA_53 = img[:, h4:h5, :w1], img[:, h4:h5, w1:w2], img[:, h4:h5, w2:]
        imgA_61, imgA_62, imgA_63 = img[:, h5:  , :w1], img[:, h5:  , w1:w2], img[:, h5:  , w2:]
        
        lblA_11, lblA_12, lblA_13 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:]
        lblA_21, lblA_22, lblA_23 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:]
        lblA_31, lblA_32, lblA_33 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:]
        lblA_41, lblA_42, lblA_43 = lbl[   h3:h4, :w1], lbl[   h3:h4, w1:w2], lbl[   h3:h4, w2:]
        lblA_51, lblA_52, lblA_53 = lbl[   h4:h5, :w1], lbl[   h4:h5, w1:w2], lbl[   h4:h5, w2:]
        lblA_61, lblA_62, lblA_63 = lbl[   h5:  , :w1], lbl[   h5:  , w1:w2], lbl[   h5:  , w2:]

        imgB_11, imgB_12, imgB_13 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:]
        imgB_21, imgB_22, imgB_23 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:]
        imgB_31, imgB_32, imgB_33 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:]
        imgB_41, imgB_42, imgB_43 = img_aux[:, h3:h4, :w1], img_aux[:, h3:h4, w1:w2], img_aux[:, h3:h4, w2:]
        imgB_51, imgB_52, imgB_53 = img_aux[:, h4:h5, :w1], img_aux[:, h4:h5, w1:w2], img_aux[:, h4:h5, w2:]
        imgB_61, imgB_62, imgB_63 = img_aux[:, h5:  , :w1], img_aux[:, h5:  , w1:w2], img_aux[:, h5:  , w2:]

        lblB_11, lblB_12, lblB_13 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:]
        lblB_21, lblB_22, lblB_23 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:]
        lblB_31, lblB_32, lblB_33 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:]
        lblB_41, lblB_42, lblB_43 = lbl_aux[   h3:h4, :w1], lbl_aux[   h3:h4, w1:w2], lbl_aux[   h3:h4, w2:]
        lblB_51, lblB_52, lblB_53 = lbl_aux[   h4:h5, :w1], lbl_aux[   h4:h5, w1:w2], lbl_aux[   h4:h5, w2:]
        lblB_61, lblB_62, lblB_63 = lbl_aux[   h5:  , :w1], lbl_aux[   h5:  , w1:w2], lbl_aux[   h5:  , w2:]

        img_aux1_ro1 = torch.cat((imgA_11, imgB_12, imgA_13), dim=-1)
        img_aux1_ro2 = torch.cat((imgB_21, imgA_22, imgB_23), dim=-1)
        img_aux1_ro3 = torch.cat((imgA_31, imgB_32, imgA_33), dim=-1)
        img_aux1_ro4 = torch.cat((imgB_41, imgA_42, imgB_43), dim=-1)
        img_aux1_ro5 = torch.cat((imgA_51, imgB_52, imgA_53), dim=-1)
        img_aux1_ro6 = torch.cat((imgB_61, imgA_62, imgB_63), dim=-1)
        img_aux1 = torch.cat((img_aux1_ro1, img_aux1_ro2, img_aux1_ro3, img_aux1_ro4, img_aux1_ro5, img_aux1_ro6), dim=-2)

        lbl_aux1_ro1 = torch.cat((lblA_11, lblB_12, lblA_13), dim=-1)
        lbl_aux1_ro2 = torch.cat((lblB_21, lblA_22, lblB_23), dim=-1)
        lbl_aux1_ro3 = torch.cat((lblA_31, lblB_32, lblA_33), dim=-1)
        lbl_aux1_ro4 = torch.cat((lblB_41, lblA_42, lblB_43), dim=-1)
        lbl_aux1_ro5 = torch.cat((lblA_51, lblB_52, lblA_53), dim=-1)
        lbl_aux1_ro6 = torch.cat((lblB_61, lblA_62, lblB_63), dim=-1)
        lbl_aux1 = torch.cat((lbl_aux1_ro1, lbl_aux1_ro2, lbl_aux1_ro3, lbl_aux1_ro4, lbl_aux1_ro5, lbl_aux1_ro6), dim=-2)

        img_aux2_ro1 = torch.cat((imgB_11, imgA_12, imgB_13), dim=-1)
        img_aux2_ro2 = torch.cat((imgA_21, imgB_22, imgA_23), dim=-1)
        img_aux2_ro3 = torch.cat((imgB_31, imgA_32, imgB_33), dim=-1)
        img_aux2_ro4 = torch.cat((imgA_41, imgB_42, imgA_43), dim=-1)
        img_aux2_ro5 = torch.cat((imgB_51, imgA_52, imgB_53), dim=-1)
        img_aux2_ro6 = torch.cat((imgA_61, imgB_62, imgA_63), dim=-1)
        img_aux2 = torch.cat((img_aux2_ro1, img_aux2_ro2, img_aux2_ro3, img_aux2_ro4, img_aux2_ro5, img_aux2_ro6), dim=-2)

        lbl_aux2_ro1 = torch.cat((lblB_11, lblA_12, lblB_13), dim=-1)
        lbl_aux2_ro2 = torch.cat((lblA_21, lblB_22, lblA_23), dim=-1)
        lbl_aux2_ro3 = torch.cat((lblB_31, lblA_32, lblB_33), dim=-1)
        lbl_aux2_ro4 = torch.cat((lblA_41, lblB_42, lblA_43), dim=-1)
        lbl_aux2_ro5 = torch.cat((lblB_51, lblA_52, lblB_53), dim=-1)
        lbl_aux2_ro6 = torch.cat((lblA_61, lblB_62, lblA_63), dim=-1)
        lbl_aux2 = torch.cat((lbl_aux2_ro1, lbl_aux2_ro2, lbl_aux2_ro3, lbl_aux2_ro4, lbl_aux2_ro5, lbl_aux2_ro6), dim=-2)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col4row1(self, img, lbl, img_aux, lbl_aux):  # [bs, 6, 32, 1920]
        w1 = int(img.size()[-1] / 4)  # 1920/4 = 480
        w2 = int(img.size()[-1] / 2)  # 1920/2 = 960
        w3 = 3 * w1                   # 3 * 480 = 1440
        
        imgA_1, imgA_2, imgA_3, imgA_4 = img[:, :, :w1], img[:, :, w1:w2], img[:, :, w2:w3], img[:, :, w3:]  # 1 - 2 - 3 - 4
        lblA_1, lblA_2, lblA_3, lblA_4 = lbl[   :, :w1], lbl[   :, w1:w2], lbl[   :, w2:w3], lbl[   :, w3:]  # 1 - 2 - 3 - 4
        
        imgB_1, imgB_2, imgB_3, imgB_4 = img_aux[:, :, :w1], img_aux[:, :, w1:w2], img_aux[:, :, w2:w3], img_aux[:, :, w3:]  # 1 - 2 - 3 - 4
        lblB_1, lblB_2, lblB_3, lblB_4 = lbl_aux[   :, :w1], lbl_aux[   :, w1:w2], lbl_aux[   :, w2:w3], lbl_aux[   :, w3:]  # 1 - 2 - 3 - 4
        
        img_aux1 = torch.cat((imgA_1, imgB_2, imgA_3, imgB_4), dim=-1)  # 1 - 2 - 3 - 4
        lbl_aux1 = torch.cat((lblA_1, lblB_2, lblA_3, lblB_4), dim=-1)  # 1 - 2 - 3 - 4

        img_aux2 = torch.cat((imgB_1, imgA_2, imgB_3, imgA_4), dim=-1)  # 1 - 2 - 3 - 4
        lbl_aux2 = torch.cat((lblB_1, lblA_2, lblB_3, lblA_4), dim=-1)  # 1 - 2 - 3 - 4
        
        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col4row2(self, img, lbl, img_aux, lbl_aux):
        w1 = int(img.size()[-1] / 4)  # 1920/4 = 480
        w2 = int(img.size()[-1] / 2)  # 1920/2 = 960
        w3 = 3 * w1                   # 3 * 480 = 1440
        h_mid = int(img.size()[-2] / 2)  # 32/2 = 16
        
        imgA_11, imgA_12, imgA_13, imgA_14 = img[:, :h_mid, :w1], img[:, :h_mid, w1:w2], img[:, :h_mid, w2:w3], img[:, :h_mid, w3:]
        imgA_21, imgA_22, imgA_23, imgA_24 = img[:, h_mid:, :w1], img[:, h_mid:, w1:w2], img[:, h_mid:, w2:w3], img[:, h_mid:, w3:]
        
        lblA_11, lblA_12, lblA_13, lblA_14 = lbl[   :h_mid, :w1], lbl[   :h_mid, w1:w2], lbl[   :h_mid, w2:w3], lbl[   :h_mid, w3:]
        lblA_21, lblA_22, lblA_23, lblA_24 = lbl[   h_mid:, :w1], lbl[   h_mid:, w1:w2], lbl[   h_mid:, w2:w3], lbl[   h_mid:, w3:]

        imgB_11, imgB_12, imgB_13, imgB_14 = img_aux[:, :h_mid, :w1], img_aux[:, :h_mid, w1:w2], img_aux[:, :h_mid, w2:w3], img_aux[:, :h_mid, w3:]
        imgB_21, imgB_22, imgB_23, imgB_24 = img_aux[:, h_mid:, :w1], img_aux[:, h_mid:, w1:w2], img_aux[:, h_mid:, w2:w3], img_aux[:, h_mid:, w3:]

        lblB_11, lblB_12, lblB_13, lblB_14 = lbl_aux[   :h_mid, :w1], lbl_aux[   :h_mid, w1:w2], lbl_aux[   :h_mid, w2:w3], lbl_aux[   :h_mid, w3:]
        lblB_21, lblB_22, lblB_23, lblB_24 = lbl_aux[   h_mid:, :w1], lbl_aux[   h_mid:, w1:w2], lbl_aux[   h_mid:, w2:w3], lbl_aux[   h_mid:, w3:]

        img_aux1_ro1 = torch.cat((imgA_11, imgB_12, imgA_13, imgB_14), dim=-1)
        img_aux1_ro2 = torch.cat((imgB_21, imgA_22, imgB_23, imgA_24), dim=-1)
        img_aux1 = torch.cat((img_aux1_ro1, img_aux1_ro2), dim=-2)

        lbl_aux1_ro1 = torch.cat((lblA_11, lblB_12, lblA_13, lblB_14), dim=-1)
        lbl_aux1_ro2 = torch.cat((lblB_21, lblA_22, lblB_23, lblA_24), dim=-1)
        lbl_aux1 = torch.cat((lbl_aux1_ro1, lbl_aux1_ro2), dim=-2)

        img_aux2_ro1 = torch.cat((imgB_11, imgA_12, imgB_13, imgA_14), dim=-1)
        img_aux2_ro2 = torch.cat((imgA_21, imgB_22, imgA_23, imgB_24), dim=-1)
        img_aux2 = torch.cat((img_aux2_ro1, img_aux2_ro2), dim=-2)

        lbl_aux2_ro1 = torch.cat((lblB_11, lblA_12, lblB_13, lblA_14), dim=-1)
        lbl_aux2_ro2 = torch.cat((lblA_21, lblB_22, lblA_23, lblB_24), dim=-1)
        lbl_aux2 = torch.cat((lbl_aux2_ro1, lbl_aux2_ro2), dim=-2)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col4row3(self, img, lbl, img_aux, lbl_aux):
        w1 = int(img.size()[-1] / 4)  # 1920/4 = 480
        w2 = int(img.size()[-1] / 2)  # 1920/2 = 960
        w3 = 3 * w1                   # 3 * 480 = 1440
        h1 = int(img.size()[-2] / 3)  # 32/3   = 10
        h2 = 2 * h1                   # 2 * 10 = 20

        imgA_11, imgA_12, imgA_13, imgA_14 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:w3], img[:,   :h1, w3:]
        imgA_21, imgA_22, imgA_23, imgA_24 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:w3], img[:, h1:h2, w3:]
        imgA_31, imgA_32, imgA_33, imgA_34 = img[:, h2:,   :w1], img[:, h2:,   w1:w2], img[:, h2:,   w2:w3], img[:, h2:,   w3:]
        
        lblA_11, lblA_12, lblA_13, lblA_14 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:w3], lbl[     :h1, w3:]
        lblA_21, lblA_22, lblA_23, lblA_24 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:w3], lbl[   h1:h2, w3:]
        lblA_31, lblA_32, lblA_33, lblA_34 = lbl[   h2:,   :w1], lbl[   h2:,   w1:w2], lbl[   h2:,   w2:w3], lbl[   h2:,   w3:]

        imgB_11, imgB_12, imgB_13, imgB_14 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:w3], img_aux[:,   :h1, w3:]
        imgB_21, imgB_22, imgB_23, imgB_24 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:w3], img_aux[:, h1:h2, w3:]
        imgB_31, imgB_32, imgB_33, imgB_34 = img_aux[:, h2:,   :w1], img_aux[:, h2:,   w1:w2], img_aux[:, h2:,   w2:w3], img_aux[:, h2:,   w3:]

        lblB_11, lblB_12, lblB_13, lblB_14 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:w3], lbl_aux[     :h1, w3:]
        lblB_21, lblB_22, lblB_23, lblB_24 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:w3], lbl_aux[   h1:h2, w3:]
        lblB_31, lblB_32, lblB_33, lblB_34 = lbl_aux[   h2:,   :w1], lbl_aux[   h2:,   w1:w2], lbl_aux[   h2:,   w2:w3], lbl_aux[   h2:,   w3:]

        img_aux1_ro1 = torch.cat((imgA_11, imgB_12, imgA_13, imgB_14), dim=-1)
        img_aux1_ro2 = torch.cat((imgB_21, imgA_22, imgB_23, imgA_24), dim=-1)
        img_aux1_ro3 = torch.cat((imgA_31, imgB_32, imgA_33, imgB_34), dim=-1)
        img_aux1 = torch.cat((img_aux1_ro1, img_aux1_ro2, img_aux1_ro3), dim=-2)

        lbl_aux1_ro1 = torch.cat((lblA_11, lblB_12, lblA_13, lblB_14), dim=-1)
        lbl_aux1_ro2 = torch.cat((lblB_21, lblA_22, lblB_23, lblA_24), dim=-1)
        lbl_aux1_ro3 = torch.cat((lblA_31, lblB_32, lblA_33, lblB_34), dim=-1)
        lbl_aux1 = torch.cat((lbl_aux1_ro1, lbl_aux1_ro2, lbl_aux1_ro3), dim=-2)

        img_aux2_ro1 = torch.cat((imgB_11, imgA_12, imgB_13, imgA_14), dim=-1)
        img_aux2_ro2 = torch.cat((imgA_21, imgB_22, imgA_23, imgB_24), dim=-1)
        img_aux2_ro3 = torch.cat((imgB_31, imgA_32, imgB_33, imgA_34), dim=-1)
        img_aux2 = torch.cat((img_aux2_ro1, img_aux2_ro2, img_aux2_ro3), dim=-2)

        lbl_aux2_ro1 = torch.cat((lblB_11, lblA_12, lblB_13, lblA_14), dim=-1)
        lbl_aux2_ro2 = torch.cat((lblA_21, lblB_22, lblA_23, lblB_24), dim=-1)
        lbl_aux2_ro3 = torch.cat((lblB_31, lblA_32, lblB_33, lblA_34), dim=-1)
        lbl_aux2 = torch.cat((lbl_aux2_ro1, lbl_aux2_ro2, lbl_aux2_ro3), dim=-2)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col4row4(self, img, lbl, img_aux, lbl_aux):
        w1 = int(img.size()[-1] / 4)  # 1920/4 = 480
        w2 = int(img.size()[-1] / 2)  # 1920/2 = 960
        w3 = 3 * w1                   # 3 * 480 = 1440
        h1 = int(img.size()[-2] / 4)  # 32/4 = 8
        h2 = int(img.size()[-2] / 2)  # 32/2 = 16
        h3 = 3 * h1                   # 3 * 24 = 24

        imgA_11, imgA_12, imgA_13, imgA_14 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:w3], img[:,   :h1, w3:]
        imgA_21, imgA_22, imgA_23, imgA_24 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:w3], img[:, h1:h2, w3:]
        imgA_31, imgA_32, imgA_33, imgA_34 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:w3], img[:, h2:h3, w3:]
        imgA_41, imgA_42, imgA_43, imgA_44 = img[:, h3:  , :w1], img[:, h3:  , w1:w2], img[:, h3:  , w2:w3], img[:, h3:  , w3:]
        
        lblA_11, lblA_12, lblA_13, lblA_14 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:w3], lbl[     :h1, w3:]
        lblA_21, lblA_22, lblA_23, lblA_24 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:w3], lbl[   h1:h2, w3:]
        lblA_31, lblA_32, lblA_33, lblA_34 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:w3], lbl[   h2:h3, w3:]
        lblA_41, lblA_42, lblA_43, lblA_44 = lbl[   h3:  , :w1], lbl[   h3:  , w1:w2], lbl[   h3:  , w2:w3], lbl[   h3:  , w3:]

        imgB_11, imgB_12, imgB_13, imgB_14 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:w3], img_aux[:,   :h1, w3:]
        imgB_21, imgB_22, imgB_23, imgB_24 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:w3], img_aux[:, h1:h2, w3:]
        imgB_31, imgB_32, imgB_33, imgB_34 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:w3], img_aux[:, h2:h3, w3:]
        imgB_41, imgB_42, imgB_43, imgB_44 = img_aux[:, h3:  , :w1], img_aux[:, h3:  , w1:w2], img_aux[:, h3:  , w2:w3], img_aux[:, h3:  , w3:]

        lblB_11, lblB_12, lblB_13, lblB_14 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:w3], lbl_aux[     :h1, w3:]
        lblB_21, lblB_22, lblB_23, lblB_24 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:w3], lbl_aux[   h1:h2, w3:]
        lblB_31, lblB_32, lblB_33, lblB_34 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:w3], lbl_aux[   h2:h3, w3:]
        lblB_41, lblB_42, lblB_43, lblB_44 = lbl_aux[   h3:  , :w1], lbl_aux[   h3:  , w1:w2], lbl_aux[   h3:  , w2:w3], lbl_aux[   h3:  , w3:]

        img_aux1_ro1 = torch.cat((imgA_11, imgB_12, imgA_13, imgB_14), dim=-1)
        img_aux1_ro2 = torch.cat((imgB_21, imgA_22, imgB_23, imgA_24), dim=-1)
        img_aux1_ro3 = torch.cat((imgA_31, imgB_32, imgA_33, imgB_34), dim=-1)
        img_aux1_ro4 = torch.cat((imgB_41, imgA_42, imgB_43, imgA_44), dim=-1)
        img_aux1 = torch.cat((img_aux1_ro1, img_aux1_ro2, img_aux1_ro3, img_aux1_ro4), dim=-2)

        lbl_aux1_ro1 = torch.cat((lblA_11, lblB_12, lblA_13, lblB_14), dim=-1)
        lbl_aux1_ro2 = torch.cat((lblB_21, lblA_22, lblB_23, lblA_24), dim=-1)
        lbl_aux1_ro3 = torch.cat((lblA_31, lblB_32, lblA_33, lblB_34), dim=-1)
        lbl_aux1_ro4 = torch.cat((lblB_41, lblA_42, lblB_43, lblA_44), dim=-1)
        lbl_aux1 = torch.cat((lbl_aux1_ro1, lbl_aux1_ro2, lbl_aux1_ro3, lbl_aux1_ro4), dim=-2)

        img_aux2_ro1 = torch.cat((imgB_11, imgA_12, imgB_13, imgA_14), dim=-1)
        img_aux2_ro2 = torch.cat((imgA_21, imgB_22, imgA_23, imgB_24), dim=-1)
        img_aux2_ro3 = torch.cat((imgB_31, imgA_32, imgB_33, imgA_34), dim=-1)
        img_aux2_ro4 = torch.cat((imgA_41, imgB_42, imgA_43, imgB_44), dim=-1)
        img_aux2 = torch.cat((img_aux2_ro1, img_aux2_ro2, img_aux2_ro3, img_aux2_ro4), dim=-2)

        lbl_aux2_ro1 = torch.cat((lblB_11, lblA_12, lblB_13, lblA_14), dim=-1)
        lbl_aux2_ro2 = torch.cat((lblA_21, lblB_22, lblA_23, lblB_24), dim=-1)
        lbl_aux2_ro3 = torch.cat((lblB_31, lblA_32, lblB_33, lblA_34), dim=-1)
        lbl_aux2_ro4 = torch.cat((lblA_41, lblB_42, lblA_43, lblB_44), dim=-1)
        lbl_aux2 = torch.cat((lbl_aux2_ro1, lbl_aux2_ro2, lbl_aux2_ro3, lbl_aux2_ro4), dim=-2)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col4row5(self, img, lbl, img_aux, lbl_aux):
        w1 = int(img.size()[-1] / 4)  # 1920/4 = 480
        w2 = int(img.size()[-1] / 2)  # 1920/2 = 960
        w3 = 3 * w1                   # 3 * 480 = 1440
        h1 = int(img.size()[-2] / 5)  # 32/5  = 6
        h2 = 2 * h1                   # 2 * 6 = 12
        h3 = 3 * h1                   # 3 * 6 = 18
        h4 = 4 * h1                   # 4 * 6 = 24

        imgA_11, imgA_12, imgA_13, imgA_14 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:w3], img[:,   :h1, w3:]
        imgA_21, imgA_22, imgA_23, imgA_24 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:w3], img[:, h1:h2, w3:]
        imgA_31, imgA_32, imgA_33, imgA_34 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:w3], img[:, h2:h3, w3:]
        imgA_41, imgA_42, imgA_43, imgA_44 = img[:, h3:h4, :w1], img[:, h3:h4, w1:w2], img[:, h3:h4, w2:w3], img[:, h3:h4, w3:]
        imgA_51, imgA_52, imgA_53, imgA_54 = img[:, h4:,   :w1], img[:, h4:,   w1:w2], img[:, h4:,   w2:w3], img[:, h4:,   w3:]
        
        lblA_11, lblA_12, lblA_13, lblA_14 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:w3], lbl[     :h1, w3:]
        lblA_21, lblA_22, lblA_23, lblA_24 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:w3], lbl[   h1:h2, w3:]
        lblA_31, lblA_32, lblA_33, lblA_34 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:w3], lbl[   h2:h3, w3:]
        lblA_41, lblA_42, lblA_43, lblA_44 = lbl[   h3:h4, :w1], lbl[   h3:h4, w1:w2], lbl[   h3:h4, w2:w3], lbl[   h3:h4, w3:]
        lblA_51, lblA_52, lblA_53, lblA_54 = lbl[   h4:,   :w1], lbl[   h4:,   w1:w2], lbl[   h4:,   w2:w3], lbl[   h4:,   w3:]

        imgB_11, imgB_12, imgB_13, imgB_14 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:w3], img_aux[:,   :h1, w3:]
        imgB_21, imgB_22, imgB_23, imgB_24 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:w3], img_aux[:, h1:h2, w3:]
        imgB_31, imgB_32, imgB_33, imgB_34 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:w3], img_aux[:, h2:h3, w3:]
        imgB_41, imgB_42, imgB_43, imgB_44 = img_aux[:, h3:h4, :w1], img_aux[:, h3:h4, w1:w2], img_aux[:, h3:h4, w2:w3], img_aux[:, h3:h4, w3:]
        imgB_51, imgB_52, imgB_53, imgB_54 = img_aux[:, h4:,   :w1], img_aux[:, h4:,   w1:w2], img_aux[:, h4:,   w2:w3], img_aux[:, h4:,   w3:]

        lblB_11, lblB_12, lblB_13, lblB_14 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:w3], lbl_aux[     :h1, w3:]
        lblB_21, lblB_22, lblB_23, lblB_24 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:w3], lbl_aux[   h1:h2, w3:]
        lblB_31, lblB_32, lblB_33, lblB_34 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:w3], lbl_aux[   h2:h3, w3:]
        lblB_41, lblB_42, lblB_43, lblB_44 = lbl_aux[   h3:h4, :w1], lbl_aux[   h3:h4, w1:w2], lbl_aux[   h3:h4, w2:w3], lbl_aux[   h3:h4, w3:]
        lblB_51, lblB_52, lblB_53, lblB_54 = lbl_aux[   h4:,   :w1], lbl_aux[   h4:,   w1:w2], lbl_aux[   h4:,   w2:w3], lbl_aux[   h4:,   w3:]

        img_aux1_ro1 = torch.cat((imgA_11, imgB_12, imgA_13, imgB_14), dim=-1)
        img_aux1_ro2 = torch.cat((imgB_21, imgA_22, imgB_23, imgA_24), dim=-1)
        img_aux1_ro3 = torch.cat((imgA_31, imgB_32, imgA_33, imgB_34), dim=-1)
        img_aux1_ro4 = torch.cat((imgB_41, imgA_42, imgB_43, imgA_44), dim=-1)
        img_aux1_ro5 = torch.cat((imgA_51, imgB_52, imgA_53, imgB_54), dim=-1)
        img_aux1 = torch.cat((img_aux1_ro1, img_aux1_ro2, img_aux1_ro3, img_aux1_ro4, img_aux1_ro5), dim=-2)

        lbl_aux1_ro1 = torch.cat((lblA_11, lblB_12, lblA_13, lblB_14), dim=-1)
        lbl_aux1_ro2 = torch.cat((lblB_21, lblA_22, lblB_23, lblA_24), dim=-1)
        lbl_aux1_ro3 = torch.cat((lblA_31, lblB_32, lblA_33, lblB_34), dim=-1)
        lbl_aux1_ro4 = torch.cat((lblB_41, lblA_42, lblB_43, lblA_44), dim=-1)
        lbl_aux1_ro5 = torch.cat((lblA_51, lblB_52, lblA_53, lblB_54), dim=-1)
        lbl_aux1 = torch.cat((lbl_aux1_ro1, lbl_aux1_ro2, lbl_aux1_ro3, lbl_aux1_ro4, lbl_aux1_ro5), dim=-2)

        img_aux2_ro1 = torch.cat((imgB_11, imgA_12, imgB_13, imgA_14), dim=-1)
        img_aux2_ro2 = torch.cat((imgA_21, imgB_22, imgA_23, imgB_24), dim=-1)
        img_aux2_ro3 = torch.cat((imgB_31, imgA_32, imgB_33, imgA_34), dim=-1)
        img_aux2_ro4 = torch.cat((imgA_41, imgB_42, imgA_43, imgB_44), dim=-1)
        img_aux2_ro5 = torch.cat((imgB_51, imgA_52, imgB_53, imgA_54), dim=-1)
        img_aux2 = torch.cat((img_aux2_ro1, img_aux2_ro2, img_aux2_ro3, img_aux2_ro4, img_aux2_ro5), dim=-2)

        lbl_aux2_ro1 = torch.cat((lblB_11, lblA_12, lblB_13, lblA_14), dim=-1)
        lbl_aux2_ro2 = torch.cat((lblA_21, lblB_22, lblA_23, lblB_24), dim=-1)
        lbl_aux2_ro3 = torch.cat((lblB_31, lblA_32, lblB_33, lblA_34), dim=-1)
        lbl_aux2_ro4 = torch.cat((lblA_41, lblB_42, lblA_43, lblB_44), dim=-1)
        lbl_aux2_ro5 = torch.cat((lblB_51, lblA_52, lblB_53, lblA_54), dim=-1)
        lbl_aux2 = torch.cat((lbl_aux2_ro1, lbl_aux2_ro2, lbl_aux2_ro3, lbl_aux2_ro4, lbl_aux2_ro5), dim=-2)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col4row6(self, img, lbl, img_aux, lbl_aux):
        w1 = int(img.size()[-1] / 4)  # 1920/4 = 480
        w2 = int(img.size()[-1] / 2)  # 1920/2 = 960
        w3 = 3 * w1                   # 3 * 480 = 1440
        h1 = int(img.size()[-2] / 6)  # 32/6 = 5
        h2 = int(img.size()[-2] / 3)  # 32/3 = 10
        h3 = int(img.size()[-2] / 2)  # 32/2 = 16
        h4 = 4 * h1                   # 4 * 5 = 20
        h5 = 5 * h1                   # 5 * 5 = 25

        imgA_11, imgA_12, imgA_13, imgA_14 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:w3], img[:,   :h1, w3:]
        imgA_21, imgA_22, imgA_23, imgA_24 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:w3], img[:, h1:h2, w3:]
        imgA_31, imgA_32, imgA_33, imgA_34 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:w3], img[:, h2:h3, w3:]
        imgA_41, imgA_42, imgA_43, imgA_44 = img[:, h3:h4, :w1], img[:, h3:h4, w1:w2], img[:, h3:h4, w2:w3], img[:, h3:h4, w3:]
        imgA_51, imgA_52, imgA_53, imgA_54 = img[:, h4:h5, :w1], img[:, h4:h5, w1:w2], img[:, h4:h5, w2:w3], img[:, h4:h5, w3:]
        imgA_61, imgA_62, imgA_63, imgA_64 = img[:, h5:  , :w1], img[:, h5:  , w1:w2], img[:, h5:  , w2:w3], img[:, h5:  , w3:]
        
        lblA_11, lblA_12, lblA_13, lblA_14 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:w3], lbl[     :h1, w3:]
        lblA_21, lblA_22, lblA_23, lblA_24 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:w3], lbl[   h1:h2, w3:]
        lblA_31, lblA_32, lblA_33, lblA_34 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:w3], lbl[   h2:h3, w3:]
        lblA_41, lblA_42, lblA_43, lblA_44 = lbl[   h3:h4, :w1], lbl[   h3:h4, w1:w2], lbl[   h3:h4, w2:w3], lbl[   h3:h4, w3:]
        lblA_51, lblA_52, lblA_53, lblA_54 = lbl[   h4:h5, :w1], lbl[   h4:h5, w1:w2], lbl[   h4:h5, w2:w3], lbl[   h4:h5, w3:]
        lblA_61, lblA_62, lblA_63, lblA_64 = lbl[   h5:  , :w1], lbl[   h5:  , w1:w2], lbl[   h5:  , w2:w3], lbl[   h5:  , w3:]

        imgB_11, imgB_12, imgB_13, imgB_14 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:w3], img_aux[:,   :h1, w3:]
        imgB_21, imgB_22, imgB_23, imgB_24 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:w3], img_aux[:, h1:h2, w3:]
        imgB_31, imgB_32, imgB_33, imgB_34 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:w3], img_aux[:, h2:h3, w3:]
        imgB_41, imgB_42, imgB_43, imgB_44 = img_aux[:, h3:h4, :w1], img_aux[:, h3:h4, w1:w2], img_aux[:, h3:h4, w2:w3], img_aux[:, h3:h4, w3:]
        imgB_51, imgB_52, imgB_53, imgB_54 = img_aux[:, h4:h5, :w1], img_aux[:, h4:h5, w1:w2], img_aux[:, h4:h5, w2:w3], img_aux[:, h4:h5, w3:]
        imgB_61, imgB_62, imgB_63, imgB_64 = img_aux[:, h5:  , :w1], img_aux[:, h5:  , w1:w2], img_aux[:, h5:  , w2:w3], img_aux[:, h5:  , w3:]

        lblB_11, lblB_12, lblB_13, lblB_14 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:w3], lbl_aux[     :h1, w3:]
        lblB_21, lblB_22, lblB_23, lblB_24 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:w3], lbl_aux[   h1:h2, w3:]
        lblB_31, lblB_32, lblB_33, lblB_34 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:w3], lbl_aux[   h2:h3, w3:]
        lblB_41, lblB_42, lblB_43, lblB_44 = lbl_aux[   h3:h4, :w1], lbl_aux[   h3:h4, w1:w2], lbl_aux[   h3:h4, w2:w3], lbl_aux[   h3:h4, w3:]
        lblB_51, lblB_52, lblB_53, lblB_54 = lbl_aux[   h4:h5, :w1], lbl_aux[   h4:h5, w1:w2], lbl_aux[   h4:h5, w2:w3], lbl_aux[   h4:h5, w3:]
        lblB_61, lblB_62, lblB_63, lblB_64 = lbl_aux[   h5:  , :w1], lbl_aux[   h5:  , w1:w2], lbl_aux[   h5:  , w2:w3], lbl_aux[   h5:  , w3:]

        img_aux1_ro1 = torch.cat((imgA_11, imgB_12, imgA_13, imgB_14), dim=-1)
        img_aux1_ro2 = torch.cat((imgB_21, imgA_22, imgB_23, imgA_24), dim=-1)
        img_aux1_ro3 = torch.cat((imgA_31, imgB_32, imgA_33, imgB_34), dim=-1)
        img_aux1_ro4 = torch.cat((imgB_41, imgA_42, imgB_43, imgA_44), dim=-1)
        img_aux1_ro5 = torch.cat((imgA_51, imgB_52, imgA_53, imgB_54), dim=-1)
        img_aux1_ro6 = torch.cat((imgB_61, imgA_62, imgB_63, imgA_64), dim=-1)
        img_aux1 = torch.cat((img_aux1_ro1, img_aux1_ro2, img_aux1_ro3, img_aux1_ro4, img_aux1_ro5, img_aux1_ro6), dim=-2)

        lbl_aux1_ro1 = torch.cat((lblA_11, lblB_12, lblA_13, lblB_14), dim=-1)
        lbl_aux1_ro2 = torch.cat((lblB_21, lblA_22, lblB_23, lblA_24), dim=-1)
        lbl_aux1_ro3 = torch.cat((lblA_31, lblB_32, lblA_33, lblB_34), dim=-1)
        lbl_aux1_ro4 = torch.cat((lblB_41, lblA_42, lblB_43, lblA_44), dim=-1)
        lbl_aux1_ro5 = torch.cat((lblA_51, lblB_52, lblA_53, lblB_54), dim=-1)
        lbl_aux1_ro6 = torch.cat((lblB_61, lblA_62, lblB_63, lblA_64), dim=-1)
        lbl_aux1 = torch.cat((lbl_aux1_ro1, lbl_aux1_ro2, lbl_aux1_ro3, lbl_aux1_ro4, lbl_aux1_ro5, lbl_aux1_ro6), dim=-2)

        img_aux2_ro1 = torch.cat((imgB_11, imgA_12, imgB_13, imgA_14), dim=-1)
        img_aux2_ro2 = torch.cat((imgA_21, imgB_22, imgA_23, imgB_24), dim=-1)
        img_aux2_ro3 = torch.cat((imgB_31, imgA_32, imgB_33, imgA_34), dim=-1)
        img_aux2_ro4 = torch.cat((imgA_41, imgB_42, imgA_43, imgB_44), dim=-1)
        img_aux2_ro5 = torch.cat((imgB_51, imgA_52, imgB_53, imgA_54), dim=-1)
        img_aux2_ro6 = torch.cat((imgA_61, imgB_62, imgA_63, imgB_64), dim=-1)
        img_aux2 = torch.cat((img_aux2_ro1, img_aux2_ro2, img_aux2_ro3, img_aux2_ro4, img_aux2_ro5, img_aux2_ro6), dim=-2)

        lbl_aux2_ro1 = torch.cat((lblB_11, lblA_12, lblB_13, lblA_14), dim=-1)
        lbl_aux2_ro2 = torch.cat((lblA_21, lblB_22, lblA_23, lblB_24), dim=-1)
        lbl_aux2_ro3 = torch.cat((lblB_31, lblA_32, lblB_33, lblA_34), dim=-1)
        lbl_aux2_ro4 = torch.cat((lblA_41, lblB_42, lblA_43, lblB_44), dim=-1)
        lbl_aux2_ro5 = torch.cat((lblB_51, lblA_52, lblB_53, lblA_54), dim=-1)
        lbl_aux2_ro6 = torch.cat((lblA_61, lblB_62, lblA_63, lblB_64), dim=-1)
        lbl_aux2 = torch.cat((lbl_aux2_ro1, lbl_aux2_ro2, lbl_aux2_ro3, lbl_aux2_ro4, lbl_aux2_ro5, lbl_aux2_ro6), dim=-2)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col6row4(self, img, lbl, img_aux, lbl_aux):  ##
        w1 = int(img.size()[-1] / 6)  # 1920/6 = 320
        w2 = int(img.size()[-1] / 3)  # 1920/3 = 640
        w3 = int(img.size()[-1] / 2)  # 1920/2 = 960
        w4 = 4 * w1                   # 4 * 320 = 1280
        w5 = 5 * w1                   # 5 * 320 = 1600
        h1 = int(img.size()[-2] / 4)  # 32/4 = 8
        h2 = int(img.size()[-2] / 2)  # 32/2 = 16
        h3 = 3 * h1                   # 3 * 24 = 24

        imgA_11, imgA_12, imgA_13, imgA_14, imgA_15, imgA_16 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:w3], img[:,   :h1, w3:w4], img[:,   :h1, w4:w5], img[:,   :h1, w5:]
        imgA_21, imgA_22, imgA_23, imgA_24, imgA_25, imgA_26 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:w3], img[:, h1:h2, w3:w4], img[:, h1:h2, w4:w5], img[:, h1:h2, w5:]
        imgA_31, imgA_32, imgA_33, imgA_34, imgA_35, imgA_36 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:w3], img[:, h2:h3, w3:w4], img[:, h2:h3, w4:w5], img[:, h2:h3, w5:]
        imgA_41, imgA_42, imgA_43, imgA_44, imgA_45, imgA_46 = img[:, h3:  , :w1], img[:, h3:  , w1:w2], img[:, h3:  , w2:w3], img[:, h3:  , w3:w4], img[:, h3:  , w4:w5], img[:, h3:  , w5:]
        
        lblA_11, lblA_12, lblA_13, lblA_14, lblA_15, lblA_16 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:w3], lbl[     :h1, w3:w4], lbl[     :h1, w4:w5], lbl[     :h1, w5:]
        lblA_21, lblA_22, lblA_23, lblA_24, lblA_25, lblA_26 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:w3], lbl[   h1:h2, w3:w4], lbl[   h1:h2, w4:w5], lbl[   h1:h2, w5:]
        lblA_31, lblA_32, lblA_33, lblA_34, lblA_35, lblA_36 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:w3], lbl[   h2:h3, w3:w4], lbl[   h2:h3, w4:w5], lbl[   h2:h3, w5:]
        lblA_41, lblA_42, lblA_43, lblA_44, lblA_45, lblA_46 = lbl[   h3:  , :w1], lbl[   h3:  , w1:w2], lbl[   h3:  , w2:w3], lbl[   h3:  , w3:w4], lbl[   h3:  , w4:w5], lbl[   h3:  , w5:]

        imgB_11, imgB_12, imgB_13, imgB_14, imgB_15, imgB_16 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:w3], img_aux[:,   :h1, w3:w4], img_aux[:,   :h1, w4:w5], img_aux[:,   :h1, w5:]
        imgB_21, imgB_22, imgB_23, imgB_24, imgB_25, imgB_26 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:w3], img_aux[:, h1:h2, w3:w4], img_aux[:, h1:h2, w4:w5], img_aux[:, h1:h2, w5:]
        imgB_31, imgB_32, imgB_33, imgB_34, imgB_35, imgB_36 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:w3], img_aux[:, h2:h3, w3:w4], img_aux[:, h2:h3, w4:w5], img_aux[:, h2:h3, w5:]
        imgB_41, imgB_42, imgB_43, imgB_44, imgB_45, imgB_46 = img_aux[:, h3:  , :w1], img_aux[:, h3:  , w1:w2], img_aux[:, h3:  , w2:w3], img_aux[:, h3:  , w3:w4], img_aux[:, h3:  , w4:w5], img_aux[:, h3:  , w5:]

        lblB_11, lblB_12, lblB_13, lblB_14, lblB_15, lblB_16 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:w3], lbl_aux[     :h1, w3:w4], lbl_aux[     :h1, w4:w5], lbl_aux[     :h1, w5:]
        lblB_21, lblB_22, lblB_23, lblB_24, lblB_25, lblB_26 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:w3], lbl_aux[   h1:h2, w3:w4], lbl_aux[   h1:h2, w4:w5], lbl_aux[   h1:h2, w5:]
        lblB_31, lblB_32, lblB_33, lblB_34, lblB_35, lblB_36 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:w3], lbl_aux[   h2:h3, w3:w4], lbl_aux[   h2:h3, w4:w5], lbl_aux[   h2:h3, w5:]
        lblB_41, lblB_42, lblB_43, lblB_44, lblB_45, lblB_46 = lbl_aux[   h3:  , :w1], lbl_aux[   h3:  , w1:w2], lbl_aux[   h3:  , w2:w3], lbl_aux[   h3:  , w3:w4], lbl_aux[   h3:  , w4:w5], lbl_aux[   h3:  , w5:]

        img_aux1_ro1 = torch.cat((imgA_11, imgB_12, imgA_13, imgB_14, imgA_15, imgB_16), dim=-1)
        img_aux1_ro2 = torch.cat((imgB_21, imgA_22, imgB_23, imgA_24, imgB_25, imgA_26), dim=-1)
        img_aux1_ro3 = torch.cat((imgA_31, imgB_32, imgA_33, imgB_34, imgA_35, imgB_36), dim=-1)
        img_aux1_ro4 = torch.cat((imgB_41, imgA_42, imgB_43, imgA_44, imgB_45, imgA_46), dim=-1)
        img_aux1 = torch.cat((img_aux1_ro1, img_aux1_ro2, img_aux1_ro3, img_aux1_ro4), dim=-2)

        lbl_aux1_ro1 = torch.cat((lblA_11, lblB_12, lblA_13, lblB_14, lblA_15, lblB_16), dim=-1)
        lbl_aux1_ro2 = torch.cat((lblB_21, lblA_22, lblB_23, lblA_24, lblB_25, lblA_26), dim=-1)
        lbl_aux1_ro3 = torch.cat((lblA_31, lblB_32, lblA_33, lblB_34, lblA_35, lblB_36), dim=-1)
        lbl_aux1_ro4 = torch.cat((lblB_41, lblA_42, lblB_43, lblA_44, lblB_45, lblA_46), dim=-1)
        lbl_aux1 = torch.cat((lbl_aux1_ro1, lbl_aux1_ro2, lbl_aux1_ro3, lbl_aux1_ro4), dim=-2)

        img_aux2_ro1 = torch.cat((imgB_11, imgA_12, imgB_13, imgA_14, imgB_15, imgA_16), dim=-1)
        img_aux2_ro2 = torch.cat((imgA_21, imgB_22, imgA_23, imgB_24, imgA_25, imgB_26), dim=-1)
        img_aux2_ro3 = torch.cat((imgB_31, imgA_32, imgB_33, imgA_34, imgB_35, imgA_36), dim=-1)
        img_aux2_ro4 = torch.cat((imgA_41, imgB_42, imgA_43, imgB_44, imgA_45, imgB_46), dim=-1)
        img_aux2 = torch.cat((img_aux2_ro1, img_aux2_ro2, img_aux2_ro3, img_aux2_ro4), dim=-2)

        lbl_aux2_ro1 = torch.cat((lblB_11, lblA_12, lblB_13, lblA_14, lblB_15, lblA_16), dim=-1)
        lbl_aux2_ro2 = torch.cat((lblA_21, lblB_22, lblA_23, lblB_24, lblA_25, lblB_26), dim=-1)
        lbl_aux2_ro3 = torch.cat((lblB_31, lblA_32, lblB_33, lblA_34, lblB_35, lblA_36), dim=-1)
        lbl_aux2_ro4 = torch.cat((lblA_41, lblB_42, lblA_43, lblB_44, lblA_45, lblB_46), dim=-1)
        lbl_aux2 = torch.cat((lbl_aux2_ro1, lbl_aux2_ro2, lbl_aux2_ro3, lbl_aux2_ro4), dim=-2)

        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


    def col16row1(self, img, lbl, img_aux, lbl_aux):  # [bs, 6, 32, 1920]
        w1 = int(img.size()[-1] / 16)  # 1920/16 = 120
        w2 = int(img.size()[-1] / 8)   # 1920/8  = 240
        w3 = 3 * w1                    # 3 * 120 = 360
        w4 = int(img.size()[-1] / 4)   # 1920/4  = 480
        w5 = 5 * w1                    # 5 * 120 = 600
        w6 = 6 * w1                    # 6 * 120 = 720
        w7 = 7 * w1                    # 7 * 120 = 840
        w8 = 8 * w1                    # 8 * 120 = 960
        w9 = 9 * w1                    # 9 * 120 = 1080
        w10 = 10 * w1                  # 10 * 120 = 1200
        w11 = 11 * w1                  # 11 * 120 = 1320
        w12 = 12 * w1                  # 12 * 120 = 1440
        w13 = 13 * w1                  # 13 * 120 = 1560
        w14 = 14 * w1                  # 14 * 120 = 1680
        w15 = 15 * w1                  # 15 * 120 = 1800
        
        imgA_1,  imgA_2,  imgA_3,  imgA_4  = img[:, :,    :w1 ], img[:, :,  w1:w2 ], img[:, :,  w2:w3 ], img[:, :,  w3:w4 ]
        imgA_5,  imgA_6,  imgA_7,  imgA_8  = img[:, :,  w4:w5 ], img[:, :,  w5:w6 ], img[:, :,  w6:w7 ], img[:, :,  w7:w8 ]
        imgA_9,  imgA_10, imgA_11, imgA_12 = img[:, :,  w8:w9 ], img[:, :,  w9:w10], img[:, :, w10:w11], img[:, :, w11:w12]
        imgA_13, imgA_14, imgA_15, imgA_16 = img[:, :, w12:w13], img[:, :, w13:w14], img[:, :, w14:w15], img[:, :, w15:   ]

        lblA_1,  lblA_2,  lblA_3,  lblA_4  = lbl[   :,    :w1 ], lbl[   :,  w1:w2 ], lbl[   :,  w2:w3 ], lbl[    :,  w3:w4 ]
        lblA_5,  lblA_6,  lblA_7,  lblA_8  = lbl[   :,  w4:w5 ], lbl[   :,  w5:w6 ], lbl[   :,  w6:w7 ], lbl[    :,  w7:w8 ]
        lblA_9,  lblA_10, lblA_11, lblA_12 = lbl[   :,  w8:w9 ], lbl[   :,  w9:w10], lbl[   :, w10:w11], lbl[    :, w11:w12]
        lblA_13, lblA_14, lblA_15, lblA_16 = lbl[   :, w12:w13], lbl[   :, w13:w14], lbl[   :, w14:w15], lbl[    :, w15:   ]
        
        imgB_1,  imgB_2,  imgB_3,  imgB_4  = img_aux[:, :,    :w1 ], img_aux[:, :,  w1:w2 ], img_aux[:, :,  w2:w3 ], img_aux[:, :,  w3:w4 ]
        imgB_5,  imgB_6,  imgB_7,  imgB_8  = img_aux[:, :,  w4:w5 ], img_aux[:, :,  w5:w6 ], img_aux[:, :,  w6:w7 ], img_aux[:, :,  w7:w8 ]
        imgB_9,  imgB_10, imgB_11, imgB_12 = img_aux[:, :,  w8:w9 ], img_aux[:, :,  w9:w10], img_aux[:, :, w10:w11], img_aux[:, :, w11:w12]
        imgB_13, imgB_14, imgB_15, imgB_16 = img_aux[:, :, w12:w13], img_aux[:, :, w13:w14], img_aux[:, :, w14:w15], img_aux[:, :, w15:   ]

        lblB_1,  lblB_2,  lblB_3,  lblB_4  = lbl_aux[   :,    :w1 ], lbl_aux[   :,  w1:w2 ], lbl_aux[   :,  w2:w3 ], lbl_aux[   :,  w3:w4 ]
        lblB_5,  lblB_6,  lblB_7,  lblB_8  = lbl_aux[   :,  w4:w5 ], lbl_aux[   :,  w5:w6 ], lbl_aux[   :,  w6:w7 ], lbl_aux[   :,  w7:w8 ]
        lblB_9,  lblB_10, lblB_11, lblB_12 = lbl_aux[   :,  w8:w9 ], lbl_aux[   :,  w9:w10], lbl_aux[   :, w10:w11], lbl_aux[   :, w11:w12]
        lblB_13, lblB_14, lblB_15, lblB_16 = lbl_aux[   :, w12:w13], lbl_aux[   :, w13:w14], lbl_aux[   :, w14:w15], lbl_aux[   :, w15:   ]
        
        img_aux1 = torch.cat((imgA_1, imgB_2, imgA_3, imgB_4, imgA_5, imgB_6, imgA_7, imgB_8, imgA_9, imgB_10, imgA_11, imgB_12, imgA_13, imgB_14, imgA_15, imgB_16), dim=-1)
        lbl_aux1 = torch.cat((lblA_1, lblB_2, lblA_3, lblB_4, lblA_5, lblB_6, lblA_7, lblB_8, lblA_9, lblB_10, lblA_11, lblB_12, lblA_13, lblB_14, lblA_15, lblB_16), dim=-1)

        img_aux2 = torch.cat((imgB_1, imgA_2, imgB_3, imgA_4, imgB_5, imgA_6, imgB_7, imgA_8, imgB_9, imgA_10, imgB_11, imgA_12, imgB_13, imgA_14, imgB_15, imgA_16), dim=-1)
        lbl_aux2 = torch.cat((lblB_1, lblA_2, lblB_3, lblA_4, lblB_5, lblA_6, lblB_7, lblA_8, lblB_9, lblA_10, lblB_11, lblA_12, lblB_13, lblA_14, lblB_15, lblA_16), dim=-1)
        
        return img_aux1, lbl_aux1, img_aux2, lbl_aux2


class MixTeacherSemkitti:
    
    def __init__(self, strategy,):
        super(MixTeacherSemkitti, self).__init__()
        self.strategy = strategy
        
    def forward(self, image, label, mask, image_aux, label_aux, mask_aux):
        """
        Arguments:
            - strategy: MixTeacher strategies.
            - image: original image, size: [6, H, W].
            - label: original label, size: [H, W].
            - mask:  original mask,  size: [H, W].
            - image_aux: auxiliary image, size: [6, H, W].
            - label_aux: auxiliary label, size: [H, W].
            - mask_aux:  auxiliary mask,  size: [H, W].
        Return:
            (2x) Augmented images, labels, and masks.
        """

        image, image_aux = np.transpose(image, (2, 0, 1)), np.transpose(image_aux, (2, 0, 1))

                
        if self.strategy == 'mixture':
            strategies = ['col1row2', 'col1row3', 'col2row1', 'col3row1', 'col2row2', 'col1row4', 'col2row4']
            strategy = np.random.choice(strategies, size=1)[0]

        elif self.strategy == 'mixtureV2':
            strategies = ['col1row3', 'col1row4', 'col1row5', 'col1row6', 'col2row3', 'col2row4', 'col2row5', 'col2row6', 'col3row3', 'col3row4', 'col3row5', 'col3row6', 'col4row3', 'col4row4', 'col4row5', 'col4row6', 'col6row4', ]
            strategy = np.random.choice(strategies, size=1)[0]

        else:
            strategy = self.strategy
            
        if strategy == 'col1row2':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col1row2(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col1row3':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col1row3(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col1row4':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col1row4(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col1row5':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col1row5(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col1row6':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col1row6(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col2row1':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col2row1(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col2row2':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col2row2(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col2row3':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col2row3(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col2row4':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col2row4(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col2row5':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col2row5(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col2row6':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col2row6(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col3row1':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col3row1(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col3row2':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col3row2(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col3row3':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col3row3(image, label, mask, image_aux, label_aux, mask_aux)
        
        elif strategy == 'col3row4':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col3row4(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col3row5':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col3row5(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col3row6':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col3row6(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col4row1':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col4row1(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col4row2':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col4row2(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col4row3':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col4row3(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col4row4':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col4row4(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col4row5':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col4row5(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col4row6':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col4row6(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col6row4':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col6row4(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'cutmix':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.cutmix(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'cutout':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.cutout(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'mixup':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.mixup(image, label, mask, image_aux, label_aux, mask_aux)


        img_aux1, img_aux2 = np.transpose(img_aux1, (1, 2, 0)), np.transpose(img_aux2, (1, 2, 0))

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2, strategy
        
        
    def col1row2(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        mid_h = int(img.shape[-2] / 2)  # 64/2 = 32

        imgA_1, imgA_2 = img[:, :mid_h, :], img[:, mid_h:, :]  # upper-half1, lower-half1
        lblA_1, lblA_2 = lbl[   :mid_h, :], lbl[   mid_h:, :]  # upper-half1, lower-half1
        mskA_1, mskA_2 = msk[   :mid_h, :], msk[   mid_h:, :]  # upper-half1, lower-half1

        imgB_1, imgB_2 = img_aux[:, :mid_h, :], img_aux[:, mid_h:, :]  # upper-half2, lower-half2
        lblB_1, lblB_2 = lbl_aux[   :mid_h, :], lbl_aux[   mid_h:, :]  # upper-half2, lower-half2
        mskB_1, mskB_2 = msk_aux[   :mid_h, :], msk_aux[   mid_h:, :]  # upper-half2, lower-half2

        img_aux1 = np.concatenate((imgA_1, imgB_2), axis=-2)  # upper-half1, lower-half2
        lbl_aux1 = np.concatenate((lblA_1, lblB_2), axis=-2)  # upper-half1, lower-half2
        msk_aux1 = np.concatenate((mskA_1, mskB_2), axis=-2)  # upper-half1, lower-half2

        img_aux2 = np.concatenate((imgB_1, imgA_2), axis=-2)  # upper-half2, lower-half1
        lbl_aux2 = np.concatenate((lblB_1, lblA_2), axis=-2)  # upper-half2, lower-half1
        msk_aux2 = np.concatenate((mskB_1, mskA_2), axis=-2)  # upper-half2, lower-half1
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col1row3(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 3)  # 64/3 = 21
        h2 = 2 * h1                   # 21*2 = 42

        imgA_1, imgA_2, imgA_3 = img[:, :h1, :], img[:, h1:h2, :], img[:, h2:, :]  # upper1, middle1, lower1
        lblA_1, lblA_2, lblA_3 = lbl[   :h1, :], lbl[   h1:h2, :], lbl[   h2:, :]  # upper1, middle1, lower1
        mskA_1, mskA_2, mskA_3 = msk[   :h1, :], msk[   h1:h2, :], msk[   h2:, :]  # upper1, middle1, lower1

        imgB_1, imgB_2, imgB_3 = img_aux[:, :h1, :], img_aux[:, h1:h2, :], img_aux[:, h2:, :]  # upper2, middle2, lower2
        lblB_1, lblB_2, lblB_3 = lbl_aux[   :h1, :], lbl_aux[   h1:h2, :], lbl_aux[   h2:, :]  # upper2, middle2, lower2
        mskB_1, mskB_2, mskB_3 = msk_aux[   :h1, :], msk_aux[   h1:h2, :], msk_aux[   h2:, :]  # upper2, middle2, lower2

        img_aux1 = np.concatenate((imgA_1, imgB_2, imgA_3), axis=-2)  # upper1, middle2, lower1
        lbl_aux1 = np.concatenate((lblA_1, lblB_2, lblA_3), axis=-2)  # upper1, middle2, lower1
        msk_aux1 = np.concatenate((mskA_1, mskB_2, mskA_3), axis=-2)  # upper1, middle2, lower1

        img_aux2 = np.concatenate((imgB_1, imgA_2, imgB_3), axis=-2)  # upper2, middle1, lower2
        lbl_aux2 = np.concatenate((lblB_1, lblA_2, lblB_3), axis=-2)  # upper2, middle1, lower2
        msk_aux2 = np.concatenate((mskB_1, mskA_2, mskB_3), axis=-2)  # upper2, middle1, lower2
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col1row4(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 4)     # 64/4 = 16
        mid_h = int(img.shape[-2] / 2)  # 64/2 = 32
        h3 = 3 * h1                     # 16*3 = 48

        imgA_1, imgA_2, imgA_3, imgA_4 = img[:, :h1, :], img[:, h1:mid_h, :], img[:, mid_h:h3, :], img[:, h3:, :]
        lblA_1, lblA_2, lblA_3, lblA_4 = lbl[   :h1, :], lbl[   h1:mid_h, :], lbl[   mid_h:h3, :], lbl[   h3:, :]
        mskA_1, mskA_2, mskA_3, mskA_4 = msk[   :h1, :], msk[   h1:mid_h, :], msk[   mid_h:h3, :], msk[   h3:, :]

        imgB_1, imgB_2, imgB_3, imgB_4 = img_aux[:, :h1, :], img_aux[:, h1:mid_h, :], img_aux[:, mid_h:h3, :], img_aux[:, h3:, :]
        lblB_1, lblB_2, lblB_3, lblB_4 = lbl_aux[   :h1, :], lbl_aux[   h1:mid_h, :], lbl_aux[   mid_h:h3, :], lbl_aux[   h3:, :]
        mskB_1, mskB_2, mskB_3, mskB_4 = msk_aux[   :h1, :], msk_aux[   h1:mid_h, :], msk_aux[   mid_h:h3, :], msk_aux[   h3:, :]

        img_aux1 = np.concatenate((imgA_1, imgB_2, imgA_3, imgB_4), axis=-2)
        lbl_aux1 = np.concatenate((lblA_1, lblB_2, lblA_3, lblB_4), axis=-2)
        msk_aux1 = np.concatenate((mskA_1, mskB_2, mskA_3, mskB_4), axis=-2)

        img_aux2 = np.concatenate((imgB_1, imgA_2, imgB_3, imgA_4), axis=-2)
        lbl_aux2 = np.concatenate((lblB_1, lblA_2, lblB_3, lblA_4), axis=-2)
        msk_aux2 = np.concatenate((mskB_1, mskA_2, mskB_3, mskA_4), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2
    
    
    def col1row5(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 5)      # 64/5 = 12
        h2 = 2 * h1                      # 2*12 = 24
        h3 = 3 * h1                      # 3*12 = 36
        h4 = 4 * h1                      # 4*12 = 48

        imgA_1, imgA_2, imgA_3, imgA_4, imgA_5 = img[:, :h1, :], img[:, h1:h2, :], img[:, h2:h3, :], img[:, h3:h4, :], img[:, h4:, :]
        lblA_1, lblA_2, lblA_3, lblA_4, lblA_5 = lbl[   :h1, :], lbl[   h1:h2, :], lbl[   h2:h3, :], lbl[   h3:h4, :], lbl[   h4:, :]
        mskA_1, mskA_2, mskA_3, mskA_4, mskA_5 = msk[   :h1, :], msk[   h1:h2, :], msk[   h2:h3, :], msk[   h3:h4, :], msk[   h4:, :]

        imgB_1, imgB_2, imgB_3, imgB_4, imgB_5 = img_aux[:, :h1, :], img_aux[:, h1:h2, :], img_aux[:, h2:h3, :], img_aux[:, h3:h4, :], img_aux[:, h4:, :]
        lblB_1, lblB_2, lblB_3, lblB_4, lblB_5 = lbl_aux[   :h1, :], lbl_aux[   h1:h2, :], lbl_aux[   h2:h3, :], lbl_aux[   h3:h4, :], lbl_aux[   h4:, :]
        mskB_1, mskB_2, mskB_3, mskB_4, mskB_5 = msk_aux[   :h1, :], msk_aux[   h1:h2, :], msk_aux[   h2:h3, :], msk_aux[   h3:h4, :], msk_aux[   h4:, :]

        img_aux1 = np.concatenate((imgA_1, imgB_2, imgA_3, imgB_4, imgA_5), axis=-2)
        lbl_aux1 = np.concatenate((lblA_1, lblB_2, lblA_3, lblB_4, lblA_5), axis=-2)
        msk_aux1 = np.concatenate((mskA_1, mskB_2, mskA_3, mskB_4, mskA_5), axis=-2)

        img_aux2 = np.concatenate((imgB_1, imgA_2, imgB_3, imgA_4, imgB_5), axis=-2)
        lbl_aux2 = np.concatenate((lblB_1, lblA_2, lblB_3, lblA_4, lblB_5), axis=-2)
        msk_aux2 = np.concatenate((mskB_1, mskA_2, mskB_3, mskA_4, mskB_5), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col1row6(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 6)     # 64/6 = 10
        h2 = 2 * h1                      # 2*10 = 20
        h3 = 3 * h1                      # 3*10 = 30
        h4 = 4 * h1                      # 4*10 = 40
        h5 = 5 * h1                      # 5*10 = 50

        imgA_1, imgA_2, imgA_3, imgA_4, imgA_5, imgA_6 = img[:, :h1, :], img[:, h1:h2, :], img[:, h2:h3, :], img[:, h3:h4, :], img[:, h4:h5, :], img[:, h5:, :]
        lblA_1, lblA_2, lblA_3, lblA_4, lblA_5, lblA_6 = lbl[   :h1, :], lbl[   h1:h2, :], lbl[   h2:h3, :], lbl[   h3:h4, :], lbl[   h4:h5, :], lbl[   h5:, :]
        mskA_1, mskA_2, mskA_3, mskA_4, mskA_5, mskA_6 = msk[   :h1, :], msk[   h1:h2, :], msk[   h2:h3, :], msk[   h3:h4, :], msk[   h4:h5, :], msk[   h5:, :]

        imgB_1, imgB_2, imgB_3, imgB_4, imgB_5, imgB_6 = img_aux[:, :h1, :], img_aux[:, h1:h2, :], img_aux[:, h2:h3, :], img_aux[:, h3:h4, :], img_aux[:, h4:h5, :], img_aux[:, h5:, :]
        lblB_1, lblB_2, lblB_3, lblB_4, lblB_5, lblB_6 = lbl_aux[   :h1, :], lbl_aux[   h1:h2, :], lbl_aux[   h2:h3, :], lbl_aux[   h3:h4, :], lbl_aux[   h4:h5, :], lbl_aux[   h5:, :]
        mskB_1, mskB_2, mskB_3, mskB_4, mskB_5, mskB_6 = msk_aux[   :h1, :], msk_aux[   h1:h2, :], msk_aux[   h2:h3, :], msk_aux[   h3:h4, :], msk_aux[   h4:h5, :], msk_aux[   h5:, :]

        img_aux1 = np.concatenate((imgA_1, imgB_2, imgA_3, imgB_4, imgA_5, imgB_6), axis=-2)
        lbl_aux1 = np.concatenate((lblA_1, lblB_2, lblA_3, lblB_4, lblA_5, lblB_6), axis=-2)
        msk_aux1 = np.concatenate((mskA_1, mskB_2, mskA_3, mskB_4, mskA_5, mskB_6), axis=-2)

        img_aux2 = np.concatenate((imgB_1, imgA_2, imgB_3, imgA_4, imgB_5, imgA_6), axis=-2)
        lbl_aux2 = np.concatenate((lblB_1, lblA_2, lblB_3, lblA_4, lblB_5, lblA_6), axis=-2)
        msk_aux2 = np.concatenate((mskB_1, mskA_2, mskB_3, mskA_4, mskB_5, mskA_6), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col2row1(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        mid_w = int(img.shape[-1] / 2)  # 2048/2 = 1024
        
        imgA_1, imgA_2 = img[:, :, :mid_w], img[:, :, mid_w:]  # left-half1, right-half1
        lblA_1, lblA_2 = lbl[   :, :mid_w], lbl[   :, mid_w:]  # left-half1, right-half1
        mskA_1, mskA_2 = msk[   :, :mid_w], msk[   :, mid_w:]  # left-half1, right-half1
        
        imgB_1, imgB_2 = img_aux[:, :, :mid_w], img_aux[:, :, mid_w:]  # left-half2, right-half2
        lblB_1, lblB_2 = lbl_aux[   :, :mid_w], lbl_aux[   :, mid_w:]  # left-half2, right-half2
        mskB_1, mskB_2 = msk_aux[   :, :mid_w], msk_aux[   :, mid_w:]  # left-half2, right-half2
        
        img_aux1 = np.concatenate((imgA_1, imgB_2), axis=-1)  # left-half1, right-half2
        lbl_aux1 = np.concatenate((lblA_1, lblB_2), axis=-1)  # left-half1, right-half2
        msk_aux1 = np.concatenate((mskA_1, mskB_2), axis=-1)  # left-half1, right-half2

        img_aux2 = np.concatenate((imgB_1, imgA_2), axis=-1)  # left-half2, right-half1
        lbl_aux2 = np.concatenate((lblB_1, lblA_2), axis=-1)  # left-half2, right-half1
        msk_aux2 = np.concatenate((mskB_1, mskA_2), axis=-1)  # left-half2, right-half1
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col2row2(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        mid_h = int(img.shape[-2] / 2)  # 64/2 = 32
        mid_w = int(img.shape[-1] / 2)  # 2048/2 = 1024
        
        imgA_11, imgA_12, imgA_21, imgA_22 = img[:, :mid_h, :mid_w], img[:, :mid_h, mid_w:], img[:, mid_h:, :mid_w], img[:, mid_h:, mid_w:]
        lblA_11, lblA_12, lblA_21, lblA_22 = lbl[   :mid_h, :mid_w], lbl[   :mid_h, mid_w:], lbl[   mid_h:, :mid_w], lbl[   mid_h:, mid_w:]
        mskA_11, mskA_12, mskA_21, mskA_22 = msk[   :mid_h, :mid_w], msk[   :mid_h, mid_w:], msk[   mid_h:, :mid_w], msk[   mid_h:, mid_w:]
        
        imgB_11, imgB_12, imgB_21, imgB_22 = img_aux[:, :mid_h, :mid_w], img_aux[:, :mid_h, mid_w:], img_aux[:, mid_h:, :mid_w], img_aux[:, mid_h:, mid_w:]
        lblB_11, lblB_12, lblB_21, lblB_22 = lbl_aux[   :mid_h, :mid_w], lbl_aux[   :mid_h, mid_w:], lbl_aux[   mid_h:, :mid_w], lbl_aux[   mid_h:, mid_w:]
        mskB_11, mskB_12, mskB_21, mskB_22 = msk_aux[   :mid_h, :mid_w], msk_aux[   :mid_h, mid_w:], msk_aux[   mid_h:, :mid_w], msk_aux[   mid_h:, mid_w:]

        concat_img_aux1_top = np.concatenate((imgA_11, imgB_12), axis=-1)
        concat_img_aux1_bot = np.concatenate((imgB_21, imgA_22), axis=-1)
        img_aux1 = np.concatenate((concat_img_aux1_top, concat_img_aux1_bot), axis=-2)
        
        concat_lbl_aux1_top = np.concatenate((lblA_11, lblB_12), axis=-1)
        concat_lbl_aux1_bot = np.concatenate((lblB_21, lblA_22), axis=-1)
        lbl_aux1 = np.concatenate((concat_lbl_aux1_top, concat_lbl_aux1_bot), axis=-2)
        
        concat_msk_aux1_top = np.concatenate((mskA_11, mskB_12), axis=-1)
        concat_msk_aux1_bot = np.concatenate((mskB_21, mskA_22), axis=-1)
        msk_aux1 = np.concatenate((concat_msk_aux1_top, concat_msk_aux1_bot), axis=-2)
        
        concat_img_aux2_top = np.concatenate((imgB_11, imgA_12), axis=-1)
        concat_img_aux2_bot = np.concatenate((imgA_21, imgB_22), axis=-1)
        img_aux2 = np.concatenate((concat_img_aux2_top, concat_img_aux2_bot), axis=-2)
        
        concat_lbl_aux2_top = np.concatenate((lblB_11, lblA_12), axis=-1)
        concat_lbl_aux2_bot = np.concatenate((lblA_21, lblB_22), axis=-1)
        lbl_aux2 = np.concatenate((concat_lbl_aux2_top, concat_lbl_aux2_bot), axis=-2)
        
        concat_msk_aux2_top = np.concatenate((mskB_11, mskA_12), axis=-1)
        concat_msk_aux2_bot = np.concatenate((mskA_21, mskB_22), axis=-1)
        msk_aux2 = np.concatenate((concat_msk_aux2_top, concat_msk_aux2_bot), axis=-2)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col2row3(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 3)      # 64/3 = 21
        h2 = 2 * h1                      # 2*21 = 42
        mid_w = int(img.shape[-1] / 2)   # 2048/2 = 1024

        imgA_11, imgA_21, imgA_31 = img[:, :h1, :mid_w], img[:, h1:h2, :mid_w], img[:, h2:, :mid_w]
        imgA_12, imgA_22, imgA_32 = img[:, :h1, mid_w:], img[:, h1:h2, mid_w:], img[:, h2:, mid_w:]
        lblA_11, lblA_21, lblA_31 = lbl[   :h1, :mid_w], lbl[   h1:h2, :mid_w], lbl[   h2:, :mid_w]
        lblA_12, lblA_22, lblA_32 = lbl[   :h1, mid_w:], lbl[   h1:h2, mid_w:], lbl[   h2:, mid_w:]
        mskA_11, mskA_21, mskA_31 = msk[   :h1, :mid_w], msk[   h1:h2, :mid_w], msk[   h2:, :mid_w]
        mskA_12, mskA_22, mskA_32 = msk[   :h1, mid_w:], msk[   h1:h2, mid_w:], msk[   h2:, mid_w:]

        imgB_11, imgB_21, imgB_31 = img_aux[:, :h1, :mid_w], img_aux[:, h1:h2, :mid_w], img_aux[:, h2:, :mid_w]
        imgB_12, imgB_22, imgB_32 = img_aux[:, :h1, mid_w:], img_aux[:, h1:h2, mid_w:], img_aux[:, h2:, mid_w:]
        lblB_11, lblB_21, lblB_31 = lbl_aux[   :h1, :mid_w], lbl_aux[   h1:h2, :mid_w], lbl_aux[   h2:, :mid_w]
        lblB_12, lblB_22, lblB_32 = lbl_aux[   :h1, mid_w:], lbl_aux[   h1:h2, mid_w:], lbl_aux[   h2:, mid_w:]
        mskB_11, mskB_21, mskB_31 = msk_aux[   :h1, :mid_w], msk_aux[   h1:h2, :mid_w], msk_aux[   h2:, :mid_w]
        mskB_12, mskB_22, mskB_32 = msk_aux[   :h1, mid_w:], msk_aux[   h1:h2, mid_w:], msk_aux[   h2:, mid_w:]

        img_aux1_l, img_aux1_r = np.concatenate((imgA_11, imgB_21, imgA_31), axis=-2), np.concatenate((imgB_12, imgA_22, imgB_32), axis=-2)
        img_aux1 = np.concatenate((img_aux1_l, img_aux1_r), axis=-1)
        lbl_aux1_l, lbl_aux1_r = np.concatenate((lblA_11, lblB_21, lblA_31), axis=-2), np.concatenate((lblB_12, lblA_22, lblB_32), axis=-2)
        lbl_aux1 = np.concatenate((lbl_aux1_l, lbl_aux1_r), axis=-1)
        msk_aux1_l, msk_aux1_r = np.concatenate((mskA_11, mskB_21, mskA_31), axis=-2), np.concatenate((mskB_12, mskA_22, mskB_32), axis=-2)
        msk_aux1 = np.concatenate((msk_aux1_l, msk_aux1_r), axis=-1)

        img_aux2_l, img_aux2_r = np.concatenate((imgB_11, imgA_21, imgB_31), axis=-2), np.concatenate((imgA_12, imgB_22, imgA_32), axis=-2)
        img_aux2 = np.concatenate((img_aux2_l, img_aux2_r), axis=-1)
        lbl_aux2_l, lbl_aux2_r = np.concatenate((lblB_11, lblA_21, lblB_31), axis=-2), np.concatenate((lblA_12, lblB_22, lblA_32), axis=-2)
        lbl_aux2 = np.concatenate((lbl_aux2_l, lbl_aux2_r), axis=-1)
        msk_aux2_l, msk_aux2_r = np.concatenate((mskB_11, mskA_21, mskB_31), axis=-2), np.concatenate((mskA_12, mskB_22, mskA_32), axis=-2)
        msk_aux2 = np.concatenate((msk_aux2_l, msk_aux2_r), axis=-1)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col2row4(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 4)      # 64/4 = 16
        mid_h = int(img.shape[-2] / 2)   # 64/2 = 32
        h3 = 3 * h1                      # 16*3 = 48
        mid_w = int(img.shape[-1] / 2)   # 2048/2 = 1024

        imgA_11, imgA_21, imgA_31, imgA_41 = img[:, :h1, :mid_w], img[:, h1:mid_h, :mid_w], img[:, mid_h:h3, :mid_w], img[:, h3:, :mid_w]
        imgA_12, imgA_22, imgA_32, imgA_42 = img[:, :h1, mid_w:], img[:, h1:mid_h, mid_w:], img[:, mid_h:h3, mid_w:], img[:, h3:, mid_w:]
        lblA_11, lblA_21, lblA_31, lblA_41 = lbl[   :h1, :mid_w], lbl[   h1:mid_h, :mid_w], lbl[   mid_h:h3, :mid_w], lbl[   h3:, :mid_w]
        lblA_12, lblA_22, lblA_32, lblA_42 = lbl[   :h1, mid_w:], lbl[   h1:mid_h, mid_w:], lbl[   mid_h:h3, mid_w:], lbl[   h3:, mid_w:]
        mskA_11, mskA_21, mskA_31, mskA_41 = msk[   :h1, :mid_w], msk[   h1:mid_h, :mid_w], msk[   mid_h:h3, :mid_w], msk[   h3:, :mid_w]
        mskA_12, mskA_22, mskA_32, mskA_42 = msk[   :h1, mid_w:], msk[   h1:mid_h, mid_w:], msk[   mid_h:h3, mid_w:], msk[   h3:, mid_w:]

        imgB_11, imgB_21, imgB_31, imgB_41 = img_aux[:, :h1, :mid_w], img_aux[:, h1:mid_h, :mid_w], img_aux[:, mid_h:h3, :mid_w], img_aux[:, h3:, :mid_w]
        imgB_12, imgB_22, imgB_32, imgB_42 = img_aux[:, :h1, mid_w:], img_aux[:, h1:mid_h, mid_w:], img_aux[:, mid_h:h3, mid_w:], img_aux[:, h3:, mid_w:]
        lblB_11, lblB_21, lblB_31, lblB_41 = lbl_aux[   :h1, :mid_w], lbl_aux[   h1:mid_h, :mid_w], lbl_aux[   mid_h:h3, :mid_w], lbl_aux[   h3:, :mid_w]
        lblB_12, lblB_22, lblB_32, lblB_42 = lbl_aux[   :h1, mid_w:], lbl_aux[   h1:mid_h, mid_w:], lbl_aux[   mid_h:h3, mid_w:], lbl_aux[   h3:, mid_w:]
        mskB_11, mskB_21, mskB_31, mskB_41 = msk_aux[   :h1, :mid_w], msk_aux[   h1:mid_h, :mid_w], msk_aux[   mid_h:h3, :mid_w], msk_aux[   h3:, :mid_w]
        mskB_12, mskB_22, mskB_32, mskB_42 = msk_aux[   :h1, mid_w:], msk_aux[   h1:mid_h, mid_w:], msk_aux[   mid_h:h3, mid_w:], msk_aux[   h3:, mid_w:]

        img_aux1_l, img_aux1_r = np.concatenate((imgA_11, imgB_21, imgA_31, imgB_41), axis=-2), np.concatenate((imgB_12, imgA_22, imgB_32, imgA_42), axis=-2)
        img_aux1 = np.concatenate((img_aux1_l, img_aux1_r), axis=-1)
        lbl_aux1_l, lbl_aux1_r = np.concatenate((lblA_11, lblB_21, lblA_31, lblB_41), axis=-2), np.concatenate((lblB_12, lblA_22, lblB_32, lblA_42), axis=-2)
        lbl_aux1 = np.concatenate((lbl_aux1_l, lbl_aux1_r), axis=-1)
        msk_aux1_l, msk_aux1_r = np.concatenate((mskA_11, mskB_21, mskA_31, mskB_41), axis=-2), np.concatenate((mskB_12, mskA_22, mskB_32, mskA_42), axis=-2)
        msk_aux1 = np.concatenate((msk_aux1_l, msk_aux1_r), axis=-1)

        img_aux2_l, img_aux2_r = np.concatenate((imgB_11, imgA_21, imgB_31, imgA_41), axis=-2), np.concatenate((imgA_12, imgB_22, imgA_32, imgB_42), axis=-2)
        img_aux2 = np.concatenate((img_aux2_l, img_aux2_r), axis=-1)
        lbl_aux2_l, lbl_aux2_r = np.concatenate((lblB_11, lblA_21, lblB_31, lblA_41), axis=-2), np.concatenate((lblA_12, lblB_22, lblA_32, lblB_42), axis=-2)
        lbl_aux2 = np.concatenate((lbl_aux2_l, lbl_aux2_r), axis=-1)
        msk_aux2_l, msk_aux2_r = np.concatenate((mskB_11, mskA_21, mskB_31, mskA_41), axis=-2), np.concatenate((mskA_12, mskB_22, mskA_32, mskB_42), axis=-2)
        msk_aux2 = np.concatenate((msk_aux2_l, msk_aux2_r), axis=-1)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col2row5(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 5)      # 64/5 = 12
        h2 = 2 * h1                      # 2*12 = 24
        h3 = 3 * h1                      # 3*12 = 36
        h4 = 4 * h1                      # 4*12 = 48
        mid_w = int(img.shape[-1] / 2)   # 2048/2 = 1024

        imgA_11, imgA_21, imgA_31, imgA_41, imgA_51 = img[:, :h1, :mid_w], img[:, h1:h2, :mid_w], img[:, h2:h3, :mid_w], img[:, h3:h4, :mid_w], img[:, h4:, :mid_w]
        imgA_12, imgA_22, imgA_32, imgA_42, imgA_52 = img[:, :h1, mid_w:], img[:, h1:h2, mid_w:], img[:, h2:h3, mid_w:], img[:, h3:h4, mid_w:], img[:, h4:, mid_w:]
        lblA_11, lblA_21, lblA_31, lblA_41, lblA_51 = lbl[   :h1, :mid_w], lbl[   h1:h2, :mid_w], lbl[   h2:h3, :mid_w], lbl[   h3:h4, :mid_w], lbl[   h4:, :mid_w]
        lblA_12, lblA_22, lblA_32, lblA_42, lblA_52 = lbl[   :h1, mid_w:], lbl[   h1:h2, mid_w:], lbl[   h2:h3, mid_w:], lbl[   h3:h4, mid_w:], lbl[   h4:, mid_w:]
        mskA_11, mskA_21, mskA_31, mskA_41, mskA_51 = msk[   :h1, :mid_w], msk[   h1:h2, :mid_w], msk[   h2:h3, :mid_w], msk[   h3:h4, :mid_w], msk[   h4:, :mid_w]
        mskA_12, mskA_22, mskA_32, mskA_42, mskA_52 = msk[   :h1, mid_w:], msk[   h1:h2, mid_w:], msk[   h2:h3, mid_w:], msk[   h3:h4, mid_w:], msk[   h4:, mid_w:]

        imgB_11, imgB_21, imgB_31, imgB_41, imgB_51 = img_aux[:, :h1, :mid_w], img_aux[:, h1:h2, :mid_w], img_aux[:, h2:h3, :mid_w], img_aux[:, h3:h4, :mid_w], img_aux[:, h4:, :mid_w]
        imgB_12, imgB_22, imgB_32, imgB_42, imgB_52 = img_aux[:, :h1, mid_w:], img_aux[:, h1:h2, mid_w:], img_aux[:, h2:h3, mid_w:], img_aux[:, h3:h4, mid_w:], img_aux[:, h4:, mid_w:]
        lblB_11, lblB_21, lblB_31, lblB_41, lblB_51 = lbl_aux[   :h1, :mid_w], lbl_aux[   h1:h2, :mid_w], lbl_aux[   h2:h3, :mid_w], lbl_aux[   h3:h4, :mid_w], lbl_aux[   h4:, :mid_w]
        lblB_12, lblB_22, lblB_32, lblB_42, lblB_52 = lbl_aux[   :h1, mid_w:], lbl_aux[   h1:h2, mid_w:], lbl_aux[   h2:h3, mid_w:], lbl_aux[   h3:h4, mid_w:], lbl_aux[   h4:, mid_w:]
        mskB_11, mskB_21, mskB_31, mskB_41, mskB_51 = msk_aux[   :h1, :mid_w], msk_aux[   h1:h2, :mid_w], msk_aux[   h2:h3, :mid_w], msk_aux[   h3:h4, :mid_w], msk_aux[   h4:, :mid_w]
        mskB_12, mskB_22, mskB_32, mskB_42, mskB_52 = msk_aux[   :h1, mid_w:], msk_aux[   h1:h2, mid_w:], msk_aux[   h2:h3, mid_w:], msk_aux[   h3:h4, mid_w:], msk_aux[   h4:, mid_w:]

        img_aux1_l, img_aux1_r = np.concatenate((imgA_11, imgB_21, imgA_31, imgB_41, imgA_51), axis=-2), np.concatenate((imgB_12, imgA_22, imgB_32, imgA_42, imgB_52), axis=-2)
        img_aux1 = np.concatenate((img_aux1_l, img_aux1_r), axis=-1)
        lbl_aux1_l, lbl_aux1_r = np.concatenate((lblA_11, lblB_21, lblA_31, lblB_41, lblA_51), axis=-2), np.concatenate((lblB_12, lblA_22, lblB_32, lblA_42, lblB_52), axis=-2)
        lbl_aux1 = np.concatenate((lbl_aux1_l, lbl_aux1_r), axis=-1)
        msk_aux1_l, msk_aux1_r = np.concatenate((mskA_11, mskB_21, mskA_31, mskB_41, mskA_51), axis=-2), np.concatenate((mskB_12, mskA_22, mskB_32, mskA_42, mskB_52), axis=-2)
        msk_aux1 = np.concatenate((msk_aux1_l, msk_aux1_r), axis=-1)

        img_aux2_l, img_aux2_r = np.concatenate((imgB_11, imgA_21, imgB_31, imgA_41, imgB_51), axis=-2), np.concatenate((imgA_12, imgB_22, imgA_32, imgB_42, imgA_52), axis=-2)
        img_aux2 = np.concatenate((img_aux2_l, img_aux2_r), axis=-1)
        lbl_aux2_l, lbl_aux2_r = np.concatenate((lblB_11, lblA_21, lblB_31, lblA_41, lblB_51), axis=-2), np.concatenate((lblA_12, lblB_22, lblA_32, lblB_42, lblA_52), axis=-2)
        lbl_aux2 = np.concatenate((lbl_aux2_l, lbl_aux2_r), axis=-1)
        msk_aux2_l, msk_aux2_r = np.concatenate((mskB_11, mskA_21, mskB_31, mskA_41, mskB_51), axis=-2), np.concatenate((mskA_12, mskB_22, mskA_32, mskB_42, mskA_52), axis=-2)
        msk_aux2 = np.concatenate((msk_aux2_l, msk_aux2_r), axis=-1)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col2row6(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 6)      # 64/6 = 10
        h2 = 2 * h1                      # 2*10 = 20
        h3 = 3 * h1                      # 3*10 = 30
        h4 = 4 * h1                      # 4*10 = 40
        h5 = 5 * h1                      # 5*10 = 50
        mid_w = int(img.shape[-1] / 2)   # 2048/2 = 1024

        imgA_11, imgA_21, imgA_31, imgA_41, imgA_51, imgA_61 = img[:, :h1, :mid_w], img[:, h1:h2, :mid_w], img[:, h2:h3, :mid_w], img[:, h3:h4, :mid_w], img[:, h4:h5, :mid_w], img[:, h5:, :mid_w]
        imgA_12, imgA_22, imgA_32, imgA_42, imgA_52, imgA_62 = img[:, :h1, mid_w:], img[:, h1:h2, mid_w:], img[:, h2:h3, mid_w:], img[:, h3:h4, mid_w:], img[:, h4:h5, mid_w:], img[:, h5:, mid_w:]
        lblA_11, lblA_21, lblA_31, lblA_41, lblA_51, lblA_61 = lbl[   :h1, :mid_w], lbl[   h1:h2, :mid_w], lbl[   h2:h3, :mid_w], lbl[   h3:h4, :mid_w], lbl[   h4:h5, :mid_w], lbl[   h5:, :mid_w]
        lblA_12, lblA_22, lblA_32, lblA_42, lblA_52, lblA_62 = lbl[   :h1, mid_w:], lbl[   h1:h2, mid_w:], lbl[   h2:h3, mid_w:], lbl[   h3:h4, mid_w:], lbl[   h4:h5, mid_w:], lbl[   h5:, mid_w:]
        mskA_11, mskA_21, mskA_31, mskA_41, mskA_51, mskA_61 = msk[   :h1, :mid_w], msk[   h1:h2, :mid_w], msk[   h2:h3, :mid_w], msk[   h3:h4, :mid_w], msk[   h4:h5, :mid_w], msk[   h5:, :mid_w]
        mskA_12, mskA_22, mskA_32, mskA_42, mskA_52, mskA_62 = msk[   :h1, mid_w:], msk[   h1:h2, mid_w:], msk[   h2:h3, mid_w:], msk[   h3:h4, mid_w:], msk[   h4:h5, mid_w:], msk[   h5:, mid_w:]

        imgB_11, imgB_21, imgB_31, imgB_41, imgB_51, imgB_61 = img_aux[:, :h1, :mid_w], img_aux[:, h1:h2, :mid_w], img_aux[:, h2:h3, :mid_w], img_aux[:, h3:h4, :mid_w], img_aux[:, h4:h5, :mid_w], img_aux[:, h5:, :mid_w]
        imgB_12, imgB_22, imgB_32, imgB_42, imgB_52, imgB_62 = img_aux[:, :h1, mid_w:], img_aux[:, h1:h2, mid_w:], img_aux[:, h2:h3, mid_w:], img_aux[:, h3:h4, mid_w:], img_aux[:, h4:h5, mid_w:], img_aux[:, h5:, mid_w:]
        lblB_11, lblB_21, lblB_31, lblB_41, lblB_51, lblB_61 = lbl_aux[   :h1, :mid_w], lbl_aux[   h1:h2, :mid_w], lbl_aux[   h2:h3, :mid_w], lbl_aux[   h3:h4, :mid_w], lbl_aux[   h4:h5, :mid_w], lbl_aux[   h5:, :mid_w]
        lblB_12, lblB_22, lblB_32, lblB_42, lblB_52, lblB_62 = lbl_aux[   :h1, mid_w:], lbl_aux[   h1:h2, mid_w:], lbl_aux[   h2:h3, mid_w:], lbl_aux[   h3:h4, mid_w:], lbl_aux[   h4:h5, mid_w:], lbl_aux[   h5:, mid_w:]
        mskB_11, mskB_21, mskB_31, mskB_41, mskB_51, mskB_61 = msk_aux[   :h1, :mid_w], msk_aux[   h1:h2, :mid_w], msk_aux[   h2:h3, :mid_w], msk_aux[   h3:h4, :mid_w], msk_aux[   h4:h5, :mid_w], msk_aux[   h5:, :mid_w]
        mskB_12, mskB_22, mskB_32, mskB_42, mskB_52, mskB_62 = msk_aux[   :h1, mid_w:], msk_aux[   h1:h2, mid_w:], msk_aux[   h2:h3, mid_w:], msk_aux[   h3:h4, mid_w:], msk_aux[   h4:h5, mid_w:], msk_aux[   h5:, mid_w:]

        img_aux1_l, img_aux1_r = np.concatenate((imgA_11, imgB_21, imgA_31, imgB_41, imgA_51, imgB_61), axis=-2), np.concatenate((imgB_12, imgA_22, imgB_32, imgA_42, imgB_52, imgA_62), axis=-2)
        img_aux1 = np.concatenate((img_aux1_l, img_aux1_r), axis=-1)
        lbl_aux1_l, lbl_aux1_r = np.concatenate((lblA_11, lblB_21, lblA_31, lblB_41, lblA_51, lblB_61), axis=-2), np.concatenate((lblB_12, lblA_22, lblB_32, lblA_42, lblB_52, lblA_62), axis=-2)
        lbl_aux1 = np.concatenate((lbl_aux1_l, lbl_aux1_r), axis=-1)
        msk_aux1_l, msk_aux1_r = np.concatenate((mskA_11, mskB_21, mskA_31, mskB_41, mskA_51, mskB_61), axis=-2), np.concatenate((mskB_12, mskA_22, mskB_32, mskA_42, mskB_52, mskA_62), axis=-2)
        msk_aux1 = np.concatenate((msk_aux1_l, msk_aux1_r), axis=-1)

        img_aux2_l, img_aux2_r = np.concatenate((imgB_11, imgA_21, imgB_31, imgA_41, imgB_51, imgA_61), axis=-2), np.concatenate((imgA_12, imgB_22, imgA_32, imgB_42, imgA_52, imgB_62), axis=-2)
        img_aux2 = np.concatenate((img_aux2_l, img_aux2_r), axis=-1)
        lbl_aux2_l, lbl_aux2_r = np.concatenate((lblB_11, lblA_21, lblB_31, lblA_41, lblB_51, lblA_61), axis=-2), np.concatenate((lblA_12, lblB_22, lblA_32, lblB_42, lblA_52, lblB_62), axis=-2)
        lbl_aux2 = np.concatenate((lbl_aux2_l, lbl_aux2_r), axis=-1)
        msk_aux2_l, msk_aux2_r = np.concatenate((mskB_11, mskA_21, mskB_31, mskA_41, mskB_51, mskA_61), axis=-2), np.concatenate((mskA_12, mskB_22, mskA_32, mskB_42, mskA_52, mskB_62), axis=-2)
        msk_aux2 = np.concatenate((msk_aux2_l, msk_aux2_r), axis=-1)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col3row1(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 3)   # 2048/3 = 683
        w2 = 2 * w1                   # 2 * 683 = 1366
        
        imgA_1, imgA_2, imgA_3 = img[:, :, :w1], img[:, :, w1:w2], img[:, :, w2:]  # left1, middle1, right1
        lblA_1, lblA_2, lblA_3 = lbl[   :, :w1], lbl[   :, w1:w2], lbl[   :, w2:]  # left1, middle1, right1
        mskA_1, mskA_2, mskA_3 = msk[   :, :w1], msk[   :, w1:w2], msk[   :, w2:]  # left1, middle1, right1
        
        imgB_1, imgB_2, imgB_3 = img_aux[:, :, :w1], img_aux[:, :, w1:w2], img_aux[:, :, w2:]  # left2, middle2, right2
        lblB_1, lblB_2, lblB_3 = lbl_aux[   :, :w1], lbl_aux[   :, w1:w2], lbl_aux[   :, w2:]  # left2, middle2, right2
        mskB_1, mskB_2, mskB_3 = msk_aux[   :, :w1], msk_aux[   :, w1:w2], msk_aux[   :, w2:]  # left2, middle2, right2
        
        img_aux1 = np.concatenate((imgA_1, imgB_2, imgA_3), axis=-1)  # left1, middle2, right1
        lbl_aux1 = np.concatenate((lblA_1, lblB_2, lblA_3), axis=-1)  # left1, middle2, right1
        msk_aux1 = np.concatenate((mskA_1, mskB_2, mskA_3), axis=-1)  # left1, middle2, right1

        img_aux2 = np.concatenate((imgB_1, imgA_2, imgB_3), axis=-1)  # left2, middle1, right1
        lbl_aux2 = np.concatenate((lblB_1, lblA_2, lblB_3), axis=-1)  # left2, middle1, right1
        msk_aux2 = np.concatenate((mskB_1, mskA_2, mskB_3), axis=-1)  # left2, middle1, right1
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col3row2(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 3)   # 2048/3 = 683
        w2 = 2 * w1                   # 2*683  = 1366
        h1 = int(img.shape[-2] / 2)   # 64/2 = 32
        
        imgA_11, imgA_12, imgA_13 = img[:, :h1, :w1], img[:, :h1, w1:w2], img[:, :h1, w2:]
        imgA_21, imgA_22, imgA_23 = img[:, h1:, :w1], img[:, h1:, w1:w2], img[:, h1:, w2:]

        lblA_11, lblA_12, lblA_13 = lbl[   :h1, :w1], lbl[   :h1, w1:w2], lbl[   :h1, w2:]
        lblA_21, lblA_22, lblA_23 = lbl[   h1:, :w1], lbl[   h1:, w1:w2], lbl[   h1:, w2:]

        mskA_11, mskA_12, mskA_13 = msk[   :h1, :w1], msk[   :h1, w1:w2], msk[   :h1, w2:]
        mskA_21, mskA_22, mskA_23 = msk[   h1:, :w1], msk[   h1:, w1:w2], msk[   h1:, w2:]
        
        imgB_11, imgB_12, imgB_13 = img_aux[:, :h1, :w1], img_aux[:, :h1, w1:w2], img_aux[:, :h1, w2:]
        imgB_21, imgB_22, imgB_23 = img_aux[:, h1:, :w1], img_aux[:, h1:, w1:w2], img_aux[:, h1:, w2:]

        lblB_11, lblB_12, lblB_13 = lbl_aux[   :h1, :w1], lbl_aux[   :h1, w1:w2], lbl_aux[   :h1, w2:]
        lblB_21, lblB_22, lblB_23 = lbl_aux[   h1:, :w1], lbl_aux[   h1:, w1:w2], lbl_aux[   h1:, w2:]

        mskB_11, mskB_12, mskB_13 = msk_aux[   :h1, :w1], msk_aux[   :h1, w1:w2], msk_aux[   :h1, w2:]
        mskB_21, mskB_22, mskB_23 = msk_aux[   h1:, :w1], msk_aux[   h1:, w1:w2], msk_aux[   h1:, w2:]
        
        img_aux1_top = np.concatenate((imgA_11, imgB_12, imgA_13), axis=-1)
        img_aux1_bot = np.concatenate((imgB_21, imgA_22, imgB_23), axis=-1)
        img_aux1 = np.concatenate((img_aux1_top, img_aux1_bot), axis=-2)

        lbl_aux1_top = np.concatenate((lblA_11, lblB_12, lblA_13), axis=-1)
        lbl_aux1_bot = np.concatenate((lblB_21, lblA_22, lblB_23), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_top, lbl_aux1_bot), axis=-2)

        msk_aux1_top = np.concatenate((mskA_11, mskB_12, mskA_13), axis=-1)
        msk_aux1_bot = np.concatenate((mskB_21, mskA_22, mskB_23), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_top, msk_aux1_bot), axis=-2)

        img_aux2_top = np.concatenate((imgB_11, imgA_12, imgB_13), axis=-1)
        img_aux2_bot = np.concatenate((imgA_21, imgB_22, imgA_23), axis=-1)
        img_aux2 = np.concatenate((img_aux2_top, img_aux2_bot), axis=-2)

        lbl_aux2_top = np.concatenate((lblB_11, lblA_12, lblB_13), axis=-1)
        lbl_aux2_bot = np.concatenate((lblA_21, lblB_22, lblA_23), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_top, lbl_aux2_bot), axis=-2)

        msk_aux2_top = np.concatenate((mskB_11, mskA_12, mskB_13), axis=-1)
        msk_aux2_bot = np.concatenate((mskA_21, mskB_22, mskA_23), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_top, msk_aux2_bot), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col3row3(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 3)   # 2048/3 = 683
        w2 = 2 * w1                   # 2*683  = 1366
        h1 = int(img.shape[-2] / 3)   # 64/3 = 21
        h2 = 2 * h1                   # 2*21 = 42
        
        imgA_11, imgA_12, imgA_13 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:]
        imgA_21, imgA_22, imgA_23 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:]
        imgA_31, imgA_32, imgA_33 = img[:, h2:  , :w1], img[:, h2:  , w1:w2], img[:, h2:  , w2:]

        lblA_11, lblA_12, lblA_13 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:]
        lblA_21, lblA_22, lblA_23 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:]
        lblA_31, lblA_32, lblA_33 = lbl[   h2:  , :w1], lbl[   h2:  , w1:w2], lbl[   h2:  , w2:]

        mskA_11, mskA_12, mskA_13 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:]
        mskA_21, mskA_22, mskA_23 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:]
        mskA_31, mskA_32, mskA_33 = msk[   h2:  , :w1], msk[   h2:  , w1:w2], msk[   h2:  , w2:]
        
        imgB_11, imgB_12, imgB_13 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:]
        imgB_21, imgB_22, imgB_23 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:]
        imgB_31, imgB_32, imgB_33 = img_aux[:, h2:  , :w1], img_aux[:, h2:  , w1:w2], img_aux[:, h2:  , w2:]

        lblB_11, lblB_12, lblB_13 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:]
        lblB_21, lblB_22, lblB_23 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:]
        lblB_31, lblB_32, lblB_33 = lbl_aux[   h2:  , :w1], lbl_aux[   h2:  , w1:w2], lbl_aux[   h2:  , w2:]

        mskB_11, mskB_12, mskB_13 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:]
        mskB_21, mskB_22, mskB_23 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:]
        mskB_31, mskB_32, mskB_33 = msk_aux[   h2:  , :w1], msk_aux[   h2:  , w1:w2], msk_aux[   h2:  , w2:]
        
        img_aux1_top = np.concatenate((imgA_11, imgB_12, imgA_13), axis=-1)
        img_aux1_mid = np.concatenate((imgB_21, imgA_22, imgB_23), axis=-1)
        img_aux1_bot = np.concatenate((imgA_31, imgB_32, imgA_33), axis=-1)
        img_aux1 = np.concatenate((img_aux1_top, img_aux1_mid, img_aux1_bot), axis=-2)

        lbl_aux1_top = np.concatenate((lblA_11, lblB_12, lblA_13), axis=-1)
        lbl_aux1_mid = np.concatenate((lblB_21, lblA_22, lblB_23), axis=-1)
        lbl_aux1_bot = np.concatenate((lblA_31, lblB_32, lblA_33), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_top, lbl_aux1_mid, lbl_aux1_bot), axis=-2)

        msk_aux1_top = np.concatenate((mskA_11, mskB_12, mskA_13), axis=-1)
        msk_aux1_mid = np.concatenate((mskB_21, mskA_22, mskB_23), axis=-1)
        msk_aux1_bot = np.concatenate((mskA_31, mskB_32, mskA_33), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_top, msk_aux1_mid, msk_aux1_bot), axis=-2)

        img_aux2_top = np.concatenate((imgB_11, imgA_12, imgB_13), axis=-1)
        img_aux2_mid = np.concatenate((imgA_21, imgB_22, imgA_23), axis=-1)
        img_aux2_bot = np.concatenate((imgB_31, imgA_32, imgB_33), axis=-1)
        img_aux2 = np.concatenate((img_aux2_top, img_aux2_mid, img_aux2_bot), axis=-2)

        lbl_aux2_top = np.concatenate((lblB_11, lblA_12, lblB_13), axis=-1)
        lbl_aux2_mid = np.concatenate((lblA_21, lblB_22, lblA_23), axis=-1)
        lbl_aux2_bot = np.concatenate((lblB_31, lblA_32, lblB_33), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_top, lbl_aux2_mid, lbl_aux2_bot), axis=-2)

        msk_aux2_top = np.concatenate((mskB_11, mskA_12, mskB_13), axis=-1)
        msk_aux2_mid = np.concatenate((mskA_21, mskB_22, mskA_23), axis=-1)
        msk_aux2_bot = np.concatenate((mskB_31, mskA_32, mskB_33), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_top, msk_aux2_mid, msk_aux2_bot), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col3row4(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 3)   # 2048/3 = 683
        w2 = 2 * w1                   # 2*683  = 1366
        h1 = int(img.shape[-2] / 4)   # 64/4 = 16
        h2 = 2 * h1                   # 2*16 = 32
        h3 = 3 * h1                   # 3*16 = 48
        
        imgA_11, imgA_12, imgA_13 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:]
        imgA_21, imgA_22, imgA_23 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:]
        imgA_31, imgA_32, imgA_33 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:]
        imgA_41, imgA_42, imgA_43 = img[:, h3:  , :w1], img[:, h3:  , w1:w2], img[:, h3:  , w2:]

        lblA_11, lblA_12, lblA_13 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:]
        lblA_21, lblA_22, lblA_23 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:]
        lblA_31, lblA_32, lblA_33 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:]
        lblA_41, lblA_42, lblA_43 = lbl[   h3:  , :w1], lbl[   h3:  , w1:w2], lbl[   h3:  , w2:]

        mskA_11, mskA_12, mskA_13 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:]
        mskA_21, mskA_22, mskA_23 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:]
        mskA_31, mskA_32, mskA_33 = msk[   h2:h3, :w1], msk[   h2:h3, w1:w2], msk[   h2:h3, w2:]
        mskA_41, mskA_42, mskA_43 = msk[   h3:  , :w1], msk[   h3:  , w1:w2], msk[   h3:  , w2:]
        
        imgB_11, imgB_12, imgB_13 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:]
        imgB_21, imgB_22, imgB_23 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:]
        imgB_31, imgB_32, imgB_33 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:]
        imgB_41, imgB_42, imgB_43 = img_aux[:, h3:  , :w1], img_aux[:, h3:  , w1:w2], img_aux[:, h3:  , w2:]

        lblB_11, lblB_12, lblB_13 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:]
        lblB_21, lblB_22, lblB_23 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:]
        lblB_31, lblB_32, lblB_33 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:]
        lblB_41, lblB_42, lblB_43 = lbl_aux[   h3:  , :w1], lbl_aux[   h3:  , w1:w2], lbl_aux[   h3:  , w2:]

        mskB_11, mskB_12, mskB_13 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:]
        mskB_21, mskB_22, mskB_23 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:]
        mskB_31, mskB_32, mskB_33 = msk_aux[   h2:h3, :w1], msk_aux[   h2:h3, w1:w2], msk_aux[   h2:h3, w2:]
        mskB_41, mskB_42, mskB_43 = msk_aux[   h3:  , :w1], msk_aux[   h3:  , w1:w2], msk_aux[   h3:  , w2:]
        
        img_aux1_1 = np.concatenate((imgA_11, imgB_12, imgA_13), axis=-1)
        img_aux1_2 = np.concatenate((imgB_21, imgA_22, imgB_23), axis=-1)
        img_aux1_3 = np.concatenate((imgA_31, imgB_32, imgA_33), axis=-1)
        img_aux1_4 = np.concatenate((imgB_41, imgA_42, imgB_43), axis=-1)
        img_aux1 = np.concatenate((img_aux1_1, img_aux1_2, img_aux1_3, img_aux1_4), axis=-2)

        lbl_aux1_1 = np.concatenate((lblA_11, lblB_12, lblA_13), axis=-1)
        lbl_aux1_2 = np.concatenate((lblB_21, lblA_22, lblB_23), axis=-1)
        lbl_aux1_3 = np.concatenate((lblA_31, lblB_32, lblA_33), axis=-1)
        lbl_aux1_4 = np.concatenate((lblB_41, lblA_42, lblB_43), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_1, lbl_aux1_2, lbl_aux1_3, lbl_aux1_4), axis=-2)

        msk_aux1_1 = np.concatenate((mskA_11, mskB_12, mskA_13), axis=-1)
        msk_aux1_2 = np.concatenate((mskB_21, mskA_22, mskB_23), axis=-1)
        msk_aux1_3 = np.concatenate((mskA_31, mskB_32, mskA_33), axis=-1)
        msk_aux1_4 = np.concatenate((mskB_41, mskA_42, mskB_43), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_1, msk_aux1_2, msk_aux1_3, msk_aux1_4), axis=-2)

        img_aux2_1 = np.concatenate((imgB_11, imgA_12, imgB_13), axis=-1)
        img_aux2_2 = np.concatenate((imgA_21, imgB_22, imgA_23), axis=-1)
        img_aux2_3 = np.concatenate((imgB_31, imgA_32, imgB_33), axis=-1)
        img_aux2_4 = np.concatenate((imgA_41, imgB_42, imgA_43), axis=-1)
        img_aux2 = np.concatenate((img_aux2_1, img_aux2_2, img_aux2_3, img_aux2_4), axis=-2)

        lbl_aux2_1 = np.concatenate((lblB_11, lblA_12, lblB_13), axis=-1)
        lbl_aux2_2 = np.concatenate((lblA_21, lblB_22, lblA_23), axis=-1)
        lbl_aux2_3 = np.concatenate((lblB_31, lblA_32, lblB_33), axis=-1)
        lbl_aux2_4 = np.concatenate((lblA_41, lblB_42, lblA_43), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_1, lbl_aux2_2, lbl_aux2_3, lbl_aux2_4), axis=-2)

        msk_aux2_1 = np.concatenate((mskB_11, mskA_12, mskB_13), axis=-1)
        msk_aux2_2 = np.concatenate((mskA_21, mskB_22, mskA_23), axis=-1)
        msk_aux2_3 = np.concatenate((mskB_31, mskA_32, mskB_33), axis=-1)
        msk_aux2_4 = np.concatenate((mskA_41, mskB_42, mskA_43), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_1, msk_aux2_2, msk_aux2_3, msk_aux2_4), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col3row5(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 3)   # 2048/3 = 683
        w2 = 2 * w1                   # 2*683  = 1366
        h1 = int(img.shape[-2] / 5)   # 64/5 = 12
        h2 = 2 * h1                   # 2*12 = 24
        h3 = 3 * h1                   # 3*12 = 36
        h4 = 4 * h1                   # 4*12 = 48
        
        imgA_11, imgA_12, imgA_13 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:]
        imgA_21, imgA_22, imgA_23 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:]
        imgA_31, imgA_32, imgA_33 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:]
        imgA_41, imgA_42, imgA_43 = img[:, h3:h4, :w1], img[:, h3:h4, w1:w2], img[:, h3:h4, w2:]
        imgA_51, imgA_52, imgA_53 = img[:, h4:  , :w1], img[:, h4:  , w1:w2], img[:, h4:  , w2:]

        lblA_11, lblA_12, lblA_13 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:]
        lblA_21, lblA_22, lblA_23 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:]
        lblA_31, lblA_32, lblA_33 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:]
        lblA_41, lblA_42, lblA_43 = lbl[   h3:h4, :w1], lbl[   h3:h4, w1:w2], lbl[   h3:h4, w2:]
        lblA_51, lblA_52, lblA_53 = lbl[   h4:  , :w1], lbl[   h4:  , w1:w2], lbl[   h4:  , w2:]

        mskA_11, mskA_12, mskA_13 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:]
        mskA_21, mskA_22, mskA_23 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:]
        mskA_31, mskA_32, mskA_33 = msk[   h2:h3, :w1], msk[   h2:h3, w1:w2], msk[   h2:h3, w2:]
        mskA_41, mskA_42, mskA_43 = msk[   h3:h4, :w1], msk[   h3:h4, w1:w2], msk[   h3:h4, w2:]
        mskA_51, mskA_52, mskA_53 = msk[   h4:  , :w1], msk[   h4:  , w1:w2], msk[   h4:  , w2:]
        
        imgB_11, imgB_12, imgB_13 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:]
        imgB_21, imgB_22, imgB_23 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:]
        imgB_31, imgB_32, imgB_33 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:]
        imgB_41, imgB_42, imgB_43 = img_aux[:, h3:h4, :w1], img_aux[:, h3:h4, w1:w2], img_aux[:, h3:h4, w2:]
        imgB_51, imgB_52, imgB_53 = img_aux[:, h4:  , :w1], img_aux[:, h4:  , w1:w2], img_aux[:, h4:  , w2:]

        lblB_11, lblB_12, lblB_13 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:]
        lblB_21, lblB_22, lblB_23 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:]
        lblB_31, lblB_32, lblB_33 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:]
        lblB_41, lblB_42, lblB_43 = lbl_aux[   h3:h4, :w1], lbl_aux[   h3:h4, w1:w2], lbl_aux[   h3:h4, w2:]
        lblB_51, lblB_52, lblB_53 = lbl_aux[   h4:  , :w1], lbl_aux[   h4:  , w1:w2], lbl_aux[   h4:  , w2:]

        mskB_11, mskB_12, mskB_13 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:]
        mskB_21, mskB_22, mskB_23 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:]
        mskB_31, mskB_32, mskB_33 = msk_aux[   h2:h3, :w1], msk_aux[   h2:h3, w1:w2], msk_aux[   h2:h3, w2:]
        mskB_41, mskB_42, mskB_43 = msk_aux[   h3:h4, :w1], msk_aux[   h3:h4, w1:w2], msk_aux[   h3:h4, w2:]
        mskB_51, mskB_52, mskB_53 = msk_aux[   h4:  , :w1], msk_aux[   h4:  , w1:w2], msk_aux[   h4:  , w2:]
        
        img_aux1_1 = np.concatenate((imgA_11, imgB_12, imgA_13), axis=-1)
        img_aux1_2 = np.concatenate((imgB_21, imgA_22, imgB_23), axis=-1)
        img_aux1_3 = np.concatenate((imgA_31, imgB_32, imgA_33), axis=-1)
        img_aux1_4 = np.concatenate((imgB_41, imgA_42, imgB_43), axis=-1)
        img_aux1_5 = np.concatenate((imgA_51, imgB_52, imgA_53), axis=-1)
        img_aux1 = np.concatenate((img_aux1_1, img_aux1_2, img_aux1_3, img_aux1_4, img_aux1_5), axis=-2)

        lbl_aux1_1 = np.concatenate((lblA_11, lblB_12, lblA_13), axis=-1)
        lbl_aux1_2 = np.concatenate((lblB_21, lblA_22, lblB_23), axis=-1)
        lbl_aux1_3 = np.concatenate((lblA_31, lblB_32, lblA_33), axis=-1)
        lbl_aux1_4 = np.concatenate((lblB_41, lblA_42, lblB_43), axis=-1)
        lbl_aux1_5 = np.concatenate((lblA_51, lblB_52, lblA_53), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_1, lbl_aux1_2, lbl_aux1_3, lbl_aux1_4, lbl_aux1_5), axis=-2)

        msk_aux1_1 = np.concatenate((mskA_11, mskB_12, mskA_13), axis=-1)
        msk_aux1_2 = np.concatenate((mskB_21, mskA_22, mskB_23), axis=-1)
        msk_aux1_3 = np.concatenate((mskA_31, mskB_32, mskA_33), axis=-1)
        msk_aux1_4 = np.concatenate((mskB_41, mskA_42, mskB_43), axis=-1)
        msk_aux1_5 = np.concatenate((mskA_51, mskB_52, mskA_53), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_1, msk_aux1_2, msk_aux1_3, msk_aux1_4, msk_aux1_5), axis=-2)

        img_aux2_1 = np.concatenate((imgB_11, imgA_12, imgB_13), axis=-1)
        img_aux2_2 = np.concatenate((imgA_21, imgB_22, imgA_23), axis=-1)
        img_aux2_3 = np.concatenate((imgB_31, imgA_32, imgB_33), axis=-1)
        img_aux2_4 = np.concatenate((imgA_41, imgB_42, imgA_43), axis=-1)
        img_aux2_5 = np.concatenate((imgB_51, imgA_52, imgB_53), axis=-1)
        img_aux2 = np.concatenate((img_aux2_1, img_aux2_2, img_aux2_3, img_aux2_4, img_aux2_5), axis=-2)

        lbl_aux2_1 = np.concatenate((lblB_11, lblA_12, lblB_13), axis=-1)
        lbl_aux2_2 = np.concatenate((lblA_21, lblB_22, lblA_23), axis=-1)
        lbl_aux2_3 = np.concatenate((lblB_31, lblA_32, lblB_33), axis=-1)
        lbl_aux2_4 = np.concatenate((lblA_41, lblB_42, lblA_43), axis=-1)
        lbl_aux2_5 = np.concatenate((lblB_51, lblA_52, lblB_53), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_1, lbl_aux2_2, lbl_aux2_3, lbl_aux2_4, lbl_aux2_5), axis=-2)

        msk_aux2_1 = np.concatenate((mskB_11, mskA_12, mskB_13), axis=-1)
        msk_aux2_2 = np.concatenate((mskA_21, mskB_22, mskA_23), axis=-1)
        msk_aux2_3 = np.concatenate((mskB_31, mskA_32, mskB_33), axis=-1)
        msk_aux2_4 = np.concatenate((mskA_41, mskB_42, mskA_43), axis=-1)
        msk_aux2_5 = np.concatenate((mskB_51, mskA_52, mskB_53), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_1, msk_aux2_2, msk_aux2_3, msk_aux2_4, msk_aux2_5), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col3row6(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 3)   # 2048/3 = 683
        w2 = 2 * w1                   # 2*683  = 1366
        h1 = int(img.shape[-2] / 6)   # 64/6 = 10
        h2 = 2 * h1                   # 2*10 = 20
        h3 = 3 * h1                   # 3*10 = 30
        h4 = 4 * h1                   # 4*10 = 40
        h5 = 5 * h1                   # 5*10 = 50
        
        imgA_11, imgA_12, imgA_13 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:]
        imgA_21, imgA_22, imgA_23 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:]
        imgA_31, imgA_32, imgA_33 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:]
        imgA_41, imgA_42, imgA_43 = img[:, h3:h4, :w1], img[:, h3:h4, w1:w2], img[:, h3:h4, w2:]
        imgA_51, imgA_52, imgA_53 = img[:, h4:h5, :w1], img[:, h4:h5, w1:w2], img[:, h4:h5, w2:]
        imgA_61, imgA_62, imgA_63 = img[:, h5:  , :w1], img[:, h5:  , w1:w2], img[:, h5:  , w2:]

        lblA_11, lblA_12, lblA_13 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:]
        lblA_21, lblA_22, lblA_23 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:]
        lblA_31, lblA_32, lblA_33 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:]
        lblA_41, lblA_42, lblA_43 = lbl[   h3:h4, :w1], lbl[   h3:h4, w1:w2], lbl[   h3:h4, w2:]
        lblA_51, lblA_52, lblA_53 = lbl[   h4:h5, :w1], lbl[   h4:h5, w1:w2], lbl[   h4:h5, w2:]
        lblA_61, lblA_62, lblA_63 = lbl[   h5:  , :w1], lbl[   h5:  , w1:w2], lbl[   h5:  , w2:]

        mskA_11, mskA_12, mskA_13 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:]
        mskA_21, mskA_22, mskA_23 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:]
        mskA_31, mskA_32, mskA_33 = msk[   h2:h3, :w1], msk[   h2:h3, w1:w2], msk[   h2:h3, w2:]
        mskA_41, mskA_42, mskA_43 = msk[   h3:h4, :w1], msk[   h3:h4, w1:w2], msk[   h3:h4, w2:]
        mskA_51, mskA_52, mskA_53 = msk[   h4:h5, :w1], msk[   h4:h5, w1:w2], msk[   h4:h5, w2:]
        mskA_61, mskA_62, mskA_63 = msk[   h5:  , :w1], msk[   h5:  , w1:w2], msk[   h5:  , w2:]
        
        imgB_11, imgB_12, imgB_13 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:]
        imgB_21, imgB_22, imgB_23 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:]
        imgB_31, imgB_32, imgB_33 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:]
        imgB_41, imgB_42, imgB_43 = img_aux[:, h3:h4, :w1], img_aux[:, h3:h4, w1:w2], img_aux[:, h3:h4, w2:]
        imgB_51, imgB_52, imgB_53 = img_aux[:, h4:h5, :w1], img_aux[:, h4:h5, w1:w2], img_aux[:, h4:h5, w2:]
        imgB_61, imgB_62, imgB_63 = img_aux[:, h5:  , :w1], img_aux[:, h5:  , w1:w2], img_aux[:, h5:  , w2:]

        lblB_11, lblB_12, lblB_13 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:]
        lblB_21, lblB_22, lblB_23 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:]
        lblB_31, lblB_32, lblB_33 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:]
        lblB_41, lblB_42, lblB_43 = lbl_aux[   h3:h4, :w1], lbl_aux[   h3:h4, w1:w2], lbl_aux[   h3:h4, w2:]
        lblB_51, lblB_52, lblB_53 = lbl_aux[   h4:h5, :w1], lbl_aux[   h4:h5, w1:w2], lbl_aux[   h4:h5, w2:]
        lblB_61, lblB_62, lblB_63 = lbl_aux[   h5:  , :w1], lbl_aux[   h5:  , w1:w2], lbl_aux[   h5:  , w2:]

        mskB_11, mskB_12, mskB_13 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:]
        mskB_21, mskB_22, mskB_23 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:]
        mskB_31, mskB_32, mskB_33 = msk_aux[   h2:h3, :w1], msk_aux[   h2:h3, w1:w2], msk_aux[   h2:h3, w2:]
        mskB_41, mskB_42, mskB_43 = msk_aux[   h3:h4, :w1], msk_aux[   h3:h4, w1:w2], msk_aux[   h3:h4, w2:]
        mskB_51, mskB_52, mskB_53 = msk_aux[   h4:h5, :w1], msk_aux[   h4:h5, w1:w2], msk_aux[   h4:h5, w2:]
        mskB_61, mskB_62, mskB_63 = msk_aux[   h5:  , :w1], msk_aux[   h5:  , w1:w2], msk_aux[   h5:  , w2:]
        
        img_aux1_1 = np.concatenate((imgA_11, imgB_12, imgA_13), axis=-1)
        img_aux1_2 = np.concatenate((imgB_21, imgA_22, imgB_23), axis=-1)
        img_aux1_3 = np.concatenate((imgA_31, imgB_32, imgA_33), axis=-1)
        img_aux1_4 = np.concatenate((imgB_41, imgA_42, imgB_43), axis=-1)
        img_aux1_5 = np.concatenate((imgA_51, imgB_52, imgA_53), axis=-1)
        img_aux1_6 = np.concatenate((imgB_61, imgA_62, imgB_63), axis=-1)
        img_aux1 = np.concatenate((img_aux1_1, img_aux1_2, img_aux1_3, img_aux1_4, img_aux1_5, img_aux1_6), axis=-2)

        lbl_aux1_1 = np.concatenate((lblA_11, lblB_12, lblA_13), axis=-1)
        lbl_aux1_2 = np.concatenate((lblB_21, lblA_22, lblB_23), axis=-1)
        lbl_aux1_3 = np.concatenate((lblA_31, lblB_32, lblA_33), axis=-1)
        lbl_aux1_4 = np.concatenate((lblB_41, lblA_42, lblB_43), axis=-1)
        lbl_aux1_5 = np.concatenate((lblA_51, lblB_52, lblA_53), axis=-1)
        lbl_aux1_6 = np.concatenate((lblB_61, lblA_62, lblB_63), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_1, lbl_aux1_2, lbl_aux1_3, lbl_aux1_4, lbl_aux1_5, lbl_aux1_6), axis=-2)

        msk_aux1_1 = np.concatenate((mskA_11, mskB_12, mskA_13), axis=-1)
        msk_aux1_2 = np.concatenate((mskB_21, mskA_22, mskB_23), axis=-1)
        msk_aux1_3 = np.concatenate((mskA_31, mskB_32, mskA_33), axis=-1)
        msk_aux1_4 = np.concatenate((mskB_41, mskA_42, mskB_43), axis=-1)
        msk_aux1_5 = np.concatenate((mskA_51, mskB_52, mskA_53), axis=-1)
        msk_aux1_6 = np.concatenate((mskB_61, mskA_62, mskB_63), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_1, msk_aux1_2, msk_aux1_3, msk_aux1_4, msk_aux1_5, msk_aux1_6), axis=-2)

        img_aux2_1 = np.concatenate((imgB_11, imgA_12, imgB_13), axis=-1)
        img_aux2_2 = np.concatenate((imgA_21, imgB_22, imgA_23), axis=-1)
        img_aux2_3 = np.concatenate((imgB_31, imgA_32, imgB_33), axis=-1)
        img_aux2_4 = np.concatenate((imgA_41, imgB_42, imgA_43), axis=-1)
        img_aux2_5 = np.concatenate((imgB_51, imgA_52, imgB_53), axis=-1)
        img_aux2_6 = np.concatenate((imgA_61, imgB_62, imgA_63), axis=-1)
        img_aux2 = np.concatenate((img_aux2_1, img_aux2_2, img_aux2_3, img_aux2_4, img_aux2_5, img_aux2_6), axis=-2)

        lbl_aux2_1 = np.concatenate((lblB_11, lblA_12, lblB_13), axis=-1)
        lbl_aux2_2 = np.concatenate((lblA_21, lblB_22, lblA_23), axis=-1)
        lbl_aux2_3 = np.concatenate((lblB_31, lblA_32, lblB_33), axis=-1)
        lbl_aux2_4 = np.concatenate((lblA_41, lblB_42, lblA_43), axis=-1)
        lbl_aux2_5 = np.concatenate((lblB_51, lblA_52, lblB_53), axis=-1)
        lbl_aux2_6 = np.concatenate((lblA_61, lblB_62, lblA_63), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_1, lbl_aux2_2, lbl_aux2_3, lbl_aux2_4, lbl_aux2_5, lbl_aux2_6), axis=-2)

        msk_aux2_1 = np.concatenate((mskB_11, mskA_12, mskB_13), axis=-1)
        msk_aux2_2 = np.concatenate((mskA_21, mskB_22, mskA_23), axis=-1)
        msk_aux2_3 = np.concatenate((mskB_31, mskA_32, mskB_33), axis=-1)
        msk_aux2_4 = np.concatenate((mskA_41, mskB_42, mskA_43), axis=-1)
        msk_aux2_5 = np.concatenate((mskB_51, mskA_52, mskB_53), axis=-1)
        msk_aux2_6 = np.concatenate((mskA_61, mskB_62, mskA_63), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_1, msk_aux2_2, msk_aux2_3, msk_aux2_4, msk_aux2_5, msk_aux2_6), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col4row1(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 4)   # 2048/4 = 512
        w2 = int(img.shape[-1] / 2)   # 2048/2 = 1024
        w3 = 3 * w1                   # 3 * 512 = 1536
        
        imgA_1, imgA_2, imgA_3, imgA_4 = img[:, :, :w1], img[:, :, w1:w2], img[:, :, w2:w3], img[:, :, w3:]  # 1 - 2 - 3 - 4
        lblA_1, lblA_2, lblA_3, lblA_4 = lbl[   :, :w1], lbl[   :, w1:w2], lbl[   :, w2:w3], lbl[   :, w3:]  # 1 - 2 - 3 - 4
        mskA_1, mskA_2, mskA_3, mskA_4 = msk[   :, :w1], msk[   :, w1:w2], msk[   :, w2:w3], msk[   :, w3:]  # 1 - 2 - 3 - 4
        
        imgB_1, imgB_2, imgB_3, imgB_4 = img_aux[:, :, :w1], img_aux[:, :, w1:w2], img_aux[:, :, w2:w3], img_aux[:, :, w3:]  # 1 - 2 - 3 - 4
        lblB_1, lblB_2, lblB_3, lblB_4 = lbl_aux[   :, :w1], lbl_aux[   :, w1:w2], lbl_aux[   :, w2:w3], lbl_aux[   :, w3:]  # 1 - 2 - 3 - 4
        mskB_1, mskB_2, mskB_3, mskB_4 = msk_aux[   :, :w1], msk_aux[   :, w1:w2], msk_aux[   :, w2:w3], msk_aux[   :, w3:]  # 1 - 2 - 3 - 4
        
        img_aux1 = np.concatenate((imgA_1, imgB_2, imgA_3, imgB_4), axis=-1)  # 1 - 2 - 3 - 4
        lbl_aux1 = np.concatenate((lblA_1, lblB_2, lblA_3, lblB_4), axis=-1)  # 1 - 2 - 3 - 4
        msk_aux1 = np.concatenate((mskA_1, mskB_2, mskA_3, mskB_4), axis=-1)  # 1 - 2 - 3 - 4

        img_aux2 = np.concatenate((imgB_1, imgA_2, imgB_3, imgA_4), axis=-1)  # 1 - 2 - 3 - 4
        lbl_aux2 = np.concatenate((lblB_1, lblA_2, lblB_3, lblA_4), axis=-1)  # 1 - 2 - 3 - 4
        msk_aux2 = np.concatenate((mskB_1, mskA_2, mskB_3, mskA_4), axis=-1)  # 1 - 2 - 3 - 4
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col4row2(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 4)   # 2048/4 = 512
        w2 = int(img.shape[-1] / 2)   # 2048/2 = 1024
        w3 = 3 * w1                   # 3 * 512 = 1536
        h1 = int(img.shape[-2] / 2)   # 64/2 = 32

        imgA_11, imgA_12, imgA_13, imgA_14 = img[:, :h1, :w1], img[:, :h1, w1:w2], img[:, :h1, w2:w3], img[:, :h1, w3:]
        imgA_21, imgA_22, imgA_23, imgA_24 = img[:, h1:, :w1], img[:, h1:, w1:w2], img[:, h1:, w2:w3], img[:, h1:, w3:]

        lblA_11, lblA_12, lblA_13, lblA_14 = lbl[   :h1, :w1], lbl[   :h1, w1:w2], lbl[   :h1, w2:w3], lbl[   :h1, w3:]
        lblA_21, lblA_22, lblA_23, lblA_24 = lbl[   h1:, :w1], lbl[   h1:, w1:w2], lbl[   h1:, w2:w3], lbl[   h1:, w3:]

        mskA_11, mskA_12, mskA_13, mskA_14 = msk[   :h1, :w1], msk[   :h1, w1:w2], msk[   :h1, w2:w3], msk[   :h1, w3:]
        mskA_21, mskA_22, mskA_23, mskA_24 = msk[   h1:, :w1], msk[   h1:, w1:w2], msk[   h1:, w2:w3], msk[   h1:, w3:]

        imgB_11, imgB_12, imgB_13, imgB_14 = img_aux[:, :h1, :w1], img_aux[:, :h1, w1:w2], img_aux[:, :h1, w2:w3], img_aux[:, :h1, w3:]
        imgB_21, imgB_22, imgB_23, imgB_24 = img_aux[:, h1:, :w1], img_aux[:, h1:, w1:w2], img_aux[:, h1:, w2:w3], img_aux[:, h1:, w3:]

        lblB_11, lblB_12, lblB_13, lblB_14 = lbl_aux[   :h1, :w1], lbl_aux[   :h1, w1:w2], lbl_aux[   :h1, w2:w3], lbl_aux[   :h1, w3:]
        lblB_21, lblB_22, lblB_23, lblB_24 = lbl_aux[   h1:, :w1], lbl_aux[   h1:, w1:w2], lbl_aux[   h1:, w2:w3], lbl_aux[   h1:, w3:]

        mskB_11, mskB_12, mskB_13, mskB_14 = msk_aux[   :h1, :w1], msk_aux[   :h1, w1:w2], msk_aux[   :h1, w2:w3], msk_aux[   :h1, w3:]
        mskB_21, mskB_22, mskB_23, mskB_24 = msk_aux[   h1:, :w1], msk_aux[   h1:, w1:w2], msk_aux[   h1:, w2:w3], msk_aux[   h1:, w3:]
        
        img_aux1_top = np.concatenate((imgA_11, imgB_12, imgA_13, imgB_14), axis=-1)
        img_aux1_bot = np.concatenate((imgB_21, imgA_22, imgB_23, imgA_24), axis=-1)
        img_aux1 = np.concatenate((img_aux1_top, img_aux1_bot), axis=-2)

        lbl_aux1_top = np.concatenate((lblA_11, lblB_12, lblA_13, lblB_14), axis=-1)
        lbl_aux1_bot = np.concatenate((lblB_21, lblA_22, lblB_23, lblA_24), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_top, lbl_aux1_bot), axis=-2)

        msk_aux1_top = np.concatenate((mskA_11, mskB_12, mskA_13, mskB_14), axis=-1)
        msk_aux1_bot = np.concatenate((mskB_21, mskA_22, mskB_23, mskA_24), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_top, msk_aux1_bot), axis=-2)

        img_aux2_top = np.concatenate((imgB_11, imgA_12, imgB_13, imgA_14), axis=-1)
        img_aux2_bot = np.concatenate((imgA_21, imgB_22, imgA_23, imgB_24), axis=-1)
        img_aux2 = np.concatenate((img_aux2_top, img_aux2_bot), axis=-2)

        lbl_aux2_top = np.concatenate((lblB_11, lblA_12, lblB_13, lblA_14), axis=-1)
        lbl_aux2_bot = np.concatenate((lblA_21, lblB_22, lblA_23, lblB_24), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_top, lbl_aux2_bot), axis=-2)

        msk_aux2_top = np.concatenate((mskB_11, mskA_12, mskB_13, mskA_14), axis=-1)
        msk_aux2_bot = np.concatenate((mskA_21, mskB_22, mskA_23, mskB_24), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_top, msk_aux2_bot), axis=-2)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col4row3(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 4)   # 2048/4 = 512
        w2 = int(img.shape[-1] / 2)   # 2048/2 = 1024
        w3 = 3 * w1                   # 3 * 512 = 1536
        h1 = int(img.shape[-2] / 3)   # 64/3 = 21
        h2 = 2 * h1                   # 2*21 = 42

        imgA_11, imgA_12, imgA_13, imgA_14 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:w3], img[:,   :h1, w3:]
        imgA_21, imgA_22, imgA_23, imgA_24 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:w3], img[:, h1:h2, w3:]
        imgA_31, imgA_32, imgA_33, imgA_34 = img[:, h2:  , :w1], img[:, h2:  , w1:w2], img[:, h2:  , w2:w3], img[:, h2:  , w3:]

        lblA_11, lblA_12, lblA_13, lblA_14 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:w3], lbl[     :h1, w3:]
        lblA_21, lblA_22, lblA_23, lblA_24 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:w3], lbl[   h1:h2, w3:]
        lblA_31, lblA_32, lblA_33, lblA_34 = lbl[   h2:  , :w1], lbl[   h2:  , w1:w2], lbl[   h2:  , w2:w3], lbl[   h2:  , w3:]

        mskA_11, mskA_12, mskA_13, mskA_14 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:w3], msk[     :h1, w3:]
        mskA_21, mskA_22, mskA_23, mskA_24 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:w3], msk[   h1:h2, w3:]
        mskA_31, mskA_32, mskA_33, mskA_34 = msk[   h2:  , :w1], msk[   h2:  , w1:w2], msk[   h2:  , w2:w3], msk[   h2:  , w3:]

        imgB_11, imgB_12, imgB_13, imgB_14 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:w3], img_aux[:,   :h1, w3:]
        imgB_21, imgB_22, imgB_23, imgB_24 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:w3], img_aux[:, h1:h2, w3:]
        imgB_31, imgB_32, imgB_33, imgB_34 = img_aux[:, h2:  , :w1], img_aux[:, h2:  , w1:w2], img_aux[:, h2:  , w2:w3], img_aux[:, h2:  , w3:]

        lblB_11, lblB_12, lblB_13, lblB_14 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:w3], lbl_aux[     :h1, w3:]
        lblB_21, lblB_22, lblB_23, lblB_24 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:w3], lbl_aux[   h1:h2, w3:]
        lblB_31, lblB_32, lblB_33, lblB_34 = lbl_aux[   h2:  , :w1], lbl_aux[   h2:  , w1:w2], lbl_aux[   h2:  , w2:w3], lbl_aux[   h2:  , w3:]

        mskB_11, mskB_12, mskB_13, mskB_14 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:w3], msk_aux[     :h1, w3:]
        mskB_21, mskB_22, mskB_23, mskB_24 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:w3], msk_aux[   h1:h2, w3:]
        mskB_31, mskB_32, mskB_33, mskB_34 = msk_aux[   h2:  , :w1], msk_aux[   h2:  , w1:w2], msk_aux[   h2:  , w2:w3], msk_aux[   h2:  , w3:]
        
        img_aux1_top = np.concatenate((imgA_11, imgB_12, imgA_13, imgB_14), axis=-1)
        img_aux1_mid = np.concatenate((imgB_21, imgA_22, imgB_23, imgA_24), axis=-1)
        img_aux1_bot = np.concatenate((imgA_31, imgB_32, imgA_33, imgB_34), axis=-1)
        img_aux1 = np.concatenate((img_aux1_top, img_aux1_mid, img_aux1_bot), axis=-2)

        lbl_aux1_top = np.concatenate((lblA_11, lblB_12, lblA_13, lblB_14), axis=-1)
        lbl_aux1_mid = np.concatenate((lblB_21, lblA_22, lblB_23, lblA_24), axis=-1)
        lbl_aux1_bot = np.concatenate((lblA_31, lblB_32, lblA_33, lblB_34), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_top, lbl_aux1_mid, lbl_aux1_bot), axis=-2)

        msk_aux1_top = np.concatenate((mskA_11, mskB_12, mskA_13, mskB_14), axis=-1)
        msk_aux1_mid = np.concatenate((mskB_21, mskA_22, mskB_23, mskA_24), axis=-1)
        msk_aux1_bot = np.concatenate((mskA_31, mskB_32, mskA_33, mskB_34), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_top, msk_aux1_mid, msk_aux1_bot), axis=-2)

        img_aux2_top = np.concatenate((imgB_11, imgA_12, imgB_13, imgA_14), axis=-1)
        img_aux2_mid = np.concatenate((imgA_21, imgB_22, imgA_23, imgB_24), axis=-1)
        img_aux2_bot = np.concatenate((imgB_31, imgA_32, imgB_33, imgA_34), axis=-1)
        img_aux2 = np.concatenate((img_aux2_top, img_aux2_mid, img_aux2_bot), axis=-2)

        lbl_aux2_top = np.concatenate((lblB_11, lblA_12, lblB_13, lblA_14), axis=-1)
        lbl_aux2_mid = np.concatenate((lblA_21, lblB_22, lblA_23, lblB_24), axis=-1)
        lbl_aux2_bot = np.concatenate((lblB_31, lblA_32, lblB_33, lblA_34), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_top, lbl_aux2_mid, lbl_aux2_bot), axis=-2)

        msk_aux2_top = np.concatenate((mskB_11, mskA_12, mskB_13, mskA_14), axis=-1)
        msk_aux2_mid = np.concatenate((mskA_21, mskB_22, mskA_23, mskB_24), axis=-1)
        msk_aux2_bot = np.concatenate((mskB_31, mskA_32, mskB_33, mskA_34), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_top, msk_aux2_mid, msk_aux2_bot), axis=-2)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col4row4(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 4)   # 2048/4 = 512
        w2 = int(img.shape[-1] / 2)   # 2048/2 = 1024
        w3 = 3 * w1                   # 3 * 512 = 1536
        h1 = int(img.shape[-2] / 4)   # 64/4 = 16
        h2 = 2 * h1                   # 2*16 = 32
        h3 = 3 * h1                   # 3*16 = 48

        imgA_11, imgA_12, imgA_13, imgA_14 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:w3], img[:,   :h1, w3:]
        imgA_21, imgA_22, imgA_23, imgA_24 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:w3], img[:, h1:h2, w3:]
        imgA_31, imgA_32, imgA_33, imgA_34 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:w3], img[:, h2:h3, w3:]
        imgA_41, imgA_42, imgA_43, imgA_44 = img[:, h3:  , :w1], img[:, h3:  , w1:w2], img[:, h3:  , w2:w3], img[:, h3:  , w3:]

        lblA_11, lblA_12, lblA_13, lblA_14 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:w3], lbl[     :h1, w3:]
        lblA_21, lblA_22, lblA_23, lblA_24 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:w3], lbl[   h1:h2, w3:]
        lblA_31, lblA_32, lblA_33, lblA_34 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:w3], lbl[   h2:h3, w3:]
        lblA_41, lblA_42, lblA_43, lblA_44 = lbl[   h3:  , :w1], lbl[   h3:  , w1:w2], lbl[   h3:  , w2:w3], lbl[   h3:  , w3:]

        mskA_11, mskA_12, mskA_13, mskA_14 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:w3], msk[     :h1, w3:]
        mskA_21, mskA_22, mskA_23, mskA_24 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:w3], msk[   h1:h2, w3:]
        mskA_31, mskA_32, mskA_33, mskA_34 = msk[   h2:h3, :w1], msk[   h2:h3, w1:w2], msk[   h2:h3, w2:w3], msk[   h2:h3, w3:]
        mskA_41, mskA_42, mskA_43, mskA_44 = msk[   h3:  , :w1], msk[   h3:  , w1:w2], msk[   h3:  , w2:w3], msk[   h3:  , w3:]

        imgB_11, imgB_12, imgB_13, imgB_14 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:w3], img_aux[:,   :h1, w3:]
        imgB_21, imgB_22, imgB_23, imgB_24 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:w3], img_aux[:, h1:h2, w3:]
        imgB_31, imgB_32, imgB_33, imgB_34 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:w3], img_aux[:, h2:h3, w3:]
        imgB_41, imgB_42, imgB_43, imgB_44 = img_aux[:, h3:  , :w1], img_aux[:, h3:  , w1:w2], img_aux[:, h3:  , w2:w3], img_aux[:, h3:  , w3:]

        lblB_11, lblB_12, lblB_13, lblB_14 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:w3], lbl_aux[     :h1, w3:]
        lblB_21, lblB_22, lblB_23, lblB_24 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:w3], lbl_aux[   h1:h2, w3:]
        lblB_31, lblB_32, lblB_33, lblB_34 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:w3], lbl_aux[   h2:h3, w3:]
        lblB_41, lblB_42, lblB_43, lblB_44 = lbl_aux[   h3:  , :w1], lbl_aux[   h3:  , w1:w2], lbl_aux[   h3:  , w2:w3], lbl_aux[   h3:  , w3:]

        mskB_11, mskB_12, mskB_13, mskB_14 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:w3], msk_aux[     :h1, w3:]
        mskB_21, mskB_22, mskB_23, mskB_24 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:w3], msk_aux[   h1:h2, w3:]
        mskB_31, mskB_32, mskB_33, mskB_34 = msk_aux[   h2:h3, :w1], msk_aux[   h2:h3, w1:w2], msk_aux[   h2:h3, w2:w3], msk_aux[   h2:h3, w3:]
        mskB_41, mskB_42, mskB_43, mskB_44 = msk_aux[   h3:  , :w1], msk_aux[   h3:  , w1:w2], msk_aux[   h3:  , w2:w3], msk_aux[   h3:  , w3:]
        
        img_aux1_1 = np.concatenate((imgA_11, imgB_12, imgA_13, imgB_14), axis=-1)
        img_aux1_2 = np.concatenate((imgB_21, imgA_22, imgB_23, imgA_24), axis=-1)
        img_aux1_3 = np.concatenate((imgA_31, imgB_32, imgA_33, imgB_34), axis=-1)
        img_aux1_4 = np.concatenate((imgB_41, imgA_42, imgB_43, imgA_44), axis=-1)
        img_aux1 = np.concatenate((img_aux1_1, img_aux1_2, img_aux1_3, img_aux1_4), axis=-2)

        lbl_aux1_1 = np.concatenate((lblA_11, lblB_12, lblA_13, lblB_14), axis=-1)
        lbl_aux1_2 = np.concatenate((lblB_21, lblA_22, lblB_23, lblA_24), axis=-1)
        lbl_aux1_3 = np.concatenate((lblA_31, lblB_32, lblA_33, lblB_34), axis=-1)
        lbl_aux1_4 = np.concatenate((lblB_41, lblA_42, lblB_43, lblA_44), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_1, lbl_aux1_2, lbl_aux1_3, lbl_aux1_4), axis=-2)

        msk_aux1_1 = np.concatenate((mskA_11, mskB_12, mskA_13, mskB_14), axis=-1)
        msk_aux1_2 = np.concatenate((mskB_21, mskA_22, mskB_23, mskA_24), axis=-1)
        msk_aux1_3 = np.concatenate((mskA_31, mskB_32, mskA_33, mskB_34), axis=-1)
        msk_aux1_4 = np.concatenate((mskB_41, mskA_42, mskB_43, mskA_44), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_1, msk_aux1_2, msk_aux1_3, msk_aux1_4), axis=-2)

        img_aux2_1 = np.concatenate((imgB_11, imgA_12, imgB_13, imgA_14), axis=-1)
        img_aux2_2 = np.concatenate((imgA_21, imgB_22, imgA_23, imgB_24), axis=-1)
        img_aux2_3 = np.concatenate((imgB_31, imgA_32, imgB_33, imgA_34), axis=-1)
        img_aux2_4 = np.concatenate((imgA_41, imgB_42, imgA_43, imgB_44), axis=-1)
        img_aux2 = np.concatenate((img_aux2_1, img_aux2_2, img_aux2_3, img_aux2_4), axis=-2)

        lbl_aux2_1 = np.concatenate((lblB_11, lblA_12, lblB_13, lblA_14), axis=-1)
        lbl_aux2_2 = np.concatenate((lblA_21, lblB_22, lblA_23, lblB_24), axis=-1)
        lbl_aux2_3 = np.concatenate((lblB_31, lblA_32, lblB_33, lblA_34), axis=-1)
        lbl_aux2_4 = np.concatenate((lblA_41, lblB_42, lblA_43, lblB_44), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_1, lbl_aux2_2, lbl_aux2_3, lbl_aux2_4), axis=-2)

        msk_aux2_1 = np.concatenate((mskB_11, mskA_12, mskB_13, mskA_14), axis=-1)
        msk_aux2_2 = np.concatenate((mskA_21, mskB_22, mskA_23, mskB_24), axis=-1)
        msk_aux2_3 = np.concatenate((mskB_31, mskA_32, mskB_33, mskA_34), axis=-1)
        msk_aux2_4 = np.concatenate((mskA_41, mskB_42, mskA_43, mskB_44), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_1, msk_aux2_2, msk_aux2_3, msk_aux2_4), axis=-2)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col4row5(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 4)   # 2048/4 = 512
        w2 = int(img.shape[-1] / 2)   # 2048/2 = 1024
        w3 = 3 * w1                   # 3 * 512 = 1536
        h1 = int(img.shape[-2] / 5)   # 64/5 = 12
        h2 = 2 * h1                   # 2*12 = 24
        h3 = 3 * h1                   # 3*12 = 36
        h4 = 4 * h1                   # 4*12 = 48

        imgA_11, imgA_12, imgA_13, imgA_14 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:w3], img[:,   :h1, w3:]
        imgA_21, imgA_22, imgA_23, imgA_24 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:w3], img[:, h1:h2, w3:]
        imgA_31, imgA_32, imgA_33, imgA_34 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:w3], img[:, h2:h3, w3:]
        imgA_41, imgA_42, imgA_43, imgA_44 = img[:, h3:h4, :w1], img[:, h3:h4, w1:w2], img[:, h3:h4, w2:w3], img[:, h3:h4, w3:]
        imgA_51, imgA_52, imgA_53, imgA_54 = img[:, h4:  , :w1], img[:, h4:  , w1:w2], img[:, h4:  , w2:w3], img[:, h4:  , w3:]

        lblA_11, lblA_12, lblA_13, lblA_14 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:w3], lbl[     :h1, w3:]
        lblA_21, lblA_22, lblA_23, lblA_24 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:w3], lbl[   h1:h2, w3:]
        lblA_31, lblA_32, lblA_33, lblA_34 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:w3], lbl[   h2:h3, w3:]
        lblA_41, lblA_42, lblA_43, lblA_44 = lbl[   h3:h4, :w1], lbl[   h3:h4, w1:w2], lbl[   h3:h4, w2:w3], lbl[   h3:h4, w3:]
        lblA_51, lblA_52, lblA_53, lblA_54 = lbl[   h4:  , :w1], lbl[   h4:  , w1:w2], lbl[   h4:  , w2:w3], lbl[   h4:  , w3:]

        mskA_11, mskA_12, mskA_13, mskA_14 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:w3], msk[     :h1, w3:]
        mskA_21, mskA_22, mskA_23, mskA_24 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:w3], msk[   h1:h2, w3:]
        mskA_31, mskA_32, mskA_33, mskA_34 = msk[   h2:h3, :w1], msk[   h2:h3, w1:w2], msk[   h2:h3, w2:w3], msk[   h2:h3, w3:]
        mskA_41, mskA_42, mskA_43, mskA_44 = msk[   h3:h4, :w1], msk[   h3:h4, w1:w2], msk[   h3:h4, w2:w3], msk[   h3:h4, w3:]
        mskA_51, mskA_52, mskA_53, mskA_54 = msk[   h4:  , :w1], msk[   h4:  , w1:w2], msk[   h4:  , w2:w3], msk[   h4:  , w3:]

        imgB_11, imgB_12, imgB_13, imgB_14 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:w3], img_aux[:,   :h1, w3:]
        imgB_21, imgB_22, imgB_23, imgB_24 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:w3], img_aux[:, h1:h2, w3:]
        imgB_31, imgB_32, imgB_33, imgB_34 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:w3], img_aux[:, h2:h3, w3:]
        imgB_41, imgB_42, imgB_43, imgB_44 = img_aux[:, h3:h4, :w1], img_aux[:, h3:h4, w1:w2], img_aux[:, h3:h4, w2:w3], img_aux[:, h3:h4, w3:]
        imgB_51, imgB_52, imgB_53, imgB_54 = img_aux[:, h4:  , :w1], img_aux[:, h4:  , w1:w2], img_aux[:, h4:  , w2:w3], img_aux[:, h4:  , w3:]

        lblB_11, lblB_12, lblB_13, lblB_14 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:w3], lbl_aux[     :h1, w3:]
        lblB_21, lblB_22, lblB_23, lblB_24 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:w3], lbl_aux[   h1:h2, w3:]
        lblB_31, lblB_32, lblB_33, lblB_34 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:w3], lbl_aux[   h2:h3, w3:]
        lblB_41, lblB_42, lblB_43, lblB_44 = lbl_aux[   h3:h4, :w1], lbl_aux[   h3:h4, w1:w2], lbl_aux[   h3:h4, w2:w3], lbl_aux[   h3:h4, w3:]
        lblB_51, lblB_52, lblB_53, lblB_54 = lbl_aux[   h4:  , :w1], lbl_aux[   h4:  , w1:w2], lbl_aux[   h4:  , w2:w3], lbl_aux[   h4:  , w3:]

        mskB_11, mskB_12, mskB_13, mskB_14 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:w3], msk_aux[     :h1, w3:]
        mskB_21, mskB_22, mskB_23, mskB_24 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:w3], msk_aux[   h1:h2, w3:]
        mskB_31, mskB_32, mskB_33, mskB_34 = msk_aux[   h2:h3, :w1], msk_aux[   h2:h3, w1:w2], msk_aux[   h2:h3, w2:w3], msk_aux[   h2:h3, w3:]
        mskB_41, mskB_42, mskB_43, mskB_44 = msk_aux[   h3:h4, :w1], msk_aux[   h3:h4, w1:w2], msk_aux[   h3:h4, w2:w3], msk_aux[   h3:h4, w3:]
        mskB_51, mskB_52, mskB_53, mskB_54 = msk_aux[   h4:  , :w1], msk_aux[   h4:  , w1:w2], msk_aux[   h4:  , w2:w3], msk_aux[   h4:  , w3:]
        
        img_aux1_1 = np.concatenate((imgA_11, imgB_12, imgA_13, imgB_14), axis=-1)
        img_aux1_2 = np.concatenate((imgB_21, imgA_22, imgB_23, imgA_24), axis=-1)
        img_aux1_3 = np.concatenate((imgA_31, imgB_32, imgA_33, imgB_34), axis=-1)
        img_aux1_4 = np.concatenate((imgB_41, imgA_42, imgB_43, imgA_44), axis=-1)
        img_aux1_5 = np.concatenate((imgA_51, imgB_52, imgA_53, imgB_54), axis=-1)
        img_aux1 = np.concatenate((img_aux1_1, img_aux1_2, img_aux1_3, img_aux1_4, img_aux1_5), axis=-2)

        lbl_aux1_1 = np.concatenate((lblA_11, lblB_12, lblA_13, lblB_14), axis=-1)
        lbl_aux1_2 = np.concatenate((lblB_21, lblA_22, lblB_23, lblA_24), axis=-1)
        lbl_aux1_3 = np.concatenate((lblA_31, lblB_32, lblA_33, lblB_34), axis=-1)
        lbl_aux1_4 = np.concatenate((lblB_41, lblA_42, lblB_43, lblA_44), axis=-1)
        lbl_aux1_5 = np.concatenate((lblA_51, lblB_52, lblA_53, lblB_54), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_1, lbl_aux1_2, lbl_aux1_3, lbl_aux1_4, lbl_aux1_5), axis=-2)

        msk_aux1_1 = np.concatenate((mskA_11, mskB_12, mskA_13, mskB_14), axis=-1)
        msk_aux1_2 = np.concatenate((mskB_21, mskA_22, mskB_23, mskA_24), axis=-1)
        msk_aux1_3 = np.concatenate((mskA_31, mskB_32, mskA_33, mskB_34), axis=-1)
        msk_aux1_4 = np.concatenate((mskB_41, mskA_42, mskB_43, mskA_44), axis=-1)
        msk_aux1_5 = np.concatenate((mskA_51, mskB_52, mskA_53, mskB_54), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_1, msk_aux1_2, msk_aux1_3, msk_aux1_4, msk_aux1_5), axis=-2)

        img_aux2_1 = np.concatenate((imgB_11, imgA_12, imgB_13, imgA_14), axis=-1)
        img_aux2_2 = np.concatenate((imgA_21, imgB_22, imgA_23, imgB_24), axis=-1)
        img_aux2_3 = np.concatenate((imgB_31, imgA_32, imgB_33, imgA_34), axis=-1)
        img_aux2_4 = np.concatenate((imgA_41, imgB_42, imgA_43, imgB_44), axis=-1)
        img_aux2_5 = np.concatenate((imgB_51, imgA_52, imgB_53, imgA_54), axis=-1)
        img_aux2 = np.concatenate((img_aux2_1, img_aux2_2, img_aux2_3, img_aux2_4, img_aux2_5), axis=-2)

        lbl_aux2_1 = np.concatenate((lblB_11, lblA_12, lblB_13, lblA_14), axis=-1)
        lbl_aux2_2 = np.concatenate((lblA_21, lblB_22, lblA_23, lblB_24), axis=-1)
        lbl_aux2_3 = np.concatenate((lblB_31, lblA_32, lblB_33, lblA_34), axis=-1)
        lbl_aux2_4 = np.concatenate((lblA_41, lblB_42, lblA_43, lblB_44), axis=-1)
        lbl_aux2_5 = np.concatenate((lblB_51, lblA_52, lblB_53, lblA_54), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_1, lbl_aux2_2, lbl_aux2_3, lbl_aux2_4, lbl_aux2_5), axis=-2)

        msk_aux2_1 = np.concatenate((mskB_11, mskA_12, mskB_13, mskA_14), axis=-1)
        msk_aux2_2 = np.concatenate((mskA_21, mskB_22, mskA_23, mskB_24), axis=-1)
        msk_aux2_3 = np.concatenate((mskB_31, mskA_32, mskB_33, mskA_34), axis=-1)
        msk_aux2_4 = np.concatenate((mskA_41, mskB_42, mskA_43, mskB_44), axis=-1)
        msk_aux2_5 = np.concatenate((mskB_51, mskA_52, mskB_53, mskA_54), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_1, msk_aux2_2, msk_aux2_3, msk_aux2_4, msk_aux2_5), axis=-2)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col4row6(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 4)   # 2048/4 = 512
        w2 = int(img.shape[-1] / 2)   # 2048/2 = 1024
        w3 = 3 * w1                   # 3 * 512 = 1536
        h1 = int(img.shape[-2] / 6)   # 64/6 = 10
        h2 = 2 * h1                   # 2*10 = 20
        h3 = 3 * h1                   # 3*10 = 30
        h4 = 4 * h1                   # 4*10 = 40
        h5 = 5 * h1                   # 5*10 = 50

        imgA_11, imgA_12, imgA_13, imgA_14 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:w3], img[:,   :h1, w3:]
        imgA_21, imgA_22, imgA_23, imgA_24 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:w3], img[:, h1:h2, w3:]
        imgA_31, imgA_32, imgA_33, imgA_34 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:w3], img[:, h2:h3, w3:]
        imgA_41, imgA_42, imgA_43, imgA_44 = img[:, h3:h4, :w1], img[:, h3:h4, w1:w2], img[:, h3:h4, w2:w3], img[:, h3:h4, w3:]
        imgA_51, imgA_52, imgA_53, imgA_54 = img[:, h4:h5, :w1], img[:, h4:h5, w1:w2], img[:, h4:h5, w2:w3], img[:, h4:h5, w3:]
        imgA_61, imgA_62, imgA_63, imgA_64 = img[:, h5:  , :w1], img[:, h5:  , w1:w2], img[:, h5:  , w2:w3], img[:, h5:  , w3:]

        lblA_11, lblA_12, lblA_13, lblA_14 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:w3], lbl[     :h1, w3:]
        lblA_21, lblA_22, lblA_23, lblA_24 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:w3], lbl[   h1:h2, w3:]
        lblA_31, lblA_32, lblA_33, lblA_34 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:w3], lbl[   h2:h3, w3:]
        lblA_41, lblA_42, lblA_43, lblA_44 = lbl[   h3:h4, :w1], lbl[   h3:h4, w1:w2], lbl[   h3:h4, w2:w3], lbl[   h3:h4, w3:]
        lblA_51, lblA_52, lblA_53, lblA_54 = lbl[   h4:h5, :w1], lbl[   h4:h5, w1:w2], lbl[   h4:h5, w2:w3], lbl[   h4:h5, w3:]
        lblA_61, lblA_62, lblA_63, lblA_64 = lbl[   h5:  , :w1], lbl[   h5:  , w1:w2], lbl[   h5:  , w2:w3], lbl[   h5:  , w3:]

        mskA_11, mskA_12, mskA_13, mskA_14 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:w3], msk[     :h1, w3:]
        mskA_21, mskA_22, mskA_23, mskA_24 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:w3], msk[   h1:h2, w3:]
        mskA_31, mskA_32, mskA_33, mskA_34 = msk[   h2:h3, :w1], msk[   h2:h3, w1:w2], msk[   h2:h3, w2:w3], msk[   h2:h3, w3:]
        mskA_41, mskA_42, mskA_43, mskA_44 = msk[   h3:h4, :w1], msk[   h3:h4, w1:w2], msk[   h3:h4, w2:w3], msk[   h3:h4, w3:]
        mskA_51, mskA_52, mskA_53, mskA_54 = msk[   h4:h5, :w1], msk[   h4:h5, w1:w2], msk[   h4:h5, w2:w3], msk[   h4:h5, w3:]
        mskA_61, mskA_62, mskA_63, mskA_64 = msk[   h5:  , :w1], msk[   h5:  , w1:w2], msk[   h5:  , w2:w3], msk[   h5:  , w3:]

        imgB_11, imgB_12, imgB_13, imgB_14 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:w3], img_aux[:,   :h1, w3:]
        imgB_21, imgB_22, imgB_23, imgB_24 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:w3], img_aux[:, h1:h2, w3:]
        imgB_31, imgB_32, imgB_33, imgB_34 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:w3], img_aux[:, h2:h3, w3:]
        imgB_41, imgB_42, imgB_43, imgB_44 = img_aux[:, h3:h4, :w1], img_aux[:, h3:h4, w1:w2], img_aux[:, h3:h4, w2:w3], img_aux[:, h3:h4, w3:]
        imgB_51, imgB_52, imgB_53, imgB_54 = img_aux[:, h4:h5, :w1], img_aux[:, h4:h5, w1:w2], img_aux[:, h4:h5, w2:w3], img_aux[:, h4:h5, w3:]
        imgB_61, imgB_62, imgB_63, imgB_64 = img_aux[:, h5:  , :w1], img_aux[:, h5:  , w1:w2], img_aux[:, h5:  , w2:w3], img_aux[:, h5:  , w3:]

        lblB_11, lblB_12, lblB_13, lblB_14 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:w3], lbl_aux[     :h1, w3:]
        lblB_21, lblB_22, lblB_23, lblB_24 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:w3], lbl_aux[   h1:h2, w3:]
        lblB_31, lblB_32, lblB_33, lblB_34 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:w3], lbl_aux[   h2:h3, w3:]
        lblB_41, lblB_42, lblB_43, lblB_44 = lbl_aux[   h3:h4, :w1], lbl_aux[   h3:h4, w1:w2], lbl_aux[   h3:h4, w2:w3], lbl_aux[   h3:h4, w3:]
        lblB_51, lblB_52, lblB_53, lblB_54 = lbl_aux[   h4:h5, :w1], lbl_aux[   h4:h5, w1:w2], lbl_aux[   h4:h5, w2:w3], lbl_aux[   h4:h5, w3:]
        lblB_61, lblB_62, lblB_63, lblB_64 = lbl_aux[   h5:  , :w1], lbl_aux[   h5:  , w1:w2], lbl_aux[   h5:  , w2:w3], lbl_aux[   h5:  , w3:]

        mskB_11, mskB_12, mskB_13, mskB_14 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:w3], msk_aux[     :h1, w3:]
        mskB_21, mskB_22, mskB_23, mskB_24 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:w3], msk_aux[   h1:h2, w3:]
        mskB_31, mskB_32, mskB_33, mskB_34 = msk_aux[   h2:h3, :w1], msk_aux[   h2:h3, w1:w2], msk_aux[   h2:h3, w2:w3], msk_aux[   h2:h3, w3:]
        mskB_41, mskB_42, mskB_43, mskB_44 = msk_aux[   h3:h4, :w1], msk_aux[   h3:h4, w1:w2], msk_aux[   h3:h4, w2:w3], msk_aux[   h3:h4, w3:]
        mskB_51, mskB_52, mskB_53, mskB_54 = msk_aux[   h4:h5, :w1], msk_aux[   h4:h5, w1:w2], msk_aux[   h4:h5, w2:w3], msk_aux[   h4:h5, w3:]
        mskB_61, mskB_62, mskB_63, mskB_64 = msk_aux[   h5:  , :w1], msk_aux[   h5:  , w1:w2], msk_aux[   h5:  , w2:w3], msk_aux[   h5:  , w3:]
        
        img_aux1_1 = np.concatenate((imgA_11, imgB_12, imgA_13, imgB_14), axis=-1)
        img_aux1_2 = np.concatenate((imgB_21, imgA_22, imgB_23, imgA_24), axis=-1)
        img_aux1_3 = np.concatenate((imgA_31, imgB_32, imgA_33, imgB_34), axis=-1)
        img_aux1_4 = np.concatenate((imgB_41, imgA_42, imgB_43, imgA_44), axis=-1)
        img_aux1_5 = np.concatenate((imgA_51, imgB_52, imgA_53, imgB_54), axis=-1)
        img_aux1_6 = np.concatenate((imgB_61, imgA_62, imgB_63, imgA_64), axis=-1)
        img_aux1 = np.concatenate((img_aux1_1, img_aux1_2, img_aux1_3, img_aux1_4, img_aux1_5, img_aux1_6), axis=-2)

        lbl_aux1_1 = np.concatenate((lblA_11, lblB_12, lblA_13, lblB_14), axis=-1)
        lbl_aux1_2 = np.concatenate((lblB_21, lblA_22, lblB_23, lblA_24), axis=-1)
        lbl_aux1_3 = np.concatenate((lblA_31, lblB_32, lblA_33, lblB_34), axis=-1)
        lbl_aux1_4 = np.concatenate((lblB_41, lblA_42, lblB_43, lblA_44), axis=-1)
        lbl_aux1_5 = np.concatenate((lblA_51, lblB_52, lblA_53, lblB_54), axis=-1)
        lbl_aux1_6 = np.concatenate((lblB_61, lblA_62, lblB_63, lblA_64), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_1, lbl_aux1_2, lbl_aux1_3, lbl_aux1_4, lbl_aux1_5, lbl_aux1_6), axis=-2)

        msk_aux1_1 = np.concatenate((mskA_11, mskB_12, mskA_13, mskB_14), axis=-1)
        msk_aux1_2 = np.concatenate((mskB_21, mskA_22, mskB_23, mskA_24), axis=-1)
        msk_aux1_3 = np.concatenate((mskA_31, mskB_32, mskA_33, mskB_34), axis=-1)
        msk_aux1_4 = np.concatenate((mskB_41, mskA_42, mskB_43, mskA_44), axis=-1)
        msk_aux1_5 = np.concatenate((mskA_51, mskB_52, mskA_53, mskB_54), axis=-1)
        msk_aux1_6 = np.concatenate((mskB_61, mskA_62, mskB_63, mskA_64), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_1, msk_aux1_2, msk_aux1_3, msk_aux1_4, msk_aux1_5, msk_aux1_6), axis=-2)

        img_aux2_1 = np.concatenate((imgB_11, imgA_12, imgB_13, imgA_14), axis=-1)
        img_aux2_2 = np.concatenate((imgA_21, imgB_22, imgA_23, imgB_24), axis=-1)
        img_aux2_3 = np.concatenate((imgB_31, imgA_32, imgB_33, imgA_34), axis=-1)
        img_aux2_4 = np.concatenate((imgA_41, imgB_42, imgA_43, imgB_44), axis=-1)
        img_aux2_5 = np.concatenate((imgB_51, imgA_52, imgB_53, imgA_54), axis=-1)
        img_aux2_6 = np.concatenate((imgA_61, imgB_62, imgA_63, imgB_64), axis=-1)
        img_aux2 = np.concatenate((img_aux2_1, img_aux2_2, img_aux2_3, img_aux2_4, img_aux2_5, img_aux2_6), axis=-2)

        lbl_aux2_1 = np.concatenate((lblB_11, lblA_12, lblB_13, lblA_14), axis=-1)
        lbl_aux2_2 = np.concatenate((lblA_21, lblB_22, lblA_23, lblB_24), axis=-1)
        lbl_aux2_3 = np.concatenate((lblB_31, lblA_32, lblB_33, lblA_34), axis=-1)
        lbl_aux2_4 = np.concatenate((lblA_41, lblB_42, lblA_43, lblB_44), axis=-1)
        lbl_aux2_5 = np.concatenate((lblB_51, lblA_52, lblB_53, lblA_54), axis=-1)
        lbl_aux2_6 = np.concatenate((lblA_61, lblB_62, lblA_63, lblB_64), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_1, lbl_aux2_2, lbl_aux2_3, lbl_aux2_4, lbl_aux2_5, lbl_aux2_6), axis=-2)

        msk_aux2_1 = np.concatenate((mskB_11, mskA_12, mskB_13, mskA_14), axis=-1)
        msk_aux2_2 = np.concatenate((mskA_21, mskB_22, mskA_23, mskB_24), axis=-1)
        msk_aux2_3 = np.concatenate((mskB_31, mskA_32, mskB_33, mskA_34), axis=-1)
        msk_aux2_4 = np.concatenate((mskA_41, mskB_42, mskA_43, mskB_44), axis=-1)
        msk_aux2_5 = np.concatenate((mskB_51, mskA_52, mskB_53, mskA_54), axis=-1)
        msk_aux2_6 = np.concatenate((mskA_61, mskB_62, mskA_63, mskB_64), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_1, msk_aux2_2, msk_aux2_3, msk_aux2_4, msk_aux2_5, msk_aux2_6), axis=-2)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col6row4(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 4)   # 2048/6 = 341
        w2 = 2 * w1                   # 2*341  = 682
        w3 = 3 * w1                   # 3*341  = 1023
        w4 = 4 * w1                   # 4*341  = 1361
        w5 = 5 * w1                   # 5*341  = 1705
        h1 = int(img.shape[-2] / 4)   # 64/4 = 16
        h2 = 2 * h1                   # 2*16 = 32
        h3 = 3 * h1                   # 3*16 = 48

        imgA_11, imgA_12, imgA_13, imgA_14, imgA_15, imgA_16 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:w3], img[:,   :h1, w3:w4], img[:,   :h1, w4:w5], img[:,   :h1, w5:]
        imgA_21, imgA_22, imgA_23, imgA_24, imgA_25, imgA_26 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:w3], img[:, h1:h2, w3:w4], img[:, h1:h2, w4:w5], img[:, h1:h2, w5:]
        imgA_31, imgA_32, imgA_33, imgA_34, imgA_35, imgA_36 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:w3], img[:, h2:h3, w3:w4], img[:, h2:h3, w4:w5], img[:, h2:h3, w5:]
        imgA_41, imgA_42, imgA_43, imgA_44, imgA_45, imgA_46 = img[:, h3:  , :w1], img[:, h3:  , w1:w2], img[:, h3:  , w2:w3], img[:, h3:  , w3:w4], img[:, h3:  , w4:w5], img[:, h3:  , w5:]

        lblA_11, lblA_12, lblA_13, lblA_14, lblA_15, lblA_16 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:w3], lbl[     :h1, w3:w4], lbl[     :h1, w4:w5], lbl[     :h1, w5:]
        lblA_21, lblA_22, lblA_23, lblA_24, lblA_25, lblA_26 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:w3], lbl[   h1:h2, w3:w4], lbl[   h1:h2, w4:w5], lbl[   h1:h2, w5:]
        lblA_31, lblA_32, lblA_33, lblA_34, lblA_35, lblA_36 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:w3], lbl[   h2:h3, w3:w4], lbl[   h2:h3, w4:w5], lbl[   h2:h3, w5:]
        lblA_41, lblA_42, lblA_43, lblA_44, lblA_45, lblA_46 = lbl[   h3:  , :w1], lbl[   h3:  , w1:w2], lbl[   h3:  , w2:w3], lbl[   h3:  , w3:w4], lbl[   h3:  , w4:w5], lbl[   h3:  , w5:]

        mskA_11, mskA_12, mskA_13, mskA_14, mskA_15, mskA_16 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:w3], msk[     :h1, w3:w4], msk[     :h1, w4:w5], msk[     :h1, w5:]
        mskA_21, mskA_22, mskA_23, mskA_24, mskA_25, mskA_26 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:w3], msk[   h1:h2, w3:w4], msk[   h1:h2, w4:w5], msk[   h1:h2, w5:]
        mskA_31, mskA_32, mskA_33, mskA_34, mskA_35, mskA_36 = msk[   h2:h3, :w1], msk[   h2:h3, w1:w2], msk[   h2:h3, w2:w3], msk[   h2:h3, w3:w4], msk[   h2:h3, w4:w5], msk[   h2:h3, w5:]
        mskA_41, mskA_42, mskA_43, mskA_44, mskA_45, mskA_46 = msk[   h3:  , :w1], msk[   h3:  , w1:w2], msk[   h3:  , w2:w3], msk[   h3:  , w3:w4], msk[   h3:  , w4:w5], msk[   h3:  , w5:]

        imgB_11, imgB_12, imgB_13, imgB_14, imgB_15, imgB_16 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:w3], img_aux[:,   :h1, w3:w4], img_aux[:,   :h1, w4:w5], img_aux[:,   :h1, w5:]
        imgB_21, imgB_22, imgB_23, imgB_24, imgB_25, imgB_26 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:w3], img_aux[:, h1:h2, w3:w4], img_aux[:, h1:h2, w4:w5], img_aux[:, h1:h2, w5:]
        imgB_31, imgB_32, imgB_33, imgB_34, imgB_35, imgB_36 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:w3], img_aux[:, h2:h3, w3:w4], img_aux[:, h2:h3, w4:w5], img_aux[:, h2:h3, w5:]
        imgB_41, imgB_42, imgB_43, imgB_44, imgB_45, imgB_46 = img_aux[:, h3:  , :w1], img_aux[:, h3:  , w1:w2], img_aux[:, h3:  , w2:w3], img_aux[:, h3:  , w3:w4], img_aux[:, h3:  , w4:w5], img_aux[:, h3:  , w5:]

        lblB_11, lblB_12, lblB_13, lblB_14, lblB_15, lblB_16 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:w3], lbl_aux[     :h1, w3:w4], lbl_aux[     :h1, w4:w5], lbl_aux[     :h1, w5:]
        lblB_21, lblB_22, lblB_23, lblB_24, lblB_25, lblB_26 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:w3], lbl_aux[   h1:h2, w3:w4], lbl_aux[   h1:h2, w4:w5], lbl_aux[   h1:h2, w5:]
        lblB_31, lblB_32, lblB_33, lblB_34, lblB_35, lblB_36 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:w3], lbl_aux[   h2:h3, w3:w4], lbl_aux[   h2:h3, w4:w5], lbl_aux[   h2:h3, w5:]
        lblB_41, lblB_42, lblB_43, lblB_44, lblB_45, lblB_46 = lbl_aux[   h3:  , :w1], lbl_aux[   h3:  , w1:w2], lbl_aux[   h3:  , w2:w3], lbl_aux[   h3:  , w3:w4], lbl_aux[   h3:  , w4:w5], lbl_aux[   h3:  , w5:]

        mskB_11, mskB_12, mskB_13, mskB_14, mskB_15, mskB_16 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:w3], msk_aux[     :h1, w3:w4], msk_aux[     :h1, w4:w5], msk_aux[     :h1, w5:]
        mskB_21, mskB_22, mskB_23, mskB_24, mskB_25, mskB_26 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:w3], msk_aux[   h1:h2, w3:w4], msk_aux[   h1:h2, w4:w5], msk_aux[   h1:h2, w5:]
        mskB_31, mskB_32, mskB_33, mskB_34, mskB_35, mskB_36 = msk_aux[   h2:h3, :w1], msk_aux[   h2:h3, w1:w2], msk_aux[   h2:h3, w2:w3], msk_aux[   h2:h3, w3:w4], msk_aux[   h2:h3, w4:w5], msk_aux[   h2:h3, w5:]
        mskB_41, mskB_42, mskB_43, mskB_44, mskB_45, mskB_46 = msk_aux[   h3:  , :w1], msk_aux[   h3:  , w1:w2], msk_aux[   h3:  , w2:w3], msk_aux[   h3:  , w3:w4], msk_aux[   h3:  , w4:w5], msk_aux[   h3:  , w5:]
        
        img_aux1_1 = np.concatenate((imgA_11, imgB_12, imgA_13, imgB_14, imgA_15, imgB_16), axis=-1)
        img_aux1_2 = np.concatenate((imgB_21, imgA_22, imgB_23, imgA_24, imgB_25, imgA_26), axis=-1)
        img_aux1_3 = np.concatenate((imgA_31, imgB_32, imgA_33, imgB_34, imgA_35, imgB_36), axis=-1)
        img_aux1_4 = np.concatenate((imgB_41, imgA_42, imgB_43, imgA_44, imgB_45, imgA_46), axis=-1)
        img_aux1 = np.concatenate((img_aux1_1, img_aux1_2, img_aux1_3, img_aux1_4), axis=-2)

        lbl_aux1_1 = np.concatenate((lblA_11, lblB_12, lblA_13, lblB_14, lblA_15, lblB_16), axis=-1)
        lbl_aux1_2 = np.concatenate((lblB_21, lblA_22, lblB_23, lblA_24, lblB_25, lblA_26), axis=-1)
        lbl_aux1_3 = np.concatenate((lblA_31, lblB_32, lblA_33, lblB_34, lblA_35, lblB_36), axis=-1)
        lbl_aux1_4 = np.concatenate((lblB_41, lblA_42, lblB_43, lblA_44, lblB_45, lblA_46), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_1, lbl_aux1_2, lbl_aux1_3, lbl_aux1_4), axis=-2)

        msk_aux1_1 = np.concatenate((mskA_11, mskB_12, mskA_13, mskB_14, mskA_15, mskB_16), axis=-1)
        msk_aux1_2 = np.concatenate((mskB_21, mskA_22, mskB_23, mskA_24, mskB_25, mskA_26), axis=-1)
        msk_aux1_3 = np.concatenate((mskA_31, mskB_32, mskA_33, mskB_34, mskA_35, mskB_36), axis=-1)
        msk_aux1_4 = np.concatenate((mskB_41, mskA_42, mskB_43, mskA_44, mskB_45, mskA_46), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_1, msk_aux1_2, msk_aux1_3, msk_aux1_4), axis=-2)

        img_aux2_1 = np.concatenate((imgB_11, imgA_12, imgB_13, imgA_14, imgB_15, imgA_16), axis=-1)
        img_aux2_2 = np.concatenate((imgA_21, imgB_22, imgA_23, imgB_24, imgA_25, imgB_26), axis=-1)
        img_aux2_3 = np.concatenate((imgB_31, imgA_32, imgB_33, imgA_34, imgB_35, imgA_36), axis=-1)
        img_aux2_4 = np.concatenate((imgA_41, imgB_42, imgA_43, imgB_44, imgA_45, imgB_46), axis=-1)
        img_aux2 = np.concatenate((img_aux2_1, img_aux2_2, img_aux2_3, img_aux2_4), axis=-2)

        lbl_aux2_1 = np.concatenate((lblB_11, lblA_12, lblB_13, lblA_14, lblB_15, lblA_16), axis=-1)
        lbl_aux2_2 = np.concatenate((lblA_21, lblB_22, lblA_23, lblB_24, lblA_25, lblB_26), axis=-1)
        lbl_aux2_3 = np.concatenate((lblB_31, lblA_32, lblB_33, lblA_34, lblB_35, lblA_36), axis=-1)
        lbl_aux2_4 = np.concatenate((lblA_41, lblB_42, lblA_43, lblB_44, lblA_45, lblB_46), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_1, lbl_aux2_2, lbl_aux2_3, lbl_aux2_4), axis=-2)

        msk_aux2_1 = np.concatenate((mskB_11, mskA_12, mskB_13, mskA_14, mskB_15, mskA_16), axis=-1)
        msk_aux2_2 = np.concatenate((mskA_21, mskB_22, mskA_23, mskB_24, mskA_25, mskB_26), axis=-1)
        msk_aux2_3 = np.concatenate((mskB_31, mskA_32, mskB_33, mskA_34, mskB_35, mskA_36), axis=-1)
        msk_aux2_4 = np.concatenate((mskA_41, mskB_42, mskA_43, mskB_44, mskA_45, mskB_46), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_1, msk_aux2_2, msk_aux2_3, msk_aux2_4), axis=-2)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


def map_state_dict(state_dict, model_dict):

    state_dict_ = {}
    for k_, v_ in state_dict.items():
        if 'backbone.' in k_:
            if 'head.' not in k_:
                state_dict_[k_] = v_
        else:
            if 'head.' not in k_:
                state_dict_['backbone.' + k_] = v_


    model_dict_backbone = [i for i in model_dict if 'backbone.' in i]

    state_dict_match = {}
    for w in state_dict_:
        if w in model_dict_backbone:
            state_dict_match[w] = state_dict_[w]

    state_dict_ = state_dict_match

    assert len(state_dict_) == len(model_dict_backbone)

    size_embed1_proj = model_dict['backbone.patch_embed1.proj.weight'].size()
    size_embed1_proj_pretrain = state_dict_['backbone.patch_embed1.proj.weight'].size()
    print(size_embed1_proj)
    print(size_embed1_proj_pretrain)

    if size_embed1_proj[0] != size_embed1_proj_pretrain[0]:
        state_dict_['backbone.patch_embed1.proj.weight'] = torch.cat((state_dict_['backbone.patch_embed1.proj.weight'], state_dict_['backbone.patch_embed1.proj.weight']), dim=0)
        print(state_dict_['backbone.patch_embed1.proj.weight'].size())

    if size_embed1_proj[1] != size_embed1_proj_pretrain[1]:
        w_embed1_proj = state_dict_['backbone.patch_embed1.proj.weight']
        for i in range(int(size_embed1_proj[1] / size_embed1_proj_pretrain[1])):
            state_dict_['backbone.patch_embed1.proj.weight'] = torch.cat((state_dict_['backbone.patch_embed1.proj.weight'], w_embed1_proj), dim=1)
        print(state_dict_['backbone.patch_embed1.proj.weight'].size())
        state_dict_['backbone.patch_embed1.proj.weight'] = state_dict_['backbone.patch_embed1.proj.weight'][:, :size_embed1_proj[1], :, :]
        print(state_dict_['backbone.patch_embed1.proj.weight'].size())

    size_embed2_proj = model_dict['backbone.patch_embed2.proj.weight'].size()
    size_embed2_proj_pretrain = state_dict_['backbone.patch_embed2.proj.weight'].size()
    print(size_embed2_proj)
    print(size_embed2_proj_pretrain)

    if size_embed2_proj[1] != size_embed2_proj_pretrain[1]:
        state_dict_['backbone.patch_embed2.proj.weight'] = torch.cat((state_dict_['backbone.patch_embed2.proj.weight'], state_dict_['backbone.patch_embed2.proj.weight']), dim=1)
        print(state_dict_['backbone.patch_embed2.proj.weight'].size())

    for k, v in model_dict.items():
        if 'backbone.' not in k:
            state_dict_[k] = v

    assert len(state_dict_) == len(model_dict)

    # map kernel size
    cc1 = model_dict['backbone.patch_embed1.proj.weight'].size()
    cc2 = model_dict['backbone.patch_embed2.proj.weight'].size()
    cc3 = model_dict['backbone.patch_embed3.proj.weight'].size()
    cc4 = model_dict['backbone.patch_embed4.proj.weight'].size()

    if state_dict_['backbone.patch_embed1.proj.weight'].size() != cc1 or state_dict_['backbone.patch_embed2.proj.weight'].size() != cc2:

        state_dict_['backbone.patch_embed1.proj.weight'] = state_dict_['backbone.patch_embed1.proj.weight'][:, :, :cc1[2], :cc1[3]]
        state_dict_['backbone.patch_embed2.proj.weight'] = state_dict_['backbone.patch_embed2.proj.weight'][:, :, :cc2[2], :cc2[3]]
        
        state_dict_['backbone.patch_embed3.proj.weight'] = state_dict_['backbone.patch_embed3.proj.weight'][:cc3[0], :cc3[1], :cc3[2], :cc3[3]]
        state_dict_['backbone.patch_embed3.proj.bias']   = state_dict_['backbone.patch_embed3.proj.bias'][:cc3[0]]
        state_dict_['backbone.patch_embed3.norm.weight'] = state_dict_['backbone.patch_embed3.norm.weight'][:cc3[0]]
        state_dict_['backbone.patch_embed3.norm.bias']   = state_dict_['backbone.patch_embed3.norm.bias'][:cc3[0]]
        state_dict_['backbone.norm3.weight'] = state_dict_['backbone.norm3.weight'][:cc3[0]]
        state_dict_['backbone.norm3.bias']   = state_dict_['backbone.norm3.bias'][:cc3[0]]

        state_dict_['backbone.patch_embed4.proj.weight'] = state_dict_['backbone.patch_embed4.proj.weight'][:cc4[0], :cc4[1], :cc4[2], :cc4[3]]
        state_dict_['backbone.patch_embed4.proj.bias']   = state_dict_['backbone.patch_embed4.proj.bias'][:cc4[0]]
        state_dict_['backbone.patch_embed4.norm.weight'] = state_dict_['backbone.patch_embed4.norm.weight'][:cc4[0]]
        state_dict_['backbone.patch_embed4.norm.bias']   = state_dict_['backbone.patch_embed4.norm.bias'][:cc4[0]]
        state_dict_['backbone.norm4.weight'] = state_dict_['backbone.norm4.weight'][:cc4[0]]
        state_dict_['backbone.norm4.bias']   = state_dict_['backbone.norm4.bias'][:cc4[0]]

        for w in state_dict_:
            if 'backbone.block3.' in w:
                if state_dict_[w].size() != model_dict[w].size():
                    if len(state_dict_[w].size()) == 1:
                        state_dict_[w] = state_dict_[w][:model_dict[w].size()[0]]
                    elif len(state_dict_[w].size()) == 2:
                        state_dict_[w] = state_dict_[w][:model_dict[w].size()[0], :model_dict[w].size()[1]]
                    elif len(state_dict_[w].size()) == 4:
                        state_dict_[w] = state_dict_[w][:model_dict[w].size()[0], :model_dict[w].size()[1], :, :]
            if 'backbone.block4.' in w:
                if state_dict_[w].size() != model_dict[w].size():
                    if len(state_dict_[w].size()) == 1:
                        state_dict_[w] = state_dict_[w][:model_dict[w].size()[0]]
                    elif len(state_dict_[w].size()) == 2:
                        state_dict_[w] = state_dict_[w][:model_dict[w].size()[0], :model_dict[w].size()[1]]
                    elif len(state_dict_[w].size()) == 4:
                        state_dict_[w] = state_dict_[w][:model_dict[w].size()[0], :model_dict[w].size()[1], :, :]

        assert state_dict_['backbone.patch_embed1.proj.weight'].size() == cc1
        assert state_dict_['backbone.patch_embed2.proj.weight'].size() == cc2
        assert state_dict_['backbone.patch_embed3.proj.weight'].size() == cc3
        assert state_dict_['backbone.patch_embed4.proj.weight'].size() == cc4

    for w in state_dict_:
        # if 'backbone.block1.' in w:
        if state_dict_[w].size() != model_dict[w].size():
            state_dict_[w] = torch.cat((state_dict_[w], state_dict_[w]), dim=0)
            if len(state_dict_[w].size()) > 1 and state_dict_[w].size()[1] != 1:
                state_dict_[w] = torch.cat((state_dict_[w], state_dict_[w]), dim=1)

    # sr_ratios
    if 'backbone.block1.0.attn.sr.weight' in state_dict_:
        if state_dict_['backbone.block1.0.attn.sr.weight'].size() != model_dict['backbone.block1.0.attn.sr.weight'].size():
            for _ in range(int(model_dict['backbone.block1.0.attn.sr.weight'].size()[2] / state_dict_['backbone.block1.0.attn.sr.weight'].size()[2])):
                if state_dict_['backbone.block1.0.attn.sr.weight'].size() != model_dict['backbone.block1.0.attn.sr.weight'].size():
                    state_dict_['backbone.block1.0.attn.sr.weight'] = torch.cat((state_dict_['backbone.block1.0.attn.sr.weight'], state_dict_['backbone.block1.0.attn.sr.weight']), dim=-2)
                    state_dict_['backbone.block1.0.attn.sr.weight'] = torch.cat((state_dict_['backbone.block1.0.attn.sr.weight'], state_dict_['backbone.block1.0.attn.sr.weight']), dim=-1)
            assert state_dict_['backbone.block1.0.attn.sr.weight'].size() == model_dict['backbone.block1.0.attn.sr.weight'].size()
        
    if 'backbone.block1.1.attn.sr.weight' in state_dict_:
        if state_dict_['backbone.block1.1.attn.sr.weight'].size() != model_dict['backbone.block1.1.attn.sr.weight'].size():
            for _ in range(int(model_dict['backbone.block1.1.attn.sr.weight'].size()[2] / state_dict_['backbone.block1.1.attn.sr.weight'].size()[2])):
                if state_dict_['backbone.block1.1.attn.sr.weight'].size() != model_dict['backbone.block1.1.attn.sr.weight'].size():
                    state_dict_['backbone.block1.1.attn.sr.weight'] = torch.cat((state_dict_['backbone.block1.1.attn.sr.weight'], state_dict_['backbone.block1.1.attn.sr.weight']), dim=-2)
                    state_dict_['backbone.block1.1.attn.sr.weight'] = torch.cat((state_dict_['backbone.block1.1.attn.sr.weight'], state_dict_['backbone.block1.1.attn.sr.weight']), dim=-1)
            assert state_dict_['backbone.block1.1.attn.sr.weight'].size() == model_dict['backbone.block1.1.attn.sr.weight'].size()
    
    if 'backbone.block1.2.attn.sr.weight' in state_dict_:
        if state_dict_['backbone.block1.2.attn.sr.weight'].size() != model_dict['backbone.block1.2.attn.sr.weight'].size():
            for _ in range(int(model_dict['backbone.block1.2.attn.sr.weight'].size()[2] / state_dict_['backbone.block1.2.attn.sr.weight'].size()[2])):
                if state_dict_['backbone.block1.2.attn.sr.weight'].size() != model_dict['backbone.block1.2.attn.sr.weight'].size():
                    state_dict_['backbone.block1.2.attn.sr.weight'] = torch.cat((state_dict_['backbone.block1.2.attn.sr.weight'], state_dict_['backbone.block1.2.attn.sr.weight']), dim=-2)
                    state_dict_['backbone.block1.2.attn.sr.weight'] = torch.cat((state_dict_['backbone.block1.2.attn.sr.weight'], state_dict_['backbone.block1.2.attn.sr.weight']), dim=-1)
            assert state_dict_['backbone.block1.2.attn.sr.weight'].size() == model_dict['backbone.block1.2.attn.sr.weight'].size()
    
    if 'backbone.block2.0.attn.sr.weight' in state_dict_:
        if state_dict_['backbone.block2.0.attn.sr.weight'].size() != model_dict['backbone.block2.0.attn.sr.weight'].size():
            s1, s2, s3, s4 = model_dict['backbone.block2.0.attn.sr.weight'].size()
            for _ in range(int(model_dict['backbone.block2.0.attn.sr.weight'].size()[2] / state_dict_['backbone.block2.0.attn.sr.weight'].size()[2])):
                if state_dict_['backbone.block2.0.attn.sr.weight'].size() != model_dict['backbone.block2.0.attn.sr.weight'].size():
                    state_dict_['backbone.block2.0.attn.sr.weight'] = torch.cat((state_dict_['backbone.block2.0.attn.sr.weight'], state_dict_['backbone.block2.0.attn.sr.weight']), dim=-2)
                    state_dict_['backbone.block2.0.attn.sr.weight'] = torch.cat((state_dict_['backbone.block2.0.attn.sr.weight'], state_dict_['backbone.block2.0.attn.sr.weight']), dim=-1)
            state_dict_['backbone.block2.0.attn.sr.weight'] = state_dict_['backbone.block2.0.attn.sr.weight'][:s1, :s2, :s3, :s4]
            assert state_dict_['backbone.block2.0.attn.sr.weight'].size() == model_dict['backbone.block2.0.attn.sr.weight'].size()
    
    if 'backbone.block2.1.attn.sr.weight' in state_dict_:
        if state_dict_['backbone.block2.1.attn.sr.weight'].size() != model_dict['backbone.block2.1.attn.sr.weight'].size():
            s1, s2, s3, s4 = model_dict['backbone.block2.1.attn.sr.weight'].size()
            for _ in range(int(model_dict['backbone.block2.1.attn.sr.weight'].size()[2] / state_dict_['backbone.block2.1.attn.sr.weight'].size()[2])):
                if state_dict_['backbone.block2.1.attn.sr.weight'].size() != model_dict['backbone.block2.1.attn.sr.weight'].size():
                    state_dict_['backbone.block2.1.attn.sr.weight'] = torch.cat((state_dict_['backbone.block2.1.attn.sr.weight'], state_dict_['backbone.block2.1.attn.sr.weight']), dim=-2)
                    state_dict_['backbone.block2.1.attn.sr.weight'] = torch.cat((state_dict_['backbone.block2.1.attn.sr.weight'], state_dict_['backbone.block2.1.attn.sr.weight']), dim=-1)
            state_dict_['backbone.block2.1.attn.sr.weight'] = state_dict_['backbone.block2.1.attn.sr.weight'][:s1, :s2, :s3, :s4]
            assert state_dict_['backbone.block2.1.attn.sr.weight'].size() == model_dict['backbone.block2.1.attn.sr.weight'].size()
    
    if 'backbone.block2.2.attn.sr.weight' in state_dict_:
        if state_dict_['backbone.block2.2.attn.sr.weight'].size() != model_dict['backbone.block2.2.attn.sr.weight'].size():
            s1, s2, s3, s4 = model_dict['backbone.block2.2.attn.sr.weight'].size()
            for _ in range(int(model_dict['backbone.block2.2.attn.sr.weight'].size()[2] / state_dict_['backbone.block2.2.attn.sr.weight'].size()[2])):
                if state_dict_['backbone.block2.2.attn.sr.weight'].size() != model_dict['backbone.block2.2.attn.sr.weight'].size():
                    state_dict_['backbone.block2.2.attn.sr.weight'] = torch.cat((state_dict_['backbone.block2.2.attn.sr.weight'], state_dict_['backbone.block2.2.attn.sr.weight']), dim=-2)
                    state_dict_['backbone.block2.2.attn.sr.weight'] = torch.cat((state_dict_['backbone.block2.2.attn.sr.weight'], state_dict_['backbone.block2.2.attn.sr.weight']), dim=-1)
            state_dict_['backbone.block2.2.attn.sr.weight'] = state_dict_['backbone.block2.2.attn.sr.weight'][:s1, :s2, :s3, :s4]
            assert state_dict_['backbone.block2.2.attn.sr.weight'].size() == model_dict['backbone.block2.2.attn.sr.weight'].size()

    if 'backbone.block2.3.attn.sr.weight' in state_dict_:
        if state_dict_['backbone.block2.3.attn.sr.weight'].size() != model_dict['backbone.block2.3.attn.sr.weight'].size():
            s1, s2, s3, s4 = model_dict['backbone.block2.3.attn.sr.weight'].size()
            for _ in range(int(model_dict['backbone.block2.3.attn.sr.weight'].size()[2] / state_dict_['backbone.block2.3.attn.sr.weight'].size()[2])):
                if state_dict_['backbone.block2.3.attn.sr.weight'].size() != model_dict['backbone.block2.3.attn.sr.weight'].size():
                    state_dict_['backbone.block2.3.attn.sr.weight'] = torch.cat((state_dict_['backbone.block2.3.attn.sr.weight'], state_dict_['backbone.block2.3.attn.sr.weight']), dim=-2)
                    state_dict_['backbone.block2.3.attn.sr.weight'] = torch.cat((state_dict_['backbone.block2.3.attn.sr.weight'], state_dict_['backbone.block2.3.attn.sr.weight']), dim=-1)
            state_dict_['backbone.block2.3.attn.sr.weight'] = state_dict_['backbone.block2.3.attn.sr.weight'][:s1, :s2, :s3, :s4]
            assert state_dict_['backbone.block2.3.attn.sr.weight'].size() == model_dict['backbone.block2.3.attn.sr.weight'].size()
    
    if 'backbone.block3.0.attn.sr.weight' in state_dict_:
        if state_dict_['backbone.block3.0.attn.sr.weight'].size() != model_dict['backbone.block3.0.attn.sr.weight'].size():
            s1, s2, s3, s4 = model_dict['backbone.block3.0.attn.sr.weight'].size()
            for _ in range(int(model_dict['backbone.block3.0.attn.sr.weight'].size()[2] / state_dict_['backbone.block3.0.attn.sr.weight'].size()[2])):
                if state_dict_['backbone.block3.0.attn.sr.weight'].size() != model_dict['backbone.block3.0.attn.sr.weight'].size():
                    state_dict_['backbone.block3.0.attn.sr.weight'] = torch.cat((state_dict_['backbone.block3.0.attn.sr.weight'], state_dict_['backbone.block3.0.attn.sr.weight']), dim=-2)
                    state_dict_['backbone.block3.0.attn.sr.weight'] = torch.cat((state_dict_['backbone.block3.0.attn.sr.weight'], state_dict_['backbone.block3.0.attn.sr.weight']), dim=-1)
            state_dict_['backbone.block3.0.attn.sr.weight'] = state_dict_['backbone.block3.0.attn.sr.weight'][:s1, :s2, :s3, :s4]
            assert state_dict_['backbone.block3.0.attn.sr.weight'].size() == model_dict['backbone.block3.0.attn.sr.weight'].size()

    if 'backbone.block3.1.attn.sr.weight' in state_dict_:
        if state_dict_['backbone.block3.1.attn.sr.weight'].size() != model_dict['backbone.block3.1.attn.sr.weight'].size():
            s1, s2, s3, s4 = model_dict['backbone.block3.1.attn.sr.weight'].size()
            for _ in range(int(model_dict['backbone.block3.1.attn.sr.weight'].size()[2] / state_dict_['backbone.block3.1.attn.sr.weight'].size()[2])):
                if state_dict_['backbone.block3.1.attn.sr.weight'].size() != model_dict['backbone.block3.1.attn.sr.weight'].size():
                    state_dict_['backbone.block3.1.attn.sr.weight'] = torch.cat((state_dict_['backbone.block3.1.attn.sr.weight'], state_dict_['backbone.block3.1.attn.sr.weight']), dim=-2)
                    state_dict_['backbone.block3.1.attn.sr.weight'] = torch.cat((state_dict_['backbone.block3.1.attn.sr.weight'], state_dict_['backbone.block3.1.attn.sr.weight']), dim=-1)
            state_dict_['backbone.block3.1.attn.sr.weight'] = state_dict_['backbone.block3.1.attn.sr.weight'][:s1, :s2, :s3, :s4]
            assert state_dict_['backbone.block3.1.attn.sr.weight'].size() == model_dict['backbone.block3.1.attn.sr.weight'].size()
    
    if 'backbone.block3.2.attn.sr.weight' in state_dict_:
        if state_dict_['backbone.block3.2.attn.sr.weight'].size() != model_dict['backbone.block3.2.attn.sr.weight'].size():
            s1, s2, s3, s4 = model_dict['backbone.block3.2.attn.sr.weight'].size()
            for _ in range(int(model_dict['backbone.block3.2.attn.sr.weight'].size()[2] / state_dict_['backbone.block3.2.attn.sr.weight'].size()[2])):
                if state_dict_['backbone.block3.2.attn.sr.weight'].size() != model_dict['backbone.block3.2.attn.sr.weight'].size():
                    state_dict_['backbone.block3.2.attn.sr.weight'] = torch.cat((state_dict_['backbone.block3.2.attn.sr.weight'], state_dict_['backbone.block3.2.attn.sr.weight']), dim=-2)
                    state_dict_['backbone.block3.2.attn.sr.weight'] = torch.cat((state_dict_['backbone.block3.2.attn.sr.weight'], state_dict_['backbone.block3.2.attn.sr.weight']), dim=-1)
            state_dict_['backbone.block3.2.attn.sr.weight'] = state_dict_['backbone.block3.2.attn.sr.weight'][:s1, :s2, :s3, :s4]
            assert state_dict_['backbone.block3.2.attn.sr.weight'].size() == model_dict['backbone.block3.2.attn.sr.weight'].size()

    if 'backbone.block3.3.attn.sr.weight' in state_dict_:
        if state_dict_['backbone.block3.3.attn.sr.weight'].size() != model_dict['backbone.block3.3.attn.sr.weight'].size():
            s1, s2, s3, s4 = model_dict['backbone.block3.3.attn.sr.weight'].size()
            for _ in range(int(model_dict['backbone.block3.3.attn.sr.weight'].size()[2] / state_dict_['backbone.block3.3.attn.sr.weight'].size()[2])):
                if state_dict_['backbone.block3.3.attn.sr.weight'].size() != model_dict['backbone.block3.3.attn.sr.weight'].size():
                    state_dict_['backbone.block3.3.attn.sr.weight'] = torch.cat((state_dict_['backbone.block3.3.attn.sr.weight'], state_dict_['backbone.block3.3.attn.sr.weight']), dim=-2)
                    state_dict_['backbone.block3.3.attn.sr.weight'] = torch.cat((state_dict_['backbone.block3.3.attn.sr.weight'], state_dict_['backbone.block3.3.attn.sr.weight']), dim=-1)
            state_dict_['backbone.block3.3.attn.sr.weight'] = state_dict_['backbone.block3.3.attn.sr.weight'][:s1, :s2, :s3, :s4]
            assert state_dict_['backbone.block3.3.attn.sr.weight'].size() == model_dict['backbone.block3.3.attn.sr.weight'].size()

    if 'backbone.block3.4.attn.sr.weight' in state_dict_:
        if state_dict_['backbone.block3.4.attn.sr.weight'].size() != model_dict['backbone.block3.4.attn.sr.weight'].size():
            s1, s2, s3, s4 = model_dict['backbone.block3.4.attn.sr.weight'].size()
            for _ in range(int(model_dict['backbone.block3.4.attn.sr.weight'].size()[2] / state_dict_['backbone.block3.4.attn.sr.weight'].size()[2])):
                if state_dict_['backbone.block3.4.attn.sr.weight'].size() != model_dict['backbone.block3.4.attn.sr.weight'].size():
                    state_dict_['backbone.block3.4.attn.sr.weight'] = torch.cat((state_dict_['backbone.block3.4.attn.sr.weight'], state_dict_['backbone.block3.4.attn.sr.weight']), dim=-2)
                    state_dict_['backbone.block3.4.attn.sr.weight'] = torch.cat((state_dict_['backbone.block3.4.attn.sr.weight'], state_dict_['backbone.block3.4.attn.sr.weight']), dim=-1)
            state_dict_['backbone.block3.4.attn.sr.weight'] = state_dict_['backbone.block3.4.attn.sr.weight'][:s1, :s2, :s3, :s4]
            assert state_dict_['backbone.block3.4.attn.sr.weight'].size() == model_dict['backbone.block3.4.attn.sr.weight'].size()

    if 'backbone.block3.5.attn.sr.weight' in state_dict_:
        if state_dict_['backbone.block3.5.attn.sr.weight'].size() != model_dict['backbone.block3.5.attn.sr.weight'].size():
            s1, s2, s3, s4 = model_dict['backbone.block3.5.attn.sr.weight'].size()
            for _ in range(int(model_dict['backbone.block3.5.attn.sr.weight'].size()[2] / state_dict_['backbone.block3.5.attn.sr.weight'].size()[2])):
                if state_dict_['backbone.block3.5.attn.sr.weight'].size() != model_dict['backbone.block3.5.attn.sr.weight'].size():
                    state_dict_['backbone.block3.5.attn.sr.weight'] = torch.cat((state_dict_['backbone.block3.5.attn.sr.weight'], state_dict_['backbone.block3.5.attn.sr.weight']), dim=-2)
                    state_dict_['backbone.block3.5.attn.sr.weight'] = torch.cat((state_dict_['backbone.block3.5.attn.sr.weight'], state_dict_['backbone.block3.5.attn.sr.weight']), dim=-1)
            state_dict_['backbone.block3.5.attn.sr.weight'] = state_dict_['backbone.block3.5.attn.sr.weight'][:s1, :s2, :s3, :s4]
            assert state_dict_['backbone.block3.5.attn.sr.weight'].size() == model_dict['backbone.block3.5.attn.sr.weight'].size()

    return state_dict_

