import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
#from mmdet.utils import get_root_logger
from functools import partial

class EQLv2(nn.Module):
    def __init__(self,
                 ignore_index=None,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=23,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 vis_grad=False):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self._pos_grad = None
        self._neg_grad = None
        self.pos_neg = None

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)
        self.ignore_index = ignore_index
        #logger = get_root_logger()
        #logger.info(f"build EQL v2, gamma: {gamma}, mu: {mu}, alpha: {alpha}")

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if cls_score.dim()>2:
            cls_score = cls_score.view(cls_score.size(0),cls_score.size(1),-1)  # N,C,H,W => N,C,H*W
            cls_score = cls_score.transpose(1,2)    # N,C,H*W => N,H*W,C
            cls_score = cls_score.contiguous().view(-1,cls_score.size(2))   # N,H*W,C => N*H*W,C
        label = label.view(-1)
        # import pdb
        # pdb.set_trace()

        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        target = expand_label(cls_score, label) # n*23

        pos_w, neg_w = self.get_weight(cls_score) #n*23

        weight = pos_w * target + neg_w * (1 - target) 

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                      reduction='none')
        if self.ignore_index != None:
            mask = ~torch.eq(label, self.ignore_index)
            mask = mask.view(-1)
            cls_loss = torch.sum(cls_loss * weight* mask.expand(self.n_c,self.n_i).transpose(0,1).float()) /(mask.float().sum()+ 1e-10)
        else:
            cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())

        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes #+ 1
        return num_channel

    def get_activation(self, cls_score, bgfgweight=False, apply_activation_func=False):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1] 
        # here use all grad
        pos_grad = torch.sum(grad * target * weight, dim=0)[1:] # ignore undefined
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[1:] # ignore undefined

        dist.all_reduce(pos_grad)
        dist.all_reduce(neg_grad)

        self._pos_grad += pos_grad
        self._neg_grad += neg_grad
        self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)

    def get_weight(self, cls_score):
        # we do not have information about pos grad and neg grad at beginning
        if self._pos_grad is None:
            self._pos_grad = cls_score.new_zeros(self.num_classes -1 ) # 22
            self._neg_grad = cls_score.new_zeros(self.num_classes -1 )
            neg_w = cls_score.new_ones((self.n_i, self.n_c))
            pos_w = cls_score.new_ones((self.n_i, self.n_c))
        else:
            # the negative weight for objectiveness is always 1
            #neg_w = self.map_func(self.pos_neg) # 
            neg_w = torch.cat([cls_score.new_ones(1), self.map_func(self.pos_neg)]) # undefined set 1
            pos_w = 1 + self.alpha * (1 - neg_w)
            neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
            pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w
