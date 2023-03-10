import torch.nn as nn

from torch.nn import CrossEntropyLoss
from .dice_loss_v0 import DiceLossV0
from .dice_loss_v1 import DiceLossV1
from .ell_loss import ELLLoss
from .wce_loss import WeightedCrossEntropyLoss
from .focalloss import FocalLoss
from .eqlv2 import EQLv2
from .group_softmax import GroupSoftmax
from .group_softmax_fgbg_2 import GroupSoftmax_fgbg_2
from tools.utils.common.lovasz_losses import lovasz_softmax


class Losses(nn.Module):
    def __init__(
        self,
        loss_types: list,
        loss_weights: list,
        cls_num_pts: list = None, 
        ignore_index: int = 0,
        knn: int = 10,
        label_smoothing: float = 0.0, 
        class_weight = None,
        class_names = None,
    ):
        super().__init__()
        self.loss_types = loss_types
        self.ignore_index = ignore_index
        self.loss_weights = loss_weights

        self.ell_loss = ELLLoss(
            ignore_index=ignore_index, 
            label_smoothing=label_smoothing,
            cls_num_pts=cls_num_pts,
        )

        self.dice_loss_v0 = DiceLossV0(
            ignore_index=ignore_index,
        )

        self.dice_loss_v1 = DiceLossV1(
            ignore_index=ignore_index,
        )
        
        self.wce_loss = WeightedCrossEntropyLoss(
            cls_num_pts=cls_num_pts,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        
        self.ce_loss = CrossEntropyLoss(
            ignore_index=ignore_index,
            weight=class_weight,
            label_smoothing=label_smoothing,
        )
        
        self.lov_loss = lovasz_softmax

        self.focalloss = FocalLoss(
            gamma=0.5,
            ignore_index=ignore_index,
        )

        self.eqlv2 = EQLv2(
            ignore_index=ignore_index,
        )

        self.groupsoftmax = GroupSoftmax(
            class_names=class_names,
            ignore_index=ignore_index,
            beta=8,
            version='fine-grained',
        )

        self.groupsoftmax_fgbg = GroupSoftmax_fgbg_2(
            class_names=class_names,
            ignore_index=ignore_index,
            beta=8,
            version='bgfg',
        )
 
    def forward(self, input, target, xyz=None, offset=None):

        loss_dict = {}
        if 'WCELoss' in self.loss_types:
            loss_dict.update(
                WCELoss=self.wce_loss(input, target) * \
                    self.loss_weights[self.loss_types.index('WCELoss')])

        if 'ELLLoss' in self.loss_types:
            loss_dict.update(
                ELLLoss=self.ell_loss(input, target) * \
                    self.loss_weights[self.loss_types.index('ELLLoss')])

        if 'DiceLossV0' in self.loss_types:
            loss_dict.update(
                DiceLoss=self.dice_loss_v0(input, target) * \
                    self.loss_weights[self.loss_types.index('DiceLossV0')])
        
        if 'DiceLossV1' in self.loss_types:
            loss_dict.update(
                DiceLoss=self.dice_loss_v1(input, target) * \
                    self.loss_weights[self.loss_types.index('DiceLossV1')])

        if 'CELoss' in self.loss_types:
            loss_dict.update(
                CELoss=self.ce_loss(input, target) * \
                    self.loss_weights[self.loss_types.index('CELoss')])

        if 'LovLoss' in self.loss_types:
            loss_dict.update(
                LovLoss=self.lov_loss(input.softmax(dim=1), target, 
                                        ignore=self.ignore_index) * \
                    self.loss_weights[self.loss_types.index('LovLoss')])

        if 'FocalLoss' in self.loss_types:
            loss_dict.update(
                FocalLoss=self.focalloss(input, target) * \
                    self.loss_weights[self.loss_types.index('FocalLoss')])

        if 'EQLv2' in self.loss_types:
            loss_dict.update(
                FocalLoss=self.eqlv2(input, target) * \
                    self.loss_weights[self.loss_types.index('EQLv2')])
        
        if 'GroupSoftmax' in self.loss_types:
            loss_dict.update(
                FocalLoss=self.groupsoftmax(input, target) * \
                    self.loss_weights[self.loss_types.index('GroupSoftmax')])
        
        if 'GroupSoftmax_fgbg_2' in self.loss_types:
            loss_dict.update(
                FocalLoss=self.groupsoftmax_fgbg(input, target) * \
                    self.loss_weights[self.loss_types.index('GroupSoftmax_fgbg_2')])

        return sum(loss_dict.values())
