import bisect
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .focalloss import FocalLoss
# from .utils import LabelSmoothSoftmaxCE


#class_names = ['UNDEFINED', 'CAR', 'TRUCK', 'BUS', 'OTHER_VEHICLE', 'MOTORCYCLIST', 'BICYCLIST', 'PEDESTRIAN', 'SIGN',
        # 'TRAFFIC_LIGHT', 'POLE', 'CONSTRUCTION_CONE', 'BICYCLE', 'MOTORCYCLE', 'BUILDING', 'VEGETATION',
        # 'TREE_TRUNK', 'CURB', 'ROAD', 'LANE_MARKER', 'OTHER_GROUND', 'WALKABLE', 'SIDEWALK']

class GroupSoftmax_fgbg_2(nn.Module):
    """
    This uses a different encoding from v1.
    v1: [cls1, cls2, ..., other1_for_group0, other_for_group_1, bg, bg_others]
    this: [group0_others, group0_cls0, ..., group1_others, group1_cls0, ...]
    """
    def __init__(self,
                 ignore_index=-1,
                 num_per_class=None,
                 class_names=None, 
                 focalloss=False,
                 focal_parameter=2,
                 use_sigmoid=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 beta=8,
                 bin_split=(10, 100, 1000),
                 version="bgfg"):
        super(GroupSoftmax_fgbg_2, self).__init__()
        if class_names == None: 
            class_names = ['UNDEFINED', 'CAR', 'TRUCK', 'BUS', 'OTHER_VEHICLE', 'MOTORCYCLIST', 'BICYCLIST', 'PEDESTRIAN', 'SIGN', 'TRAFFIC_LIGHT', 'POLE', 'CONSTRUCTION_CONE', 'BICYCLE', 'MOTORCYCLE', 'BUILDING', 'VEGETATION','TREE_TRUNK', 'CURB', 'ROAD', 'LANE_MARKER', 'OTHER_GROUND', 'WALKABLE', 'SIDEWALK']
        self.use_sigmoid = False
        self.group = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.focalloss = focalloss
        self.class_weight = class_weight
        self.beta = beta
        self.bin_split = bin_split
        self.version = version
        # self.cat_instance_count = num_per_class
        self.class_names = class_names
        self.num_classes = len(self.class_names)  
        assert not self.use_sigmoid
        self.ignore_index = ignore_index
        self._get_group_bgfg() 
        self._prepare_for_label_remapping()
        if self.focalloss:
            self.FocalLoss = FocalLoss(gamma=focal_parameter, ignore_index=0)

       
    def _get_group(self, version="bg_fg"): # fg_bg, fine-grained
        self.num_group = 6 # 1 + 5
        self.group_cls = [[] for _ in range(self.num_group)] 
        self.group_cls_ids = [[] for _ in range(self.num_group)]

        # all classes = (4 + 5 + 4 + 3 + 6) + 5(group_num) + 2(fg bg) + 1(undefined)
        self.group_cls[0] = ['CAR', 'TRUCK', 'BUS', 'OTHER_VEHICLE'] # 4 fg
        self.group_cls[1] = ['MOTORCYCLIST', 'BICYCLIST', 'PEDESTRIAN', 'BICYCLE', 'MOTORCYCLE'] #5 fg
        self.group_cls[2] = ['SIGN', 'TRAFFIC_LIGHT', 'POLE', 'CONSTRUCTION_CONE'] #4 fg
        self.group_cls[3] = ['BUILDING', 'VEGETATION', 'TREE_TRUNK'] #3 bg
        self.group_cls[4] = ['CURB', 'ROAD', 'LANE_MARKER', 'OTHER_GROUND', 'WALKABLE', 'SIDEWALK'] #6 bg
        self.group_cls[5] = ['fg', 'bg']

        for i in range(len(self.group_cls)-1):
            #group_cls_ids[i] = list(map(lambda x: self.class_names.index(x), group_cls[0]))
            for cls in self.group_cls[i]:
                 self.group_cls_ids[i].append(self.class_names.index(cls))
        self.n_cls_group = list(map(lambda x: len(x), self.group_cls))

        # get fg bg group
        self.fg_bg_cls_ids = [[] for _ in range(len(self.group_cls[-1]))]
        self.fg_bg_cls = [[] for _ in range(len(self.group_cls[-1]))] 
        self.fg_bg_cls[0] = self.group_cls[0] + self.group_cls[1] + self.group_cls[2] 
        self.fg_bg_cls[1] = self.group_cls[3] + self.group_cls[4]

        for i in range(len(self.fg_bg_cls)):
            #group_cls_ids[i] = list(map(lambda x: self.class_names.index(x), group_cls[0]))
            for cls in self.fg_bg_cls[i]:
                 self.fg_bg_cls_ids[i].append(self.class_names.index(cls))

    def _get_group_bgfg(self): # fg_bg, fine-grained
        self.num_group = 2 # 1 + 5
        self.group_cls = [[] for _ in range(self.num_group)] 
        self.group_cls_ids = [[] for _ in range(self.num_group)]

        # all classes = (4 + 5 + 4 + 3 + 6) + 5(group_num) + 2(fg bg) + 1(undefined)
        self.group_cls[0] =  self.class_names
        self.group_cls[1] = ['undefined', 'fg', 'bg']

        for i in range(len(self.group_cls)-1):
            #group_cls_ids[i] = list(map(lambda x: self.class_names.index(x), group_cls[0]))
            for cls in self.group_cls[i]:
                 self.group_cls_ids[i].append(self.class_names.index(cls))
        self.n_cls_group = list(map(lambda x: len(x), self.group_cls))

        # get fg bg group
        self.fg_bg_cls_ids = [[] for _ in range(len(self.group_cls[-1]))]
        self.fg_bg_cls = [[] for _ in range(len(self.group_cls[-1]))] 
        self.fg_bg_cls[1] =  self.class_names[1:14]
        self.fg_bg_cls[2] = self.class_names[14:]
        for i in range(len(self.fg_bg_cls)):
            #group_cls_ids[i] = list(map(lambda x: self.class_names.index(x), group_cls[0]))
            for cls in self.fg_bg_cls[i]:
                 self.fg_bg_cls_ids[i].append(self.class_names.index(cls))

    def _prepare_for_label_remapping(self):
        # obtain label map for each group
        group_label_maps = []
        for group_id in range(self.num_group): # add a group for bg_fg
            label_map = [0 for _ in range(self.num_classes)] # 23 classes
            label_map[self.ignore_index] = 0 # undefined assign -1
            group_label_maps.append(label_map)

        for i in range(1, self.num_classes):
            group_label_maps[0][i] = i
        
        # for bg_fg class
        for index in range(len(self.fg_bg_cls_ids)):
            for cls in self.fg_bg_cls_ids[index]:
                group_label_maps[self.num_group-1][cls] = index  # the last group is for bg_fg

        self.group_label_maps = torch.LongTensor(group_label_maps)

    def _get_group_pred(self, cls_score, apply_activation_func=False):
        group_pred = []
        start = 0
        for group_id, n_cls in enumerate(self.n_cls_group):
            num_logits = n_cls
            pred = cls_score.narrow(1, start, num_logits)
            start = start + num_logits
            if apply_activation_func:
                pred = F.softmax(pred, dim=1) #log_softmaxn
            group_pred.append(pred)
        assert start == self.num_classes + 3
        return group_pred

    def _remap_labels(self, labels):
        new_labels = []
        for group_id in range(len(self.group_label_maps)):
            mapping = self.group_label_maps[group_id]
            new_bin_label = mapping[labels]
            new_bin_label = torch.LongTensor(new_bin_label).to(labels.device)
            new_labels.append(new_bin_label)
        return new_labels

    def _sample_others(self, label):

        # only works for non bg-fg bins

        fg = torch.where(label > 0, torch.ones_like(label), torch.zeros_like(label))
        fg_idx = fg.nonzero(as_tuple=True)
        fg_num = fg_idx[0].shape[0]
        if fg_num == 0:
            return torch.zeros_like(label)

        bg = torch.where(label == 0, torch.ones_like(label), torch.zeros_like(label))
        bg_idx = bg.nonzero(as_tuple=True)
        bg_num = bg_idx[0].shape[0]

        bg_sample_num = int(fg_num * self.beta)

        if bg_sample_num >= bg_num:
            sample_idx = bg_idx
        else:
            sample_index = torch.randperm(bg_idx[0].size(0))[:bg_sample_num].to(label)
            sample_idx = []
            for idx in bg_idx: sample_idx.append(idx[sample_index])
        fg[sample_idx] = 1
        weight = fg

        return weight.to(label.device)

    def forward(self,
                cls_score,
                label,
                focal_loss=False,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        group_preds = self._get_group_pred(cls_score, apply_activation_func=False)
        new_labels = self._remap_labels(label)
        
        cls_loss = []
        for group_id in range(self.num_group):
            pred_in_group = group_preds[group_id]
            label_in_group = new_labels[group_id]
            if self.focalloss and self.is_background_group(group_id):
                loss_in_group = self.FocalLoss(pred_in_group,
                                            label_in_group)
            else:
                loss_in_group = F.cross_entropy(pred_in_group,
                                            label_in_group,
                                            ignore_index=0) #  reduction='none',
            cls_loss.append(loss_in_group)
        cls_loss = sum(cls_loss)    
        return cls_loss * self.loss_weight

    def get_activation(self, cls_score, bgfgweight=False, apply_activation_func=False):
        sizes = list(cls_score.size())
        sizes[1] = self.num_classes
        sizes = tuple(sizes)
        group_activation = self._get_group_pred(cls_score, apply_activation_func=apply_activation_func)
        bg_score = group_activation[-1]
        activation = cls_score.new_zeros(sizes) #len(self.group_ids)))
        for group_id, cls_ids in enumerate(self.group_cls_ids[:-1]):
            activation[:, cls_ids] = group_activation[group_id]

        if bgfgweight:
            for group_id, cls_ids in enumerate(self.fg_bg_cls_ids):
                activation[:, cls_ids] *= bg_score[:, [group_id]]
        
        return activation


    def get_channel_num(self):
        num_channel = self.num_classes + 3 # 25 for waymo
        return num_channel

    def is_background_group(self, group_id):
        return group_id == self.num_group - 1