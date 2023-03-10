'''
This file is modified from https://github.com/mit-han-lab/spvnas
'''


import numpy as np
import torch
from torch.utils import data
from .semantickitti import SemantickittiDataset
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from itertools import accumulate
from tools.utils.common.seg_utils import aug_points


class SemkittiVoxelDataset(data.Dataset):
    def __init__(
        self,
        data_cfgs=None,
        training=True,
        root_path=None,
        logger=None,
    ):
        super().__init__()
        self.data_cfgs = data_cfgs
        self.training = training
        self.class_names = [
            "unlabeled",  # ignored
            "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist",  # dynamic
            "road", "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign"  # static
        ]
        self.root_path = root_path if root_path is not None else self.data_cfgs.DATA_PATH
        self.logger = logger

        self.point_cloud_dataset = SemantickittiDataset(
            data_cfgs=data_cfgs,
            training=training,
            class_names=self.class_names,
            root_path=self.root_path,
            logger=logger,
            if_scribble=True if self.data_cfgs.DATASET == 'scribblekitti' else False,
        )

        self.voxel_size = data_cfgs.VOXEL_SIZE
        self.num_points = data_cfgs.NUM_POINTS

        self.if_flip = data_cfgs.get('FLIP_AUG', True)
        self.if_scale = data_cfgs.get('SCALE_AUG', True)
        self.scale_axis = data_cfgs.get('SCALE_AUG_AXIS', 'xyz')
        self.scale_range = data_cfgs.get('SCALE_AUG_RANGE', [0.9, 1.1])
        self.if_jitter = data_cfgs.get('TRANSFORM_AUG', True)
        self.if_rotate = data_cfgs.get('ROTATE_AUG', True)
        
        self.if_tta = self.data_cfgs.get('TTA', False)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)


    def __getitem__(self, index):
        if self.if_tta:
            data_total = []
            voting = 10  
            for idx in range(voting):
                data_single = self.get_single_sample(index, idx)
                data_total.append(data_single)
            return data_total
        else:
            data = self.get_single_sample(index)
            return data

    def get_single_sample(self, index, voting_idx=0):
        'Generates one sample of data'
        pc_data = self.point_cloud_dataset[index]
        point_label = pc_data['labels'].reshape(-1)
        point = pc_data['xyzret'][:,:4].astype(np.float32)

        num_points_current_frame = point.shape[0]
        ret = {}
        if self.training:
            point[:, 0:3] = aug_points(
                xyz=point[:, :3],
                if_flip=self.if_flip,
                if_scale=self.if_scale,
                scale_axis=self.scale_axis,
                scale_range=self.scale_range,
                if_jitter=self.if_jitter,
                if_rotate=self.if_rotate,
                if_tta=self.if_tta,
            )

        elif self.if_tta:
            self.if_flip = False
            self.if_scale = True
            self.scale_aug_range = [0.95, 1.05]
            self.if_jitter = False
            self.if_rotate = True
            point[:, 0:3] = aug_points(
                xyz=point[:, :3],
                if_flip=self.if_flip,
                if_scale=self.if_scale,
                scale_axis=self.scale_axis,
                scale_range=self.scale_range,
                if_jitter=self.if_jitter,
                if_rotate=self.if_rotate,
                if_tta=True,
                num_vote=voting_idx,
        )

        pc_ = np.round(point[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)
        feat_ = point
        _, inds, inverse_map = sparse_quantize(
            pc_,
            return_index=True,
            return_inverse=True,
        
        )
        if self.training and len(inds) > self.num_points:  # NOTE: num_points must always bigger than self.num_points
            raise RuntimeError('droping point')
            inds = np.random.choice(inds, self.num_points, replace=False)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = point_label[inds]
        
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(point_label, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)
        ret = {
            'name': pc_data['path'],
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'num_points': np.array([num_points_current_frame]), # for multi frames
        }

        return ret

    @staticmethod
    def collate_batch(inputs):
        offset = [sample['lidar'].C.shape[0] for sample in inputs] # for point transformer
        offsets = {}
        ret = sparse_collate_fn(inputs)
        ret.update(dict(
            offset=torch.tensor(list(accumulate(offset))).int()
        ))
        return ret
    
    @staticmethod
    def collate_batch_tta(inputs):
        inputs = inputs[0]
        offset = [sample['lidar'].C.shape[0] for sample in inputs] # for point transformer
        offsets = {}

        ret = sparse_collate_fn(inputs)
        ret.update(dict(
            offset=torch.tensor(list(accumulate(offset))).int()
        ))
        return ret

