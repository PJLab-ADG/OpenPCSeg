'''
This file is modified from https://github.com/mit-han-lab/spvnas
'''


import os
import numpy as np
import torch
from torch.utils import data
from .semantickitti import SemantickittiDataset
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from itertools import accumulate
from tools.utils.common.seg_utils import aug_points
import cv2
from collections import defaultdict


class SemkittiFusionDataset(data.Dataset):
    def __init__(
        self, 
        data_cfgs=None,
        training=True, 
        root_path=None, 
        logger=None
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

    def get_range_image(self, points):
        '''
        :param points: xyzir
        :return: depth_image, reflectivity_image, px(-1~1), py(-1~1)
        '''
        INIT_HW =  [64, 2048]
        UP_HW = [64, 2048]
        # get depth of all points
        depth = np.linalg.norm(points[:, 0:3], 2, axis=1)  # xyz
        # get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        reflectivity = points[:, 3]
        # get angles of all points
        yaw = np.arctan2(scan_y, -scan_x) + (np.random.rand() - 0.5) * 2 * np.pi  # random cut
        yaw = yaw % (2 * np.pi) - np.pi  # (-pi, pi]
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = points[:, 4]  # ringID

        assert np.max(proj_y) <= (INIT_HW[0] - 1), f'{proj_y} not less than {(INIT_HW[0] - 1)}'
        # scale to image size using angular resolution
        proj_x = proj_x * (INIT_HW[1] - 1)
        px = proj_x.copy()
        py = proj_y.copy()
        proj_x = np.round(proj_x).astype(np.int32)
        proj_y = np.round(proj_y).astype(np.int32)

        # range image
        proj_range = np.zeros((INIT_HW[0], INIT_HW[1]))
        proj_range[proj_y, proj_x] = 1.0 / depth
        proj_reflectivity = np.zeros((INIT_HW[0], INIT_HW[1]))
        proj_reflectivity[proj_y, proj_x] = reflectivity
        proj_xyz = np.zeros((INIT_HW[0], INIT_HW[1], 3))
        proj_xyz[proj_y, proj_x] = points[:, :3]

        # norm
        px = 2.0 * (proj_x / (INIT_HW[1] - 1) - 0.5)  # (-1~1)
        py = 2.0 * (proj_y / (INIT_HW[0] - 1) - 0.5)  # (-1~1)
        # resize
        proj_range = cv2.resize(proj_range, (UP_HW[1], UP_HW[0]), interpolation=cv2.INTER_LINEAR)
        proj_reflectivity = cv2.resize(proj_reflectivity, (UP_HW[1], UP_HW[0]),interpolation=cv2.INTER_LINEAR)
        proj_xyz = cv2.resize(proj_xyz, (UP_HW[1], UP_HW[0]), interpolation=cv2.INTER_LINEAR)
        proj_xyz = proj_xyz.transpose(2, 0, 1)

        # normalize values to be between -10 and 10
        proj_range = 25 * (proj_range - 0.4)
        proj_reflectivity = 20 * (proj_reflectivity - 0.5)
        range_image = np.concatenate([proj_range[np.newaxis, :], proj_reflectivity[np.newaxis, :], proj_xyz]).astype(np.float32)

        range_pxpy = np.hstack([px.reshape(-1, 1), py.reshape(-1, 1)])
        return range_image, range_pxpy

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
        point = pc_data['xyzret'].astype(np.float32)

        num_points_current_frame =point.shape[0]
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
        _, inds, inverse_map = sparse_quantize(pc_,
                                               return_index=True,
                                               return_inverse=True)
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
        range_image, range_pxpy = self.get_range_image(lidar.F)
        ret = {
            'name': pc_data['path'],
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'num_points': np.array([num_points_current_frame]), # for multi frames
            'range_image' : range_image,
            'range_pxpy' : range_pxpy,
        }

        return ret

    @staticmethod
    def collate_batch(inputs):
        offset = [sample['lidar'].C.shape[0] for sample in inputs] # for point transformer
        offsets = {}
        data_dict = defaultdict(list)
        for i, cur_sample in enumerate(inputs):
            for key in list(cur_sample.keys()):
                if key == 'range_pxpy':
                   data_dict[key].append(cur_sample['range_pxpy'])
                   inputs[i].pop('range_pxpy')
        ret = sparse_collate_fn(inputs)
        ret.update(dict(
            offset=torch.tensor(list(accumulate(offset))).int()
        ))
        for key,val in data_dict.items():
            if key in ['range_pxpy']:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                coors_b = torch.from_numpy(np.concatenate(coors, axis=0)).float()
        ret.update(dict(range_pxpy=coors_b))
        return ret

    @staticmethod
    def collate_batch_tta(inputs):
        inputs = inputs[0]
        offset = [sample['lidar'].C.shape[0] for sample in inputs] # for point transformer
        offsets = {}
        data_dict = defaultdict(list)
        for i, cur_sample in enumerate(inputs):
            for key in list(cur_sample.keys()):
                if key == 'range_pxpy':
                   data_dict[key].append(cur_sample['range_pxpy'])
                   inputs[i].pop('range_pxpy')
        ret = sparse_collate_fn(inputs)
        ret.update(dict(
            offset=torch.tensor(list(accumulate(offset))).int()
        ))
        for key,val in data_dict.items():
            if key in ['range_pxpy']:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                coors_b = torch.from_numpy(np.concatenate(coors, axis=0)).float()
        ret.update(dict(range_pxpy=coors_b))
        return ret
