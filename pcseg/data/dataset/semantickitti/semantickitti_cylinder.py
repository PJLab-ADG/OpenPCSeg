'''
This file is modified from https://github.com/xinge008/Cylinder3D
'''


from collections import defaultdict
from itertools import accumulate
import numpy as np
import torch
from torch.utils import data
from torchsparse.utils.quantize import sparse_quantize

from .semantickitti import SemantickittiDataset

from tools.utils.common.seg_utils import aug_points


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


def voxelize_with_label(point_coords, point_labels, num_classes):
    voxel_coords, inds, inverse_map = sparse_quantize(
        point_coords,
        return_index=True,
        return_inverse=True,
    )
    voxel_label_counter = np.zeros([voxel_coords.shape[0], num_classes])

    for ind in range(len(inverse_map)):
        if point_labels[ind] != 67:
            voxel_label_counter[inverse_map[ind]][point_labels[ind]] += 1
    voxel_labels = np.argmax(voxel_label_counter, axis=1)

    return voxel_coords, voxel_labels, inds, inverse_map


class SemkittiCylinderDataset(data.Dataset):
    def __init__(
        self,
        data_cfgs = None,
        training = True,
        root_path = None,
        logger = None
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
        )

        self.cylinder_space_max = np.array(data_cfgs.CYLINDER_SPACE_MAX)
        self.cylinder_space_min = np.array(data_cfgs.CYLINDER_SPACE_MIN)
        self.grid_size = np.array(data_cfgs.CYLINDER_GRID_SIZE)
        
        self.if_flip = data_cfgs.get('FLIP_AUG', True)
        self.if_scale = data_cfgs.get('SCALE_AUG', True)
        self.scale_axis = data_cfgs.get('SCALE_AUG_AXIS', 'xyz')
        self.scale_range = data_cfgs.get('SCALE_AUG_RANGE', [0.95, 1.05])
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
        point = pc_data['xyzret'][:, :4] # pc_data['xyzret'][:, :5]

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

        xyz_pol = cart2polar(point[:, :3])
        xyz_pol[:, 1] = xyz_pol[:, 1] / np.pi * 180.
        max_bound = self.cylinder_space_max
        min_bound = self.cylinder_space_min
        
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        point_coord = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)
        voxel_coord, voxel_label, inds, inverse_map = voxelize_with_label(
            point_coord, point_label, len(self.class_names))
        voxel_centers = (voxel_coord.astype(np.float32) + 0.5) * intervals + min_bound
        voxel_feature = np.concatenate([voxel_centers, xyz_pol[inds], point[inds][:, :2], point[inds][:, 3:]], axis=1)
        point_voxel_centers = (point_coord.astype(np.float32) + 0.5) * intervals + min_bound

        point_feature = np.concatenate([point_voxel_centers, xyz_pol, point[:, :2], point[:, 3:]], axis=1)

        ret.update({
            'name': pc_data['path'],
            'point_feature': point_feature.astype(np.float32),
            'point_coord': point_coord.astype(np.float32),
            'point_label': point_label.astype(np.int),
            'voxel_feature': voxel_feature.astype(np.float32),
            'voxel_coord': voxel_coord.astype(np.int),
            'voxel_label': voxel_label.astype(np.int),
            'inverse_map': inverse_map.astype(np.int),
            'num_points': np.array([num_points_current_frame]),
        })

        return ret

    @staticmethod
    def collate_batch(batch_list):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        point_coord = []
        voxel_coord = []
        for i_batch in range(batch_size):
            point_coord.append(
                np.pad(data_dict['point_coord'][i_batch], ((0, 0), (0, 1)), mode='constant', constant_values=i_batch))
            voxel_coord.append(
                np.pad(data_dict['voxel_coord'][i_batch], ((0, 0), (0, 1)), mode='constant', constant_values=i_batch))

        ret['point_coord'] = torch.from_numpy(np.concatenate(point_coord)).type(torch.LongTensor)
        ret['voxel_coord'] = torch.from_numpy(np.concatenate(voxel_coord)).type(torch.LongTensor)

        ret['point_feature'] = torch.from_numpy(np.concatenate(data_dict['point_feature'])).type(torch.FloatTensor)
        ret['point_label'] = torch.from_numpy(np.concatenate(data_dict['point_label'])).type(torch.LongTensor)
        ret['voxel_feature'] = torch.from_numpy(np.concatenate(data_dict['voxel_feature'])).type(torch.FloatTensor)
        ret['voxel_label'] = torch.from_numpy(np.concatenate(data_dict['voxel_label'])).type(torch.LongTensor)
        ret['inverse_map'] = torch.from_numpy(np.concatenate(data_dict['inverse_map'])).type(torch.LongTensor)
        ret['num_points']= torch.from_numpy(np.concatenate(data_dict['num_points'])).type(torch.LongTensor)
        offset = [sample['voxel_coord'].shape[0] for sample in batch_list] 
        ret['offset'] = torch.tensor(list(accumulate(offset))).int()
        ret['name'] = data_dict['name']

        for k, v in data_dict.items():
            if k.startswith('flag'):
                ret[k] = data_dict[k]
            elif k.startswith('augmented_point_coord'):
                temp = []
                for i_batch in range(batch_size):
                    temp.append(
                        np.pad(data_dict[k][i_batch], ((0, 0), (0, 1)), mode='constant', constant_values=i_batch))
                ret[k] = torch.from_numpy(np.concatenate(temp)).type(torch.LongTensor)
            elif k.startswith('augmented_point_feature'):
                ret[k] = torch.from_numpy(np.concatenate(data_dict[k])).type(torch.FloatTensor)
            elif k.startswith('augmented_point_label') or k.startswith('augmented_inverse_map'):
                ret[k] = torch.from_numpy(np.concatenate(data_dict[k])).type(torch.LongTensor)

        return ret
    
    @staticmethod
    def collate_batch_tta(batch_list):
        batch_list = batch_list[0]
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        point_coord = []
        voxel_coord = []
        for i_batch in range(batch_size):
            point_coord.append(
                np.pad(data_dict['point_coord'][i_batch], ((0, 0), (0, 1)), mode='constant', constant_values=i_batch))
            voxel_coord.append(
                np.pad(data_dict['voxel_coord'][i_batch], ((0, 0), (0, 1)), mode='constant', constant_values=i_batch))

        ret['point_coord'] = torch.from_numpy(np.concatenate(point_coord)).type(torch.LongTensor)
        ret['voxel_coord'] = torch.from_numpy(np.concatenate(voxel_coord)).type(torch.LongTensor)

        ret['point_feature'] = torch.from_numpy(np.concatenate(data_dict['point_feature'])).type(torch.FloatTensor)
        ret['point_label'] = torch.from_numpy(np.concatenate(data_dict['point_label'])).type(torch.LongTensor)
        ret['voxel_feature'] = torch.from_numpy(np.concatenate(data_dict['voxel_feature'])).type(torch.FloatTensor)
        ret['voxel_label'] = torch.from_numpy(np.concatenate(data_dict['voxel_label'])).type(torch.LongTensor)
        ret['inverse_map'] = torch.from_numpy(np.concatenate(data_dict['inverse_map'])).type(torch.LongTensor)
        ret['num_points']= torch.from_numpy(np.concatenate(data_dict['num_points'])).type(torch.LongTensor)
        offset = [sample['voxel_coord'].shape[0] for sample in batch_list] 
        ret['offset'] = torch.tensor(list(accumulate(offset))).int()
        ret['name'] = data_dict['name']

        for k, v in data_dict.items():
            if k.startswith('flag'):
                ret[k] = data_dict[k]
            elif k.startswith('augmented_point_coord'):
                temp = []
                for i_batch in range(batch_size):
                    temp.append(
                        np.pad(data_dict[k][i_batch], ((0, 0), (0, 1)), mode='constant', constant_values=i_batch))
                ret[k] = torch.from_numpy(np.concatenate(temp)).type(torch.LongTensor)
            elif k.startswith('augmented_point_feature'):
                ret[k] = torch.from_numpy(np.concatenate(data_dict[k])).type(torch.FloatTensor)
            elif k.startswith('augmented_point_label') or k.startswith('augmented_inverse_map'):
                ret[k] = torch.from_numpy(np.concatenate(data_dict[k])).type(torch.LongTensor)

        return ret
