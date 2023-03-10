'''
This file is modified from https://github.com/mit-han-lab/spvnas
'''


from typing import List

import numpy as np
import torch
import torch_scatter

import torchsparse
import torchsparse.nn
import torchsparse.nn.functional
import torchsparse.nn.functional as F

from torchsparse.nn.utils import get_kernel_offsets
from torchsparse import PointTensor, SparseTensor
from torchsparse.utils.quantize import sparse_quantize


def torch_unique(x):
    unique, inverse = torch.unique(x, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inds = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return unique, inds, inverse


def voxelize_with_label(point_coords, point_labels, num_classes):
    voxel_coords, inds, inverse_map = sparse_quantize(
        point_coords,
        return_index=True,
        return_inverse=True
    )
    voxel_label_counter = np.zeros([voxel_coords.shape[0], num_classes])
    for ind in range(len(inverse_map)):
        voxel_label_counter[inverse_map[ind]][point_labels[ind]] += 1
    voxel_labels = np.argmax(voxel_label_counter, axis=1)

    return voxel_coords, voxel_labels, inds, inverse_map


def aug_points(
    xyz: np.array,
    if_flip: bool = False,
    if_scale: bool = False,
    scale_axis: str = 'xyz',
    scale_range: list = [0.9, 1.1],
    if_jitter: bool = False,
    if_rotate: bool = False,
    if_tta: bool = False,
    num_vote: int = 0,
) -> List[np.ndarray]:

    # aug (random rotate)
    if if_rotate:
        if if_tta:
            angle_vec = [0, 1, -1, 2, -2, 6, -6, 7, -7, 8]
            assert len(angle_vec) == 10
            angle_vec_new = [cnt * np.pi / 8.0 for cnt in angle_vec]
            theta = angle_vec_new[num_vote]
        else:
            theta = np.random.uniform(0, 2 * np.pi)
        rot_mat = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        xyz = np.dot(xyz, rot_mat)

    # aug (random scale)
    if if_scale:
        #scale_range = [0.95, 1.05]
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        xyz = xyz * scale_factor

    # aug (random flip)
    if if_flip:
        if if_tta:
            flip_type = num_vote
        else:
            flip_type = np.random.choice(4, 1)
        
        if flip_type == 1:
            xyz[:, 0] = -xyz[:, 0]
        elif flip_type == 2:
            xyz[:, 1] = -xyz[:, 1]
        elif flip_type == 3:
            xyz[:, :2] = -xyz[:, :2]
    
    # aug (random jitter)
    if if_jitter:
        noise_translate = np.array([
            np.random.normal(0, 0.1, 1),
            np.random.normal(0, 0.1, 1),
            np.random.normal(0, 0.1, 1),
        ]).T
        xyz += noise_translate
    
    return xyz


def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get(
            'idx_query') is None or z.additional_features['idx_query'].get(
                x.s) is None:
        pc_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F.sphash(x.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        old_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = F.sphash(x.C.to(z.F.device))
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = F.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor


# z: PointTensor(the C is float())
# return: SparseTensor
def voxelize(z, init_res=None, after_res=None, voxel_type='max'):
    pc_hash = torchsparse.nn.functional.sphash(z.C.int())
    sparse_hash, inds, idx_query = torch_unique(pc_hash)
    counts = torchsparse.nn.functional.spcount(idx_query.int(), len(sparse_hash))
    inserted_coords = z.C[inds].int()
    if voxel_type == 'max':
        inserted_feat = torch_scatter.scatter_max(z.F, idx_query, dim=0)[0]
    elif voxel_type == 'mean':
        inserted_feat = torch_scatter.scatter_mean(z.F, idx_query, dim=0)
    else:
        raise NotImplementedError("Wrong voxel_type")
    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts

    return new_tensor
