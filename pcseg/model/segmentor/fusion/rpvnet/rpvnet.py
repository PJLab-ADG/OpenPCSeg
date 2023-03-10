'''
Implementation for RPVNet: A Deep and Efficient Range-Point-Voxel Fusion Network for LiDAR Point Cloud Segmentation

Also inspired by:
    [1] https://github.com/mit-han-lab/spvnas
    [2] https://github.com/TiagoCortinhal/SalsaNext
'''


import torch
from torch import nn
import torch.nn.functional as F

import torchsparse
import torchsparse.nn as spnn
from torchsparse.nn.utils import fapply
from torchsparse import PointTensor
from torchsparse import SparseTensor

from pcseg.model.segmentor.base_segmentors import BaseSegmentor
from .utils import initial_voxelize, point_to_voxel, voxel_to_point

from pcseg.loss import Losses


import range_utils.nn.functional as rnf


__all__ = ['RPVNet', 'SalsaNext']


def resample_grid_stacked(predictions, pxpy, grid_sample_mode='bilinear'):
    '''
    :param predictions: NCHW
    :param pxpy: Nx3 3: batch_idx, px, py
    :return:
    '''
    resampled = []
    for cnt, one_batch in enumerate(predictions):
        bs_mask = (pxpy[:, 0] == cnt)
        one_batch = one_batch.unsqueeze(0)
        one_pxpy = pxpy[bs_mask][:, 1:].unsqueeze(0).unsqueeze(0)
        
        one_resampled = F.grid_sample(one_batch, one_pxpy, mode=grid_sample_mode)
        one_resampled = one_resampled.squeeze().transpose(0, 1)  # NxC
        resampled.append(one_resampled)
    return torch.cat(resampled, dim=0)

def range_to_point(feature_map, pxpy, grid_sample_mode='bilinear'):
    """convert 2d range feature map to points feature"""
    return resample_grid_stacked(feature_map, pxpy, grid_sample_mode)


def resample_grid_stacked_forfusion(predictions, pxpy, cnt, grid_sample_mode='bilinear'):
    '''
    :param predictions: NCHW
    :param pxpy: Nx3 3: batch_idx, px, py
    :return:
      '''
    resampled = []
    one_pxpy = pxpy[:, 1:].unsqueeze(0).unsqueeze(0)
    one_resampled = F.grid_sample(predictions, one_pxpy, mode=grid_sample_mode)
    one_resampled = one_resampled.squeeze().transpose(0, 1)  # NxC
    resampled.append(one_resampled)
    return torch.cat(resampled, dim=0)


def range_to_point_forfusion(feature_map, pxpy,cnt, grid_sample_mode='bilinear'):
    """convert 2d range feature map to points feature"""
    return resample_grid_stacked_forfusion(feature_map, pxpy,cnt, grid_sample_mode)


def point_to_range(pf, pxpy, b, h, w):
    """
    args:
        pf: N x C point features
        pxpy: N x 3 batch_id,px,py
            px range: -1~1
            py range: -1~1
        b: batch_size
        h: output feature map height
        w: output feature map width
    return:
        feature map: B C H W
    """
    # imitate from pytorch grid_sample C ops [-1,1] => [0,h-1] or [0, w-1]
    batch_id = pxpy[:,:1]
    int_pxpy = torch.cat([batch_id,(pxpy[:,1:] + 1)/2 * torch.Tensor([w-1,h-1]).cuda()],1).int() 
    cm = rnf.map_count(int_pxpy, b, h, w)
    fm = rnf.denselize(pf, cm, int_pxpy)
    return fm


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)
        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)
        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)
        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True, return_skip=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.return_skip = return_skip
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)
        resA = shortcut + resA1

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)
            if self.return_skip:
                return resB, resA
            else:
                return resB
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):

    def __init__(self, in_filters, out_filters, dropout_rate=0.2, drop_out=True, mid_filters=None):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.mid_filters = mid_filters if mid_filters else in_filters // 4 + 2 * out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(self.mid_filters, out_filters, (3, 3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA, skip), dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = upE1
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


class SalsaNext(nn.Module):
    def __init__(self, model_cfgs, input_channels=5):
        super(SalsaNext, self).__init__()
        self.in_channels = input_channels
        first_r, r = 1, 1
        self.cr = model_cfgs.get('cr', 1.75)
        self.cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]

        self.cs = [int(self.cr * x) for x in self.cs]
        # input_channels = 64
        self.stem = nn.Sequential(
            ResContextBlock(input_channels, self.cs[0]),
            ResContextBlock(self.cs[0], self.cs[0]),
            ResContextBlock(self.cs[0], self.cs[0])
        )
        self.stage1 = ResBlock(self.cs[0] * first_r, self.cs[1], 0.2, pooling=True, drop_out=False)
        self.stage2 = ResBlock(self.cs[1], self.cs[2], 0.2, pooling=True)
        self.stage3 = ResBlock(self.cs[2], self.cs[3], 0.2, pooling=True)
        self.stage4 = ResBlock(self.cs[3], self.cs[4], 0.2, pooling=True)

        self.mid_stage = ResBlock(self.cs[4], self.cs[4], 0.2, pooling=False)

        self.up1 = UpBlock(self.cs[4] * r, self.cs[5], 0.2, mid_filters=self.cs[4] * r // 4 + self.cs[4])
        self.up2 = UpBlock(self.cs[5], self.cs[6], 0.2, mid_filters=self.cs[5] // 4 + self.cs[3])
        self.up3 = UpBlock(self.cs[6] * r, self.cs[7], 0.2, mid_filters=self.cs[6] * r // 4 + self.cs[2])
        self.up4 = UpBlock(self.cs[7], self.cs[8], 0.2, drop_out=False, mid_filters=self.cs[7] // 4 + self.cs[1])

        self.num_point_features = self.cs[8]

    def forward(self, batch_dict):
        range_pxpy = batch_dict['range_pxpy']
        range_image = batch_dict['range_image']

        x = self.stem(range_image)
        x1, s1 = self.stage1(x)
        x2, s2 = self.stage2(x1)
        x3, s3 = self.stage3(x2)
        x4, s4 = self.stage4(x3)

        x5 = self.mid_stage(x4)

        u1 = self.up1(x5, s4)
        u2 = self.up2(u1, s3)
        u3 = self.up3(u2, s2)
        u4 = self.up4(u3, s1)

        z = resample_grid_stacked(u4, range_pxpy)
        batch_dict['point_features'] = z
        return batch_dict


class SyncBatchNorm(nn.SyncBatchNorm):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)
    
class BatchNorm(nn.BatchNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)

class BasicConvolutionBlock(nn.Module):
    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                stride=stride,
                transposed=True,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(
                outc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=1,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
        )
        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    inc, outc * self.expansion,
                    kernel_size=1,
                    dilation=1,
                    stride=stride,
                ),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=1,
                bias=False,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.Conv3d(
                outc, outc,
                kernel_size=ks,
                stride=stride,
                bias=False,
                dilation=dilation,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.Conv3d(
                outc, outc * self.expansion,
                kernel_size=1,
                bias=False,
            ),
            SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
        )
        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    inc, outc * self.expansion,
                    kernel_size=1,
                    dilation=1,
                    stride=stride,
                ),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class RPVNet(BaseSegmentor):

    def _make_layer(self, block, out_channels, num_block, stride=1, if_dist=False):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, if_dist=if_dist))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels, if_dist=if_dist))
        return layers

    def __init__(self, model_cfgs, num_class):
        super().__init__(model_cfgs, num_class)
        self.name = "rpvnet"
        self.in_feature_dim = model_cfgs.IN_FEATURE_DIM

        # Default is MinkUNet50
        self.num_layer = model_cfgs.get('NUM_LAYER', [2, 3, 4, 6, 2, 2, 2, 2])
        self.block = {
            'ResBlock': ResidualBlock,
            'Bottleneck': Bottleneck,
        }[model_cfgs.get('BLOCK', 'Bottleneck')]

        cr = model_cfgs.get('cr', 1.75)
        cs = model_cfgs.get('PLANES', [32, 32, 64, 128, 256, 256, 128, 96, 96])
        cs = [int(cr * x) for x in cs]

        self.pres = model_cfgs.get('pres', 0.05)
        self.vres = model_cfgs.get('vres', 0.05)
        if_dist = model_cfgs.IF_DIST

        self.stem = nn.Sequential(
            spnn.Conv3d(
                self.in_feature_dim, cs[0],
                kernel_size=3,
                stride=1,
            ),
            SyncBatchNorm(cs[0]) if if_dist else BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(
                cs[0], cs[0],
                kernel_size=3,
                stride=1,
            ),
            SyncBatchNorm(cs[0]) if if_dist else BatchNorm(cs[0]),
            spnn.ReLU(True),
        )


        self.in_channels = cs[0]

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=if_dist,
            ),
            *self._make_layer(self.block, cs[1], self.num_layer[0] ,if_dist=if_dist),
        )
        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=if_dist,
            ),
            *self._make_layer(self.block, cs[2], self.num_layer[1], if_dist=if_dist),
        )
        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=if_dist,
            ),
            *self._make_layer(self.block, cs[3], self.num_layer[2], if_dist=if_dist),
        )
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=if_dist,
            ),
            *self._make_layer(self.block, cs[4], self.num_layer[3], if_dist=if_dist),
        )

        self.up1 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[5],
                ks=2,
                stride=2,
                if_dist=if_dist,
            )
        ]
        self.in_channels = cs[5] + cs[3] * self.block.expansion
        self.up1.append(
            nn.Sequential(*self._make_layer(self.block, cs[5], self.num_layer[4], if_dist=if_dist))
        )
        self.up1 = nn.ModuleList(self.up1)

        self.up2 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[6],
                ks=2,
                stride=2,
                if_dist=if_dist,
            )
        ]
        self.in_channels = cs[6] + cs[2] * self.block.expansion
        self.up2.append(
            nn.Sequential(*self._make_layer(self.block, cs[6], self.num_layer[5], if_dist=if_dist))
        )
        self.up2 = nn.ModuleList(self.up2)

        self.up3 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[7],
                ks=2,
                stride=2,
                if_dist=if_dist,
            )
        ]
        self.in_channels = cs[7] + cs[1] * self.block.expansion
        self.up3.append(
            nn.Sequential(*self._make_layer(self.block, cs[7], self.num_layer[6], if_dist=if_dist))
        )
        self.up3 = nn.ModuleList(self.up3)

        self.up4 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[8],
                ks=2,
                stride=2,
                if_dist=if_dist,
            )
        ]
        self.in_channels = cs[8] + cs[0]
        self.up4.append(
            nn.Sequential(*self._make_layer(self.block, cs[8], self.num_layer[7], if_dist=if_dist))
        )
        self.up4 = nn.ModuleList(self.up4)

        self.multi_scale = self.model_cfgs.get('MULTI_SCALE', 'concat')
        if self.multi_scale == 'concat':
            self.classifier = nn.Sequential(nn.Linear((cs[4] + cs[6] + cs[8]) * self.block.expansion, self.num_class))

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
               nn.Linear(self.in_feature_dim, cs[0]),
               nn.SyncBatchNorm(cs[0]) if if_dist else BatchNorm(cs[0]),
               nn.ReLU(True),
           ),
            nn.Sequential(
                nn.Linear(cs[0], cs[4] * self.block.expansion),
                nn.SyncBatchNorm(cs[4] * self.block.expansion) if if_dist else BatchNorm(cs[4] * self.block.expansion),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[4] * self.block.expansion, cs[6] * self.block.expansion),
                nn.SyncBatchNorm(cs[6] * self.block.expansion) if if_dist else BatchNorm(cs[6] * self.block.expansion),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[6] * self.block.expansion, cs[8] * self.block.expansion),
                nn.SyncBatchNorm(cs[8] * self.block.expansion) if if_dist else BatchNorm(cs[8] * self.block.expansion),
                nn.ReLU(True),
            )
        ])

        self.range_branch = SalsaNext(model_cfgs=model_cfgs, input_channels= 5)
        if if_dist:
            self.range_branch = nn.SyncBatchNorm.convert_sync_batchnorm(self.range_branch)
        self.grid_sample_mode = 'bilinear'

        self.weight_initialization()

        dropout_p = model_cfgs.get('DROPOUT_P', 0.3)
        self.dropout = nn.Dropout(dropout_p, True)
        
        label_smoothing = model_cfgs.get('LABEL_SMOOTHING', 0.0)

        # loss
        default_loss_config = {
            # 'LOSS_TYPES': ['GeoLoss', 'LovLoss'],
            'LOSS_TYPES': ['CELoss', 'LovLoss'],
            # 'LOSS_WEIGHTS': [1.5, 1.0],
            'LOSS_WEIGHTS': [1.0, 1.0],
            'KNN': 10,
            }
        loss_config = self.model_cfgs.get('LOSS_CONFIG', default_loss_config)

        loss_types = loss_config.get('LOSS_TYPES', default_loss_config['LOSS_TYPES'])
        loss_weights = loss_config.get('LOSS_WEIGHTS', default_loss_config['LOSS_WEIGHTS'])
        assert len(loss_types) == len(loss_weights)
        k_nearest_neighbors = loss_config.get('KNN', default_loss_config['KNN'])
        self.criterion_losses = Losses(loss_types=loss_types, 
                                        loss_weights=loss_weights, 
                                        ignore_index=model_cfgs.IGNORE_LABEL,
                                        knn=k_nearest_neighbors,
                                        label_smoothing=label_smoothing) 

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict, return_logit=False, return_tta=False): 
        x = batch_dict['lidar']
        range_image = batch_dict['range_image']
        range_pxpy = batch_dict['range_pxpy']
        batch_size = range_image.size(0)
        h = range_image.size(2)
        w = range_image.size(3)

        x.F = x.F[:, :self.in_feature_dim]
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())
        x0 = initial_voxelize(z, self.pres, self.vres)
        
        r_x0 = self.range_branch.stem(range_image)
        x0 = self.stem(x0)

        z0 = voxel_to_point(x0, z, nearest=False)
        r_z0 = range_to_point(r_x0, range_pxpy, self.grid_sample_mode)
        z0_point = self.point_transforms[0](z.F)
        z0.F = z0.F + r_z0 + z0_point 

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        r_x1 = point_to_range(z0.F, range_pxpy, batch_size, h, w)
        r_x1, r_s1 = self.range_branch.stage1(r_x1)
        r_x2, r_s2 = self.range_branch.stage2(r_x1)
        r_x3, r_s3 = self.range_branch.stage3(r_x2)
        r_x4, r_s4 = self.range_branch.stage4(r_x3)
        r_x4 = self.range_branch.mid_stage(r_x4)
        
        z1 = voxel_to_point(x4, z0)
        r_z1 = range_to_point(r_x4, range_pxpy, self.grid_sample_mode)
        z1_point =self.point_transforms[1](z0.F)
        z1.F = z1.F + r_z1 + z1_point

        y1 = point_to_voxel(x4, z1)
        r_y1 = point_to_range(z1.F, range_pxpy, batch_size, r_x4.size(2), r_x4.size(3))
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)
        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        r_y1 = self.range_branch.up1(r_y1, r_s4)
        r_y2 = self.range_branch.up2(r_y1, r_s3)


        z2 = voxel_to_point(y2, z1)
        r_z2 = range_to_point(r_y2, range_pxpy, self.grid_sample_mode)
        z2_point = self.point_transforms[2](z1.F)
        z2.F = z2.F + r_z2 + z2_point

        y3 = point_to_voxel(y2, z2)
        r_y3 = point_to_range(z2.F, range_pxpy, batch_size, r_y2.size(2), r_y2.size(3))
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)
        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        r_y3 = self.range_branch.up3(r_y3, r_s2)
        r_y4 = self.range_branch.up4(r_y3, r_s1)


        z3 = voxel_to_point(y4, z2)
        r_z3 = range_to_point(r_y4, range_pxpy, self.grid_sample_mode)
        z3_point = self.point_transforms[3](z2.F)
        z3.F = z3.F + r_z3 + z3_point

        if self.multi_scale == 'concat':
            out = self.classifier(torch.cat([z1.F, z2.F, z3.F], dim=1))
        elif self.multi_scale == 'sum':
            out = self.classifier(self.l1(z1.F) + self.l2(z2.F) + z3.F)
        elif self.multi_scale == 'se':
            attn = torch.cat([z1.F, z2.F, z3.F], dim=1)
            attn = self.pool(attn.permute(1, 0)).permute(1, 0)
            attn = self.attn(attn)
            out = self.classifier(torch.cat([z1.F, z2.F, z3.F], dim=1) * attn)
        else:
            out = self.classifier(z3.F)

        if self.training:
            target = batch_dict['targets'].F.long().cuda(non_blocking=True)

            coords_xyz = batch_dict['lidar'].C[:, :3].float()
            offset = batch_dict['offset']
            loss = self.criterion_losses(out, target, xyz=coords_xyz, offset=offset)
            
            ret_dict = {'loss': loss}
            disp_dict = {'loss': loss.item()}
            tb_dict = {'loss': loss.item()}
            return ret_dict, tb_dict, disp_dict
        else:
            invs = batch_dict['inverse_map']
            all_labels = batch_dict['targets_mapped']
            point_predict = []
            point_labels = []
            point_predict_logits = []
            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (x.C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                if return_logit or return_tta:
                    outputs_mapped = out[cur_scene_pts][cur_inv].softmax(1)
                else:
                    outputs_mapped = out[cur_scene_pts][cur_inv].argmax(1)
                    outputs_mapped_logits = out[cur_scene_pts][cur_inv]
                targets_mapped = all_labels.F[cur_label]
                point_predict.append(outputs_mapped[:batch_dict['num_points'][idx]].cpu().numpy())
                point_labels.append(targets_mapped[:batch_dict['num_points'][idx]].cpu().numpy())
                point_predict_logits.append(outputs_mapped_logits[:batch_dict['num_points'][idx]].cpu().numpy())

            return {'point_predict': point_predict, 'point_labels': point_labels, 'name': batch_dict['name'],'point_predict_logits': point_predict_logits}

    def forward_ensemble(self, batch_dict):
        return self.forward(batch_dict, ensemble=True)
