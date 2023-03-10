import torch
import torch.nn as nn
from torch.nn import functional as F

from pcseg.model.segmentor.base_segmentors import BaseSegmentor
from pcseg.model.segmentor.range.utils import ClassWeightSemikitti, CrossEntropyDiceLoss, Lovasz_softmax, BoundaryLoss


class FIDNet(BaseSegmentor):
    def __init__(
        self,
        model_cfgs,
        num_class: int,
    ):
        super(FIDNet, self).__init__(model_cfgs, num_class)
        # backbone
        self.backend=Backbone(
            if_BN=model_cfgs.IF_BN,
            if_remission=model_cfgs.IF_INTENSITY,
            if_range=model_cfgs.IF_RANGE,
            with_normal=model_cfgs.WITH_NORM,
        )

        # segmentation head
        self.semantic_head = SemanticHead(
            num_class=num_class,
            input_channel=1024,
        )

        # loss func
        self.if_ls_loss = model_cfgs.IF_LS_LOSS
        self.if_bd_loss = model_cfgs.IF_BD_LOSS
        self.build_loss_funs(model_cfgs)

    def build_loss_funs(self, model_cfgs):
        self.top_k_percent_pixels = model_cfgs.TOP_K_PERCENT_PIXELS

        if model_cfgs.LOSS == 'wce':
            weight = torch.tensor(ClassWeightSemikitti.get_weight()).cuda()
            self.WCE = torch.nn.CrossEntropyLoss(reduction='none', weight=weight).cuda()
        elif model_cfgs.LOSS == 'dice':
            self.WCE = CrossEntropyDiceLoss(reduction='none').cuda()
        
        if self.if_ls_loss:
            self.LS = Lovasz_softmax(ignore=0).cuda()
        
        if self.if_bd_loss:
            self.BD = BoundaryLoss().cuda()
        
    def forward(self, batch):
        scan_rv = batch['scan_rv']  # [bs, 6, H, W]
        label_rv = batch['label_rv']
        if len(label_rv.size()) != 3:
            label_rv = label_rv.squeeze(dim=1)  # [bs, H, W]

        middle_feature_maps = self.backend(scan_rv)  # [bs, 1024, H, W]
        logits = self.semantic_head(middle_feature_maps)  # [bs, cls, H, W]

        if logits.size()[-1] != label_rv.size()[-1] and logits.size()[-2] != label_rv.size()[-2]:
            logits = F.interpolate(logits, size=label_rv.size()[1:], mode='bilinear', align_corners=True)  # [bs, cls, H, W]

        if self.training:
            pixel_losses = self.WCE(logits, label_rv)
            pixel_losses = pixel_losses.contiguous().view(-1)

            if self.top_k_percent_pixels == 1.0:
                loss_ce = pixel_losses.mean()
            else:
                top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
                pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
                loss_ce = pixel_losses.mean()

            if self.if_ls_loss:
                loss_ls = self.LS(F.softmax(logits, dim=1), label_rv)
                loss_ls = loss_ls.mean()
            else:
                loss_ls = 0.

            if self.if_bd_loss:
                loss_bd = self.BD(F.softmax(logits, dim=1), label_rv)
            else:
                loss_bd = 0.

            loss = 1.0 * loss_ce + 3.0 * loss_ls + 1.0 * loss_bd

            ret_dict = {'loss': loss}
            disp_dict = {'loss': loss.item()}
            tb_dict = {'loss': loss.item()}

            return ret_dict, tb_dict, disp_dict
        
        else:
            return {'point_predict': logits, 'point_labels': label_rv}


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, if_BN=None):
        super(BasicBlock, self).__init__()

        self.if_BN = if_BN
        if self.if_BN: norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64: raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1: raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.if_BN: self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = conv3x3(planes, planes)
        if self.if_BN: self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.if_BN: out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.if_BN: out = self.bn2(out)

        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, if_BN=None):
        super(Bottleneck, self).__init__()
        self.if_BN = if_BN
        if self.if_BN: norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        if self.if_BN: self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        if self.if_BN: self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        if self.if_BN: self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.LeakyReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.if_BN: out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.if_BN: out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.if_BN: out = self.bn3(out)

        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class SemanticHead(nn.Module):
    def __init__(self, num_class, input_channel=1024):
        super(SemanticHead,self).__init__()

        self.conv_1=nn.Conv2d(input_channel, 512, 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu_1 = nn.LeakyReLU()

        self.conv_2=nn.Conv2d(512, 128, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu_2 = nn.LeakyReLU()

        self.semantic_output=nn.Conv2d(128, num_class, 1)

    def forward(self, input_tensor):    # [bs, 1024, 64, 512]
        res=self.conv_1(input_tensor)   # [bs, 512, 64, 512]
        res=self.bn1(res)
        res=self.relu_1(res)
        
        res=self.conv_2(res)            # [bs, 128, 64, 512]
        res=self.bn2(res)
        res=self.relu_2(res)
        
        res=self.semantic_output(res)   # [bs, cls, 64, 512]
        return res


class SemanticBackbone(nn.Module):
    def __init__(self, block, layers, if_BN, if_remission, if_range, with_normal, norm_layer=None, groups=1, width_per_group=64):
        super(SemanticBackbone, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.if_BN=if_BN
        self.if_remission=if_remission
        self.if_range=if_range
        self.with_normal=with_normal
        self.inplanes = 512
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        if not self.if_remission and not self.if_range and not self.with_normal:        
            self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=True)
            self.bn_0 = nn.BatchNorm2d(64)
            self.relu_0 = nn.LeakyReLU()

        if self.if_remission and not self.if_range and not self.with_normal:
            self.conv1 = nn.Conv2d(4, 64, kernel_size=1, stride=1, padding=0, bias=True)
            self.bn_0 = nn.BatchNorm2d(64)
            self.relu_0 = nn.LeakyReLU()

        if self.if_remission and self.if_range and not self.with_normal:
            self.conv1 = nn.Conv2d(6, 64, kernel_size=1, stride=1, padding=0, bias=True)
            self.bn_0 = nn.BatchNorm2d(64)
            self.relu_0 = nn.LeakyReLU()

        if self.if_remission and self.if_range and self.with_normal:
            self.conv1 = nn.Conv2d(9, 64, kernel_size=1, stride=1, padding=0, bias=True)
            self.bn_0 = nn.BatchNorm2d(64)
            self.relu_0 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0,bias=True)
        self.bn_1 = nn.BatchNorm2d(256)
        self.relu_1 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0,bias=True)
        self.bn_2 = nn.BatchNorm2d(512)
        self.relu_2 = nn.LeakyReLU()
        
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.if_BN:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, if_BN=self.if_BN))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, if_BN=self.if_BN))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):  # [bs, 6, 64, 512]

        x = self.conv1(x)  # [bs, 64, 64, 512]
        x = self.bn_0(x)
        x = self.relu_0(x)

        x = self.conv2(x)  # [bs, 128, 64, 512]
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv3(x)  # [bs, 256, 64, 512]
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv4(x)  # [bs, 512, 64, 512]
        x = self.bn_2(x)
        x = self.relu_2(x)

        x_1 = self.layer1(x)     # [bs, 128, 64, 512]
        x_2 = self.layer2(x_1)   # [bs, 128, 32, 256]
        x_3 = self.layer3(x_2)   # [bs, 128, 16, 128]
        x_4 = self.layer4(x_3)   # [bs, 128, 8, 64]

        res_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)  # [bs, 128, 64, 512]
        res_3 = F.interpolate(x_3, size=x.size()[2:], mode='bilinear', align_corners=True)  # [bs, 128, 64, 512]
        res_4 = F.interpolate(x_4, size=x.size()[2:], mode='bilinear', align_corners=True)  # [bs, 128, 64, 512]
        
        res=[x, x_1, res_2, res_3, res_4]

        return torch.cat(res, dim=1)

    def forward(self, x):
        return self._forward_impl(x)


def _backbone(arch, block, layers, if_BN, if_remission, if_range, with_normal):
    model = SemanticBackbone(block, layers, if_BN, if_remission, if_range, with_normal)
    return model

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def Backbone(if_BN, if_remission, if_range, with_normal):
    """ResNet-34 model from "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>"""
    return _backbone('resnet34', BasicBlock, [3, 4, 6, 3], if_BN, if_remission, if_range, with_normal)

