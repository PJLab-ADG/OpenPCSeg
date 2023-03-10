import torch
import torch.nn as nn
import torch.nn.functional as F

from pcseg.model.segmentor.base_segmentors import BaseSegmentor
from pcseg.model.segmentor.range.utils import ClassWeightSemikitti, CrossEntropyDiceLoss, Lovasz_softmax, BoundaryLoss


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
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
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        dropout_rate: float,
        kernel_size: tuple = (3, 3),
        stride: int = 1,
        pooling: bool = True,
        drop_out: bool = True,
    ):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

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

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1,resA2,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA,skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1, upE2, upE3), dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


class SalsaNext(BaseSegmentor):
    def __init__(
        self,
        model_cfgs,
        num_class: int,
    ):
        super(SalsaNext, self).__init__(model_cfgs, num_class)
        self.num_class = num_class
        self.if_ls_loss = model_cfgs.IF_LS_LOSS
        self.if_bd_loss = model_cfgs.IF_BD_LOSS

        # encoder
        self.downCntx = ResContextBlock(6, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

        # decoder
        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)

        # head
        self.logits = nn.Conv2d(32, num_class, kernel_size=(1, 1))
        
        # loss
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

        downCntx = self.downCntx(scan_rv)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)

        up4e = self.upBlock1(down5c,down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)

        logits = self.logits(up1e)

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
