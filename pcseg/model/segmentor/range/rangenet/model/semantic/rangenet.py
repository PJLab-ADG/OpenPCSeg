import imp

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcseg.model.segmentor.range.rangenet.postproc.CRF import CRF
from pcseg.model.segmentor.base_segmentors import BaseSegmentor
from pcseg.model.segmentor.range.utils import ClassWeightSemikitti, CrossEntropyDiceLoss, Lovasz_softmax, BoundaryLoss
# import __init__ as booger
from pcseg.model.segmentor.range.rangenet.module.darknet import Backbone, Decoder

#https://github.com/RnRi/RangeNet-/blob/main/tasks/semantic/config/arch/darknet53-crf.yaml
class RangeNet(BaseSegmentor): # TODO CRF
    def __init__(
		self,
        model_cfgs,
		num_class: int,
	):
        super().__init__(model_cfgs, num_class)
        self.num_class = num_class
        self.H = 64
        self.W = 512
        # Module = imp.load_source( "bboneModule", '../module/' + 'darknet' + '.py')

		# backbone
        self.backbone = Backbone() #Module.Backbone()

        # do a pass of the backbone to initialize the skip connections
        stub = torch.zeros((1, 5, self.H, self.W))
        if torch.cuda.is_available():
            stub = stub.cuda()
            self.backbone.cuda()
        _, stub_skips = self.backbone(stub)

		# decoder
        self.decoder = Decoder(stub_skips=stub_skips) #Module.Decoder(stub_skips=stub_skips)

		# head
        self.head = nn.Sequential(
			nn.Dropout2d(0.01),
            nn.Conv2d(
				self.decoder.get_last_depth(), self.num_class,
				kernel_size=3, stride=1, padding=1,
			)
		)

		# CRF
        self.CRF = None
        # if self.ARCH["post"]["CRF"]["use"]:
        #     self.CRF = CRF(self.ARCH["post"]["CRF"]["params"], self.num_class)
        #     weights_crf = sum(p.numel() for p in self.CRF.parameters())
        #     print("Param CRF ", weights_crf)
        # else:
        #     self.CRF = None

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
		
    def forward(self, batch, mask=None):
        scan_rv = batch['scan_rv']  # [bs, 6, H, W]
        label_rv = batch['label_rv']
        if len(label_rv.size()) != 3:
            label_rv = label_rv.squeeze(dim=1)  # [bs, H, W]

        y, skips = self.backbone(scan_rv)
        y = self.decoder(y, skips)
        logits = self.head(y)

        if self.CRF:
            assert (mask is not None)
            logits = self.CRF(scan_rv, logits, mask)

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
