import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dice_coefficient(pred, target, smooth=1, exponent=2):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    num = torch.sum(torch.mul(pred, target), dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    neg_samples_idx = torch.where(target == 0)[0]
    pos_samples_idx = torch.where(target == 1)[0]
    mask = torch.zeros(target.shape[0], device=target.device)  # specify device -> gpu accelerating
    mask[pos_samples_idx] = 1
    tot = 3 * len(pos_samples_idx)
    if tot >= neg_samples_idx.shape[0]:
        tot = neg_samples_idx.shape[0]
    # specify device -> gpu accelerating
    random_sample_neg_idx = torch.randperm(neg_samples_idx.size(0), device=target.device)[:tot]
    random_sample_neg_idx = neg_samples_idx[random_sample_neg_idx]
    mask[random_sample_neg_idx] = 1

    coef = num / den
    coef_ = ((coef * mask).sum() + smooth) / (mask.sum() + smooth)  
    # add smoothing, or `mask` filling with only 0 will make `coef_` = 0
    # -log(0) = inf !!!!! loss ==> inf ==> wrong training
    # denominator also need smoothing, `coef_` <= 1 ==> `-log(coef_)` >= 0

    return coef_


def dice_loss(pred, target, smooth=1, exponent=2, class_weight=None,
              ignore_index=0, gamma_dice=1):
    """
    dice coef for each class
    return mean
    """
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_coef = dice_coefficient(
                pred[:, i],
                target[..., i],
                smooth=smooth,
                exponent=exponent)
            if class_weight is not None:
                dice_coef *= class_weight[i]
            total_loss += (-torch.log(dice_coef)).pow(gamma_dice)
    return total_loss / num_classes


class ELLLoss(nn.Module):
    def __init__(self, ignore_index, label_smoothing, cls_num_pts, normal_w=False, 
                    w_dice=0.8, w_cross=0.2, gamma_dice=1, gamma_cross=1, smooth=1, exponent=2):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.exponent = exponent
        self.smooth = smooth
        self.gamma_dice = gamma_dice
        self.gamma_cross = gamma_cross
        self.w_dice = w_dice
        self.w_cross = w_cross

        # frequency of each class
        if cls_num_pts is not None:
            f_c = torch.Tensor(cls_num_pts) / sum(cls_num_pts)        
            self.w_l = 1 / torch.sqrt(f_c)
            self.w_l[ignore_index] = 0
            if normal_w:
                self.w_l = self.w_l / self.w_l.sum() * len(cls_num_pts)

    
    def forward(self, input, target):
        """
        L_exp = w_dice * L_dice + w_cross * L_cross

        L_dice = E[(-ln(Dice))^{\gamma_{dice}}]  E -> mean value
        Dice_i = [2(pred_score * gt_label) + eps] / [(pred_score^2 + gt^2) + eps]
            pred_score: softmax probability
            gt: gt label
            Dice_i : per class dice_coef

        L_cross = E[w_l(-ln(pred_score))^{\gamma_{cross}}]
            w_l = 1 / sqrt(f_c)     f_c: class frequency

        gamma_dice = gamma_cross = 0.3 | 1
        
        """
        ### mask ignore_index, no loss for ignore_index
        mask = (target != self.ignore_index)
        input = input[mask]
        target = target[mask]

        pred_score = input.softmax(dim=-1)
        one_hot_target = F.one_hot(target, num_classes=pred_score.shape[1])
        
        L_dice = dice_loss(pred_score, one_hot_target, smooth=self.smooth, 
                            exponent=self.exponent, ignore_index=self.ignore_index)

        point_weights = self.w_l[target].type_as(pred_score)

        pred_score = pred_score.gather(1, index=target.reshape(-1, 1)).reshape(-1)
        L_cross = (point_weights * (-torch.log(pred_score)).pow(self.gamma_cross)).reshape(-1)

        loss = self.w_dice * L_dice.mean() + self.w_cross * L_cross.mean()


        return L_dice.mean()
