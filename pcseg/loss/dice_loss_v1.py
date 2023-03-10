import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.reshape(predict.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) * 2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        ###
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
        ###

        loss = 1 - num / den

        if self.reduction == 'mean':
            return (loss * mask).sum() / (mask.sum() + 1e-10)


class DiceLossV1(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, ignore_index, smooth=1, exponent=2):
        super(DiceLossV1, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.exponent = exponent

    def forward(self, input, target):
        mask = (target != self.ignore_index)
        target = target[mask]
        input = input[mask]

        dice = BinaryDiceLoss(self.smooth, self.exponent)
        total_loss = 0
        input = F.softmax(input, dim=1)
        target_ = target.reshape(-1, 1)
        target = make_one_hot(target_, num_classes=input.shape[1]).type_as(target_)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(input[:, i], target[:, i])
                total_loss += dice_loss

        return total_loss / target.shape[1]
