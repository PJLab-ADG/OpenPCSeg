import torch
import torch.nn as nn


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_pts, ignore_index, label_smoothing, normal_w=True):
        """
        Args:
            cls_num_pts (list): num_points of each class in the whole dataset
            normal_wce (bools): whether normalize weights
            ignore_index (int): index class not cal loss
            label_smoothing (float): 
        """
        super().__init__()
        # frequency of each class
        if cls_num_pts is not None:
            f_c = torch.Tensor(cls_num_pts) / sum(cls_num_pts)        
            weights = 1 / torch.sqrt(f_c)
            weights[ignore_index] = 0
            cls_num_pts = torch.tensor(cls_num_pts)
            if normal_w:
                weights = weights / weights.sum() * len(cls_num_pts)
            weights = weights / ((weights * cls_num_pts).sum() / cls_num_pts.sum())
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                                weight=weights,
                                                label_smoothing=label_smoothing)

    def forward(self, input ,target):
        return self.ce_loss(input, target)
