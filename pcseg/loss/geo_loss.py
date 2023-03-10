import torch
import torch.nn as nn
from lib.pointops.functions import pointops
import torch.nn.functional as F

class GeoLoss(nn.Module):
    """
        Local Geometric Anisotropic
    """
    def __init__(self, ignore_index, nsample=10):
        """
        nsample: KNN -> K neighbor of current point
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.nsample = nsample

    def forward(self, input, target, xyz, offset):
        """
        Args:
            target: points gt labels for one batch
            xyz: coords of all points -> voxelized xyz : lidar.C[:, :3].float() or labels.C[:, :3].float()
            offset: each batch points num offset
        """

        mask = (target != self.ignore_index)  # remove ignore_index classes
        target = target[mask]
        input = input[mask]
        # mask xyz
        new_xyz = xyz = xyz[mask]
        # mask offset
        new_offset = offset = \
            torch.tensor([mask[:offset_][mask[:offset_] == True].shape[0] \
                        for offset_ in offset]).type_as(offset)

        idx, _ = pointops.knnquery(self.nsample, xyz, new_xyz, offset, new_offset)
        # idx: (m, nsample) -> n nearest points indices for xyz[index]
        # dist2: (m, nsample) -> n nearest points distance for xyz[index]

        # mapping indices to labels
        knn_labels = target[idx.long()]
        target_ = torch.repeat_interleave(target, repeats=idx.shape[1]).reshape(-1, idx.shape[1])
        xor_ = ~(target_ == knn_labels)

        # local geometric anisotropic
        points_lga = xor_.sum(dim=1).float()
        lamda = 1
        alpha = 0.5
        point_weights = (lamda + alpha * points_lga) / self.nsample
        ### set weights close to its mean
        point_weights = point_weights / point_weights.mean()

        # get log softmax score for each point
        logits = F.log_softmax(input, dim=1)
        # get pred score for each point (according to gt label)
        pred_score = logits.gather(dim=1, index=target.reshape(-1, 1)).view(-1)

        loss = (-pred_score * point_weights).mean()

        return loss
