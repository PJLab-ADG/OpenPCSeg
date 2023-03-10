import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler
from tools.utils.common import common_utils

from .dataset.semantickitti import SemkittiRangeViewDataset, SemkittiVoxelDataset, SemkittiCylinderDataset, SemkittiFusionDataset
from .dataset.waymo import WaymoVoxelDataset, WaymoCylinderDataset, WaymoFusionDataset

__all__ = {
    # SemanticKITTI
    'SemkittiRangeViewDataset': SemkittiRangeViewDataset,
    'SemkittiVoxelDataset': SemkittiVoxelDataset,
    'SemkittiCylinderDataset': SemkittiCylinderDataset,
    'SemkittiFusionDataset': SemkittiFusionDataset,

    # Waymo
    'WaymoVoxelDataset': WaymoVoxelDataset,
    'WaymoCylinderDataset': WaymoCylinderDataset,
    'WaymoFusionDataset': WaymoFusionDataset,
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(
    data_cfgs,
    modality: str,
    batch_size: int,
    dist: bool = False,
    root_path: str = None,
    workers: int = 10,
    logger = None,
    training: bool = True,
    merge_all_iters_to_one_epoch: bool = False,
    total_epochs: int = 0,
):
    if modality == 'range':
        if data_cfgs.DATASET == 'nuscenes':
            db = 'NuscRangeViewDataset'
        elif data_cfgs.DATASET == 'semantickitti' or data_cfgs.DATASET == 'scribblekitti':
            db = 'SemkittiRangeViewDataset'
        else:
            raise NotImplementedError
    elif modality == 'voxel':
        if data_cfgs.DATASET == 'nuscenes':
            db = 'NuscVoxelDataset'
        elif data_cfgs.DATASET == 'semantickitti' or data_cfgs.DATASET == 'scribblekitti':
            db = 'SemkittiVoxelDataset'
        elif data_cfgs.DATASET == 'waymo':
            db = 'WaymoVoxelDataset'
        else:
            raise NotImplementedError
    elif modality == 'cylinder':
        if data_cfgs.DATASET == 'nuscenes':
            db = 'NuscCylinderDataset'
        elif data_cfgs.DATASET == 'semantickitti' or data_cfgs.DATASET == 'scribblekitti':
            db = 'SemkittiCylinderDataset'
        elif data_cfgs.DATASET == 'waymo':
            db = 'WaymoCylinderDataset'
        else:
            raise NotImplementedError
    elif modality == 'bev':
        raise NotImplementedError
    elif modality == 'fusion':
        if data_cfgs.DATASET == 'nuscenes':
            db  = 'NuscFusionDataset'
        elif data_cfgs.DATASET == 'semantickitti' or data_cfgs.DATASET == 'scribblekitti':
            db = 'SemkittiFusionDataset'
        elif data_cfgs.DATASET == 'waymo':
            db = 'WaymoFusionDataset'
        else:
            raise NotImplementedError
    
    dataset = eval(db)(
        data_cfgs=data_cfgs,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    
    tta = data_cfgs.get('TTA', False)
    if tta:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=workers,
            shuffle = (sampler is None) and training,
            collate_fn=dataset.collate_batch_tta,
            drop_last=False,
            sampler=sampler,
            timeout=0,
            persistent_workers=(workers > 0),
        )
    else:
        if modality == 'range':
            sampler_train = torch.utils.data.RandomSampler(dataset)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler_train,
                num_workers=workers,
                drop_last=True,
                pin_memory=True,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=workers,
                shuffle=(sampler is None) and training,
                collate_fn=dataset.collate_batch,
                drop_last=False,
                sampler=sampler,
                timeout=0,
                persistent_workers=(workers > 0),
            )
    
    return dataset, dataloader, sampler
