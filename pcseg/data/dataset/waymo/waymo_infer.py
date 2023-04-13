import os
import numpy as np
from torch.utils import data
import random
import pickle


class WaymoInferDataset(data.Dataset):
    '''
    Inference dataset for loading an unpacked sequence of Waymo
    '''
    def __init__(
        self,
        data_cfgs=None,
        training: bool = True,
        class_names: list = None,
        root_path: str = None,
        logger = None,
        nusc=None,
        if_scribble: bool = False,
    ):
        super().__init__()
        self.data_cfgs = data_cfgs
        self.root_path = root_path
        self.training = training
        self.logger = logger
        self.class_names = class_names
        self.tta = data_cfgs.get('TTA', False)
        self.train_val = data_cfgs.get('TRAINVAL', False)
        self.augment = data_cfgs.AUGMENT

        annos = []
        point_npys = os.listdir(self.data_cfgs.INPUT_DIR)
        point_npys.sort()
        for npy in point_npys:
            annos.append(os.path.join(self.data_cfgs.INPUT_DIR, npy))
        
        self.annos = annos

        self.annos_another = self.annos.copy()
        random.shuffle(self.annos_another)
        print(f'The total sample is {len(self.annos)}')

        self.sample_idx = self._sample_idx = np.arange(len(self.annos))

        self.samples_per_epoch = self.data_cfgs.get('SAMPLES_PER_EPOCH', -1)
        if self.samples_per_epoch == -1 or not self.training:
            self.samples_per_epoch = len(self.annos)

    def __len__(self):
        return len(self.sample_idx)

    def resample(self):
        self.sample_idx = np.random.choice(self._sample_idx, self.samples_per_epoch)
    
    def __getitem__(self, index):
        index = self.sample_idx[index]
        ann_info = self.annos[index]
        raw_data = np.load(ann_info).astype(np.float32)
    
        # Placeholder label
        annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)

        pc_data = {
            'xyzret': raw_data.astype(np.float32),
            'labels': annotated_data.astype(np.uint8),
            'path': ann_info,
        }

        return pc_data

    @staticmethod
    def collate_batch(batch_list):
        raise NotImplementedError
