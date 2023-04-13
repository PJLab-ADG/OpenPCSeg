import os
import numpy as np
from torch.utils import data
import random
import pickle

class WaymoDataset(data.Dataset):
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


        if self.training and not self.train_val:
            self.split = 'train'
        else:
            if self.training and self.train_val:
                self.split = 'train_val'
            else:
                self.split = 'val'
        if self.tta:
            self.split = 'test'

        annos = []
        if self.split == 'train':
            with open('./data_root/Waymo/train-0-31.txt', 'r') as f:
                for line in f.readlines():
                    annos.append(line.strip())
        elif self.split == 'val':
            with open('./data_root/Waymo/val-0-7.txt', 'r') as f:
                for line in f.readlines():
                    annos.append(line.strip())
        else:
            raise Exception('split must be train/val/train_val/test.')

        self.annos = annos
        

        self.annos_another = self.annos.copy()
        random.shuffle(self.annos_another)
        print(f'The total sample is {len(self.annos)}')

        self._sample_idx = np.arange(len(self.annos))

        self.samples_per_epoch = self.data_cfgs.get('SAMPLES_PER_EPOCH', -1)
        if self.samples_per_epoch == -1 or not self.training:
            self.samples_per_epoch = len(self.annos)

        if self.training:
            self.resample()
        else:
            self.sample_idx = self._sample_idx

    def __len__(self):
        return len(self.sample_idx)

    def resample(self):
        self.sample_idx = np.random.choice(self._sample_idx, self.samples_per_epoch)

    def __getitem__(self, index):
        index = self.sample_idx[index]
        ann_info = self.annos[index]
        raw_xyz = np.load(ann_info)[:,3:6].reshape((-1,3)).astype(np.float32)
        intenel = np.load(ann_info)[:,1:3].reshape((-1,2)).astype(np.float32)
        pc_first = np.concatenate((raw_xyz,intenel),1)

        sec_path = ann_info.replace('first/', 'second/')
        raw_xyz1 = np.load(sec_path)[:, 3:6].reshape((-1, 3)).astype(np.float32)
        intenel1 = np.load(sec_path)[:, 1:3].reshape((-1, 2)).astype(np.float32)
        pc_second = np.concatenate((raw_xyz1, intenel1), 1)

        raw_data = np.concatenate((pc_first, pc_second), 0).astype(np.float32).copy()
        # NORMALIZE INTENSITY and elongation
        raw_data[:, 3:5] = np.tanh(raw_data[:, 3:5])
    
        if self.split == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data_first = np.load(ann_info)[:,-1].reshape((-1,1)).astype(np.int32)
            annotated_data_second =  np.load(sec_path)[:, -1].reshape((-1, 1)).astype(np.int32)
            annotated_data = np.concatenate((annotated_data_first,annotated_data_second), 0)
            
        pc_data = {
            'xyzret': raw_data.astype(np.float32),
            'labels': annotated_data.astype(np.uint8),
            'path': ann_info,
        }

        return pc_data

    @staticmethod
    def collate_batch(batch_list):
        raise NotImplementedError

