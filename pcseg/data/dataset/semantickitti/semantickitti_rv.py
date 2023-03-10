import os
import cv2
import glob
import random
import yaml

import numpy as np
import torch
from torch.utils import data
from torchvision.transforms import functional as F
from .semantickitti_utils import LEARNING_MAP, color_map
from .laserscan import SemLaserScan


class SemkittiRangeViewDataset(data.Dataset):
    
    def __init__(
        self,
        data_cfgs = None,
        training: bool = True,
        root_path: bool = None,
        logger = None,
    ):
        self.data_cfgs = data_cfgs
        self.training = training
        self.root = root_path if root_path is not None else self.data_cfgs.DATA_PATH
        self.logger = logger
        self.split = self.data_cfgs.DATA_SPLIT['train'] if self.training else self.data_cfgs.DATA_SPLIT['test']
        self.H, self.W = self.data_cfgs.H, self.data_cfgs.W  # (H, W)
        self.color_dict = color_map
        self.label_transfer_dict = LEARNING_MAP  # label mapping
        self.nclasses = len(self.color_dict)  # 34

        self.class_names = [
            "unlabeled",  # ignored
            "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist",  # dynamic
            "road", "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign"  # static
        ]

        if self.data_cfgs.DATASET == 'scribblekitti':
            self.if_scribble = True
        else:
            self.if_scribble = False

        # common aug
        self.if_drop = False if not self.training else self.data_cfgs.IF_DROP
        self.if_flip = False if not self.training else self.data_cfgs.IF_FLIP
        self.if_scale = False if not self.training else self.data_cfgs.IF_SCALE
        self.if_rotate = False if not self.training else self.data_cfgs.IF_ROTATE
        self.if_jitter = False if not self.training else self.data_cfgs.IF_JITTER

        # range aug
        self.if_range_mix = False if not self.training else self.data_cfgs.IF_RANGE_MIX
        self.if_range_shift = False if not self.training else self.data_cfgs.IF_RANGE_SHIFT
        self.if_range_paste = False if not self.training else self.data_cfgs.IF_RANGE_PASTE
        self.instance_list = [
            'bicycle', 'motorcycle', 'truck' 'other-vehicle', 
            'person', 'bicyclist', 'motorcyclist', 'other-ground', 
            'trunk', 'pole', 'traffic-sign'
        ]
        self.if_range_union = False if not self.training else self.data_cfgs.IF_RANGE_UNION

        self.A=SemLaserScan(
            nclasses = self.nclasses,
            sem_color_dict = self.color_dict,
            project = True,
            H = self.H,
            W = self.W, 
            fov_up = 3.0,
            fov_down = -25.0,
            if_drop = self.if_drop,
            if_flip = self.if_flip,
            if_scale = self.if_scale,
            if_rotate = self.if_rotate,
            if_jitter = self.if_jitter,
            if_range_mix = self.if_range_mix,
            if_range_paste=self.if_range_paste,
            if_range_union=self.if_range_union,
        )
        if self.split == 'train': folders = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
        elif self.split == 'val': folders = ['08']
        elif self.split == 'test': folders = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        
        self.lidar_list = []
        for folder in folders:
            self.lidar_list += glob.glob(self.root + 'sequences/' + folder + '/velodyne/*.bin') 
        print("Loading '{}' samples from SemanticKITTI under '{}' split".format(len(self.lidar_list), self.split))

        self.label_list = [i.replace("velodyne", "labels") for i in self.lidar_list]
        self.label_list = [i.replace("bin", "label") for i in self.label_list]

        if self.split == 'train_test':
            root_psuedo_labels = '/mnt/lustre/konglingdong/data/sets/sequences/'
            folders_test = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
            for i in self.label_list:
                if i.split('sequences/')[1][:2] in folders_test:
                    i.replace(self.root + 'sequences/', root_psuedo_labels)

        if self.if_range_mix:
            strategy = 'mixtureV2'
            # strategy = 'col4row1'
            self.BeamMix = MixTeacherSemkitti(strategy=strategy)
            print("Lasermix strategy: '{}'".format(strategy))

        print("Prob (RangeMix): {}.".format(self.if_range_mix))
        print("Prob (RangePaste): {}.".format(self.if_range_paste))
        print("Prob (RangeUnion): {}.".format(self.if_range_union))

        if self.if_scribble:
            self.label_list = [i.replace("SemanticKITTI", "ScribbleKITTI") for i in self.label_list]
            self.label_list = [i.replace("labels", "scribbles") for i in self.label_list]
            print("Loading '{}' labels from ScribbleKITTI under '{}' split.\n".format(len(self.label_list), self.split))

        else:
            print("Loading '{}' labels from SemanticKITTI under '{}' split.\n".format(len(self.label_list), self.split))


    def __len__(self):
        return len(self.lidar_list)

    def __getitem__(self, index):
        self.A.open_scan(self.lidar_list[index])
        self.A.open_label(self.label_list[index])

        # prepare attributes
        dataset_dict = {}

        dataset_dict['xyz'] = self.A.proj_xyz
        dataset_dict['intensity'] = self.A.proj_remission
        dataset_dict['range_img'] = self.A.proj_range
        dataset_dict['xyz_mask'] = self.A.proj_mask
        
        semantic_label = self.A.proj_sem_label
        semantic_train_label = self.generate_label(semantic_label)
        dataset_dict['semantic_label'] = semantic_train_label

        # data aug (range shift)
        if np.random.random() >= (1 - self.if_range_shift):
            split_point = random.randint(100, self.W-100)
            dataset_dict = self.sample_transform(dataset_dict, split_point)

        scan, label, mask = self.prepare_input_label_semantic_with_mask(dataset_dict)

        if self.if_range_mix > 0 or self.if_range_paste > 0 or self.if_range_union > 0:

            idx = np.random.randint(0, len(self.lidar_list))

            self.A.open_scan(self.lidar_list[idx])
            self.A.open_label(self.label_list[idx])

            dataset_dict_ = {}
            dataset_dict_['xyz'] = self.A.proj_xyz
            dataset_dict_['intensity'] = self.A.proj_remission
            dataset_dict_['range_img'] = self.A.proj_range
            dataset_dict_['xyz_mask'] = self.A.proj_mask

            semantic_label = self.A.proj_sem_label
            semantic_train_label = self.generate_label(semantic_label)
            dataset_dict_['semantic_label'] = semantic_train_label

            # data aug (range shift)
            if np.random.random() >= (1 - self.if_range_shift):
                split_point_ = random.randint(100, self.W-100)
                dataset_dict_ = self.sample_transform(dataset_dict_, split_point_)

            scan_, label_, mask_ = self.prepare_input_label_semantic_with_mask(dataset_dict_)

            # data aug (range mix)
            if np.random.random() >= (1 - self.if_range_mix):
                scan_mix1, label_mix1, mask_mix1, scan_mix2, label_mix2, mask_mix2, s = self.BeamMix.forward(scan, label, mask, scan_, label_, mask_)

                if np.random.random() >= 0.5:
                    scan, label, mask = scan_mix1, label_mix1, mask_mix1
                else:
                    scan, label, mask = scan_mix2, label_mix2, mask_mix2

            # data aug (range paste)
            if np.random.random() >= (1 - self.if_range_paste):
                scan, label, mask = self.RangePaste(scan, label, mask, scan_, label_, mask_)

            # data aug (range union)
            if np.random.random() >= (1 - self.if_range_union):
                scan, label, mask = self.RangeUnion(scan, label, mask, scan_, label_, mask_)

        data_dict = {
            'scan_rv': F.to_tensor(scan),
            'label_rv': F.to_tensor(label).to(dtype=torch.long),
            'mask_rv': F.to_tensor(mask),
            'scan_name': self.lidar_list[index],
        }

        return data_dict

        # return F.to_tensor(scan), F.to_tensor(label).to(dtype=torch.long), F.to_tensor(mask), self.lidar_list[index]


    def RangeUnion(self, scan, label, mask, scan_, label_, mask_):
        pix_empty = mask == 0

        scan_new = scan.copy()
        label_new = label.copy()
        mask_new = mask.copy()

        scan_new[pix_empty]  = scan_[pix_empty]
        label_new[pix_empty] = label_[pix_empty]
        mask_new[pix_empty]  = mask_[pix_empty]
        return scan_new, label_new, mask_new


    def RangePaste(self, scan, label, mask, scan_, label_, mask_):
        scan_new = scan.copy()
        label_new = label.copy()
        mask_new = mask.copy()

        pix_bicycle = label_ == 2  # cls: 2 (bicycle)
        if np.sum(pix_bicycle) > 20:
            scan_new[pix_bicycle]  = scan_[pix_bicycle]
            label_new[pix_bicycle] = label_[pix_bicycle]
            mask_new[pix_bicycle]  = mask_[pix_bicycle]
        
        pix_motorcycle = label_ == 3  # cls: 3 (motorcycle)
        if np.sum(pix_motorcycle) > 20:
            scan_new[pix_motorcycle]  = scan_[pix_motorcycle]
            label_new[pix_motorcycle] = label_[pix_motorcycle]
            mask_new[pix_motorcycle]  = mask_[pix_motorcycle]

        pix_truck = label_ == 4  # cls: 4 (truck)
        if np.sum(pix_truck) > 20:
            scan_new[pix_truck]  = scan_[pix_truck]
            label_new[pix_truck] = label_[pix_truck]
            mask_new[pix_truck]  = mask_[pix_truck]

        pix_other_vehicle = label_ == 5  # cls: 5 (other-vehicle)
        if np.sum(pix_other_vehicle) > 20:
            scan_new[pix_other_vehicle]  = scan_[pix_other_vehicle]
            label_new[pix_other_vehicle] = label_[pix_other_vehicle]
            mask_new[pix_other_vehicle]  = mask_[pix_other_vehicle]

        pix_person = label_ == 6  # cls: 6 (person)
        if np.sum(pix_person) > 20:
            scan_new[pix_person]  = scan_[pix_person]
            label_new[pix_person] = label_[pix_person]
            mask_new[pix_person]  = mask_[pix_person]

        pix_bicyclist = label_ == 7  # cls: 7 (bicyclist)
        if np.sum(pix_bicyclist) > 20:
            scan_new[pix_bicyclist]  = scan_[pix_bicyclist]
            label_new[pix_bicyclist] = label_[pix_bicyclist]
            mask_new[pix_bicyclist]  = mask_[pix_bicyclist]

        pix_motorcyclist = label_ == 8  # cls: 8 (motorcyclist)
        if np.sum(pix_motorcyclist) > 20:
            scan_new[pix_motorcyclist]  = scan_[pix_motorcyclist]
            label_new[pix_motorcyclist] = label_[pix_motorcyclist]
            mask_new[pix_motorcyclist]  = mask_[pix_motorcyclist]

        pix_other_ground = label_ == 12  # cls: 12 (other-ground)
        if np.sum(pix_other_ground) > 20:
            scan_new[pix_other_ground]  = scan_[pix_other_ground]
            label_new[pix_other_ground] = label_[pix_other_ground]
            mask_new[pix_other_ground]  = mask_[pix_other_ground]

        pix_other_trunk = label_ == 16  # cls: 16 (trunk)
        if np.sum(pix_other_trunk) > 20:
            scan_new[pix_other_trunk]  = scan_[pix_other_trunk]
            label_new[pix_other_trunk] = label_[pix_other_trunk]
            mask_new[pix_other_trunk]  = mask_[pix_other_trunk]
        
        pix_pole = label_ == 18  # cls: 18 (pole)
        if np.sum(pix_pole) > 20:
            scan_new[pix_pole]  = scan_[pix_pole]
            label_new[pix_pole] = label_[pix_pole]
            mask_new[pix_pole]  = mask_[pix_pole]

        pix_traffic_sign = label_ == 19  # cls: 19 (traffic-sign)
        if np.sum(pix_traffic_sign) > 20:
            scan_new[pix_traffic_sign]  = scan_[pix_traffic_sign]
            label_new[pix_traffic_sign] = label_[pix_traffic_sign]
            mask_new[pix_traffic_sign]  = mask_[pix_traffic_sign]

        return scan_new, label_new, mask_new


    def prepare_input_label_semantic_with_mask(self, sample):
        scale_x = np.expand_dims(np.ones([self.H, self.W]) * 50.0, axis=-1).astype(np.float32)
        scale_y = np.expand_dims(np.ones([self.H, self.W]) * 50.0, axis=-1).astype(np.float32)
        scale_z = np.expand_dims(np.ones([self.H, self.W]) * 3.0,  axis=-1).astype(np.float32)
        scale_matrx = np.concatenate([scale_x, scale_y, scale_z], axis=2)

        each_input = [
            sample['xyz'] / scale_matrx,
            np.expand_dims(sample['intensity'], axis=-1), 
            np.expand_dims(sample['range_img']/80.0, axis=-1),
            np.expand_dims(sample['xyz_mask'], axis=-1)
        ]
        input_tensor = np.concatenate(each_input, axis=-1)

        semantic_label = sample['semantic_label'][:, :]
        semantic_label_mask = sample['xyz_mask'][:, :]

        return input_tensor, semantic_label, semantic_label_mask


    def sample_transform(self, dataset_dict, split_point):
        dataset_dict['xyz'] = np.concatenate(
            [dataset_dict['xyz'][:, split_point:, :], dataset_dict['xyz'][:, :split_point, :]], axis=1
        )
        dataset_dict['xyz_mask'] = np.concatenate(
            [dataset_dict['xyz_mask'][:, split_point:], dataset_dict['xyz_mask'][:, :split_point]], axis=1
        )
        dataset_dict['intensity'] = np.concatenate(
            [dataset_dict['intensity'][:, split_point:], dataset_dict['intensity'][:, :split_point]], axis=1
        )
        dataset_dict['range_img'] = np.concatenate(
            [dataset_dict['range_img'][:, split_point:], dataset_dict['range_img'][:, :split_point]], axis=1
        )
        dataset_dict['semantic_label'] = np.concatenate(
            [dataset_dict['semantic_label'][:, split_point:], dataset_dict['semantic_label'][:, :split_point]], axis=1
        )
        return dataset_dict


    def sem_label_transform(self,raw_label_map):
        for i in self.label_transfer_dict.keys():
            raw_label_map[raw_label_map==i]=self.label_transfer_dict[i]
        
        return raw_label_map


    def generate_label(self,semantic_label):
        original_label=np.copy(semantic_label)
        label_new=self.sem_label_transform(original_label)
        
        return label_new


    def fill_spherical(self,range_image):
        # fill in spherical image for calculating normal vector
        height,width=np.shape(range_image)[:2]
        value_mask=np.asarray(1.0-np.squeeze(range_image)>0.1).astype(np.uint8)
        dt, lbl = cv2.distanceTransformWithLabels(value_mask, cv2.DIST_L1, 5, labelType=cv2.DIST_LABEL_PIXEL)
        with_value=np.squeeze(range_image)>0.1
        depth_list=np.squeeze(range_image)[with_value]
        label_list=np.reshape(lbl,[1,height*width])
        depth_list_all=depth_list[label_list-1]
        depth_map=np.reshape(depth_list_all,(height,width))
        depth_map = cv2.GaussianBlur(depth_map,(7,7),0)
        depth_map=range_image*with_value+depth_map*(1-with_value)
        
        return depth_map
                   

def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


class MixTeacherSemkitti:
    
    def __init__(self, strategy,):
        super(MixTeacherSemkitti, self).__init__()
        self.strategy = strategy
        
    def forward(self, image, label, mask, image_aux, label_aux, mask_aux):
        """
        Arguments:
            - strategy: MixTeacher strategies.
            - image: original image, size: [6, H, W].
            - label: original label, size: [H, W].
            - mask:  original mask,  size: [H, W].
            - image_aux: auxiliary image, size: [6, H, W].
            - label_aux: auxiliary label, size: [H, W].
            - mask_aux:  auxiliary mask,  size: [H, W].
        Return:
            (2x) Augmented images, labels, and masks.
        """

        image, image_aux = np.transpose(image, (2, 0, 1)), np.transpose(image_aux, (2, 0, 1))

                
        if self.strategy == 'mixture':
            strategies = ['col1row2', 'col1row3', 'col2row1', 'col3row1', 'col2row2', 'col1row4', 'col2row4']
            strategy = np.random.choice(strategies, size=1)[0]

        elif self.strategy == 'mixtureV2':
            strategies = ['col1row3', 'col1row4', 'col1row5', 'col1row6', 'col2row3', 'col2row4', 'col2row5', 'col2row6', 'col3row3', 'col3row4', 'col3row5', 'col3row6', 'col4row3', 'col4row4', 'col4row5', 'col4row6', 'col6row4', ]
            strategy = np.random.choice(strategies, size=1)[0]

        else:
            strategy = self.strategy
            
        if strategy == 'col1row2':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col1row2(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col1row3':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col1row3(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col1row4':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col1row4(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col1row5':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col1row5(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col1row6':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col1row6(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col2row1':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col2row1(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col2row2':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col2row2(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col2row3':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col2row3(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col2row4':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col2row4(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col2row5':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col2row5(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col2row6':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col2row6(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col3row1':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col3row1(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col3row2':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col3row2(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col3row3':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col3row3(image, label, mask, image_aux, label_aux, mask_aux)
        
        elif strategy == 'col3row4':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col3row4(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col3row5':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col3row5(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col3row6':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col3row6(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col4row1':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col4row1(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col4row2':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col4row2(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col4row3':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col4row3(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col4row4':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col4row4(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col4row5':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col4row5(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col4row6':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col4row6(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'col6row4':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.col6row4(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'cutmix':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.cutmix(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'cutout':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.cutout(image, label, mask, image_aux, label_aux, mask_aux)

        elif strategy == 'mixup':
            img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2 = self.mixup(image, label, mask, image_aux, label_aux, mask_aux)


        img_aux1, img_aux2 = np.transpose(img_aux1, (1, 2, 0)), np.transpose(img_aux2, (1, 2, 0))

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2, strategy
        
        
    def col1row2(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        mid_h = int(img.shape[-2] / 2)  # 64/2 = 32

        imgA_1, imgA_2 = img[:, :mid_h, :], img[:, mid_h:, :]  # upper-half1, lower-half1
        lblA_1, lblA_2 = lbl[   :mid_h, :], lbl[   mid_h:, :]  # upper-half1, lower-half1
        mskA_1, mskA_2 = msk[   :mid_h, :], msk[   mid_h:, :]  # upper-half1, lower-half1

        imgB_1, imgB_2 = img_aux[:, :mid_h, :], img_aux[:, mid_h:, :]  # upper-half2, lower-half2
        lblB_1, lblB_2 = lbl_aux[   :mid_h, :], lbl_aux[   mid_h:, :]  # upper-half2, lower-half2
        mskB_1, mskB_2 = msk_aux[   :mid_h, :], msk_aux[   mid_h:, :]  # upper-half2, lower-half2

        img_aux1 = np.concatenate((imgA_1, imgB_2), axis=-2)  # upper-half1, lower-half2
        lbl_aux1 = np.concatenate((lblA_1, lblB_2), axis=-2)  # upper-half1, lower-half2
        msk_aux1 = np.concatenate((mskA_1, mskB_2), axis=-2)  # upper-half1, lower-half2

        img_aux2 = np.concatenate((imgB_1, imgA_2), axis=-2)  # upper-half2, lower-half1
        lbl_aux2 = np.concatenate((lblB_1, lblA_2), axis=-2)  # upper-half2, lower-half1
        msk_aux2 = np.concatenate((mskB_1, mskA_2), axis=-2)  # upper-half2, lower-half1
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col1row3(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 3)  # 64/3 = 21
        h2 = 2 * h1                   # 21*2 = 42

        imgA_1, imgA_2, imgA_3 = img[:, :h1, :], img[:, h1:h2, :], img[:, h2:, :]  # upper1, middle1, lower1
        lblA_1, lblA_2, lblA_3 = lbl[   :h1, :], lbl[   h1:h2, :], lbl[   h2:, :]  # upper1, middle1, lower1
        mskA_1, mskA_2, mskA_3 = msk[   :h1, :], msk[   h1:h2, :], msk[   h2:, :]  # upper1, middle1, lower1

        imgB_1, imgB_2, imgB_3 = img_aux[:, :h1, :], img_aux[:, h1:h2, :], img_aux[:, h2:, :]  # upper2, middle2, lower2
        lblB_1, lblB_2, lblB_3 = lbl_aux[   :h1, :], lbl_aux[   h1:h2, :], lbl_aux[   h2:, :]  # upper2, middle2, lower2
        mskB_1, mskB_2, mskB_3 = msk_aux[   :h1, :], msk_aux[   h1:h2, :], msk_aux[   h2:, :]  # upper2, middle2, lower2

        img_aux1 = np.concatenate((imgA_1, imgB_2, imgA_3), axis=-2)  # upper1, middle2, lower1
        lbl_aux1 = np.concatenate((lblA_1, lblB_2, lblA_3), axis=-2)  # upper1, middle2, lower1
        msk_aux1 = np.concatenate((mskA_1, mskB_2, mskA_3), axis=-2)  # upper1, middle2, lower1

        img_aux2 = np.concatenate((imgB_1, imgA_2, imgB_3), axis=-2)  # upper2, middle1, lower2
        lbl_aux2 = np.concatenate((lblB_1, lblA_2, lblB_3), axis=-2)  # upper2, middle1, lower2
        msk_aux2 = np.concatenate((mskB_1, mskA_2, mskB_3), axis=-2)  # upper2, middle1, lower2
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col1row4(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 4)     # 64/4 = 16
        mid_h = int(img.shape[-2] / 2)  # 64/2 = 32
        h3 = 3 * h1                     # 16*3 = 48

        imgA_1, imgA_2, imgA_3, imgA_4 = img[:, :h1, :], img[:, h1:mid_h, :], img[:, mid_h:h3, :], img[:, h3:, :]
        lblA_1, lblA_2, lblA_3, lblA_4 = lbl[   :h1, :], lbl[   h1:mid_h, :], lbl[   mid_h:h3, :], lbl[   h3:, :]
        mskA_1, mskA_2, mskA_3, mskA_4 = msk[   :h1, :], msk[   h1:mid_h, :], msk[   mid_h:h3, :], msk[   h3:, :]

        imgB_1, imgB_2, imgB_3, imgB_4 = img_aux[:, :h1, :], img_aux[:, h1:mid_h, :], img_aux[:, mid_h:h3, :], img_aux[:, h3:, :]
        lblB_1, lblB_2, lblB_3, lblB_4 = lbl_aux[   :h1, :], lbl_aux[   h1:mid_h, :], lbl_aux[   mid_h:h3, :], lbl_aux[   h3:, :]
        mskB_1, mskB_2, mskB_3, mskB_4 = msk_aux[   :h1, :], msk_aux[   h1:mid_h, :], msk_aux[   mid_h:h3, :], msk_aux[   h3:, :]

        img_aux1 = np.concatenate((imgA_1, imgB_2, imgA_3, imgB_4), axis=-2)
        lbl_aux1 = np.concatenate((lblA_1, lblB_2, lblA_3, lblB_4), axis=-2)
        msk_aux1 = np.concatenate((mskA_1, mskB_2, mskA_3, mskB_4), axis=-2)

        img_aux2 = np.concatenate((imgB_1, imgA_2, imgB_3, imgA_4), axis=-2)
        lbl_aux2 = np.concatenate((lblB_1, lblA_2, lblB_3, lblA_4), axis=-2)
        msk_aux2 = np.concatenate((mskB_1, mskA_2, mskB_3, mskA_4), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2
    
    
    def col1row5(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 5)      # 64/5 = 12
        h2 = 2 * h1                      # 2*12 = 24
        h3 = 3 * h1                      # 3*12 = 36
        h4 = 4 * h1                      # 4*12 = 48

        imgA_1, imgA_2, imgA_3, imgA_4, imgA_5 = img[:, :h1, :], img[:, h1:h2, :], img[:, h2:h3, :], img[:, h3:h4, :], img[:, h4:, :]
        lblA_1, lblA_2, lblA_3, lblA_4, lblA_5 = lbl[   :h1, :], lbl[   h1:h2, :], lbl[   h2:h3, :], lbl[   h3:h4, :], lbl[   h4:, :]
        mskA_1, mskA_2, mskA_3, mskA_4, mskA_5 = msk[   :h1, :], msk[   h1:h2, :], msk[   h2:h3, :], msk[   h3:h4, :], msk[   h4:, :]

        imgB_1, imgB_2, imgB_3, imgB_4, imgB_5 = img_aux[:, :h1, :], img_aux[:, h1:h2, :], img_aux[:, h2:h3, :], img_aux[:, h3:h4, :], img_aux[:, h4:, :]
        lblB_1, lblB_2, lblB_3, lblB_4, lblB_5 = lbl_aux[   :h1, :], lbl_aux[   h1:h2, :], lbl_aux[   h2:h3, :], lbl_aux[   h3:h4, :], lbl_aux[   h4:, :]
        mskB_1, mskB_2, mskB_3, mskB_4, mskB_5 = msk_aux[   :h1, :], msk_aux[   h1:h2, :], msk_aux[   h2:h3, :], msk_aux[   h3:h4, :], msk_aux[   h4:, :]

        img_aux1 = np.concatenate((imgA_1, imgB_2, imgA_3, imgB_4, imgA_5), axis=-2)
        lbl_aux1 = np.concatenate((lblA_1, lblB_2, lblA_3, lblB_4, lblA_5), axis=-2)
        msk_aux1 = np.concatenate((mskA_1, mskB_2, mskA_3, mskB_4, mskA_5), axis=-2)

        img_aux2 = np.concatenate((imgB_1, imgA_2, imgB_3, imgA_4, imgB_5), axis=-2)
        lbl_aux2 = np.concatenate((lblB_1, lblA_2, lblB_3, lblA_4, lblB_5), axis=-2)
        msk_aux2 = np.concatenate((mskB_1, mskA_2, mskB_3, mskA_4, mskB_5), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col1row6(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 6)     # 64/6 = 10
        h2 = 2 * h1                      # 2*10 = 20
        h3 = 3 * h1                      # 3*10 = 30
        h4 = 4 * h1                      # 4*10 = 40
        h5 = 5 * h1                      # 5*10 = 50

        imgA_1, imgA_2, imgA_3, imgA_4, imgA_5, imgA_6 = img[:, :h1, :], img[:, h1:h2, :], img[:, h2:h3, :], img[:, h3:h4, :], img[:, h4:h5, :], img[:, h5:, :]
        lblA_1, lblA_2, lblA_3, lblA_4, lblA_5, lblA_6 = lbl[   :h1, :], lbl[   h1:h2, :], lbl[   h2:h3, :], lbl[   h3:h4, :], lbl[   h4:h5, :], lbl[   h5:, :]
        mskA_1, mskA_2, mskA_3, mskA_4, mskA_5, mskA_6 = msk[   :h1, :], msk[   h1:h2, :], msk[   h2:h3, :], msk[   h3:h4, :], msk[   h4:h5, :], msk[   h5:, :]

        imgB_1, imgB_2, imgB_3, imgB_4, imgB_5, imgB_6 = img_aux[:, :h1, :], img_aux[:, h1:h2, :], img_aux[:, h2:h3, :], img_aux[:, h3:h4, :], img_aux[:, h4:h5, :], img_aux[:, h5:, :]
        lblB_1, lblB_2, lblB_3, lblB_4, lblB_5, lblB_6 = lbl_aux[   :h1, :], lbl_aux[   h1:h2, :], lbl_aux[   h2:h3, :], lbl_aux[   h3:h4, :], lbl_aux[   h4:h5, :], lbl_aux[   h5:, :]
        mskB_1, mskB_2, mskB_3, mskB_4, mskB_5, mskB_6 = msk_aux[   :h1, :], msk_aux[   h1:h2, :], msk_aux[   h2:h3, :], msk_aux[   h3:h4, :], msk_aux[   h4:h5, :], msk_aux[   h5:, :]

        img_aux1 = np.concatenate((imgA_1, imgB_2, imgA_3, imgB_4, imgA_5, imgB_6), axis=-2)
        lbl_aux1 = np.concatenate((lblA_1, lblB_2, lblA_3, lblB_4, lblA_5, lblB_6), axis=-2)
        msk_aux1 = np.concatenate((mskA_1, mskB_2, mskA_3, mskB_4, mskA_5, mskB_6), axis=-2)

        img_aux2 = np.concatenate((imgB_1, imgA_2, imgB_3, imgA_4, imgB_5, imgA_6), axis=-2)
        lbl_aux2 = np.concatenate((lblB_1, lblA_2, lblB_3, lblA_4, lblB_5, lblA_6), axis=-2)
        msk_aux2 = np.concatenate((mskB_1, mskA_2, mskB_3, mskA_4, mskB_5, mskA_6), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col2row1(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        mid_w = int(img.shape[-1] / 2)  # 2048/2 = 1024
        
        imgA_1, imgA_2 = img[:, :, :mid_w], img[:, :, mid_w:]  # left-half1, right-half1
        lblA_1, lblA_2 = lbl[   :, :mid_w], lbl[   :, mid_w:]  # left-half1, right-half1
        mskA_1, mskA_2 = msk[   :, :mid_w], msk[   :, mid_w:]  # left-half1, right-half1
        
        imgB_1, imgB_2 = img_aux[:, :, :mid_w], img_aux[:, :, mid_w:]  # left-half2, right-half2
        lblB_1, lblB_2 = lbl_aux[   :, :mid_w], lbl_aux[   :, mid_w:]  # left-half2, right-half2
        mskB_1, mskB_2 = msk_aux[   :, :mid_w], msk_aux[   :, mid_w:]  # left-half2, right-half2
        
        img_aux1 = np.concatenate((imgA_1, imgB_2), axis=-1)  # left-half1, right-half2
        lbl_aux1 = np.concatenate((lblA_1, lblB_2), axis=-1)  # left-half1, right-half2
        msk_aux1 = np.concatenate((mskA_1, mskB_2), axis=-1)  # left-half1, right-half2

        img_aux2 = np.concatenate((imgB_1, imgA_2), axis=-1)  # left-half2, right-half1
        lbl_aux2 = np.concatenate((lblB_1, lblA_2), axis=-1)  # left-half2, right-half1
        msk_aux2 = np.concatenate((mskB_1, mskA_2), axis=-1)  # left-half2, right-half1
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col2row2(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        mid_h = int(img.shape[-2] / 2)  # 64/2 = 32
        mid_w = int(img.shape[-1] / 2)  # 2048/2 = 1024
        
        imgA_11, imgA_12, imgA_21, imgA_22 = img[:, :mid_h, :mid_w], img[:, :mid_h, mid_w:], img[:, mid_h:, :mid_w], img[:, mid_h:, mid_w:]
        lblA_11, lblA_12, lblA_21, lblA_22 = lbl[   :mid_h, :mid_w], lbl[   :mid_h, mid_w:], lbl[   mid_h:, :mid_w], lbl[   mid_h:, mid_w:]
        mskA_11, mskA_12, mskA_21, mskA_22 = msk[   :mid_h, :mid_w], msk[   :mid_h, mid_w:], msk[   mid_h:, :mid_w], msk[   mid_h:, mid_w:]
        
        imgB_11, imgB_12, imgB_21, imgB_22 = img_aux[:, :mid_h, :mid_w], img_aux[:, :mid_h, mid_w:], img_aux[:, mid_h:, :mid_w], img_aux[:, mid_h:, mid_w:]
        lblB_11, lblB_12, lblB_21, lblB_22 = lbl_aux[   :mid_h, :mid_w], lbl_aux[   :mid_h, mid_w:], lbl_aux[   mid_h:, :mid_w], lbl_aux[   mid_h:, mid_w:]
        mskB_11, mskB_12, mskB_21, mskB_22 = msk_aux[   :mid_h, :mid_w], msk_aux[   :mid_h, mid_w:], msk_aux[   mid_h:, :mid_w], msk_aux[   mid_h:, mid_w:]

        concat_img_aux1_top = np.concatenate((imgA_11, imgB_12), axis=-1)
        concat_img_aux1_bot = np.concatenate((imgB_21, imgA_22), axis=-1)
        img_aux1 = np.concatenate((concat_img_aux1_top, concat_img_aux1_bot), axis=-2)
        
        concat_lbl_aux1_top = np.concatenate((lblA_11, lblB_12), axis=-1)
        concat_lbl_aux1_bot = np.concatenate((lblB_21, lblA_22), axis=-1)
        lbl_aux1 = np.concatenate((concat_lbl_aux1_top, concat_lbl_aux1_bot), axis=-2)
        
        concat_msk_aux1_top = np.concatenate((mskA_11, mskB_12), axis=-1)
        concat_msk_aux1_bot = np.concatenate((mskB_21, mskA_22), axis=-1)
        msk_aux1 = np.concatenate((concat_msk_aux1_top, concat_msk_aux1_bot), axis=-2)
        
        concat_img_aux2_top = np.concatenate((imgB_11, imgA_12), axis=-1)
        concat_img_aux2_bot = np.concatenate((imgA_21, imgB_22), axis=-1)
        img_aux2 = np.concatenate((concat_img_aux2_top, concat_img_aux2_bot), axis=-2)
        
        concat_lbl_aux2_top = np.concatenate((lblB_11, lblA_12), axis=-1)
        concat_lbl_aux2_bot = np.concatenate((lblA_21, lblB_22), axis=-1)
        lbl_aux2 = np.concatenate((concat_lbl_aux2_top, concat_lbl_aux2_bot), axis=-2)
        
        concat_msk_aux2_top = np.concatenate((mskB_11, mskA_12), axis=-1)
        concat_msk_aux2_bot = np.concatenate((mskA_21, mskB_22), axis=-1)
        msk_aux2 = np.concatenate((concat_msk_aux2_top, concat_msk_aux2_bot), axis=-2)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col2row3(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 3)      # 64/3 = 21
        h2 = 2 * h1                      # 2*21 = 42
        mid_w = int(img.shape[-1] / 2)   # 2048/2 = 1024

        imgA_11, imgA_21, imgA_31 = img[:, :h1, :mid_w], img[:, h1:h2, :mid_w], img[:, h2:, :mid_w]
        imgA_12, imgA_22, imgA_32 = img[:, :h1, mid_w:], img[:, h1:h2, mid_w:], img[:, h2:, mid_w:]
        lblA_11, lblA_21, lblA_31 = lbl[   :h1, :mid_w], lbl[   h1:h2, :mid_w], lbl[   h2:, :mid_w]
        lblA_12, lblA_22, lblA_32 = lbl[   :h1, mid_w:], lbl[   h1:h2, mid_w:], lbl[   h2:, mid_w:]
        mskA_11, mskA_21, mskA_31 = msk[   :h1, :mid_w], msk[   h1:h2, :mid_w], msk[   h2:, :mid_w]
        mskA_12, mskA_22, mskA_32 = msk[   :h1, mid_w:], msk[   h1:h2, mid_w:], msk[   h2:, mid_w:]

        imgB_11, imgB_21, imgB_31 = img_aux[:, :h1, :mid_w], img_aux[:, h1:h2, :mid_w], img_aux[:, h2:, :mid_w]
        imgB_12, imgB_22, imgB_32 = img_aux[:, :h1, mid_w:], img_aux[:, h1:h2, mid_w:], img_aux[:, h2:, mid_w:]
        lblB_11, lblB_21, lblB_31 = lbl_aux[   :h1, :mid_w], lbl_aux[   h1:h2, :mid_w], lbl_aux[   h2:, :mid_w]
        lblB_12, lblB_22, lblB_32 = lbl_aux[   :h1, mid_w:], lbl_aux[   h1:h2, mid_w:], lbl_aux[   h2:, mid_w:]
        mskB_11, mskB_21, mskB_31 = msk_aux[   :h1, :mid_w], msk_aux[   h1:h2, :mid_w], msk_aux[   h2:, :mid_w]
        mskB_12, mskB_22, mskB_32 = msk_aux[   :h1, mid_w:], msk_aux[   h1:h2, mid_w:], msk_aux[   h2:, mid_w:]

        img_aux1_l, img_aux1_r = np.concatenate((imgA_11, imgB_21, imgA_31), axis=-2), np.concatenate((imgB_12, imgA_22, imgB_32), axis=-2)
        img_aux1 = np.concatenate((img_aux1_l, img_aux1_r), axis=-1)
        lbl_aux1_l, lbl_aux1_r = np.concatenate((lblA_11, lblB_21, lblA_31), axis=-2), np.concatenate((lblB_12, lblA_22, lblB_32), axis=-2)
        lbl_aux1 = np.concatenate((lbl_aux1_l, lbl_aux1_r), axis=-1)
        msk_aux1_l, msk_aux1_r = np.concatenate((mskA_11, mskB_21, mskA_31), axis=-2), np.concatenate((mskB_12, mskA_22, mskB_32), axis=-2)
        msk_aux1 = np.concatenate((msk_aux1_l, msk_aux1_r), axis=-1)

        img_aux2_l, img_aux2_r = np.concatenate((imgB_11, imgA_21, imgB_31), axis=-2), np.concatenate((imgA_12, imgB_22, imgA_32), axis=-2)
        img_aux2 = np.concatenate((img_aux2_l, img_aux2_r), axis=-1)
        lbl_aux2_l, lbl_aux2_r = np.concatenate((lblB_11, lblA_21, lblB_31), axis=-2), np.concatenate((lblA_12, lblB_22, lblA_32), axis=-2)
        lbl_aux2 = np.concatenate((lbl_aux2_l, lbl_aux2_r), axis=-1)
        msk_aux2_l, msk_aux2_r = np.concatenate((mskB_11, mskA_21, mskB_31), axis=-2), np.concatenate((mskA_12, mskB_22, mskA_32), axis=-2)
        msk_aux2 = np.concatenate((msk_aux2_l, msk_aux2_r), axis=-1)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col2row4(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 4)      # 64/4 = 16
        mid_h = int(img.shape[-2] / 2)   # 64/2 = 32
        h3 = 3 * h1                      # 16*3 = 48
        mid_w = int(img.shape[-1] / 2)   # 2048/2 = 1024

        imgA_11, imgA_21, imgA_31, imgA_41 = img[:, :h1, :mid_w], img[:, h1:mid_h, :mid_w], img[:, mid_h:h3, :mid_w], img[:, h3:, :mid_w]
        imgA_12, imgA_22, imgA_32, imgA_42 = img[:, :h1, mid_w:], img[:, h1:mid_h, mid_w:], img[:, mid_h:h3, mid_w:], img[:, h3:, mid_w:]
        lblA_11, lblA_21, lblA_31, lblA_41 = lbl[   :h1, :mid_w], lbl[   h1:mid_h, :mid_w], lbl[   mid_h:h3, :mid_w], lbl[   h3:, :mid_w]
        lblA_12, lblA_22, lblA_32, lblA_42 = lbl[   :h1, mid_w:], lbl[   h1:mid_h, mid_w:], lbl[   mid_h:h3, mid_w:], lbl[   h3:, mid_w:]
        mskA_11, mskA_21, mskA_31, mskA_41 = msk[   :h1, :mid_w], msk[   h1:mid_h, :mid_w], msk[   mid_h:h3, :mid_w], msk[   h3:, :mid_w]
        mskA_12, mskA_22, mskA_32, mskA_42 = msk[   :h1, mid_w:], msk[   h1:mid_h, mid_w:], msk[   mid_h:h3, mid_w:], msk[   h3:, mid_w:]

        imgB_11, imgB_21, imgB_31, imgB_41 = img_aux[:, :h1, :mid_w], img_aux[:, h1:mid_h, :mid_w], img_aux[:, mid_h:h3, :mid_w], img_aux[:, h3:, :mid_w]
        imgB_12, imgB_22, imgB_32, imgB_42 = img_aux[:, :h1, mid_w:], img_aux[:, h1:mid_h, mid_w:], img_aux[:, mid_h:h3, mid_w:], img_aux[:, h3:, mid_w:]
        lblB_11, lblB_21, lblB_31, lblB_41 = lbl_aux[   :h1, :mid_w], lbl_aux[   h1:mid_h, :mid_w], lbl_aux[   mid_h:h3, :mid_w], lbl_aux[   h3:, :mid_w]
        lblB_12, lblB_22, lblB_32, lblB_42 = lbl_aux[   :h1, mid_w:], lbl_aux[   h1:mid_h, mid_w:], lbl_aux[   mid_h:h3, mid_w:], lbl_aux[   h3:, mid_w:]
        mskB_11, mskB_21, mskB_31, mskB_41 = msk_aux[   :h1, :mid_w], msk_aux[   h1:mid_h, :mid_w], msk_aux[   mid_h:h3, :mid_w], msk_aux[   h3:, :mid_w]
        mskB_12, mskB_22, mskB_32, mskB_42 = msk_aux[   :h1, mid_w:], msk_aux[   h1:mid_h, mid_w:], msk_aux[   mid_h:h3, mid_w:], msk_aux[   h3:, mid_w:]

        img_aux1_l, img_aux1_r = np.concatenate((imgA_11, imgB_21, imgA_31, imgB_41), axis=-2), np.concatenate((imgB_12, imgA_22, imgB_32, imgA_42), axis=-2)
        img_aux1 = np.concatenate((img_aux1_l, img_aux1_r), axis=-1)
        lbl_aux1_l, lbl_aux1_r = np.concatenate((lblA_11, lblB_21, lblA_31, lblB_41), axis=-2), np.concatenate((lblB_12, lblA_22, lblB_32, lblA_42), axis=-2)
        lbl_aux1 = np.concatenate((lbl_aux1_l, lbl_aux1_r), axis=-1)
        msk_aux1_l, msk_aux1_r = np.concatenate((mskA_11, mskB_21, mskA_31, mskB_41), axis=-2), np.concatenate((mskB_12, mskA_22, mskB_32, mskA_42), axis=-2)
        msk_aux1 = np.concatenate((msk_aux1_l, msk_aux1_r), axis=-1)

        img_aux2_l, img_aux2_r = np.concatenate((imgB_11, imgA_21, imgB_31, imgA_41), axis=-2), np.concatenate((imgA_12, imgB_22, imgA_32, imgB_42), axis=-2)
        img_aux2 = np.concatenate((img_aux2_l, img_aux2_r), axis=-1)
        lbl_aux2_l, lbl_aux2_r = np.concatenate((lblB_11, lblA_21, lblB_31, lblA_41), axis=-2), np.concatenate((lblA_12, lblB_22, lblA_32, lblB_42), axis=-2)
        lbl_aux2 = np.concatenate((lbl_aux2_l, lbl_aux2_r), axis=-1)
        msk_aux2_l, msk_aux2_r = np.concatenate((mskB_11, mskA_21, mskB_31, mskA_41), axis=-2), np.concatenate((mskA_12, mskB_22, mskA_32, mskB_42), axis=-2)
        msk_aux2 = np.concatenate((msk_aux2_l, msk_aux2_r), axis=-1)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col2row5(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 5)      # 64/5 = 12
        h2 = 2 * h1                      # 2*12 = 24
        h3 = 3 * h1                      # 3*12 = 36
        h4 = 4 * h1                      # 4*12 = 48
        mid_w = int(img.shape[-1] / 2)   # 2048/2 = 1024

        imgA_11, imgA_21, imgA_31, imgA_41, imgA_51 = img[:, :h1, :mid_w], img[:, h1:h2, :mid_w], img[:, h2:h3, :mid_w], img[:, h3:h4, :mid_w], img[:, h4:, :mid_w]
        imgA_12, imgA_22, imgA_32, imgA_42, imgA_52 = img[:, :h1, mid_w:], img[:, h1:h2, mid_w:], img[:, h2:h3, mid_w:], img[:, h3:h4, mid_w:], img[:, h4:, mid_w:]
        lblA_11, lblA_21, lblA_31, lblA_41, lblA_51 = lbl[   :h1, :mid_w], lbl[   h1:h2, :mid_w], lbl[   h2:h3, :mid_w], lbl[   h3:h4, :mid_w], lbl[   h4:, :mid_w]
        lblA_12, lblA_22, lblA_32, lblA_42, lblA_52 = lbl[   :h1, mid_w:], lbl[   h1:h2, mid_w:], lbl[   h2:h3, mid_w:], lbl[   h3:h4, mid_w:], lbl[   h4:, mid_w:]
        mskA_11, mskA_21, mskA_31, mskA_41, mskA_51 = msk[   :h1, :mid_w], msk[   h1:h2, :mid_w], msk[   h2:h3, :mid_w], msk[   h3:h4, :mid_w], msk[   h4:, :mid_w]
        mskA_12, mskA_22, mskA_32, mskA_42, mskA_52 = msk[   :h1, mid_w:], msk[   h1:h2, mid_w:], msk[   h2:h3, mid_w:], msk[   h3:h4, mid_w:], msk[   h4:, mid_w:]

        imgB_11, imgB_21, imgB_31, imgB_41, imgB_51 = img_aux[:, :h1, :mid_w], img_aux[:, h1:h2, :mid_w], img_aux[:, h2:h3, :mid_w], img_aux[:, h3:h4, :mid_w], img_aux[:, h4:, :mid_w]
        imgB_12, imgB_22, imgB_32, imgB_42, imgB_52 = img_aux[:, :h1, mid_w:], img_aux[:, h1:h2, mid_w:], img_aux[:, h2:h3, mid_w:], img_aux[:, h3:h4, mid_w:], img_aux[:, h4:, mid_w:]
        lblB_11, lblB_21, lblB_31, lblB_41, lblB_51 = lbl_aux[   :h1, :mid_w], lbl_aux[   h1:h2, :mid_w], lbl_aux[   h2:h3, :mid_w], lbl_aux[   h3:h4, :mid_w], lbl_aux[   h4:, :mid_w]
        lblB_12, lblB_22, lblB_32, lblB_42, lblB_52 = lbl_aux[   :h1, mid_w:], lbl_aux[   h1:h2, mid_w:], lbl_aux[   h2:h3, mid_w:], lbl_aux[   h3:h4, mid_w:], lbl_aux[   h4:, mid_w:]
        mskB_11, mskB_21, mskB_31, mskB_41, mskB_51 = msk_aux[   :h1, :mid_w], msk_aux[   h1:h2, :mid_w], msk_aux[   h2:h3, :mid_w], msk_aux[   h3:h4, :mid_w], msk_aux[   h4:, :mid_w]
        mskB_12, mskB_22, mskB_32, mskB_42, mskB_52 = msk_aux[   :h1, mid_w:], msk_aux[   h1:h2, mid_w:], msk_aux[   h2:h3, mid_w:], msk_aux[   h3:h4, mid_w:], msk_aux[   h4:, mid_w:]

        img_aux1_l, img_aux1_r = np.concatenate((imgA_11, imgB_21, imgA_31, imgB_41, imgA_51), axis=-2), np.concatenate((imgB_12, imgA_22, imgB_32, imgA_42, imgB_52), axis=-2)
        img_aux1 = np.concatenate((img_aux1_l, img_aux1_r), axis=-1)
        lbl_aux1_l, lbl_aux1_r = np.concatenate((lblA_11, lblB_21, lblA_31, lblB_41, lblA_51), axis=-2), np.concatenate((lblB_12, lblA_22, lblB_32, lblA_42, lblB_52), axis=-2)
        lbl_aux1 = np.concatenate((lbl_aux1_l, lbl_aux1_r), axis=-1)
        msk_aux1_l, msk_aux1_r = np.concatenate((mskA_11, mskB_21, mskA_31, mskB_41, mskA_51), axis=-2), np.concatenate((mskB_12, mskA_22, mskB_32, mskA_42, mskB_52), axis=-2)
        msk_aux1 = np.concatenate((msk_aux1_l, msk_aux1_r), axis=-1)

        img_aux2_l, img_aux2_r = np.concatenate((imgB_11, imgA_21, imgB_31, imgA_41, imgB_51), axis=-2), np.concatenate((imgA_12, imgB_22, imgA_32, imgB_42, imgA_52), axis=-2)
        img_aux2 = np.concatenate((img_aux2_l, img_aux2_r), axis=-1)
        lbl_aux2_l, lbl_aux2_r = np.concatenate((lblB_11, lblA_21, lblB_31, lblA_41, lblB_51), axis=-2), np.concatenate((lblA_12, lblB_22, lblA_32, lblB_42, lblA_52), axis=-2)
        lbl_aux2 = np.concatenate((lbl_aux2_l, lbl_aux2_r), axis=-1)
        msk_aux2_l, msk_aux2_r = np.concatenate((mskB_11, mskA_21, mskB_31, mskA_41, mskB_51), axis=-2), np.concatenate((mskA_12, mskB_22, mskA_32, mskB_42, mskA_52), axis=-2)
        msk_aux2 = np.concatenate((msk_aux2_l, msk_aux2_r), axis=-1)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col2row6(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        h1 = int(img.shape[-2] / 6)      # 64/6 = 10
        h2 = 2 * h1                      # 2*10 = 20
        h3 = 3 * h1                      # 3*10 = 30
        h4 = 4 * h1                      # 4*10 = 40
        h5 = 5 * h1                      # 5*10 = 50
        mid_w = int(img.shape[-1] / 2)   # 2048/2 = 1024

        imgA_11, imgA_21, imgA_31, imgA_41, imgA_51, imgA_61 = img[:, :h1, :mid_w], img[:, h1:h2, :mid_w], img[:, h2:h3, :mid_w], img[:, h3:h4, :mid_w], img[:, h4:h5, :mid_w], img[:, h5:, :mid_w]
        imgA_12, imgA_22, imgA_32, imgA_42, imgA_52, imgA_62 = img[:, :h1, mid_w:], img[:, h1:h2, mid_w:], img[:, h2:h3, mid_w:], img[:, h3:h4, mid_w:], img[:, h4:h5, mid_w:], img[:, h5:, mid_w:]
        lblA_11, lblA_21, lblA_31, lblA_41, lblA_51, lblA_61 = lbl[   :h1, :mid_w], lbl[   h1:h2, :mid_w], lbl[   h2:h3, :mid_w], lbl[   h3:h4, :mid_w], lbl[   h4:h5, :mid_w], lbl[   h5:, :mid_w]
        lblA_12, lblA_22, lblA_32, lblA_42, lblA_52, lblA_62 = lbl[   :h1, mid_w:], lbl[   h1:h2, mid_w:], lbl[   h2:h3, mid_w:], lbl[   h3:h4, mid_w:], lbl[   h4:h5, mid_w:], lbl[   h5:, mid_w:]
        mskA_11, mskA_21, mskA_31, mskA_41, mskA_51, mskA_61 = msk[   :h1, :mid_w], msk[   h1:h2, :mid_w], msk[   h2:h3, :mid_w], msk[   h3:h4, :mid_w], msk[   h4:h5, :mid_w], msk[   h5:, :mid_w]
        mskA_12, mskA_22, mskA_32, mskA_42, mskA_52, mskA_62 = msk[   :h1, mid_w:], msk[   h1:h2, mid_w:], msk[   h2:h3, mid_w:], msk[   h3:h4, mid_w:], msk[   h4:h5, mid_w:], msk[   h5:, mid_w:]

        imgB_11, imgB_21, imgB_31, imgB_41, imgB_51, imgB_61 = img_aux[:, :h1, :mid_w], img_aux[:, h1:h2, :mid_w], img_aux[:, h2:h3, :mid_w], img_aux[:, h3:h4, :mid_w], img_aux[:, h4:h5, :mid_w], img_aux[:, h5:, :mid_w]
        imgB_12, imgB_22, imgB_32, imgB_42, imgB_52, imgB_62 = img_aux[:, :h1, mid_w:], img_aux[:, h1:h2, mid_w:], img_aux[:, h2:h3, mid_w:], img_aux[:, h3:h4, mid_w:], img_aux[:, h4:h5, mid_w:], img_aux[:, h5:, mid_w:]
        lblB_11, lblB_21, lblB_31, lblB_41, lblB_51, lblB_61 = lbl_aux[   :h1, :mid_w], lbl_aux[   h1:h2, :mid_w], lbl_aux[   h2:h3, :mid_w], lbl_aux[   h3:h4, :mid_w], lbl_aux[   h4:h5, :mid_w], lbl_aux[   h5:, :mid_w]
        lblB_12, lblB_22, lblB_32, lblB_42, lblB_52, lblB_62 = lbl_aux[   :h1, mid_w:], lbl_aux[   h1:h2, mid_w:], lbl_aux[   h2:h3, mid_w:], lbl_aux[   h3:h4, mid_w:], lbl_aux[   h4:h5, mid_w:], lbl_aux[   h5:, mid_w:]
        mskB_11, mskB_21, mskB_31, mskB_41, mskB_51, mskB_61 = msk_aux[   :h1, :mid_w], msk_aux[   h1:h2, :mid_w], msk_aux[   h2:h3, :mid_w], msk_aux[   h3:h4, :mid_w], msk_aux[   h4:h5, :mid_w], msk_aux[   h5:, :mid_w]
        mskB_12, mskB_22, mskB_32, mskB_42, mskB_52, mskB_62 = msk_aux[   :h1, mid_w:], msk_aux[   h1:h2, mid_w:], msk_aux[   h2:h3, mid_w:], msk_aux[   h3:h4, mid_w:], msk_aux[   h4:h5, mid_w:], msk_aux[   h5:, mid_w:]

        img_aux1_l, img_aux1_r = np.concatenate((imgA_11, imgB_21, imgA_31, imgB_41, imgA_51, imgB_61), axis=-2), np.concatenate((imgB_12, imgA_22, imgB_32, imgA_42, imgB_52, imgA_62), axis=-2)
        img_aux1 = np.concatenate((img_aux1_l, img_aux1_r), axis=-1)
        lbl_aux1_l, lbl_aux1_r = np.concatenate((lblA_11, lblB_21, lblA_31, lblB_41, lblA_51, lblB_61), axis=-2), np.concatenate((lblB_12, lblA_22, lblB_32, lblA_42, lblB_52, lblA_62), axis=-2)
        lbl_aux1 = np.concatenate((lbl_aux1_l, lbl_aux1_r), axis=-1)
        msk_aux1_l, msk_aux1_r = np.concatenate((mskA_11, mskB_21, mskA_31, mskB_41, mskA_51, mskB_61), axis=-2), np.concatenate((mskB_12, mskA_22, mskB_32, mskA_42, mskB_52, mskA_62), axis=-2)
        msk_aux1 = np.concatenate((msk_aux1_l, msk_aux1_r), axis=-1)

        img_aux2_l, img_aux2_r = np.concatenate((imgB_11, imgA_21, imgB_31, imgA_41, imgB_51, imgA_61), axis=-2), np.concatenate((imgA_12, imgB_22, imgA_32, imgB_42, imgA_52, imgB_62), axis=-2)
        img_aux2 = np.concatenate((img_aux2_l, img_aux2_r), axis=-1)
        lbl_aux2_l, lbl_aux2_r = np.concatenate((lblB_11, lblA_21, lblB_31, lblA_41, lblB_51, lblA_61), axis=-2), np.concatenate((lblA_12, lblB_22, lblA_32, lblB_42, lblA_52, lblB_62), axis=-2)
        lbl_aux2 = np.concatenate((lbl_aux2_l, lbl_aux2_r), axis=-1)
        msk_aux2_l, msk_aux2_r = np.concatenate((mskB_11, mskA_21, mskB_31, mskA_41, mskB_51, mskA_61), axis=-2), np.concatenate((mskA_12, mskB_22, mskA_32, mskB_42, mskA_52, mskB_62), axis=-2)
        msk_aux2 = np.concatenate((msk_aux2_l, msk_aux2_r), axis=-1)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col3row1(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 3)   # 2048/3 = 683
        w2 = 2 * w1                   # 2 * 683 = 1366
        
        imgA_1, imgA_2, imgA_3 = img[:, :, :w1], img[:, :, w1:w2], img[:, :, w2:]  # left1, middle1, right1
        lblA_1, lblA_2, lblA_3 = lbl[   :, :w1], lbl[   :, w1:w2], lbl[   :, w2:]  # left1, middle1, right1
        mskA_1, mskA_2, mskA_3 = msk[   :, :w1], msk[   :, w1:w2], msk[   :, w2:]  # left1, middle1, right1
        
        imgB_1, imgB_2, imgB_3 = img_aux[:, :, :w1], img_aux[:, :, w1:w2], img_aux[:, :, w2:]  # left2, middle2, right2
        lblB_1, lblB_2, lblB_3 = lbl_aux[   :, :w1], lbl_aux[   :, w1:w2], lbl_aux[   :, w2:]  # left2, middle2, right2
        mskB_1, mskB_2, mskB_3 = msk_aux[   :, :w1], msk_aux[   :, w1:w2], msk_aux[   :, w2:]  # left2, middle2, right2
        
        img_aux1 = np.concatenate((imgA_1, imgB_2, imgA_3), axis=-1)  # left1, middle2, right1
        lbl_aux1 = np.concatenate((lblA_1, lblB_2, lblA_3), axis=-1)  # left1, middle2, right1
        msk_aux1 = np.concatenate((mskA_1, mskB_2, mskA_3), axis=-1)  # left1, middle2, right1

        img_aux2 = np.concatenate((imgB_1, imgA_2, imgB_3), axis=-1)  # left2, middle1, right1
        lbl_aux2 = np.concatenate((lblB_1, lblA_2, lblB_3), axis=-1)  # left2, middle1, right1
        msk_aux2 = np.concatenate((mskB_1, mskA_2, mskB_3), axis=-1)  # left2, middle1, right1
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col3row2(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 3)   # 2048/3 = 683
        w2 = 2 * w1                   # 2*683  = 1366
        h1 = int(img.shape[-2] / 2)   # 64/2 = 32
        
        imgA_11, imgA_12, imgA_13 = img[:, :h1, :w1], img[:, :h1, w1:w2], img[:, :h1, w2:]
        imgA_21, imgA_22, imgA_23 = img[:, h1:, :w1], img[:, h1:, w1:w2], img[:, h1:, w2:]

        lblA_11, lblA_12, lblA_13 = lbl[   :h1, :w1], lbl[   :h1, w1:w2], lbl[   :h1, w2:]
        lblA_21, lblA_22, lblA_23 = lbl[   h1:, :w1], lbl[   h1:, w1:w2], lbl[   h1:, w2:]

        mskA_11, mskA_12, mskA_13 = msk[   :h1, :w1], msk[   :h1, w1:w2], msk[   :h1, w2:]
        mskA_21, mskA_22, mskA_23 = msk[   h1:, :w1], msk[   h1:, w1:w2], msk[   h1:, w2:]
        
        imgB_11, imgB_12, imgB_13 = img_aux[:, :h1, :w1], img_aux[:, :h1, w1:w2], img_aux[:, :h1, w2:]
        imgB_21, imgB_22, imgB_23 = img_aux[:, h1:, :w1], img_aux[:, h1:, w1:w2], img_aux[:, h1:, w2:]

        lblB_11, lblB_12, lblB_13 = lbl_aux[   :h1, :w1], lbl_aux[   :h1, w1:w2], lbl_aux[   :h1, w2:]
        lblB_21, lblB_22, lblB_23 = lbl_aux[   h1:, :w1], lbl_aux[   h1:, w1:w2], lbl_aux[   h1:, w2:]

        mskB_11, mskB_12, mskB_13 = msk_aux[   :h1, :w1], msk_aux[   :h1, w1:w2], msk_aux[   :h1, w2:]
        mskB_21, mskB_22, mskB_23 = msk_aux[   h1:, :w1], msk_aux[   h1:, w1:w2], msk_aux[   h1:, w2:]
        
        img_aux1_top = np.concatenate((imgA_11, imgB_12, imgA_13), axis=-1)
        img_aux1_bot = np.concatenate((imgB_21, imgA_22, imgB_23), axis=-1)
        img_aux1 = np.concatenate((img_aux1_top, img_aux1_bot), axis=-2)

        lbl_aux1_top = np.concatenate((lblA_11, lblB_12, lblA_13), axis=-1)
        lbl_aux1_bot = np.concatenate((lblB_21, lblA_22, lblB_23), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_top, lbl_aux1_bot), axis=-2)

        msk_aux1_top = np.concatenate((mskA_11, mskB_12, mskA_13), axis=-1)
        msk_aux1_bot = np.concatenate((mskB_21, mskA_22, mskB_23), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_top, msk_aux1_bot), axis=-2)

        img_aux2_top = np.concatenate((imgB_11, imgA_12, imgB_13), axis=-1)
        img_aux2_bot = np.concatenate((imgA_21, imgB_22, imgA_23), axis=-1)
        img_aux2 = np.concatenate((img_aux2_top, img_aux2_bot), axis=-2)

        lbl_aux2_top = np.concatenate((lblB_11, lblA_12, lblB_13), axis=-1)
        lbl_aux2_bot = np.concatenate((lblA_21, lblB_22, lblA_23), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_top, lbl_aux2_bot), axis=-2)

        msk_aux2_top = np.concatenate((mskB_11, mskA_12, mskB_13), axis=-1)
        msk_aux2_bot = np.concatenate((mskA_21, mskB_22, mskA_23), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_top, msk_aux2_bot), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col3row3(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 3)   # 2048/3 = 683
        w2 = 2 * w1                   # 2*683  = 1366
        h1 = int(img.shape[-2] / 3)   # 64/3 = 21
        h2 = 2 * h1                   # 2*21 = 42
        
        imgA_11, imgA_12, imgA_13 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:]
        imgA_21, imgA_22, imgA_23 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:]
        imgA_31, imgA_32, imgA_33 = img[:, h2:  , :w1], img[:, h2:  , w1:w2], img[:, h2:  , w2:]

        lblA_11, lblA_12, lblA_13 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:]
        lblA_21, lblA_22, lblA_23 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:]
        lblA_31, lblA_32, lblA_33 = lbl[   h2:  , :w1], lbl[   h2:  , w1:w2], lbl[   h2:  , w2:]

        mskA_11, mskA_12, mskA_13 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:]
        mskA_21, mskA_22, mskA_23 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:]
        mskA_31, mskA_32, mskA_33 = msk[   h2:  , :w1], msk[   h2:  , w1:w2], msk[   h2:  , w2:]
        
        imgB_11, imgB_12, imgB_13 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:]
        imgB_21, imgB_22, imgB_23 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:]
        imgB_31, imgB_32, imgB_33 = img_aux[:, h2:  , :w1], img_aux[:, h2:  , w1:w2], img_aux[:, h2:  , w2:]

        lblB_11, lblB_12, lblB_13 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:]
        lblB_21, lblB_22, lblB_23 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:]
        lblB_31, lblB_32, lblB_33 = lbl_aux[   h2:  , :w1], lbl_aux[   h2:  , w1:w2], lbl_aux[   h2:  , w2:]

        mskB_11, mskB_12, mskB_13 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:]
        mskB_21, mskB_22, mskB_23 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:]
        mskB_31, mskB_32, mskB_33 = msk_aux[   h2:  , :w1], msk_aux[   h2:  , w1:w2], msk_aux[   h2:  , w2:]
        
        img_aux1_top = np.concatenate((imgA_11, imgB_12, imgA_13), axis=-1)
        img_aux1_mid = np.concatenate((imgB_21, imgA_22, imgB_23), axis=-1)
        img_aux1_bot = np.concatenate((imgA_31, imgB_32, imgA_33), axis=-1)
        img_aux1 = np.concatenate((img_aux1_top, img_aux1_mid, img_aux1_bot), axis=-2)

        lbl_aux1_top = np.concatenate((lblA_11, lblB_12, lblA_13), axis=-1)
        lbl_aux1_mid = np.concatenate((lblB_21, lblA_22, lblB_23), axis=-1)
        lbl_aux1_bot = np.concatenate((lblA_31, lblB_32, lblA_33), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_top, lbl_aux1_mid, lbl_aux1_bot), axis=-2)

        msk_aux1_top = np.concatenate((mskA_11, mskB_12, mskA_13), axis=-1)
        msk_aux1_mid = np.concatenate((mskB_21, mskA_22, mskB_23), axis=-1)
        msk_aux1_bot = np.concatenate((mskA_31, mskB_32, mskA_33), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_top, msk_aux1_mid, msk_aux1_bot), axis=-2)

        img_aux2_top = np.concatenate((imgB_11, imgA_12, imgB_13), axis=-1)
        img_aux2_mid = np.concatenate((imgA_21, imgB_22, imgA_23), axis=-1)
        img_aux2_bot = np.concatenate((imgB_31, imgA_32, imgB_33), axis=-1)
        img_aux2 = np.concatenate((img_aux2_top, img_aux2_mid, img_aux2_bot), axis=-2)

        lbl_aux2_top = np.concatenate((lblB_11, lblA_12, lblB_13), axis=-1)
        lbl_aux2_mid = np.concatenate((lblA_21, lblB_22, lblA_23), axis=-1)
        lbl_aux2_bot = np.concatenate((lblB_31, lblA_32, lblB_33), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_top, lbl_aux2_mid, lbl_aux2_bot), axis=-2)

        msk_aux2_top = np.concatenate((mskB_11, mskA_12, mskB_13), axis=-1)
        msk_aux2_mid = np.concatenate((mskA_21, mskB_22, mskA_23), axis=-1)
        msk_aux2_bot = np.concatenate((mskB_31, mskA_32, mskB_33), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_top, msk_aux2_mid, msk_aux2_bot), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col3row4(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 3)   # 2048/3 = 683
        w2 = 2 * w1                   # 2*683  = 1366
        h1 = int(img.shape[-2] / 4)   # 64/4 = 16
        h2 = 2 * h1                   # 2*16 = 32
        h3 = 3 * h1                   # 3*16 = 48
        
        imgA_11, imgA_12, imgA_13 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:]
        imgA_21, imgA_22, imgA_23 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:]
        imgA_31, imgA_32, imgA_33 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:]
        imgA_41, imgA_42, imgA_43 = img[:, h3:  , :w1], img[:, h3:  , w1:w2], img[:, h3:  , w2:]

        lblA_11, lblA_12, lblA_13 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:]
        lblA_21, lblA_22, lblA_23 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:]
        lblA_31, lblA_32, lblA_33 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:]
        lblA_41, lblA_42, lblA_43 = lbl[   h3:  , :w1], lbl[   h3:  , w1:w2], lbl[   h3:  , w2:]

        mskA_11, mskA_12, mskA_13 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:]
        mskA_21, mskA_22, mskA_23 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:]
        mskA_31, mskA_32, mskA_33 = msk[   h2:h3, :w1], msk[   h2:h3, w1:w2], msk[   h2:h3, w2:]
        mskA_41, mskA_42, mskA_43 = msk[   h3:  , :w1], msk[   h3:  , w1:w2], msk[   h3:  , w2:]
        
        imgB_11, imgB_12, imgB_13 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:]
        imgB_21, imgB_22, imgB_23 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:]
        imgB_31, imgB_32, imgB_33 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:]
        imgB_41, imgB_42, imgB_43 = img_aux[:, h3:  , :w1], img_aux[:, h3:  , w1:w2], img_aux[:, h3:  , w2:]

        lblB_11, lblB_12, lblB_13 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:]
        lblB_21, lblB_22, lblB_23 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:]
        lblB_31, lblB_32, lblB_33 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:]
        lblB_41, lblB_42, lblB_43 = lbl_aux[   h3:  , :w1], lbl_aux[   h3:  , w1:w2], lbl_aux[   h3:  , w2:]

        mskB_11, mskB_12, mskB_13 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:]
        mskB_21, mskB_22, mskB_23 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:]
        mskB_31, mskB_32, mskB_33 = msk_aux[   h2:h3, :w1], msk_aux[   h2:h3, w1:w2], msk_aux[   h2:h3, w2:]
        mskB_41, mskB_42, mskB_43 = msk_aux[   h3:  , :w1], msk_aux[   h3:  , w1:w2], msk_aux[   h3:  , w2:]
        
        img_aux1_1 = np.concatenate((imgA_11, imgB_12, imgA_13), axis=-1)
        img_aux1_2 = np.concatenate((imgB_21, imgA_22, imgB_23), axis=-1)
        img_aux1_3 = np.concatenate((imgA_31, imgB_32, imgA_33), axis=-1)
        img_aux1_4 = np.concatenate((imgB_41, imgA_42, imgB_43), axis=-1)
        img_aux1 = np.concatenate((img_aux1_1, img_aux1_2, img_aux1_3, img_aux1_4), axis=-2)

        lbl_aux1_1 = np.concatenate((lblA_11, lblB_12, lblA_13), axis=-1)
        lbl_aux1_2 = np.concatenate((lblB_21, lblA_22, lblB_23), axis=-1)
        lbl_aux1_3 = np.concatenate((lblA_31, lblB_32, lblA_33), axis=-1)
        lbl_aux1_4 = np.concatenate((lblB_41, lblA_42, lblB_43), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_1, lbl_aux1_2, lbl_aux1_3, lbl_aux1_4), axis=-2)

        msk_aux1_1 = np.concatenate((mskA_11, mskB_12, mskA_13), axis=-1)
        msk_aux1_2 = np.concatenate((mskB_21, mskA_22, mskB_23), axis=-1)
        msk_aux1_3 = np.concatenate((mskA_31, mskB_32, mskA_33), axis=-1)
        msk_aux1_4 = np.concatenate((mskB_41, mskA_42, mskB_43), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_1, msk_aux1_2, msk_aux1_3, msk_aux1_4), axis=-2)

        img_aux2_1 = np.concatenate((imgB_11, imgA_12, imgB_13), axis=-1)
        img_aux2_2 = np.concatenate((imgA_21, imgB_22, imgA_23), axis=-1)
        img_aux2_3 = np.concatenate((imgB_31, imgA_32, imgB_33), axis=-1)
        img_aux2_4 = np.concatenate((imgA_41, imgB_42, imgA_43), axis=-1)
        img_aux2 = np.concatenate((img_aux2_1, img_aux2_2, img_aux2_3, img_aux2_4), axis=-2)

        lbl_aux2_1 = np.concatenate((lblB_11, lblA_12, lblB_13), axis=-1)
        lbl_aux2_2 = np.concatenate((lblA_21, lblB_22, lblA_23), axis=-1)
        lbl_aux2_3 = np.concatenate((lblB_31, lblA_32, lblB_33), axis=-1)
        lbl_aux2_4 = np.concatenate((lblA_41, lblB_42, lblA_43), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_1, lbl_aux2_2, lbl_aux2_3, lbl_aux2_4), axis=-2)

        msk_aux2_1 = np.concatenate((mskB_11, mskA_12, mskB_13), axis=-1)
        msk_aux2_2 = np.concatenate((mskA_21, mskB_22, mskA_23), axis=-1)
        msk_aux2_3 = np.concatenate((mskB_31, mskA_32, mskB_33), axis=-1)
        msk_aux2_4 = np.concatenate((mskA_41, mskB_42, mskA_43), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_1, msk_aux2_2, msk_aux2_3, msk_aux2_4), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col3row5(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 3)   # 2048/3 = 683
        w2 = 2 * w1                   # 2*683  = 1366
        h1 = int(img.shape[-2] / 5)   # 64/5 = 12
        h2 = 2 * h1                   # 2*12 = 24
        h3 = 3 * h1                   # 3*12 = 36
        h4 = 4 * h1                   # 4*12 = 48
        
        imgA_11, imgA_12, imgA_13 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:]
        imgA_21, imgA_22, imgA_23 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:]
        imgA_31, imgA_32, imgA_33 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:]
        imgA_41, imgA_42, imgA_43 = img[:, h3:h4, :w1], img[:, h3:h4, w1:w2], img[:, h3:h4, w2:]
        imgA_51, imgA_52, imgA_53 = img[:, h4:  , :w1], img[:, h4:  , w1:w2], img[:, h4:  , w2:]

        lblA_11, lblA_12, lblA_13 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:]
        lblA_21, lblA_22, lblA_23 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:]
        lblA_31, lblA_32, lblA_33 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:]
        lblA_41, lblA_42, lblA_43 = lbl[   h3:h4, :w1], lbl[   h3:h4, w1:w2], lbl[   h3:h4, w2:]
        lblA_51, lblA_52, lblA_53 = lbl[   h4:  , :w1], lbl[   h4:  , w1:w2], lbl[   h4:  , w2:]

        mskA_11, mskA_12, mskA_13 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:]
        mskA_21, mskA_22, mskA_23 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:]
        mskA_31, mskA_32, mskA_33 = msk[   h2:h3, :w1], msk[   h2:h3, w1:w2], msk[   h2:h3, w2:]
        mskA_41, mskA_42, mskA_43 = msk[   h3:h4, :w1], msk[   h3:h4, w1:w2], msk[   h3:h4, w2:]
        mskA_51, mskA_52, mskA_53 = msk[   h4:  , :w1], msk[   h4:  , w1:w2], msk[   h4:  , w2:]
        
        imgB_11, imgB_12, imgB_13 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:]
        imgB_21, imgB_22, imgB_23 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:]
        imgB_31, imgB_32, imgB_33 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:]
        imgB_41, imgB_42, imgB_43 = img_aux[:, h3:h4, :w1], img_aux[:, h3:h4, w1:w2], img_aux[:, h3:h4, w2:]
        imgB_51, imgB_52, imgB_53 = img_aux[:, h4:  , :w1], img_aux[:, h4:  , w1:w2], img_aux[:, h4:  , w2:]

        lblB_11, lblB_12, lblB_13 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:]
        lblB_21, lblB_22, lblB_23 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:]
        lblB_31, lblB_32, lblB_33 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:]
        lblB_41, lblB_42, lblB_43 = lbl_aux[   h3:h4, :w1], lbl_aux[   h3:h4, w1:w2], lbl_aux[   h3:h4, w2:]
        lblB_51, lblB_52, lblB_53 = lbl_aux[   h4:  , :w1], lbl_aux[   h4:  , w1:w2], lbl_aux[   h4:  , w2:]

        mskB_11, mskB_12, mskB_13 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:]
        mskB_21, mskB_22, mskB_23 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:]
        mskB_31, mskB_32, mskB_33 = msk_aux[   h2:h3, :w1], msk_aux[   h2:h3, w1:w2], msk_aux[   h2:h3, w2:]
        mskB_41, mskB_42, mskB_43 = msk_aux[   h3:h4, :w1], msk_aux[   h3:h4, w1:w2], msk_aux[   h3:h4, w2:]
        mskB_51, mskB_52, mskB_53 = msk_aux[   h4:  , :w1], msk_aux[   h4:  , w1:w2], msk_aux[   h4:  , w2:]
        
        img_aux1_1 = np.concatenate((imgA_11, imgB_12, imgA_13), axis=-1)
        img_aux1_2 = np.concatenate((imgB_21, imgA_22, imgB_23), axis=-1)
        img_aux1_3 = np.concatenate((imgA_31, imgB_32, imgA_33), axis=-1)
        img_aux1_4 = np.concatenate((imgB_41, imgA_42, imgB_43), axis=-1)
        img_aux1_5 = np.concatenate((imgA_51, imgB_52, imgA_53), axis=-1)
        img_aux1 = np.concatenate((img_aux1_1, img_aux1_2, img_aux1_3, img_aux1_4, img_aux1_5), axis=-2)

        lbl_aux1_1 = np.concatenate((lblA_11, lblB_12, lblA_13), axis=-1)
        lbl_aux1_2 = np.concatenate((lblB_21, lblA_22, lblB_23), axis=-1)
        lbl_aux1_3 = np.concatenate((lblA_31, lblB_32, lblA_33), axis=-1)
        lbl_aux1_4 = np.concatenate((lblB_41, lblA_42, lblB_43), axis=-1)
        lbl_aux1_5 = np.concatenate((lblA_51, lblB_52, lblA_53), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_1, lbl_aux1_2, lbl_aux1_3, lbl_aux1_4, lbl_aux1_5), axis=-2)

        msk_aux1_1 = np.concatenate((mskA_11, mskB_12, mskA_13), axis=-1)
        msk_aux1_2 = np.concatenate((mskB_21, mskA_22, mskB_23), axis=-1)
        msk_aux1_3 = np.concatenate((mskA_31, mskB_32, mskA_33), axis=-1)
        msk_aux1_4 = np.concatenate((mskB_41, mskA_42, mskB_43), axis=-1)
        msk_aux1_5 = np.concatenate((mskA_51, mskB_52, mskA_53), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_1, msk_aux1_2, msk_aux1_3, msk_aux1_4, msk_aux1_5), axis=-2)

        img_aux2_1 = np.concatenate((imgB_11, imgA_12, imgB_13), axis=-1)
        img_aux2_2 = np.concatenate((imgA_21, imgB_22, imgA_23), axis=-1)
        img_aux2_3 = np.concatenate((imgB_31, imgA_32, imgB_33), axis=-1)
        img_aux2_4 = np.concatenate((imgA_41, imgB_42, imgA_43), axis=-1)
        img_aux2_5 = np.concatenate((imgB_51, imgA_52, imgB_53), axis=-1)
        img_aux2 = np.concatenate((img_aux2_1, img_aux2_2, img_aux2_3, img_aux2_4, img_aux2_5), axis=-2)

        lbl_aux2_1 = np.concatenate((lblB_11, lblA_12, lblB_13), axis=-1)
        lbl_aux2_2 = np.concatenate((lblA_21, lblB_22, lblA_23), axis=-1)
        lbl_aux2_3 = np.concatenate((lblB_31, lblA_32, lblB_33), axis=-1)
        lbl_aux2_4 = np.concatenate((lblA_41, lblB_42, lblA_43), axis=-1)
        lbl_aux2_5 = np.concatenate((lblB_51, lblA_52, lblB_53), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_1, lbl_aux2_2, lbl_aux2_3, lbl_aux2_4, lbl_aux2_5), axis=-2)

        msk_aux2_1 = np.concatenate((mskB_11, mskA_12, mskB_13), axis=-1)
        msk_aux2_2 = np.concatenate((mskA_21, mskB_22, mskA_23), axis=-1)
        msk_aux2_3 = np.concatenate((mskB_31, mskA_32, mskB_33), axis=-1)
        msk_aux2_4 = np.concatenate((mskA_41, mskB_42, mskA_43), axis=-1)
        msk_aux2_5 = np.concatenate((mskB_51, mskA_52, mskB_53), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_1, msk_aux2_2, msk_aux2_3, msk_aux2_4, msk_aux2_5), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col3row6(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 3)   # 2048/3 = 683
        w2 = 2 * w1                   # 2*683  = 1366
        h1 = int(img.shape[-2] / 6)   # 64/6 = 10
        h2 = 2 * h1                   # 2*10 = 20
        h3 = 3 * h1                   # 3*10 = 30
        h4 = 4 * h1                   # 4*10 = 40
        h5 = 5 * h1                   # 5*10 = 50
        
        imgA_11, imgA_12, imgA_13 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:]
        imgA_21, imgA_22, imgA_23 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:]
        imgA_31, imgA_32, imgA_33 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:]
        imgA_41, imgA_42, imgA_43 = img[:, h3:h4, :w1], img[:, h3:h4, w1:w2], img[:, h3:h4, w2:]
        imgA_51, imgA_52, imgA_53 = img[:, h4:h5, :w1], img[:, h4:h5, w1:w2], img[:, h4:h5, w2:]
        imgA_61, imgA_62, imgA_63 = img[:, h5:  , :w1], img[:, h5:  , w1:w2], img[:, h5:  , w2:]

        lblA_11, lblA_12, lblA_13 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:]
        lblA_21, lblA_22, lblA_23 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:]
        lblA_31, lblA_32, lblA_33 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:]
        lblA_41, lblA_42, lblA_43 = lbl[   h3:h4, :w1], lbl[   h3:h4, w1:w2], lbl[   h3:h4, w2:]
        lblA_51, lblA_52, lblA_53 = lbl[   h4:h5, :w1], lbl[   h4:h5, w1:w2], lbl[   h4:h5, w2:]
        lblA_61, lblA_62, lblA_63 = lbl[   h5:  , :w1], lbl[   h5:  , w1:w2], lbl[   h5:  , w2:]

        mskA_11, mskA_12, mskA_13 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:]
        mskA_21, mskA_22, mskA_23 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:]
        mskA_31, mskA_32, mskA_33 = msk[   h2:h3, :w1], msk[   h2:h3, w1:w2], msk[   h2:h3, w2:]
        mskA_41, mskA_42, mskA_43 = msk[   h3:h4, :w1], msk[   h3:h4, w1:w2], msk[   h3:h4, w2:]
        mskA_51, mskA_52, mskA_53 = msk[   h4:h5, :w1], msk[   h4:h5, w1:w2], msk[   h4:h5, w2:]
        mskA_61, mskA_62, mskA_63 = msk[   h5:  , :w1], msk[   h5:  , w1:w2], msk[   h5:  , w2:]
        
        imgB_11, imgB_12, imgB_13 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:]
        imgB_21, imgB_22, imgB_23 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:]
        imgB_31, imgB_32, imgB_33 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:]
        imgB_41, imgB_42, imgB_43 = img_aux[:, h3:h4, :w1], img_aux[:, h3:h4, w1:w2], img_aux[:, h3:h4, w2:]
        imgB_51, imgB_52, imgB_53 = img_aux[:, h4:h5, :w1], img_aux[:, h4:h5, w1:w2], img_aux[:, h4:h5, w2:]
        imgB_61, imgB_62, imgB_63 = img_aux[:, h5:  , :w1], img_aux[:, h5:  , w1:w2], img_aux[:, h5:  , w2:]

        lblB_11, lblB_12, lblB_13 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:]
        lblB_21, lblB_22, lblB_23 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:]
        lblB_31, lblB_32, lblB_33 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:]
        lblB_41, lblB_42, lblB_43 = lbl_aux[   h3:h4, :w1], lbl_aux[   h3:h4, w1:w2], lbl_aux[   h3:h4, w2:]
        lblB_51, lblB_52, lblB_53 = lbl_aux[   h4:h5, :w1], lbl_aux[   h4:h5, w1:w2], lbl_aux[   h4:h5, w2:]
        lblB_61, lblB_62, lblB_63 = lbl_aux[   h5:  , :w1], lbl_aux[   h5:  , w1:w2], lbl_aux[   h5:  , w2:]

        mskB_11, mskB_12, mskB_13 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:]
        mskB_21, mskB_22, mskB_23 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:]
        mskB_31, mskB_32, mskB_33 = msk_aux[   h2:h3, :w1], msk_aux[   h2:h3, w1:w2], msk_aux[   h2:h3, w2:]
        mskB_41, mskB_42, mskB_43 = msk_aux[   h3:h4, :w1], msk_aux[   h3:h4, w1:w2], msk_aux[   h3:h4, w2:]
        mskB_51, mskB_52, mskB_53 = msk_aux[   h4:h5, :w1], msk_aux[   h4:h5, w1:w2], msk_aux[   h4:h5, w2:]
        mskB_61, mskB_62, mskB_63 = msk_aux[   h5:  , :w1], msk_aux[   h5:  , w1:w2], msk_aux[   h5:  , w2:]
        
        img_aux1_1 = np.concatenate((imgA_11, imgB_12, imgA_13), axis=-1)
        img_aux1_2 = np.concatenate((imgB_21, imgA_22, imgB_23), axis=-1)
        img_aux1_3 = np.concatenate((imgA_31, imgB_32, imgA_33), axis=-1)
        img_aux1_4 = np.concatenate((imgB_41, imgA_42, imgB_43), axis=-1)
        img_aux1_5 = np.concatenate((imgA_51, imgB_52, imgA_53), axis=-1)
        img_aux1_6 = np.concatenate((imgB_61, imgA_62, imgB_63), axis=-1)
        img_aux1 = np.concatenate((img_aux1_1, img_aux1_2, img_aux1_3, img_aux1_4, img_aux1_5, img_aux1_6), axis=-2)

        lbl_aux1_1 = np.concatenate((lblA_11, lblB_12, lblA_13), axis=-1)
        lbl_aux1_2 = np.concatenate((lblB_21, lblA_22, lblB_23), axis=-1)
        lbl_aux1_3 = np.concatenate((lblA_31, lblB_32, lblA_33), axis=-1)
        lbl_aux1_4 = np.concatenate((lblB_41, lblA_42, lblB_43), axis=-1)
        lbl_aux1_5 = np.concatenate((lblA_51, lblB_52, lblA_53), axis=-1)
        lbl_aux1_6 = np.concatenate((lblB_61, lblA_62, lblB_63), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_1, lbl_aux1_2, lbl_aux1_3, lbl_aux1_4, lbl_aux1_5, lbl_aux1_6), axis=-2)

        msk_aux1_1 = np.concatenate((mskA_11, mskB_12, mskA_13), axis=-1)
        msk_aux1_2 = np.concatenate((mskB_21, mskA_22, mskB_23), axis=-1)
        msk_aux1_3 = np.concatenate((mskA_31, mskB_32, mskA_33), axis=-1)
        msk_aux1_4 = np.concatenate((mskB_41, mskA_42, mskB_43), axis=-1)
        msk_aux1_5 = np.concatenate((mskA_51, mskB_52, mskA_53), axis=-1)
        msk_aux1_6 = np.concatenate((mskB_61, mskA_62, mskB_63), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_1, msk_aux1_2, msk_aux1_3, msk_aux1_4, msk_aux1_5, msk_aux1_6), axis=-2)

        img_aux2_1 = np.concatenate((imgB_11, imgA_12, imgB_13), axis=-1)
        img_aux2_2 = np.concatenate((imgA_21, imgB_22, imgA_23), axis=-1)
        img_aux2_3 = np.concatenate((imgB_31, imgA_32, imgB_33), axis=-1)
        img_aux2_4 = np.concatenate((imgA_41, imgB_42, imgA_43), axis=-1)
        img_aux2_5 = np.concatenate((imgB_51, imgA_52, imgB_53), axis=-1)
        img_aux2_6 = np.concatenate((imgA_61, imgB_62, imgA_63), axis=-1)
        img_aux2 = np.concatenate((img_aux2_1, img_aux2_2, img_aux2_3, img_aux2_4, img_aux2_5, img_aux2_6), axis=-2)

        lbl_aux2_1 = np.concatenate((lblB_11, lblA_12, lblB_13), axis=-1)
        lbl_aux2_2 = np.concatenate((lblA_21, lblB_22, lblA_23), axis=-1)
        lbl_aux2_3 = np.concatenate((lblB_31, lblA_32, lblB_33), axis=-1)
        lbl_aux2_4 = np.concatenate((lblA_41, lblB_42, lblA_43), axis=-1)
        lbl_aux2_5 = np.concatenate((lblB_51, lblA_52, lblB_53), axis=-1)
        lbl_aux2_6 = np.concatenate((lblA_61, lblB_62, lblA_63), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_1, lbl_aux2_2, lbl_aux2_3, lbl_aux2_4, lbl_aux2_5, lbl_aux2_6), axis=-2)

        msk_aux2_1 = np.concatenate((mskB_11, mskA_12, mskB_13), axis=-1)
        msk_aux2_2 = np.concatenate((mskA_21, mskB_22, mskA_23), axis=-1)
        msk_aux2_3 = np.concatenate((mskB_31, mskA_32, mskB_33), axis=-1)
        msk_aux2_4 = np.concatenate((mskA_41, mskB_42, mskA_43), axis=-1)
        msk_aux2_5 = np.concatenate((mskB_51, mskA_52, mskB_53), axis=-1)
        msk_aux2_6 = np.concatenate((mskA_61, mskB_62, mskA_63), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_1, msk_aux2_2, msk_aux2_3, msk_aux2_4, msk_aux2_5, msk_aux2_6), axis=-2)
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col4row1(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 4)   # 2048/4 = 512
        w2 = int(img.shape[-1] / 2)   # 2048/2 = 1024
        w3 = 3 * w1                   # 3 * 512 = 1536
        
        imgA_1, imgA_2, imgA_3, imgA_4 = img[:, :, :w1], img[:, :, w1:w2], img[:, :, w2:w3], img[:, :, w3:]  # 1 - 2 - 3 - 4
        lblA_1, lblA_2, lblA_3, lblA_4 = lbl[   :, :w1], lbl[   :, w1:w2], lbl[   :, w2:w3], lbl[   :, w3:]  # 1 - 2 - 3 - 4
        mskA_1, mskA_2, mskA_3, mskA_4 = msk[   :, :w1], msk[   :, w1:w2], msk[   :, w2:w3], msk[   :, w3:]  # 1 - 2 - 3 - 4
        
        imgB_1, imgB_2, imgB_3, imgB_4 = img_aux[:, :, :w1], img_aux[:, :, w1:w2], img_aux[:, :, w2:w3], img_aux[:, :, w3:]  # 1 - 2 - 3 - 4
        lblB_1, lblB_2, lblB_3, lblB_4 = lbl_aux[   :, :w1], lbl_aux[   :, w1:w2], lbl_aux[   :, w2:w3], lbl_aux[   :, w3:]  # 1 - 2 - 3 - 4
        mskB_1, mskB_2, mskB_3, mskB_4 = msk_aux[   :, :w1], msk_aux[   :, w1:w2], msk_aux[   :, w2:w3], msk_aux[   :, w3:]  # 1 - 2 - 3 - 4
        
        img_aux1 = np.concatenate((imgA_1, imgB_2, imgA_3, imgB_4), axis=-1)  # 1 - 2 - 3 - 4
        lbl_aux1 = np.concatenate((lblA_1, lblB_2, lblA_3, lblB_4), axis=-1)  # 1 - 2 - 3 - 4
        msk_aux1 = np.concatenate((mskA_1, mskB_2, mskA_3, mskB_4), axis=-1)  # 1 - 2 - 3 - 4

        img_aux2 = np.concatenate((imgB_1, imgA_2, imgB_3, imgA_4), axis=-1)  # 1 - 2 - 3 - 4
        lbl_aux2 = np.concatenate((lblB_1, lblA_2, lblB_3, lblA_4), axis=-1)  # 1 - 2 - 3 - 4
        msk_aux2 = np.concatenate((mskB_1, mskA_2, mskB_3, mskA_4), axis=-1)  # 1 - 2 - 3 - 4
        
        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col4row2(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 4)   # 2048/4 = 512
        w2 = int(img.shape[-1] / 2)   # 2048/2 = 1024
        w3 = 3 * w1                   # 3 * 512 = 1536
        h1 = int(img.shape[-2] / 2)   # 64/2 = 32

        imgA_11, imgA_12, imgA_13, imgA_14 = img[:, :h1, :w1], img[:, :h1, w1:w2], img[:, :h1, w2:w3], img[:, :h1, w3:]
        imgA_21, imgA_22, imgA_23, imgA_24 = img[:, h1:, :w1], img[:, h1:, w1:w2], img[:, h1:, w2:w3], img[:, h1:, w3:]

        lblA_11, lblA_12, lblA_13, lblA_14 = lbl[   :h1, :w1], lbl[   :h1, w1:w2], lbl[   :h1, w2:w3], lbl[   :h1, w3:]
        lblA_21, lblA_22, lblA_23, lblA_24 = lbl[   h1:, :w1], lbl[   h1:, w1:w2], lbl[   h1:, w2:w3], lbl[   h1:, w3:]

        mskA_11, mskA_12, mskA_13, mskA_14 = msk[   :h1, :w1], msk[   :h1, w1:w2], msk[   :h1, w2:w3], msk[   :h1, w3:]
        mskA_21, mskA_22, mskA_23, mskA_24 = msk[   h1:, :w1], msk[   h1:, w1:w2], msk[   h1:, w2:w3], msk[   h1:, w3:]

        imgB_11, imgB_12, imgB_13, imgB_14 = img_aux[:, :h1, :w1], img_aux[:, :h1, w1:w2], img_aux[:, :h1, w2:w3], img_aux[:, :h1, w3:]
        imgB_21, imgB_22, imgB_23, imgB_24 = img_aux[:, h1:, :w1], img_aux[:, h1:, w1:w2], img_aux[:, h1:, w2:w3], img_aux[:, h1:, w3:]

        lblB_11, lblB_12, lblB_13, lblB_14 = lbl_aux[   :h1, :w1], lbl_aux[   :h1, w1:w2], lbl_aux[   :h1, w2:w3], lbl_aux[   :h1, w3:]
        lblB_21, lblB_22, lblB_23, lblB_24 = lbl_aux[   h1:, :w1], lbl_aux[   h1:, w1:w2], lbl_aux[   h1:, w2:w3], lbl_aux[   h1:, w3:]

        mskB_11, mskB_12, mskB_13, mskB_14 = msk_aux[   :h1, :w1], msk_aux[   :h1, w1:w2], msk_aux[   :h1, w2:w3], msk_aux[   :h1, w3:]
        mskB_21, mskB_22, mskB_23, mskB_24 = msk_aux[   h1:, :w1], msk_aux[   h1:, w1:w2], msk_aux[   h1:, w2:w3], msk_aux[   h1:, w3:]
        
        img_aux1_top = np.concatenate((imgA_11, imgB_12, imgA_13, imgB_14), axis=-1)
        img_aux1_bot = np.concatenate((imgB_21, imgA_22, imgB_23, imgA_24), axis=-1)
        img_aux1 = np.concatenate((img_aux1_top, img_aux1_bot), axis=-2)

        lbl_aux1_top = np.concatenate((lblA_11, lblB_12, lblA_13, lblB_14), axis=-1)
        lbl_aux1_bot = np.concatenate((lblB_21, lblA_22, lblB_23, lblA_24), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_top, lbl_aux1_bot), axis=-2)

        msk_aux1_top = np.concatenate((mskA_11, mskB_12, mskA_13, mskB_14), axis=-1)
        msk_aux1_bot = np.concatenate((mskB_21, mskA_22, mskB_23, mskA_24), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_top, msk_aux1_bot), axis=-2)

        img_aux2_top = np.concatenate((imgB_11, imgA_12, imgB_13, imgA_14), axis=-1)
        img_aux2_bot = np.concatenate((imgA_21, imgB_22, imgA_23, imgB_24), axis=-1)
        img_aux2 = np.concatenate((img_aux2_top, img_aux2_bot), axis=-2)

        lbl_aux2_top = np.concatenate((lblB_11, lblA_12, lblB_13, lblA_14), axis=-1)
        lbl_aux2_bot = np.concatenate((lblA_21, lblB_22, lblA_23, lblB_24), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_top, lbl_aux2_bot), axis=-2)

        msk_aux2_top = np.concatenate((mskB_11, mskA_12, mskB_13, mskA_14), axis=-1)
        msk_aux2_bot = np.concatenate((mskA_21, mskB_22, mskA_23, mskB_24), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_top, msk_aux2_bot), axis=-2)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col4row3(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 4)   # 2048/4 = 512
        w2 = int(img.shape[-1] / 2)   # 2048/2 = 1024
        w3 = 3 * w1                   # 3 * 512 = 1536
        h1 = int(img.shape[-2] / 3)   # 64/3 = 21
        h2 = 2 * h1                   # 2*21 = 42

        imgA_11, imgA_12, imgA_13, imgA_14 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:w3], img[:,   :h1, w3:]
        imgA_21, imgA_22, imgA_23, imgA_24 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:w3], img[:, h1:h2, w3:]
        imgA_31, imgA_32, imgA_33, imgA_34 = img[:, h2:  , :w1], img[:, h2:  , w1:w2], img[:, h2:  , w2:w3], img[:, h2:  , w3:]

        lblA_11, lblA_12, lblA_13, lblA_14 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:w3], lbl[     :h1, w3:]
        lblA_21, lblA_22, lblA_23, lblA_24 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:w3], lbl[   h1:h2, w3:]
        lblA_31, lblA_32, lblA_33, lblA_34 = lbl[   h2:  , :w1], lbl[   h2:  , w1:w2], lbl[   h2:  , w2:w3], lbl[   h2:  , w3:]

        mskA_11, mskA_12, mskA_13, mskA_14 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:w3], msk[     :h1, w3:]
        mskA_21, mskA_22, mskA_23, mskA_24 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:w3], msk[   h1:h2, w3:]
        mskA_31, mskA_32, mskA_33, mskA_34 = msk[   h2:  , :w1], msk[   h2:  , w1:w2], msk[   h2:  , w2:w3], msk[   h2:  , w3:]

        imgB_11, imgB_12, imgB_13, imgB_14 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:w3], img_aux[:,   :h1, w3:]
        imgB_21, imgB_22, imgB_23, imgB_24 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:w3], img_aux[:, h1:h2, w3:]
        imgB_31, imgB_32, imgB_33, imgB_34 = img_aux[:, h2:  , :w1], img_aux[:, h2:  , w1:w2], img_aux[:, h2:  , w2:w3], img_aux[:, h2:  , w3:]

        lblB_11, lblB_12, lblB_13, lblB_14 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:w3], lbl_aux[     :h1, w3:]
        lblB_21, lblB_22, lblB_23, lblB_24 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:w3], lbl_aux[   h1:h2, w3:]
        lblB_31, lblB_32, lblB_33, lblB_34 = lbl_aux[   h2:  , :w1], lbl_aux[   h2:  , w1:w2], lbl_aux[   h2:  , w2:w3], lbl_aux[   h2:  , w3:]

        mskB_11, mskB_12, mskB_13, mskB_14 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:w3], msk_aux[     :h1, w3:]
        mskB_21, mskB_22, mskB_23, mskB_24 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:w3], msk_aux[   h1:h2, w3:]
        mskB_31, mskB_32, mskB_33, mskB_34 = msk_aux[   h2:  , :w1], msk_aux[   h2:  , w1:w2], msk_aux[   h2:  , w2:w3], msk_aux[   h2:  , w3:]
        
        img_aux1_top = np.concatenate((imgA_11, imgB_12, imgA_13, imgB_14), axis=-1)
        img_aux1_mid = np.concatenate((imgB_21, imgA_22, imgB_23, imgA_24), axis=-1)
        img_aux1_bot = np.concatenate((imgA_31, imgB_32, imgA_33, imgB_34), axis=-1)
        img_aux1 = np.concatenate((img_aux1_top, img_aux1_mid, img_aux1_bot), axis=-2)

        lbl_aux1_top = np.concatenate((lblA_11, lblB_12, lblA_13, lblB_14), axis=-1)
        lbl_aux1_mid = np.concatenate((lblB_21, lblA_22, lblB_23, lblA_24), axis=-1)
        lbl_aux1_bot = np.concatenate((lblA_31, lblB_32, lblA_33, lblB_34), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_top, lbl_aux1_mid, lbl_aux1_bot), axis=-2)

        msk_aux1_top = np.concatenate((mskA_11, mskB_12, mskA_13, mskB_14), axis=-1)
        msk_aux1_mid = np.concatenate((mskB_21, mskA_22, mskB_23, mskA_24), axis=-1)
        msk_aux1_bot = np.concatenate((mskA_31, mskB_32, mskA_33, mskB_34), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_top, msk_aux1_mid, msk_aux1_bot), axis=-2)

        img_aux2_top = np.concatenate((imgB_11, imgA_12, imgB_13, imgA_14), axis=-1)
        img_aux2_mid = np.concatenate((imgA_21, imgB_22, imgA_23, imgB_24), axis=-1)
        img_aux2_bot = np.concatenate((imgB_31, imgA_32, imgB_33, imgA_34), axis=-1)
        img_aux2 = np.concatenate((img_aux2_top, img_aux2_mid, img_aux2_bot), axis=-2)

        lbl_aux2_top = np.concatenate((lblB_11, lblA_12, lblB_13, lblA_14), axis=-1)
        lbl_aux2_mid = np.concatenate((lblA_21, lblB_22, lblA_23, lblB_24), axis=-1)
        lbl_aux2_bot = np.concatenate((lblB_31, lblA_32, lblB_33, lblA_34), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_top, lbl_aux2_mid, lbl_aux2_bot), axis=-2)

        msk_aux2_top = np.concatenate((mskB_11, mskA_12, mskB_13, mskA_14), axis=-1)
        msk_aux2_mid = np.concatenate((mskA_21, mskB_22, mskA_23, mskB_24), axis=-1)
        msk_aux2_bot = np.concatenate((mskB_31, mskA_32, mskB_33, mskA_34), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_top, msk_aux2_mid, msk_aux2_bot), axis=-2)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col4row4(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 4)   # 2048/4 = 512
        w2 = int(img.shape[-1] / 2)   # 2048/2 = 1024
        w3 = 3 * w1                   # 3 * 512 = 1536
        h1 = int(img.shape[-2] / 4)   # 64/4 = 16
        h2 = 2 * h1                   # 2*16 = 32
        h3 = 3 * h1                   # 3*16 = 48

        imgA_11, imgA_12, imgA_13, imgA_14 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:w3], img[:,   :h1, w3:]
        imgA_21, imgA_22, imgA_23, imgA_24 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:w3], img[:, h1:h2, w3:]
        imgA_31, imgA_32, imgA_33, imgA_34 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:w3], img[:, h2:h3, w3:]
        imgA_41, imgA_42, imgA_43, imgA_44 = img[:, h3:  , :w1], img[:, h3:  , w1:w2], img[:, h3:  , w2:w3], img[:, h3:  , w3:]

        lblA_11, lblA_12, lblA_13, lblA_14 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:w3], lbl[     :h1, w3:]
        lblA_21, lblA_22, lblA_23, lblA_24 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:w3], lbl[   h1:h2, w3:]
        lblA_31, lblA_32, lblA_33, lblA_34 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:w3], lbl[   h2:h3, w3:]
        lblA_41, lblA_42, lblA_43, lblA_44 = lbl[   h3:  , :w1], lbl[   h3:  , w1:w2], lbl[   h3:  , w2:w3], lbl[   h3:  , w3:]

        mskA_11, mskA_12, mskA_13, mskA_14 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:w3], msk[     :h1, w3:]
        mskA_21, mskA_22, mskA_23, mskA_24 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:w3], msk[   h1:h2, w3:]
        mskA_31, mskA_32, mskA_33, mskA_34 = msk[   h2:h3, :w1], msk[   h2:h3, w1:w2], msk[   h2:h3, w2:w3], msk[   h2:h3, w3:]
        mskA_41, mskA_42, mskA_43, mskA_44 = msk[   h3:  , :w1], msk[   h3:  , w1:w2], msk[   h3:  , w2:w3], msk[   h3:  , w3:]

        imgB_11, imgB_12, imgB_13, imgB_14 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:w3], img_aux[:,   :h1, w3:]
        imgB_21, imgB_22, imgB_23, imgB_24 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:w3], img_aux[:, h1:h2, w3:]
        imgB_31, imgB_32, imgB_33, imgB_34 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:w3], img_aux[:, h2:h3, w3:]
        imgB_41, imgB_42, imgB_43, imgB_44 = img_aux[:, h3:  , :w1], img_aux[:, h3:  , w1:w2], img_aux[:, h3:  , w2:w3], img_aux[:, h3:  , w3:]

        lblB_11, lblB_12, lblB_13, lblB_14 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:w3], lbl_aux[     :h1, w3:]
        lblB_21, lblB_22, lblB_23, lblB_24 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:w3], lbl_aux[   h1:h2, w3:]
        lblB_31, lblB_32, lblB_33, lblB_34 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:w3], lbl_aux[   h2:h3, w3:]
        lblB_41, lblB_42, lblB_43, lblB_44 = lbl_aux[   h3:  , :w1], lbl_aux[   h3:  , w1:w2], lbl_aux[   h3:  , w2:w3], lbl_aux[   h3:  , w3:]

        mskB_11, mskB_12, mskB_13, mskB_14 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:w3], msk_aux[     :h1, w3:]
        mskB_21, mskB_22, mskB_23, mskB_24 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:w3], msk_aux[   h1:h2, w3:]
        mskB_31, mskB_32, mskB_33, mskB_34 = msk_aux[   h2:h3, :w1], msk_aux[   h2:h3, w1:w2], msk_aux[   h2:h3, w2:w3], msk_aux[   h2:h3, w3:]
        mskB_41, mskB_42, mskB_43, mskB_44 = msk_aux[   h3:  , :w1], msk_aux[   h3:  , w1:w2], msk_aux[   h3:  , w2:w3], msk_aux[   h3:  , w3:]
        
        img_aux1_1 = np.concatenate((imgA_11, imgB_12, imgA_13, imgB_14), axis=-1)
        img_aux1_2 = np.concatenate((imgB_21, imgA_22, imgB_23, imgA_24), axis=-1)
        img_aux1_3 = np.concatenate((imgA_31, imgB_32, imgA_33, imgB_34), axis=-1)
        img_aux1_4 = np.concatenate((imgB_41, imgA_42, imgB_43, imgA_44), axis=-1)
        img_aux1 = np.concatenate((img_aux1_1, img_aux1_2, img_aux1_3, img_aux1_4), axis=-2)

        lbl_aux1_1 = np.concatenate((lblA_11, lblB_12, lblA_13, lblB_14), axis=-1)
        lbl_aux1_2 = np.concatenate((lblB_21, lblA_22, lblB_23, lblA_24), axis=-1)
        lbl_aux1_3 = np.concatenate((lblA_31, lblB_32, lblA_33, lblB_34), axis=-1)
        lbl_aux1_4 = np.concatenate((lblB_41, lblA_42, lblB_43, lblA_44), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_1, lbl_aux1_2, lbl_aux1_3, lbl_aux1_4), axis=-2)

        msk_aux1_1 = np.concatenate((mskA_11, mskB_12, mskA_13, mskB_14), axis=-1)
        msk_aux1_2 = np.concatenate((mskB_21, mskA_22, mskB_23, mskA_24), axis=-1)
        msk_aux1_3 = np.concatenate((mskA_31, mskB_32, mskA_33, mskB_34), axis=-1)
        msk_aux1_4 = np.concatenate((mskB_41, mskA_42, mskB_43, mskA_44), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_1, msk_aux1_2, msk_aux1_3, msk_aux1_4), axis=-2)

        img_aux2_1 = np.concatenate((imgB_11, imgA_12, imgB_13, imgA_14), axis=-1)
        img_aux2_2 = np.concatenate((imgA_21, imgB_22, imgA_23, imgB_24), axis=-1)
        img_aux2_3 = np.concatenate((imgB_31, imgA_32, imgB_33, imgA_34), axis=-1)
        img_aux2_4 = np.concatenate((imgA_41, imgB_42, imgA_43, imgB_44), axis=-1)
        img_aux2 = np.concatenate((img_aux2_1, img_aux2_2, img_aux2_3, img_aux2_4), axis=-2)

        lbl_aux2_1 = np.concatenate((lblB_11, lblA_12, lblB_13, lblA_14), axis=-1)
        lbl_aux2_2 = np.concatenate((lblA_21, lblB_22, lblA_23, lblB_24), axis=-1)
        lbl_aux2_3 = np.concatenate((lblB_31, lblA_32, lblB_33, lblA_34), axis=-1)
        lbl_aux2_4 = np.concatenate((lblA_41, lblB_42, lblA_43, lblB_44), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_1, lbl_aux2_2, lbl_aux2_3, lbl_aux2_4), axis=-2)

        msk_aux2_1 = np.concatenate((mskB_11, mskA_12, mskB_13, mskA_14), axis=-1)
        msk_aux2_2 = np.concatenate((mskA_21, mskB_22, mskA_23, mskB_24), axis=-1)
        msk_aux2_3 = np.concatenate((mskB_31, mskA_32, mskB_33, mskA_34), axis=-1)
        msk_aux2_4 = np.concatenate((mskA_41, mskB_42, mskA_43, mskB_44), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_1, msk_aux2_2, msk_aux2_3, msk_aux2_4), axis=-2)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col4row5(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 4)   # 2048/4 = 512
        w2 = int(img.shape[-1] / 2)   # 2048/2 = 1024
        w3 = 3 * w1                   # 3 * 512 = 1536
        h1 = int(img.shape[-2] / 5)   # 64/5 = 12
        h2 = 2 * h1                   # 2*12 = 24
        h3 = 3 * h1                   # 3*12 = 36
        h4 = 4 * h1                   # 4*12 = 48

        imgA_11, imgA_12, imgA_13, imgA_14 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:w3], img[:,   :h1, w3:]
        imgA_21, imgA_22, imgA_23, imgA_24 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:w3], img[:, h1:h2, w3:]
        imgA_31, imgA_32, imgA_33, imgA_34 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:w3], img[:, h2:h3, w3:]
        imgA_41, imgA_42, imgA_43, imgA_44 = img[:, h3:h4, :w1], img[:, h3:h4, w1:w2], img[:, h3:h4, w2:w3], img[:, h3:h4, w3:]
        imgA_51, imgA_52, imgA_53, imgA_54 = img[:, h4:  , :w1], img[:, h4:  , w1:w2], img[:, h4:  , w2:w3], img[:, h4:  , w3:]

        lblA_11, lblA_12, lblA_13, lblA_14 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:w3], lbl[     :h1, w3:]
        lblA_21, lblA_22, lblA_23, lblA_24 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:w3], lbl[   h1:h2, w3:]
        lblA_31, lblA_32, lblA_33, lblA_34 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:w3], lbl[   h2:h3, w3:]
        lblA_41, lblA_42, lblA_43, lblA_44 = lbl[   h3:h4, :w1], lbl[   h3:h4, w1:w2], lbl[   h3:h4, w2:w3], lbl[   h3:h4, w3:]
        lblA_51, lblA_52, lblA_53, lblA_54 = lbl[   h4:  , :w1], lbl[   h4:  , w1:w2], lbl[   h4:  , w2:w3], lbl[   h4:  , w3:]

        mskA_11, mskA_12, mskA_13, mskA_14 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:w3], msk[     :h1, w3:]
        mskA_21, mskA_22, mskA_23, mskA_24 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:w3], msk[   h1:h2, w3:]
        mskA_31, mskA_32, mskA_33, mskA_34 = msk[   h2:h3, :w1], msk[   h2:h3, w1:w2], msk[   h2:h3, w2:w3], msk[   h2:h3, w3:]
        mskA_41, mskA_42, mskA_43, mskA_44 = msk[   h3:h4, :w1], msk[   h3:h4, w1:w2], msk[   h3:h4, w2:w3], msk[   h3:h4, w3:]
        mskA_51, mskA_52, mskA_53, mskA_54 = msk[   h4:  , :w1], msk[   h4:  , w1:w2], msk[   h4:  , w2:w3], msk[   h4:  , w3:]

        imgB_11, imgB_12, imgB_13, imgB_14 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:w3], img_aux[:,   :h1, w3:]
        imgB_21, imgB_22, imgB_23, imgB_24 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:w3], img_aux[:, h1:h2, w3:]
        imgB_31, imgB_32, imgB_33, imgB_34 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:w3], img_aux[:, h2:h3, w3:]
        imgB_41, imgB_42, imgB_43, imgB_44 = img_aux[:, h3:h4, :w1], img_aux[:, h3:h4, w1:w2], img_aux[:, h3:h4, w2:w3], img_aux[:, h3:h4, w3:]
        imgB_51, imgB_52, imgB_53, imgB_54 = img_aux[:, h4:  , :w1], img_aux[:, h4:  , w1:w2], img_aux[:, h4:  , w2:w3], img_aux[:, h4:  , w3:]

        lblB_11, lblB_12, lblB_13, lblB_14 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:w3], lbl_aux[     :h1, w3:]
        lblB_21, lblB_22, lblB_23, lblB_24 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:w3], lbl_aux[   h1:h2, w3:]
        lblB_31, lblB_32, lblB_33, lblB_34 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:w3], lbl_aux[   h2:h3, w3:]
        lblB_41, lblB_42, lblB_43, lblB_44 = lbl_aux[   h3:h4, :w1], lbl_aux[   h3:h4, w1:w2], lbl_aux[   h3:h4, w2:w3], lbl_aux[   h3:h4, w3:]
        lblB_51, lblB_52, lblB_53, lblB_54 = lbl_aux[   h4:  , :w1], lbl_aux[   h4:  , w1:w2], lbl_aux[   h4:  , w2:w3], lbl_aux[   h4:  , w3:]

        mskB_11, mskB_12, mskB_13, mskB_14 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:w3], msk_aux[     :h1, w3:]
        mskB_21, mskB_22, mskB_23, mskB_24 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:w3], msk_aux[   h1:h2, w3:]
        mskB_31, mskB_32, mskB_33, mskB_34 = msk_aux[   h2:h3, :w1], msk_aux[   h2:h3, w1:w2], msk_aux[   h2:h3, w2:w3], msk_aux[   h2:h3, w3:]
        mskB_41, mskB_42, mskB_43, mskB_44 = msk_aux[   h3:h4, :w1], msk_aux[   h3:h4, w1:w2], msk_aux[   h3:h4, w2:w3], msk_aux[   h3:h4, w3:]
        mskB_51, mskB_52, mskB_53, mskB_54 = msk_aux[   h4:  , :w1], msk_aux[   h4:  , w1:w2], msk_aux[   h4:  , w2:w3], msk_aux[   h4:  , w3:]
        
        img_aux1_1 = np.concatenate((imgA_11, imgB_12, imgA_13, imgB_14), axis=-1)
        img_aux1_2 = np.concatenate((imgB_21, imgA_22, imgB_23, imgA_24), axis=-1)
        img_aux1_3 = np.concatenate((imgA_31, imgB_32, imgA_33, imgB_34), axis=-1)
        img_aux1_4 = np.concatenate((imgB_41, imgA_42, imgB_43, imgA_44), axis=-1)
        img_aux1_5 = np.concatenate((imgA_51, imgB_52, imgA_53, imgB_54), axis=-1)
        img_aux1 = np.concatenate((img_aux1_1, img_aux1_2, img_aux1_3, img_aux1_4, img_aux1_5), axis=-2)

        lbl_aux1_1 = np.concatenate((lblA_11, lblB_12, lblA_13, lblB_14), axis=-1)
        lbl_aux1_2 = np.concatenate((lblB_21, lblA_22, lblB_23, lblA_24), axis=-1)
        lbl_aux1_3 = np.concatenate((lblA_31, lblB_32, lblA_33, lblB_34), axis=-1)
        lbl_aux1_4 = np.concatenate((lblB_41, lblA_42, lblB_43, lblA_44), axis=-1)
        lbl_aux1_5 = np.concatenate((lblA_51, lblB_52, lblA_53, lblB_54), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_1, lbl_aux1_2, lbl_aux1_3, lbl_aux1_4, lbl_aux1_5), axis=-2)

        msk_aux1_1 = np.concatenate((mskA_11, mskB_12, mskA_13, mskB_14), axis=-1)
        msk_aux1_2 = np.concatenate((mskB_21, mskA_22, mskB_23, mskA_24), axis=-1)
        msk_aux1_3 = np.concatenate((mskA_31, mskB_32, mskA_33, mskB_34), axis=-1)
        msk_aux1_4 = np.concatenate((mskB_41, mskA_42, mskB_43, mskA_44), axis=-1)
        msk_aux1_5 = np.concatenate((mskA_51, mskB_52, mskA_53, mskB_54), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_1, msk_aux1_2, msk_aux1_3, msk_aux1_4, msk_aux1_5), axis=-2)

        img_aux2_1 = np.concatenate((imgB_11, imgA_12, imgB_13, imgA_14), axis=-1)
        img_aux2_2 = np.concatenate((imgA_21, imgB_22, imgA_23, imgB_24), axis=-1)
        img_aux2_3 = np.concatenate((imgB_31, imgA_32, imgB_33, imgA_34), axis=-1)
        img_aux2_4 = np.concatenate((imgA_41, imgB_42, imgA_43, imgB_44), axis=-1)
        img_aux2_5 = np.concatenate((imgB_51, imgA_52, imgB_53, imgA_54), axis=-1)
        img_aux2 = np.concatenate((img_aux2_1, img_aux2_2, img_aux2_3, img_aux2_4, img_aux2_5), axis=-2)

        lbl_aux2_1 = np.concatenate((lblB_11, lblA_12, lblB_13, lblA_14), axis=-1)
        lbl_aux2_2 = np.concatenate((lblA_21, lblB_22, lblA_23, lblB_24), axis=-1)
        lbl_aux2_3 = np.concatenate((lblB_31, lblA_32, lblB_33, lblA_34), axis=-1)
        lbl_aux2_4 = np.concatenate((lblA_41, lblB_42, lblA_43, lblB_44), axis=-1)
        lbl_aux2_5 = np.concatenate((lblB_51, lblA_52, lblB_53, lblA_54), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_1, lbl_aux2_2, lbl_aux2_3, lbl_aux2_4, lbl_aux2_5), axis=-2)

        msk_aux2_1 = np.concatenate((mskB_11, mskA_12, mskB_13, mskA_14), axis=-1)
        msk_aux2_2 = np.concatenate((mskA_21, mskB_22, mskA_23, mskB_24), axis=-1)
        msk_aux2_3 = np.concatenate((mskB_31, mskA_32, mskB_33, mskA_34), axis=-1)
        msk_aux2_4 = np.concatenate((mskA_41, mskB_42, mskA_43, mskB_44), axis=-1)
        msk_aux2_5 = np.concatenate((mskB_51, mskA_52, mskB_53, mskA_54), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_1, msk_aux2_2, msk_aux2_3, msk_aux2_4, msk_aux2_5), axis=-2)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col4row6(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 4)   # 2048/4 = 512
        w2 = int(img.shape[-1] / 2)   # 2048/2 = 1024
        w3 = 3 * w1                   # 3 * 512 = 1536
        h1 = int(img.shape[-2] / 6)   # 64/6 = 10
        h2 = 2 * h1                   # 2*10 = 20
        h3 = 3 * h1                   # 3*10 = 30
        h4 = 4 * h1                   # 4*10 = 40
        h5 = 5 * h1                   # 5*10 = 50

        imgA_11, imgA_12, imgA_13, imgA_14 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:w3], img[:,   :h1, w3:]
        imgA_21, imgA_22, imgA_23, imgA_24 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:w3], img[:, h1:h2, w3:]
        imgA_31, imgA_32, imgA_33, imgA_34 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:w3], img[:, h2:h3, w3:]
        imgA_41, imgA_42, imgA_43, imgA_44 = img[:, h3:h4, :w1], img[:, h3:h4, w1:w2], img[:, h3:h4, w2:w3], img[:, h3:h4, w3:]
        imgA_51, imgA_52, imgA_53, imgA_54 = img[:, h4:h5, :w1], img[:, h4:h5, w1:w2], img[:, h4:h5, w2:w3], img[:, h4:h5, w3:]
        imgA_61, imgA_62, imgA_63, imgA_64 = img[:, h5:  , :w1], img[:, h5:  , w1:w2], img[:, h5:  , w2:w3], img[:, h5:  , w3:]

        lblA_11, lblA_12, lblA_13, lblA_14 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:w3], lbl[     :h1, w3:]
        lblA_21, lblA_22, lblA_23, lblA_24 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:w3], lbl[   h1:h2, w3:]
        lblA_31, lblA_32, lblA_33, lblA_34 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:w3], lbl[   h2:h3, w3:]
        lblA_41, lblA_42, lblA_43, lblA_44 = lbl[   h3:h4, :w1], lbl[   h3:h4, w1:w2], lbl[   h3:h4, w2:w3], lbl[   h3:h4, w3:]
        lblA_51, lblA_52, lblA_53, lblA_54 = lbl[   h4:h5, :w1], lbl[   h4:h5, w1:w2], lbl[   h4:h5, w2:w3], lbl[   h4:h5, w3:]
        lblA_61, lblA_62, lblA_63, lblA_64 = lbl[   h5:  , :w1], lbl[   h5:  , w1:w2], lbl[   h5:  , w2:w3], lbl[   h5:  , w3:]

        mskA_11, mskA_12, mskA_13, mskA_14 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:w3], msk[     :h1, w3:]
        mskA_21, mskA_22, mskA_23, mskA_24 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:w3], msk[   h1:h2, w3:]
        mskA_31, mskA_32, mskA_33, mskA_34 = msk[   h2:h3, :w1], msk[   h2:h3, w1:w2], msk[   h2:h3, w2:w3], msk[   h2:h3, w3:]
        mskA_41, mskA_42, mskA_43, mskA_44 = msk[   h3:h4, :w1], msk[   h3:h4, w1:w2], msk[   h3:h4, w2:w3], msk[   h3:h4, w3:]
        mskA_51, mskA_52, mskA_53, mskA_54 = msk[   h4:h5, :w1], msk[   h4:h5, w1:w2], msk[   h4:h5, w2:w3], msk[   h4:h5, w3:]
        mskA_61, mskA_62, mskA_63, mskA_64 = msk[   h5:  , :w1], msk[   h5:  , w1:w2], msk[   h5:  , w2:w3], msk[   h5:  , w3:]

        imgB_11, imgB_12, imgB_13, imgB_14 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:w3], img_aux[:,   :h1, w3:]
        imgB_21, imgB_22, imgB_23, imgB_24 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:w3], img_aux[:, h1:h2, w3:]
        imgB_31, imgB_32, imgB_33, imgB_34 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:w3], img_aux[:, h2:h3, w3:]
        imgB_41, imgB_42, imgB_43, imgB_44 = img_aux[:, h3:h4, :w1], img_aux[:, h3:h4, w1:w2], img_aux[:, h3:h4, w2:w3], img_aux[:, h3:h4, w3:]
        imgB_51, imgB_52, imgB_53, imgB_54 = img_aux[:, h4:h5, :w1], img_aux[:, h4:h5, w1:w2], img_aux[:, h4:h5, w2:w3], img_aux[:, h4:h5, w3:]
        imgB_61, imgB_62, imgB_63, imgB_64 = img_aux[:, h5:  , :w1], img_aux[:, h5:  , w1:w2], img_aux[:, h5:  , w2:w3], img_aux[:, h5:  , w3:]

        lblB_11, lblB_12, lblB_13, lblB_14 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:w3], lbl_aux[     :h1, w3:]
        lblB_21, lblB_22, lblB_23, lblB_24 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:w3], lbl_aux[   h1:h2, w3:]
        lblB_31, lblB_32, lblB_33, lblB_34 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:w3], lbl_aux[   h2:h3, w3:]
        lblB_41, lblB_42, lblB_43, lblB_44 = lbl_aux[   h3:h4, :w1], lbl_aux[   h3:h4, w1:w2], lbl_aux[   h3:h4, w2:w3], lbl_aux[   h3:h4, w3:]
        lblB_51, lblB_52, lblB_53, lblB_54 = lbl_aux[   h4:h5, :w1], lbl_aux[   h4:h5, w1:w2], lbl_aux[   h4:h5, w2:w3], lbl_aux[   h4:h5, w3:]
        lblB_61, lblB_62, lblB_63, lblB_64 = lbl_aux[   h5:  , :w1], lbl_aux[   h5:  , w1:w2], lbl_aux[   h5:  , w2:w3], lbl_aux[   h5:  , w3:]

        mskB_11, mskB_12, mskB_13, mskB_14 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:w3], msk_aux[     :h1, w3:]
        mskB_21, mskB_22, mskB_23, mskB_24 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:w3], msk_aux[   h1:h2, w3:]
        mskB_31, mskB_32, mskB_33, mskB_34 = msk_aux[   h2:h3, :w1], msk_aux[   h2:h3, w1:w2], msk_aux[   h2:h3, w2:w3], msk_aux[   h2:h3, w3:]
        mskB_41, mskB_42, mskB_43, mskB_44 = msk_aux[   h3:h4, :w1], msk_aux[   h3:h4, w1:w2], msk_aux[   h3:h4, w2:w3], msk_aux[   h3:h4, w3:]
        mskB_51, mskB_52, mskB_53, mskB_54 = msk_aux[   h4:h5, :w1], msk_aux[   h4:h5, w1:w2], msk_aux[   h4:h5, w2:w3], msk_aux[   h4:h5, w3:]
        mskB_61, mskB_62, mskB_63, mskB_64 = msk_aux[   h5:  , :w1], msk_aux[   h5:  , w1:w2], msk_aux[   h5:  , w2:w3], msk_aux[   h5:  , w3:]
        
        img_aux1_1 = np.concatenate((imgA_11, imgB_12, imgA_13, imgB_14), axis=-1)
        img_aux1_2 = np.concatenate((imgB_21, imgA_22, imgB_23, imgA_24), axis=-1)
        img_aux1_3 = np.concatenate((imgA_31, imgB_32, imgA_33, imgB_34), axis=-1)
        img_aux1_4 = np.concatenate((imgB_41, imgA_42, imgB_43, imgA_44), axis=-1)
        img_aux1_5 = np.concatenate((imgA_51, imgB_52, imgA_53, imgB_54), axis=-1)
        img_aux1_6 = np.concatenate((imgB_61, imgA_62, imgB_63, imgA_64), axis=-1)
        img_aux1 = np.concatenate((img_aux1_1, img_aux1_2, img_aux1_3, img_aux1_4, img_aux1_5, img_aux1_6), axis=-2)

        lbl_aux1_1 = np.concatenate((lblA_11, lblB_12, lblA_13, lblB_14), axis=-1)
        lbl_aux1_2 = np.concatenate((lblB_21, lblA_22, lblB_23, lblA_24), axis=-1)
        lbl_aux1_3 = np.concatenate((lblA_31, lblB_32, lblA_33, lblB_34), axis=-1)
        lbl_aux1_4 = np.concatenate((lblB_41, lblA_42, lblB_43, lblA_44), axis=-1)
        lbl_aux1_5 = np.concatenate((lblA_51, lblB_52, lblA_53, lblB_54), axis=-1)
        lbl_aux1_6 = np.concatenate((lblB_61, lblA_62, lblB_63, lblA_64), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_1, lbl_aux1_2, lbl_aux1_3, lbl_aux1_4, lbl_aux1_5, lbl_aux1_6), axis=-2)

        msk_aux1_1 = np.concatenate((mskA_11, mskB_12, mskA_13, mskB_14), axis=-1)
        msk_aux1_2 = np.concatenate((mskB_21, mskA_22, mskB_23, mskA_24), axis=-1)
        msk_aux1_3 = np.concatenate((mskA_31, mskB_32, mskA_33, mskB_34), axis=-1)
        msk_aux1_4 = np.concatenate((mskB_41, mskA_42, mskB_43, mskA_44), axis=-1)
        msk_aux1_5 = np.concatenate((mskA_51, mskB_52, mskA_53, mskB_54), axis=-1)
        msk_aux1_6 = np.concatenate((mskB_61, mskA_62, mskB_63, mskA_64), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_1, msk_aux1_2, msk_aux1_3, msk_aux1_4, msk_aux1_5, msk_aux1_6), axis=-2)

        img_aux2_1 = np.concatenate((imgB_11, imgA_12, imgB_13, imgA_14), axis=-1)
        img_aux2_2 = np.concatenate((imgA_21, imgB_22, imgA_23, imgB_24), axis=-1)
        img_aux2_3 = np.concatenate((imgB_31, imgA_32, imgB_33, imgA_34), axis=-1)
        img_aux2_4 = np.concatenate((imgA_41, imgB_42, imgA_43, imgB_44), axis=-1)
        img_aux2_5 = np.concatenate((imgB_51, imgA_52, imgB_53, imgA_54), axis=-1)
        img_aux2_6 = np.concatenate((imgA_61, imgB_62, imgA_63, imgB_64), axis=-1)
        img_aux2 = np.concatenate((img_aux2_1, img_aux2_2, img_aux2_3, img_aux2_4, img_aux2_5, img_aux2_6), axis=-2)

        lbl_aux2_1 = np.concatenate((lblB_11, lblA_12, lblB_13, lblA_14), axis=-1)
        lbl_aux2_2 = np.concatenate((lblA_21, lblB_22, lblA_23, lblB_24), axis=-1)
        lbl_aux2_3 = np.concatenate((lblB_31, lblA_32, lblB_33, lblA_34), axis=-1)
        lbl_aux2_4 = np.concatenate((lblA_41, lblB_42, lblA_43, lblB_44), axis=-1)
        lbl_aux2_5 = np.concatenate((lblB_51, lblA_52, lblB_53, lblA_54), axis=-1)
        lbl_aux2_6 = np.concatenate((lblA_61, lblB_62, lblA_63, lblB_64), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_1, lbl_aux2_2, lbl_aux2_3, lbl_aux2_4, lbl_aux2_5, lbl_aux2_6), axis=-2)

        msk_aux2_1 = np.concatenate((mskB_11, mskA_12, mskB_13, mskA_14), axis=-1)
        msk_aux2_2 = np.concatenate((mskA_21, mskB_22, mskA_23, mskB_24), axis=-1)
        msk_aux2_3 = np.concatenate((mskB_31, mskA_32, mskB_33, mskA_34), axis=-1)
        msk_aux2_4 = np.concatenate((mskA_41, mskB_42, mskA_43, mskB_44), axis=-1)
        msk_aux2_5 = np.concatenate((mskB_51, mskA_52, mskB_53, mskA_54), axis=-1)
        msk_aux2_6 = np.concatenate((mskA_61, mskB_62, mskA_63, mskB_64), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_1, msk_aux2_2, msk_aux2_3, msk_aux2_4, msk_aux2_5, msk_aux2_6), axis=-2)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2


    def col6row4(self, img, lbl, msk, img_aux, lbl_aux, msk_aux):
        w1 = int(img.shape[-1] / 4)   # 2048/6 = 341
        w2 = 2 * w1                   # 2*341  = 682
        w3 = 3 * w1                   # 3*341  = 1023
        w4 = 4 * w1                   # 4*341  = 1361
        w5 = 5 * w1                   # 5*341  = 1705
        h1 = int(img.shape[-2] / 4)   # 64/4 = 16
        h2 = 2 * h1                   # 2*16 = 32
        h3 = 3 * h1                   # 3*16 = 48

        imgA_11, imgA_12, imgA_13, imgA_14, imgA_15, imgA_16 = img[:,   :h1, :w1], img[:,   :h1, w1:w2], img[:,   :h1, w2:w3], img[:,   :h1, w3:w4], img[:,   :h1, w4:w5], img[:,   :h1, w5:]
        imgA_21, imgA_22, imgA_23, imgA_24, imgA_25, imgA_26 = img[:, h1:h2, :w1], img[:, h1:h2, w1:w2], img[:, h1:h2, w2:w3], img[:, h1:h2, w3:w4], img[:, h1:h2, w4:w5], img[:, h1:h2, w5:]
        imgA_31, imgA_32, imgA_33, imgA_34, imgA_35, imgA_36 = img[:, h2:h3, :w1], img[:, h2:h3, w1:w2], img[:, h2:h3, w2:w3], img[:, h2:h3, w3:w4], img[:, h2:h3, w4:w5], img[:, h2:h3, w5:]
        imgA_41, imgA_42, imgA_43, imgA_44, imgA_45, imgA_46 = img[:, h3:  , :w1], img[:, h3:  , w1:w2], img[:, h3:  , w2:w3], img[:, h3:  , w3:w4], img[:, h3:  , w4:w5], img[:, h3:  , w5:]

        lblA_11, lblA_12, lblA_13, lblA_14, lblA_15, lblA_16 = lbl[     :h1, :w1], lbl[     :h1, w1:w2], lbl[     :h1, w2:w3], lbl[     :h1, w3:w4], lbl[     :h1, w4:w5], lbl[     :h1, w5:]
        lblA_21, lblA_22, lblA_23, lblA_24, lblA_25, lblA_26 = lbl[   h1:h2, :w1], lbl[   h1:h2, w1:w2], lbl[   h1:h2, w2:w3], lbl[   h1:h2, w3:w4], lbl[   h1:h2, w4:w5], lbl[   h1:h2, w5:]
        lblA_31, lblA_32, lblA_33, lblA_34, lblA_35, lblA_36 = lbl[   h2:h3, :w1], lbl[   h2:h3, w1:w2], lbl[   h2:h3, w2:w3], lbl[   h2:h3, w3:w4], lbl[   h2:h3, w4:w5], lbl[   h2:h3, w5:]
        lblA_41, lblA_42, lblA_43, lblA_44, lblA_45, lblA_46 = lbl[   h3:  , :w1], lbl[   h3:  , w1:w2], lbl[   h3:  , w2:w3], lbl[   h3:  , w3:w4], lbl[   h3:  , w4:w5], lbl[   h3:  , w5:]

        mskA_11, mskA_12, mskA_13, mskA_14, mskA_15, mskA_16 = msk[     :h1, :w1], msk[     :h1, w1:w2], msk[     :h1, w2:w3], msk[     :h1, w3:w4], msk[     :h1, w4:w5], msk[     :h1, w5:]
        mskA_21, mskA_22, mskA_23, mskA_24, mskA_25, mskA_26 = msk[   h1:h2, :w1], msk[   h1:h2, w1:w2], msk[   h1:h2, w2:w3], msk[   h1:h2, w3:w4], msk[   h1:h2, w4:w5], msk[   h1:h2, w5:]
        mskA_31, mskA_32, mskA_33, mskA_34, mskA_35, mskA_36 = msk[   h2:h3, :w1], msk[   h2:h3, w1:w2], msk[   h2:h3, w2:w3], msk[   h2:h3, w3:w4], msk[   h2:h3, w4:w5], msk[   h2:h3, w5:]
        mskA_41, mskA_42, mskA_43, mskA_44, mskA_45, mskA_46 = msk[   h3:  , :w1], msk[   h3:  , w1:w2], msk[   h3:  , w2:w3], msk[   h3:  , w3:w4], msk[   h3:  , w4:w5], msk[   h3:  , w5:]

        imgB_11, imgB_12, imgB_13, imgB_14, imgB_15, imgB_16 = img_aux[:,   :h1, :w1], img_aux[:,   :h1, w1:w2], img_aux[:,   :h1, w2:w3], img_aux[:,   :h1, w3:w4], img_aux[:,   :h1, w4:w5], img_aux[:,   :h1, w5:]
        imgB_21, imgB_22, imgB_23, imgB_24, imgB_25, imgB_26 = img_aux[:, h1:h2, :w1], img_aux[:, h1:h2, w1:w2], img_aux[:, h1:h2, w2:w3], img_aux[:, h1:h2, w3:w4], img_aux[:, h1:h2, w4:w5], img_aux[:, h1:h2, w5:]
        imgB_31, imgB_32, imgB_33, imgB_34, imgB_35, imgB_36 = img_aux[:, h2:h3, :w1], img_aux[:, h2:h3, w1:w2], img_aux[:, h2:h3, w2:w3], img_aux[:, h2:h3, w3:w4], img_aux[:, h2:h3, w4:w5], img_aux[:, h2:h3, w5:]
        imgB_41, imgB_42, imgB_43, imgB_44, imgB_45, imgB_46 = img_aux[:, h3:  , :w1], img_aux[:, h3:  , w1:w2], img_aux[:, h3:  , w2:w3], img_aux[:, h3:  , w3:w4], img_aux[:, h3:  , w4:w5], img_aux[:, h3:  , w5:]

        lblB_11, lblB_12, lblB_13, lblB_14, lblB_15, lblB_16 = lbl_aux[     :h1, :w1], lbl_aux[     :h1, w1:w2], lbl_aux[     :h1, w2:w3], lbl_aux[     :h1, w3:w4], lbl_aux[     :h1, w4:w5], lbl_aux[     :h1, w5:]
        lblB_21, lblB_22, lblB_23, lblB_24, lblB_25, lblB_26 = lbl_aux[   h1:h2, :w1], lbl_aux[   h1:h2, w1:w2], lbl_aux[   h1:h2, w2:w3], lbl_aux[   h1:h2, w3:w4], lbl_aux[   h1:h2, w4:w5], lbl_aux[   h1:h2, w5:]
        lblB_31, lblB_32, lblB_33, lblB_34, lblB_35, lblB_36 = lbl_aux[   h2:h3, :w1], lbl_aux[   h2:h3, w1:w2], lbl_aux[   h2:h3, w2:w3], lbl_aux[   h2:h3, w3:w4], lbl_aux[   h2:h3, w4:w5], lbl_aux[   h2:h3, w5:]
        lblB_41, lblB_42, lblB_43, lblB_44, lblB_45, lblB_46 = lbl_aux[   h3:  , :w1], lbl_aux[   h3:  , w1:w2], lbl_aux[   h3:  , w2:w3], lbl_aux[   h3:  , w3:w4], lbl_aux[   h3:  , w4:w5], lbl_aux[   h3:  , w5:]

        mskB_11, mskB_12, mskB_13, mskB_14, mskB_15, mskB_16 = msk_aux[     :h1, :w1], msk_aux[     :h1, w1:w2], msk_aux[     :h1, w2:w3], msk_aux[     :h1, w3:w4], msk_aux[     :h1, w4:w5], msk_aux[     :h1, w5:]
        mskB_21, mskB_22, mskB_23, mskB_24, mskB_25, mskB_26 = msk_aux[   h1:h2, :w1], msk_aux[   h1:h2, w1:w2], msk_aux[   h1:h2, w2:w3], msk_aux[   h1:h2, w3:w4], msk_aux[   h1:h2, w4:w5], msk_aux[   h1:h2, w5:]
        mskB_31, mskB_32, mskB_33, mskB_34, mskB_35, mskB_36 = msk_aux[   h2:h3, :w1], msk_aux[   h2:h3, w1:w2], msk_aux[   h2:h3, w2:w3], msk_aux[   h2:h3, w3:w4], msk_aux[   h2:h3, w4:w5], msk_aux[   h2:h3, w5:]
        mskB_41, mskB_42, mskB_43, mskB_44, mskB_45, mskB_46 = msk_aux[   h3:  , :w1], msk_aux[   h3:  , w1:w2], msk_aux[   h3:  , w2:w3], msk_aux[   h3:  , w3:w4], msk_aux[   h3:  , w4:w5], msk_aux[   h3:  , w5:]
        
        img_aux1_1 = np.concatenate((imgA_11, imgB_12, imgA_13, imgB_14, imgA_15, imgB_16), axis=-1)
        img_aux1_2 = np.concatenate((imgB_21, imgA_22, imgB_23, imgA_24, imgB_25, imgA_26), axis=-1)
        img_aux1_3 = np.concatenate((imgA_31, imgB_32, imgA_33, imgB_34, imgA_35, imgB_36), axis=-1)
        img_aux1_4 = np.concatenate((imgB_41, imgA_42, imgB_43, imgA_44, imgB_45, imgA_46), axis=-1)
        img_aux1 = np.concatenate((img_aux1_1, img_aux1_2, img_aux1_3, img_aux1_4), axis=-2)

        lbl_aux1_1 = np.concatenate((lblA_11, lblB_12, lblA_13, lblB_14, lblA_15, lblB_16), axis=-1)
        lbl_aux1_2 = np.concatenate((lblB_21, lblA_22, lblB_23, lblA_24, lblB_25, lblA_26), axis=-1)
        lbl_aux1_3 = np.concatenate((lblA_31, lblB_32, lblA_33, lblB_34, lblA_35, lblB_36), axis=-1)
        lbl_aux1_4 = np.concatenate((lblB_41, lblA_42, lblB_43, lblA_44, lblB_45, lblA_46), axis=-1)
        lbl_aux1 = np.concatenate((lbl_aux1_1, lbl_aux1_2, lbl_aux1_3, lbl_aux1_4), axis=-2)

        msk_aux1_1 = np.concatenate((mskA_11, mskB_12, mskA_13, mskB_14, mskA_15, mskB_16), axis=-1)
        msk_aux1_2 = np.concatenate((mskB_21, mskA_22, mskB_23, mskA_24, mskB_25, mskA_26), axis=-1)
        msk_aux1_3 = np.concatenate((mskA_31, mskB_32, mskA_33, mskB_34, mskA_35, mskB_36), axis=-1)
        msk_aux1_4 = np.concatenate((mskB_41, mskA_42, mskB_43, mskA_44, mskB_45, mskA_46), axis=-1)
        msk_aux1 = np.concatenate((msk_aux1_1, msk_aux1_2, msk_aux1_3, msk_aux1_4), axis=-2)

        img_aux2_1 = np.concatenate((imgB_11, imgA_12, imgB_13, imgA_14, imgB_15, imgA_16), axis=-1)
        img_aux2_2 = np.concatenate((imgA_21, imgB_22, imgA_23, imgB_24, imgA_25, imgB_26), axis=-1)
        img_aux2_3 = np.concatenate((imgB_31, imgA_32, imgB_33, imgA_34, imgB_35, imgA_36), axis=-1)
        img_aux2_4 = np.concatenate((imgA_41, imgB_42, imgA_43, imgB_44, imgA_45, imgB_46), axis=-1)
        img_aux2 = np.concatenate((img_aux2_1, img_aux2_2, img_aux2_3, img_aux2_4), axis=-2)

        lbl_aux2_1 = np.concatenate((lblB_11, lblA_12, lblB_13, lblA_14, lblB_15, lblA_16), axis=-1)
        lbl_aux2_2 = np.concatenate((lblA_21, lblB_22, lblA_23, lblB_24, lblA_25, lblB_26), axis=-1)
        lbl_aux2_3 = np.concatenate((lblB_31, lblA_32, lblB_33, lblA_34, lblB_35, lblA_36), axis=-1)
        lbl_aux2_4 = np.concatenate((lblA_41, lblB_42, lblA_43, lblB_44, lblA_45, lblB_46), axis=-1)
        lbl_aux2 = np.concatenate((lbl_aux2_1, lbl_aux2_2, lbl_aux2_3, lbl_aux2_4), axis=-2)

        msk_aux2_1 = np.concatenate((mskB_11, mskA_12, mskB_13, mskA_14, mskB_15, mskA_16), axis=-1)
        msk_aux2_2 = np.concatenate((mskA_21, mskB_22, mskA_23, mskB_24, mskA_25, mskB_26), axis=-1)
        msk_aux2_3 = np.concatenate((mskB_31, mskA_32, mskB_33, mskA_34, mskB_35, mskA_36), axis=-1)
        msk_aux2_4 = np.concatenate((mskA_41, mskB_42, mskA_43, mskB_44, mskA_45, mskB_46), axis=-1)
        msk_aux2 = np.concatenate((msk_aux2_1, msk_aux2_2, msk_aux2_3, msk_aux2_4), axis=-2)

        return img_aux1, lbl_aux1, msk_aux1, img_aux2, lbl_aux2, msk_aux2
