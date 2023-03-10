# Data Preparation

### Overall Structure

```
└── data_root
    └── NuScenes (upcoming)
    └── SemanticKitti
    └── ScribbleKitti
    └── Waymo
```

### SemanticKITTI

To install the [SemanticKITTI](http://semantic-kitti.org/index) dataset, download the data, annotations, and other files from http://semantic-kitti.org/dataset. Unpack the compressed file(s) into `./data_root/semantickitti` and re-organize the data structure. Your folder structure should end up looking like this:

```
└── SemanticKitti  
    └── dataset
        ├── velodyne <- contains the .bin files; a .bin file contains the points in a point cloud
        │    └── 00
        │    └── ···
        │    └── 21
        ├── labels   <- contains the .label files; a .label file contains the labels of the points in a point cloud
        │    └── 00
        │    └── ···
        │    └── 10
        ├── calib
        │    └── 00
        │    └── ···
        │    └── 21
        └── semantic-kitti.yaml
```

### ScribbleKITTI

To install the [ScribbleKITTI](https://arxiv.org/abs/2203.08537) dataset, download the annotations from https://data.vision.ee.ethz.ch/ouenal/scribblekitti.zip. Note that you only need to download these annotation files (~118.2MB); the data is the same as [SemanticKITTI](http://semantic-kitti.org/index). Unpack the compressed file(s) into `./data_root/scribblekitti` and re-organize the data structure. Your folder structure should end up looking like this:


```
└── ScribbleKITTI 
    └── dataset
        └── scribbles <- contains the .label files; a .label file contains the scribble labels of the points in a point cloud
             └── 00
             └── ···
             └── 10
```

### Waymo Open

To acquire the [Waymo Open](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Scalability_in_Perception_for_Autonomous_Driving_Waymo_Open_Dataset_CVPR_2020_paper.pdf) dataset, download the annotations from [here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_3_2/archived_files?authuser=1&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false). Note that you only need to download data from training and testing prefix. Unpack the compressed file(s) into `./data_root/Waymo/raw_data` and re-organize the data structure.

Installation. Note that we only test it with python 3.6.
```
rm -rf waymo-od > /dev/null
git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
cd waymo-od && git branch -a
git checkout remotes/origin/master
pip3 install --upgrade pip
pip3 install waymo-open-dataset-tf-2-6-0==1.4.3
```

Next, execute the following script:
```shell
python data/dataset/waymo/scripts/preprocess_waymo_data.py
```

Lastly, download files from [`train-0-31.txt`](https://www.dropbox.com/s/ijnxe9skn3r8dbg/train-0-31.txt?dl=0) and [`val-0-7.txt`](https://www.dropbox.com/s/cqcm9mftidik0fu/val-0-7.txt?dl=0), and put them into `Waymo` folder

Your folder structure should end up looking like this:
```
└── Waymo
    └── raw_data
        └── training
        └── validation
        └── testing
    └── train
        └── first
        └── second
    └── val_with_label
        └── first
        └── second
    └── train-0-31.txt
    └── val-0-7.txt
```


### References

Please consider site the original papers of the datasets if you find them helpful to your research.

#### nuScenes
```bibtex
@article{fong2022panopticnuscenes,
    author = {W. K. Fong and R. Mohan and J. V. Hurtado and L. Zhou and H. Caesar and O. Beijbom and A. Valada},
    title = {Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking},
    journal = {IEEE Robotics and Automation Letters},
    volume = {7},
    number = {2},
    pages = {3795--3802},
    year = {2022}
}
```
```bibtex
@inproceedings{caesar2020nuscenes,
    author = {H. Caesar and V. Bankiti and A. H. Lang and S. Vora and V. E. Liong and Q. Xu and A. Krishnan and Y. Pan and G. Baldan and O. Beijbom},
    title = {nuScenes: A Multimodal Dataset for Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {11621--11631},
    year = {2020}
}
```

#### SemanticKITTI

```bibtex
@inproceedings{behley2019semantickitti,
    author = {J. Behley and M. Garbade and A. Milioto and J. Quenzel and S. Behnke and C. Stachniss and J. Gall},
    title = {SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages = {9297--9307},
    year = {2019}
}
```
```bibtex
@inproceedings{geiger2012kitti,
    author = {A. Geiger and P. Lenz and R. Urtasun},
    title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {3354--3361},
    year = {2012}
}
```

#### ScribbleKITTI

```bibtex
@inproceedings{unal2022scribble,
    author = {O. Unal and D. Dai and L. Van Gool},
    title = {Scribble-Supervised LiDAR Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {2697--2707},
    year = {2022}
}
```

#### Waymo Open

```bibtex
@inproceedings{sun2020waymoopen,
    author = {P. Sun and H. Kretzschmar and X. Dotiwalla and A. Chouard and V. Patnaik and P. Tsui and J. Guo and Y. Zhou and Y. Chai and B. Caine and V. Vasudevan and W. Han and J. Ngiam and H. Zhao and A. Timofeev and S. Ettinger and M. Krivokon and A. Gao and A. Joshi and Y. Zhang and J. Shlens and Z. Chen and D. Anguelov},
    title = {Scalability in Perception for Autonomous Driving: Waymo Open Dataset},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {2446--2454},
    year = {2020}
}
```


### nuScenes (Coming Soon)

To install the [nuScenes-lidarseg](https://www.nuscenes.org/nuscenes) dataset, download the data, annotations, and other files from https://www.nuscenes.org/download. Unpack the compressed file(s) into `./data_root/nuscenes` and your folder structure should end up looking like this:

```
└── nuscenes  
    ├── Usual nuscenes folders (i.e. samples, sweep)
    │
    ├── lidarseg
    │   └── v1.0-{mini, test, trainval} <- contains the .bin files; a .bin file 
    │                                      contains the labels of the points in a 
    │                                      point cloud (note that v1.0-test does not 
    │                                      have any .bin files associated with it)
    │
    └── v1.0-{mini, test, trainval}
        ├── Usual files (e.g. attribute.json, calibrated_sensor.json etc.)
        ├── lidarseg.json  <- contains the mapping of each .bin file to the token   
        └── category.json  <- contains the categories of the labels (note that the 
                              category.json from nuScenes v1.0 is overwritten)
```
