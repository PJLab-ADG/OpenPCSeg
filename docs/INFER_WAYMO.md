# Inference and Visualization on Waymo Dataset

### Environment Setup

We encourage you to create a separate environment for the unpacking of Waymo Open Dataset, execute below in `PCSeg` folder:
```
conda create -n wod python=3.6
conda activate wod
pip install opencv-python
pip install waymo-open-dataset-tf-2-1-0
pip install protobuf==3.19.0
pip install tqdm
python setup.py develop
```

### Prepare Waymo Data

First, download a sequence from [Waymo Perception Dataset v1.3.2](https://waymo.com/intl/en_us/open/download/).

For example, you can select `segment-10082223140073588526_6140_000_6160_000_with_camera_labels.tfrecord` from the training set.

```
mkdir ./infer_data
mkdir ./infer_data/raw_waymo/
```

Further, place `segment-10082223140073588526_6140_000_6160_000_with_camera_labels.tfrecord` in `./infer_data/raw_waymo/` directory. 

```
conda activate wod
python ./tools/scripts/unpack_wod_sequence.py \
    --segment_path ./infer_data/raw_waymo/segment-10082223140073588526_6140_000_6160_000_with_camera_labels.tfrecord \
    --output_dir ./infer_data/output/segment-10082223140073588526_6140_000_6160_000_with_camera_labels.unpacked
```

### Infer Results with a Pretrained Model

```
conda activate pcseg
CUDA_VISIBLE_DEVICES=0 sh infer.sh 1 --cfg_file tools/cfgs/voxel/waymo/minkunet_mk34_cr16_infer.yaml --batch_size 1 --extra_tag default
```

### Visualize Predictions

We encourage you to create a separate environment for the visualization of Waymo Open Dataset results, on a machine with GUI:

```
conda create -n open3d python=3.6
conda activate open3d
pip install open3d==0.9.0
```

```
# Make sure you have a GUI -- in open3d env:
python tools/scripts/vis_waymo.py \
    --pc_path infer_data/output/segment-10082223140073588526_6140_000_6160_000_with_camera_labels.unpacked/LiDAR/0000000100.npy \
    --label_path infer_data/output/segment-10082223140073588526_6140_000_6160_000_with_camera_labels.unpacked/PCSeg/0000000100.npy
```

Here are some inference & visualization results:
<img src="./figs/wod_vis_01.png" align="center" width="80%">
<img src="./figs/wod_vis_02.png" align="center" width="80%">
