# Getting Started

### SemanticKITTI

For example, if you want to train the following models with 2 GPUs:

```
CUDA_VISIBLE_DEVICES=0,1 sh dist_train.sh 2 --cfg_file tools/cfgs/voxel/semantic_kitti/minkunet_mk34_cr10.yaml           

CUDA_VISIBLE_DEVICES=2,3 sh dist_train.sh 2 --cfg_file tools/cfgs/voxel/semantic_kitti/cylinder_cy480_cr10.yaml

CUDA_VISIBLE_DEVICES=4,5 sh dist_train.sh 2 --cfg_file tools/cfgs/fusion/semantic_kitti/spvcnn_mk18_cr10.yaml

CUDA_VISIBLE_DEVICES=6,7 sh dist_train.sh 2 --cfg_file tools/cfgs/fusion/semantic_kitti/rpvnet_mk18_cr10.yaml
```

### Waymo Open Dataset

For example, if you want to train the following models with 2 GPUs:

```
CUDA_VISIBLE_DEVICES=0,1 sh dist_train.sh 2 --cfg_file tools/cfgs/voxel/waymo/minkunet_mk34_cr16.yaml       

CUDA_VISIBLE_DEVICES=2,3 sh dist_train.sh 2 --cfg_file tools/cfgs/voxel/waymo/cylinder_cy480_cr10.yaml
```

### Docker

#### Docker

After building according to [installation instructions](./INSTALL.md), run the container while mounting the dataset, e.g. SemanticKITTI:

```bash
docker run -it --rm --gpus all -v /path/to/logs:/home/pcseg -v /path/to/kitti/semanticKITTI/dataset:/home/pcseg/data_root/SemanticKITTI/ pcseg
```

and follow the train instructions above