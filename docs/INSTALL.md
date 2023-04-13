# Installation

### General Requirements

This codebase is tested with `torch==1.10.0` and `torchvision==0.11.0`, with `CUDA 11.3` and `gcc 7.3.0`. In order to successfully reproduce the results reported in our paper, we recommend you to follow the exact same configuation with us. However, similar versions that came out lately should be good as well.


### Step 1: Create Enviroment
```Shell
conda create -n pcseg python=3.7
```

### Step 2: Activate Enviroment
```Shell
conda activate pcseg
```

### Step 3: Install PyTorch
```Shell
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

### Step 4: Install Necessary Libraries
#### 4.1 - [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit)
:oncoming_automobile: **Note:** This toolkit is **required** in order to run experiments on the [nuScenes](https://www.nuscenes.org/nuscenes) dataset.
```Shell
pip install nuscenes-devkit 
```

#### 4.2 - [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter)
```Shell
conda install pytorch-scatter -c pyg
```

#### 4.3 - [TorchSparse](https://github.com/mit-han-lab/torchsparse)
**Note:** The following steps are **required** in order to use the `voxel` and `fusion` backbones in this codebase.

- Make a directory named `torchsparse_dir`
```Shell
cd package/
mkdir torchsparse_dir/
```

- Unzip the `.zip` files in `package/`
```Shell
unzip sparsehash.zip
unzip torchsparse.zip

mv sparsehash-master/ sparsehash/
```

- Setup `sparsehash` (Note that `${ROOT}` should be your home path to the `PCSeg` folder)
```Shell
cd sparsehash/
./configure --prefix=/${ROOT}/PCSeg/package/torchsparse_dir/sphash/
```
```Shell
make
```
```Shell
make install
```

- Compile `torchsparse`
```Shell
cd ..
pip install ./torchsparse
```

- It takes a while to build wheels. After successfully building `torchsparse`, you should see the following:
```Shell
Successfully built torchsparse
Installing collected packages: torchsparse
Successfully installed torchsparse-1.4.0
```
#### 4.4 - Range Image Library
```Shell
cd package/
```
- Unzip the `range_lib.zip` file in `package/`
```Shell
unzip range_lib.zip
cd range_lib/
python setup.py install
```
- After successfully building `range_lib`, you should see the following:
```Shell
Processing dependencies for rangelib==1.0.0
Finished processing dependencies for rangelib==1.0.0
```
#### 4.5 - Other Packages
```Shell
pip install pyyaml easydict numba torchpack strictyaml llvmlite easydict scikit-image tqdm SharedArray prettytable opencv-python
```
```Shell
pip uninstall setuptools
pip install setuptools==59.5.0
```

#### 4.6 - Register PCSeg

Inside `PCSeg` directory:

```Shell
python setup.py develop
```

