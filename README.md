# TRM: Phrase-level Temporal Relationship Mining for Temporal Sentence Localization
> Mindsopre implementation of Phrase-level Temporal Relationship Mining for Temporal Sentence Localization (AAAI2023).


## Requiments

-  ~~pytorch~~: mindspore, mindformers
```shell
# mindspore
conda install mindspore=2.2.11 -c mindspore -c conda-forge
# mindformers
git clone -b dev https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```
- h5py
- yacs
- terminaltables
- tqdm
- transformers

## Quick Start

### Data Preparation

We use the C3D feature for the ActivityNet Captions dataset. Please download from [here](http://activity-net.org/challenges/2016/download.html) and save as `dataset/ActivityNet/sub_activitynet_v1-3.c3d.hdf5`. We use the VGG feature provided by [2D-TAN](https://github.com/microsoft/VideoX) for the Charades-STA dataset, which can be downloaded from [here](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav/folder/137471415879). Please save it as `dataset/Charades-STA/vgg_rgb_features.hdf5`.


### Training

To train on the ActivityNet Captions dataset:
```bash
python train_net.py --config-file configs/activitynet.yaml OUTPUT_DIR outputs/activitynet
```

To train on the Charades-STA dataset:
```bash
sh scripts/charades_train.sh
python train_net.py --config-file configs/charades.yaml OUTPUT_DIR outputs/charade
```

You can change the options in the shell scripts, such as the GPU id, configuration file, et al.


### Inference

Run the following commands for ActivityNet Captions evaluation:

```shell
python test_net.py --config configs/activitynet.yaml --ckpt trm_act_e8_all.ckpt 
```
Run the following commands for Charades-STA evaluation:

```
python test_net.py --config configs/charades.yaml --ckpt trm_charades_e9_all.ckpt
```

### Parameters Transfer (Optional)
1. change the checkpoint path in `param_convert.py` 


2. run the command
```shell
python param_convert.py --config-file configs/charades.yaml #  configs/activitynet.yaml
```

### Pytorch version
