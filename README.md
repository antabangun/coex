# CoEx

PyTorch implementation of our paper: 


**Correlate-and-Excite: Real-Time Stereo Matching via Guided Cost Volume Excitation**  
*Authors: Antyanta Bangunharcana, Jae Won Cho, Seokju Lee, In So Kweon, Kyung-Soo Kim, Soohyun Kim*  
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2021

\[[Project page](https://antabangun.github.io/projects/CoEx/)\]

We propose a Guided Cost volume Excitation (GCE) and top-k soft-argmax disparity regression for real-time and accurate stereo matching. 

## Contents
- [Installation](#installation)
- [Datasets](#datasets)
    - [Data for demo](#data-for-demo)
    - [If you want to re-train the models](#if-you-want-to-re-train-the-models)
- [Demo on KITTI raw data](#demo-on-kitti-raw-data)
    - [Model zoo](#model-zoo)
- [Re-training the models](#re-training-the-models)

##Installation

We recommend using [conda](https://www.anaconda.com/distribution/) for installation: 
```shell
conda env create -f environment.yml
```
Then activate the newly created env
```shell
conda activate coex
```

## Datasets

```
data
└── datasets
    ├── KITTI_raw
    |   ├── 2011_09_26
    |   │   ├── 2011_09_26_drive_0001_sync
    |   │   ├── 2011_09_26_drive_0002_sync
    |   |       :
    |   |
    |   ├── 2011_09_28
    |   │   ├── 2011_09_28_drive_0001_sync
    |   │   └── 2011_09_28_drive_0002_sync
    |   |       :
    |   |   :    
    |
    └── SceneFlow
        ├── driving
        │   ├── disparity
        │   └── frames_finalpass
        ├── flyingthings3d_final
        │   ├── disparity
        │   └── frames_finalpass
        ├── monkaa
        │   ├── disparity
        │   └── frames_finalpass
        ├── kitti12
        │   ├── testing
        │   └── training
        └── kitti15
            ├── testing
            └── training
```

### Data for demo

For a demo of our code on the KITTI dataset, download the "\[synced+rectified data\]" from [raw KITTI data](http://www.cvlibs.net/datasets/kitti/raw_data.php). Unzip and place the extracted folders following the above directory tree. 
       
### If you want to re-train the models
**Sceneflow dataset**  
Download the *finalpass* data of the [Sceneflow dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) as well as the *Disparity* data.

**KITTI 2015**  
Download [kitti15](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) dataset, and unzip data_scene_flow.zip, rename it as kitti15, and move it into SceneFlow directory as shown in the tree above.

**KITTI 2012**  
Download [kitti12](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) dataset. Unzip data_stereo_flow.zip, rename it as kitti12, and move it into SceneFlow directory as shown in the tree above.

The paths where the code can find the dataset can be modified inside the [config files](/configs/stereo/cfg_coex.yaml), but make sure the directory names of driving, flyingthings3d_final, monkaa.

## Demo on KITTI raw data
### Model zoo


## Re-training the models


## Citation

If you find our work useful in your research, please consider citing our paper

## Acknowledgements

Part of the code is adopted from previous works: [PSMNet](https://github.com/JiaRenChang/PSMNet), [AANet](https://github.com/haofeixu/aanet)