# Coex

PyTorch implementation of our paper: 

Correlate-and-Excite: Real-Time Stereo Matching via Guided Cost Volume Excitation

Authors: Antyanta Bangunharcana, Jae Won Cho, Seokju Lee, In So Kweon, Kyung-Soo Kim, Soohyun Kim

We propose a Guided Cost volume Excitation (GCE) and top-k soft-argmax disparity regression for real-time and accurate stereo matching. 


## Installation

We recommend using [conda](https://www.anaconda.com/distribution/) for installation: 

```shell
conda env create -f environment.yml
```


## Datasets
```
data
├── KITTI
│   ├── kitti_2012
│   │   └── data_stereo_flow
│   ├── kitti_2015
│   │   └── data_scene_flow
└── SceneFlow
    ├── Driving
    │   ├── disparity
    │   └── frames_finalpass
    ├── FlyingThings3D
    │   ├── disparity
    │   └── frames_finalpass
    ├── Monkaa
    │   ├── disparity
    │   └── frames_finalpass
    ├── kitti12
    │   ├── disparity
    │   └── frames_finalpass
    └── kitti15
        ├── disparity
        └── frames_finalpass
```

## Demo on KITTI raw data
### Model zoo


## Training your own model


## Citation

If you find our work useful in your research, please consider citing our paper:

## Acknowledgements

Part of the code is adopted from previous works: [PSMNet](https://github.com/JiaRenChang/PSMNet), [AANet](https://github.com/haofeixu/aanet)