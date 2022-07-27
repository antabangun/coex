import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from ruamel.yaml import YAML

from dataloaders import KITTIRawLoader as KRL

import torch_tensorrt


config = 'cfg_coex.yaml'
vid_date = "2011_09_26"
vid_num = '0093'


def load_configs(path):
    cfg = YAML().load(open(path, 'r'))
    backbone_cfg = YAML().load(
        open(cfg['model']['stereo']['backbone']['cfg_path'], 'r'))
    cfg['model']['stereo']['backbone'].update(backbone_cfg)
    return cfg


if __name__ == '__main__':
    cfg = load_configs(
        './configs/stereo/{}'.format(config))
    stereo = torch.jit.load('zoo/tensorrt/trt_ts_module.ts')
