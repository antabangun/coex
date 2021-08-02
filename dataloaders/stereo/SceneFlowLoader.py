import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from albumentations import Compose, OneOf
from PIL import Image, ImageOps
from . import preprocess 
from .stereo_albumentation import RandomShiftRotate, GaussNoiseStereo, RGBShiftStereo, \
    RandomBrightnessContrastStereo, random_crop, horizontal_flip
from . import transforms
from .transforms import RandomColor
from . import readpfm as rp
import numpy as np
import cv2

import pdb

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return cv2.imread(path)
    # return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path)


class ImageLoader(data.Dataset):
    def __init__(self, left, right,
                 focal, left_disparity, training,
                 loader=default_loader, dploader=disparity_loader,
                 th=256, tw=512):

        self.left = left
        self.right = right
        self.focal = focal
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.th = th
        self.tw = tw
        self.training = training

    def __getitem__(self, index):
        batch = dict()

        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]
        disp_R = disp_L.replace('left', 'right')
        focal = self.focal[index]*30

        K = np.array([[focal, 0, 479.5],
                      [0, focal, 269.5],
                      [0, 0, 1]])
        K = torch.Tensor(K)

        left_img = self.loader(left)
        right_img = self.loader(right)

        dataL, scaleL = self.dploader(disp_L)
        dataR, scaleR = self.dploader(disp_R)

        if disp_L.split('/')[-5] == 'flyingthings3d':
            dataL = -dataL
            dataR = -dataR
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)
        dataR = np.ascontiguousarray(dataR, dtype=np.float32)

        if self.training:
            left_img, right_img, dataL = horizontal_flip(left_img, right_img, dataL, dataR)

            h, w = left_img.shape[:2]

            x1 = random.randint(0, w - self.tw)
            y1 = random.randint(0, h - self.th)

            left_img = left_img[y1: y1 + self.th, x1: x1 + self.tw]
            right_img = right_img[y1: y1 + self.th, x1: x1 + self.tw]

            dataL = dataL[y1:y1 + self.th, x1:x1 + self.tw]

            img = {'left': left_img, 'right': right_img}
            # img = self.train_aug(img)

            left_img, right_img = img['left'], img['right']

            processed = preprocess.get_transform(augment=True)
            left_img = processed(left_img)
            right_img = processed(right_img)

            batch['imgL'], batch['imgR'], batch['disp_true'] = left_img, right_img, dataL
            batch['K'], batch['x1'], batch['y1'] = K, x1, y1

            return batch
        else:
            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            batch['imgL'], batch['imgR'], batch['disp_true'] = left_img, right_img, dataL
            batch['K'] = K

            return batch

    def __len__(self):
        return len(self.left)

    def train_aug(self, img):
        transformation = Compose([
                # RandomShiftRotate(always_apply=True),
                RGBShiftStereo(always_apply=True, p_asym=0.3),
                OneOf([
                    GaussNoiseStereo(always_apply=True, p_asym=1),
                    RandomBrightnessContrastStereo(always_apply=True, p_asym=0.5)
                ], p=1)
                ])
        return transformation(**img)

        # transformation = transforms.Compose([
        #         RandomColor()
        #         ])
        # return transformation(img)
