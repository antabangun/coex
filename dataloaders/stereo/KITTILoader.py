import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from albumentations import Compose, OneOf
from PIL import Image, ImageOps
import numpy as np
from . import preprocess 
from .stereo_albumentation import RandomShiftRotate, GaussNoiseStereo, RGBShiftStereo, \
    RandomBrightnessContrastStereo, random_crop, horizontal_flip
from . import transforms
from .transforms import RandomColor
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
    #return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)


class ImageLoader(data.Dataset):
    def __init__(self, left, right, left_disparity, calib, th=256, tw=512, shift=0, training=True, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.calib = calib
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.th = th
        self.tw = tw
        self.shift = shift

    def __getitem__(self, index):
        batch = dict()

        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]
        calib = self.calib[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)
        file = open(calib,"r")
        cal = file.read()
        if calib.find('kitti15')==-1:
            P2 = np.array(cal.split('\n')[2].split(' ')[1:]).astype(np.float32)
            P3 = np.array(cal.split('\n')[3].split(' ')[1:]).astype(np.float32)
        else:
            P2 = np.array(cal.split('\n')[-10].split(' ')[1:]).astype(np.float32)
            P3 = np.array(cal.split('\n')[-2].split(' ')[1:]).astype(np.float32)
        P2 = P2.reshape(3,4)
        P3 = P3.reshape(3,4)

        calib = self.kitti_calib(P2,P3)

        dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

        # if 'kitti15' in left:
        #     disp_R = disp_L.replace('occ_0','occ_1')
        #     dataR = self.dploader(disp_R)
        #     dataR = np.ascontiguousarray(dataR,dtype=np.float32)/256

        if self.training:  
            # if 'kitti15' in left:
            #     left_img, right_img, dataL = horizontal_flip(left_img, right_img, dataL, dataR)

            pad_h, pad_w = 384-left_img.shape[0], 1280-left_img.shape[1]

            left_img = np.pad(left_img,((0,pad_h),(0,pad_w),(0,0)))
            right_img = np.pad(right_img,((0,pad_h),(0,pad_w),(0,0)))

            h,w,_ = left_img.shape
            th, tw = self.th, self.tw

            shift = random.randint(-self.shift,self.shift)
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            # if x1 + shift < 0 or  x1 + shift + tw > w:
            shift = 0

            left_img_raw = left_img[y1:y1+th,x1+shift:x1+shift+tw,:]
            right_img_raw = right_img[y1:y1+th,x1:x1+tw,:]

            imL_lab = cv2.cvtColor(
                left_img_raw,#cv2.resize(left_img,None,None,0.25,0.25),
                cv2.COLOR_BGR2LAB)

            dataL = np.pad(dataL[:,:,np.newaxis],((0,pad_h),(0,pad_w),(0,0)))[:,:,0]
            dataL = dataL[y1:y1 + th, x1 + shift:x1 + tw + shift]
            dataL = dataL - shift

            img = {'left':left_img_raw,'right':right_img_raw}
            # img = self.train_aug(img)

            left_img_raw, right_img_raw = img['left'], img['right']

            processed = preprocess.get_transform(augment=False)  
            left_img   = processed(left_img_raw)
            right_img  = processed(right_img_raw)

            left_img_raw = np.transpose(left_img_raw,(2,0,1)).astype(np.float32)
            right_img_raw = np.transpose(right_img_raw,(2,0,1)).astype(np.float32)

            batch['imgL'], batch['imgR'], batch['disp_true'] = left_img, right_img, dataL
            batch['imgLRaw'], batch['imgRRaw'], batch['imgLLab'] = left_img_raw, right_img_raw, imL_lab
            batch['calib'], batch['x1'], batch['y1'] = calib, x1, y1

            return batch
        else:
            h,w,_ = left_img.shape
            imL = left_img
            pad_h, pad_w = 384-h, 1280-w

            # left_img_raw = left_img[h-352:h,w-1216:w,:]
            # right_img_raw = right_img[h-352:h,w-1216:w,:]
            left_img_raw = left_img  # np.pad(left_img,((0,pad_h),(0,pad_w),(0,0)))
            right_img_raw = right_img  # np.pad(right_img,((0,pad_h),(0,pad_w),(0,0)))

            imL_lab = cv2.cvtColor(
                left_img_raw,#cv2.resize(left_img,None,None,0.25,0.25),
                cv2.COLOR_BGR2LAB)

            # dataL = dataL.crop((w-1216, h-352, w, h))
            # dataL = np.pad(dataL,((0,pad_h),(0,pad_w)))

            processed = preprocess.get_transform(augment=False)  
            left_img       = processed(left_img_raw)
            right_img      = processed(right_img_raw)

            batch['imgL'], batch['imgR'], batch['disp_true'] = left_img, right_img, dataL
            batch['imgLLab'] = imL_lab
            batch['calib'] = calib

            return batch

    def __len__(self):
        return len(self.left)

    def train_aug(self, img):
        transformation = Compose([
                RGBShiftStereo(always_apply=True, p_asym=0.5),
                RandomBrightnessContrastStereo(always_apply=True, p_asym=0.5)
                ])
        return transformation(**img)

        # transformation = transforms.Compose([
        #         RandomColor()
        #         ])
        # return transformation(img)

    def kitti_calib(self, P2, P3):
        t2 = np.array([P2[0,-1]/P2[0,0],P2[1,-1]/P2[1,1],P2[2,-1]])
        t3 = np.array([P3[0,-1]/P3[0,0],P3[1,-1]/P3[1,1],P3[2,-1]])
        t = t2-t3
        baseline = np.linalg.norm(t,2)

        K = P2[:,:-1]

        return {'K':K,'baseline':baseline}
