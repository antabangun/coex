import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import cv2
from . import preprocess 

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def listfiles(filepath, dataname):

    left_fold = '/image_2/' if dataname == 'kitti15' else '/colored_0/'
    right_fold = '/image_3/' if dataname == 'kitti15' else '/colored_1/'

    image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]
    image.sort()

    left_test = [filepath+left_fold+img for img in image]
    right_test = [filepath+right_fold+img for img in image]

    calib_path = '/calib_cam_to_cam/' if dataname == 'kitti15' else '/calib/'
    f = [txt for txt in os.listdir(filepath+calib_path)]
    f.sort()

    calib_test = [filepath+calib_path+f_ for f_ in f]

    return left_test, right_test, calib_test


def default_loader(path):
    return cv2.imread(path)
    # return Image.open(path).convert('RGB')


def disparity_loader(path):
    return Image.open(path)


class ImageLoader(data.Dataset):
    def __init__(self, left, right, calib, loader=default_loader, dploader=disparity_loader):

        self.left = left
        self.right = right
        self.calib = calib
        self.loader = loader
        self.dploader = dploader

    def __getitem__(self, index):
        batch = dict()

        left = self.left[index]
        right = self.right[index]
        calib = self.calib[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        file = open(calib, "r")
        cal = file.read()
        if calib.find('kitti15') == -1:
            P2 = np.array(cal.split('\n')[2].split(' ')[1:]).astype(np.float32)
            P3 = np.array(cal.split('\n')[3].split(' ')[1:]).astype(np.float32)
            dataname = 'kitti12_test'
        else:
            P2 = np.array(cal.split('\n')[-10].split(' ')[1:]).astype(np.float32)
            P3 = np.array(cal.split('\n')[-2].split(' ')[1:]).astype(np.float32)
            dataname = 'kitti15_test'
        filename = self.left[index].split('/')[-1].split('.')[0]
        P2 = P2.reshape(3, 4)
        P3 = P3.reshape(3, 4)

        calib = self.kitti_calib(P2, P3)

        processed = preprocess.get_transform(augment=False)
        left_img = processed(left_img)
        right_img = processed(right_img)

        batch['imgL'], batch['imgR'] = left_img, right_img
        batch['calib'], batch['dataname'], batch['filename'] = calib, dataname, filename

        return batch

    def __len__(self):
        return len(self.left)

    def kitti_calib(self, P2, P3):
        t2 = np.array([P2[0, -1]/P2[0, 0], P2[1, -1]/P2[1, 1], P2[2, -1]])
        t3 = np.array([P3[0, -1]/P3[0, 0], P3[1, -1]/P3[1, 1], P3[2, -1]])
        t = t2-t3
        baseline = np.linalg.norm(t, 2)

        K = P2[:, :-1]

        return {'K': K, 'baseline': baseline}
