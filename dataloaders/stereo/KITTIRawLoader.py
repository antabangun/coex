import torch.utils.data as data

from PIL import Image
import os
import os.path
import glob
import random
import numpy as np
import cv2
from . import preprocess

import pdb


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def listfiles(cfg, date=None, num=None, test=False):

    train_datanames = ['kittiraw', 'kitti360']
    left_train = {'current': [], 'next': [], 'prev': []}
    right_train = {'current': [], 'next': [], 'prev': []}
    for train_dataname in train_datanames:
        if cfg['training']['train_on'][train_dataname]:
            left_train_, right_train_ = listtrainfiles(
                cfg['training']['paths'][train_dataname],
                train_dataname,
                date=date,
                num=num,
                test=test)

            left_train['current'] = left_train['current'] + left_train_['current']
            right_train['current'] = right_train['current'] + right_train_['current']
            left_train['prev'] = left_train['prev'] + left_train_['prev']
            right_train['prev'] = right_train['prev'] + right_train_['prev']
            left_train['next'] = left_train['next'] + left_train_['next']
            right_train['next'] = right_train['next'] + right_train_['next']

    if not test:
        val_datanames = ['kitti12', 'kitti15']
        left_val = {'current': []}
        right_val = {'current': []}
        disp_val = {'current': []}
        for val_dataname in val_datanames:
            left_val_, right_val_, disp_val_ = listvalfiles(cfg['training']['paths'][val_dataname])
            left_val['current'] = left_val['current'] + left_val_
            right_val['current'] = right_val['current'] + right_val_
            disp_val['current'] = disp_val['current'] + disp_val_

    # left_train['prev'] = left_train['prev'][200:201]
    # left_train['current'] = left_train['current'][200:201]
    # left_train['next'] = left_train['next'][200:201]
    # right_train['prev'] = right_train['prev'][200:201]
    # right_train['current'] = right_train['current'][200:201]
    # right_train['next'] = right_train['next'][200:201]

        return left_train, right_train, left_val, right_val, disp_val

    else:
        return left_train, right_train


def listtrainfiles(filepath, train_dataname, date=None, num=None, test=False):
    if train_dataname == 'kittiraw' and not test:
        with open(filepath + '/data_splits/eigen_zhou_files.txt') as f:
            data_splits = f.read()

        data_splits = data_splits.split('\n')
        data_splits_ = []
        for data_split in data_splits:
            data_splits_.append(data_split.split(' ')[0])
    else:
        data_splits_ = None

    left_fold = 'image_02' if 'raw' in filepath else 'image_00'
    right_fold = 'image_03' if 'raw' in filepath else 'image_01'
    data = 'data' if 'raw' in filepath else 'data_rect'
    if (train_dataname == 'kittiraw' and test
            and date is not None and num is not None):
        date_dir_ = date
    else:
        date_dir_ = '20' if 'raw' in filepath else '2d'

    left_train = {'current': [], 'next': [], 'prev': []}
    right_train = {'current': [], 'next': [], 'prev': []}
    for date_dir in os.listdir(filepath):
        if date_dir_ in date_dir:
            datepath = os.path.join(filepath, date_dir)
            for time_dir in os.listdir(os.path.join(datepath)):
                if num is not None:
                    if num not in time_dir:
                        continue
                if 'sync' in time_dir:
                    timepath = os.path.join(datepath, time_dir)
                    datadir_left = '{}/{}/{}'.format(timepath, left_fold, data)
                    datadir_right = '{}/{}/{}'.format(timepath, right_fold, data)

                    img_names = os.listdir(datadir_left)
                    img_names.sort()

                    for i in range(len(img_names)-2):
                        if (train_dataname == 'kitti360' or data_splits_ is None or
                            os.path.join(*os.path.join(datadir_left, img_names[i]).split('/')[-5:]) in data_splits_):

                            left_train['prev'].append(os.path.join(datadir_left, img_names[i]))
                            left_train['current'].append(os.path.join(datadir_left, img_names[i+1]))
                            left_train['next'].append(os.path.join(datadir_left, img_names[i+2]))

                            right_train['prev'].append(os.path.join(datadir_right, img_names[i]))
                            right_train['current'].append(os.path.join(datadir_right, img_names[i+1]))
                            right_train['next'].append(os.path.join(datadir_right, img_names[i+2]))

    return left_train, right_train


def listvalfiles(filepath):

    left_fold = '/image_2/' if 'kitti15' in filepath else '/colored_0/'
    right_fold = '/image_3/' if 'kitti15' in filepath else '/colored_1/'
    disp_fold = '/disp_occ_0/' if 'kitti15' in filepath else '/disp_noc/'

    image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]
    image.sort()

    left_val = [filepath+left_fold+img for img in image]
    right_val = [filepath+right_fold+img for img in image]
    disp_val = [filepath+disp_fold+img for img in image]
    left_val.sort()
    right_val.sort()
    disp_val.sort()

    return left_val, right_val, disp_val


def default_loader(path):
    return cv2.imread(path)


def disparity_loader(path):
    return Image.open(path)


class ImageLoader(data.Dataset):
    def __init__(self, left, right, cfg, disp=None, training=True, demo=False,
                 loader=default_loader, dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp = disp
        self.training = training
        self.demo = demo
        self.th, self.tw = cfg['training']['th'], cfg['training']['tw']
        self.with_context = cfg['training']['with_context']
        self.extract_feature = cfg['training']['extract_feature']
        if self.extract_feature:
            self.extractor = load_feature(cfg['training']['feature_extractor'])
            self.feature_matcher = cv2.BFMatcher(
                cv2.NORM_HAMMING, crossCheck=False)

        self.loader = loader
        self.dploader = dploader

    def __getitem__(self, index):
        batch = dict()

        left = self.left['current'][index]
        right = self.right['current'][index]
        left_img = self.loader(left)
        right_img = self.loader(right)

        h, w, _ = left_img.shape

        processed = preprocess.get_transform(augment=False)

        calib_path = None
        if 'KITTI_raw' in self.left['current'][index]:
            calib_path = os.path.join(
                *self.left['current'][index].split('/')[:-4],
                'calib_cam_to_cam.txt')

        elif 'KITTI-360' in self.left['current'][index]:
            calib_path = os.path.join(
                *self.left['current'][index].split('/')[:-4],
                'calibration/perspective.txt')

        if calib_path is not None:

            file = open(calib_path, "r")
            cal = file.read()

            P2 = np.array(cal.split('\n')[-10].split(' ')[1:]).astype(np.float32)
            P3 = np.array(cal.split('\n')[-2].split(' ')[1:]).astype(np.float32)
            P2 = P2.reshape(3, 4)
            P3 = P3.reshape(3, 4)

            K, baseline = self.kitti_calib(P2, P3)

            calib = {'P2': P2, 'P3': P3, 'K': K, 'baseline': baseline}
            batch['calib'] = calib

        if self.training:

            if not self.demo:
                th, tw = self.th, self.tw
            else:
                th, tw = h, w

            if not self.demo:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
            else:
                x1, y1 = 0, 0

            left_img = left_img[y1: y1+th, x1: x1+tw, :]
            right_img = right_img[y1: y1+th, x1: x1+tw, :]

            left_img_p = processed(left_img)
            right_img_p = processed(right_img)

            left_img_ = np.transpose(left_img, (2, 0, 1)).astype(np.float32)
            right_img_ = np.transpose(right_img, (2, 0, 1)).astype(np.float32)

            batch['imgL'], batch['imgR'] = left_img_p, right_img_p
            batch['imgLRaw'], batch['imgRRaw'] = left_img_, right_img_
            batch['x1'], batch['y1'] = x1, y1

            if self.with_context:
                left_prev = self.left['prev'][index]
                right_prev = self.right['prev'][index]
                left_img_prev = self.loader(left_prev)[y1: y1+th, x1: x1+tw, :]
                right_img_prev = self.loader(right_prev)[y1: y1+th, x1: x1+tw, :]

                left_img_prev_p = processed(left_img_prev)
                right_img_prev_p = processed(right_img_prev)
                left_img_prev_ = np.transpose(
                    left_img_prev, (2, 0, 1)).astype(np.float32)
                right_img_prev_ = np.transpose(
                    right_img_prev, (2, 0, 1)).astype(np.float32)

                left_next = self.left['next'][index]
                right_next = self.right['next'][index]
                left_img_next = self.loader(left_next)[y1: y1+th, x1: x1+tw, :]
                right_img_next = self.loader(right_next)[y1: y1+th, x1: x1+tw, :]

                left_img_next_p = processed(left_img_next)
                right_img_next_p = processed(right_img_next)
                left_img_next_ = np.transpose(
                    left_img_next, (2, 0, 1)).astype(np.float32)
                right_img_next_ = np.transpose(
                    right_img_next, (2, 0, 1)).astype(np.float32)

                contexts = {
                    'imgLPrev': left_img_prev_p, 'imgLPrevRaw': left_img_prev_,
                    'imgRPrev': right_img_prev_p, 'imgRPrevRaw': right_img_prev_,
                    'imgLNext': left_img_next_p, 'imgLNextRaw': left_img_next_,
                    'imgRNext': right_img_next_p, 'imgRNextRaw': right_img_next_,
                }

                if self.extract_feature:
                    kp_curr, des_curr = self.extractor.detectAndCompute(
                        left_img, None)
                    kp_prev, des_prev = self.extractor.detectAndCompute(
                        left_img_prev, None)
                    kp_next, des_next = self.extractor.detectAndCompute(
                        left_img_next, None)

                    pxscp_curr, pxscp_prev = match(
                        kp_curr, des_curr, kp_prev, des_prev,
                        self.feature_matcher)
                    pxscn_curr, pxscn_next = match(
                        kp_curr, des_curr, kp_next, des_next,
                        self.feature_matcher)

                    pxs = {
                        'pxscp_curr': pxscp_curr, 'pxscp_prev': pxscp_prev,
                        'pxscn_curr': pxscn_curr, 'pxscn_next': pxscn_next
                    }
                    contexts.update(pxs)

                batch['contexts'] = contexts

        else:
            # left_img, right_img = left_img[:352, :1216], right_img[:352, :1216]

            left_img_p = processed(left_img)
            right_img_p = processed(right_img)

            left_img_ = np.transpose(left_img, (2, 0, 1)).astype(np.float32)
            right_img_ = np.transpose(right_img, (2, 0, 1)).astype(np.float32)

            batch['imgL'], batch['imgR'] = left_img_p, right_img_p
            batch['imgLRaw'], batch['imgRRaw'] = left_img_, right_img_

            disp = self.disp['current'][index]
            dataL = self.dploader(disp)
            dataL = np.ascontiguousarray(dataL, dtype=np.float32)/256
            batch['dispL'] = dataL

            # # Load context images
            # left_prev = self.left['prev'][index]
            # left_img_prev = self.loader(left_prev)
            # left_img_prev = left_img_prev[:352, :1216]

            # left_img_prev_p = processed(left_img_prev)
            # left_img_prev_ = np.transpose(
            #     left_img_prev, (2, 0, 1)).astype(np.float32)

            # contexts = {
            #     'imgLPrev': left_img_prev_p, 'imgLPrevRaw': left_img_prev_,
            # }

            # batch['contexts'] = contexts

        return batch

    def __len__(self):
        return len(self.left['current'])

    def kitti_calib(self, P2, P3):
        t2 = np.array([P2[0, -1]/P2[0, 0], P2[1, -1]/P2[1, 1], P2[2, -1]])
        t3 = np.array([P3[0, -1]/P3[0, 0], P3[1, -1]/P3[1, 1], P3[2, -1]])
        t = t2-t3
        baseline = np.linalg.norm(t, 2)

        K = P2[:, :-1]

        return K, baseline


def load_feature(cfg):
    if cfg['type'] == 'ORB':
        extractor = cv2.ORB_create(
            nfeatures=cfg['max_num'],
            edgeThreshold=cfg['thresh'], fastThreshold=cfg['thresh'])
    elif cfg['type'] == 'AKAZE':
        extractor = cv2.AKAZE_create(threshold=cfg['thresh'])

    return extractor


def match(kp1, des1, kp2, des2, matcher):
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    max_num = 1000
    pxs1, pxs2 = -np.ones((max_num, 2)), -np.ones((max_num, 2))
    matches = []
    pxs1_, pxs2_ = [], []
    for (m, n) in raw_matches:
        if m.distance < 0.70*n.distance:
            matches.append(m)
            pxs1_.append(kp1[m.queryIdx].pt)
            pxs2_.append(kp2[n.trainIdx].pt)
    pxs1_, pxs2_ = np.array(pxs1_), np.array(pxs2_)

    px1_len = min(max_num, pxs1_.shape[0])
    px2_len = min(max_num, pxs2_.shape[0])
    pxs1[:px1_len] = pxs1_[:px1_len]
    pxs2[:px2_len] = pxs2_[:px2_len]
    return pxs1, pxs2
