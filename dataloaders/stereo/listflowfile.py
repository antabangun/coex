import torch.utils.data as data

from PIL import Image
import os
import os.path
import glob
import numpy as np

import pdb

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    all_focal = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []
    test_focal = []

    # # MONKAAS ##
    monkaa_path = os.path.join(filepath, 'monkaa', 'frames_finalpass')
    monkaa_disp = os.path.join(filepath, 'monkaa', 'disparity')

    monkaa_dir = os.listdir(monkaa_path)

    for dd in monkaa_dir:
        for im in os.listdir(os.path.join(monkaa_path, dd, 'left')):
            if is_image_file(os.path.join(monkaa_path, dd, 'left', im)):
                all_left_img.append(os.path.join(monkaa_path, dd, 'left', im))
                all_left_disp.append(os.path.join(monkaa_disp, dd, 'left', im.split(".")[0]+'.pfm'))
                all_focal.append(35)

        for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
            if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
                all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)

    # # FLYINGTHINGS 3D ##
    flying_path = os.path.join(filepath, 'flyingthings3d_final', 'frames_finalpass', 'TRAIN')
    flying_disp = os.path.join(filepath, 'flyingthings3d_final', 'disparity', 'TRAIN')

    flying_dir = os.listdir(flying_path)

    left_paths, right_paths, disp_paths = [], [], []
    for dd in flying_dir:
        for nn in os.listdir(os.path.join(flying_path, dd)):
            for im in os.listdir(os.path.join(flying_path, dd, nn, 'left')):
                if is_image_file(os.path.join(flying_path, dd, nn, 'left', im)):
                    left_paths.append(os.path.join(flying_path, dd, nn, 'left', im))
                    disp_paths.append(os.path.join(flying_disp, dd, nn, 'left', im.split(".")[0]+'.pfm'))
            for im in os.listdir(os.path.join(flying_path, dd, nn, 'right')):
                if is_image_file(os.path.join(flying_path, dd, nn, 'right', im)):
                    right_paths.append(os.path.join(flying_path, dd, nn, 'right', im))

    # left_paths = glob.glob(flying_path+'/*/*/left/*.png')
    # right_paths = glob.glob(flying_path+'/*/*/right/*.png')
    # disp_paths = glob.glob(flying_disp+'/*/*/left/*.pfm')

    flying_path_val = os.path.join(filepath, 'flyingthings3d_final', 'frames_finalpass', 'TEST')
    flying_disp_val = os.path.join(filepath, 'flyingthings3d_final', 'disparity', 'TEST')

    flying_dir_val = os.listdir(flying_path_val)

    left_paths_val, right_paths_val, disp_paths_val = [], [], []
    for dd in flying_dir_val:
        for nn in os.listdir(os.path.join(flying_path_val, dd)):
            for im in os.listdir(os.path.join(flying_path_val, dd, nn, 'left')):
                if is_image_file(os.path.join(flying_path_val, dd, nn, 'left', im)):
                    left_paths_val.append(os.path.join(flying_path_val, dd, nn, 'left', im))
                    disp_paths_val.append(os.path.join(flying_disp_val, dd, nn, 'left', im.split(".")[0]+'.pfm'))
            for im in os.listdir(os.path.join(flying_path_val, dd, nn, 'right')):
                if is_image_file(os.path.join(flying_path_val, dd, nn, 'right', im)):
                    right_paths_val.append(os.path.join(flying_path_val, dd, nn, 'right', im))

    # left_paths_val = glob.glob(flying_path_val+'/*/*/left/*.png')
    # right_paths_val = glob.glob(flying_path_val+'/*/*/right/*.png')
    # disp_paths_val = glob.glob(flying_disp_val+'/*/*/left/*.pfm')

    left_paths.sort()
    right_paths.sort()
    disp_paths.sort()
    left_paths_val.sort()
    right_paths_val.sort()
    disp_paths_val.sort()

    focal = (35*np.ones(len(left_paths), dtype=int)).tolist()
    all_left_img = all_left_img + left_paths
    all_right_img = all_right_img + right_paths
    all_left_disp = all_left_disp + disp_paths
    all_focal = all_focal + focal

    focal_val = (35*np.ones(len(left_paths_val), dtype=int)).tolist()
    test_left_img = test_left_img + left_paths_val
    test_right_img = test_right_img + right_paths_val
    test_left_disp = test_left_disp + disp_paths_val
    test_focal = test_focal + focal_val

    # # DRIVING ##
    driving_dir = os.path.join(filepath, 'driving', 'frames_finalpass')
    driving_disp = os.path.join(filepath, 'driving', 'disparity')

    subdir1 = ['35mm_focallength', '15mm_focallength']
    subdir2 = ['scene_backwards', 'scene_forwards']
    subdir3 = ['fast', 'slow']

    for i in subdir1:
        for j in subdir2:
            for k in subdir3:
                imm_l = os.listdir(os.path.join(driving_dir, i, j, k, 'left'))
                for im in imm_l:
                    if is_image_file(os.path.join(driving_dir, i, j, k, 'left', im)):
                        all_left_img.append(os.path.join(driving_dir, i, j, k, 'left', im))
                        if i == '35mm_focallength':
                            all_focal.append(35)
                        else:
                            all_focal.append(15)
                        all_left_disp.append(os.path.join(driving_disp, i, j, k, 'left', im.split(".")[0]+'.pfm'))

                    if is_image_file(os.path.join(driving_dir, i, j, k, 'right', im)):
                        all_right_img.append(os.path.join(driving_dir, i, j, k, 'right', im))

    return all_left_img, all_right_img, all_left_disp, all_focal, test_left_img, test_right_img, test_left_disp, test_focal
