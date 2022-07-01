
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath, returnCalib=False, trainAll=False):
    train_list_f = open('dataloaders/stereo/lists/kitti2015_train180.list', 'r')
    test_list_f = open('dataloaders/stereo/lists/kitti2015_val20.list', 'r')
    train_list_ = train_list_f.readlines()
    test_list_ = test_list_f.readlines()
    if trainAll:
        train_list = train_list_ + test_list_
        test_list = train_list_ + test_list_
    else:
        train_list = train_list_
        test_list = test_list_

    left_train = []
    right_train = []
    disp_train_L = []
    calib_train = []
    for i in range(len(train_list)):
        name = train_list[i].split('.')[0] + '.png'
        left_train.append(filepath + '/image_2/' + name)
        right_train.append(filepath + '/image_3/' + name)
        disp_train_L.append(filepath + '/disp_occ_0/' + name)
        cal = name.split('_')[0] + '.txt'
        calib_train.append(filepath + '/calib_cam_to_cam/' + cal)

    left_val = []
    right_val = []
    disp_val_L = []
    calib_val = []
    for i in range(len(test_list)):
        name = test_list[i].split('.')[0] + '.png'
        left_val.append(filepath + '/image_2/' + name)
        right_val.append(filepath + '/image_3/' + name)
        disp_val_L.append(filepath + '/disp_occ_0/' + name)
        cal = name.split('_')[0] + '.txt'
        calib_val.append(filepath + '/calib_cam_to_cam/' + cal)

    return left_train, right_train, disp_train_L, calib_train, left_val, right_val, disp_val_L, calib_val
