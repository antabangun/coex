
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath, returnCalib=False, trainAll=False):
    train_list_f = open('dataloaders/stereo/lists/kitti2012_train170.list', 'r')
    test_list_f = open('dataloaders/stereo/lists/kitti2012_val24.list', 'r')
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
        left_train.append(filepath + '/colored_0/' + name)
        right_train.append(filepath + '/colored_1/' + name)
        disp_train_L.append(filepath + '/disp_noc/' + name)
        cal = name.split('_')[0] + '.txt'
        calib_train.append(filepath + '/calib/' + cal)

    left_val = []
    right_val = []
    disp_val_L = []
    calib_val = []
    for i in range(len(test_list)):
        name = test_list[i].split('.')[0] + '.png'
        left_val.append(filepath + '/colored_0/' + name)
        right_val.append(filepath + '/colored_1/' + name)
        disp_val_L.append(filepath + '/disp_noc/' + name)
        cal = name.split('_')[0] + '.txt'
        calib_val.append(filepath + '/calib/' + cal)

    return left_train, right_train, disp_train_L, calib_train, left_val, right_val, disp_val_L, calib_val
