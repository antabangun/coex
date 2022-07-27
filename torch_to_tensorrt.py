import cv2
import numpy as np
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ruamel.yaml import YAML

from dataloaders import KITTIRawLoader as KRL

from stereo import StereoTRT

import torch_tensorrt

import pdb

torch.backends.cudnn.benchmark = True


torch.set_grad_enabled(False)

config = 'cfg_coex.yaml'
version = 0  # CoEx

vid_date = "2011_09_26"
vid_num = '0093'
half_precision = False


def load_configs(path):
    cfg = YAML().load(open(path, 'r'))
    backbone_cfg = YAML().load(
        open(cfg['model']['stereo']['backbone']['cfg_path'], 'r'))
    cfg['model']['stereo']['backbone'].update(backbone_cfg)
    return cfg


def postprocess(outputs):
    cost, spx_pred = outputs

    b, _, h, w = spx_pred.shape

    corr, ind = cost.squeeze().sort(0, True)
    corr = F.softmax(corr[:2], 0)
    disp = ind[:2]

    disp_ = torch.mul(corr, disp)

    disp_4 = disp_[0] + disp_[1]
    disp_4 = disp_4.reshape(b, 1, disp_4.shape[-2], disp_4.shape[-1])

    x = F.pad(disp_4, (1,1,1,1))
    feat = torch.cat([
            x[:, :, :-2, :-2],
            x[:, :, :-2, 1:-1],
            x[:, :, :-2, 2:],
            x[:, :, 1:-1, :-2],
            x[:, :, 1:-1, 1:-1],
            x[:, :, 1:-1, 2:],
            x[:, :, 2:, :-2],
            x[:, :, 2:, 1:-1],
            x[:, :, 2:, 2:],
        ], 1)
    feat = torch.repeat_interleave(feat, 4, 2)
    feat = torch.repeat_interleave(feat, 4, 3)
    disp_1a = (feat*spx_pred)
    disp_1 = disp_1a.sum(1)

    disp_1 = disp_1*4  # + 1.5

    return disp_1[0]


if __name__ == '__main__':
    cfg = load_configs(
        './configs/stereo/{}'.format(config))
    cfg['model']['name'] = 'CoExTRT'
    cfg['model']['stereo']['name'] = 'CoExTRT'

    ckpt = '{}/CoEx/version_{}/checkpoints/last.ckpt'.format(
        'logs/stereo', version)
    cfg['stereo_ckpt'] = ckpt
    stereo = StereoTRT.load_from_checkpoint(cfg['stereo_ckpt'],
                                         strict=False,
                                         cfg=cfg).cuda()
    stereo.eval()

    if half_precision:
        enabled_precisions = {torch.float, torch.half}
        dtype = torch.half
        stereo = stereo.half()
    else:
        enabled_precisions = {torch.float}
        dtype = torch.float

    trt_model = torch_tensorrt.compile(
        stereo, inputs = [torch_tensorrt.Input((2, 3, 384, 1248), dtype=dtype)],
        enabled_precisions = enabled_precisions, # Run with FP32
        # workspace_size = 1 << 22
    )

    if not os.path.exists("zoo/tensorrt"):
        os.makedirs("zoo/tensorrt")

    torch.jit.save(trt_model, "zoo/tensorrt/trt_ts_module.ts")

    left_cam, right_cam = KRL.listfiles(
        cfg,
        vid_date,
        vid_num,
        True)
    cfg['training']['th'] = 0
    cfg['training']['tw'] = 0
    kitti_train = KRL.ImageLoader(
        left_cam, right_cam, cfg, training=True, demo=True)
    kitti_train = DataLoader(
        kitti_train, batch_size=1,
        num_workers=4, shuffle=False, drop_last=False)
    
    fps_list = np.array([])

    for i, batch in enumerate(kitti_train):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        imgL, imgR = batch['imgL'].cuda(), batch['imgR'].cuda()
        imgLRaw = batch['imgLRaw']

        im = torch.cat([imgL, imgR], 0)
        h, w = im.shape[-2:]
        h_pad = 384-h
        w_pad = 1248-w
        im = F.pad(im, (0, w_pad, 0, h_pad))

        end.record()
        torch.cuda.synchronize()
        runtime = start.elapsed_time(end)
        print('Data Preparation: {:.3f}'.format(runtime))

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # disp = postprocess(stereo(im.type(dtype)))[:h, :w]
        disp = postprocess(trt_model(im.type(dtype)))[:h, :w]
        
        end.record()
        torch.cuda.synchronize()
        runtime = start.elapsed_time(end)
        # print('Stereo runtime: {:.3f}'.format(runtime))

        fps = 1000/runtime
        fps_list = np.append(fps_list, fps)
        if len(fps_list) > 5:
            fps_list = fps_list[-5:]
        avg_fps = np.mean(fps_list)
        print('Stereo runtime: {:.3f}'.format(1000/avg_fps))

        disp_np = (2*disp).data.cpu().numpy().astype(np.uint8)
        disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_MAGMA)

        image_np = (imgLRaw[0].permute(1, 2, 0).numpy()).astype(np.uint8)
        
        out_img = np.concatenate((image_np, disp_np), 0)
        cv2.putText(
            out_img,
            "%.1f fps" % (avg_fps),
            (10, image_np.shape[0]+30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('img', out_img)
        cv2.waitKey(1)
