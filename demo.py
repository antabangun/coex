import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from ruamel.yaml import YAML

from dataloaders import KITTIRawLoader as KRL

from stereo import Stereo

torch.backends.cudnn.benchmark = True


torch.set_grad_enabled(False)

config = 'cfg_coex.yaml'
version = 0  # CoEx

vid_date = "2011_09_26"
vid_num = '0093'
half_precision = True


def load_configs(path):
    cfg = YAML().load(open(path, 'r'))
    backbone_cfg = YAML().load(
        open(cfg['model']['stereo']['backbone']['cfg_path'], 'r'))
    cfg['model']['stereo']['backbone'].update(backbone_cfg)
    return cfg


if __name__ == '__main__':
    cfg = load_configs(
        './configs/stereo/{}'.format(config))

    ckpt = '{}/{}/version_{}/checkpoints/last.ckpt'.format(
        'logs/stereo', cfg['model']['name'], version)
    cfg['stereo_ckpt'] = ckpt
    pose_ssstereo = Stereo.load_from_checkpoint(cfg['stereo_ckpt'],
                                                strict=False,
                                                cfg=cfg).cuda()

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

    pose_ssstereo.eval()
    for i, batch in enumerate(kitti_train):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        imgL, imgR = batch['imgL'].cuda(), batch['imgR'].cuda()
        imgLRaw = batch['imgLRaw']
        imgLRaw = imgLRaw.cuda()

        end.record()
        torch.cuda.synchronize()
        runtime = start.elapsed_time(end)
        print('Data Preparation: {:.3f}'.format(runtime))

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=half_precision):
                disp = pose_ssstereo(imgL, imgR, False)
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

        disp_np = (2*disp[0]).data.cpu().numpy().astype(np.uint8)
        disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_PLASMA)

        image_np = (imgLRaw[0].permute(1, 2, 0).data.cpu().numpy()).astype(np.uint8)
        
        out_img = np.concatenate((image_np, disp_np), 0)
        cv2.putText(
            out_img,
            "%.1f fps" % (avg_fps),
            (10, image_np.shape[0]+30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('img', out_img)
        cv2.waitKey(1)
