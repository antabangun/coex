import os
import argparse
import shutil
import glob
import cv2
import numpy as np
import OpenGL.GL as gl
import pangolin

from multiprocessing import Process, Queue

import torch
from torch.utils.data import DataLoader

from ruamel.yaml import YAML

from dataloaders import KITTIRawLoader as KRL

from stereo import Stereo

from geometry.pose import Pose
from geometry.camera import Camera

torch.backends.cudnn.benchmark = True

import pdb


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.set_grad_enabled(False)
configs = [
    'cfg_directweightedss_iros21.yaml',
    ]
config_num = 0

version = 21080  # CoEx
# version = 21010  # CoEx-48
# version = 1120  # CoEx-6
# version = 1122  # CoEx-3
# version = 2123  # CoEx-2
# version = 11020  # CoEx-One-48
# version = 21040  # CoEx-Full-48
# version = 11060  # CoEx-Full-6
# version = 11070  # CoEx-Full-3
# version = 2000  # PSMNet
# version = 1009  # PSMNet-6
# version = 11011  # PSMNet-One-192
# version = 11012  # PSMNet-One-6
# version = 1001  # PSMNet corr-192
# version = 1010  # PSMNet corr-6
# version = 1002  # PSMNet corr-one-192
# version = 2003  # PSMNet corr-one-6
# version = 1004  # PSMNet corr-one-2
vid_date = "2011_09_26"
vid_num = '0005'
# vid_num = '0009'
# vid_num = '0093'
# vid_num = '0039'
half_precision = False

vis_3d = False
semantics = False


class pang(object):
    def __init__(self):

        self.q_points = Queue()
        self.q_colors = Queue()
        self.q_pose = Queue()
        self.q_image = Queue()
        self.q_graph = Queue()

        self.plot_thread = Process(target=self.plot)
        self.plot_thread.start()

    def stop(self):
        self.plot_thread.join()
        qtype = type(Queue())
        for x in self.__dict__.values():
            if isinstance(x, qtype):
                while not x.empty():
                    _ = x.get()
        print('viewer stopped')

    def put(self, points, colors, image):
        self.q_points.put(points)
        self.q_colors.put(colors)
        self.q_image.put(image)

    def plot(self):
        points = np.random.rand(100*3).reshape(-1, 3)
        colors = np.random.rand(100*3).reshape(-1, 3)

        pangolin.CreateWindowAndBind('Main', 1920, 1080)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100000),
            pangolin.ModelViewLookAt(0, -5, -10, 0, 0, 0, 0, -1, 0))
        handler = pangolin.Handler3D(scam)

        # Create Interactive View in window
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -1920.0/1080.0)
        dcam.SetHandler(handler)

        dimg = pangolin.Display('image')
        dimg.SetBounds(0, 352 / 1 / 1080., 0.0, 1216 / 2 / 1920., 1920 / 1080.)
        dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)
        texture = pangolin.GlTexture(
            int(1216 / 2), int(352 / 1),
            gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

        image = None
        while not pangolin.ShouldQuit():

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            dcam.Activate(scam)

            if not self.q_points.empty():
                while not self.q_points.empty():
                    points = self.q_points.get()
                    colors = self.q_colors.get()[:, ::-1]/255

            gl.glPointSize(2)
            gl.glColor3f(1.0, 0.0, 0.0)
            # access numpy array directly(without copying data),
            # array should be contiguous.
            pangolin.DrawPoints(points[::1], colors[::1])

            # Draw image
            if not self.q_image.empty():
                image = self.q_image.get()
                image = cv2.resize(image[::-1, :, ], (608, 352))
            if image is not None:
                # image = 100*np.ones((352, 1216, 3), 'uint8')
                texture.Upload(image, gl.GL_BGR, gl.GL_UNSIGNED_BYTE)
                dimg.Activate()
                gl.glColor3f(1.0, 1.0, 1.0)
                texture.RenderToViewport()

            pangolin.FinishFrame()


def load_configs(path):
    cfg = YAML().load(open(path, 'r'))
    backbone_cfg = YAML().load(
        open(cfg['model']['stereo']['backbone']['cfg_path'], 'r'))
    cfg['model']['stereo']['backbone'].update(backbone_cfg)
    return cfg


def load_calibs(calib):
    for key, value in calib.items():
        if torch.is_tensor(value):
            calib[key] = value.cuda()
        if isinstance(value, dict):
            calib[key] = load_calibs(calib[key])
    return calib


def boundary_mask(disp, ksize=5, lim=1):
    disp_ = disp.permute(1, 2, 0).data.cpu().numpy()
    kernel = np.ones((ksize, ksize), np.uint8)
    laplacian = cv2.Laplacian(disp_, cv2.CV_32F)
    mask = cv2.dilate(255*((laplacian > lim).astype(np.uint8)), kernel, iterations=1)
    disp__ = disp_ - disp_.min()
    disp__ = disp__/disp__.max()*255
    # cv2.imshow('mask', mask); cv2.imshow('disp', disp__.astype(np.uint8)); cv2.waitKey(0)
    mask_t = torch.tensor(~mask, dtype=torch.bool, device=disp.device)
    return mask_t


if semantics:
    import torchvision

    def color_pred2(preds):
        preds = torch.argmax(preds, dim=1, keepdim=True)
        bgnd_mask = torch.logical_and(preds != 2, preds != 6)
        bgnd_mask = torch.logical_and(bgnd_mask, preds != 7)
        bgnd_mask = torch.logical_and(bgnd_mask, preds != 14)
        bgnd_mask = torch.logical_and(bgnd_mask, preds != 15)
        preds[bgnd_mask] = 0
        preds[preds == 2] = 1
        preds[preds == 6] = 2
        preds[preds == 7] = 3
        preds[preds == 14] = 4
        preds[preds == 15] = 5
        pred_color = (preds).repeat(1, 3, 1, 1)
        pred_color[:, :1][preds == 0], pred_color[:, 1:2][preds == 0], pred_color[:, 2:3][preds == 0] = 0, 0, 0
        pred_color[:, :1][preds == 1], pred_color[:, 1:2][preds == 1], pred_color[:, 2:3][preds == 1] = 0, 255, 255
        pred_color[:, :1][preds == 2], pred_color[:, 1:2][preds == 2], pred_color[:, 2:3][preds == 2] = 192, 192, 0
        pred_color[:, :1][preds == 3], pred_color[:, 1:2][preds == 3], pred_color[:, 2:3][preds == 3] = 128, 0, 128
        pred_color[:, :1][preds == 4], pred_color[:, 1:2][preds == 4], pred_color[:, 2:3][preds == 4] = 255, 255, 0
        pred_color[:, :1][preds == 5], pred_color[:, 1:2][preds == 5], pred_color[:, 2:3][preds == 5] = 0, 128, 128
        pred_color = pred_color.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        return pred_color

    ddr_model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained=True)

    ddr_model = ddr_model.cuda()
    ddr_model.eval()

if __name__ == '__main__':
    cfg = load_configs(
        './logs/stereo/IROS21/version_{}/project/configs/stereo/{}'.format(
            version, configs[config_num]))
    ###
    # fish_model_scale = cfg['model']['monodepth']['downsize']
    # fish_scale = fish_model_scale * cfg['training']['fish']['resize']

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

    if vis_3d:
        vis_pang = pang()
    else:
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

        if semantics:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            predLRaw = ddr_model(torch.flip(imgL, [1]))['out']

            predLRaw = color_pred2(predLRaw)
            imgLRaw = 0.5 * imgLRaw + 0.5 * torch.tensor(
                predLRaw, device=imgLRaw.device).permute(2, 0, 1).unsqueeze(0)
            
            end.record()
            torch.cuda.synchronize()
            runtime = start.elapsed_time(end)
            print('Semantic runtime: {:.3f}'.format(runtime))

        if i == 0:
            # Initialize calibrations and poses of cameras
            calib = load_calibs(batch['calib'])
            cam = Camera(K=calib['K'].float()).to(imgL.device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=half_precision):
                disp = pose_ssstereo(imgL, imgR, False)
                if vis_3d:
                    depth = [(calib['K'][:, 0:1, 0:1] *
                              calib['baseline'].reshape(-1, 1, 1) /
                              disp)]
        end.record()
        torch.cuda.synchronize()
        runtime = start.elapsed_time(end)
        # print('Stereo runtime: {:.3f}'.format(runtime))

        if vis_3d:
            # Get X3d of stereo camera points
            cam_points = cam.reconstruct(depth[0].unsqueeze(1), frame='w')
            cam_points_ = cam_points.permute(0, 2, 3, 1).reshape(-1, 3)
            mask_ = cam_points_[:, 1] > -4
            mask_ = torch.logical_and(mask_, boundary_mask(depth[0], 3).reshape(-1))
            cam_points_ = cam_points_[mask_]
            cam_colors_ = 1*(imgLRaw.permute(0, 2, 3, 1).reshape(-1, 3)[mask_])

            # Compute world coordinates of points
            Xc = cam_points_
            Xw_np = Xc.data.cpu().numpy()

            Ic = cam_colors_
            Ic_np = Ic.data.cpu().numpy()
            image_np = (imgLRaw[0].permute(1, 2, 0).data.cpu().numpy()).astype(np.uint8)
            disp_np = (2*disp[0]).data.cpu().numpy().astype(np.uint8)
            # disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_JET)
            disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_PLASMA)
            
            out_img = np.concatenate((image_np, disp_np), 0)

            # cv2.imshow('img', image_np)
            # cv2.waitKey(1)

            vis_pang.put(Xw_np, Ic_np, out_img)
            # end.record()
            # torch.cuda.synchronize()
            # runtime = start.elapsed_time(end)
            # print('runtime: ', runtime)

        else:
            fps = 1000/runtime
            fps_list = np.append(fps_list, fps)
            if len(fps_list) > 5:
                fps_list = fps_list[-5:]
            avg_fps = np.mean(fps_list)
            print('Stereo runtime: {:.3f}'.format(1000/avg_fps))

            disp_np = (2*disp[0]).data.cpu().numpy().astype(np.uint8)
            # disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_JET)
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
