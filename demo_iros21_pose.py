import os
import argparse
import shutil
import glob
import cv2
import numpy as np
import cupy as cp
import OpenGL.GL as gl
import pangolin

from multiprocessing import Process, Queue

import torch
from torch.utils.data import DataLoader

from ruamel.yaml import YAML

from dataloaders import KITTIRawLoader as KRL

from pose_ssstereo import Pose_SSStereo

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

version = 21084
vid_date = "2011_09_26"
# vid_num = '0009'
vid_num = '0093'
# vid_num = '0039'
half_precision = True

semantics = False
plot_map = True


class DynamicArray(object):
    def __init__(self, shape=3):
        if isinstance(shape, int):
            shape = (shape,)
        assert isinstance(shape, tuple)

        self.data = np.zeros((10000000, *shape))
        self.shape = shape
        self.ind = 0

        self.extend_length = 10000000

    def clear(self):
        self.ind = 0

    def append(self, x):
        self.extend([x])
    
    def extend(self, xs):
        if len(xs) == 0:
            return
        assert np.array(xs[0]).shape == self.shape

        if self.ind + len(xs) >= len(self.data):
            self.data.resize(
                (self.extend_length+len(self.data), *self.shape) , refcheck=False)

        if isinstance(xs, np.ndarray):
            self.data[self.ind:self.ind+len(xs)] = xs
            self.ind += len(xs)
        else:
            for i, x in enumerate(xs):
                self.data[self.ind+i] = x
            self.ind += len(xs)

    def array(self):
        return self.data[:self.ind]

    def __len__(self):
        return self.ind

    def __getitem__(self, i):
        assert i < self.ind
        return self.data[i]

    def __iter__(self):
        for x in self.data[:self.ind]:
            yield x


class pang(object):
    def __init__(self):

        self.q_points = Queue()
        self.q_colors = Queue()
        self.q_pose = Queue()
        self.q_image = Queue()
        self.q_graph = Queue()
        self.q_follow = Queue()
        self.q_mappoints = Queue()
        self.q_mapcolors = Queue()

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

    def put(self, points, colors, pose, graph, image,
            follow=None, mappoints=None, mapcolors=None):
        self.q_points.put(points)
        self.q_colors.put(colors)
        self.q_pose.put(pose)
        self.q_graph.put(graph)
        self.q_image.put(image)
        if follow is not None:
            self.q_follow.put(follow)
        if mappoints is not None:
            self.q_mappoints.put(mappoints)
        if mapcolors is not None:
            self.q_mapcolors.put(mapcolors)

    def plot(self):
        mappoints = DynamicArray(shape=(3,))
        mapcolors = DynamicArray(shape=(3,))
        follow = True

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

        pose = pangolin.OpenGlMatrix()
        dimg = pangolin.Display('image')
        dimg.SetBounds(0, 352 / 1 / 1080., 0.0, 1216 / 2 / 1920., 1920 / 1080.)
        dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)
        texture = pangolin.GlTexture(
            int(1216 / 2), int(352 / 1),
            gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

        lines = []
        image = None
        while not pangolin.ShouldQuit():
            if not self.q_pose.empty():
                pose.m = self.q_pose.get()
            if not self.q_follow.empty():
                follow = self.q_follow.get()
            if follow:
                scam.Follow(pose, True)

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            dcam.Activate(scam)

            if not self.q_graph.empty():
                graph = self.q_graph.get()
                lines.append(graph)

            if len(lines) > 0:
                gl.glLineWidth(1)
                gl.glColor3f(0.0, 1.0, 0.0)
                pangolin.DrawLines(np.array(lines), 3)

            gl.glPointSize(1)
            gl.glColor3f(1.0, 0.0, 0.0)
            gl.glBegin(gl.GL_POINTS)
            gl.glVertex3d(pose[0, 3], pose[1, 3], pose[2, 3])
            gl.glEnd()

            if not self.q_points.empty():
                while not self.q_points.empty():
                    points = self.q_points.get()
                    colors = self.q_colors.get()[:, ::-1]/255

            gl.glPointSize(2)
            gl.glColor3f(1.0, 0.0, 0.0)
            # access numpy array directly(without copying data),
            # array should be contiguous.
            pangolin.DrawPoints(points[::1], colors[::1])

            if not self.q_mappoints.empty():
                while not self.q_mappoints.empty():
                    mappoints_ = self.q_mappoints.get()
                    mapcolors_ = self.q_mapcolors.get()[:, ::-1]/255
                mappoints.extend(mappoints_)
                mapcolors.extend(mapcolors_)
            gl.glPointSize(2)
            gl.glColor3f(1.0, 0.0, 0.0)
            pangolin.DrawPoints(mappoints.array(), mapcolors.array())
            # print(mappoints.array().shape)

            # # Draw camera
            # gl.glLineWidth(1)
            # gl.glColor3f(0.0, 0.0, 1.0)
            # pangolin.DrawCameras(pose.m[np.newaxis, :, :], 1)

            # Draw image
            if not self.q_image.empty():
                image = self.q_image.get()
                image = cv2.resize(image[::-1, :, ], (608, 352))
                # cv2.imshow('image', image)
                # cv2.waitKey(1)
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

    if cfg['model']['pose']['backbone']['cfg_path'] is not None:
        backbone_cfg = YAML().load(
            open(cfg['model']['pose']['backbone']['cfg_path'], 'r'))
        cfg['model']['pose']['backbone'].update(backbone_cfg)
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
        './logs/stereo', cfg['model']['name'], version)
    cfg['stereo_ckpt'] = ckpt
    pose_ssstereo = Pose_SSStereo.load_from_checkpoint(cfg['stereo_ckpt'],
                                                       strict=False,
                                                       cfg=cfg,
                                                       load_as_module=True
                                                       ).cuda()

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

    vis_pang = pang()

    ################################################################
    # vis_pang2 = pang()
    pts_history = np.empty((0, 3))
    cnt_history = np.empty((0, 1))
    cls_history = np.empty((0, 3))
    history_max_size = 100000
    voxel_size = 1/4
    voxel_min = 3

    pts_good = np.empty((0, 3))
    cls_good = np.empty((0, 3))
    put_dense_map = False
    ################################################################

    pose_ssstereo.eval()
    for i, batch in enumerate(kitti_train):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        imgL, imgR = batch['imgL'].cuda(), batch['imgR'].cuda()
        imgLRaw = batch['imgLRaw']
        # cv2.imshow('imgLRaw', imgLRaw[0].permute(1, 2, 0).numpy().astype(np.uint8))
        # cv2.waitKey(1)
        imgLRaw = imgLRaw.cuda()

        tgt_imgs = [batch['contexts']['imgLPrev'].cuda()]
        src_img_raw = imgLRaw
        tgt_imgs_raw = [batch['contexts']['imgLPrevRaw'].cuda()]

        # cv2.imshow('img', imgLRaw[0].permute(1, 2, 0).data.cpu().numpy().astype(np.uint8))
        # cv2.waitKey(1)

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
            pose = torch.eye(4, device=imgL.device)
            calib = load_calibs(batch['calib'])
            cam = Camera(K=calib['K'].float()).to(imgL.device)
            Tinit = None

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=half_precision):
                disp, poses, depth = pose_ssstereo(
                    imgL, imgR,
                    calib,
                    tgt_imgs,
                    src_img_raw, tgt_imgs_raw,
                    Tinit=Tinit, return_depth=True)
        end.record()
        torch.cuda.synchronize()
        runtime = start.elapsed_time(end)
        print('Stereo-Pose runtime: {:.3f}'.format(runtime))

        # Get X3d of stereo camera points
        cam_points = cam.reconstruct(depth[0].unsqueeze(1), frame='w')
        cam_points_ = cam_points.permute(0, 2, 3, 1).reshape(-1, 3)
        mask_ = cam_points_[:, 1] > -4
        mask_ = torch.logical_and(mask_, boundary_mask(depth[0], 3).reshape(-1))
        cam_points_ = cam_points_[mask_]
        cam_colors_ = 1*(imgLRaw.permute(0, 2, 3, 1).reshape(-1, 3)[mask_])

        # Compute world coordinates of points
        Xc = cam_points_
        prev_position = Pose(pose).inverse().mat[0, :3, -1].data.cpu().numpy()
        with torch.cuda.amp.autocast(enabled=False):
            T_ = poses[0].inverse().mat
            if cfg['model']['pose']['init_next_by_prev']:
                Tinit = poses[0].mat  # T_
            pose = torch.mm(T_[0], pose)
            pose_inv = Pose(pose).inverse().mat[0]

            Xw = (torch.mm(pose_inv[:3, :3], Xc.permute(1, 0)) +
                  pose_inv[:3, -1].unsqueeze(-1)).permute(1, 0)
            # Xw = Xc
        Xw_np = Xw.data.cpu().numpy()
        current_position = pose_inv[:3, -1].data.cpu().numpy()

        Ic = cam_colors_
        Ic_np = Ic.data.cpu().numpy()

        pose_np = pose_inv.data.cpu().numpy()
        graph_np = [*current_position, *prev_position]
        image_np = (imgLRaw[0].permute(1, 2, 0).data.cpu().numpy()).astype(np.uint8)
        disp_np = (2*disp[0]).data.cpu().numpy().astype(np.uint8)
        # disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_JET)
        disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_PLASMA)
        out_img = np.concatenate((image_np, disp_np), 0)

        # cv2.imshow('img', image_np)
        # cv2.waitKey(1)

        ################################################################
        if plot_map:

            cv2.imshow('img', out_img)
            k = cv2.waitKey(1)
            follow = None
            if k == ord('p'):
                put_dense_map = True
            elif k == ord('n'):
                put_dense_map = False
            elif k == ord('f'):
                follow = True
            elif k == ord('s'):
                follow = False

            if put_dense_map:
                skips = 1
                Xw = np.round(voxel_size*Xw_np[::skips], 2)/voxel_size

                points = np.concatenate((pts_history, Xw), 0)
                colors = np.concatenate((cls_history, Ic_np[::skips]), 0)
                counts = np.concatenate((cnt_history-1, np.zeros((Xw.shape[0], 1))), 0)

                # points_unique, points_idx, points_count = np.unique(
                #     points, True, False, True, 0)
                # # points_unique, points_idx, points_count = cp.unique(
                # #     cp.asarray(points), True, False, True, 0)
                # # points_unique, points_idx, points_count = cp.asnumpy(points_unique), cp.asnumpy(points_idx), cp.asnumpy(points_count)
                
                points_unique_, points_idx_, points_count_ = torch.unique(
                    torch.tensor(points, device='cuda'), True, True, True, 0)
                colors_ = points_unique_.new_zeros(points_unique_.shape)
                colors_[points_idx_] = torch.tensor(colors, device='cuda')
                colors_ = colors_.data.cpu().numpy()
                counts_ = points_unique_.new_zeros((points_unique_.shape[0], 1))
                counts_[points_idx_] = torch.tensor(counts, device='cuda')
                counts_ = points_count_.unsqueeze(-1) + counts_
                counts_ = counts_.data.cpu().numpy()
                points_unique, points_idx, points_count = points_unique_.data.cpu().numpy(), points_idx_.data.cpu().numpy(), points_count_.data.cpu().numpy()

                # colors_ = colors[points_idx]
                # counts_ = points_count[:, np.newaxis] + counts[points_idx]
                points_good = points_unique[points_count >= voxel_min]
                colors_good = colors_[points_count >= voxel_min]  

                pts_history = points_unique[points_count < voxel_min]
                cls_history = colors_[points_count < voxel_min]
                cnt_history = counts_[points_count < voxel_min]

                if pts_history.shape[0] > history_max_size:
                    pts_history = pts_history[-history_max_size:]
                    cls_history = cls_history[-history_max_size:]
                    cnt_history = cnt_history[-history_max_size:]
                    
                vis_pang.put(
                    Xw_np, Ic_np, pose_np, graph_np, out_img, follow,
                    points_good, colors_good)
            else:
                vis_pang.put(
                    Xw_np, Ic_np, pose_np, graph_np, out_img, follow)

        ################################################################
        else:
            vis_pang.put(Xw_np, Ic_np, pose_np, graph_np, out_img)