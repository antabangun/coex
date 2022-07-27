import os
import shutil
import glob
import numpy as np
import cv2
import skimage
import skimage.io

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TestTubeLogger

from ruamel.yaml import YAML

# from models import *
from utils.load import load_class
from dataloaders import listflowfile as lt
from dataloaders import SceneFlowLoader as SFL
from dataloaders import KITTIloader2012 as ls12
from dataloaders import KITTIloader2015 as ls15
from dataloaders import KITTILoader as KL
from dataloaders import KITTI_submission_loader as KSL

import pdb


configs = [
    'cfg_coex.yaml',
    'cfg_psm.yaml'
    ]
config_num = 0


class Stereo(LightningModule):

    def __init__(self, cfg, dataname=None):
        super().__init__()
        self.cfg = cfg
        self.dataname = dataname

        self.stereo = load_class(
            cfg['model']['stereo']['name'],
            ['models.stereo'])(cfg['model']['stereo'])

    def forward(self, imgL, imgR=None, training=False):
        if training:
            disp_pred = self.stereo(imgL, imgR=imgR, training=training)
        else:
            h, w = imgL.shape[-2:]
            h_pad = 32-h % 32
            w_pad = 32-w % 32

            imgL = F.pad(imgL, (0, w_pad, 0, h_pad))
            if imgR is not None:
                imgR = F.pad(imgR, (0, w_pad, 0, h_pad))

            disp_pred = self.stereo(imgL, imR=imgR, training=training)[0][:, :h, :w]
            
        return disp_pred

    def training_step(self, batch, batch_idx):
        imgL, imgR, disp_true = batch['imgL'], batch['imgR'], batch['disp_true']
        x1, y1 = batch['x1'], batch['y1']
        disp_pred = self.stereo(imgL, imgR, u0=x1, v0=y1, training=True)

        losses = []
        train_weights = self.cfg['training']['training_scales_weighting']
        for disp_pred_ in disp_pred[:-1]:

            disp_pred_ = F.interpolate(
                disp_pred_.unsqueeze(1),
                size=(disp_true.shape[1], disp_true.shape[2]),
                mode='bilinear').squeeze(1)
            # ---------
            if self.dataname == 'sceneflow':
                mask = torch.logical_and(
                    disp_true <= self.cfg['model']['stereo']['max_disparity'],
                    disp_true > 0)
            else:
                mask = disp_true > 0.001
            mask.detach_()
            # ----
            losses.append(F.smooth_l1_loss(
                disp_pred_[mask], disp_true[mask], reduction='mean'))
        loss = sum([losses[i] * train_weights[i] for i in range(len(disp_pred[:-1]))]) /\
            sum([1 * train_weights[i] for i in range(len(disp_pred[:-1]))])

        self.log('{}_train_loss'.format(self.dataname), loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgL, imgR, disp_true = batch['imgL'], batch['imgR'], batch['disp_true']

        h, w = imgL.shape[-2:]
        h_pad = (32-(h % 32)) % 32
        w_pad = (32-(w % 32)) % 32

        imgL = F.pad(imgL, (0, w_pad, 0, h_pad))
        imgR = F.pad(imgR, (0, w_pad, 0, h_pad))

        disp_pred = self.stereo(imgL, imgR)[0][:, :h, :w]

        # ---------
        mask = torch.logical_and(disp_true <= self.cfg['model']['stereo']['max_disparity'],
                                 disp_true > 0)
        mask.detach_()
        # ----
        if mask.sum() == 0:
            epe = 0
        else:
            epe = torch.mean(torch.abs(disp_pred[mask]-disp_true[mask])).detach()
            if torch.isnan(epe).sum() > 0:
                pdb.set_trace()

        # ---------
        mask = disp_true > 0.001
        mask.detach_()
        # ----
        if mask.sum() == 0:
            error_3px, error_2px, error_1px = 0, 0, 0
        else:
            delta = torch.abs(disp_pred[mask] - disp_true[mask])
            error_mat_3px = (torch.logical_and((delta >= 3.0), (delta >= 0.05 * (disp_true[mask]))) == True)
            error_3px = torch.sum(error_mat_3px).item() / torch.numel(disp_pred[mask]) * 100
            error_mat_2px = (torch.logical_and((delta >= 2.0), (delta >= 0.05 * (disp_true[mask]))) == True)
            error_2px = torch.sum(error_mat_2px).item() / torch.numel(disp_pred[mask]) * 100
            error_mat_1px = (torch.logical_and((delta >= 1.0), (delta >= 0.05 * (disp_true[mask]))) == True)
            error_1px = torch.sum(error_mat_1px).item() / torch.numel(disp_pred[mask]) * 100

        self.log_dict(
            {
                '{}_val_epe'.format(self.dataname): epe,
                '{}_val_3pxError'.format(self.dataname): error_3px,
                '{}_val_2pxError'.format(self.dataname): error_2px,
                '{}_val_1pxError'.format(self.dataname): error_1px
            }
        )

        disp_img = disp_pred[0].data.cpu().numpy().astype(np.uint8)
        disp_img = cv2.applyColorMap(disp_img*2, cv2.COLORMAP_JET)
        self.save_disp_imgs(disp_img, '{:05d}'.format(batch_idx))
        return epe

    def test_step(self, batch, batch_idx):
        if self.dataname == 'sceneflow':
            imgL, imgR, disp_true = batch['imgL'], batch['imgR'], batch['disp_true']
            dataname, filename = None, '{:05d}'.format(batch_idx)
        else:
            imgL, imgR = batch['imgL'], batch['imgR']
            dataname, filename = batch['dataname'][0], batch['filename'][0]

        h, w = imgL.shape[-2:]
        h_pad = (32-(h % 32)) % 32
        w_pad = (32-(w % 32)) % 32

        imgL = F.pad(imgL, (0, w_pad, 0, h_pad))
        imgR = F.pad(imgR, (0, w_pad, 0, h_pad))

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        disp_pred = self.stereo(imgL, imgR)[0][:, :h, :w]
        end.record()
        torch.cuda.synchronize()
        runtime = start.elapsed_time(end)
        fps = 1000/runtime
        # print('total runtime : ', runtime)

        if self.dataname == 'sceneflow':
            if self.cfg['testing']['save_disp_imgs']:
                disp_img = disp_pred[0].data.cpu().numpy().astype(np.uint8)
                disp_img = cv2.applyColorMap(disp_img*1, cv2.COLORMAP_JET)
                self.save_disp_imgs(disp_img, filename, dataname)
                disp_img = disp_true[0].data.cpu().numpy().astype(np.uint8)
                disp_img = cv2.applyColorMap(disp_img*1, cv2.COLORMAP_JET)
                self.save_disp_imgs(disp_img, filename+"_gt", dataname)

            if self.cfg['testing']['compute_metrics']:
                # ---------
                mask = torch.logical_and(disp_true <= self.cfg['model']['stereo']['max_disparity'],
                                         disp_true > 0)
                mask.detach_()
                # ----
                if mask.sum() == 0:
                    epe = 0
                else:
                    epe = torch.mean(torch.abs(disp_pred[mask]-disp_true[mask])).detach()
                    if torch.isnan(epe).sum() > 0:
                        pdb.set_trace()
                # ---------
                mask = disp_true > 0.001
                mask.detach_()
                # ----
                if mask.sum() == 0:
                    error_3px, error_2px, error_1px = 0, 0, 0
                else:
                    delta = torch.abs(disp_pred[mask] - disp_true[mask])
                    error_mat_3px = delta >= 3.0
                    error_3px = torch.sum(error_mat_3px).item() / torch.numel(disp_pred[mask]) * 100
                    error_mat_2px = delta >= 2.0
                    error_2px = torch.sum(error_mat_2px).item() / torch.numel(disp_pred[mask]) * 100
                    error_mat_1px = delta >= 1.0
                    error_1px = torch.sum(error_mat_1px).item() / torch.numel(disp_pred[mask]) * 100

                self.log_dict({'{}_test_epe'.format(self.dataname): epe,
                               '{}_test_3pxError'.format(self.dataname): error_3px,
                               '{}_test_2pxError'.format(self.dataname): error_2px,
                               '{}_test_1pxError'.format(self.dataname): error_1px,
                               '{}_test_fps'.format(self.dataname): fps})
                return {'{}_epe'.format(self.dataname): epe,
                        '{}_3pxError'.format(self.dataname): error_3px,
                        '{}_2pxError'.format(self.dataname): error_2px,
                        '{}_1pxError'.format(self.dataname): error_1px,
                        '{}_fps'.format(self.dataname): fps}
            else:
                self.log_dict({'{}_test_fps'.format(self.dataname): fps})
                return {'{}_fps'.format(self.dataname): fps}

        else:
            if self.cfg['testing']['save_disp_imgs']:
                disp_img = (256*disp_pred[0].data.cpu().numpy()).astype('uint16')
                self.save_disp_imgs(disp_img, filename, dataname, True)

            self.log_dict({'{}_test_fps'.format(self.dataname): fps})

            return {'{}_fps'.format(self.dataname): fps}

    def configure_optimizers(self):
        dataname = 'kitti' if 'kitti' in self.dataname else 'sceneflow'
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.cfg['training']['lr'], betas=(0.9, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.cfg['training']['{}_milestones'.format(dataname)],
            gamma=self.cfg['training']['{}_gamma'.format(dataname)])
        return [optimizer], [lr_scheduler]

    def save_disp_imgs(self, disp_img, filename,
                       dataname=None, use_skimage=False):
        # Save imgs
        if dataname is None:
            dataname = self.dataname
        savedir = '{}/{}/version_{}/media/{}'.format(
            self.logger.save_dir, self.logger.name,
            self.logger.version, dataname)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        if use_skimage:
            skimage.io.imsave('{}/{}/version_{}/media/{}/{}.png'.format(
                self.logger.save_dir, self.logger.name,
                self.logger.version, dataname, filename), disp_img)
        else:
            cv2.imwrite('{}/{}/version_{}/media/{}/{}.png'.format(
                self.logger.save_dir, self.logger.name,
                self.logger.version, dataname, filename), disp_img)


def load_configs(path):
    cfg = YAML().load(open(path, 'r'))
    backbone_cfg = YAML().load(
        open(cfg['model']['stereo']['backbone']['cfg_path'], 'r'))
    cfg['model']['stereo']['backbone'].update(backbone_cfg)
    return cfg


def copy_dir(save_dir, name, save_version):
    savedir = '{}/{}/version_{}/project/'.format(save_dir, name, save_version)
    datadirs = ['configs', 'dataloaders', 'models', 'utils']
    if os.path.exists(savedir):
        shutil.rmtree(savedir)
    os.makedirs(savedir)
    for file in glob.glob('./*.py'):
        shutil.copyfile('./{}'.format(file),
                        '{}{}'.format(savedir, file))
    for datadir in datadirs:
        shutil.copytree('./{}'.format(datadir),
                        '{}{}'.format(savedir, datadir))
    return save_version


if __name__ == '__main__':
    pl.seed_everything(42)
    cfg = load_configs('./configs/stereo/{}'.format(configs[config_num]))
    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg['device']
    logging_pth = cfg['training']['paths']['logging']
    ###
    th, tw = cfg['training']['th'], cfg['training']['tw']
    ''' SceneFlow Training Part '''
    if cfg['training']['train_on']['sceneflow']:
        sceneflowpath = cfg['training']['paths']['sceneflow']
        all_left_img, all_right_img, all_left_disp, all_focal, test_left_img, test_right_img, test_left_disp, test_focal = lt.dataloader(sceneflowpath)
        sceneflow_train = SFL.ImageLoader(
            all_left_img, all_right_img, all_focal, all_left_disp,
            True, th=th, tw=tw)
        sceneflow_train = DataLoader(
            sceneflow_train, batch_size=cfg['training']['batch_size'],
            num_workers=16, shuffle=True, drop_last=False)
        sceneflow_test = SFL.ImageLoader(
            test_left_img, test_right_img, test_focal, test_left_disp,
            False)
        sceneflow_test = DataLoader(
            sceneflow_test, batch_size=1, num_workers=16,
            shuffle=False, drop_last=False)

        # Model
        log_name = 'sceneflow'
        if cfg['training']['load_version'] is not None:
            load_version = cfg['training']['load_version']
            ckpt = '{}/{}/version_{}/checkpoints/sceneflow-epoch={}.ckpt'.format(
                logging_pth, cfg['model']['name'], load_version,
                cfg['training']['sceneflow_max_epochs']-1)
            stereo = Stereo.load_from_checkpoint(ckpt, cfg=cfg, dataname=log_name)
            
        else:
            stereo = Stereo(cfg, 'sceneflow')

        version = copy_dir(
            logging_pth, cfg['model']['name'], cfg['training']['save_version'])
        # resume_from_checkpoint = None

        logger = TestTubeLogger(
            logging_pth,
            cfg['model']['name'],
            version=version)
        gpu_stats = pl.callbacks.GPUStatsMonitor()
        lr_monitor = pl.callbacks.LearningRateMonitor()
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename=log_name+'-{epoch}',
            save_last=True,
            save_top_k=-1,
            monitor=log_name+'_train_loss_epoch')
        trainer = pl.Trainer(
            accelerator='dp',
            logger=logger,
            callbacks=[gpu_stats, lr_monitor, checkpoint_callback],
            precision=cfg['precision'],
            gpus=cfg['device'],
            max_epochs=cfg['training']['sceneflow_max_epochs'],
            # resume_from_checkpoint=resume_from_checkpoint,
            benchmark=True,
            accumulate_grad_batches=1,
            gradient_clip_val=0.1,
            stochastic_weight_avg=True,
            # track_grad_norm=2,
            weights_summary='full',
            )

        trainer.fit(stereo, sceneflow_train, )
        trainer.test(stereo, sceneflow_test)

    ''' KITTI Training Part '''
    all_left_img, all_right_img, all_left_disp, all_calib, val_left_img, val_right_img, val_left_disp, val_calib = [], [], [], [], [], [], [], []
    if cfg['training']['train_on']['kitti12']:
        all_left_img12, all_right_img12, all_left_disp12, all_calib12, val_left_img12, val_right_img12, val_left_disp12, val_calib12 = ls12.dataloader(
            cfg['training']['paths']['kitti12'], True)
        all_left_img += all_left_img12
        all_right_img += all_right_img12
        all_left_disp += all_left_disp12
        all_calib += all_calib12
        val_left_img += val_left_img12
        val_right_img += val_right_img12
        val_left_disp += val_left_disp12
        val_calib += val_calib12
        log_name = 'kitti12'

    if cfg['training']['train_on']['kitti15']:
        all_left_img15, all_right_img15, all_left_disp15, all_calib15, val_left_img15, val_right_img15, val_left_disp15, val_calib15 = ls15.dataloader(
            cfg['training']['paths']['kitti15'], True)
        all_left_img += all_left_img15
        all_right_img += all_right_img15
        all_left_disp += all_left_disp15
        all_calib += all_calib15
        val_left_img += val_left_img15
        val_right_img += val_right_img15
        val_left_disp += val_left_disp15
        val_calib += val_calib15
        log_name = 'kitti15'

    if cfg['training']['train_on']['kitti12'] and cfg['training']['train_on']['kitti15']:
        log_name = 'kitti'

    if cfg['training']['train_on']['kitti12'] or cfg['training']['train_on']['kitti15']:
        kitti_train = KL.ImageLoader(
            all_left_img, all_right_img, all_left_disp, all_calib, th, tw,
            training=True)
        kitti_train = DataLoader(
            kitti_train, batch_size=cfg['training']['batch_size'],
            num_workers=16, shuffle=True, drop_last=False)
        kitti_val = KL.ImageLoader(
            val_left_img, val_right_img, val_left_disp, val_calib,
            training=False)
        kitti_val = DataLoader(
            kitti_val, batch_size=1,
            num_workers=16, shuffle=False, drop_last=False)

        # Model
        if cfg['training']['load_version'] is not None:
            load_version = cfg['training']['load_version']
        else:
            assert(cfg['training']['train_on']['sceneflow'])
            load_version = cfg['training']['save_version']

        ckpt = '{}/{}/version_{}/checkpoints/sceneflow-epoch={}.ckpt'.format(
            logging_pth, cfg['model']['name'], load_version,
            cfg['training']['sceneflow_max_epochs']-1)
        # ckpt = '{}/{}/version_{}/checkpoints/last.ckpt'.format(
        #     logging_pth, cfg['model']['name'], load_version)
        stereo = Stereo.load_from_checkpoint(ckpt, cfg=cfg, dataname=log_name)

        logger = TestTubeLogger(
            logging_pth,
            cfg['model']['name'],
            version=cfg['training']['save_version'])
        gpu_stats = pl.callbacks.GPUStatsMonitor()
        lr_monitor = pl.callbacks.LearningRateMonitor()
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename=log_name+'-{epoch}',
            save_last=True,
            save_top_k=2,
            monitor=log_name+'_val_3pxError')
        trainer = pl.Trainer(
            accelerator='dp',
            logger=logger,
            callbacks=[gpu_stats, lr_monitor, checkpoint_callback],
            precision=cfg['precision'],
            gpus=cfg['device'],
            max_epochs=cfg['training']['kitti_max_epochs'],
            # resume_from_checkpoint=resume_from_checkpoint,
            benchmark=True,
            accumulate_grad_batches=1,
            gradient_clip_val=0.1,
            stochastic_weight_avg=True,
            # track_grad_norm=2,
            weights_summary='full',
            )

        trainer.fit(stereo, kitti_train, kitti_val)

    if cfg['training']['train_on']['kitti12']:
        datapath = cfg['training']['paths']['kitti12'].replace('training', 'testing')
        test_left_img, test_right_img, test_calib = KSL.listfiles(datapath, 'kitti12')
        kitti_test = KSL.ImageLoader(test_left_img, test_right_img, test_calib)
        kitti_test = DataLoader(
            kitti_test, batch_size=1,
            num_workers=16, shuffle=False, drop_last=False)
        trainer.test(stereo, kitti_test)

    if cfg['training']['train_on']['kitti15']:
        datapath = cfg['training']['paths']['kitti15'].replace('training','testing')
        test_left_img, test_right_img, test_calib = KSL.listfiles(datapath, 'kitti15')
        kitti_test = KSL.ImageLoader(test_left_img, test_right_img, test_calib)
        kitti_test = DataLoader(
            kitti_test, batch_size=1,
            num_workers=16, shuffle=False, drop_last=False)
        trainer.test(stereo, kitti_test)



class StereoTRT(Stereo):

    def forward(self, imgL):

        cost, spx_pred = self.stereo(imgL)
            
        return cost, spx_pred