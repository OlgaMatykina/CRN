import unittest
import pytest
import torch
import numpy as np

from datasets.nusc_sbd_dataset import NuscDatasetSparseBeatsDense, collate_fn
from layers.backbones.rvt_lss_fpn_sbd import RVTLSSFPNSBD
from layers.backbones.pts_backbone import PtsBackbone
from functools import partial

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]
H = 900
W = 1600
final_dim = (256, 704)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)

rda_aug_conf = {
    'N_sweeps': 6,
    'N_use': 5,
    'drop_ratio': 0.1,
}


class TestRVTLSSFPN(unittest.TestCase):
    def setUp(self) -> None:
        self.optimizer_config = dict(
            type='AdamW',
            lr=2e-4,
            weight_decay=1e-4)
        ################################################
        self.ida_aug_conf = {
            'resize_lim': (0.386, 0.55),
            'final_dim': (256, 704),
            'rot_lim': (0., 0.),
            'H': 900,
            'W': 1600,
            'rand_flip': True,
            'bot_pct_lim': (0.0, 0.0),
            # 'cams': [
            #     'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            #     'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            # ],
            # 'Ncams': 6,
            'cams': [
                'CAM_FRONT'
            ],
            'Ncams': 1,
        }
        self.bda_aug_conf = {
            'rot_ratio': 1.0,
            'rot_lim': (-22.5, 22.5),
            'scale_lim': (0.9, 1.1),
            'flip_dx_ratio': 0.5,
            'flip_dy_ratio': 0.5
        }
        ################################################
        self.backbone_img_conf = {
            'x_bound': [-51.2, 51.2, 0.8],
            'y_bound': [-51.2, 51.2, 0.8],
            'z_bound': [-5, 3, 8],
            'd_bound': [2.0, 58.0, 0.8],
            'final_dim': (256, 704),
            'downsample_factor': 16,
            'img_backbone_conf': dict(
                type='ResNet',
                depth=18,
                frozen_stages=0,
                out_indices=[0, 1, 2, 3],
                norm_eval=False,
                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
            ),
            'img_neck_conf': dict(
                type='SECONDFPN',
                in_channels=[64, 128, 256, 512],
                upsample_strides=[0.25, 0.5, 1, 2],
                out_channels=[64, 64, 64, 64],
            ),
            'depth_net_conf':
                dict(in_channels=256, mid_channels=256),
            'radar_view_transform': True,
            'camera_aware': False,
            'output_channels': 80,
        }
        ################################################
        self.backbone_pts_conf = {
            'pts_voxel_layer': dict(
                max_num_points=8,
                voxel_size=[8, 0.4, 2],
                point_cloud_range=[0, 2.0, 0, 704, 58.0, 2],
                max_voxels=(768, 1024)
            ),
            'pts_voxel_encoder': dict(
                type='PillarFeatureNet',
                in_channels=5,
                feat_channels=[32, 64],
                with_distance=False,
                with_cluster_center=False,
                with_voxel_center=True,
                voxel_size=[8, 0.4, 2],
                point_cloud_range=[0, 2.0, 0, 704, 58.0, 2],
                norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                legacy=True
            ),
            'pts_middle_encoder': dict(
                type='PointPillarsScatter',
                in_channels=64,
                output_shape=(140, 88)
            ),
            'pts_backbone': dict(
                type='SECOND',
                in_channels=64,
                out_channels=[64, 128, 256],
                layer_nums=[2, 3, 3],
                layer_strides=[1, 2, 2],
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                conv_cfg=dict(type='Conv2d', bias=True, padding_mode='reflect')
            ),
            'pts_neck': dict(
                type='SECONDFPN',
                in_channels=[64, 128, 256],
                out_channels=[64, 64, 64],
                upsample_strides=[0.5, 1, 2],
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                upsample_cfg=dict(type='deconv', bias=False),
                use_conv_for_no_stride=True
                ),
            'out_channels_pts': 80,
        }
        self.rvt_lss_fpn_sbd = RVTLSSFPNSBD(**self.backbone_img_conf).cuda()

    @pytest.mark.skipif(torch.cuda.is_available() is False, reason='No GPU available.')
    def test_forward(self):
        # sweep_imgs = torch.rand(2, 2, 6, 3, 64, 64).cuda()
        # sensor2ego_mats = torch.rand(2, 2, 6, 4, 4).cuda()
        # intrin_mats = torch.rand(2, 2, 6, 4, 4).cuda()
        # ida_mats = torch.rand(2, 2, 6, 4, 4).cuda()
        # sensor2sensor_mats = torch.rand(2, 2, 6, 4, 4).cuda()
        # bda_mat = torch.rand(2, 4, 4).cuda()

        np.random.seed(0)
        torch.random.manual_seed(0)
        nusc_dataset = NuscDatasetSparseBeatsDense(self.ida_aug_conf,
                                self.bda_aug_conf,
                                rda_aug_conf,
                                CLASSES,
                                '../HPR1/nuscenes',
                                '../HPR1/nuscenes_radar_5sweeps_infos_val.pkl',
                                is_train=True,
                                return_image=True,
                                return_depth=True,
                                sweep_idxes=[],
                                return_radar_pv=True)

        dataloader = torch.utils.data.DataLoader(
            nusc_dataset,
            batch_size=2,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_image=True,
                               is_return_depth=True,
                               is_return_radar_pv=True),
            sampler=None,
        )
        dataiter = iter(dataloader)
        batch = next(dataiter)
        
        (sweep_imgs, mats, _, gt_boxes_3d, gt_labels_3d, _, 
            img_batch,
            padding_radar_pts_batch,
            valid_radar_pts_cnt_batch,
            radar_batch,
            lidar_batch,
            lidar_mask_batch,
            seg_mask_roi_batch, depth_labels, pts_pv) = batch

        sweep_imgs = sweep_imgs.cuda()
        mats = {k: v.cuda() for k, v in mats.items()}
        img_batch = img_batch.cuda()
        padding_radar_pts_batch = padding_radar_pts_batch.cuda()
        valid_radar_pts_cnt_batch = valid_radar_pts_cnt_batch.cuda()
        radar_batch = radar_batch.cuda()
        lidar_batch = lidar_batch.cuda()
        lidar_mask_batch = lidar_mask_batch.cuda()
        seg_mask_roi_batch = seg_mask_roi_batch.cuda()
        # depth_labels = depth_labels.cuda()
        pts_pv = pts_pv.cuda()

        self.backbone_pts = PtsBackbone(**self.backbone_pts_conf).cuda()

        ptss_context, ptss_occupancy, _ = self.backbone_pts(pts_pv)

        # print(img_batch.shape)
        # print(radar_batch.shape)


        preds = self.rvt_lss_fpn_sbd.forward(sweep_imgs, mats, ptss_context, ptss_occupancy, \
                                             img_batch, radar_batch, padding_radar_pts_batch, valid_radar_pts_cnt_batch)

if __name__ == '__main__':
    unittest.main()