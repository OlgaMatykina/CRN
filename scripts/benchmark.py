import torch
import time
from models.roburcdet import RobuRCDet

# === Фиктивные конфиги ===
backbone_img_conf = {
    'radar_view_transform': None,  # если нужно, подставь свой
    'img_channels': 3,
    'num_cams': 6,
    'img_size': [256, 704],
    'feature_dim': 64,
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

backbone_pts_conf = {
    'input_dim': 4,
    'voxel_size': [0.2, 0.2, 8],
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

fuser_conf = {
    'img_dims': 80,
    'pts_dims': 80,
    'embed_dims': 128,
    'num_layers': 6,
    'num_heads': 4,
    'bev_shape': (128, 128),
}

head_conf = {
    'bev_backbone_conf': dict(
        type='ResNet',
        in_channels=128,
        depth=18,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=[0, 1, 2],
        norm_eval=False,
        base_channels=128,
    ),
    'bev_neck_conf': dict(
        type='SECONDFPN',
        in_channels=[128, 128, 256, 512],
        upsample_strides=[1, 2, 4, 8],
        out_channels=[64, 64, 64, 64]
    ),
    'tasks': [
        dict(num_class=1, class_names=['car']),
        dict(num_class=2, class_names=['truck', 'construction_vehicle']),
        dict(num_class=2, class_names=['bus', 'trailer']),
        dict(num_class=1, class_names=['barrier']),
        dict(num_class=2, class_names=['motorcycle', 'bicycle']),
        dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
    ],
    'common_heads': dict(
        reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
    'bbox_coder': dict(
        type='CenterPointBBoxCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_num=500,
        score_threshold=0.01,
        out_size_factor=4,
        voxel_size=[0.2, 0.2, 8],
        pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
        code_size=9,
    ),
    'train_cfg': dict(
        point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
        grid_size=[512, 512, 1],
        voxel_size=[0.2, 0.2, 8],
        out_size_factor=4,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ),
    'test_cfg': dict(
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        score_threshold=0.01,
        out_size_factor=4,
        voxel_size=[0.2, 0.2, 8],
        nms_type='circle',
        pre_max_size=1000,
        post_max_size=200,
        nms_thr=0.2,
    ),
    'in_channels': 256,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
}

# === Фейковые данные ===
B = 1  # batch size
num_cams = 6
H, W = 256, 704
num_sweeps = 1

# Sweep images: (B, num_cams, 3, H, W)
sweep_imgs = torch.randn(B, num_sweeps, num_cams, 3, H, W).cuda()

# Sweep points: (B, N, 4) — например, N = 20000 точек
sweep_ptss = torch.randn(B, num_sweeps, num_cams, 2000, 5).cuda()

# Матрицы: (B, num_sweeps, num_cams, 4, 4)
mats_dict = {
    'sensor2ego_mats': torch.eye(4).repeat(B, num_sweeps, num_cams, 1, 1).cuda(),
    'intrin_mats': torch.eye(4).repeat(B, num_sweeps, num_cams, 1, 1).cuda(),
    'ida_mats': torch.eye(4).repeat(B, num_sweeps, num_cams, 1, 1).cuda(),
    'sensor2sensor_mats': torch.eye(4).repeat(B, num_sweeps, num_cams, 1, 1).cuda(),
    'bda_mat': torch.eye(4).repeat(B, 1, 1).cuda()
}

# === Модель ===
model = RobuRCDet(backbone_img_conf, backbone_pts_conf, fuser_conf, head_conf)
model.cuda()
model.eval()

# === Прогон ===
with torch.no_grad():
    for i in range(1001):
        outputs = model(sweep_imgs, mats_dict, sweep_ptss=sweep_ptss, is_train=False)
