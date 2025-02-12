import os
import cv2
import scipy

import torch
import mmcv
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from nuscenes.utils.data_classes import Box, RadarPointCloud, LidarPointCloud
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from datasets.nusc_det_dataset import NuscDatasetRadarDet, collate_fn

from torchvision.transforms import InterpolationMode

from utils.utils import (
    map_pointcloud_to_image,
    get_depth_map,
    get_radar_map,
    canvas_filter,
)

class conf:
    input_h, input_w = 900, 1600
    max_depth = 80
    min_depth = 0


rng = np.random.default_rng()

__all__ = ['NuscDatasetSparseBeatsDense']

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, resize, resize_dims, crop, flip, rotate):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    ida_rot *= resize
    ida_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    ida_rot = A.matmul(ida_rot)
    ida_tran = A.matmul(ida_tran) + b
    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    return img, ida_mat


def bev_det_transform(gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy):
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                            [0, 0, 1]])
    scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                              [0, 0, scale_ratio]])
    flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rot_mat = flip_mat @ (scale_mat @ rot_mat)
    if gt_boxes.shape[0] > 0:
        gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
        gt_boxes[:, 3:6] *= scale_ratio
        gt_boxes[:, 6] += rotate_angle
        if flip_dx:
            gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
        if flip_dy:
            gt_boxes[:, 6] = -gt_boxes[:, 6]
        gt_boxes[:, 7:] = (
            rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
    return gt_boxes, rot_mat

class NuscDatasetSparseBeatsDense(NuscDatasetRadarDet):
    def __init__(self,
                 ida_aug_conf,
                 bda_aug_conf,
                 rda_aug_conf,
                 classes,
                 data_root,
                 info_paths,
                 is_train,
                 load_interval=1,
                 num_sweeps=1,
                 img_conf=dict(img_mean=[123.675, 116.28, 103.53],
                               img_std=[58.395, 57.12, 57.375],
                               to_rgb=True),
                 img_backbone_conf=dict(
                     x_bound=[-51.2, 51.2, 0.8],
                     y_bound=[-51.2, 51.2, 0.8],
                     z_bound=[-5, 3, 8],
                     d_bound=[2.0, 58.0, 0.5]
                 ),
                 drop_aug_conf=None,
                 return_image=True,
                 return_depth=False,
                 return_radar_pv=False,
                 depth_path='depth_gt',
                 radar_pv_path='radar_pv_filter',
                 semantic_path = 'seg_mask',
                 remove_z_axis=False,
                 use_cbgs=False,
                 gt_for_radar_only=False,
                 sweep_idxes=list(),
                 key_idxes=list()):
        super().__init__(ida_aug_conf,
                 bda_aug_conf,
                 rda_aug_conf,
                 classes,
                 data_root,
                 info_paths,
                 is_train,
                 load_interval,
                 num_sweeps,
                 img_conf,
                 img_backbone_conf,
                 drop_aug_conf,
                 return_image,
                 return_depth,
                 return_radar_pv,
                 depth_path,
                 radar_pv_path,
                 remove_z_axis,
                 use_cbgs,
                 gt_for_radar_only,
                 sweep_idxes,
                 key_idxes)
        
        self.semantic_path = semantic_path
        self.radar_load_dim = 18 # self.radar_data_conf["radar_load_dim"]
        self.radar_use_dims = [0, 1, 2, 5, 6, 7, 8, 9, 12, 13, 16, 17] # [x y z] dyn_prop id [rcs vx vy vx_comp vy_comp] is_quality_valid ambig_state [x_rms y_rms] invalid_state pdh0 [vx_rms vy_rms] + [timestamp_diff]
        
        self.semantic_mask_used_mask = [0, 1, 4, 12, 20, 32, 80, 83, 93, 127, 102, 116] 
        # {"wall": 0, "building": 1, "sky": 2, "floor": 3, "tree": 4, "ceiling": 5, "road": 6, "bed ": 7, "windowpane": 8, "grass": 9, "cabinet": 10, "sidewalk": 11, "person": 12, "earth": 13, "door": 14, "table": 15,
        # "mountain": 16, "plant": 17, "curtain": 18, "chair": 19, "car": 20, "water": 21, "painting": 22, "sofa": 23, "shelf": 24, "house": 25, "sea": 26, "mirror": 27, "rug": 28, "field": 29, "armchair": 30, "seat": 31, 
        # "fence": 32, "desk": 33, "rock": 34, "wardrobe": 35, "lamp": 36, "bathtub": 37, "railing": 38, "cushion": 39, "base": 40, "box": 41, "column": 42, "signboard": 43, "chest of drawers": 44, "counter": 45, "sand": 46,
        # "sink": 47, "skyscraper": 48, "fireplace": 49, "refrigerator": 50, "grandstand": 51, "path": 52, "stairs": 53, "runway": 54, "case": 55, "pool table": 56, "pillow": 57, "screen door": 58, "stairway": 59, "river": 60,
        # "bridge": 61, "bookcase": 62, "blind": 63, "coffee table": 64, "toilet": 65, "flower": 66, "book": 67, "hill": 68, "bench": 69, "countertop": 70, "stove": 71, "palm": 72, "kitchen island": 73, "computer": 74, "swivel chair": 75,
        # "boat": 76, "bar": 77, "arcade machine": 78, "hovel": 79, "bus": 80, "towel": 81, "light": 82, "truck": 83, "tower": 84, "chandelier": 85, "awning": 86, "streetlight": 87, "booth": 88, "television receiver": 89, "airplane": 90, 
        # "dirt track": 91, "apparel": 92, "pole": 93, "land": 94, "bannister": 95, "escalator": 96, "ottoman": 97, "bottle": 98, "buffet": 99, "poster": 100, "stage": 101, "van": 102, "ship": 103, "fountain": 104, "conveyer belt": 105, 
        # "canopy": 106, "washer": 107, "plaything": 108, "swimming pool": 109, "stool": 110, "barrel": 111, "basket": 112, "waterfall": 113, "tent": 114, "bag": 115, "minibike": 116, "cradle": 117, "oven": 118, "ball": 119, "food": 120,
        # "step": 121, "tank": 122, "trade name": 123, "microwave": 124, "pot": 125, "animal": 126, "bicycle": 127, "lake": 128, "dishwasher": 129, "screen": 130, "blanket": 131, "sculpture": 132, "hood": 133, "sconce": 134, "vase": 135,
        # "traffic light": 136, "tray": 137, "ashcan": 138, "fan": 139, "pier": 140, "crt screen": 141, "plate": 142, "monitor": 143, "bulletin board": 144, "shower": 145, "radiator": 146, "glass": 147, "clock": 148, "flag": 149}
        
        self.RADAR_PTS_NUM = 200
        
        # Todo support multi-view Depth Completion
        # Now we follow the previous research, only use the front Camera and Radar
        self.radar_use_type = 'RADAR_FRONT'
        self.camera_use_type = 'CAM_FRONT'
        self.lidar_use_type = 'LIDAR_TOP'

    def get_params(self, data):
        params = dict()
        if 'calibrated_sensor' in data.keys():
            params['sensor2ego'] = data['calibrated_sensor']
        else:
            params['sensor2ego'] = dict()
            params['sensor2ego']['translation'] = data['sensor2ego_translation']
            params['sensor2ego']['rotation'] = data['sensor2ego_rotation']
        
        if 'ego_pose' in data.keys():
            params['ego2global'] = data['ego_pose']
        else:
            params['ego2global'] = dict()
            params['ego2global']['translation'] = data['ego2global_translation']
            params['ego2global']['rotation'] = data['ego2global_rotation']
        
        return params
         
    def __getitem__(self, idx):
        if self.use_cbgs:
            idx = self.sample_indices[idx]
        cam_infos = list()
        pts_infos = list()
        cams = self.choose_cams()
        for key_idx in self.key_idxes:
            cur_idx = key_idx + idx
            while self.infos[cur_idx]['scene_token'] != self.infos[idx]['scene_token']:
                cur_idx += 1
            info = self.infos[cur_idx]
            cam_infos.append(info['cam_infos'])
            pts_infos.append([info['lidar_infos']] + info['lidar_sweeps'])
            for sweep_idx in self.sweeps_idx:
                if len(info['cam_sweeps']) == 0:
                    cam_infos.append(info['cam_infos'])
                else:
                    for i in range(min(len(info['cam_sweeps']) - 1, sweep_idx), -1, -1):
                        if sum([cam in info['cam_sweeps'][i] for cam in cams]) == len(cams):
                            cam_infos.append(info['cam_sweeps'][i])
                            break

        if self.return_image or self.return_depth or self.return_radar_pv:
            image_data_list = self.get_image(cam_infos, cams)
            # print(image_data_list)
            (
                sweep_imgs,
                sweep_sensor2ego_mats,
                sweep_intrins,
                sweep_ida_mats,
                sweep_sensor2sensor_mats,
                sweep_timestamps,
            ) = image_data_list[:6]
        else:
            (
                sweep_imgs,
                sweep_intrins,
                sweep_ida_mats,
                sweep_sensor2sensor_mats,
                sweep_timestamps,
            ) = None, None, None, None, None
            sweep_sensor2ego_mats = self.get_image_sensor2ego_mats(cam_infos, cams)

        img_metas = self.get_image_meta(cam_infos, cams)
        img_metas['token'] = self.infos[idx]['sample_token']
        gt_boxes_3d, gt_labels_3d, gt_corners = self.get_gt(self.infos[idx], cams, return_corners=False)

        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        gt_boxes_3d, bda_rot = bev_det_transform(gt_boxes_3d, rotate_bda, scale_bda, flip_dx, flip_dy)

        bda_mat = torch.zeros(4, 4, dtype=torch.float32)
        bda_mat[:3, :3] = bda_rot
        bda_mat[3, 3] = 1

        data = self.infos[idx]

        # get cameras images only for front 
        camera_infos = data['cam_infos'][self.camera_use_type]
        camera_params = self.get_params(camera_infos)
        camera_filename = camera_infos['filename'].split('samples/')[-1]
        img = cv2.imread(os.path.join(self.data_root, 'samples', camera_filename))

        # get radars only for front
        radar_infos = data['radar_infos'][self.radar_use_type][0]
        radar_params = self.get_params(radar_infos)
        path = radar_infos['data_path'].split('samples/')[-1]
        radar_obj = RadarPointCloud.from_file(os.path.join(self.data_root, 'samples', path))
        radar_all = radar_obj.points.transpose(1,0)[:, self.radar_use_dims]
        radar = np.concatenate((radar_all[:, :3], np.ones([radar_all.shape[0], 1])), axis=1)
        
        # get lidar top
        lidar_infos = data['lidar_infos'][self.lidar_use_type]
        lidar_params = self.get_params(lidar_infos)
        path = lidar_infos['filename'].split('samples/')[-1]
        lidar_obj = LidarPointCloud.from_file(os.path.join(self.data_root, 'samples', path))
        lidar = lidar_obj.points.transpose(1,0)[:, :3]
        lidar = np.concatenate((lidar, np.ones([lidar.shape[0], 1])), axis=1)
        
        # get semantic mask of images
        name = camera_filename.split('/')[-1].replace('.jpg', '.png')
        seg_mask_path = os.path.join(self.data_root, self.semantic_path, name)
        seg_mask = cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE)
        seg_mask_roi = list()
        for i in self.semantic_mask_used_mask:
            seg_mask_roi.append(np.where(seg_mask==i, 1, 0))
        seg_mask_roi = np.sum(np.stack(seg_mask_roi, axis=0), axis=0)
        
        lidar_pts, lidar = map_pointcloud_to_image(lidar, lidar_params['sensor2ego'], lidar_params['ego2global'],
                                        camera_params['sensor2ego'], camera_params['ego2global'])
        radar_pts, radar = map_pointcloud_to_image(radar, radar_params['sensor2ego'], radar_params['ego2global'],
                                        camera_params['sensor2ego'], camera_params['ego2global'])
        
        radar_pts = radar_pts[:, :3]
        valid_radar_pts_cnt = radar_pts.shape[0]
        if valid_radar_pts_cnt <= self.RADAR_PTS_NUM:
            padding_radar_pts = np.zeros((self.RADAR_PTS_NUM, 3), dtype=radar_pts.dtype)
            padding_radar_pts[:valid_radar_pts_cnt,:] = radar_pts
        else:
            random_idx = sorted(rng.choice(range(valid_radar_pts_cnt), size=(self.RADAR_PTS_NUM,), replace=False))
            padding_radar_pts = radar_pts[random_idx,:]

        # inds = (lidar[:, 2] > conf.min_depth) & (lidar[:, 2] < conf.max_depth)
        # lidar = lidar[inds]
        
        # # Filter out the Lidar point cloud with overlapping near and far depth
        # uvs, depths = lidar[:, :2], lidar[:, -1]
        # tree = scipy.spatial.KDTree(uvs)
        # res = tree.query_ball_point(uvs, conf.query_radius)
        # filter_mask = np.array([
        #     (depths[i] - min(depths[inds])) / depths[i] > 0.1
        #     for i, inds in enumerate(res)])
        # lidar[filter_mask] = 0
        lidar = get_depth_map(lidar[:, :3], img.shape[:2])
        
        inds = canvas_filter(radar[:, :2], img.shape[:2])
        radar = radar[inds]
        radar = get_radar_map(radar[:, :3], img.shape[:2])
        
        img        = Image.fromarray(img[...,::-1])  # BGR->RGB
        lidar      = Image.fromarray(lidar.astype('float32'), mode='F')
        radar      = Image.fromarray(radar.astype('float32'), mode='F')

        seg_mask_roi = seg_mask_roi.astype('float32')
        if isinstance(seg_mask_roi, np.float32):
            seg_mask_roi = np.array([seg_mask_roi])
        seg_mask_roi   = torch.from_numpy(seg_mask_roi)[None]
        
        # Aug
        try:
            img, lidar, radar, seg_mask_roi = augmention(img, lidar, radar, seg_mask_roi)
        except:
            pass
        
        lidar, radar = (np.array(d) for d in (lidar, radar))
        
        lidar_mask, radar_mask = (
            (d > 0).astype(np.uint8) for d in (lidar, radar))
        
        lidar, radar = (d[None] for d in (lidar, radar))
        lidar_mask, radar_mask = (
            d[None] for d in (lidar_mask, radar_mask))
        
        img = np.array(img)[...,::-1] # RGB -> BGR
        img = np.ascontiguousarray(img.transpose(2, 0, 1))

        ret_list = [
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes_3d,
            gt_labels_3d,
        ]

        if self.return_depth:
            ret_list.append(image_data_list[6])
        else:
            ret_list.append(None)
        if self.return_radar_pv:
            ret_list.append(image_data_list[7])
        else:
            ret_list.append(None)

        img = torch.tensor(img)
        img = img.float()
        padding_radar_pts = torch.tensor(padding_radar_pts)
        valid_radar_pts_cnt = torch.tensor(valid_radar_pts_cnt)
        radar = torch.tensor(radar)
        lidar = torch.tensor(lidar)
        lidar_mask = torch.tensor(lidar_mask)
        seg_mask_roi = torch.tensor(seg_mask_roi)

        ret_list.extend([img, padding_radar_pts, valid_radar_pts_cnt, radar, lidar, lidar_mask, seg_mask_roi])

        # print('IMG', img.shape, 'RADAR', radar.shape)

        return ret_list


def augmention(img:Image, lidar:Image, radar:Image, seg_mask:torch.Tensor):
    width, height = img.size
    _scale = rng.uniform(1.0, 1.3) # resize scale > 1.0, no info loss
    scale  = int(height * _scale)
    degree = np.random.uniform(-5.0, 5.0)
    flip   = rng.uniform(0.0, 1.0)
    # Horizontal flip
    if flip > 0.5:
        img   = TF.hflip(img)
        lidar = TF.hflip(lidar)
        radar = TF.hflip(radar)
        seg_mask   = TF.hflip(seg_mask)
    
    # Color jitter
    brightness = rng.uniform(0.6, 1.4)
    contrast   = rng.uniform(0.6, 1.4)
    saturation = rng.uniform(0.6, 1.4)

    img = TF.adjust_brightness(img, brightness)
    img = TF.adjust_contrast(img, contrast)
    img = TF.adjust_saturation(img, saturation)
    
    # Resize
    img        = TF.resize(img,   scale, interpolation=InterpolationMode.BICUBIC)
    lidar      = TF.resize(lidar, scale, interpolation=InterpolationMode.NEAREST)
    radar      = TF.resize(radar, scale, interpolation=InterpolationMode.NEAREST)
    seg_mask   = TF.resize(seg_mask, scale, interpolation=InterpolationMode.NEAREST)

    # Crop
    width, height = img.size
    ch, cw = conf.input_h, conf.input_w
    h_start = rng.integers(0, height - ch)
    w_start = rng.integers(0, width - cw)

    img          = TF.crop(img,   h_start, w_start, ch, cw)
    lidar        = TF.crop(lidar, h_start, w_start, ch, cw)
    radar        = TF.crop(radar, h_start, w_start, ch, cw)
    seg_mask     = TF.crop(seg_mask, h_start, w_start, ch, cw)

    img     = TF.gaussian_blur(img, kernel_size=3, )

    return img, lidar, radar, seg_mask

def collate_fn(data,
               is_return_image=True,
               is_return_depth=False,
               is_return_radar_pv=False):
    assert (is_return_image or is_return_depth or is_return_radar_pv) is True
    imgs_batch = list()
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    sensor2sensor_mats_batch = list()
    bda_mat_batch = list()
    gt_boxes_3d_batch = list()
    gt_labels_3d_batch = list()
    img_metas_batch = list()
    depth_labels_batch = list()
    radar_pv_batch = list()
    img_batch = list()
    padding_radar_pts_batch = list()
    valid_radar_pts_cnt_batch = list()
    radar_batch = list()
    lidar_batch = list()
    lidar_mask_batch = list()
    seg_mask_roi_batch = list()

    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
        ) = iter_data[:10]
        if is_return_depth:
            gt_depth = iter_data[10]
            depth_labels_batch.append(gt_depth)
        if is_return_radar_pv:
            radar_pv = iter_data[11]
            radar_pv_batch.append(radar_pv)

        (
            img,
            padding_radar_pts,
            valid_radar_pts_cnt,
            radar,
            lidar,
            lidar_mask,
            seg_mask_roi,
        ) = iter_data[-7:]

        imgs_batch.append(sweep_imgs)
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
        bda_mat_batch.append(bda_mat)
        img_metas_batch.append(img_metas)
        gt_boxes_3d_batch.append(gt_boxes)
        gt_labels_3d_batch.append(gt_labels)

        img_batch.append(img)
        padding_radar_pts_batch.append(padding_radar_pts)
        valid_radar_pts_cnt_batch.append(valid_radar_pts_cnt)
        radar_batch.append(radar)
        lidar_batch.append(lidar)
        lidar_mask_batch.append(lidar_mask)
        seg_mask_roi_batch.append(seg_mask_roi)

    if is_return_image:
        mats_dict = dict()
        mats_dict['sensor2ego_mats'] = torch.stack(sensor2ego_mats_batch)
        mats_dict['intrin_mats'] = torch.stack(intrin_mats_batch)
        mats_dict['ida_mats'] = torch.stack(ida_mats_batch)
        mats_dict['sensor2sensor_mats'] = torch.stack(sensor2sensor_mats_batch)
        mats_dict['bda_mat'] = torch.stack(bda_mat_batch)
        ret_list = [
            torch.stack(imgs_batch),
            mats_dict,
            img_metas_batch,
            gt_boxes_3d_batch,
            gt_labels_3d_batch,
            None,  # reserve for segmentation
            torch.stack(img_batch),
            torch.stack(padding_radar_pts_batch),
            torch.stack(valid_radar_pts_cnt_batch),
            torch.stack(radar_batch),
            torch.stack(lidar_batch),
            torch.stack(lidar_mask_batch),
            torch.stack(seg_mask_roi_batch),
        ]
    else:
        ret_list = [
            None,
            None,
            img_metas_batch,
            gt_boxes_3d_batch,
            gt_labels_3d_batch,
            None,
            img_batch,
            padding_radar_pts_batch,
            valid_radar_pts_cnt_batch,
            radar_batch,
            lidar_batch,
            lidar_mask_batch,
            seg_mask_roi_batch,
        ]
    if is_return_depth:
        ret_list.append(torch.stack(depth_labels_batch))
    else:
        ret_list.append(None)
    if is_return_radar_pv:
        ret_list.append(torch.stack(radar_pv_batch))
    else:
        ret_list.append(None)

    return ret_list