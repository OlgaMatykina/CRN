import unittest

import numpy as np
import torch

from datasets.nusc_det_dataset import NuscDatasetRadarDet, collate_fn
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
ida_aug_conf = {
    'resize_lim': (0.4, 0.4),
    'final_dim':
    final_dim,
    'rot_lim': (0, 0),
    'H':
    H,
    'W':
    W,
    'rand_flip':
    True,
    'bot_pct_lim': (0.0, 0.0),
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
}

bda_aug_conf = {
    'rot_ratio': 1.0,
    'rot_lim': (0, 0),
    'scale_lim': (1, 1),
    'flip_dx_ratio': 0,
    'flip_dy_ratio': 0
}

rda_aug_conf = {
    'N_sweeps': 6,
    'N_use': 5,
    'drop_ratio': 0.1,
}

class TestNuscMVDetData(unittest.TestCase):
    def test_voxel_pooling(self):
        np.random.seed(0)
        torch.random.manual_seed(0)
        nusc = NuscDatasetRadarDet(ida_aug_conf,
                                bda_aug_conf,
                                rda_aug_conf,
                                CLASSES,
                                # './test/data/nuscenes',
                                # './test/data/nuscenes/infos.pkl',
                                # '../HPR1/nuscenes',
                                # '../HPR1/nuscenes_radar_5sweeps_infos_val.pkl',
                                './data/nuScenes',
                                './data/nuScenes/nuscenes_infos_val.pkl',
                                True,
                                sweep_idxes=[],
                                return_depth=True,
                                return_radar_pv=True)
        ret_list = nusc[1]
        # dataloader = torch.utils.data.DataLoader(
        #     nusc,
        #     batch_size=3,
        #     num_workers=4,
        #     drop_last=True,
        #     shuffle=False,
        #     collate_fn=partial(collate_fn,
        #                        is_return_image=True,
        #                        is_return_depth=True,
        #                        is_return_radar_pv=True),
        #     sampler=None,
        # )
        # dataiter = iter(dataloader)
        # batch = next(dataiter)
        # print('BATCH', batch)
        # assert torch.isclose(ret_list[0].mean(),
        #                      torch.tensor(-0.4667),
        #                      rtol=1e-3)
        # assert torch.isclose(ret_list[1].mean(),
        #                      torch.tensor(0.1678),
        #                      rtol=1e-3)
        # assert torch.isclose(ret_list[2].mean(),
        #                      torch.tensor(230.0464),
        #                      rtol=1e-3)
        # assert torch.isclose(ret_list[3].mean(),
        #                      torch.tensor(8.3250),
        #                      rtol=1e-3)
        # assert torch.isclose(ret_list[4].mean(), torch.tensor(0.25), rtol=1e-3)
        # assert torch.isclose(ret_list[5].mean(), torch.tensor(0.25), rtol=1e-3)

if __name__ == '__main__':
    unittest.main()