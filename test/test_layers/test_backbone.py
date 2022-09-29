import unittest

import pytest
import torch

from layers.backbones.base_lss_fpn import BaseLSSFPN


class TestLSSFPN(unittest.TestCase):
    def setUp(self) -> None:
        backbone_conf = {
            'x_bound': [-10, 10, 0.5],
            'y_bound': [-10, 10, 0.5],
            'z_bound': [-5, 3, 8],
            'd_bound': [2.0, 22, 1.0],
            'final_dim': [64, 64],
            'output_channels':
            10,
            'downsample_factor':
            16,
            'img_backbone_conf':
            dict(type='ResNet',
                 depth=18,
                 frozen_stages=0,
                 out_indices=[0, 1, 2, 3],
                 norm_eval=False,
                 base_channels=8),
            'img_neck_conf':
            dict(
                type='SECONDFPN',
                in_channels=[8, 16, 32, 64],
                upsample_strides=[0.25, 0.5, 1, 2],
                out_channels=[16, 16, 16, 16],
            ),
            'depth_net_conf':
            dict(in_channels=64, mid_channels=64),
        }
        self.lss_fpn = BaseLSSFPN(**backbone_conf).cuda()

    @pytest.mark.skipif(torch.cuda.is_available() is False,
                        reason='No gpu available.')
    def test_forward(self):
        sweep_imgs = torch.rand(2, 2, 6, 3, 64, 64).cuda()
        sensor2ego_mats = torch.rand(2, 2, 6, 4, 4).cuda()
        intrin_mats = torch.rand(2, 2, 6, 4, 4).cuda()
        ida_mats = torch.rand(2, 2, 6, 4, 4).cuda()
        sensor2sensor_mats = torch.rand(2, 2, 6, 4, 4).cuda()
        bda_mat = torch.rand(2, 4, 4).cuda()
        mats_dict = dict()
        mats_dict['sensor2ego_mats'] = sensor2ego_mats
        mats_dict['intrin_mats'] = intrin_mats
        mats_dict['ida_mats'] = ida_mats
        mats_dict['sensor2sensor_mats'] = sensor2sensor_mats
        mats_dict['bda_mat'] = bda_mat
        preds = self.lss_fpn.forward(sweep_imgs, mats_dict)
        assert preds.shape == torch.Size([2, 20, 40, 40])
