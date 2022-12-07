import unittest

import torch

from bevdepth.layers.backbones.matrixvt import MatrixVT


class TestMatrixVT(unittest.TestCase):

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
            dict(
                type='ResNet',
                depth=18,
                frozen_stages=0,
                out_indices=[0, 1, 2, 3],
                norm_eval=False,
                base_channels=8,
            ),
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

        model = MatrixVT(**backbone_conf)

        return model

    def test_forward(self):
        model = self.setUp()
        bev_feature, depth = model(
            torch.rand((2, 1, 6, 3, 64, 64)),
            {
                'sensor2ego_mats': torch.rand((2, 1, 6, 4, 4)),
                'intrin_mats': torch.rand((2, 1, 6, 4, 4)),
                'ida_mats': torch.rand((2, 1, 6, 4, 4)),
                'sensor2sensor_mats': torch.rand((2, 1, 6, 4, 4)),
                'bda_mat': torch.rand((2, 4, 4)),
            },
            is_return_depth=True,
        )
        print(bev_feature.shape)
        print(depth.shape)
        assert bev_feature.shape == torch.Size([2, 10, 40, 40])
        assert depth.shape == torch.Size([12, 20, 4, 4])
