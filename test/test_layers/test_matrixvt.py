import unittest

import torch
from layers.backbones.matrixvt import MatrixVT


class TestMatrixVT(unittest.TestCase):
    def setUp(self) -> None:
            backbone_conf = {
                "x_bound": [-51.2, 51.2, 0.8],  # BEV grids bounds and size (m)
                "y_bound": [-51.2, 51.2, 0.8],  # BEV grids bounds and size (m)
                "z_bound": [-5, 3, 8],  # BEV grids bounds and size (m)
                "d_bound": [2.0, 58.0, 0.5],  # Categorical Depth bounds and devision (m)
                "final_dim": (256, 704),  # img size for model input (pix)
                "output_channels": 80,  # BEV feature channels
                "downsample_factor": 16,  # ds factor of the feature to be projected to BEV (e.g. 256x704 -> 16x44)
                "img_backbone_conf": dict(
                    type="ResNet",
                    depth=50,
                    frozen_stages=0,
                    out_indices=[0, 1, 2, 3],
                    norm_eval=False,
                    init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
                ),
                "img_neck_conf": dict(
                    type="SECONDFPN",
                    in_channels=[256, 512, 1024, 2048],
                    upsample_strides=[0.25, 0.5, 1, 2],
                    out_channels=[128, 128, 128, 128],
                ),
                "depth_net_conf": dict(in_channels=512, mid_channels=512),
            }

            model = MatrixVT(**backbone_conf)
            
            return model

    def test_forward(self):
        model = self.setUp()
        bev_feature, depth = model(
            torch.rand((2, 1, 6, 3, 256, 704)),
            {
                "sensor2ego_mats": torch.rand((2, 1, 6, 4, 4)),
                "intrin_mats": torch.rand((2, 1, 6, 4, 4)),
                "ida_mats": torch.rand((2, 1, 6, 4, 4)),
                "sensor2sensor_mats": torch.rand((2, 1, 6, 4, 4)),
                "bda_mat": torch.rand((2, 4, 4)),
            },
            is_return_depth=True
        )

        assert bev_feature.shape == torch.Size([2, 80, 128, 128])
        assert depth.shape == torch.Size([12, 112, 16, 44])