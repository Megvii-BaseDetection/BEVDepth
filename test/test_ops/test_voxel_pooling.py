import unittest

import pytest
import torch

from ops.voxel_pooling import voxel_pooling


class TestLSSFPN(unittest.TestCase):
    @pytest.mark.skipif(condition=torch.cuda.is_available() is False,
                        reason='No gpu available.')
    def test_voxel_pooling(self):
        import numpy as np

        np.random.seed(0)
        torch.manual_seed(0)
        geom_xyz = torch.rand([2, 6, 10, 10, 10, 3]) * 160 - 80
        geom_xyz[..., 2] /= 100
        geom_xyz = geom_xyz.reshape(2, -1, 3)
        features = torch.rand([2, 6, 10, 10, 10, 80]) - 0.5
        gt_features = features.reshape(2, -1, 80)
        gt_bev_featuremap = features.new_zeros(2, 128, 128, 80)
        for i in range(2):
            for j in range(geom_xyz.shape[1]):
                x = geom_xyz[i, j, 0].int()
                y = geom_xyz[i, j, 1].int()
                z = geom_xyz[i, j, 2].int()
                if x < 0 or x >= 128 or y < 0 or y >= 128 or z < 0 or z >= 1:
                    continue
                gt_bev_featuremap[i, y, x, :] += gt_features[i, j, :]
        gt_bev_featuremap = gt_bev_featuremap.permute(0, 3, 1, 2).cuda()
        bev_featuremap = voxel_pooling(
            geom_xyz.cuda().int(), features.cuda(),
            torch.tensor([128, 128, 1], dtype=torch.int, device='cuda'))
        assert torch.allclose(gt_bev_featuremap.cuda(),
                              bev_featuremap,
                              rtol=1e-3)
