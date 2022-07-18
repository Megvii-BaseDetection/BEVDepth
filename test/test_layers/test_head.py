import unittest

import pytest
import torch
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

from layers.heads.bev_depth_head import BEVDepthHead


class TestLSSFPN(unittest.TestCase):
    def setUp(self) -> None:
        bev_backbone = dict(
            type='ResNet',
            in_channels=10,
            depth=18,
            num_stages=3,
            strides=(1, 2, 2),
            dilations=(1, 1, 1),
            out_indices=[0, 1, 2],
            norm_eval=False,
            base_channels=20,
        )

        bev_neck = dict(type='SECONDFPN',
                        in_channels=[10, 20, 40, 80],
                        upsample_strides=[1, 2, 4, 8],
                        out_channels=[8, 8, 8, 8])

        TASKS = [
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ]

        common_heads = dict(reg=(2, 2),
                            height=(1, 2),
                            dim=(3, 2),
                            rot=(2, 2),
                            vel=(2, 2))

        bbox_coder = dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=32,
            voxel_size=[0.2, 0.2, 8],
            pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
            code_size=9,
        )

        train_cfg = dict(
            point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            out_size_factor=32,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )

        test_cfg = dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=[0.2, 0.2, 8],
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
        )

        head_conf = {
            'bev_backbone_conf': bev_backbone,
            'bev_neck_conf': bev_neck,
            'tasks': TASKS,
            'common_heads': common_heads,
            'bbox_coder': bbox_coder,
            'train_cfg': train_cfg,
            'test_cfg': test_cfg,
            'in_channels': 32,  # Equal to bev_neck output_channels.
            'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
            'loss_bbox': dict(type='L1Loss',
                              reduction='mean',
                              loss_weight=0.25),
            'gaussian_overlap': 0.1,
            'min_radius': 2,
        }
        self.bevdet_head = BEVDepthHead(**head_conf).cuda()

    @pytest.mark.skipif(torch.cuda.is_available() is False,
                        reason='No gpu available.')
    def test_forward(self):
        x = torch.rand(2, 10, 32, 32).cuda()
        ret_results = self.bevdet_head.forward(x)
        assert len(ret_results) == 6
        assert ret_results[0][0]['reg'].shape == torch.Size([2, 2, 32, 32])
        assert ret_results[0][0]['height'].shape == torch.Size([2, 1, 32, 32])
        assert ret_results[0][0]['dim'].shape == torch.Size([2, 3, 32, 32])
        assert ret_results[0][0]['rot'].shape == torch.Size([2, 2, 32, 32])
        assert ret_results[0][0]['vel'].shape == torch.Size([2, 2, 32, 32])
        assert ret_results[0][0]['heatmap'].shape == torch.Size([2, 1, 32, 32])

    @pytest.mark.skipif(torch.cuda.is_available() is False,
                        reason='No gpu available.')
    def test_get_targets(self):
        gt_boxes_3d_0 = torch.rand(10, 9).cuda()
        gt_boxes_3d_1 = torch.rand(15, 9).cuda()
        gt_boxes_3d_0[:, :2] *= 10
        gt_boxes_3d_1[:, :2] *= 10
        gt_labels_3d_0 = torch.randint(0, 10, (10, )).cuda()
        gt_labels_3d_1 = torch.randint(0, 10, (15, )).cuda()
        gt_boxes_3d = [gt_boxes_3d_0, gt_boxes_3d_1]
        gt_labels_3d = [gt_labels_3d_0, gt_labels_3d_1]
        heatmaps, anno_boxes, inds, masks = self.bevdet_head.get_targets(
            gt_boxes_3d, gt_labels_3d)
        assert len(heatmaps) == 6
        assert len(anno_boxes) == 6
        assert len(inds) == 6
        assert len(masks) == 6
        assert heatmaps[0].shape == torch.Size([2, 1, 16, 16])
        assert anno_boxes[0].shape == torch.Size([2, 500, 10])
        assert inds[0].shape == torch.Size([2, 500])
        assert masks[0].shape == torch.Size([2, 500])

    @pytest.mark.skipif(torch.cuda.is_available() is False,
                        reason='No gpu available.')
    def test_get_bboxes(self):
        x = torch.rand(2, 10, 32, 32).cuda()
        ret_results = self.bevdet_head.forward(x)
        img_metas = [
            dict(box_type_3d=LiDARInstance3DBoxes),
            dict(box_type_3d=LiDARInstance3DBoxes)
        ]
        pred_bboxes = self.bevdet_head.get_bboxes(ret_results,
                                                  img_metas=img_metas)
        assert len(pred_bboxes) == 2
        assert len(pred_bboxes[0]) == 3
        assert pred_bboxes[0][1].shape == torch.Size([498])
        assert pred_bboxes[0][2].shape == torch.Size([498])
