# Copyright (c) Megvii Inc. All rights reserved.
import torch
from exps.base_cli import run_cli

# Basic Experiment
from exps.mv.bev_depth_lss_r50_256x704_128x128_24e_ema_cbgs import (
    BEVDepthLightningModel as BaseExp,
)

from exps.MatrixVT.matrixvt import MatrixVT
from layers.heads.bev_depth_head import BEVDepthHead


class MatrixVT_Det(torch.nn.Module):
    def __init__(self, backbone_conf, head_conf, is_train_depth=False):
        super().__init__()
        self.backbone = MatrixVT(**backbone_conf)
        self.head = BEVDepthHead(**head_conf)
        self.is_train_depth = is_train_depth

    def forward(
        self,
        x,
        mats_dict,
        timestamps=None,
    ):
        """Forward function

        Args:
            x (Tensor): Input ferature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if self.is_train_depth and self.training:
            x, depth_pred = self.backbone(
                x, mats_dict, timestamps, is_return_depth=True
            )
            preds = self.head(x)
            return preds, depth_pred
        else:
            x = self.backbone(x, mats_dict, timestamps)
            preds = self.head(x)
            return preds

    def get_targets(self, gt_boxes, gt_labels):
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)


class MatrixVT_Exp(BaseExp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = MatrixVT_Det(
            self.backbone_conf, self.head_conf, is_train_depth=True
        )
        self.data_use_cbgs = True


if __name__ == "__main__":
    run_cli(
        MatrixVT_Exp,
        "matrixvt_bev_depth_lss_r50_256x704_128x128_24e_ema_cbgs",
        use_ema=True,
    )
