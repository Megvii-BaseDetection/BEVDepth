from perceptron.layers.blocks_3d.mmdet3d.lss_fpn import LSSFPN
from torch import nn

from layers.heads.bevdepth_head import BEVDepthHead

__all__ = ['BEVDepth']


class BEVDepth(nn.Module):
    """Source code of `BEVDepth`, `https://arxiv.org/abs/2112.11790`.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, backbone_conf, head_conf, is_train_depth=False):
        super(BEVDepth, self).__init__()
        self.backbone = LSSFPN(**backbone_conf)
        self.head = BEVDepthHead(**head_conf)
        self.is_train_depth = is_train_depth

    # TODO: Merge to one input
    def forward(
        self,
        x,
        mats,
        timestamps=None,
    ):
        if self.is_train_depth and self.training:
            x, depth_pred = self.backbone(x,
                                          mats,
                                          timestamps,
                                          is_return_depth=True)
            preds = self.head(x)
            return preds, depth_pred
        else:
            x = self.backbone(x, mats, timestamps)
            preds = self.head(x)
            return preds

    def get_targets(self, gt_boxes, gt_labels):
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)
