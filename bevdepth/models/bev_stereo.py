from bevdepth.layers.backbones.bevstereo_lss_fpn import BEVStereoLSSFPN
from bevdepth.models.base_bev_depth import BaseBEVDepth

__all__ = ['BEVStereo']


class BEVStereo(BaseBEVDepth):
    """Source code of `BEVStereo`, `https://arxiv.org/abs/2209.10248`.

    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, backbone_conf, head_conf, is_train_depth=False):
        super(BEVStereo, self).__init__(backbone_conf, head_conf,
                                        is_train_depth)
        self.backbone = BEVStereoLSSFPN(**backbone_conf)
