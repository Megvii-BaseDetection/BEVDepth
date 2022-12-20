from bevdepth.layers.backbones.matrixvt import MatrixVT
from bevdepth.models.base_bev_depth import BaseBEVDepth


class MatrixVT_Det(BaseBEVDepth):
    """Implementation of MatrixVT for Object Detection.

        Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    def __init__(self, backbone_conf, head_conf, is_train_depth=False):
        super().__init__(backbone_conf, head_conf, is_train_depth)
        self.backbone = MatrixVT(**backbone_conf)
