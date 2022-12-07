# Copyright (c) Megvii Inc. All rights reserved.
from bevdepth.exps.base_cli import run_cli
# Basic Experiment
from bevdepth.exps.mv.bev_depth_lss_r50_256x704_128x128_24e_ema_cbgs import \
    BEVDepthLightningModel as BaseExp
# new model
from bevdepth.models.matrixvt_det import MatrixVT_Det


class MatrixVT_Exp(BaseExp):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = MatrixVT_Det(self.backbone_conf,
                                  self.head_conf,
                                  is_train_depth=True)
        self.data_use_cbgs = True


if __name__ == '__main__':
    run_cli(
        MatrixVT_Exp,
        'matrixvt_bev_depth_lss_r50_256x704_128x128_24e_ema_cbgs',
        use_ema=True,
    )
