# Copyright (c) Megvii Inc. All rights reserved.
from bevdepth.exps.nuscenes.base_cli import run_cli
from bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel  # noqa
from bevdepth.models.base_bev_depth import BaseBEVDepth


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        final_dim = (512, 1408)
        self.backbone_conf['final_dim'] = final_dim
        self.ida_aug_conf['resize_lim'] = (0.386 * 2, 0.55 * 2)
        self.ida_aug_conf['final_dim'] = final_dim
        self.model = BaseBEVDepth(self.backbone_conf,
                                  self.head_conf,
                                  is_train_depth=True)


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_lss_r50_256x704_128x128_24e_2key')
