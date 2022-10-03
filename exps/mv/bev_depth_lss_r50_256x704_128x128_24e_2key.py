# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3304
mATE: 0.7021
mASE: 0.2795
mAOE: 0.5346
mAVE: 0.5530
mAAE: 0.2274
NDS: 0.4355
Eval time: 171.8s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.499   0.540   0.165   0.211   0.650   0.233
truck   0.278   0.719   0.218   0.265   0.547   0.215
bus     0.386   0.661   0.211   0.171   1.132   0.274
trailer 0.168   1.034   0.235   0.548   0.408   0.168
construction_vehicle    0.075   1.124   0.510   1.177   0.111   0.385
pedestrian      0.284   0.757   0.298   0.966   0.578   0.301
motorcycle      0.335   0.624   0.263   0.621   0.734   0.237
bicycle 0.305   0.554   0.264   0.653   0.263   0.006
traffic_cone    0.462   0.516   0.355   nan     nan     nan
barrier 0.512   0.491   0.275   0.200   nan     nan
"""
from exps.base_cli import run_cli
from exps.base_exp import BEVDepthLightningModel as BaseBEVDepthLightningModel
from models.base_bev_depth import BaseBEVDepth


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_idxes = [-1]
        self.head_conf['bev_backbone_conf']['in_channels'] = 80 * (
            len(self.key_idxes) + 1)
        self.head_conf['bev_neck_conf']['in_channels'] = [
            80 * (len(self.key_idxes) + 1), 160, 320, 640
        ]
        self.head_conf['train_cfg']['code_weight'] = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
        self.model = BaseBEVDepth(self.backbone_conf,
                                  self.head_conf,
                                  is_train_depth=True)


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_lss_r50_256x704_128x128_24e_2key')
