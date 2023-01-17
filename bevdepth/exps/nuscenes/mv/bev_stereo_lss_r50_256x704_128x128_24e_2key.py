# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3456
mATE: 0.6589
mASE: 0.2774
mAOE: 0.5500
mAVE: 0.4980
mAAE: 0.2278
NDS: 0.4516
Eval time: 158.2s
Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.510   0.525   0.165   0.188   0.510   0.226
truck   0.288   0.698   0.220   0.205   0.443   0.227
bus     0.378   0.622   0.210   0.135   0.896   0.289
trailer 0.156   1.003   0.219   0.482   0.609   0.179
construction_vehicle    0.094   0.929   0.502   1.209   0.108   0.365
pedestrian      0.356   0.728   0.297   1.005   0.579   0.319
motorcycle      0.361   0.571   0.258   0.734   0.631   0.211
bicycle 0.318   0.533   0.269   0.793   0.208   0.007
traffic_cone    0.488   0.501   0.355   nan     nan     nan
barrier 0.506   0.478   0.277   0.200   nan     nan
"""
from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.base_exp import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel
from bevdepth.models.bev_stereo import BEVStereo


class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_idxes = [-1]
        self.head_conf['bev_backbone_conf']['in_channels'] = 80 * (
            len(self.key_idxes) + 1)
        self.head_conf['bev_neck_conf']['in_channels'] = [
            80 * (len(self.key_idxes) + 1), 160, 320, 640
        ]
        self.head_conf['train_cfg']['code_weights'] = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
        self.head_conf['test_cfg']['thresh_scale'] = [
            0.6, 0.4, 0.4, 0.7, 0.8, 0.9
        ]
        self.head_conf['test_cfg']['nms_type'] = 'size_aware_circle'
        self.model = BEVStereo(self.backbone_conf,
                               self.head_conf,
                               is_train_depth=True)


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_stereo_lss_r50_256x704_128x128_24e_2key')
