# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3427
mATE: 0.6560
mASE: 0.2784
mAOE: 0.5982
mAVE: 0.5347
mAAE: 0.2228
NDS: 0.4423
Eval time: 116.3s
Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.508   0.518   0.163   0.188   0.534   0.230
truck   0.268   0.709   0.214   0.215   0.510   0.226
bus     0.379   0.640   0.207   0.142   1.049   0.315
trailer 0.151   0.953   0.240   0.541   0.618   0.113
construction_vehicle    0.092   0.955   0.514   1.360   0.113   0.394
pedestrian      0.350   0.727   0.300   1.013   0.598   0.328
motorcycle      0.371   0.576   0.259   0.777   0.634   0.175
bicycle 0.325   0.512   0.261   0.942   0.221   0.002
traffic_cone    0.489   0.503   0.345   nan     nan     nan
barrier 0.495   0.468   0.280   0.206   nan     nan
"""
from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.mv.bev_stereo_lss_r50_256x704_128x128_24e_key4 import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel  # noqa


class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_sweeps = 2
        self.sweep_idxes = [4]
        self.key_idxes = list()


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_stereo_lss_r50_256x704_128x128_24e_key4_ema')
