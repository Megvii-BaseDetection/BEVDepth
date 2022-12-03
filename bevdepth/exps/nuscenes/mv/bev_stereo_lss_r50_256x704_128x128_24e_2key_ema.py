# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3494
mATE: 0.6672
mASE: 0.2785
mAOE: 0.5607
mAVE: 0.4687
mAAE: 0.2295
NDS: 0.4542
Eval time: 166.7s
Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.509   0.522   0.163   0.187   0.507   0.228
truck   0.287   0.694   0.213   0.202   0.449   0.229
bus     0.390   0.681   0.207   0.152   0.902   0.261
trailer 0.167   0.945   0.248   0.491   0.340   0.185
construction_vehicle    0.087   1.057   0.515   1.199   0.104   0.377
pedestrian      0.351   0.729   0.299   0.987   0.575   0.321
motorcycle      0.368   0.581   0.262   0.721   0.663   0.226
bicycle 0.338   0.494   0.258   0.921   0.209   0.008
traffic_cone    0.494   0.502   0.341   nan     nan     nan
barrier 0.502   0.467   0.278   0.185   nan     nan
"""
from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.mv.bev_stereo_lss_r50_256x704_128x128_24e_2key import \
    BEVDepthLightningModel  # noqa

if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_stereo_lss_r50_256x704_128x128_24e_2key_ema',
            use_ema=True)
