# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3329
mATE: 0.6832
mASE: 0.2761
mAOE: 0.5446
mAVE: 0.5258
mAAE: 0.2259
NDS: 0.4409

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.505   0.531   0.165   0.189   0.618   0.234
truck   0.274   0.731   0.206   0.211   0.546   0.223
bus     0.394   0.673   0.219   0.148   1.061   0.274
trailer 0.174   0.934   0.228   0.544   0.369   0.183
construction_vehicle    0.079   1.043   0.528   1.162   0.112   0.376
pedestrian      0.284   0.748   0.294   0.973   0.575   0.297
motorcycle      0.345   0.633   0.256   0.719   0.667   0.214
bicycle 0.314   0.544   0.252   0.778   0.259   0.007
traffic_cone    0.453   0.519   0.335   nan     nan     nan
barrier 0.506   0.475   0.279   0.178   nan     nan
"""
from exps.base_cli import run_cli
from exps.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import \
    BEVDepthLightningModel

if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_lss_r50_256x704_128x128_24e_2key_ema',
            use_ema=True)
