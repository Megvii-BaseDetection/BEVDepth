# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3589
mATE: 0.6119
mASE: 0.2692
mAOE: 0.5074
mAVE: 0.4086
mAAE: 0.2009
NDS: 0.4797
Eval time: 183.3s
Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.559   0.475   0.157   0.112   0.370   0.205
truck   0.270   0.659   0.196   0.103   0.356   0.181
bus     0.374   0.651   0.184   0.072   0.846   0.326
trailer 0.179   0.963   0.227   0.512   0.294   0.127
construction_vehicle    0.081   0.825   0.481   1.352   0.094   0.345
pedestrian      0.363   0.690   0.297   0.831   0.491   0.244
motorcycle      0.354   0.580   0.255   0.545   0.615   0.164
bicycle 0.301   0.447   0.280   0.920   0.203   0.015
traffic_cone    0.539   0.435   0.324   nan     nan     nan
barrier 0.569   0.394   0.293   0.120   nan     nan
"""
from exps.base_cli import run_cli
from exps.mv.bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da import \
    BEVDepthLightningModel

if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema',
            use_ema=True)
