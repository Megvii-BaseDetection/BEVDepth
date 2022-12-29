# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3721
mATE: 0.5980
mASE: 0.2701
mAOE: 0.4381
mAVE: 0.3672
mAAE: 0.1898
NDS: 0.4997
Eval time: 138.0s
Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.567   0.457   0.156   0.104   0.343   0.204
truck   0.299   0.650   0.205   0.103   0.321   0.197
bus     0.394   0.613   0.203   0.106   0.643   0.252
trailer 0.178   0.991   0.239   0.433   0.345   0.070
construction_vehicle    0.102   0.826   0.458   1.055   0.114   0.372
pedestrian      0.402   0.653   0.297   0.803   0.479   0.249
motorcycle      0.356   0.553   0.251   0.450   0.512   0.168
bicycle 0.311   0.440   0.265   0.779   0.180   0.006
traffic_cone    0.552   0.420   0.336   nan     nan     nan
barrier 0.561   0.377   0.291   0.111   nan     nan
"""
from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.mv.bev_stereo_lss_r50_256x704_128x128_20e_cbgs_2key_da import \
    BEVDepthLightningModel  # noqa

if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_stereo_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema',
            use_ema=True,
            extra_trainer_config_args={'epochs': 20})
