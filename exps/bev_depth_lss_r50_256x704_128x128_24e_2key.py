"""train: python exps/bev_depth_lss_r50_256x704_128x128_24e_2key.py --amp -b 8
   eval: python exps/bev_depth_lss_r50_256x704_128x128_24e_2key.py --eval -b 4

mAP: 0.3361
mATE: 0.6974
mASE: 0.2813
mAOE: 0.5518
mAVE: 0.4931
mAAE: 0.2281
NDS: 0.4429
Eval time: 201.2s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.504   0.535   0.163   0.186   0.569   0.230
truck   0.274   0.755   0.215   0.203   0.484   0.218
bus     0.401   0.678   0.208   0.155   0.958   0.287
trailer 0.190   0.961   0.263   0.586   0.385   0.179
construction_vehicle    0.078   1.078   0.526   1.164   0.105   0.368
pedestrian      0.283   0.752   0.298   0.926   0.569   0.304
motorcycle      0.346   0.650   0.259   0.793   0.636   0.237
bicycle 0.311   0.571   0.263   0.756   0.239   0.004
traffic_cone    0.463   0.518   0.337   nan     nan     nan
barrier 0.513   0.477   0.279   0.196   nan     nan
"""

from perceptron.engine.cli import BaseCli

from exps.bev_depth_lss_r50_256x704_128x128_24e import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self, **kwargs):
        super(Exp, self).__init__(**kwargs)
        self.key_idxes = [-1]
        self.head_conf['bev_backbone_conf']['in_channels'] = 80 * (
            len(self.key_idxes) + 1)
        self.head_conf['bev_neck_conf']['in_channels'] = [
            80 * (len(self.key_idxes) + 1), 160, 320, 640
        ]
        self.head_conf['train_cfg']['code_weight'] = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]


if __name__ == '__main__':
    BaseCli(Exp).run()
