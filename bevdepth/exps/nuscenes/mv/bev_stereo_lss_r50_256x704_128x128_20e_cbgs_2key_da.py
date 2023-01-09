# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3576
mATE: 0.6071
mASE: 0.2684
mAOE: 0.4157
mAVE: 0.3928
mAAE: 0.2021
NDS: 0.4902
Eval time: 129.7s
Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.559   0.465   0.157   0.110   0.350   0.205
truck   0.285   0.633   0.205   0.101   0.304   0.209
bus     0.373   0.667   0.204   0.076   0.896   0.345
trailer 0.167   0.956   0.228   0.482   0.289   0.100
construction_vehicle    0.077   0.869   0.454   1.024   0.108   0.335
pedestrian      0.402   0.652   0.299   0.821   0.493   0.253
motorcycle      0.321   0.544   0.255   0.484   0.529   0.159
bicycle 0.276   0.466   0.272   0.522   0.173   0.011
traffic_cone    0.551   0.432   0.321   nan     nan     nan
barrier 0.565   0.386   0.287   0.121   nan     nan
"""
import torch
from torch.optim.lr_scheduler import MultiStepLR

from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.mv.bev_stereo_lss_r50_256x704_128x128_24e_2key import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel  # noqa
from bevdepth.models.bev_stereo import BEVStereo


class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone_conf['use_da'] = True
        self.data_use_cbgs = True
        self.basic_lr_per_img = 2e-4 / 32
        self.model = BEVStereo(self.backbone_conf,
                               self.head_conf,
                               is_train_depth=True)

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-2)
        scheduler = MultiStepLR(optimizer, [16, 19])
        return [[optimizer], [scheduler]]


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_stereo_lss_r50_256x704_128x128_20e_cbgs_2key_da',
            extra_trainer_config_args={'epochs': 20})
