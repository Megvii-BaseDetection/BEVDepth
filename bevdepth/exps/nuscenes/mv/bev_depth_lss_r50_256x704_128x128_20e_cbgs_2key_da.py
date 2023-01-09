# Copyright (c) Megvii Inc. All rights reserved.
"""
mAP: 0.3484
mATE: 0.6159
mASE: 0.2716
mAOE: 0.4144
mAVE: 0.4402
mAAE: 0.1954
NDS: 0.4805
Eval time: 110.7s
Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.553   0.480   0.157   0.117   0.386   0.205
truck   0.252   0.645   0.202   0.097   0.381   0.185
bus     0.378   0.674   0.197   0.090   0.871   0.298
trailer 0.163   0.932   0.230   0.409   0.543   0.098
construction_vehicle    0.076   0.878   0.495   1.015   0.103   0.344
pedestrian      0.361   0.694   0.300   0.816   0.491   0.247
motorcycle      0.319   0.569   0.252   0.431   0.552   0.181
bicycle 0.286   0.457   0.255   0.630   0.194   0.006
traffic_cone    0.536   0.438   0.339   nan     nan     nan
barrier 0.559   0.392   0.289   0.124   nan     nan
"""
import torch
from torch.optim.lr_scheduler import MultiStepLR

from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel  # noqa
from bevdepth.models.base_bev_depth import BaseBEVDepth as BaseBEVDepth


class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone_conf['use_da'] = True
        self.data_use_cbgs = True
        self.model = BaseBEVDepth(self.backbone_conf,
                                  self.head_conf,
                                  is_train_depth=True)

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, [16, 19])
        return [[optimizer], [scheduler]]


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da',
            extra_trainer_config_args={'epochs': 20})
