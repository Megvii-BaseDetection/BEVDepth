# Copyright (c) Megvii Inc. All rights reserved.
import torch
from torch.optim.lr_scheduler import MultiStepLR

from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel  # noqa
from bevdepth.models.base_bev_depth import BaseBEVDepth


class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        final_dim = (640, 1600)
        self.backbone_conf['final_dim'] = final_dim
        self.ida_aug_conf['resize_lim'] = (0.386 * 2, 0.55 * 2)
        self.ida_aug_conf['final_dim'] = final_dim
        self.model = BaseBEVDepth(self.backbone_conf,
                                  self.head_conf,
                                  is_train_depth=True)

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-3)
        scheduler = MultiStepLR(optimizer, [19, 23])
        return [[optimizer], [scheduler]]


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_lss_r50_512x1408_128x128_24e_2key')
