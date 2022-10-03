# Copyright (c) Megvii Inc. All rights reserved.
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed

from exps.base_cli import run_cli
from exps.base_exp import BEVDepthLightningModel as BaseBEVDepthLightningModel


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        return [optimizer]


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_lss_r50_256x704_128x128_24e_ema',
            use_ema=True)
