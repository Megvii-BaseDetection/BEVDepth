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
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast

from callbacks.ema import EMACallback
from exps.bev_depth_lss_r50_256x704_128x128_24e_2key import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel
from layers.backbones.lss_fpn import LSSFPN as BaseLSSFPN
from layers.heads.bev_depth_head import BEVDepthHead
from models.bev_depth import BEVDepth as BaseBEVDepth


class DepthAggregation(nn.Module):
    """
    pixel cloud feature extraction
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x


class LSSFPN(BaseLSSFPN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.depth_aggregation_net = self._configure_depth_aggregation_net()

    def _configure_depth_aggregation_net(self):
        """build pixel cloud feature extractor"""
        return DepthAggregation(self.output_channels, self.output_channels,
                                self.output_channels)

    def _forward_voxel_net(self, img_feat_with_depth):
        # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
        img_feat_with_depth = img_feat_with_depth.permute(
            0, 3, 1, 4, 2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
        n, h, c, w, d = img_feat_with_depth.shape
        img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
        img_feat_with_depth = (
            self.depth_aggregation_net(img_feat_with_depth).view(
                n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous().float())
        return img_feat_with_depth


class BEVDepth(BaseBEVDepth):
    def __init__(self, backbone_conf, head_conf, is_train_depth=True):
        super(BaseBEVDepth, self).__init__()
        self.backbone = LSSFPN(**backbone_conf)
        self.head = BEVDepthHead(**head_conf)
        self.is_train_depth = is_train_depth


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = BEVDepth(self.backbone_conf,
                              self.head_conf,
                              is_train_depth=True)
        self.data_use_cbgs = True

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-2)
        return [optimizer]


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = BEVDepthLightningModel(**vars(args))
    train_dataloader = model.train_dataloader()
    ema_callback = EMACallback(len(train_dataloader.dataset) * args.max_epochs)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[ema_callback])
    if args.evaluate:
        trainer.test(model, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parser = BEVDepthLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(profiler='simple',
                        deterministic=False,
                        max_epochs=20,
                        accelerator='ddp',
                        num_sanity_val_steps=0,
                        gradient_clip_val=5,
                        limit_val_batches=0,
                        enable_checkpointing=False,
                        precision=16,
                        default_root_dir='./outputs/bev_depth_lss_r50_'
                        '256x704_128x128_20e_cbgs_2key_da')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
