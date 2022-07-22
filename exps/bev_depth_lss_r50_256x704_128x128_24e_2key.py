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
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl

from callbacks.ema import EMACallback
from exps.bev_depth_lss_r50_256x704_128x128_24e import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel
from models.bev_depth import BEVDepth


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_idxes = [-1]
        self.head_conf['bev_backbone_conf']['in_channels'] = 80 * (
            len(self.key_idxes) + 1)
        self.head_conf['bev_neck_conf']['in_channels'] = [
            80 * (len(self.key_idxes) + 1), 160, 320, 640
        ]
        self.head_conf['train_cfg']['code_weight'] = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
        self.model = BEVDepth(self.backbone_conf,
                              self.head_conf,
                              is_train_depth=True)


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
    parser.set_defaults(
        profiler='simple',
        deterministic=False,
        max_epochs=24,
        accelerator='ddp',
        num_sanity_val_steps=0,
        gradient_clip_val=5,
        limit_val_batches=0,
        enable_checkpointing=False,
        precision=16,
        default_root_dir='./outputs/bev_depth_lss_r50_256x704_128x128_24e_2key'
    )
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
