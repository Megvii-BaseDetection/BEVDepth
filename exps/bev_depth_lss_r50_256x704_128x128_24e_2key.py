"""train: python exps/bev_depth_lss_r50_256x704_128x128_24e_2key.py --amp_backend native --gpus 8 -b 8 # noqa
   eval: python exps/bev_depth_lss_r50_256x704_128x128_24e_2key.py --ckpt_path [CKPT_PATH] -e -b 8 # noqa

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
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
