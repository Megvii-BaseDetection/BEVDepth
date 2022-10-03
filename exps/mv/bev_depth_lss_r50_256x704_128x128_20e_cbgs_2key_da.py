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
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR

from exps.base_cli import run_cli
from exps.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel
from layers.backbones.base_lss_fpn import BaseLSSFPN as BaseLSSFPN
from layers.heads.bev_depth_head import BEVDepthHead
from models.base_bev_depth import BaseBEVDepth as BaseBEVDepth


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
                                      weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, [16, 19])
        return [[optimizer], [scheduler]]


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da')
