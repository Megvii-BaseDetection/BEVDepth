import torch.nn as nn
from perceptron.engine.cli import BaseCli

from exps.bev_depth_lss_r50_256x704_128x128_24e_key4 import Exp as BaseExp
from layers.backbones.lss_fpn import LSSFPN as BaseLSSFPN
from layers.heads.bev_depth_head import BEVDepthHead
from models.bev_depth import BEVDepth as BaseBEVDepth


class PCFE(nn.Module):
    """
    pixel cloud feature extraction
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(PCFE, self).__init__()

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

    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x


class LSSFPN(BaseLSSFPN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pcfe = self._configure_pcfe()

    def _configure_pcfe(self):
        """build pixel cloud feature extractor"""
        return PCFE(self.output_channels, self.output_channels,
                    self.output_channels)

    def _forward_voxel_net(self, img_feat_with_depth):
        # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
        img_feat_with_depth = img_feat_with_depth.permute(
            0, 3, 1, 4, 2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
        n, h, c, w, d = img_feat_with_depth.shape
        img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
        img_feat_with_depth = (self.pcfe(img_feat_with_depth).view(
            n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous().float())
        return img_feat_with_depth


class BEVDepth(BaseBEVDepth):
    def __init__(self, backbone_conf, head_conf, is_train_depth=True):
        super(BaseBEVDepth, self).__init__()
        self.backbone = LSSFPN(**backbone_conf)
        self.head = BEVDepthHead(**head_conf)
        self.is_train_depth = is_train_depth


class Exp(BaseExp):
    def __init__(self, **kwargs):
        super(Exp, self).__init__(**kwargs)
        self._max_epoch = 20
        self.model_class = BEVDepth
        self.data_use_cbgs = True


if __name__ == '__main__':
    BaseCli(Exp).run()
