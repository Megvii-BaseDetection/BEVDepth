# Copyright (c) Megvii Inc. All rights reserved.
import math

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmdet.models.backbones.resnet import BasicBlock
from scipy.special import erf
from scipy.stats import norm
from torch import nn

from bevdepth.layers.backbones.base_lss_fpn import (ASPP, BaseLSSFPN, Mlp,
                                                    SELayer)

try:
    from bevdepth.ops.voxel_pooling_inference import voxel_pooling_inference
    from bevdepth.ops.voxel_pooling_train import voxel_pooling_train
except ImportError:
    print('Import VoxelPooling fail.')

__all__ = ['BEVStereoLSSFPN']


class ConvBnReLU3D(nn.Module):
    """Implements of 3d convolution + batch normalization + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
    ) -> None:
        """initialization method for convolution3D +
            batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=pad,
                              dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class DepthNet(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 d_bound,
                 num_ranges=4):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_feat_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
        )
        self.mu_sigma_range_net = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            nn.ConvTranspose2d(mid_channels,
                               mid_channels,
                               3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels,
                               mid_channels,
                               3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      num_ranges * 3,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        self.mono_depth_net = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        self.d_bound = d_bound
        self.num_ranges = num_ranges

    # @autocast(False)
    def forward(self, x, mats_dict, scale_depth_factor=1000.0):
        B, _, H, W = x.shape
        intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict['ida_mats'][:, 0:1, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
        bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4,
                                        4).repeat(1, 1, num_cams, 1, 1)
        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 3],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, 1, num_cams, -1),
            ],
            -1,
        )
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth_feat = self.depth_se(x, depth_se)
        depth_feat = self.depth_feat_conv(depth_feat)
        mono_depth = self.mono_depth_net(depth_feat)
        mu_sigma_score = self.mu_sigma_range_net(depth_feat)
        d_coords = torch.arange(*self.d_bound,
                                dtype=torch.float).reshape(1, -1, 1, 1).cuda()
        d_coords = d_coords.repeat(B, 1, H, W)
        mu = mu_sigma_score[:, 0:self.num_ranges, ...]
        sigma = mu_sigma_score[:, self.num_ranges:2 * self.num_ranges, ...]
        range_score = mu_sigma_score[:,
                                     2 * self.num_ranges:3 * self.num_ranges,
                                     ...]
        sigma = F.elu(sigma) + 1.0 + 1e-10
        return x, context, mu, sigma, range_score, mono_depth


class BEVStereoLSSFPN(BaseLSSFPN):

    def __init__(self,
                 x_bound,
                 y_bound,
                 z_bound,
                 d_bound,
                 final_dim,
                 downsample_factor,
                 output_channels,
                 img_backbone_conf,
                 img_neck_conf,
                 depth_net_conf,
                 use_da=False,
                 sampling_range=3,
                 num_samples=3,
                 stereo_downsample_factor=4,
                 em_iteration=3,
                 min_sigma=1,
                 num_groups=8,
                 num_ranges=4,
                 range_list=[[2, 8], [8, 16], [16, 28], [28, 58]],
                 k_list=None,
                 use_mask=True):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.
        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
            sampling_range (int): The base range of sampling candidates.
                Defaults to 3.
            num_samples (int): Number of samples. Defaults to 3.
            stereo_downsample_factor (int): Downsample factor from input image
                and stereo depth. Defaults to 4.
            em_iteration (int): Number of iterations for em. Defaults to 3.
            min_sigma (float): Minimal value for sigma. Defaults to 1.
            num_groups (int): Number of groups to keep after inner product.
                Defaults to 8.
            num_ranges (int): Number of split ranges. Defaults to 1.
            range_list (list): Start and end of every range, Defaults to None.
            k_list (list): Depth of all candidates inside the range.
                Defaults to None.
            use_mask (bool): Whether to use mask_net. Defaults to True.
        """
        self.num_ranges = num_ranges
        self.sampling_range = sampling_range
        self.num_samples = num_samples
        super(BEVStereoLSSFPN,
              self).__init__(x_bound, y_bound, z_bound, d_bound, final_dim,
                             downsample_factor, output_channels,
                             img_backbone_conf, img_neck_conf, depth_net_conf,
                             use_da)

        self.depth_channels, _, _, _ = self.frustum.shape
        self.use_mask = use_mask
        if k_list is None:
            self.register_buffer('k_list', torch.Tensor(self.depth_sampling()))
        else:
            self.register_buffer('k_list', torch.Tensor(k_list))
        self.stereo_downsample_factor = stereo_downsample_factor
        self.em_iteration = em_iteration
        self.register_buffer(
            'depth_values',
            torch.arange((self.d_bound[1] - self.d_bound[0]) / self.d_bound[2],
                         dtype=torch.float))
        self.num_groups = num_groups
        self.similarity_net = nn.Sequential(
            ConvBnReLU3D(in_channels=num_groups,
                         out_channels=16,
                         kernel_size=1,
                         stride=1,
                         pad=0),
            ConvBnReLU3D(in_channels=16,
                         out_channels=8,
                         kernel_size=1,
                         stride=1,
                         pad=0),
            nn.Conv3d(in_channels=8,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        if range_list is None:
            range_length = (d_bound[1] - d_bound[0]) / num_ranges
            self.range_list = [[
                d_bound[0] + range_length * i,
                d_bound[0] + range_length * (i + 1)
            ] for i in range(num_ranges)]
        else:
            assert len(range_list) == num_ranges
            self.range_list = range_list

        self.min_sigma = min_sigma
        self.depth_downsample_net = nn.Sequential(
            nn.Conv2d(self.depth_channels, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.depth_channels, 1, 1, 0),
        )
        self.context_downsample_net = nn.Identity()
        if self.use_mask:
            self.mask_net = nn.Sequential(
                nn.Conv2d(224, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                BasicBlock(64, 64),
                BasicBlock(64, 64),
                nn.Conv2d(64, 1, 1, 1, 0),
                nn.Sigmoid(),
            )

    def depth_sampling(self):
        """Generate sampling range of candidates.
        Returns:
            list[float]: List of all candidates.
        """
        P_total = erf(self.sampling_range /
                      np.sqrt(2))  # Probability covered by the sampling range
        idx_list = np.arange(0, self.num_samples + 1)
        p_list = (1 - P_total) / 2 + ((idx_list / self.num_samples) * P_total)
        k_list = norm.ppf(p_list)
        k_list = (k_list[1:] + k_list[:-1]) / 2
        return list(k_list)

    def _generate_cost_volume(
        self,
        sweep_index,
        stereo_feats_all_sweeps,
        mats_dict,
        depth_sample,
        depth_sample_frustum,
        sensor2sensor_mats,
    ):
        """Generate cost volume based on depth sample.
        Args:
            sweep_index (int): Index of sweep.
            stereo_feats_all_sweeps (list[Tensor]): Stereo feature
                of all sweeps.
            mats_dict (dict):
                sensor2ego_mats (Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats (Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats (Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats (Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat (Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            depth_sample (Tensor): Depth map of all candidates.
            depth_sample_frustum (Tensor): Pre-generated frustum.
            sensor2sensor_mats (Tensor): Transformation matrix from reference
                sensor to source sensor.
        Returns:
            Tensor: Depth score for all sweeps.
        """
        batch_size, num_channels, height, width = stereo_feats_all_sweeps[
            0].shape
        num_sweeps = len(stereo_feats_all_sweeps)
        depth_score_all_sweeps = list()
        for idx in range(num_sweeps):
            if idx == sweep_index:
                continue
            warped_stereo_fea = self.homo_warping(
                stereo_feats_all_sweeps[idx],
                mats_dict['intrin_mats'][:, sweep_index, ...],
                mats_dict['intrin_mats'][:, idx, ...],
                sensor2sensor_mats[idx],
                mats_dict['ida_mats'][:, sweep_index, ...],
                mats_dict['ida_mats'][:, idx, ...],
                depth_sample,
                depth_sample_frustum.type_as(stereo_feats_all_sweeps[idx]),
            )
            warped_stereo_fea = warped_stereo_fea.reshape(
                batch_size, self.num_groups, num_channels // self.num_groups,
                self.num_samples, height, width)
            ref_stereo_feat = stereo_feats_all_sweeps[sweep_index].reshape(
                batch_size, self.num_groups, num_channels // self.num_groups,
                height, width)
            feat_cost = torch.mean(
                (ref_stereo_feat.unsqueeze(3) * warped_stereo_fea), axis=2)
            depth_score = self.similarity_net(feat_cost).squeeze(1)
            depth_score_all_sweeps.append(depth_score)
        return torch.stack(depth_score_all_sweeps).mean(0)

    def homo_warping(
        self,
        stereo_feat,
        key_intrin_mats,
        sweep_intrin_mats,
        sensor2sensor_mats,
        key_ida_mats,
        sweep_ida_mats,
        depth_sample,
        frustum,
    ):
        """Used for mvs method to transfer sweep image feature to
            key image feature.
        Args:
            src_fea(Tensor): image features.
            key_intrin_mats(Tensor): Intrin matrix for key sensor.
            sweep_intrin_mats(Tensor): Intrin matrix for sweep sensor.
            sensor2sensor_mats(Tensor): Transformation matrix from key
                sensor to sweep sensor.
            key_ida_mats(Tensor): Ida matrix for key frame.
            sweep_ida_mats(Tensor): Ida matrix for sweep frame.
            depth_sample (Tensor): Depth map of all candidates.
            depth_sample_frustum (Tensor): Pre-generated frustum.
        """
        batch_size_with_num_cams, channels = stereo_feat.shape[
            0], stereo_feat.shape[1]
        height, width = stereo_feat.shape[2], stereo_feat.shape[3]
        with torch.no_grad():
            points = frustum
            points = points.reshape(points.shape[0], -1, points.shape[-1])
            points[..., 2] = 1
            # Undo ida for key frame.
            points = key_ida_mats.reshape(batch_size_with_num_cams, *
                                          key_ida_mats.shape[2:]).inverse(
                                          ).unsqueeze(1) @ points.unsqueeze(-1)
            # Convert points from pixel coord to key camera coord.
            points[..., :3, :] *= depth_sample.reshape(
                batch_size_with_num_cams, -1, 1, 1)
            num_depth = frustum.shape[1]
            points = (key_intrin_mats.reshape(
                batch_size_with_num_cams, *
                key_intrin_mats.shape[2:]).inverse().unsqueeze(1) @ points)
            points = (sensor2sensor_mats.reshape(
                batch_size_with_num_cams, *
                sensor2sensor_mats.shape[2:]).unsqueeze(1) @ points)
            # points in sweep sensor coord.
            points = (sweep_intrin_mats.reshape(
                batch_size_with_num_cams, *
                sweep_intrin_mats.shape[2:]).unsqueeze(1) @ points)
            # points in sweep pixel coord.
            points[..., :2, :] = points[..., :2, :] / points[
                ..., 2:3, :]  # [B, 2, Ndepth, H*W]

            points = (sweep_ida_mats.reshape(
                batch_size_with_num_cams, *
                sweep_ida_mats.shape[2:]).unsqueeze(1) @ points).squeeze(-1)
            neg_mask = points[..., 2] < 1e-3
            points[..., 0][neg_mask] = width * self.stereo_downsample_factor
            points[..., 1][neg_mask] = height * self.stereo_downsample_factor
            points[..., 2][neg_mask] = 1
            proj_x_normalized = points[..., 0] / (
                (width * self.stereo_downsample_factor - 1) / 2) - 1
            proj_y_normalized = points[..., 1] / (
                (height * self.stereo_downsample_factor - 1) / 2) - 1
            grid = torch.stack([proj_x_normalized, proj_y_normalized],
                               dim=2)  # [B, Ndepth, H*W, 2]

        warped_stereo_fea = F.grid_sample(
            stereo_feat,
            grid.view(batch_size_with_num_cams, num_depth * height, width, 2),
            mode='bilinear',
            padding_mode='zeros',
        )
        warped_stereo_fea = warped_stereo_fea.view(batch_size_with_num_cams,
                                                   channels, num_depth, height,
                                                   width)

        return warped_stereo_fea

    def _forward_stereo(
        self,
        sweep_index,
        stereo_feats_all_sweeps,
        mono_depth_all_sweeps,
        mats_dict,
        sensor2sensor_mats,
        mu_all_sweeps,
        sigma_all_sweeps,
        range_score_all_sweeps,
        depth_feat_all_sweeps,
    ):
        """Forward function to generate stereo depth.
        Args:
            sweep_index (int): Index of sweep.
            stereo_feats_all_sweeps (list[Tensor]): Stereo feature
                of all sweeps.
            mono_depth_all_sweeps (list[Tensor]):
            mats_dict (dict):
                sensor2ego_mats (Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats (Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats (Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats (Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat (Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            sensor2sensor_mats(Tensor): Transformation matrix from key
                sensor to sweep sensor.
            mu_all_sweeps (list[Tensor]): List of mu for all sweeps.
            sigma_all_sweeps (list[Tensor]): List of sigma for all sweeps.
            range_score_all_sweeps (list[Tensor]): List of all range score
                for all sweeps.
            depth_feat_all_sweeps (list[Tensor]): List of all depth feat for
                all sweeps.
        Returns:
            Tensor: stereo_depth
        """
        batch_size_with_cams, _, feat_height, feat_width = \
            stereo_feats_all_sweeps[0].shape
        device = stereo_feats_all_sweeps[0].device
        d_coords = torch.arange(*self.d_bound,
                                dtype=torch.float,
                                device=device).reshape(1, -1, 1, 1)
        d_coords = d_coords.repeat(batch_size_with_cams, 1, feat_height,
                                   feat_width)
        stereo_depth = stereo_feats_all_sweeps[0].new_zeros(
            batch_size_with_cams, self.depth_channels, feat_height, feat_width)
        mask_score = stereo_feats_all_sweeps[0].new_zeros(
            batch_size_with_cams,
            self.depth_channels,
            feat_height * self.stereo_downsample_factor //
            self.downsample_factor,
            feat_width * self.stereo_downsample_factor //
            self.downsample_factor,
        )
        score_all_ranges = list()
        range_score = range_score_all_sweeps[sweep_index].softmax(1)
        for range_idx in range(self.num_ranges):
            # Map mu to the corresponding interval.
            range_start = self.range_list[range_idx][0]
            mu_all_sweeps_single_range = [
                mu[:, range_idx:range_idx + 1, ...].sigmoid() *
                (self.range_list[range_idx][1] - self.range_list[range_idx][0])
                + range_start for mu in mu_all_sweeps
            ]
            sigma_all_sweeps_single_range = [
                sigma[:, range_idx:range_idx + 1, ...]
                for sigma in sigma_all_sweeps
            ]
            batch_size_with_cams, _, feat_height, feat_width =\
                stereo_feats_all_sweeps[0].shape
            mu = mu_all_sweeps_single_range[sweep_index]
            sigma = sigma_all_sweeps_single_range[sweep_index]
            for _ in range(self.em_iteration):
                depth_sample = torch.cat([mu + sigma * k for k in self.k_list],
                                         1)
                depth_sample_frustum = self.create_depth_sample_frustum(
                    depth_sample, self.stereo_downsample_factor)
                mu_score = self._generate_cost_volume(
                    sweep_index,
                    stereo_feats_all_sweeps,
                    mats_dict,
                    depth_sample,
                    depth_sample_frustum,
                    sensor2sensor_mats,
                )
                mu_score = mu_score.softmax(1)
                scale_factor = torch.clamp(
                    0.5 / (1e-4 + mu_score[:, self.num_samples //
                                           2:self.num_samples // 2 + 1, ...]),
                    min=0.1,
                    max=10)

                sigma = torch.clamp(sigma * scale_factor, min=0.1, max=10)
                mu = (depth_sample * mu_score).sum(1, keepdim=True)
                del depth_sample
                del depth_sample_frustum
            range_length = int(
                (self.range_list[range_idx][1] - self.range_list[range_idx][0])
                // self.d_bound[2])
            if self.use_mask:
                depth_sample = F.avg_pool2d(
                    mu,
                    self.downsample_factor // self.stereo_downsample_factor,
                    self.downsample_factor // self.stereo_downsample_factor,
                )
                depth_sample_frustum = self.create_depth_sample_frustum(
                    depth_sample, self.downsample_factor)
                mask = self._forward_mask(
                    sweep_index,
                    mono_depth_all_sweeps,
                    mats_dict,
                    depth_sample,
                    depth_sample_frustum,
                    sensor2sensor_mats,
                )
                mask_score[:,
                           int((range_start - self.d_bound[0]) //
                               self.d_bound[2]):range_length +
                           int((range_start - self.d_bound[0]) //
                               self.d_bound[2]), ..., ] += mask
                del depth_sample
                del depth_sample_frustum
            sigma = torch.clamp(sigma, self.min_sigma)
            mu_repeated = mu.repeat(1, range_length, 1, 1)
            eps = 1e-6
            depth_score_single_range = (-1 / 2 * (
                (d_coords[:,
                          int((range_start - self.d_bound[0]) //
                              self.d_bound[2]):range_length + int(
                                  (range_start - self.d_bound[0]) //
                                  self.d_bound[2]), ..., ] - mu_repeated) /
                torch.sqrt(sigma))**2)
            depth_score_single_range = depth_score_single_range.exp()
            score_all_ranges.append(mu_score.sum(1).unsqueeze(1))
            depth_score_single_range = depth_score_single_range / (
                sigma * math.sqrt(2 * math.pi) + eps)
            stereo_depth[:,
                         int((range_start - self.d_bound[0]) //
                             self.d_bound[2]):range_length +
                         int((range_start - self.d_bound[0]) //
                             self.d_bound[2]), ..., ] = (
                                 depth_score_single_range *
                                 range_score[:, range_idx:range_idx + 1, ...])
            del depth_score_single_range
            del mu_repeated
        if self.use_mask:
            return stereo_depth, mask_score
        else:
            return stereo_depth

    def create_depth_sample_frustum(self, depth_sample, downsample_factor=16):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // downsample_factor, ogfW // downsample_factor
        batch_size, num_depth, _, _ = depth_sample.shape
        x_coords = (torch.linspace(0,
                                   ogfW - 1,
                                   fW,
                                   dtype=torch.float,
                                   device=depth_sample.device).view(
                                       1, 1, 1,
                                       fW).expand(batch_size, num_depth, fH,
                                                  fW))
        y_coords = (torch.linspace(0,
                                   ogfH - 1,
                                   fH,
                                   dtype=torch.float,
                                   device=depth_sample.device).view(
                                       1, 1, fH,
                                       1).expand(batch_size, num_depth, fH,
                                                 fW))
        paddings = torch.ones_like(depth_sample)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, depth_sample, paddings), -1)
        return frustum

    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
            self.d_bound,
            self.num_ranges,
        )

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams,
                                   num_channels, imH, imW)
        backbone_feats = self.img_backbone(imgs)
        img_feats = self.img_neck(backbone_feats)[0]
        img_feats_reshape = img_feats.reshape(batch_size, num_sweeps, num_cams,
                                              img_feats.shape[1],
                                              img_feats.shape[2],
                                              img_feats.shape[3])
        return img_feats_reshape, backbone_feats[0].detach()

    def _forward_mask(
        self,
        sweep_index,
        mono_depth_all_sweeps,
        mats_dict,
        depth_sample,
        depth_sample_frustum,
        sensor2sensor_mats,
    ):
        """Forward function to generate mask.
        Args:
            sweep_index (int): Index of sweep.
            mono_depth_all_sweeps (list[Tensor]): List of mono_depth for
                all sweeps.
            mats_dict (dict):
                sensor2ego_mats (Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats (Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats (Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats (Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat (Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            depth_sample (Tensor): Depth map of all candidates.
            depth_sample_frustum (Tensor): Pre-generated frustum.
            sensor2sensor_mats (Tensor): Transformation matrix from reference
                sensor to source sensor.
        Returns:
            Tensor: Generated mask.
        """
        num_sweeps = len(mono_depth_all_sweeps)
        mask_all_sweeps = list()
        for idx in range(num_sweeps):
            if idx == sweep_index:
                continue
            warped_mono_depth = self.homo_warping(
                mono_depth_all_sweeps[idx],
                mats_dict['intrin_mats'][:, sweep_index, ...],
                mats_dict['intrin_mats'][:, idx, ...],
                sensor2sensor_mats[idx],
                mats_dict['ida_mats'][:, sweep_index, ...],
                mats_dict['ida_mats'][:, idx, ...],
                depth_sample,
                depth_sample_frustum.type_as(mono_depth_all_sweeps[idx]),
            )
            mask = self.mask_net(
                torch.cat([
                    mono_depth_all_sweeps[sweep_index].detach(),
                    warped_mono_depth.mean(2).detach()
                ], 1))
            mask_all_sweeps.append(mask)
        return torch.stack(mask_all_sweeps).mean(0)

    def _forward_single_sweep(self,
                              sweep_index,
                              context,
                              mats_dict,
                              depth_score,
                              is_return_depth=False):
        """Forward function for single sweep.
        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.
        Returns:
            Tensor: BEV feature map.
        """
        batch_size, num_cams = context.shape[0], context.shape[1]
        context = context.reshape(batch_size * num_cams, *context.shape[2:])
        depth = depth_score
        geom_xyz = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
        )
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        if self.training or self.use_da:
            img_feat_with_depth = depth.unsqueeze(1) * context.unsqueeze(2)

            img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

            img_feat_with_depth = img_feat_with_depth.reshape(
                batch_size,
                num_cams,
                img_feat_with_depth.shape[1],
                img_feat_with_depth.shape[2],
                img_feat_with_depth.shape[3],
                img_feat_with_depth.shape[4],
            )
            img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)

            feature_map = voxel_pooling_train(geom_xyz,
                                              img_feat_with_depth.contiguous(),
                                              self.voxel_num.cuda())
        else:
            feature_map = voxel_pooling_inference(geom_xyz, depth.contiguous(),
                                                  context.contiguous(),
                                                  self.voxel_num.cuda())
        if is_return_depth:
            return feature_map.contiguous(), depth
        return feature_map.contiguous()

    def forward(self,
                sweep_imgs,
                mats_dict,
                timestamps=None,
                is_return_depth=False):
        """Forward function.
        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).
        Return:
            Tensor: bev feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        context_all_sweeps = list()
        depth_feat_all_sweeps = list()
        img_feats_all_sweeps = list()
        stereo_feats_all_sweeps = list()
        mu_all_sweeps = list()
        sigma_all_sweeps = list()
        mono_depth_all_sweeps = list()
        range_score_all_sweeps = list()
        for sweep_index in range(0, num_sweeps):
            if sweep_index > 0:
                with torch.no_grad():
                    img_feats, stereo_feats = self.get_cam_feats(
                        sweep_imgs[:, sweep_index:sweep_index + 1, ...])
                    img_feats_all_sweeps.append(
                        img_feats.view(batch_size * num_cams,
                                       *img_feats.shape[3:]))
                    stereo_feats_all_sweeps.append(stereo_feats)
                    depth_feat, context, mu, sigma, range_score, mono_depth =\
                        self.depth_net(img_feats.view(batch_size * num_cams,
                                       *img_feats.shape[3:]), mats_dict)
                    context_all_sweeps.append(
                        self.context_downsample_net(
                            context.reshape(batch_size * num_cams,
                                            *context.shape[1:])))
                    depth_feat_all_sweeps.append(depth_feat)
            else:
                img_feats, stereo_feats = self.get_cam_feats(
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...])
                img_feats_all_sweeps.append(
                    img_feats.view(batch_size * num_cams,
                                   *img_feats.shape[3:]))
                stereo_feats_all_sweeps.append(stereo_feats)
                depth_feat, context, mu, sigma, range_score, mono_depth =\
                    self.depth_net(img_feats.view(batch_size * num_cams,
                                   *img_feats.shape[3:]), mats_dict)
                depth_feat_all_sweeps.append(depth_feat)
                context_all_sweeps.append(
                    self.context_downsample_net(
                        context.reshape(batch_size * num_cams,
                                        *context.shape[1:])))
            mu_all_sweeps.append(mu)
            sigma_all_sweeps.append(sigma)
            mono_depth_all_sweeps.append(mono_depth)
            range_score_all_sweeps.append(range_score)
        depth_score_all_sweeps = list()
        final_depth = None
        for ref_idx in range(num_sweeps):
            sensor2sensor_mats = list()
            for src_idx in range(num_sweeps):
                ref2keysensor_mats = mats_dict[
                    'sensor2sensor_mats'][:, ref_idx, ...].inverse()
                key2srcsensor_mats = mats_dict['sensor2sensor_mats'][:,
                                                                     src_idx,
                                                                     ...]
                ref2srcsensor_mats = key2srcsensor_mats @ ref2keysensor_mats
                sensor2sensor_mats.append(ref2srcsensor_mats)
            if ref_idx == 0:
                # last iteration on stage 1 does not have propagation
                # (photometric consistency filtering)
                if self.use_mask:
                    stereo_depth, mask = self._forward_stereo(
                        ref_idx,
                        stereo_feats_all_sweeps,
                        mono_depth_all_sweeps,
                        mats_dict,
                        sensor2sensor_mats,
                        mu_all_sweeps,
                        sigma_all_sweeps,
                        range_score_all_sweeps,
                        depth_feat_all_sweeps,
                    )
                else:
                    stereo_depth = self._forward_stereo(
                        ref_idx,
                        stereo_feats_all_sweeps,
                        mono_depth_all_sweeps,
                        mats_dict,
                        sensor2sensor_mats,
                        mu_all_sweeps,
                        sigma_all_sweeps,
                        range_score_all_sweeps,
                        depth_feat_all_sweeps,
                    )
            else:
                with torch.no_grad():
                    # last iteration on stage 1 does not have
                    # propagation (photometric consistency filtering)
                    if self.use_mask:
                        stereo_depth, mask = self._forward_stereo(
                            ref_idx,
                            stereo_feats_all_sweeps,
                            mono_depth_all_sweeps,
                            mats_dict,
                            sensor2sensor_mats,
                            mu_all_sweeps,
                            sigma_all_sweeps,
                            range_score_all_sweeps,
                            depth_feat_all_sweeps,
                        )
                    else:
                        stereo_depth = self._forward_stereo(
                            ref_idx,
                            stereo_feats_all_sweeps,
                            mono_depth_all_sweeps,
                            mats_dict,
                            sensor2sensor_mats,
                            mu_all_sweeps,
                            sigma_all_sweeps,
                            range_score_all_sweeps,
                            depth_feat_all_sweeps,
                        )
            if self.use_mask:
                depth_score = (
                    mono_depth_all_sweeps[ref_idx] +
                    self.depth_downsample_net(stereo_depth) * mask).softmax(
                        1, dtype=stereo_depth.dtype)
            else:
                depth_score = (
                    mono_depth_all_sweeps[ref_idx] +
                    self.depth_downsample_net(stereo_depth)).softmax(
                        1, dtype=stereo_depth.dtype)
            depth_score_all_sweeps.append(depth_score)
            if ref_idx == 0:
                # final_depth has to be fp32, otherwise the
                # depth loss will colapse during the traing process.
                final_depth = (
                    mono_depth_all_sweeps[ref_idx] +
                    self.depth_downsample_net(stereo_depth)).softmax(1)
        key_frame_res = self._forward_single_sweep(
            0,
            context_all_sweeps[0].reshape(batch_size, num_cams,
                                          *context_all_sweeps[0].shape[1:]),
            mats_dict,
            depth_score_all_sweeps[0],
            is_return_depth=is_return_depth,
        )
        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[
            0] if is_return_depth else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    context_all_sweeps[sweep_index].reshape(
                        batch_size, num_cams,
                        *context_all_sweeps[sweep_index].shape[1:]),
                    mats_dict,
                    depth_score_all_sweeps[sweep_index],
                    is_return_depth=False,
                )
                ret_feature_list.append(feature_map)

        if is_return_depth:
            return torch.cat(ret_feature_list, 1), final_depth
        else:
            return torch.cat(ret_feature_list, 1)
