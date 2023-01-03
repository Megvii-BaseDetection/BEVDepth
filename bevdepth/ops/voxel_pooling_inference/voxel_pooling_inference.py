# Copyright (c) Megvii Inc. All rights reserved.
import torch
from torch.autograd import Function

from . import voxel_pooling_inference_ext


class VoxelPoolingInference(Function):

    @staticmethod
    def forward(ctx, geom_xyz: torch.Tensor, depth_features: torch.Tensor,
                context_features: torch.Tensor,
                voxel_num: torch.Tensor) -> torch.Tensor:
        """Forward function for `voxel pooling.

        Args:
            geom_xyz (Tensor): xyz coord for each voxel with the shape
                of [B, N, 3].
            input_features (Tensor): feature for each voxel with the
                shape of [B, N, C].
            voxel_num (Tensor): Number of voxels for each dim with the
                shape of [3].

        Returns:
            Tensor: (B, C, H, W) bev feature map.
        """
        assert geom_xyz.is_contiguous()
        assert depth_features.is_contiguous()
        assert context_features.is_contiguous()
        # no gradient for input_features and geom_feats
        ctx.mark_non_differentiable(geom_xyz)
        batch_size = geom_xyz.shape[0]
        num_cams = geom_xyz.shape[1]
        num_depth = geom_xyz.shape[2]
        num_height = geom_xyz.shape[3]
        num_width = geom_xyz.shape[4]
        num_channels = context_features.shape[1]
        output_features = depth_features.new_zeros(
            (batch_size, voxel_num[1], voxel_num[0], num_channels))
        voxel_pooling_inference_ext.voxel_pooling_inference_forward_wrapper(
            batch_size,
            num_cams,
            num_depth,
            num_height,
            num_width,
            num_channels,
            voxel_num[0],
            voxel_num[1],
            voxel_num[2],
            geom_xyz,
            depth_features,
            context_features,
            output_features,
        )
        return output_features.permute(0, 3, 1, 2)


voxel_pooling_inference = VoxelPoolingInference.apply
