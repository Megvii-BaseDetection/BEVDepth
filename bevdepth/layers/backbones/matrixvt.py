# Copyright (c) Megvii Inc. All rights reserved.
import torch
from torch import nn
from torch.cuda.amp import autocast

from layers.backbones.base_lss_fpn import DepthNet

class HoriConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, cat_dim=0):
        """HoriConv that reduce the image feature in height dimension and refine it.

        Args:
            in_channels (int): in_channels
            mid_channels (int): mid_channels
            out_channels (int): output channels
            cat_dim (int, optional): channels of position embedding. Defaults to 0.
        """        
        super().__init__()

        self.merger = nn.Sequential(
            nn.Conv2d(in_channels + cat_dim, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
        )

        self.reduce_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x, pe=None):
        # [N,C,H,W]
        if pe is not None:
            x = self.merger(torch.cat([x, pe], 1))
        else:
            x = self.merger(x)
        x = x.max(2)[0]
        x = self.reduce_conv(x)
        x = self.conv1(x) + x
        x = self.conv2(x) + x
        x = self.out_conv(x)
        return x


class DepthReducer(nn.Module):
    def __init__(self, img_channels, mid_channels):
        """Module that compresses the predicted categorical depth in height dimension

        Args:
            img_channels (int): in_channels
            mid_channels (int): mid_channels
        """        
        super().__init__()
        self.vertical_weighter = nn.Sequential(
            nn.Conv2d(img_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, stride=1, padding=1),
        )

    @autocast(False)
    def forward(self, feat, depth):
        vert_weight = self.vertical_weighter(feat).softmax(2)  # [N,1,H,W]
        depth = (depth * vert_weight).sum(2)
        return depth

# NOTE Modified Lift-Splat
class MatrixVT(nn.Module):
    def __init__(
        self,
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
    ):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels

        self.register_buffer(
            "voxel_size", torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]])
        )
        self.register_buffer(
            "voxel_coord",
            torch.Tensor(
                [row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]]
            ),
        )
        self.register_buffer(
            "voxel_num",
            torch.LongTensor(
                [(row[1] - row[0]) / row[2] for row in [x_bound, y_bound, z_bound]]
            ),
        )
        self.register_buffer("frustum", self.create_frustum())
        self.register_buffer("bev_anchors", self.create_bev_anchors(x_bound, y_bound))
        self.depth_channels, _, _, _ = self.frustum.shape

        from mmdet3d.models import build_neck
        from mmdet.models import build_backbone

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)

        self.img_neck.init_weights()
        self.img_backbone.init_weights()

        self.depth_net = DepthNet(
            depth_net_conf["in_channels"],
            depth_net_conf["mid_channels"],
            self.output_channels,
            self.depth_channels,
        )

        self.horiconv = HoriConv(self.output_channels, 512, self.output_channels)
        self.depth_reducer = DepthReducer(self.output_channels, self.output_channels)
        self.static_mat = None

    def create_bev_anchors(self, x_bound, y_bound, ds_rate=1):
        """Create anchors in BEV space

        Args:
            x_bound (list): xbound in meters [start, end, step]
            y_bound (list): ybound in meters [start, end, step]
            ds_rate (iint, optional): downsample rate. Defaults to 1.

        Returns:
            anchors: anchors in [W, H, 2]
        """        
        x_coords = (
            (
                torch.linspace(
                    x_bound[0],
                    x_bound[1] - x_bound[2] * ds_rate,
                    self.voxel_num[0] // ds_rate,
                    dtype=torch.float,
                )
                + x_bound[2] * ds_rate / 2
            )
            .view(self.voxel_num[0] // ds_rate, 1)
            .expand(self.voxel_num[0] // ds_rate, self.voxel_num[1] // ds_rate)
        )
        y_coords = (
            (
                torch.linspace(
                    y_bound[0],
                    y_bound[1] - y_bound[2] * ds_rate,
                    self.voxel_num[1] // ds_rate,
                    dtype=torch.float,
                )
                + y_bound[2] * ds_rate / 2
            )
            .view(1, self.voxel_num[1] // ds_rate)
            .expand(self.voxel_num[0] // ds_rate, self.voxel_num[1] // ds_rate)
        )

        anchors = torch.stack([x_coords, y_coords]).permute(1, 2, 0)
        return anchors

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = (
            torch.arange(*self.d_bound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = d_coords.shape
        x_coords = (
            torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        y_coords = (
            torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))
        # cam_to_ego
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:],
            ),
            5,
        )

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points)
        if bda_mat is not None:
            bda_mat = (
                bda_mat.unsqueeze(1)
                .repeat(1, num_cams, 1, 1)
                .view(batch_size, num_cams, 1, 1, 1, 4, 4)
            )
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    def get_proj_mat(self, mats_dict=None):
        """Create the Ring Matrix and Ray Matrix

        Args:
            mats_dict (dict, optional): dictionary that contains intrin- and extrin- parameters.
            Defaults to None.

        Returns:
            tuple: Ring Matrix in [B, D, L, L] and Ray Matrix in [B, W, L, L]
        """        
        if self.static_mat is not None:
            return self.static_mat

        bev_size = int(self.voxel_num[0])  # only consider square BEV
        geom_sep = self.get_geometry(
            mats_dict["sensor2ego_mats"][:, 0, ...],
            mats_dict["intrin_mats"][:, 0, ...],
            mats_dict["ida_mats"][:, 0, ...],
            mats_dict.get("bda_mat", None),
        )
        geom_sep = (
            geom_sep - (self.voxel_coord - self.voxel_size / 2.0)
        ) / self.voxel_size
        geom_sep = geom_sep.mean(3).permute(0, 1, 3, 2, 4).contiguous()  # B,Ncam,W,D,2
        B, Nc, W, D, _ = geom_sep.shape
        geom_sep = geom_sep.long().view(B, Nc * W, D, -1)[..., :2]

        invalid1 = torch.logical_or((geom_sep < 0)[..., 0], (geom_sep < 0)[..., 1])
        invalid2 = torch.logical_or(
            (geom_sep > (bev_size - 1))[..., 0], (geom_sep > (bev_size - 1))[..., 1]
        )
        geom_sep[(invalid1 | invalid2)] = int(bev_size / 2)
        geom_idx = geom_sep[..., 1] * bev_size + geom_sep[..., 0]

        geom_uni = self.bev_anchors[None].repeat([B, 1, 1, 1])  # B,128,128,2
        B, L, L, _ = geom_uni.shape

        circle_map = geom_uni.new_zeros((B, D, L * L))

        ray_map = geom_uni.new_zeros((B, Nc * W, L * L))
        for b in range(B):
            for dir in range(Nc * W):
                ray_map[b, dir, geom_idx[b, dir]] += 1
            for d in range(D):
                circle_map[b, d, geom_idx[b, :, d]] += 1
        null_point = int((bev_size / 2) * (bev_size + 1))
        circle_map[..., null_point] = 0
        ray_map[..., null_point] = 0
        circle_map = circle_map.view(B, D, L * L)
        ray_map = ray_map.view(B, -1, L * L)
        circle_map /= circle_map.max(1)[0].clip(min=1)[:, None]
        ray_map /= ray_map.max(1)[0].clip(min=1)[:, None]

        return circle_map, ray_map

    @autocast(False)
    def reduce_and_project(self, feature, depth, mats_dict):
        """reduce the feature and depth in height dimension and make BEV feature

        Args:
            feature (Tensor): image feature in [B, C, H, W]
            depth (Tensor): Depth Prediction in [B, D, H, W]
            mats_dict (dict): dictionary that contains intrin- annd extrin- parameters

        Returns:
            Tensor: BEV feature in B, C, L, L
        """        
        # [N,112,H,W], [N,256,H,W]
        depth = self.depth_reducer(feature, depth)

        B = mats_dict["intrin_mats"].shape[0]

        # N, C, H, W = feature.shape
        # feature=feature.reshape(N,C*H,W)
        feature = self.horiconv(feature)
        # feature = feature.max(2)[0]
        # [N.112,W], [N,C,W]
        depth = depth.permute(0, 2, 1).reshape(B, -1, self.depth_channels)
        feature = feature.permute(0, 2, 1).reshape(B, -1, self.output_channels)
        circle_map, ray_map = self.get_proj_mat(mats_dict)

        proj_mat = depth.matmul(circle_map)
        proj_mat = (proj_mat * ray_map).permute(0, 2, 1)
        img_feat_with_depth = proj_mat.matmul(feature)
        img_feat_with_depth = img_feat_with_depth.permute(0, 2, 1).reshape(
            B, -1, *self.voxel_num[:2]
        )

        return img_feat_with_depth

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(
            batch_size * num_sweeps * num_cams, num_channels, imH, imW
        )
        img_feats = self.img_neck(self.img_backbone(imgs))[0]
        img_feats = img_feats.reshape(
            batch_size,
            num_sweeps,
            num_cams,
            img_feats.shape[1],
            img_feats.shape[2],
            img_feats.shape[3],
        )
        return img_feats

    def _forward_single_sweep(
        self, sweep_index, sweep_imgs, mats_dict, is_return_depth=False
    ):
        (
            batch_size,
            num_sweeps,
            num_cams,
            num_channels,
            img_height,
            img_width,
        ) = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        depth_feature = self.depth_net(
            source_features.reshape(
                batch_size * num_cams,
                source_features.shape[2],
                source_features.shape[3],
                source_features.shape[4],
            ),
            mats_dict,
        )
        with autocast(enabled=False):
            feature = depth_feature[
                :, self.depth_channels : (self.depth_channels + self.output_channels)
            ].float()
            depth = depth_feature[:, : self.depth_channels].float().softmax(1)

            img_feat_with_depth = self.reduce_and_project(
                feature, depth, mats_dict
            )  # [b*n, c, d, w]

            if is_return_depth:
                return img_feat_with_depth.contiguous(), depth
            return img_feat_with_depth.contiguous()

    def forward(self, sweep_imgs, mats_dict, timestamps=None, is_return_depth=False):
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
        (
            batch_size,
            num_sweeps,
            num_cams,
            num_channels,
            img_height,
            img_width,
        ) = sweep_imgs.shape

        key_frame_res = self._forward_single_sweep(
            0, sweep_imgs[:, 0:1, ...], mats_dict, is_return_depth=is_return_depth
        )
        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[0] if is_return_depth else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index : sweep_index + 1, ...],
                    mats_dict,
                    is_return_depth=False,
                )
                ret_feature_list.append(feature_map)

        if is_return_depth:
            return torch.cat(ret_feature_list, 1), key_frame_res[1]
        else:
            return torch.cat(ret_feature_list, 1)


if __name__ == "__main__":
    backbone_conf = {
        "x_bound": [-51.2, 51.2, 0.8],  # BEV grids bounds and size (m)
        "y_bound": [-51.2, 51.2, 0.8],  # BEV grids bounds and size (m)
        "z_bound": [-5, 3, 8],  # BEV grids bounds and size (m)
        "d_bound": [2.0, 58.0, 0.5],  # Categorical Depth bounds and devision (m)
        "final_dim": (256, 704),  # img size for model input (pix)
        "output_channels": 80,  # BEV feature channels
        "downsample_factor": 16,  # ds factor of the feature to be projected to BEV (e.g. 256x704 -> 16x44)
        "img_backbone_conf": dict(
            type="ResNet",
            depth=50,
            frozen_stages=0,
            out_indices=[0, 1, 2, 3],
            norm_eval=False,
            init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
        ),
        "img_neck_conf": dict(
            type="SECONDFPN",
            in_channels=[256, 512, 1024, 2048],
            upsample_strides=[0.25, 0.5, 1, 2],
            out_channels=[128, 128, 128, 128],
        ),
        "depth_net_conf": dict(in_channels=512, mid_channels=512),
    }

    model = MatrixVT(**backbone_conf)
    # for inference and deployment where intrin & extrin mats are static
    # model.static_mat = model.get_proj_mat(mats_dict)

    bev_feature, depth = model(
        torch.rand((2, 1, 6, 3, 256, 704)),
        {
            "sensor2ego_mats": torch.rand((2, 1, 6, 4, 4)),
            "intrin_mats": torch.rand((2, 1, 6, 4, 4)),
            "ida_mats": torch.rand((2, 1, 6, 4, 4)),
            "sensor2sensor_mats": torch.rand((2, 1, 6, 4, 4)),
            "bda_mat": torch.rand((2, 4, 4)),
        },
    )

    print(bev_feature.shape, depth.shape)
