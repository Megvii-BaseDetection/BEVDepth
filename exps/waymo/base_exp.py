# Copyright (c) Megvii Inc. All rights reserved.
from functools import partial

import mmcv
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from pytorch_lightning.core import LightningModule
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR

from datasets.waymo_det_dataset import WaymoDetDataset, collate_fn
from evaluators.waymo_det_evaluator import DetWaymoEvaluator
from models.base_bev_depth import BaseBEVDepth
from utils.torch_dist import all_gather_object, synchronize

LIDAR_KEYS = ['TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR']
H = 1280
W = 1920
final_dim = (256, 704)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)

backbone_conf = {
    'x_bound': [-64, 64, 1.],
    'y_bound': [-64, 64, 1.],
    'z_bound': [-2, 4, 6],
    'd_bound': [2.0, 66, 0.5],
    'final_dim':
    final_dim,
    'output_channels':
    80,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
        type='ResNet',
        depth=50,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'depth_net_conf':
    dict(in_channels=512, mid_channels=512)
}
ida_aug_conf = {
    'resize_lim': (0.2, 0.4),
    'final_dim': final_dim,
    'rot_lim': (-5.4, 5.4),
    'H': H,
    'W': W,
    'rand_flip': True,
    'bot_pct_lim': (0.0, 0.0),
    'cams': ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT'],
    'Ncams': 5,
}

bda_aug_conf = {
    'rot_lim': (-22.5, 22.5),
    'scale_lim': (0.95, 1.05),
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5
}

bev_backbone = dict(
    type='ResNet',
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck = dict(type='SECONDFPN',
                in_channels=[80, 160, 320, 640],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64])

CLASSES = ['Vehicle', 'Pedestrian', 'Cyclist']

TASKS = [
    dict(num_class=1, class_names=['Vehicle']),
    dict(num_class=2, class_names=['Pedestrian', 'Cyclist'])
]

common_heads = dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2))

bbox_coder = dict(
    type='CenterPointBBoxCoder',
    post_center_range=[-70, -70, -10.0, 70, 70, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.25, 0.25, 6.0],
    pc_range=[-64, -64, -2, 64, 64, 4.0],
    code_size=7,
)

train_cfg = dict(
    point_cloud_range=[-64, -64, -2, 64, 64, 4.0],
    grid_size=[512, 512, 1],
    voxel_size=[0.25, 0.25, 6.0],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
)

test_cfg = dict(
    post_center_limit_range=[-70, -70, -10.0, 70, 70, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 0.25],
    score_threshold=0.3,
    out_size_factor=4,
    voxel_size=[0.25, 0.25, 6],
    nms_type='circle',
    pre_max_size=1000,
    post_max_size=200,
    nms_thr=0.2,
)

head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 256,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
}


class BEVDepthLightningModel(LightningModule):
    MODEL_NAMES = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith('__')
                         and callable(models.__dict__[name]))

    def __init__(self,
                 gpus: int = 1,
                 data_root='data/waymo/v1.4',
                 eval_interval=1,
                 batch_size_per_device=8,
                 class_names=CLASSES,
                 backbone_conf=backbone_conf,
                 head_conf=head_conf,
                 ida_aug_conf=ida_aug_conf,
                 bda_aug_conf=bda_aug_conf,
                 default_root_dir='./outputs/',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.gpus = gpus
        self.eval_interval = eval_interval
        self.batch_size_per_device = batch_size_per_device
        self.data_root = data_root
        self.basic_lr_per_img = 2e-4 / 64
        self.class_names = class_names
        self.backbone_conf = backbone_conf
        self.head_conf = head_conf
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        mmcv.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir
        self.evaluator = DetWaymoEvaluator(class_names=self.class_names)
        self.model = BaseBEVDepth(self.backbone_conf,
                                  self.head_conf,
                                  is_train_depth=True)
        self.img_conf = img_conf
        self.data_use_cbgs = False
        self.num_sweeps = 1
        self.sweep_idxes = list()
        self.key_idxes = list()
        self.data_return_depth = True
        self.downsample_factor = self.backbone_conf['downsample_factor']
        self.dbound = self.backbone_conf['d_bound']
        self.depth_channels = int(
            (self.dbound[1] - self.dbound[0]) / self.dbound[2])
        self.use_fusion = False
        self.train_info_paths = 'data/waymo/v1.4/waymo_infos_training.pkl'
        self.val_info_paths = 'data/waymo/v1.4/waymo_infos_validation.pkl'
        # self.predict_info_paths = 'data/waymo/v1.4/waymo_infos_train.pkl'
        self.lidar_keys = LIDAR_KEYS

    def forward(self, sweep_imgs, mats):
        return self.model(sweep_imgs, mats)

    def training_step(self, batch):
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, _, lidar_depth) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
        preds, depth_preds = self(sweep_imgs, mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            targets = self.model.module.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.module.loss(targets, preds)
        else:
            targets = self.model.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.loss(targets, preds)
        if len(lidar_depth.shape) == 5:
            # only key-frame will calculate depth loss
            lidar_depth = lidar_depth[:, 0, ...]
        depth_loss = self.get_depth_loss(lidar_depth.cuda(), depth_preds)
        self.log('detection_loss', detection_loss)
        self.log('depth_loss', depth_loss)
        return detection_loss + depth_loss

    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return 3.0 * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]

        return gt_depths.float()

    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, img_metas, gt_boxes, _, gt_classes3d) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
        preds = self.model(sweep_imgs, mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
            results[i].append(gt_boxes[i].detach().cpu().numpy())
            results[i].append(gt_classes3d[i])
        return results

    def test_epoch_end(self, test_step_outputs):
        prediction_infos = list()
        gt_infos = list()
        for test_step_output in test_step_outputs:
            for i in range(len(test_step_output)):
                pred_bboxes = test_step_output[i][0]
                pred_scores = test_step_output[i][1]
                pred_classes = [
                    self.class_names[pred_id]
                    for pred_id in test_step_output[i][2]
                ]
                prediction_infos.append(
                    [pred_bboxes, pred_scores, pred_classes])
                gt_infos.append(test_step_output[i][3:])
        synchronize()
        # TODO: Change another way.
        dataset_length = len(self.val_dataloader().dataset)
        prediction_infos = sum(
            map(list, zip(*all_gather_object(prediction_infos))),
            [])[:dataset_length]
        gt_infos = sum(map(list, zip(*all_gather_object(gt_infos))),
                       [])[:dataset_length]
        results = self.evaluator.evaluate(prediction_infos, gt_infos)
        print(results)

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, [19, 23])
        return [[optimizer], [scheduler]]

    def train_dataloader(self):
        train_dataset = WaymoDetDataset(ida_aug_conf=self.ida_aug_conf,
                                        bda_aug_conf=self.bda_aug_conf,
                                        classes=self.class_names,
                                        data_root=self.data_root,
                                        info_paths=self.train_info_paths,
                                        is_train=True,
                                        use_cbgs=self.data_use_cbgs,
                                        img_conf=self.img_conf,
                                        num_sweeps=self.num_sweeps,
                                        sweep_idxes=self.sweep_idxes,
                                        key_idxes=self.key_idxes,
                                        return_depth=self.data_return_depth,
                                        use_fusion=self.use_fusion,
                                        lidar_keys=self.lidar_keys)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth
                               or self.use_fusion),
            sampler=None,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = WaymoDetDataset(ida_aug_conf=self.ida_aug_conf,
                                      bda_aug_conf=self.bda_aug_conf,
                                      classes=self.class_names,
                                      data_root=self.data_root,
                                      info_paths=self.val_info_paths,
                                      is_train=False,
                                      img_conf=self.img_conf,
                                      num_sweeps=self.num_sweeps,
                                      sweep_idxes=self.sweep_idxes,
                                      key_idxes=self.key_idxes,
                                      return_depth=self.use_fusion,
                                      use_fusion=self.use_fusion,
                                      lidar_keys=self.lidar_keys)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=self.use_fusion),
            num_workers=4,
            sampler=None,
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        predict_dataset = WaymoDetDataset(ida_aug_conf=self.ida_aug_conf,
                                          bda_aug_conf=self.bda_aug_conf,
                                          classes=self.class_names,
                                          data_root=self.data_root,
                                          info_paths=self.predict_info_paths,
                                          is_train=False,
                                          img_conf=self.img_conf,
                                          num_sweeps=self.num_sweeps,
                                          sweep_idxes=self.sweep_idxes,
                                          key_idxes=self.key_idxes,
                                          return_depth=self.use_fusion,
                                          use_fusion=self.use_fusion)
        predict_loader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=self.use_fusion),
            num_workers=4,
            sampler=None,
        )
        return predict_loader

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def predict_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'predict')

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser
