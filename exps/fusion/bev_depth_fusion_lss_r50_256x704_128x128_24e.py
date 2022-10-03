# Copyright (c) Megvii Inc. All rights reserved.
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed

from exps.base_cli import run_cli
from exps.base_exp import BEVDepthLightningModel as BaseBEVDepthLightningModel
from models.fusion_bev_depth import FusionBEVDepth


class BEVDepthLightningModel(BaseBEVDepthLightningModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = FusionBEVDepth(self.backbone_conf,
                                    self.head_conf,
                                    is_train_depth=False)
        self.use_fusion = True

    def forward(self, sweep_imgs, mats, lidar_depth):
        return self.model(sweep_imgs, mats, lidar_depth)

    def training_step(self, batch):
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, lidar_depth) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
        preds = self(sweep_imgs, mats, lidar_depth)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            targets = self.model.module.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.module.loss(targets, preds)
        else:
            targets = self.model.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.loss(targets, preds)

        if len(lidar_depth.shape) == 5:
            # only key-frame will calculate depth loss
            lidar_depth = lidar_depth[:, 0, ...]
        self.log('detection_loss', detection_loss)
        return detection_loss

    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, img_metas, _, _, lidar_depth) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
        preds = self.model(sweep_imgs, mats, lidar_depth)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].tensor.detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_fusion_lss_r50_256x704_128x128_24e')
