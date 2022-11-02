import os

import mmcv
import numpy as np
import torch
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from PIL import Image

from .base_det_dataset import BaseDetDataset
from .utils import bev_transform, depth_transform, img_transform

__all__ = ['WaymoDetDataset']


class WaymoDetDataset(BaseDetDataset):
    def get_lidar_depth(self, lidar_points, img, lidar_info, cam_info):
        ego2sensor = np.linalg.inv(cam_info['sensor2ego'])
        lidar_coords = np.ones_like(lidar_points,
                                    shape=(lidar_points.shape[0], 4))
        lidar_coords[:, :3] = lidar_points[:, :3]
        sensor_coords = ego2sensor @ lidar_coords[:, :, np.newaxis]
        intrin_mat = np.zeros_like(lidar_points, shape=(3, 3))
        intrin_mat[0, 0] = cam_info['intrinsic'][0]
        intrin_mat[1, 1] = cam_info['intrinsic'][1]
        intrin_mat[0, 2] = cam_info['intrinsic'][2]
        intrin_mat[1, 2] = cam_info['intrinsic'][3]
        intrin_mat[2, 2] = 1
        img_coords = (intrin_mat @ sensor_coords[:, :3, :]).squeeze(-1)[:, :3]
        img_coords[:, :2] /= img_coords[:, 2:]
        mask = np.ones(img_coords.shape[0], dtype=bool)
        mask = np.logical_and(mask, img_coords[:, 2] > 0)
        mask = np.logical_and(mask, img_coords[:, 0] > 1)
        mask = np.logical_and(mask, img_coords[:, 0] < img.size[0] - 1)
        mask = np.logical_and(mask, img_coords[:, 1] > 1)
        mask = np.logical_and(mask, img_coords[:, 1] < img.size[1] - 1)
        valid_points = img_coords[mask]

        return valid_points

    def get_image(self, cam_infos_sweeps, cams, lidar_infos_sweeps=None):
        """Given data and cam_names, return image data needed.

        Args:
            sweeps_data (list): Raw data used to generate the data we needed.
            cams (list): Camera names.

        Returns:
            Tensor: Image data after processing.
            Tensor: Transformation matrix from camera to ego.
            Tensor: Intrinsic matrix.
            Tensor: Transformation matrix for ida.
            Tensor: Transformation matrix from key
                frame camera to sweep frame camera.
            Tensor: timestamps.
            dict: meta infos needed for evaluation.
        """
        assert len(cam_infos_sweeps) > 0
        sweep_imgs = list()
        sweep_sensor2ego_mats = list()
        sweep_intrin_mats = list()
        sweep_ida_mats = list()
        sweep_sensor2sensor_mats = list()
        sweep_timestamps = list()
        sweep_lidar_points = list()
        sweep_lidar_depth = list()
        if self.return_depth or self.use_fusion:
            for lidar_infos in lidar_infos_sweeps:
                lidar_points = []
                for lidar_key in self.lidar_keys:
                    range_image = mmcv.load(
                        os.path.join(self.data_root,
                                     lidar_infos[lidar_key]['filename']))
                    lidar_points.extend(range_image['point_clouds'])
                lidar_points = np.concatenate(lidar_points, axis=0)
                lidar_points = lidar_points.astype(np.float32)[:, :5]
                sweep_lidar_points.append(lidar_points)
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0, 0.0],
                                       [0.0, 0.0, -1.0, 0.0],
                                       [1.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 1.0]])
        for cam in cams:
            imgs = list()
            sensor2ego_mats = list()
            intrin_mats = list()
            ida_mats = list()
            sensor2sensor_mats = list()
            timestamps = list()
            lidar_depth = list()
            key_info = cam_infos_sweeps[0]
            for sweep_idx, cam_info in enumerate(cam_infos_sweeps):

                img = Image.open(
                    os.path.join(self.data_root, cam_info[cam]['filename']))
                if sweep_idx == 0:
                    H = img.size[1]
                    W = img.size[0]
                    resize, resize_dims, crop, flip, \
                        rotate_ida = self.sample_ida_augmentation(
                            H, W
                        )
                # sweep sensor to sweep ego
                sweepsensor2sweepego = np.linalg.inv(
                    T_front_cam_to_ref @ np.linalg.inv(
                        cam_info[cam]['extrinsic'].reshape(4, 4)))
                # sweep ego to global
                sweepego2global = cam_info['ref_pose']

                # global to cur ego
                global2keyego = np.linalg.inv(key_info['ref_pose'])

                # key ego to key sensor
                keysensor2keyego = np.linalg.inv(
                    T_front_cam_to_ref @ np.linalg.inv(
                        key_info[cam]['extrinsic'].reshape(4, 4)))
                keyego2keysensor = np.linalg.inv(keysensor2keyego)
                keysensor2sweepsensor = np.linalg.inv(
                    keyego2keysensor @ global2keyego @ sweepego2global
                    @ sweepsensor2sweepego)
                sweepsensor2keyego = global2keyego @ sweepego2global @\
                    sweepsensor2sweepego
                sensor2ego_mats.append(torch.from_numpy(sweepsensor2keyego))
                sensor2sensor_mats.append(
                    torch.from_numpy(keysensor2sweepsensor))
                intrin_mat = torch.zeros((4, 4))
                intrin_mat[3, 3] = 1
                intrin_mat[0, 0] = cam_info[cam]['intrinsic'][0]
                intrin_mat[1, 1] = cam_info[cam]['intrinsic'][1]
                intrin_mat[0, 2] = cam_info[cam]['intrinsic'][2]
                intrin_mat[1, 2] = cam_info[cam]['intrinsic'][3]
                intrin_mat[2, 2] = 1
                cam_info[cam]['sensor2ego'] = sweepsensor2sweepego
                if self.return_depth and (self.use_fusion or sweep_idx == 0):
                    point_depth = self.get_lidar_depth(
                        sweep_lidar_points[sweep_idx], img,
                        lidar_infos_sweeps[sweep_idx], cam_info[cam])
                    point_depth_augmented = depth_transform(
                        point_depth, resize, self.ida_aug_conf['final_dim'],
                        crop, flip, rotate_ida)
                    lidar_depth.append(point_depth_augmented)
                img, ida_mat = img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate_ida,
                )
                ida_mats.append(ida_mat)
                img = mmcv.imnormalize(np.array(img), self.img_mean,
                                       self.img_std, self.to_rgb)
                img = torch.from_numpy(img).permute(2, 0, 1)
                imgs.append(img)
                intrin_mats.append(intrin_mat)
                timestamps.append(cam_info[cam]['pose_timestamp'])
            sweep_imgs.append(torch.stack(imgs))
            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_ida_mats.append(torch.stack(ida_mats))
            sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats))
            sweep_timestamps.append(torch.tensor(timestamps))
            if self.return_depth:
                sweep_lidar_depth.append(torch.stack(lidar_depth))
        img_metas = dict(box_type_3d=LiDARInstance3DBoxes)

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3).float(),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_timestamps).permute(1, 0), img_metas
        ]
        if self.return_depth:
            ret_list.append(torch.stack(sweep_lidar_depth).permute(1, 0, 2, 3))
        return ret_list

    def get_gt(self, info, cams, img_metas):
        """Generate gt labels from info.

        Args:
            info(dict): Infos needed to generate gt labels.
            cams(list): Camera names.

        Returns:
            Tensor: GT bboxes.
            Tensor: GT labels.
        """
        gt_boxes = list()
        gt_labels = list()
        gt_classes3d = list()
        num_points_in_gt = list()
        difficultys = list()
        for i, (gt_box3d, gt_class3d,
                gt_most_visible_camera_name) in enumerate(
                    zip(info['gt_boxes3d'], info['gt_classes3d'],
                        info['gt_most_visible_camera_names'])):
            # Use ego coordinate.
            # Temporary solution for masking gt_boxes that
            # cameras are unable to see.
            if (gt_class3d
                    not in self.classes) or gt_most_visible_camera_name == '':
                continue
            gt_boxes.append(gt_box3d)
            gt_labels.append(self.classes.index(gt_class3d))
            gt_classes3d.append(gt_class3d)
            num_points_in_gt.append(info['num_points_in_gt'][i])
            difficultys.append(info['difficultys'][i])
        img_metas['num_points_in_gt'] = np.array(num_points_in_gt)
        img_metas['difficultys'] = np.array(difficultys)
        img_metas['timestamp'] = info['timestamp']
        img_metas['scene_name'] = info['scene_name']
        img_metas['gt_most_visible_camera_names'] = info[
            'gt_most_visible_camera_names']
        img_metas['obj_ids'] = info['obj_ids']
        if len(gt_boxes) == 0:
            return torch.zeros((0, 7)), torch.zeros((0)), gt_classes3d
        return torch.Tensor(gt_boxes), torch.tensor(gt_labels), gt_classes3d

    def choose_cams(self):
        """Choose cameras randomly.

        Returns:
            list: Cameras to be used.
        """
        if self.is_train and self.ida_aug_conf['Ncams'] < len(
                self.ida_aug_conf['cams']):
            cams = np.random.choice(self.ida_aug_conf['cams'],
                                    self.ida_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.ida_aug_conf['cams']
        return cams

    def __getitem__(self, idx):
        if self.use_cbgs:
            idx = self.sample_indices[idx]
        cam_infos_sweeps = list()
        lidar_infos_sweeps = list()
        # TODO: Check if it still works when number of cameras is reduced.
        cams = self.choose_cams()
        for key_idx in self.key_idxes:
            cur_idx = key_idx + idx
            # Handle scenarios when current idx doesn't have previous key
            # frame or previous key frame is from another scene.
            if cur_idx < 0:
                cur_idx = idx
            elif self.infos[cur_idx]['scene_name'] != self.infos[idx][
                    'scene_name']:
                cur_idx = idx
            info = self.infos[cur_idx]
            cam_infos = info['cam_infos']
            lidar_infos = info['lidar_infos']
            cam_infos['ref_pose'] = info['ref_pose']
            lidar_infos['ref_pose'] = info['ref_pose']
            cam_infos_sweeps.append(cam_infos)
            lidar_infos_sweeps.append(lidar_infos)
        if self.return_depth or self.use_fusion:
            image_data_list = self.get_image(cam_infos_sweeps, cams,
                                             lidar_infos_sweeps)

        else:
            image_data_list = self.get_image(cam_infos_sweeps, cams)
        ret_list = list()
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            sweep_timestamps,
            img_metas,
        ) = image_data_list[:7]
        gt_boxes, gt_labels, gt_classes3d = self.get_gt(
            self.infos[idx], cams, img_metas)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        bda_mat = sweep_imgs.new_zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = bev_transform(gt_boxes, rotate_bda, scale_bda,
                                          flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        ret_list = [
            sweep_imgs, sweep_sensor2ego_mats, sweep_intrins, sweep_ida_mats,
            sweep_sensor2sensor_mats, bda_mat, sweep_timestamps, img_metas,
            gt_boxes, gt_labels, gt_classes3d
        ]
        if self.return_depth:
            ret_list.append(image_data_list[7])
        return ret_list

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: \
            {"train" if self.is_train else "val"}.
                    Augmentation Conf: {self.ida_aug_conf}"""

    def __len__(self):
        if self.use_cbgs:
            return len(self.sample_indices)
        else:
            return len(self.infos)


def collate_fn(data, is_return_depth=False):
    imgs_batch = list()
    sensor2ego_mats_batch = list()
    intrin_mats_batch = list()
    ida_mats_batch = list()
    sensor2sensor_mats_batch = list()
    bda_mat_batch = list()
    timestamps_batch = list()
    gt_boxes_batch = list()
    gt_labels_batch = list()
    gt_classes3d_batch = list()
    img_metas_batch = list()
    depth_labels_batch = list()
    for iter_data in data:
        (
            sweep_imgs,
            sweep_sensor2ego_mats,
            sweep_intrins,
            sweep_ida_mats,
            sweep_sensor2sensor_mats,
            bda_mat,
            sweep_timestamps,
            img_metas,
            gt_boxes,
            gt_labels,
            gt_classes3d,
        ) = iter_data[:11]
        if is_return_depth:
            gt_depth = iter_data[11]
            depth_labels_batch.append(gt_depth)
        imgs_batch.append(sweep_imgs)
        sensor2ego_mats_batch.append(sweep_sensor2ego_mats)
        intrin_mats_batch.append(sweep_intrins)
        ida_mats_batch.append(sweep_ida_mats)
        sensor2sensor_mats_batch.append(sweep_sensor2sensor_mats)
        bda_mat_batch.append(bda_mat)
        timestamps_batch.append(sweep_timestamps)
        img_metas_batch.append(img_metas)
        gt_boxes_batch.append(gt_boxes)
        gt_labels_batch.append(gt_labels)
        gt_classes3d_batch.append(gt_classes3d)
    mats_dict = dict()
    mats_dict['sensor2ego_mats'] = torch.stack(sensor2ego_mats_batch)
    mats_dict['intrin_mats'] = torch.stack(intrin_mats_batch)
    mats_dict['ida_mats'] = torch.stack(ida_mats_batch)
    mats_dict['sensor2sensor_mats'] = torch.stack(sensor2sensor_mats_batch)
    mats_dict['bda_mat'] = torch.stack(bda_mat_batch)
    ret_list = [
        torch.stack(imgs_batch), mats_dict,
        torch.stack(timestamps_batch), img_metas_batch, gt_boxes_batch,
        gt_labels_batch, gt_classes3d_batch
    ]
    if is_return_depth:
        ret_list.append(torch.stack(depth_labels_batch))
    return ret_list
