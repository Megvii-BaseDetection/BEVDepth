import os

import mmcv
import numpy as np
import torch
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from PIL import Image
from pyquaternion import Quaternion

from .base_det_dataset import BaseDetDataset
from .utils import bev_transform, depth_transform, img_transform

__all__ = ['NuscDetDataset']

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


def map_pointcloud_to_image(
    lidar_points,
    img,
    lidar_calibrated_sensor,
    lidar_ego_pose,
    cam_calibrated_sensor,
    cam_ego_pose,
    min_dist: float = 0.0,
):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    lidar_points = LidarPointCloud(lidar_points.T)
    lidar_points.rotate(
        Quaternion(lidar_calibrated_sensor['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_calibrated_sensor['translation']))

    # Second step: transform from ego to the global frame.
    lidar_points.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    lidar_points.translate(np.array(lidar_ego_pose['translation']))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    lidar_points.translate(-np.array(cam_ego_pose['translation']))
    lidar_points.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    lidar_points.translate(-np.array(cam_calibrated_sensor['translation']))
    lidar_points.rotate(
        Quaternion(cam_calibrated_sensor['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = lidar_points.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(lidar_points.points[:3, :],
                         np.array(cam_calibrated_sensor['camera_intrinsic']),
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring


class NuscDetDataset(BaseDetDataset):
    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        sample_indices = super()._get_sample_indices(
            map_name_from_general_to_detection)
        return sample_indices

    def get_lidar_depth(self, lidar_points, img, lidar_info, cam_info):
        lidar_calibrated_sensor = lidar_info['LIDAR_TOP']['calibrated_sensor']
        lidar_ego_pose = lidar_info['LIDAR_TOP']['ego_pose']
        cam_calibrated_sensor = cam_info['calibrated_sensor']
        cam_ego_pose = cam_info['ego_pose']
        pts_img, depth = map_pointcloud_to_image(
            lidar_points.copy(), img, lidar_calibrated_sensor.copy(),
            lidar_ego_pose.copy(), cam_calibrated_sensor, cam_ego_pose)
        return np.concatenate([pts_img[:2, :].T, depth[:, None]],
                              axis=1).astype(np.float32)

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
        sweep_lidar_depth = list()
        if self.return_depth or self.use_fusion:
            sweep_lidar_points = list()
            for lidar_info in lidar_infos_sweeps:
                lidar_path = lidar_info['LIDAR_TOP']['filename']
                lidar_points = np.fromfile(os.path.join(
                    self.data_root, lidar_path),
                                           dtype=np.float32,
                                           count=-1).reshape(-1, 5)[..., :4]
                sweep_lidar_points.append(lidar_points)
        for cam in cams:
            imgs = list()
            sensor2ego_mats = list()
            intrin_mats = list()
            ida_mats = list()
            sensor2sensor_mats = list()
            timestamps = list()
            lidar_depth = list()
            key_info = cam_infos_sweeps[0]
            resize, resize_dims, crop, flip, \
                rotate_ida = self.sample_ida_augmentation(
                    )
            for sweep_idx, cam_info in enumerate(cam_infos_sweeps):

                img = Image.open(
                    os.path.join(self.data_root, cam_info[cam]['filename']))
                # img = Image.fromarray(img)
                w, x, y, z = cam_info[cam]['calibrated_sensor']['rotation']
                # sweep sensor to sweep ego
                sweepsensor2sweepego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepsensor2sweepego_tran = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['translation'])
                sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros(
                    (4, 4))
                sweepsensor2sweepego[3, 3] = 1
                sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
                sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
                # sweep ego to global
                w, x, y, z = cam_info[cam]['ego_pose']['rotation']
                sweepego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                sweepego2global_tran = torch.Tensor(
                    cam_info[cam]['ego_pose']['translation'])
                sweepego2global = sweepego2global_rot.new_zeros((4, 4))
                sweepego2global[3, 3] = 1
                sweepego2global[:3, :3] = sweepego2global_rot
                sweepego2global[:3, -1] = sweepego2global_tran

                # global sensor to cur ego
                w, x, y, z = key_info[cam]['ego_pose']['rotation']
                keyego2global_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keyego2global_tran = torch.Tensor(
                    key_info[cam]['ego_pose']['translation'])
                keyego2global = keyego2global_rot.new_zeros((4, 4))
                keyego2global[3, 3] = 1
                keyego2global[:3, :3] = keyego2global_rot
                keyego2global[:3, -1] = keyego2global_tran
                global2keyego = keyego2global.inverse()

                # cur ego to sensor
                w, x, y, z = key_info[cam]['calibrated_sensor']['rotation']
                keysensor2keyego_rot = torch.Tensor(
                    Quaternion(w, x, y, z).rotation_matrix)
                keysensor2keyego_tran = torch.Tensor(
                    key_info[cam]['calibrated_sensor']['translation'])
                keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
                keysensor2keyego[3, 3] = 1
                keysensor2keyego[:3, :3] = keysensor2keyego_rot
                keysensor2keyego[:3, -1] = keysensor2keyego_tran
                keyego2keysensor = keysensor2keyego.inverse()
                keysensor2sweepsensor = (
                    keyego2keysensor @ global2keyego @ sweepego2global
                    @ sweepsensor2sweepego).inverse()
                sweepsensor2keyego = global2keyego @ sweepego2global @\
                    sweepsensor2sweepego
                sensor2ego_mats.append(sweepsensor2keyego)
                sensor2sensor_mats.append(keysensor2sweepsensor)
                intrin_mat = torch.zeros((4, 4))
                intrin_mat[3, 3] = 1
                intrin_mat[:3, :3] = torch.Tensor(
                    cam_info[cam]['calibrated_sensor']['camera_intrinsic'])
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
                timestamps.append(cam_info[cam]['timestamp'])
            sweep_imgs.append(torch.stack(imgs))
            sweep_sensor2ego_mats.append(torch.stack(sensor2ego_mats))
            sweep_intrin_mats.append(torch.stack(intrin_mats))
            sweep_ida_mats.append(torch.stack(ida_mats))
            sweep_sensor2sensor_mats.append(torch.stack(sensor2sensor_mats))
            sweep_timestamps.append(torch.tensor(timestamps))
            if self.return_depth:
                sweep_lidar_depth.append(torch.stack(lidar_depth))
        # Get mean pose of all cams.
        ego2global_rotation = np.mean(
            [key_info[cam]['ego_pose']['rotation'] for cam in cams], 0)
        ego2global_translation = np.mean(
            [key_info[cam]['ego_pose']['translation'] for cam in cams], 0)
        img_metas = dict(
            box_type_3d=LiDARInstance3DBoxes,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
        )

        ret_list = [
            torch.stack(sweep_imgs).permute(1, 0, 2, 3, 4),
            torch.stack(sweep_sensor2ego_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_intrin_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_ida_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_sensor2sensor_mats).permute(1, 0, 2, 3),
            torch.stack(sweep_timestamps).permute(1, 0),
            img_metas,
        ]
        if self.return_depth:
            ret_list.append(torch.stack(sweep_lidar_depth).permute(1, 0, 2, 3))
        return ret_list

    def get_gt(self, info, cams):
        """Generate gt labels from info.

        Args:
            info(dict): Infos needed to generate gt labels.
            cams(list): Camera names.

        Returns:
            Tensor: GT bboxes.
            Tensor: GT labels.
        """
        ego2global_rotation = np.mean(
            [info['cam_infos'][cam]['ego_pose']['rotation'] for cam in cams],
            0)
        ego2global_translation = np.mean([
            info['cam_infos'][cam]['ego_pose']['translation'] for cam in cams
        ], 0)
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        gt_boxes = list()
        gt_labels = list()
        for ann_info in info['ann_infos']:
            # Use ego coordinate.
            if (map_name_from_general_to_detection[ann_info['category_name']]
                    not in self.classes
                    or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <=
                    0):
                continue
            box = Box(
                ann_info['translation'],
                ann_info['size'],
                Quaternion(ann_info['rotation']),
                velocity=ann_info['velocity'],
            )
            box.translate(trans)
            box.rotate(rot)
            box_xyz = np.array(box.center)
            box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
            box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
            box_velo = np.array(box.velocity[:2])
            gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
            gt_boxes.append(gt_box)
            gt_labels.append(
                self.classes.index(map_name_from_general_to_detection[
                    ann_info['category_name']]))
        return torch.Tensor(gt_boxes), torch.tensor(gt_labels)

    def __getitem__(self, idx):
        if self.use_cbgs:
            idx = self.sample_indices[idx]
        cam_infos = list()
        lidar_infos = list()
        # TODO: Check if it still works when number of cameras is reduced.
        cams = self.choose_cams()
        for key_idx in self.key_idxes:
            cur_idx = key_idx + idx
            # Handle scenarios when current idx doesn't have previous key
            # frame or previous key frame is from another scene.
            if cur_idx < 0:
                cur_idx = idx
            elif self.infos[cur_idx]['scene_token'] != self.infos[idx][
                    'scene_token']:
                cur_idx = idx
            info = self.infos[cur_idx]
            cam_infos.append(info['cam_infos'])
            lidar_infos.append(info['lidar_infos'])
            lidar_sweep_timestamps = [
                lidar_sweep['LIDAR_TOP']['timestamp']
                for lidar_sweep in info['lidar_sweeps']
            ]
            for sweep_idx in self.sweeps_idx:
                if len(info['cam_sweeps']) == 0:
                    cam_infos.append(info['cam_infos'])
                    lidar_infos.append(info['lidar_infos'])
                else:
                    # Handle scenarios when current sweep doesn't have all
                    # cam keys.
                    for i in range(min(len(info['cam_sweeps']) - 1, sweep_idx),
                                   -1, -1):
                        if sum([cam in info['cam_sweeps'][i]
                                for cam in cams]) == len(cams):
                            cam_infos.append(info['cam_sweeps'][i])
                            cam_timestamp = np.mean([
                                val['timestamp']
                                for val in info['cam_sweeps'][i].values()
                            ])
                            # Find the closest lidar frame to the cam frame.
                            lidar_idx = np.abs(lidar_sweep_timestamps -
                                               cam_timestamp).argmin()
                            lidar_infos.append(info['lidar_sweeps'][lidar_idx])
                            break
        if self.return_depth or self.use_fusion:
            image_data_list = self.get_image(cam_infos, cams, lidar_infos)

        else:
            image_data_list = self.get_image(cam_infos, cams)
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
        img_metas['token'] = self.infos[idx]['sample_token']
        if self.is_train:
            gt_boxes, gt_labels = self.get_gt(self.infos[idx], cams)
        # Temporary solution for test.
        else:
            gt_boxes = sweep_imgs.new_zeros(0, 7)
            gt_labels = sweep_imgs.new_zeros(0, )

        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(
        )
        bda_mat = sweep_imgs.new_zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = bev_transform(gt_boxes, rotate_bda, scale_bda,
                                          flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        ret_list = [
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
        ]
        if self.return_depth:
            ret_list.append(image_data_list[7])
        return ret_list

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: \
            {"train" if self.is_train else "val"}.
                    Augmentation Conf: {self.ida_aug_conf}"""
