import os
import struct
from argparse import ArgumentParser
from concurrent import futures

import mmcv
import numpy as np
import tensorflow as tf
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.utils import (frame_utils, range_image_utils,
                                      transform_utils)

WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
LIDAR_NAMES = ['UNKNOWN', 'TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR']
CAMERA_NAMES = [
    'UNKNOWN', 'FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT'
]


def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('dataset_type',
                        default='nuscenes',
                        help='Type of the dataset to be processed.')
    parser.add_argument('dataset_version',
                        default='v1.0',
                        help='Version of the dataset to be processed.')
    parser.add_argument('--num-workers',
                        default=8,
                        type=int,
                        help='Number of wprkers to use for processing data.')
    args = parser.parse_args()
    return args


def save_image(frame, cur_save_path, file_name, cam_names, cam_infos):
    """
    Extract image and acquire camera infos.

    Args:
        frame (obj): open dataset frame
        cur_save_path (str): Path of the file to be saved.
        file_name (str): Name of the file to be saved.
        cam_names (list[str]): Name of the sensors.
        cam_infos (dict): Dict to save infos of camera.
            Defaults to [0, 1]
    """
    for frame_image in frame.images:
        cam_info = dict()
        img = mmcv.imfrombytes(frame_image.image)
        save_path = os.path.join(cur_save_path, cam_names[frame_image.name])
        mmcv.mkdir_or_exist(save_path)
        mmcv.imwrite(img, os.path.join(save_path, file_name))
        cam_info['pose'] = np.array(frame_image.pose.transform)
        velocity = dict()
        velocity['v_x'] = frame_image.velocity.v_x
        velocity['v_y'] = frame_image.velocity.v_y
        velocity['v_z'] = frame_image.velocity.v_z
        velocity['v_x'] = frame_image.velocity.v_x
        velocity['v_y'] = frame_image.velocity.v_y
        velocity['v_z'] = frame_image.velocity.v_z
        cam_info['velocity'] = velocity
        cam_info['pose_timestamp'] = frame_image.pose_timestamp
        cam_info['shutter'] = frame_image.shutter
        cam_info['cam_trigger_time'] = frame_image.camera_trigger_time
        cam_info[
            'cam_readout_done_time'] = frame_image.camera_readout_done_time
        cam_info['filename'] = os.path.join(*save_path.split(os.sep)[-3:],
                                            file_name)
        cam_infos[cam_names[frame_image.name]] = cam_info
    for camera in frame.context.camera_calibrations:
        # extrinsic parameters
        cam_infos[cam_names[camera.name]]['extrinsic'] = np.array(
            camera.extrinsic.transform)
        # intrinsic parameters
        cam_infos[cam_names[camera.name]]['intrinsic'] = np.array(
            camera.intrinsic)


def save_label(frame, objects, version='multi-view', cam_sync=True):
    """Modified from https://github.com/Tai-Wang/Depth-from-Motion/blob/
        main/tools/create_waymo_gt_bin.py.
        Parse and save gt bin file for camera-only 3D detection on Waymo.
    Args:
        frame (:obj:`Frame`): Open dataset frame proto.
        objects (:obj:`Object`): Ground truths in waymo dataset Object proto.
        version (str): Version of gt bin file. Choices include 'multi-view'
            and 'front-view'. Defaults to 'multi-view'.
        cam_sync (bool): Whether to generate camera synced gt bin. Defaults to
            True.
    """
    id_to_bbox = dict()
    id_to_name = dict()
    for labels in frame.projected_lidar_labels:
        name = labels.name  # 0 unknown, 1-5 corresponds to 5 cameras
        for label in labels.labels:
            # TODO: need a workaround as bbox may not belong to front cam
            bbox = [
                label.box.center_x - label.box.length / 2,
                label.box.center_y - label.box.width / 2,
                label.box.center_x + label.box.length / 2,
                label.box.center_y + label.box.width / 2
            ]
            # object id in one frame
            id_to_bbox[label.id] = bbox
            id_to_name[label.id] = name - 1
    if version == 'multi-view':
        cam_list = [
            '_FRONT', '_FRONT_LEFT', '_FRONT_RIGHT', '_SIDE_LEFT',
            '_SIDE_RIGHT'
        ]
    elif version == 'front-view':
        cam_list = ['_FRONT']
    else:
        raise NotImplementedError
    for obj in frame.laser_labels:
        bounding_box = None
        name = None
        id = obj.id
        for cam in cam_list:
            if id + cam in id_to_bbox:
                bounding_box = id_to_bbox.get(id + cam)
                name = str(id_to_name.get(id + cam))
                break
        num_pts = obj.num_lidar_points_in_box

        if cam_sync:
            if obj.most_visible_camera_name:
                box3d = obj.camera_synced_box
            else:
                continue
        else:
            box3d = obj.box

        if bounding_box is not None and obj.type > 0 and num_pts >= 1:
            o = metrics_pb2.Object()
            o.context_name = frame.context.name
            o.frame_timestamp_micros = frame.timestamp_micros
            o.score = 0.5
            o.object.CopyFrom(obj)
            o.object.box.CopyFrom(box3d)
            objects.objects.append(o)


def save_range_image(frame,
                     cur_save_path,
                     file_name,
                     lidar_names,
                     lidar_infos,
                     return_indexes=[0, 1]):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.

    Args:
        frame (obj): open dataset frame
        cur_save_path (str): Path of the file to be saved.
        file_name (str): Name of the file to be saved.
        lidar_names (list[str]): Name of the sensors.
        lidar_infos (dict): Dict to save infos of camera.
        return_indexes (list[int]): 0 for the first return, 1
            for the second return.
            Defaults to [0, 1]

    """

    range_images, _, _, range_image_top_pose = \
        frame_utils.parse_range_image_and_camera_projection(frame)
    calibrations = sorted(frame.context.laser_calibrations,
                          key=lambda c: c.name)

    for calib in calibrations:
        point_clouds = []
        range_image_idxes = []
        range_image_shapes = []
        range_image_infos = []
        for ri_index in return_indexes:
            single_lidar_info = dict()
            point_cloud = range_images[calib.name][ri_index]
            if len(calib.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([
                        calib.beam_inclination_min, calib.beam_inclination_max
                    ]),
                    height=point_cloud.shape.dims[0])
            else:
                beam_inclinations = tf.constant(calib.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(calib.extrinsic.transform), [4, 4])
            single_lidar_info['beam_inclinations'] = beam_inclinations.numpy()
            single_lidar_info['extrinsic'] = extrinsic
            range_image_infos.append(single_lidar_info)
            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(point_cloud.data), point_cloud.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if calib.name == dataset_pb2.LaserName.TOP:
                # frame pose and pixel pose is needed to transform to
                # global frame.
                frame_pose = tf.convert_to_tensor(
                    np.reshape(np.array(frame.pose.transform), [4, 4]))
                # [H, W, 6]
                range_image_top_pose_tensor = tf.reshape(
                    tf.convert_to_tensor(range_image_top_pose.data),
                    range_image_top_pose.shape.dims)
                # [H, W, 3, 3]
                range_image_top_pose_tensor_rotation = \
                    transform_utils.get_rotation_matrix(
                        range_image_top_pose_tensor[..., 0],
                        range_image_top_pose_tensor[..., 1],
                        range_image_top_pose_tensor[..., 2])
                range_image_top_pose_tensor_translation = \
                    range_image_top_pose_tensor[..., 3:]
                range_image_top_pose_tensor = transform_utils.get_transform(
                    range_image_top_pose_tensor_rotation,
                    range_image_top_pose_tensor_translation)
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_depth = range_image_tensor[..., 0:1].numpy()
            range_image_NLZ = range_image_tensor[..., 3:4].numpy()
            range_image_intensity = range_image_tensor[..., 1:2].numpy()
            range_image_elongation = range_image_tensor[..., 2:3].numpy()
            range_image_cartesian = \
                range_image_utils.extract_point_cloud_from_range_image(
                    tf.expand_dims(range_image_tensor[..., 0], axis=0),
                    tf.expand_dims(extrinsic, axis=0),
                    tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),  # noqa: E501
                    pixel_pose=pixel_pose_local,
                    frame_pose=frame_pose_local).numpy()

            range_image_np = range_image_tensor.numpy()
            range_image_idx = np.where(range_image_np[..., 0] > 0)
            range_image_cartesian = np.squeeze(range_image_cartesian, axis=0)
            point_cloud = np.concatenate([
                range_image_cartesian, range_image_intensity,
                range_image_elongation, range_image_NLZ, range_image_depth
            ],
                                         axis=-1)
            point_clouds.append(point_cloud[range_image_idx])
            range_image_idxes.append(range_image_idx)
            range_image_shapes.append(range_image_np.shape)
        lidar_infos[lidar_names[calib.name]] = dict()
        lidar_infos[lidar_names[
            calib.name]]['range_image_infos'] = range_image_infos
        save_path = os.path.join(cur_save_path, lidar_names[calib.name])
        lidar_infos[lidar_names[calib.name]]['filename'] = os.path.join(
            *save_path.split(os.sep)[-3:], file_name)
        mmcv.mkdir_or_exist(save_path)

        save_dict = dict(point_clouds=point_clouds,
                         range_image_idxes=range_image_idxes,
                         range_image_shapes=range_image_shapes)
        mmcv.dump(save_dict, os.path.join(save_path, file_name))


class tfrecord_decode_iterator(object):
    """An iterator to yield sample and its offset in tfrecord.

    The iterator uses a hacking way to decode sample data and its
    offset and length in tfrecord. Also see TFRecords format details:
    https://www.tensorflow.org/tutorials/load_data/tfrecord

    Args:
        tfrecord_path (str): file path of tfrecord.
    """
    def __init__(self, tfrecord_path):
        self.f = open(tfrecord_path, mode='rb')

    def __next__(self):
        length_str = self.f.read(8)
        offset = 8
        while length_str:
            length = struct.unpack('<LL', length_str)[0]
            self.f.seek(4, 1)  # skip crc
            offset += 4
            data = self.f.read(length)

            yield data, offset, length

            offset += length
            self.f.seek(4, 1)  # skip crc
            offset += 4
            length_str = self.f.read(8)
            offset += 8
        self.f.close()

    def __iter__(self):
        return next(self)

    def __del__(self):
        self.f.close()


def load_tfrecord_offset(tfrecord_file, offset, length):
    """Load a sample located at the `offset` in `tfrecord_file`.

    Args:
        tfrecord_file (str): File name of tfrecord.
        offset (int): Offset which the sample is located at.
        length (int): The byte length of the sample.
    """
    with open(tfrecord_file, mode='rb') as f:
        f.seek(offset, 0)
        data = f.read(length)
    example = tf.train.Example()
    example.ParseFromString(data)
    return example.features.feature


def drop_info_with_name(info, name):
    """Given names, drop these names in info.

    Args:
        info (dict): Infos.
        name (str): Names to drop.

    Returns:
        dict: Infos after dropping.
    """
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['gt_classes3d']) if x != name]
    for key in info.keys():
        if isinstance(info[key], np.ndarray):
            ret_info[key] = info[key][keep_indices]
        elif isinstance(info[key], list):
            ret_info[key] = [info[key][idx] for idx in keep_indices]
    return ret_info


def parse_waymo_samples(example_proto, idx, directory, tfrecord):
    """Parse sample infos for waymo dataset. Modified from
    `https://github.com/open-
    mmlab/OpenPCDet/blob/master/pcdet/datasets/waymo/waymo_dataset.py`.

    Args:
        example_proto (tf.train.Example): Proto in tfrecord.

    Returns:
        dict: Info of the input proto.
            - gt_classes3d (list[str]): Class names of each gt boxes.
            - difficultys (list[str]): Difficulty names of each gt boxes.
            - obj_ids (list[str]): Id of each object.
            - tracking_difficultys (list[str]): Tracking difficulty
                names of each gt boxes.
            - gt_boxes3d (np.ndarray): Gt boxes in gravity center
                (x, y, z, dx, dy, dz, yaw).
            - veh_to_global (np.ndarray): Transform matrix from vehicle
                to global.
            - scene_name (str): Name of the scene.
            - filename (str): Name of the frame.
            - time_of_day (str): Time of the day for this scene.
            - timestamp (int): Timestamp of the frame.
            - range_image_path (dict): Path of the range image for each sensor.
    """
    frame = dataset_pb2.Frame()
    frame.ParseFromString(example_proto)
    gt_classes3d, difficultys, gt_boxes3d = [], [], []
    gt_most_visible_camera_names = list()
    # TODO: should also consider speeds and accelerations
    tracking_difficultys, obj_ids = [], []
    num_points_in_gt = []
    laser_labels = frame.laser_labels
    for i, laser_label in enumerate(laser_labels):
        if laser_label.num_lidar_points_in_box == 0:
            continue
        box = laser_labels[i].box
        class_ind = laser_label.type
        gt_box3d = [
            box.center_x, box.center_y, box.center_z, box.length, box.width,
            box.height, box.heading
        ]
        gt_boxes3d.append(gt_box3d)
        gt_classes3d.append(WAYMO_CLASSES[class_ind])
        difficultys.append(laser_label.detection_difficulty_level)
        tracking_difficultys.append(laser_label.tracking_difficulty_level)
        obj_ids.append(laser_label.id)
        num_points_in_gt.append(laser_label.num_lidar_points_in_box)
        gt_most_visible_camera_names.append(
            laser_label.most_visible_camera_name)

    info = dict()
    lidar_infos = dict()
    cam_infos = dict()
    info['gt_classes3d'] = gt_classes3d
    info['difficultys'] = np.array(difficultys)
    gt_boxes3d = np.array(gt_boxes3d, dtype=np.float32)

    info['obj_ids'] = obj_ids
    info['tracking_difficultys'] = np.array(tracking_difficultys)
    info['num_points_in_gt'] = np.array(num_points_in_gt)
    info['gt_most_visible_camera_names'] = gt_most_visible_camera_names
    info = drop_info_with_name(info, name='unknown')
    if len(info['gt_classes3d']) == 0:
        gt_boxes3d = np.zeros((0, 7), dtype=np.float32)
    # Ground truth bboxes with the shape of [N, 7],
    # each bbox contains [x, y, z, dx, dy, dz, yaw].
    info['gt_boxes3d'] = gt_boxes3d
    ref_pose = np.reshape(np.array(frame.pose.transform), [4, 4])
    # Transform matrix from vehicle to global.
    info['ref_pose'] = ref_pose
    # Name of the scene.
    info['scene_name'] = frame.context.name
    # Name of frame.
    info['filename'] = info['scene_name'] + f'_{idx}'
    # Where this scene is taken, e.g., location_phx.
    info['location'] = frame.context.stats.location
    info['time_of_day'] = frame.context.stats.time_of_day
    info['timestamp'] = frame.timestamp_micros
    tf_name = tfrecord.split('.')[0]
    range_image_target_path = os.path.join(directory, 'range_images')
    image_target_path = os.path.join(directory, 'images')
    mmcv.mkdir_or_exist(range_image_target_path)
    lidar_file_name = f'{tf_name}_{idx}.pkl'
    cam_file_name = f'{tf_name}_{idx}.png'

    # Remove it when generating range_image is not needed.
    save_range_image(frame, range_image_target_path, lidar_file_name,
                     LIDAR_NAMES, lidar_infos)
    save_image(frame, image_target_path, cam_file_name, CAMERA_NAMES,
               cam_infos)
    # range_image_path of all sensors.
    range_image_path = dict()
    for sensor_name in LIDAR_NAMES[1:]:
        range_image_path[sensor_name] = os.path.join('range_image',
                                                     sensor_name,
                                                     lidar_file_name)
    lidar_infos['range_image_path'] = range_image_path
    info['lidar_infos'] = lidar_infos
    info['cam_infos'] = cam_infos
    return info


def generate_waymo_info(tfrecord_dir, tfrecord_file):
    tfrecord_path = os.path.join(tfrecord_dir, tfrecord_file)
    record_iterator = tfrecord_decode_iterator(tfrecord_path)

    samples = list()
    for idx, (serialized_sample, offset, length) in enumerate(record_iterator):
        sample = parse_waymo_samples(serialized_sample, idx, tfrecord_dir,
                                     tfrecord_file)
        sample['tfrecord_file'] = os.path.basename(tfrecord_path)
        sample['sample_offset'] = offset
        sample['sample_length'] = length
        samples.append(sample)
    return samples


def generate_nuscenes_info(nusc,
                           scenes,
                           max_cam_sweeps=6,
                           max_lidar_sweeps=10):
    infos = list()
    for cur_scene in tqdm(nusc.scene):
        if cur_scene['name'] not in scenes:
            continue
        first_sample_token = cur_scene['first_sample_token']
        cur_sample = nusc.get('sample', first_sample_token)
        while True:
            info = dict()
            sweep_cam_info = dict()
            cam_datas = list()
            lidar_datas = list()
            info['sample_token'] = cur_sample['token']
            info['timestamp'] = cur_sample['timestamp']
            info['scene_token'] = cur_sample['scene_token']
            cam_names = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
                'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
            ]
            lidar_names = ['LIDAR_TOP']
            cam_infos = dict()
            lidar_infos = dict()
            for cam_name in cam_names:
                cam_data = nusc.get('sample_data',
                                    cur_sample['data'][cam_name])
                cam_datas.append(cam_data)
                sweep_cam_info = dict()
                sweep_cam_info['sample_token'] = cam_data['sample_token']
                sweep_cam_info['ego_pose'] = nusc.get(
                    'ego_pose', cam_data['ego_pose_token'])
                sweep_cam_info['timestamp'] = cam_data['timestamp']
                sweep_cam_info['is_key_frame'] = cam_data['is_key_frame']
                sweep_cam_info['height'] = cam_data['height']
                sweep_cam_info['width'] = cam_data['width']
                sweep_cam_info['filename'] = cam_data['filename']
                sweep_cam_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', cam_data['calibrated_sensor_token'])
                cam_infos[cam_name] = sweep_cam_info
            for lidar_name in lidar_names:
                lidar_data = nusc.get('sample_data',
                                      cur_sample['data'][lidar_name])
                lidar_datas.append(lidar_data)
                sweep_lidar_info = dict()
                sweep_lidar_info['sample_token'] = lidar_data['sample_token']
                sweep_lidar_info['ego_pose'] = nusc.get(
                    'ego_pose', lidar_data['ego_pose_token'])
                sweep_lidar_info['timestamp'] = lidar_data['timestamp']
                sweep_lidar_info['filename'] = lidar_data['filename']
                sweep_lidar_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', lidar_data['calibrated_sensor_token'])
                lidar_infos[lidar_name] = sweep_lidar_info

            lidar_sweeps = [dict() for _ in range(max_lidar_sweeps)]
            cam_sweeps = [dict() for _ in range(max_cam_sweeps)]
            info['cam_infos'] = cam_infos
            info['lidar_infos'] = lidar_infos
            # for i in range(max_cam_sweeps):
            #     cam_sweeps.append(dict())
            for k, cam_data in enumerate(cam_datas):
                sweep_cam_data = cam_data
                for j in range(max_cam_sweeps):
                    if sweep_cam_data['prev'] == '':
                        break
                    else:
                        sweep_cam_data = nusc.get('sample_data',
                                                  sweep_cam_data['prev'])
                        sweep_cam_info = dict()
                        sweep_cam_info['sample_token'] = sweep_cam_data[
                            'sample_token']
                        if sweep_cam_info['sample_token'] != cam_data[
                                'sample_token']:
                            break
                        sweep_cam_info['ego_pose'] = nusc.get(
                            'ego_pose', cam_data['ego_pose_token'])
                        sweep_cam_info['timestamp'] = sweep_cam_data[
                            'timestamp']
                        sweep_cam_info['is_key_frame'] = sweep_cam_data[
                            'is_key_frame']
                        sweep_cam_info['height'] = sweep_cam_data['height']
                        sweep_cam_info['width'] = sweep_cam_data['width']
                        sweep_cam_info['filename'] = sweep_cam_data['filename']
                        sweep_cam_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            cam_data['calibrated_sensor_token'])
                        cam_sweeps[j][cam_names[k]] = sweep_cam_info

            for k, lidar_data in enumerate(lidar_datas):
                sweep_lidar_data = lidar_data
                for j in range(max_lidar_sweeps):
                    if sweep_lidar_data['prev'] == '':
                        break
                    else:
                        sweep_lidar_data = nusc.get('sample_data',
                                                    sweep_lidar_data['prev'])
                        sweep_lidar_info = dict()
                        sweep_lidar_info['sample_token'] = sweep_lidar_data[
                            'sample_token']
                        if sweep_lidar_info['sample_token'] != lidar_data[
                                'sample_token']:
                            break
                        sweep_lidar_info['ego_pose'] = nusc.get(
                            'ego_pose', sweep_lidar_data['ego_pose_token'])
                        sweep_lidar_info['timestamp'] = sweep_lidar_data[
                            'timestamp']
                        sweep_lidar_info['is_key_frame'] = sweep_lidar_data[
                            'is_key_frame']
                        sweep_lidar_info['filename'] = sweep_lidar_data[
                            'filename']
                        sweep_lidar_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            cam_data['calibrated_sensor_token'])
                        lidar_sweeps[j][lidar_names[k]] = sweep_lidar_info
            # Remove empty sweeps.
            for i, sweep in enumerate(cam_sweeps):
                if len(sweep.keys()) == 0:
                    cam_sweeps = cam_sweeps[:i]
                    break
            for i, sweep in enumerate(lidar_sweeps):
                if len(sweep.keys()) == 0:
                    lidar_sweeps = lidar_sweeps[:i]
                    break
            info['cam_sweeps'] = cam_sweeps
            info['lidar_sweeps'] = lidar_sweeps
            ann_infos = list()
            if hasattr(cur_sample, 'anns'):
                for ann in cur_sample['anns']:
                    ann_info = nusc.get('sample_annotation', ann)
                    velocity = nusc.box_velocity(ann_info['token'])
                    if np.any(np.isnan(velocity)):
                        velocity = np.zeros(3)
                    ann_info['velocity'] = velocity
                    ann_infos.append(ann_info)
                info['ann_infos'] = ann_infos
            infos.append(info)
            if cur_sample['next'] == '':
                break
            else:
                cur_sample = nusc.get('sample', cur_sample['next'])
    return infos


def main():
    args = parse_args()
    assert args.dataset_type in ['nuscenes', 'waymo']
    if args.dataset_type == 'nuscenes':
        trainval_nusc = NuScenes(version='v1.0-trainval',
                                 dataroot='./data/nuScenes/',
                                 verbose=True)
        train_scenes = splits.train
        val_scenes = splits.val
        train_infos = generate_nuscenes_info(trainval_nusc, train_scenes)
        val_infos = generate_nuscenes_info(trainval_nusc, val_scenes)
        mmcv.dump(train_infos, './data/nuScenes/nuscenes_infos_train.pkl')
        mmcv.dump(val_infos, './data/nuScenes/nuscenes_infos_val.pkl')
        test_nusc = NuScenes(version='v1.0-test',
                             dataroot='./data/nuScenes/',
                             verbose=True)
        test_scenes = splits.test
        test_infos = generate_nuscenes_info(test_nusc, test_scenes)
        mmcv.dump(test_infos, './data/nuScenes/nuscenes_infos_test.pkl')
    elif args.dataset_type == 'waymo':
        training_dir = os.path.join('./data', 'waymo', args.dataset_version,
                                    'training')
        training_tfrecords = list(
            filter(lambda x: x.endswith('.tfrecord'),
                   os.listdir(training_dir)))
        with futures.ThreadPoolExecutor(args.num_workers) as executor:
            training_tfrecord_infos = list(
                tqdm(executor.map(generate_waymo_info,
                                  [training_dir] * len(training_tfrecords),
                                  training_tfrecords),
                     total=len(training_tfrecords)))
        waymo_infos_training = list()
        for training_tfrecord_info in training_tfrecord_infos:
            waymo_infos_training.extend(training_tfrecord_info)
        mmcv.dump(
            waymo_infos_training,
            os.path.join('./data', 'waymo', args.dataset_version,
                         'waymo_infos_training.pkl'))
        validation_dir = os.path.join('./data', 'waymo', args.dataset_version,
                                      'validation')
        validation_tfrecords = list(
            filter(lambda x: x.endswith('.tfrecord'),
                   os.listdir(validation_dir)))
        with futures.ThreadPoolExecutor(args.num_workers) as executor:
            validation_tfrecord_infos = list(
                tqdm(executor.map(generate_waymo_info,
                                  [validation_dir] * len(validation_tfrecords),
                                  validation_tfrecords),
                     total=len(validation_tfrecords)))
        waymo_infos_valdiation = list()
        for validation_tfrecord_info in validation_tfrecord_infos:
            waymo_infos_valdiation.extend(validation_tfrecord_info)
        mmcv.dump(
            waymo_infos_valdiation,
            os.path.join('./data', 'waymo', args.dataset_version,
                         'waymo_infos_validation.pkl'))
        objects = metrics_pb2.Objects()
        progress_bar = mmcv.ProgressBar(len(validation_tfrecords))
        for i in range(len(validation_tfrecords)):
            pathname = os.path.join('./data', 'waymo', args.dataset_version,
                                    'validation', validation_tfrecords[i])
            dataset = tf.data.TFRecordDataset(pathname, compression_type='')
            for frame_idx, data in enumerate(dataset):
                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                save_label(frame, objects)
            progress_bar.update()

        # Write objects to a file.
        f = open(
            os.path.join('./data', 'waymo', args.dataset_version,
                         'cam_gt.bin'), 'wb')
        f.write(objects.SerializeToString())
        f.close()


if __name__ == '__main__':
    main()
