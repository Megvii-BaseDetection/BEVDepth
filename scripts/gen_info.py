import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm


def generate_info(nusc, scenes):
    infos = list()
    for cur_scene in tqdm(nusc.scene):
        if cur_scene['name'] not in scenes:
            continue
        first_sample_token = cur_scene['first_sample_token']
        cur_sample = nusc.get('sample', first_sample_token)
        while True:
            info = dict()
            cam_info = dict()
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
                cam_info = dict()
                cam_info['sample_token'] = cam_data['sample_token']
                cam_info['ego_pose'] = nusc.get('ego_pose',
                                                cam_data['ego_pose_token'])
                cam_info['timestamp'] = cam_data['timestamp']
                cam_info['is_key_frame'] = cam_data['is_key_frame']
                cam_info['height'] = cam_data['height']
                cam_info['width'] = cam_data['width']
                cam_info['filename'] = cam_data['filename']
                cam_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', cam_data['calibrated_sensor_token'])
                cam_infos[cam_name] = cam_info
            for lidar_name in lidar_names:
                lidar_data = nusc.get('sample_data',
                                      cur_sample['data'][lidar_name])
                lidar_info = dict()
                lidar_info['sample_token'] = lidar_data['sample_token']
                lidar_info['ego_pose'] = nusc.get('ego_pose',
                                                  lidar_data['ego_pose_token'])
                lidar_info['timestamp'] = lidar_data['timestamp']
                lidar_info['filename'] = lidar_data['filename']
                lidar_info['calibrated_sensor'] = nusc.get(
                    'calibrated_sensor', lidar_data['calibrated_sensor_token'])
                lidar_infos[lidar_name] = lidar_info

            sweeps = list()
            info['cam_infos'] = cam_infos
            info['lidar_infos'] = lidar_infos
            cam_datas = list()
            for i in range(6):
                sweeps.append(dict())
            for cam_name in cam_names:
                cam_datas.append(
                    nusc.get('sample_data', cur_sample['data'][cam_name]))
            for k, cam_data in enumerate(cam_datas):
                for j in range(6):
                    if cam_data['prev'] == '':
                        break
                    else:
                        cam_data = nusc.get('sample_data', cam_data['prev'])
                        cam_info = dict()
                        cam_info['sample_token'] = cam_data['sample_token']
                        assert cam_info['sample_token'] == cam_info[
                            'sample_token']
                        cam_info['ego_pose'] = nusc.get(
                            'ego_pose', cam_data['ego_pose_token'])
                        cam_info['timestamp'] = cam_data['timestamp']
                        cam_info['is_key_frame'] = cam_data['is_key_frame']
                        cam_info['height'] = cam_data['height']
                        cam_info['width'] = cam_data['width']
                        cam_info['filename'] = cam_data['filename']
                        cam_info['calibrated_sensor'] = nusc.get(
                            'calibrated_sensor',
                            cam_data['calibrated_sensor_token'])
                        sweeps[j][cam_names[k]] = cam_info
            # Remove empty sweeps.
            for i, sweep in enumerate(sweeps):
                if len(sweep.keys()) == 0:
                    sweeps = sweeps[:i]
                    break
            info['sweeps'] = sweeps
            ann_infos = list()
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
    nusc = NuScenes(version='v1.0-trainval',
                    dataroot='./data/nuScenes/',
                    verbose=True)
    train_scenes = splits.train
    val_scenes = splits.val
    train_infos = generate_info(nusc, train_scenes)
    val_infos = generate_info(nusc, val_scenes)
    mmcv.dump(train_infos, './data/nuScenes/nuscenes_12hz_infos_train.pkl')
    mmcv.dump(val_infos, './data/nuScenes/nuscenes_12hz_infos_val.pkl')


if __name__ == '__main__':
    main()
