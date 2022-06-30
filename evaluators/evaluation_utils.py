# flake8: noqa
import json
import math
import os
from multiprocessing import Process, Queue
from typing import Dict, List

import mmcv
import numpy as np
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from tqdm import tqdm


def box3d_to_nuscenesbox(
    meta_info: NuScenes,
    box_3d: List[float],
    token: str,
) -> Dict:
    """
    convert once_type box to nuscenes_type box
    parameter:
        - meta_info: meta info of Nuscenes dataset. In order to get token info.
        - box_3d: once_type box [x, y, z, dx, dy, dz, heading] w/o [vx, vy]
            (7d or 9d).
        - sample_data_token: the sample token of box_3d. In order to get
            sensor info.
    return:
        - nuscenesbox: nuscenes_type box instance
    """
    nusc = meta_info

    # extract key infos from box_3d
    translation = box_3d[:3]
    size = np.array(box_3d[3:6])[[1, 0, 2]].tolist()  # lwh == > wlh
    rot = box_3d[6]
    if len(box_3d) == 9:
        velocity = tuple(box_3d[7:9] + [0])
    else:
        velocity = (np.nan, np.nan, np.nan)

    # sensor pose box
    nuscenesbox = Box(
        center=translation,
        size=size,
        orientation=Quaternion(math.cos(rot / 2), 0, 0,
                               math.sin(rot / 2)),  # convert rot to Quaternion
        velocity=velocity,
    )

    # sensor pose box => ego pose box
    sample = nusc.get('sample', token)
    ref_chan = 'LIDAR_TOP'
    sample_data_token = sample['data'][ref_chan]

    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    nuscenesbox.rotate(Quaternion(cs_record['rotation']))
    nuscenesbox.translate(np.array(cs_record['translation']))
    nuscenesbox.rotate(Quaternion(pose_record['rotation']))
    nuscenesbox.translate(np.array(pose_record['translation']))
    return nuscenesbox


def generate_submission_results(
    meta_info: NuScenes,
    gt: Dict,
    dt: Dict,
    result_dir: str,
    meta_type_list: List = ['use_lidar'],
    num_workers: int = 16,
) -> Dict:
    """
    generate submission results from once_type det infos(result.pkl)
    params:
        - meta info: meta info of Nuscenes dataset. In order to get token info.
        - gt: once_type gt infos.
        - dt: once_type dt infos.
        - result_dir: where submission json file saved.
        - meta_type_list: submission meta type.
            ["use_camera", "use_lidar", "use_radar", "use_map",
            "use_external"].
        - num_workers: num of multiprocessing workers
    return:
        - submit_json: submission results json

    ===============================
        Nuscenes submission format:

        submission {
            "meta": {
                "use_camera":   <bool>          -- Whether this submission uses camera data as an input.
                "use_lidar":    <bool>          -- Whether this submission uses lidar data as an input.
                "use_radar":    <bool>          -- Whether this submission uses radar data as an input.
                "use_map":      <bool>          -- Whether this submission uses map data as an input.
                "use_external": <bool>          -- Whether this submission uses external data as an input.
            },
            "results": {
                sample_token <str>: List[sample_result] -- Maps each sample_token to a list of sample_results.
            }
        }

        sample_result {
            "sample_token":       <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
            "translation":        <float> [3]   -- Estimated bounding box location in m in the global frame: center_x, center_y, center_z.
            "size":               <float> [3]   -- Estimated bounding box size in m: width, length, height.
            "rotation":           <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
            "velocity":           <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
            "detection_name":     <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
            "detection_score":    <float>       -- Object prediction score between 0 and 1 for the class identified by detection_name.
            "attribute_name":     <str>         -- Name of the predicted attribute or empty string for classes without attributes.
                                                See table below for valid attributes for each class, e.g. cycle.with_rider.
                                                Attributes are ignored for classes without attributes.
                                                There are a few cases (0.4%) where attributes are missing also for classes
                                                that should have them. We ignore the predicted attributes for these cases.
        }
    ===============================
    """
    def worker(split_records, meta_info, result_queue):
        gt, dt = split_records
        for i in range(len(dt)):
            token = gt[i]['token']
            names, scores, boxes_3d = dt[i]['name'], dt[i]['score'], dt[i][
                'boxes_3d']
            assert len(names) == len(scores) == len(boxes_3d)
            num_dt = len(boxes_3d)
            dt_boxes = []
            for box_id in range(num_dt):
                box_item = {
                    'sample_token': token,
                    'detection_name': str(names[box_id]),
                    'detection_score': float(scores[box_id]),
                    'attribute_name': '',  # TODO: deal with attribute name
                }
                box = box3d_to_nuscenesbox(meta_info,
                                           boxes_3d[box_id].tolist(), token)
                box_info = {
                    'translation': box.center.tolist(),
                    'size': box.wlh.tolist(),
                    'rotation': list(box.orientation),
                    'velocity': box.velocity.tolist()[:2],
                }
                box_item.update(box_info)
                dt_boxes.append(box_item)
            result_queue.put({token: dt_boxes})

    nr_records = len(dt)
    pbar = tqdm(total=nr_records)

    nr_split = math.ceil(nr_records / num_workers)
    result_queue = Queue(10000)
    procs = []
    dt_res_json = {}

    print('Generating submission results...')
    for i in range(num_workers):
        start = i * nr_split
        end = min(start + nr_split, nr_records)
        split_records = (gt[start:end], dt[start:end])
        proc = Process(target=worker,
                       args=(split_records, meta_info, result_queue))
        print('process:%d, start:%d, end:%d' % (i, start, end))
        proc.start()
        procs.append(proc)

    for i in range(nr_records):
        dt_res_json.update(result_queue.get())
        pbar.update(1)

    for p in procs:
        p.join()

    submit_json = {
        'meta': {
            'use_camera': 'use_camera' in meta_type_list,
            'use_lidar': 'use_lidar' in meta_type_list,
            'use_radar': 'use_radar' in meta_type_list,
            'use_map': 'use_map' in meta_type_list,
            'use_external': 'use_external' in meta_type_list,
        },
        'results': dt_res_json,
    }

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    mmcv.dump(submit_json, os.path.join(result_dir, 'nuscenes_results.json'))
    return submit_json


def get_evaluation_results(nusc_meta_info: NuScenes,
                           result_path: str,
                           output_dir: str,
                           config_path: str = '',
                           eval_set: str = 'val',
                           verbose: bool = False,
                           plot_examples: int = 0,
                           render_curves: bool = False,
                           **kwargs) -> Dict:

    if config_path == '':
        cfg = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg = DetectionConfig.deserialize(json.load(_f))

    print('Loading Nuscenes ground truths...')
    nusc_eval = DetectionEval(nusc_meta_info,
                              config=cfg,
                              result_path=result_path,
                              eval_set=eval_set,
                              output_dir=output_dir,
                              verbose=verbose)

    print('Evaluation starts...')
    eval_res = nusc_eval.main(plot_examples=plot_examples,
                              render_curves=render_curves)

    return eval_res
