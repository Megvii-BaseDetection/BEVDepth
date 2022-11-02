import copy
import os

import numpy as np
import tensorflow as tf
import torch
from waymo_open_dataset import label_pb2
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import breakdown_pb2, metrics_pb2


def build_let_metrics_config():
    let_metric_config = metrics_pb2.Config.LongitudinalErrorTolerantConfig(
        enabled=True,
        sensor_location=metrics_pb2.Config.LongitudinalErrorTolerantConfig.
        Location3D(x=1.43, y=0, z=2.18),
        longitudinal_tolerance_percentage=0.1,  # 10% tolerance.
        min_longitudinal_tolerance_meter=0.5,
    )
    config = metrics_pb2.Config(
        box_type=label_pb2.Label.Box.TYPE_3D,
        matcher_type=metrics_pb2.MatcherProto.TYPE_HUNGARIAN,
        iou_thresholds=[0.0, 0.3, 0.5, 0.5, 0.5],
        score_cutoffs=[i * 0.01 for i in range(100)] + [1.0],
        let_metric_config=let_metric_config)

    config.breakdown_generator_ids.append(breakdown_pb2.Breakdown.OBJECT_TYPE)
    config.difficulties.append(
        metrics_pb2.Difficulty(levels=[label_pb2.Label.LEVEL_2]))
    config.breakdown_generator_ids.append(breakdown_pb2.Breakdown.CAMERA)
    config.difficulties.append(
        metrics_pb2.Difficulty(levels=[label_pb2.Label.LEVEL_2]))
    config.breakdown_generator_ids.append(breakdown_pb2.Breakdown.RANGE)
    config.difficulties.append(
        metrics_pb2.Difficulty(levels=[label_pb2.Label.LEVEL_2]))
    return config


def compute_let_detection_metrics(prediction_frame_id,
                                  prediction_bbox,
                                  prediction_type,
                                  prediction_score,
                                  ground_truth_frame_id,
                                  ground_truth_bbox,
                                  ground_truth_type,
                                  ground_truth_difficulty,
                                  recall_at_precision=None,
                                  name_filter=None,
                                  config=build_let_metrics_config()):
    """Returns dict of metric name to metric values`.

  Notation:
    * M: number of predicted boxes.
    * D: number of box dimensions. The number of box dimensions can be one of
         the following:
           4: Used for boxes with type TYPE_AA_2D (center_x, center_y, length,
              width)
           5: Used for boxes with type TYPE_2D (center_x, center_y, length,
              width, heading).
           7: Used for boxes with type TYPE_3D (center_x, center_y, center_z,
              length, width, height, heading).
    * N: number of ground truth boxes.

  Args:
    prediction_frame_id: [M] int64 tensor that identifies frame for each
      prediction.
    prediction_bbox: [M, D] tensor encoding the
        predicted bounding boxes.
    prediction_type: [M] tensor encoding the object type
        of each prediction.
    prediction_score: [M] tensor encoding the score of each
        prediciton.
    ground_truth_frame_id: [N] int64 tensor that identifies
        frame for each
      ground truth.
    ground_truth_bbox: [N, D] tensor encoding the ground
        truth bounding boxes.
    ground_truth_type: [N] tensor encoding the object type
        of each ground truth.
    ground_truth_difficulty: [N] tensor encoding the difficulty
        level of each
      ground truth.
    config: The metrics config defined in protos/metrics.proto.

  Returns:
    A dictionary of metric names to metrics values.
  """
    num_ground_truths = tf.shape(ground_truth_bbox)[0]
    num_predictions = tf.shape(prediction_bbox)[0]
    ground_truth_speed = tf.zeros((num_ground_truths, 2), tf.float32)
    prediction_overlap_nlz = tf.zeros((num_predictions), tf.bool)

    config_str = config.SerializeToString()
    ap, aph, apl, pr, _, _, _ = py_metrics_ops.detection_metrics(
        prediction_frame_id=tf.cast(prediction_frame_id, tf.int64),
        prediction_bbox=tf.cast(prediction_bbox, tf.float32),
        prediction_type=tf.cast(prediction_type, tf.uint8),
        prediction_score=tf.cast(prediction_score, tf.float32),
        prediction_overlap_nlz=prediction_overlap_nlz,
        ground_truth_frame_id=tf.cast(ground_truth_frame_id, tf.int64),
        ground_truth_bbox=tf.cast(ground_truth_bbox, tf.float32),
        ground_truth_type=tf.cast(ground_truth_type, tf.uint8),
        ground_truth_difficulty=tf.cast(ground_truth_difficulty, tf.uint8),
        ground_truth_speed=ground_truth_speed,
        config=config_str)
    breakdown_names = config_util.get_breakdown_names_from_config(config)
    metric_values = {}
    for i, name in enumerate(breakdown_names):
        if name_filter is not None and name_filter not in name:
            continue
        metric_values['{}/LET-mAP'.format(name)] = ap[i]
        metric_values['{}/LET-mAPH'.format(name)] = aph[i]
        metric_values['{}/LET-mAPL'.format(name)] = apl[i]
    return metric_values


def parse_metrics_objects_binary_files(ground_truths_path, predictions_path):
    with tf.io.gfile.GFile(ground_truths_path, 'rb') as f:
        ground_truth_objects = metrics_pb2.Objects.FromString(f.read())
    with tf.io.gfile.GFile(predictions_path, 'rb') as f:
        predictions_objects = metrics_pb2.Objects.FromString(f.read())
    eval_dict = {
        'prediction_frame_id': [],
        'prediction_bbox': [],
        'prediction_type': [],
        'prediction_score': [],
        'ground_truth_frame_id': [],
        'ground_truth_bbox': [],
        'ground_truth_type': [],
        'ground_truth_difficulty': [],
    }

    # Parse and filter ground truths.
    for obj in ground_truth_objects.objects:
        # Ignore objects that are not in Cameras' FOV.
        if not obj.object.most_visible_camera_name:
            continue
        # Ignore objects that are fully-occluded to cameras.
        if obj.object.num_lidar_points_in_box == 0:
            continue
        # Fill in unknown difficulties.
        if obj.object.detection_difficulty_level == label_pb2.Label.UNKNOWN:
            obj.object.detection_difficulty_level = label_pb2.Label.LEVEL_2
        eval_dict['ground_truth_frame_id'].append(obj.frame_timestamp_micros)
        # Note that we use `camera_synced_box` for evaluation.
        ground_truth_box = obj.object.camera_synced_box
        eval_dict['ground_truth_bbox'].append(
            np.asarray([
                ground_truth_box.center_x,
                ground_truth_box.center_y,
                ground_truth_box.center_z,
                ground_truth_box.length,
                ground_truth_box.width,
                ground_truth_box.height,
                ground_truth_box.heading,
            ], np.float32))
        eval_dict['ground_truth_type'].append(obj.object.type)
        eval_dict['ground_truth_difficulty'].append(
            np.uint8(obj.object.detection_difficulty_level))

    # Parse predictions.
    for obj in predictions_objects.objects:
        eval_dict['prediction_frame_id'].append(obj.frame_timestamp_micros)
        prediction_box = obj.object.box
        eval_dict['prediction_bbox'].append(
            np.asarray([
                prediction_box.center_x,
                prediction_box.center_y,
                prediction_box.center_z,
                prediction_box.length,
                prediction_box.width,
                prediction_box.height,
                prediction_box.heading,
            ], np.float32))
        eval_dict['prediction_type'].append(obj.object.type)
        eval_dict['prediction_score'].append(obj.score)

    for key, value in eval_dict.items():
        eval_dict[key] = tf.stack(value)
    return eval_dict


def convert_numpy_to_torch(x):
    """Convert to torch.tensor if the input is numpy.ndarray.

    Args:
        x (np.ndarray | torch.Tensor): Input data to be check and convert.

    Returns:
        tuple:  If the type of x is np.ndarray, then convert it to
            torch.Tensor and return True. If the type of x is torch.Tensor,
            it returns itself and False.
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x), True
    elif isinstance(x, torch.Tensor):
        return x, False
    else:
        raise TypeError('The type of the input must be np.ndarray'
                        f' or torch.Tensor, but got {type(x)}')


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    The return value is limited into [-offset * period, (1-offset) * period].

    Args:
        val (np.ndarray): The value to be converted.
        offset (float, optional): Offset to set the value range.
            Defaults to 0.5.
        period (float, optional): Period of the value. Defaults to np.pi.

    Returns:
        torch.Tensor: Value in the range of
            [-offset * period, (1-offset) * period]
    """
    val, is_numpy = convert_numpy_to_torch(val)
    res = val - torch.floor(val / period + offset) * period
    return res.numpy() if is_numpy else res


class DetWaymoEvaluator(tf.test.TestCase):
    """Evaluation for 3D detection.

    Args:
        class_names (list[str]): Class to be evaluated.
        iou_thr (list[float]): IoU thresholds of gt and predictions.
            Defaults to (0.3, 0.5, 0.8).
        eval_types (str): Bbox type to be evaluated. Defaults to 'bev'.
    """
    CLASS_NAME_TO_ID = {
        'Vehicle': 1,
        'Pedestrian': 2,
        'Sign': 3,
        'Cyclist': 4,
    }

    def __init__(self,
                 class_names,
                 distance_thresh=100,
                 reserved_digits=4,
                 dump_path='./'):
        super().__init__()
        assert len(class_names) > 0, 'must contain at least one class'
        self.class_names = class_names
        self.distance_thresh = distance_thresh
        self.reserved_digits = reserved_digits
        self.dump_path = dump_path

    def format(self, results, gt_for_eval, dump_path=None):
        assert len(results) == len(gt_for_eval)
        results_for_dump = list()
        for cur_result, cur_gt in zip(results, gt_for_eval):
            # parse predicted results
            labels_class = list()
            bbox3d_filtered = list()
            score_filtered = list()
            for i in range(len(cur_result[0])):
                cur_class_name = cur_result[2][i]
                bbox3d = cur_result[0][i]
                score = cur_result[1][i]
                bbox3d_filtered.append(bbox3d)
                score_filtered.append(score)
                labels_class.append(cur_class_name)
            cur_result_dict = dict()
            cur_result_dict['labels3d'] = labels_class
            cur_result_dict['bboxes3d'] = torch.from_numpy(
                np.stack(bbox3d_filtered)) if len(
                    bbox3d_filtered) else torch.zeros(
                        (0, 7), dtype=torch.float32)
            cur_result_dict['scores3d'] = torch.Tensor(score_filtered) if len(
                bbox3d_filtered) else torch.Tensor([])
            results_for_dump.append(cur_result_dict)
        self.dump_results(copy.deepcopy(results_for_dump),
                          copy.deepcopy(gt_for_eval),
                          os.path.join(self.dump_path, './predictions.bin'))

    def dump_results(self, results, gt_for_eval, dump_path):
        # Generate pred.
        pred_objects = metrics_pb2.Objects()
        for i, pred in enumerate(results):
            bboxes3d = pred['bboxes3d']
            scores3d = pred['scores3d']
            labels3d = pred['labels3d']
            gt = gt_for_eval[i][0]

            for j, bbox3d in enumerate(bboxes3d):
                score = scores3d[j]
                label = labels3d[j]
                pred_object = metrics_pb2.Object()
                pred_object.context_name = gt['scene_name']
                pred_object.frame_timestamp_micros = int(gt['timestamp'])
                # Populating box and score.
                box = label_pb2.Label.Box()
                box.center_x = bbox3d[0]
                box.center_y = bbox3d[1]
                box.center_z = bbox3d[2]
                box.length = bbox3d[3]
                box.width = bbox3d[4]
                box.height = bbox3d[5]
                box.heading = bbox3d[-1]
                pred_object.object.box.CopyFrom(box)
                pred_object.score = score
                # Use correct type.
                pred_object.object.type = self.CLASS_NAME_TO_ID[label]
                pred_objects.objects.append(pred_object)

        with open(dump_path, 'wb') as f:
            f.write(pred_objects.SerializeToString())

    def evaluate(self, prediction_infos, gt_infos):
        self.format(prediction_infos, gt_infos)

        GROUND_TRUTHS_BIN = './data/waymo/v1.4/cam_gt.bin'
        PREDICTIONS_BIN = os.path.join(self.dump_path, 'predictions.bin')
        eval_dict = parse_metrics_objects_binary_files(GROUND_TRUTHS_BIN,
                                                       PREDICTIONS_BIN)
        metrics_dict = compute_let_detection_metrics(**eval_dict)
        for key, value in metrics_dict.items():
            if 'SIGN' in key:
                continue
            print(f'{key:<55}: {value}')
