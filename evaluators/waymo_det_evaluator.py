import numpy as np
import prettytable as pt
import tensorflow as tf
import torch
from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2


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
    WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']
    # Map between class name and waymo results keys.
    NAME_LEVEL1_AP_MAP = {
        'Vehicle': 'OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP',
        'Pedestrian': 'OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP',
        'Sign': 'OBJECT_TYPE_TYPE_SIGN_LEVEL_1/AP',
        'Cyclist': 'OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP'
    }
    NAME_LEVEL2_AP_MAP = {
        'Vehicle': 'OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP',
        'Pedestrian': 'OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/AP',
        'Sign': 'OBJECT_TYPE_TYPE_SIGN_LEVEL_2/AP',
        'Cyclist': 'OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/AP'
    }
    NAME_LEVEL1_APH_MAP = {
        'Vehicle': 'OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/APH',
        'Pedestrian': 'OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/APH',
        'Sign': 'OBJECT_TYPE_TYPE_SIGN_LEVEL_1/APH',
        'Cyclist': 'OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/APH'
    }
    NAME_LEVEL2_APH_MAP = {
        'Vehicle': 'OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/APH',
        'Pedestrian': 'OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/APH',
        'Sign': 'OBJECT_TYPE_TYPE_SIGN_LEVEL_2/APH',
        'Cyclist': 'OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/APH'
    }

    def __init__(self, class_names, distance_thresh=100, reserved_digits=4):
        super().__init__()
        assert len(class_names) > 0, 'must contain at least one class'
        self.class_names = class_names
        self.distance_thresh = distance_thresh
        self.reserved_digits = reserved_digits

    def generate_waymo_type_results(self, infos, class_names, is_gt=False):

        frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty \
            = [], [], [], [], [], []
        for frame_index, info in enumerate(infos):
            img_metas, gt_boxes3d, gt_classes3d = info
            if is_gt:
                box_mask = np.array([n in class_names for n in gt_classes3d],
                                    dtype=np.bool_)
                if 'num_points_in_gt' in img_metas:

                    zero_difficulty_mask = img_metas['difficultys'] == 0
                    img_metas['difficultys'][
                        (img_metas['num_points_in_gt'] > 5)
                        & zero_difficulty_mask] = 1
                    img_metas['difficultys'][
                        (img_metas['num_points_in_gt'] <= 5)
                        & zero_difficulty_mask] = 2
                    nonzero_mask = img_metas['num_points_in_gt'] > 0
                    box_mask = box_mask & nonzero_mask
                else:
                    print('Please provide the num_points_in_gt for evaluating '
                          'on Waymo Dataset '
                          '(If you create Waymo Infos before 20201126, please '
                          're-create the validation infos '
                          'with version 1.2 Waymo dataset to get this '
                          'attribute). SSS of OpenPCDet')
                    raise NotImplementedError

                num_boxes = box_mask.sum()
                gt_classes3d = np.array(gt_classes3d)[box_mask]
                box_name = np.array([
                    self.class_names.index(gt_class3d)
                    for gt_class3d in gt_classes3d
                ])
                difficulty.append(img_metas['difficultys'][box_mask])
                score.append(np.ones(num_boxes))

                boxes3d.append(gt_boxes3d[box_mask])
            else:
                # bboxes3d
                num_boxes = len(info[0])
                difficulty.append([0] * num_boxes)
                # scores3d
                score.append(info[1])
                boxes3d.append(info[0])
                box_name = info[2]

            obj_type.append(box_name)
            frame_id.append(np.array([frame_index] * num_boxes))
            overlap_nlz.append(np.zeros(num_boxes))  # set zero currently

        frame_id = np.concatenate(frame_id).reshape(-1).astype(np.int64)
        boxes3d = np.concatenate(boxes3d, axis=0)
        obj_type = np.concatenate(obj_type).reshape(-1)
        score = np.concatenate(score).reshape(-1)
        overlap_nlz = np.concatenate(overlap_nlz).reshape(-1)
        difficulty = np.concatenate(difficulty).reshape(-1).astype(np.int8)

        boxes3d[:, -1] = limit_period(boxes3d[:, -1],
                                      offset=0.5,
                                      period=np.pi * 2)

        return frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty

    def build_config(self):
        config = metrics_pb2.Config()
        config_text = """
        breakdown_generator_ids: OBJECT_TYPE
        difficulties {
        levels:1
        levels:2
        }
        matcher_type: TYPE_HUNGARIAN
        iou_thresholds: 0.0
        iou_thresholds: 0.7
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        box_type: TYPE_3D
        """

        for x in range(0, 100):
            config.score_cutoffs.append(x * 0.01)
        config.score_cutoffs.append(1.0)

        text_format.Merge(config_text, config)
        return config

    def build_graph(self, graph):
        with graph.as_default():
            self._pd_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._pd_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._pd_score = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_overlap_nlz = tf.compat.v1.placeholder(dtype=tf.bool)

            self._gt_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._gt_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._gt_difficulty = tf.compat.v1.placeholder(dtype=tf.uint8)
            metrics = detection_metrics.get_detection_metric_ops(
                config=self.build_config(),
                prediction_frame_id=self._pd_frame_id,
                prediction_bbox=self._pd_bbox,
                prediction_type=self._pd_type,
                prediction_score=self._pd_score,
                prediction_overlap_nlz=self._pd_overlap_nlz,
                ground_truth_bbox=self._gt_bbox,
                ground_truth_type=self._gt_type,
                ground_truth_frame_id=self._gt_frame_id,
                ground_truth_difficulty=self._gt_difficulty,
            )
            return metrics

    def run_eval_ops(
        self,
        sess,
        graph,
        metrics,
        prediction_frame_id,
        prediction_bbox,
        prediction_type,
        prediction_score,
        prediction_overlap_nlz,
        ground_truth_frame_id,
        ground_truth_bbox,
        ground_truth_type,
        ground_truth_difficulty,
    ):
        sess.run(
            [tf.group([value[1] for value in metrics.values()])],
            feed_dict={
                self._pd_bbox: prediction_bbox,
                self._pd_frame_id: prediction_frame_id,
                self._pd_type: prediction_type,
                self._pd_score: prediction_score,
                self._pd_overlap_nlz: prediction_overlap_nlz,
                self._gt_bbox: ground_truth_bbox,
                self._gt_type: ground_truth_type,
                self._gt_frame_id: ground_truth_frame_id,
                self._gt_difficulty: ground_truth_difficulty,
            },
        )

    def eval_value_ops(self, sess, graph, metrics):
        return {item[0]: sess.run([item[1][0]]) for item in metrics.items()}

    def mask_by_distance(self, distance_thresh, boxes_3d, *args):
        mask = np.linalg.norm(boxes_3d[:, 0:2], axis=1) < distance_thresh + 0.5
        boxes_3d = boxes_3d[mask]
        ret_ans = [boxes_3d]
        for arg in args:
            ret_ans.append(arg[mask])

        return tuple(ret_ans)

    def evaluate(self, prediction_infos, gt_infos):
        print('Start the waymo evaluation...')
        assert len(prediction_infos) == len(
            gt_infos), f'{len(prediction_infos)} vs {len(gt_infos)}'
        tf.compat.v1.disable_eager_execution()
        pd_frameid, pd_boxes3d, pd_type, pd_score, pd_overlap_nlz, _ \
            = self.generate_waymo_type_results(
                prediction_infos, self.class_names, is_gt=False)
        gt_frameid, gt_boxes3d, gt_type, gt_score, gt_overlap_nlz,\
            gt_difficulty = self.generate_waymo_type_results(
                gt_infos, self.class_names, is_gt=True)

        pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz \
            = self.mask_by_distance(
                self.distance_thresh, pd_boxes3d, pd_frameid, pd_type,
                pd_score, pd_overlap_nlz)
        gt_boxes3d, gt_frameid, gt_type, gt_score, gt_difficulty \
            = self.mask_by_distance(
                self.distance_thresh, gt_boxes3d, gt_frameid, gt_type,
                gt_score, gt_difficulty)
        print(f'Number: (pd, {len(pd_boxes3d)}) VS. (gt, {len(gt_boxes3d)})')
        print(f'Level 1: {(gt_difficulty == 1).sum()}, \
                Level2: {(gt_difficulty == 2).sum()})')
        if pd_score.max() > 1:
            # assert pd_score.max() <= 1.0, 'Waymo evaluation
            # only supports normalized scores'
            pd_score = 1 / (1 + np.exp(-pd_score))
            print('Warning: Waymo evaluation only supports normalized scores')
        graph = tf.Graph()
        metrics = self.build_graph(graph)
        with self.test_session(graph=graph) as sess:
            sess.run(tf.compat.v1.initializers.local_variables())
            self.run_eval_ops(
                sess,
                graph,
                metrics,
                pd_frameid,
                pd_boxes3d,
                pd_type,
                pd_score,
                pd_overlap_nlz,
                gt_frameid,
                gt_boxes3d,
                gt_type,
                gt_difficulty,
            )
            with tf.compat.v1.variable_scope('detection_metrics', reuse=True):
                aps = self.eval_value_ops(sess, graph, metrics)
        level1_ap_results = [
            round(aps[self.NAME_LEVEL1_AP_MAP[class_name]][0],
                  self.reserved_digits) for class_name in self.class_names
        ]
        level2_ap_results = [
            round(aps[self.NAME_LEVEL2_AP_MAP[class_name]][0],
                  self.reserved_digits) for class_name in self.class_names
        ]
        level1_aph_results = [
            round(aps[self.NAME_LEVEL1_APH_MAP[class_name]][0],
                  self.reserved_digits) for class_name in self.class_names
        ]
        level2_aph_results = [
            round(aps[self.NAME_LEVEL2_APH_MAP[class_name]][0],
                  self.reserved_digits) for class_name in self.class_names
        ]
        level1_ap_results.append(
            np.mean(level1_ap_results).round(self.reserved_digits))
        level1_ap_results.insert(0, 'level1_ap')
        level2_ap_results.append(
            np.mean(level2_ap_results).round(self.reserved_digits))
        level2_ap_results.insert(0, 'level2_ap')
        level1_aph_results.append(
            np.mean(level1_aph_results).round(self.reserved_digits))
        level1_aph_results.insert(0, 'level1_aph')
        level2_aph_results.append(
            np.mean(level2_aph_results).round(self.reserved_digits))
        level2_aph_results.insert(0, 'level2_aph')
        tb = pt.PrettyTable()
        tb.field_names = ['metrics'] + self.class_names + ['mean']
        tb.add_row(level1_ap_results)
        tb.add_row(level1_aph_results)
        tb.add_row(level2_ap_results)
        tb.add_row(level2_aph_results)
        return '\n' + tb.get_string()
