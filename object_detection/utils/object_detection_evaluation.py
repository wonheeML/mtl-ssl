# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""object_detection_evaluation module.

ObjectDetectionEvaluation is a class which manages ground truth information of a
object detection dataset, and computes frequently used detection metrics such as
Precision, Recall, CorLoc of the provided detection results.
It supports the following operations:
1) Add ground truth information of images sequentially.
2) Add detection result of images sequentially.
3) Evaluate detection metrics on already inserted detection results.
4) Write evaluation result into a pickle file for future processing or
   visualization.

Note: This module operates on numpy boxes and box lists.
"""

import logging
import numpy as np
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from object_detection.utils import metrics
from object_detection.utils import per_image_evaluation
from object_detection.utils import np_box_list

class ObjectDetectionEvaluation(object):
  """Evaluate Object Detection Result."""

  def __init__(self,
               num_groundtruth_classes,
               matching_iou_threshold=0.5,
               nms_type='standard',
               nms_iou_threshold=1.0,
               nms_max_output_boxes=10000,
               soft_nms_sigma=0.5,
               subset_names=('default',)):
    self.per_image_eval = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes,
        matching_iou_threshold,
        nms_type,
        nms_iou_threshold,
        nms_max_output_boxes,
        soft_nms_sigma)
    self.num_class = num_groundtruth_classes
    self.subset_names = subset_names

    self.groundtruth_boxes = {}
    self.groundtruth_class_labels = {}
    self.groundtruth_subset = {s: {} for s in self.subset_names}
    self.num_gt_instances_per_class = {s: np.zeros(self.num_class, dtype=int)
                                       for s in self.subset_names}
    self.num_gt_imgs_per_class = np.zeros(self.num_class, dtype=int)

    self.detection_keys = set()
    self.scores_per_class = {s: [[] for _ in range(self.num_class)]
                             for s in self.subset_names}
    self.tp_fp_labels_per_class = {s: [[] for _ in range(self.num_class)]
                                   for s in self.subset_names}
    self.num_images_correctly_detected_per_class = np.zeros(self.num_class)
    self.average_precision_per_class \
        = {s: np.empty(self.num_class, dtype=float)
           for s in self.subset_names}
    for s in self.subset_names:
      self.average_precision_per_class[s].fill(np.nan)
    self.precisions_per_class = {s: [] for s in self.subset_names}
    self.recalls_per_class = {s: [] for s in self.subset_names}
    self.corloc_per_class = np.ones(self.num_class, dtype=float)

  def clear_groundtruths(self):
    self.groundtruth_boxes = {}
    self.groundtruth_class_labels = {}
    self.groundtruth_subset = {s: {} for s in self.subset_names}
    self.num_gt_instances_per_class = {s: np.zeros(self.num_class, dtype=int)
                                       for s in self.subset_names}
    self.num_gt_imgs_per_class = np.zeros(self.num_class, dtype=int)

  def clear_detections(self):
    self.detection_keys = set()
    self.scores_per_class = {s: [[] for _ in range(self.num_class)]
                             for s in self.subset_names}
    self.tp_fp_labels_per_class = {s: [[] for _ in range(self.num_class)]
                                   for s in self.subset_names}
    self.num_images_correctly_detected_per_class = np.zeros(self.num_class)
    self.average_precision_per_class \
        = {s: np.empty(self.num_class, dtype=float)
           for s in self.subset_names}
    for s in self.subset_names:
      self.average_precision_per_class[s].fill(np.nan)
    self.precisions_per_class = {s: [] for s in self.subset_names}
    self.recalls_per_class = {s: [] for s in self.subset_names}
    self.corloc_per_class = np.ones(self.num_class, dtype=float)

  def add_single_ground_truth_image_info(self,
                                         image_key,
                                         groundtruth_boxes,
                                         groundtruth_class_labels,
                                         groundtruth_subset=None):
    """Add ground truth info of a single image into the evaluation database.

    Args:
      image_key: sha256 key of image content
      groundtruth_boxes: A numpy array of shape [M, 4] representing object box
          coordinates[y_min, x_min, y_max, x_max]
      groundtruth_class_labels: A 1-d numpy array of length M representing class
          labels
      groundtruth_subset: A list of M subset strings, each of which is subset
          names joined with '|'. An object box may belong to multiple subsets.
          If this is not None, ignore groundtruth_is_difficult_list.
    """
    if image_key in self.groundtruth_boxes:
      logging.warn(
        'image %s has already been added to the ground truth database.',
        image_key)
      return

    self.groundtruth_boxes[image_key] = groundtruth_boxes
    self.groundtruth_class_labels[image_key] = groundtruth_class_labels
    num_boxes = groundtruth_boxes.shape[0]

    # determine subset for each setting
    if groundtruth_subset is None:
      groundtruth_subset = ['default'] * num_boxes

    # initialize groundtruth subset of current image
    for subset in self.subset_names:
      self.groundtruth_subset[subset][image_key] \
          = np.zeros((num_boxes,)).astype(np.bool)

    for box_idx, subsets in enumerate(groundtruth_subset):
      for subset in subsets.split('|'):
        if subset == '':
          continue

        if subset not in self.subset_names:
          raise ValueError('%s is not found in subset_names')

        self.groundtruth_subset[subset][image_key][box_idx] = True

    subset_of_current_img = {s: self.groundtruth_subset[s][image_key]
                             for s in self.subset_names}
    self._update_ground_truth_statistics(groundtruth_class_labels,
                                         subset_of_current_img)

  def add_single_detected_image_info(self, image_key, detected_boxes,
                                     detected_scores, detected_class_labels):
    """Add detected result of a single image into the evaluation database.

    Args:
      image_key: sha256 key of image content
      detected_boxes: A numpy array of shape [N, 4] representing detected box
          coordinates[y_min, x_min, y_max, x_max]
      detected_scores: A 1-d numpy array of length N representing classification
          score
      detected_class_labels: A 1-d numpy array of length N representing class
          labels
    Raises:
      ValueError: if detected_boxes, detected_scores and detected_class_labels
                  do not have the same length.
    """
    if (len(detected_boxes) != len(detected_scores) or
        len(detected_boxes) != len(detected_class_labels)):
      raise ValueError('detected_boxes, detected_scores and '
                       'detected_class_labels should all have same lengths. Got'
                       '[%d, %d, %d]' % (len(detected_boxes),
                                         len(detected_scores),
                                         len(detected_class_labels)))

    if image_key in self.detection_keys:
      logging.warn(
          'image %s has already been added to the detection result database',
          image_key)
      return

    self.detection_keys.add(image_key)
    if image_key in self.groundtruth_boxes:
      groundtruth_boxes = self.groundtruth_boxes[image_key]
      groundtruth_class_labels = self.groundtruth_class_labels[image_key]
    else:
      groundtruth_boxes = np.empty(shape=[0, 4], dtype=float)
      groundtruth_class_labels = np.array([], dtype=int)

    is_class_correctly_detected_in_image = 0
    for subset in self.groundtruth_subset:
      if image_key in self.groundtruth_boxes:
        groundtruth_subset = self.groundtruth_subset[subset][image_key]
      else:
        groundtruth_subset = np.array([], dtype=bool)
      scores, tp_fp_labels, is_class_correctly_detected_in_image = (
          self.per_image_eval.compute_object_detection_metrics(
              detected_boxes, detected_scores, detected_class_labels,
              groundtruth_boxes, groundtruth_class_labels,
              ~groundtruth_subset))
      for i in range(self.num_class):
        self.scores_per_class[subset][i].append(scores[i])
        self.tp_fp_labels_per_class[subset][i].append(tp_fp_labels[i])
    self.num_images_correctly_detected_per_class \
        += is_class_correctly_detected_in_image

  def _update_ground_truth_statistics(self, groundtruth_class_labels,
                                      is_subset):
    """Update grouth truth statitistics.

    1. Difficult boxes are ignored when counting the number of ground truth
    instances as done in Pascal VOC devkit.
    2. Difficult boxes are treated as normal boxes when computing CorLoc related
    statistics.

    Args:
      groundtruth_class_labels: An integer numpy array of length M,
          representing M class labels of object instances in ground truth
      is_subset: A dict of boolean numpy arrays, each denoting whether a ground
          truth box belongs to the corresponding subset.
          Its inverse is considered as the difficult
    """
    for class_index in range(self.num_class):
      for subset in self.subset_names:
        num_gt_instances = np.sum(groundtruth_class_labels[is_subset[subset]]
                                  == class_index)
        self.num_gt_instances_per_class[subset][class_index] += num_gt_instances
      if np.any(groundtruth_class_labels == class_index):
        self.num_gt_imgs_per_class[class_index] += 1

  def evaluate(self):
    """Compute evaluation result.

    Returns:
      average_precision_per_class: float numpy array of average precision for
          each class.
      mean_ap: mean average precision of all classes, float scalar
      precisions_per_class: List of precisions, each precision is a float numpy
          array
      recalls_per_class: List of recalls, each recall is a float numpy array
      corloc_per_class: numpy float array
      mean_corloc: Mean CorLoc score for each class, float scalar
    """

    # compute mAP
    mean_ap = {}
    for subset in self.subset_names:
      if (self.num_gt_instances_per_class[subset] == 0).any():
        logging.warning(
            'The following classes in subset %s have no ground truth examples: '
            '%s', subset,
            np.squeeze(np.argwhere(self.num_gt_instances_per_class == 0)))
      for class_index in range(self.num_class):
        if self.num_gt_instances_per_class[subset][class_index] == 0:
          continue
        scores = np.concatenate(self.scores_per_class[subset][class_index])
        tp_fp_labels = np.concatenate(
            self.tp_fp_labels_per_class[subset][class_index])
        precision, recall = metrics.compute_precision_recall(
            scores, tp_fp_labels,
            self.num_gt_instances_per_class[subset][class_index])
        self.precisions_per_class[subset].append(precision)
        self.recalls_per_class[subset].append(recall)
        average_precision = metrics.compute_average_precision(precision, recall)
        self.average_precision_per_class[subset][class_index] = \
            average_precision

      mean_ap[subset] = np.nanmean(self.average_precision_per_class[subset])

    # compute CorLoc
    self.corloc_per_class = metrics.compute_cor_loc(
        self.num_gt_imgs_per_class,
        self.num_images_correctly_detected_per_class)
    mean_corloc = np.nanmean(self.corloc_per_class)

    return (self.average_precision_per_class, mean_ap,
            self.precisions_per_class, self.recalls_per_class,
            self.corloc_per_class, mean_corloc)

  def get_eval_result(self):
    return EvalResult(self.average_precision_per_class,
                      self.precisions_per_class, self.recalls_per_class,
                      self.corloc_per_class)


class CocoEvaluation(ObjectDetectionEvaluation):

  def __init__(self,
               num_groundtruth_classes,
               matching_iou_threshold=0.5,
               nms_type='standard',
               nms_iou_threshold=1.0,
               nms_max_output_boxes=256,
               soft_nms_sigma=0.5):
    super(CocoEvaluation, self).__init__(
        num_groundtruth_classes,
        matching_iou_threshold,
        nms_type,
        nms_iou_threshold,
        nms_max_output_boxes,
        soft_nms_sigma)

    self.detection_result = np.zeros((0), dtype=np.float64).reshape((0,7))
    self.gt_result = []
    self.max_detections_per_image = 100


  def add_single_detected_image_info(self, image_key, detected_boxes,
                                     detected_scores, detected_class_labels):
    """Add detected result of a single image into the evaluation database.

    Args:
      image_key: sha256 key of image content
      detected_boxes: A numpy array of shape [N, 4] representing detected box
          coordinates[y_min, x_min, y_max, x_max]
      detected_scores: A 1-d numpy array of length N representing classification
          score
      detected_class_labels: A 1-d numpy array of length N representing class
          labels
    Raises:
      ValueError: if detected_boxes, detected_scores and detected_class_labels
                  do not have the same length.
    """
    if (len(detected_boxes) != len(detected_scores) or
        len(detected_boxes) != len(detected_class_labels)):
      raise ValueError('detected_boxes, detected_scores and '
                       'detected_class_labels should all have same lengths. Got'
                       '[%d, %d, %d]' % (len(detected_boxes),
                                         len(detected_scores),
                                         len(detected_class_labels)))

    detected_boxes, detected_scores, detected_class_labels = (
      self.per_image_eval._remove_invalid_boxes(detected_boxes, detected_scores, detected_class_labels))

    category_id_list = np.zeros(0)
    bbox_list = np.zeros((0,0)).reshape(0, 4)
    score_list = np.zeros(0)
    for i in range(self.num_class):
      detected_boxes_at_ith_class = detected_boxes[(detected_class_labels == i), :]
      detected_scores_at_ith_class = detected_scores[detected_class_labels == i]

      if len(detected_scores_at_ith_class) == 0:
        continue

      detected_boxlist = np_box_list.BoxList(detected_boxes_at_ith_class)
      detected_boxlist.add_field('scores', detected_scores_at_ith_class)

      kwargs = {}
      detected_boxlist = self.per_image_eval.nms_fn(boxlist=detected_boxlist, **kwargs)

      boxes = detected_boxlist.get_field('boxes')
      scores = detected_boxlist.get_field('scores')
      categories = np.array([i+1] * len(boxes), dtype=np.int32)

      # [x,y,width,height]
      for idx, box in enumerate(boxes):
        boxes[idx] = [box[1], box[0], box[3] - box[1], box[2] - box[0]]

      category_id_list = np.hstack([category_id_list, categories])
      bbox_list = np.vstack([bbox_list, boxes])
      score_list = np.hstack([score_list, scores])

    dec_index = list(reversed(sorted(range(len(score_list)), key=lambda k: score_list[k])))

    category_id_list = [category_id_list[idx] for idx in dec_index]
    bbox_list = [bbox_list[idx] for idx in dec_index]
    score_list = [score_list[idx] for idx in dec_index]

    if len(score_list) > self.max_detections_per_image:
      category_id_list = category_id_list[0:100]
      bbox_list = bbox_list[0:100]
      score_list = score_list[0:100]

    detection_result = np.hstack((
      np.expand_dims([image_key] * len(category_id_list), axis=1),
      np.array(bbox_list),
      np.expand_dims(score_list, axis=1),
      np.expand_dims(category_id_list, axis=1)
    ))

    self.detection_result = np.vstack((self.detection_result, detection_result))

  def evaluate(self, eval_cat_index, eval_ann_filename):
    """Compute evaluation result.

    Returns:
      metrics: COCO evaluation metriccs per class
    """
    if not os.path.exists(eval_ann_filename):
      logging.warn('%s does not exists: create tf record for val', eval_ann_filename)
      return None

    cocoGt = COCO(eval_ann_filename)
    cocoDt = cocoGt.loadRes(self.detection_result)

    # running evaluation
    annType = 'bbox'
    cocoEval = COCOeval(cocoGt, cocoDt, annType)

    coco_metrics = dict()

    # total class
    cocoEval.params.imgIds = cocoGt.getImgIds()
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    coco_metrics[0] = cocoEval.stats

    # per class
    for cat_id in cocoGt.cats.keys():
      if not cat_id in eval_cat_index:
        continue
      cocoEval.params.catIds = cat_id
      cocoEval.evaluate()
      cocoEval.accumulate()
      cocoEval.summarize()
      coco_metrics[cat_id] = cocoEval.stats

    return coco_metrics

class EvalResult(object):

  def __init__(self, average_precisions, precisions, recalls, all_corloc):
    self.precisions = precisions
    self.recalls = recalls
    self.all_corloc = all_corloc
    self.average_precisions = average_precisions
