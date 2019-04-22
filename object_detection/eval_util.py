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

"""Common functions for repeatedly evaluating a checkpoint.
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import copy
import logging
import os
import time
import pickle
import scipy
import json
import math

import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import object_detection_evaluation
from object_detection.utils import visualization_utils as vis_utils
from object_detection.utils import mtl_util
from object_detection.utils import np_box_ops

from global_utils.custom_utils import log
from itertools import chain
import shutil

slim = tf.contrib.slim


def get_string_list_for_nms(nms_type, nms_iou_threshold, soft_nms_sigma):
  if nms_type == 'standard' or nms_type == 'soft-linear':
    str_list = [nms_type, str(nms_iou_threshold)]
  elif nms_type == 'soft-gaussian':
    str_list = [nms_type, str(soft_nms_sigma)]
  else:
    raise ValueError('Cannot identify NMS type.')
  return str_list

def write_metrics(metrics, global_step, summary_dir, eval_config=None):
  """Write metrics to a summary directory.

  Args:
    metrics: A dictionary containing metric names and values.
    global_step: Global step at which the metrics are computed.
    summary_dir: Directory to write tensorflow summaries to.
  """
  log.infov('Writing metrics to tf summary.')
  summary_writer = tf.summary.FileWriter(summary_dir)
  if eval_config is not None:
    str_list = get_string_list_for_nms(eval_config.nms_type,
                                       eval_config.nms_threshold,
                                       eval_config.soft_nms_sigma)
    nms_str = ' / '.join(str_list).upper()
    log.warn('[Post Process: %s]', nms_str)
  for key in sorted(metrics):
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=key, simple_value=metrics[key]),
    ])
    summary_writer.add_summary(summary, global_step)
    log.info('%s: %f', key, metrics[key])
  summary_writer.close()
  log.infov('Metrics written to tf summary.')


def compute_pr_curve(precisions_per_class, recalls_per_class):
  """Compute macro averaged precision and recall curve for total class

  1) remove duplicate values from recall and leave only the highest precision
  2) scan the recall in decreasing order and replace the precision with the historical max
  3) collect recall values for each class and eliminate redundancies, use it for threshold
  4) calculate the macro averaged precision by lowering the threshold one by one

  Args:
    precisions_per_class: precision list of each class
    recalls_per_class: recall list of each class

Raises:
  ValueError: if the numbers of classes of precisions and recalls are not equal

  Returns:
    precisions, recalls
"""
  # exception handling
  class_num = len(precisions_per_class)
  class_num1 = len(recalls_per_class)

  if class_num != class_num1:
    raise ValueError('precision/recall number of class not matched')

  for class_ind in xrange(class_num):
    num_precision = len(precisions_per_class[class_ind])
    num_recall = len(recalls_per_class[class_ind])
    if num_precision != num_recall:
      raise ValueError('precision/recall number of class not matched')

  precision_new = []
  recall_new = []
  for precision, recall in zip(precisions_per_class, recalls_per_class):
    if precision is None or recall is None:
      continue
    if len(precision) <= 1 or len(recall) <= 1:
      continue

    # 1) remove duplicate values from recall and leave only the highest precision
    precision_max = [precision[0]]
    recall_no_dup = [recall[0]]
    for index in xrange(1, len(precision)):
      if recall[index] != recall_no_dup[-1]:
        precision_max.append(precision[index])
        recall_no_dup.append(recall[index])
      else:
        precision_max[-1] = max(precision_max[-1], precision[index])

    # 2) scan the recall in decreasing order and replace the precision with the historical max
    max_val = 0
    for index in reversed(xrange(len(precision_max))):
      max_val = max(precision_max[index], max_val)
      precision_max[index] = max_val

    precision_new.append(precision_max)
    recall_new.append(recall_no_dup)


  # 3) collect recall values for each class and eliminate redundancies, use it for threshold
  recall_total = list(set(chain.from_iterable(recall_new)))
  recall_total.sort(reverse=True)


  # 4) calculate the macro averaged precision by lowering the threshold one by one
  precisions = []
  recalls = []
  recall_index = [len(recall) for recall in recall_new]
  class_num = len(precision_new)

  for recall_th in recall_total:
    precision_averaged = 0.0

    for i in xrange(class_num):
      recall = recall_new[i]
      precision = precision_new[i]

      while recall_index[i] > 0 and recall[recall_index[i]-1] >= recall_th:
        recall_index[i] -= 1

      if recall_index[i] < len(recall):
        precision_averaged += precision[recall_index[i]]

    if class_num:
      precision_averaged /= class_num

    precisions.append(precision_averaged)
    recalls.append(recall_th)

  return precisions, recalls


def visualize_pr_curve(per_class_precisions, per_class_recalls, global_step, eval_dir,
                       nms_type, nms_iou_threshold, soft_nms_sigma):
  """Visualizes pr curve and writes visualizations to image summaries.

  Args:
    per_class_precisions: precision list of each class
    per_class_recalls: recall list of each class

  """

  subset_keys = per_class_precisions.keys()
  for key in subset_keys:
    precisions_per_class = per_class_precisions[key]
    recalls_per_class = per_class_recalls[key]
    precisions, recalls = compute_pr_curve(precisions_per_class, recalls_per_class)

    # file save
    nms_str = '_'.join(get_string_list_for_nms(nms_type,
                                               nms_iou_threshold,
                                               soft_nms_sigma))

    filename = os.path.join(eval_dir,  'pr_curve_' + key + '_' + nms_str + '.txt')
    # with open(filename, 'wb') as fp:  # Pickling
    #   pickle.dump([precisions, recalls], fp)

    with open(filename, 'w') as f:
      for p,r in zip(precisions, recalls):
        f.write('%f\t%f\n'%(p,r))


    # Unpickling & plot pr curve
    # with open(filename, 'rb') as fp:  # Unpickling
    #   [precisions, recalls] = pickle.load(fp)
    # plt.plot(recalls, precisions, ls='-', lw=1.0)

    # get the figure contents as RGB pixel values
    fig = plt.figure(figsize=[8, 8])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.plot(recalls, precisions, ls='-', lw=1.0)
    fig.canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    tag = 'PRcurve_' + key + '_'
    summary = tf.Summary(value=[
      tf.Summary.Value(tag=tag, image=tf.Summary.Image(
        encoded_image_string=vis_utils.encode_image_array_as_png_str(image)))
    ])
    summary_writer = tf.summary.FileWriter(eval_dir)
    summary_writer.add_summary(summary, global_step)
    summary_writer.close()
    time.sleep(1)


def evaluate_detection_results_pascal_voc(result_lists,
                                          categories,
                                          label_id_offset=1,
                                          iou_thres=0.5,
                                          corloc_summary=False,
                                          nms_type='standard',
                                          nms_thres=1.0,
                                          soft_nms_sigma=0.5,
                                          global_step=0,
                                          eval_dir='',
                                          eval_config=None):
  """Computes Pascal VOC detection metrics given groundtruth and detections.

  This function computes Pascal VOC metrics. This function by default
  takes detections and groundtruth boxes encoded in result_lists and writes
  evaluation results to tf summaries which can be viewed on tensorboard.

  Args:
    result_lists: a dictionary holding lists of groundtruth and detection
      data corresponding to each image being evaluated.  The following keys
      are required:
        'image_id': a list of string ids
        'detection_boxes': a list of float32 numpy arrays of shape [N, 4]
        'detection_scores': a list of float32 numpy arrays of shape [N]
        'detection_classes': a list of int32 numpy arrays of shape [N]
        'groundtruth_boxes': a list of float32 numpy arrays of shape [M, 4]
        'groundtruth_classes': a list of int32 numpy arrays of shape [M]
      and the remaining fields below are optional:
        'difficult': a list of boolean arrays of shape [M] indicating the
          difficulty of groundtruth boxes. Some datasets like PASCAL VOC provide
          this information and it is used to remove difficult examples from eval
          in order to not penalize the models on them.
      Note that it is okay to have additional fields in result_lists --- they
      are simply ignored.
    categories: a list of dictionaries representing all possible categories.
      Each dict in this list has the following keys:
          'id': (required) an integer id uniquely identifying this category
          'name': (required) string representing category name
            e.g., 'cat', 'dog', 'pizza'
    label_id_offset: an integer offset for the label space.
    iou_thres: float determining the IoU threshold at which a box is considered
        correct. Defaults to the standard 0.5.
    corloc_summary: boolean. If True, also outputs CorLoc metrics.
    nms_type: type of NMS (standard|soft-linear|soft_gaussian)
    nms_thres: iou threshold for non maximum suppression.
    soft_nms_sigma: Soft NMS sigma.

  Returns:
    A dictionary of metric names to scalar values.

  Raises:
    ValueError: if the set of keys in result_lists is not a superset of the
      expected list of keys.  Unexpected keys are ignored.
    ValueError: if the lists in result_lists have inconsistent sizes.
  """
  # check for expected keys in result_lists
  expected_keys = [
      'detection_boxes', 'detection_scores', 'detection_classes', 'image_id'
  ]
  expected_keys += ['groundtruth_boxes', 'groundtruth_classes']
  if not set(expected_keys).issubset(set(result_lists.keys())):
    raise ValueError('result_lists does not have expected key set.')
  num_results = len(result_lists[expected_keys[0]])
  for key in expected_keys:
    if len(result_lists[key]) != num_results:
      raise ValueError('Inconsistent list sizes in result_lists')

  if 'groundtruth_subset' in result_lists.keys():
    subset_list = result_lists['groundtruth_subset']
    subset_names = set()
    for subset in subset_list:
      for subset2 in subset:
        for subset3 in subset2.split('|'):
          subset_names.add(subset3)
    if not subset_names:
      subset_names.add('default')

  # Pascal VOC evaluator assumes foreground index starts from zero.
  categories = copy.deepcopy(categories)
  for idx in range(len(categories)):
    categories[idx]['id'] -= label_id_offset

  # num_classes (maybe encoded as categories)
  num_classes = max([cat['id'] for cat in categories]) + 1
  log.infov('Computing Pascal VOC metrics on results.')
  if all(image_id.isdigit() for image_id in result_lists['image_id']):
    image_ids = [int(image_id) for image_id in result_lists['image_id']]
  else:
    image_ids = range(num_results)

  evaluator = object_detection_evaluation.ObjectDetectionEvaluation(
      num_classes,
      matching_iou_threshold=iou_thres,
      nms_type=nms_type,
      nms_iou_threshold=nms_thres,
      soft_nms_sigma=soft_nms_sigma,
      subset_names=subset_names
  )

  difficult_lists = None
  if 'difficult' in result_lists and result_lists['difficult']:
    difficult_lists = result_lists['difficult']

  for idx, image_id in enumerate(image_ids):
    subset = None
    if len(result_lists['groundtruth_subset']) > 0 and \
        result_lists['groundtruth_subset'][idx].shape[0] \
        == result_lists['groundtruth_boxes'][idx].shape[0]:
      subset = result_lists['groundtruth_subset'][idx]

    if difficult_lists is not None \
        and difficult_lists[idx].size:
      # build subset using difficult
      difficult = difficult_lists[idx].astype(np.bool)
      if subset is None:
        subset = np.where(difficult, '', 'default')
      else:
        subset = np.where(difficult, '', subset)

    evaluator.add_single_ground_truth_image_info(
        image_id, result_lists['groundtruth_boxes'][idx],
        result_lists['groundtruth_classes'][idx] - label_id_offset,
        subset)
    evaluator.add_single_detected_image_info(
        image_id, result_lists['detection_boxes'][idx],
        result_lists['detection_scores'][idx],
        result_lists['detection_classes'][idx] - label_id_offset)

    if idx%500 == 0:
      log.infov('idx(%d)/total(%d)'%(idx, len(image_ids)))

  per_class_ap, mean_ap, per_class_precisions, per_class_recalls, per_class_corloc, mean_corloc = (
      evaluator.evaluate())

  visualize_pr_curve(per_class_precisions, per_class_recalls, global_step, eval_dir,
                     nms_type = nms_type,
                     nms_iou_threshold = nms_thres,
                     soft_nms_sigma = soft_nms_sigma)

  metrics = {'Subset {:10} mAP@{}IOU'.format(s, iou_thres): mean_ap[s]
             for s in mean_ap}
  category_index = label_map_util.create_category_index(categories)
  for subset in per_class_ap:
    for idx in range(per_class_ap[subset].size):
      if idx in category_index:
        display_name = ('Subset {:10} mAP@{}IOU/{}'
                        .format(subset, iou_thres, category_index[idx]['name']))
        metrics[display_name] = per_class_ap[subset][idx]

  if corloc_summary:
    metrics['CorLoc/CorLoc@{}IOU'.format(iou_thres)] = mean_corloc
    for idx in range(per_class_corloc.size):
      if idx in category_index:
        display_name = (
            'PerformanceByCategory/CorLoc@{}IOU/{}'.format(
                iou_thres, category_index[idx]['name']))
        metrics[display_name] = per_class_corloc[idx]
  return metrics


def evaluate_detection_results_coco(result_lists,
                                          categories,
                                          label_id_offset=1,
                                          iou_thres=0.5,
                                          corloc_summary=False,
                                          nms_type='standard',
                                          nms_thres=1.0,
                                          soft_nms_sigma=0.5,
                                          global_step=0,
                                          eval_dir='',
                                          eval_config=None):
  """Computes MS-COCO detection metrics given groundtruth and detections.

  This function computes MS-COCO metrics. This function by default
  takes detections and groundtruth boxes encoded in result_lists and writes
  evaluation results to tf summaries which can be viewed on tensorboard.

  Args:
    result_lists: a dictionary holding lists of groundtruth and detection
      data corresponding to each image being evaluated.  The following keys
      are required:
        'image_id': a list of string ids
        'detection_boxes': a list of float32 numpy arrays of shape [N, 4]
        'detection_scores': a list of float32 numpy arrays of shape [N]
        'detection_classes': a list of int32 numpy arrays of shape [N]
        'groundtruth_boxes': a list of float32 numpy arrays of shape [M, 4]
        'groundtruth_classes': a list of int32 numpy arrays of shape [M]
      and the remaining fields below are optional:
        'difficult': a list of boolean arrays of shape [M] indicating the
          difficulty of groundtruth boxes. Some datasets like PASCAL VOC provide
          this information and it is used to remove difficult examples from eval
          in order to not penalize the models on them.
      Note that it is okay to have additional fields in result_lists --- they
      are simply ignored.
    categories: a list of dictionaries representing all possible categories.
      Each dict in this list has the following keys:
          'id': (required) an integer id uniquely identifying this category
          'name': (required) string representing category name
            e.g., 'cat', 'dog', 'pizza'
    label_id_offset: an integer offset for the label space.
    iou_thres: float determining the IoU threshold at which a box is considered
        correct. Defaults to the standard 0.5.
    corloc_summary: boolean. If True, also outputs CorLoc metrics.
    nms_type: type of NMS (standard|soft-linear|soft_gaussian)
    nms_thres: iou threshold for non maximum suppression.
    soft_nms_sigma: Soft NMS sigma.

  Returns:
    A dictionary of metric names to scalar values.

  Raises:
    ValueError: if the set of keys in result_lists is not a superset of the
      expected list of keys.  Unexpected keys are ignored.
    ValueError: if the lists in result_lists have inconsistent sizes.
  """
  # check for expected keys in result_lists
  expected_keys = [
      'detection_boxes', 'detection_scores', 'detection_classes', 'image_id'
  ]
  expected_keys += ['groundtruth_boxes', 'groundtruth_classes']
  if not set(expected_keys).issubset(set(result_lists.keys())):
    raise ValueError('result_lists does not have expected key set.')
  num_results = len(result_lists[expected_keys[0]])
  for key in expected_keys:
    if len(result_lists[key]) != num_results:
      raise ValueError('Inconsistent list sizes in result_lists')

  categories = copy.deepcopy(categories)
  for idx in range(len(categories)):
    categories[idx]['id'] -= label_id_offset

  # num_classes (maybe encoded as categories)
  num_classes = max([cat['id'] for cat in categories]) + 1
  log.infov('Computing COCO metrics on results.')
  if all(image_id.isdigit() for image_id in result_lists['image_id']):
    image_ids = [int(image_id) for image_id in result_lists['image_id']]
  else:
    image_ids = range(num_results)

  evaluator = object_detection_evaluation.CocoEvaluation(
      num_classes,
      matching_iou_threshold=iou_thres,
      nms_type=nms_type,
      nms_iou_threshold=nms_thres,
      soft_nms_sigma=soft_nms_sigma
  )

  for idx, image_id in enumerate(image_ids):
    evaluator.add_single_ground_truth_image_info(
        image_id, result_lists['groundtruth_boxes'][idx],
        result_lists['groundtruth_classes'][idx] - label_id_offset)
    evaluator.add_single_detected_image_info(
        image_id, result_lists['detection_boxes'][idx],
        result_lists['detection_scores'][idx],
        result_lists['detection_classes'][idx] - label_id_offset)

  metric_names = ['AP', 'AP_IoU50', 'AP_IoU75',
                  'AP_small', 'AP_medium', 'AP_large',
                  'AR_max1', 'AR_max10', 'AR_max100',
                  'AR_small', 'AR_medium', 'AR_large']
  metrics = dict()

  if eval_config is None:
    eval_metric_index = [0]
    eval_class_type = 1
    eval_ann_filename = '../data/mscoco/annotations/instances_eval2014.json'
  else:
    eval_metric_index = eval_config.coco_eval_options.eval_metric_index
    eval_class_type = eval_config.coco_eval_options.eval_class_type
    eval_ann_filename = eval_config.coco_eval_options.eval_ann_filename


  if len(eval_metric_index) == 0:
    eval_metric_index = [0]

  if eval_ann_filename == '':
    eval_ann_filename = '../data/mscoco/annotations/instances_eval2014.json'

  if min(eval_metric_index) < 0 or max(eval_metric_index) > 11:
    raise ValueError('eval_metric_index')
    return metrics

  if eval_class_type < 0 or eval_class_type > 1:
    raise ValueError('eval_class_type')
    return metrics

  if eval_class_type == 0:
    eval_cat_index = [0]
  else:
    eval_cat_index = range(num_classes+1)

  coco_metrics = evaluator.evaluate(eval_cat_index, eval_ann_filename) # coco_metrics[num_classes+1(all)][12]

  if coco_metrics is None:
    return metrics

  category_index = label_map_util.create_category_index(categories)
  for cat_id in range(num_classes+1):

    if cat_id == 0:
      class_name = 'All'
    elif cat_id-1 in category_index.keys():
      class_name = category_index[cat_id-1]['name']
    else:
      continue

    if not cat_id in eval_cat_index:
      continue

    for metric_id, metric_name in enumerate(metric_names):
      if not metric_id in eval_metric_index:
        continue

      display_name = 'COCO_Eval/%s/%s'%(class_name, metric_name)
      metrics[display_name] = coco_metrics[cat_id][metric_id]

  return metrics


# TODO: Add tests.
def visualize_detection_results(result_dict,
                                tag,
                                global_step,
                                categories,
                                summary_dir='',
                                export_dir='',
                                agnostic_mode=False,
                                show_groundtruth=False,
                                min_score_thresh=.5,
                                max_num_predictions=20):
  """Visualizes detection results and writes visualizations to image summaries.

  This function visualizes an image with its detected bounding boxes and writes
  to image summaries which can be viewed on tensorboard.  It optionally also
  writes images to a directory. In the case of missing entry in the label map,
  unknown class name in the visualization is shown as "N/A".

  Args:
    result_dict: a dictionary holding groundtruth and detection
      data corresponding to each image being evaluated.  The following keys
      are required:
        'original_image': a numpy array representing the image with shape
          [1, height, width, 3]
        'detection_boxes': a numpy array of shape [N, 4]
        'detection_scores': a numpy array of shape [N]
        'detection_classes': a numpy array of shape [N]
      The following keys are optional:
        'groundtruth_boxes': a numpy array of shape [N, 4]
        'groundtruth_keypoints': a numpy array of shape [N, num_keypoints, 2]
      Detections are assumed to be provided in decreasing order of score and for
      display, and we assume that scores are probabilities between 0 and 1.
    tag: tensorboard tag (string) to associate with image.
    global_step: global step at which the visualization are generated.
    categories: a list of dictionaries representing all possible categories.
      Each dict in this list has the following keys:
          'id': (required) an integer id uniquely identifying this category
          'name': (required) string representing category name
            e.g., 'cat', 'dog', 'pizza'
          'supercategory': (optional) string representing the supercategory
            e.g., 'animal', 'vehicle', 'food', etc
    summary_dir: the output directory to which the image summaries are written.
    export_dir: the output directory to which images are written.  If this is
      empty (default), then images are not exported.
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.
    show_groundtruth: boolean (default: False) controlling whether to show
      groundtruth boxes in addition to detected boxes
    min_score_thresh: minimum score threshold for a box to be visualized
    max_num_predictions: maximum number of detections to visualize
  Raises:
    ValueError: if result_dict does not contain the expected keys (i.e.,
      'original_image', 'detection_boxes', 'detection_scores',
      'detection_classes')
  """
  if not set([
      'original_image', 'detection_boxes', 'detection_scores',
      'detection_classes'
  ]).issubset(set(result_dict.keys())):
    raise ValueError('result_dict does not contain all expected keys.')
  if show_groundtruth and 'groundtruth_boxes' not in result_dict:
    raise ValueError('If show_groundtruth is enabled, result_dict must contain '
                     'groundtruth_boxes.')
  log.infov('Creating detection visualizations.')
  category_index = label_map_util.create_category_index(categories)

  image = np.squeeze(result_dict['original_image'], axis=0)
  detection_boxes = result_dict['detection_boxes']
  detection_scores = result_dict['detection_scores']
  detection_classes = np.int32((result_dict['detection_classes']))
  detection_keypoints = result_dict.get('detection_keypoints', None)
  detection_masks = result_dict.get('detection_masks', None)

  # Plot groundtruth underneath detections
  if show_groundtruth:
    groundtruth_boxes = result_dict['groundtruth_boxes']

    boxes_area = (groundtruth_boxes[:, 2] - groundtruth_boxes[:, 0]) * (groundtruth_boxes[:,3] - groundtruth_boxes[:,1])
    area_index_total = np.argsort(boxes_area) # ascending order
    groundtruth_boxes_ordered = np.array([groundtruth_boxes[i] for i in area_index_total])

    groundtruth_keypoints = result_dict.get('groundtruth_keypoints', None)
    vis_utils.visualize_boxes_and_labels_on_image_array(
        image,
        groundtruth_boxes_ordered,
        None,
        None,
        category_index,
        keypoints=groundtruth_keypoints,
        use_normalized_coordinates=False,
        max_boxes_to_draw=None)
  vis_utils.visualize_boxes_and_labels_on_image_array(
      image,
      detection_boxes,
      detection_classes,
      detection_scores,
      category_index,
      instance_masks=detection_masks,
      keypoints=detection_keypoints,
      use_normalized_coordinates=False,
      max_boxes_to_draw=max_num_predictions,
      min_score_thresh=min_score_thresh,
      agnostic_mode=agnostic_mode)

  if export_dir:
    export_path = os.path.join(export_dir, 'export-{}.png'.format(tag))
    vis_utils.save_image_array_as_png(image, export_path)

  summary_value = [
      tf.Summary.Value(tag=tag+'/image', image=tf.Summary.Image(
          encoded_image_string=vis_utils.encode_image_array_as_png_str(
              image)))
  ]

  summary = tf.Summary(value=summary_value)
  summary_writer = tf.summary.FileWriter(summary_dir)
  summary_writer.add_summary(summary, global_step)
  summary_writer.close()
  time.sleep(1)

  log.warn('Detection visualizations written to summary with tag %s.', tag)


# TODO: Add tests.
# TODO: Have an argument called `aggregated_processor_tensor_keys` that contains
# a whitelist of tensors used by the `aggregated_result_processor` instead of a
# blacklist. This will prevent us from inadvertently adding any evaluated
# tensors into the `results_list` data structure that are not needed by
# `aggregated_result_preprocessor`.
def run_checkpoint_once(tensor_dict,
                        update_op,
                        summary_dir,
                        aggregated_result_processor=None,
                        batch_processor=None,
                        checkpoint_dirs=None,
                        variables_to_restore=None,
                        restore_fn=None,
                        num_batches=1,
                        master='',
                        save_graph=False,
                        save_graph_dir='',
                        metric_names_to_values=None,
                        keys_to_exclude_from_results=(),
                        eval_config=None,
                        gpu_fraction=0.0,
                        categories=[]):
  """Evaluates both python metrics and tensorflow slim metrics.

  Python metrics are processed in batch by the aggregated_result_processor,
  while tensorflow slim metrics statistics are computed by running
  metric_names_to_updates tensors and aggregated using metric_names_to_values
  tensor.

  Args:
    tensor_dict: a dictionary holding tensors representing a batch of detections
      and corresponding groundtruth annotations.
    update_op: a tensorflow update op that will run for each batch along with
      the tensors in tensor_dict..
    summary_dir: a directory to write metrics summaries.
    aggregated_result_processor: a function taking one arguments:
      1. result_lists: a dictionary with keys matching those in tensor_dict
        and corresponding values being the list of results for each tensor
        in tensor_dict.  The length of each such list is num_batches.
    batch_processor: a function taking four arguments:
      1. tensor_dict: the same tensor_dict that is passed in as the first
        argument to this function.
      2. sess: a tensorflow session
      3. batch_index: an integer representing the index of the batch amongst
        all batches
      4. update_op: a tensorflow update op that will run for each batch.
      and returns result_dict, a dictionary of results for that batch.
      By default, batch_processor is None, which defaults to running:
        return sess.run(tensor_dict)
      To skip an image, it suffices to return an empty dictionary in place of
      result_dict.
    checkpoint_dirs: list of directories to load into an EnsembleModel. If it
      has only one directory, EnsembleModel will not be used -- a DetectionModel
      will be instantiated directly. Not used if restore_fn is set.
    variables_to_restore: None, or a dictionary mapping variable names found in
      a checkpoint to model variables. The dictionary would normally be
      generated by creating a tf.train.ExponentialMovingAverage object and
      calling its variables_to_restore() method. Not used if restore_fn is set.
    restore_fn: None, or a function that takes a tf.Session object and correctly
      restores all necessary variables from the correct checkpoint file. If
      None, attempts to restore from the first directory in checkpoint_dirs.
    num_batches: the number of batches to use for evaluation.
    master: the location of the Tensorflow session.
    save_graph: whether or not the Tensorflow graph is stored as a pbtxt file.
    save_graph_dir: where to store the Tensorflow graph on disk. If save_graph
      is True this must be non-empty.
    metric_names_to_values: A dictionary containing metric names to tensors
      which will be evaluated after processing all batches
      of [tensor_dict, update_op]. If any metrics depend on statistics computed
      during each batch ensure that `update_op` tensor has a control dependency
      on the update ops that compute the statistics.
    keys_to_exclude_from_results: keys in tensor_dict that will be excluded
      from results_list. Note that the tensors corresponding to these keys will
      still be evaluated for each batch, but won't be added to results_list.
    gpu_fraction: GPU memory fraction for evaluation.

  Raises:
    ValueError: if restore_fn is None and checkpoint_dirs doesn't have at least
      one element.
    ValueError: if save_graph is True and save_graph_dir is not defined.
  """
  if save_graph and not save_graph_dir:
    raise ValueError('`save_graph_dir` must be defined.')

  tf_config=tf.ConfigProto()
  if gpu_fraction == 0.0:
    tf_config.gpu_options.allow_growth = True
  else:
    tf_config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
  sess = tf.Session(master, graph=tf.get_default_graph(),
                    config=tf_config)
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  if restore_fn:
    checkpoint_file = restore_fn(sess)
  else:
    if not checkpoint_dirs:
      raise ValueError('`checkpoint_dirs` must have at least one entry.')
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dirs[0])
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, checkpoint_file)

  if save_graph:
    tf.train.write_graph(sess.graph_def, save_graph_dir, 'eval.pbtxt')

  valid_keys = list(set(tensor_dict.keys()) - set(keys_to_exclude_from_results))

  if eval_config.submission_format_output:
    for key in tensor_dict.keys():
      if not key in valid_keys:
        del tensor_dict[key]


  result_lists = {key: [] for key in valid_keys}
  counters = {'skipped': 0, 'success': 0}
  other_metrics = None
  with tf.contrib.slim.queues.QueueRunners(sess):
    try:
      for batch in range(int(num_batches)):
        if (batch + 1) % 100 == 0:
          log.info('Running eval ops batch %d/%d', batch + 1, num_batches)
        if not batch_processor:
          try:
            (result_dict, _) = sess.run([tensor_dict, update_op])
            counters['success'] += 1
          except tf.errors.InvalidArgumentError:
            log.warn('Skipping image')
            counters['skipped'] += 1
            result_dict = {}
        else:
          result_dict = batch_processor(
              tensor_dict, sess, batch, counters, update_op)

        for key in result_dict:
          if key in valid_keys:
            result_lists[key].append(result_dict[key])
      log.info('Running eval ops batch %d/%d', batch + 1, num_batches) # print final loop also
      if metric_names_to_values is not None:
        other_metrics = sess.run(metric_names_to_values)
      log.infov('Running eval batches done.')
    except tf.errors.OutOfRangeError:
      log.warn('Done evaluating -- epoch limit reached')
    finally:
      # Save raw detection results
      detection_results = {}
      for idx, image_id in enumerate(result_lists['image_id']):
        box = result_lists['detection_boxes'][idx]
        converted_boxes = np.column_stack([
          (box[:, 3] + box[:, 1]) / 2,  # cx
          (box[:, 2] + box[:, 0]) / 2,  # cy
          box[:, 3] - box[:, 1],  # w
          box[:, 2] - box[:, 0],  # h
        ])
        det = np.hstack([
          converted_boxes,
          np.expand_dims(result_lists['detection_scores'][idx], 1)])
        detection_results[image_id.split('/')[-1].split('.')[0]] \
          = np.array(sorted(det, key=lambda x: x[4]))
      result_path = os.path.join(summary_dir, 'detection_results')
      if not os.path.exists(result_path):
        os.makedirs(result_path)

      # Save detection results for submission
      if eval_config.submission_format_output:
        save_detection_results_for_submission(
          result_lists, categories, summary_dir, eval_config.metrics_set)
      else:
        # checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        last_step = checkpoint_file.split('-')[-1]
        if last_step.find('/') != -1:
          last_step = '0'
        # result_path = os.path.join(result_path, 'detection_result_' + last_step)
        # with open(result_path, 'w') as f:
        #   pickle.dump(detection_results, f)
        # log.info('Detection results saved to %s' % result_path)

        # evaluate results
        metrics = aggregated_result_processor(result_lists, sess)

        mtl_metrics = mtl_util.get_mtl_metrics(result_lists)
        metrics.update(mtl_metrics)

        if other_metrics is not None:
          metrics.update(other_metrics)

        # Loss
        if eval_config.calc_loss:
          losses = aggregated_loss(result_lists)
          metrics.update(losses)

        global_step = tf.train.global_step(sess, slim.get_global_step())
        write_metrics(metrics, global_step, summary_dir, eval_config)

        # save best ckpt
        save_best_ckpt(sess, metrics, variables_to_restore, checkpoint_file, eval_config)

        log.warn('# success: %d', counters['success'])
        log.warn('# skipped: %d', counters['skipped'])

  sess.close()

def aggregated_loss(result_lists):
  losses = dict()
  for name, val in result_lists.iteritems():
    if 'Loss' in name:
      losses[name] = float(np.mean(val))
  return losses

def save_detection_results_for_submission(result_lists, categories, summary_dir, metrics_set):
  detection_classes = result_lists['detection_classes']
  detection_boxes = result_lists['detection_boxes']
  detection_scores = result_lists['detection_scores']
  image_id = result_lists['image_id']

  if metrics_set == 'coco_metrics':
    filename = os.path.join(summary_dir, 'detection_results', 'detection_results.json')
    f = open(filename, 'w')
    f.write('[')
    first = True
    for idx in range(len(image_id)):
      image_name = image_id[idx]
      for box, score, class_id in zip(detection_boxes[idx], detection_scores[idx], detection_classes[idx]):
        if first:
          first = False
        else:
          f.write(',')
        t, l, b, r = box
        w = r - l
        h = b - t
        bbox = '[%.1f,%.1f,%.1f,%.1f]'%(l,t,w,h)
        out_str = '{"image_id":%s,"category_id":%d,"bbox":%s,"score":%.3f}'%(image_name, class_id, bbox, score)
        f.write(out_str)
    f.write(']')
    f.close()
  elif metrics_set == 'pascal_voc_metrics':
    fout = dict()
    for category in categories:
      filename = os.path.join(summary_dir, 'detection_results', 'comp4_det_test_' + category['name'] + '.txt')
      fout[category['id']] = open(filename, 'w')

    for idx in range(len(image_id)):
      image_name = image_id[idx]
      image_name = image_name.replace('.jpg', '')
      image_name = image_name.replace('.png', '')
      for box, score, class_id in zip(detection_boxes[idx], detection_scores[idx], detection_classes[idx]):
        f = fout[int(class_id)]
        t, l, b, r = box
        # if l < 0 or t < 0:
        #   temp = 0
        # l = max(1, l)
        # t = max(1, t)

        f.write('%s %f %f %f %f %f\n' % (image_name, score, l, t, r, b))

    for key in fout.keys():
      fout[key].close()


def save_best_ckpt(sess, metrics, variables_to_restore, checkpoint_file, eval_config):
  metrics_new = dict()
  for key in metrics:
    metrics_new[key] = float(metrics[key])
  metrics = metrics_new

  main_subset = eval_config.main_subset
  metrics_set = eval_config.metrics_set

  miss_rate = False
  mAP_dic = dict()
  if metrics_set == 'pascal_voc_metrics':
    for key in metrics.keys():
      if not '/' in key:
        if len(main_subset) == 0 and 'Subset all' in key:
          mAP_dic[key] = metrics[key]
          break
        elif len(main_subset) > 0 and (main_subset in key):
          mAP_dic[key] = metrics[key]
          break
  elif metrics_set == 'coco_metrics':
    key = 'COCO_Eval/All/AP'
    mAP_dic[key] = metrics[key]

  if len(mAP_dic.keys()) == 1:
    mAP = mAP_dic[mAP_dic.keys()[0]]
  else:
    assert(False)

  # mAP = np.mean(mAP_dic.values())

  output_dic = dict()
  output_dic['checkpoint_file'] = checkpoint_file
  if miss_rate:
    output_dic['miss_rate'] = float(mAP)
  else:
    output_dic['mAP'] = float(mAP)

  output_dic = dict(output_dic.items() + metrics.items())

  dir_name = os.path.dirname(checkpoint_file)
  best_path = os.path.join(dir_name, 'best')

  summary_filename = os.path.join(best_path, 'summary.json')

  if os.path.exists(summary_filename):
    with open(summary_filename, 'r') as f:
      summary = json.load(f)
      if miss_rate:
        if 'miss_rate' in summary.keys():
          if math.isnan(mAP) or mAP > summary['miss_rate']:
            return
      else:
        if 'mAP' in summary.keys():
          if math.isnan(mAP) or mAP < summary['mAP']:
            return
  else:
    if not os.path.exists(best_path):
      os.mkdir(best_path)

  with open(summary_filename, 'w') as f:
    json.dump(output_dic, f, indent=2, sort_keys=True)
    saver = tf.train.Saver(variables_to_restore)
    saver.save(sess, os.path.join(best_path, 'model.ckpt'))

# TODO: Add tests.
def repeated_checkpoint_run(tensor_dict,
                            update_op,
                            summary_dir,
                            aggregated_result_processor=None,
                            batch_processor=None,
                            checkpoint_dirs=None,
                            variables_to_restore=None,
                            restore_fn=None,
                            num_batches=1,
                            eval_interval_secs=120,
                            max_number_of_evaluations=None,
                            master='',
                            save_graph=False,
                            save_graph_dir='',
                            metric_names_to_values=None,
                            keys_to_exclude_from_results=(),
                            eval_config=None,
                            gpu_fraction=0.0,
                            categories=[]):
  """Periodically evaluates desired tensors using checkpoint_dirs or restore_fn.

  This function repeatedly loads a checkpoint and evaluates a desired
  set of tensors (provided by tensor_dict) and hands the resulting numpy
  arrays to a function result_processor which can be used to further
  process/save/visualize the results.

  Args:
    tensor_dict: a dictionary holding tensors representing a batch of detections
      and corresponding groundtruth annotations.
    update_op: a tensorflow update op that will run for each batch along with
      the tensors in tensor_dict.
    summary_dir: a directory to write metrics summaries.
    aggregated_result_processor: a function taking one argument:
      1. result_lists: a dictionary with keys matching those in tensor_dict
        and corresponding values being the list of results for each tensor
        in tensor_dict.  The length of each such list is num_batches.
    batch_processor: a function taking three arguments:
      1. tensor_dict: the same tensor_dict that is passed in as the first
        argument to this function.
      2. sess: a tensorflow session
      3. batch_index: an integer representing the index of the batch amongst
        all batches
      4. update_op: a tensorflow update op that will run for each batch.
      and returns result_dict, a dictionary of results for that batch.
      By default, batch_processor is None, which defaults to running:
        return sess.run(tensor_dict)
    checkpoint_dirs: list of directories to load into a DetectionModel or an
      EnsembleModel if restore_fn isn't set. Also used to determine when to run
      next evaluation. Must have at least one element.
    variables_to_restore: None, or a dictionary mapping variable names found in
      a checkpoint to model variables. The dictionary would normally be
      generated by creating a tf.train.ExponentialMovingAverage object and
      calling its variables_to_restore() method. Not used if restore_fn is set.
    restore_fn: a function that takes a tf.Session object and correctly restores
      all necessary variables from the correct checkpoint file.
    num_batches: the number of batches to use for evaluation.
    eval_interval_secs: the number of seconds between each evaluation run.
    max_number_of_evaluations: the max number of iterations of the evaluation.
      If the value is left as None the evaluation continues indefinitely.
    master: the location of the Tensorflow session.
    save_graph: whether or not the Tensorflow graph is saved as a pbtxt file.
    save_graph_dir: where to save on disk the Tensorflow graph. If store_graph
      is True this must be non-empty.
    metric_names_to_values: A dictionary containing metric names to tensors
      which will be evaluated after processing all batches
      of [tensor_dict, update_op]. If any metrics depend on statistics computed
      during each batch ensure that `update_op` tensor has a control dependency
      on the update ops that compute the statistics.
    keys_to_exclude_from_results: keys in tensor_dict that will be excluded
      from results_list. Note that the tensors corresponding to these keys will
      still be evaluated for each batch, but won't be added to results_list.
    gpu_fraction: GPU memory fraction for evaluation.

  Raises:
    ValueError: if max_num_of_evaluations is not None or a positive number.
    ValueError: if checkpoint_dirs doesn't have at least one element.
  """
  if max_number_of_evaluations and max_number_of_evaluations <= 0:
    raise ValueError(
        '`number_of_steps` must be either None or a positive number.')

  if not checkpoint_dirs:
    raise ValueError('`checkpoint_dirs` must have at least one entry.')

  last_evaluated_model_path = None
  number_of_evaluations = 0
  while True:
    start = time.time()
    log.infov('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                        time.gmtime()))
    model_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
    if not model_path:
      log.warn('No model found in %s. Will try again in %d seconds',
               checkpoint_dirs[0], eval_interval_secs)
    elif model_path == last_evaluated_model_path:
      log.warn('Found already evaluated checkpoint. Will try again in %d '
               'seconds', eval_interval_secs)
    else:
      last_evaluated_model_path = model_path
      run_checkpoint_once(tensor_dict, update_op, summary_dir,
                          aggregated_result_processor,
                          batch_processor, checkpoint_dirs,
                          variables_to_restore, restore_fn, num_batches, master,
                          save_graph, save_graph_dir, metric_names_to_values,
                          keys_to_exclude_from_results,
                          eval_config, gpu_fraction, categories)
    number_of_evaluations += 1

    if (max_number_of_evaluations and
        number_of_evaluations >= max_number_of_evaluations):
      log.infov('Finished evaluation!')
      break
    time_to_next_eval = start + eval_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)
