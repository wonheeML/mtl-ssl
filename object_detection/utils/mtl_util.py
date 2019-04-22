
"""Multi-task utility functions."""

from object_detection.core import standard_fields as fields
from object_detection.utils import metrics
from object_detection.utils import np_box_ops
from skimage.transform import resize
import numpy as np

def _softmax(x):
  y = x - np.expand_dims(np.max(x, axis=-1), -1)
  y = np.exp(y)
  ax_sum = np.expand_dims(np.sum(y, axis=-1), -1)
  p = y / ax_sum
  return p

def _sigmoid(x):
  return 1 / (1 + np.exp(-1.0 * x))

def get_mtl_metrics(result_lists):
  mtl_metrics = dict()
  gt_boxes_list = result_lists[fields.InputDataFields.groundtruth_boxes]
  detection_box_list = result_lists['detection_boxes']

  b_window = False
  b_closeness = False
  b_edgemask = False

  if 'window_classes_gt' in result_lists.keys():
    window_classes_gt_list = result_lists['window_classes_gt']
    window_classes_dt_list = result_lists['window_classes_dt']
    b_window = True
  if 'closeness_gt' in result_lists.keys():
    closeness_gt_list = result_lists['closeness_gt']
    closeness_dt_list = result_lists['closeness_dt']
    b_closeness = True
  if 'edgemask_gt' in result_lists.keys():
    edgemask_gt_list = result_lists['edgemask_gt']
    edgemask_dt_list = result_lists['edgemask_dt']
    b_edgemask = True


  if b_window:
    map_list = []
    for window_classes_gt, window_classes_dt in zip(window_classes_gt_list, window_classes_dt_list):
      ap_list = []
      for window_class_gt, window_class_dt in zip(window_classes_gt, window_classes_dt):
        window_class_dt = _softmax(window_class_dt)
        window_class_gt = [float(val_str) for val_str in window_class_gt.split(' ')]

        scores = window_class_dt
        tp_fp_labels = np.asarray([gt > 0 for gt in window_class_gt], dtype=np.bool)
        num_gt = int(np.sum(np.asarray(tp_fp_labels, dtype=np.int32)))
        precision, recall = metrics.compute_precision_recall(scores, tp_fp_labels, num_gt)
        average_precision = metrics.compute_average_precision(precision, recall)
        ap_list.append(average_precision)
      map_list.append(float(np.mean(ap_list)))
    window_map = float(np.mean(map_list))
    mtl_metrics['mtl/window_map'] = window_map


  gt_dt_index_list = []
  for gt_boxes, dt_boxes in zip(gt_boxes_list, detection_box_list):
    intersection = np_box_ops.intersection(gt_boxes, dt_boxes)
    gt_dt_index = np.argmax(intersection, axis=1)
    gt_dt_index_list.append(gt_dt_index)

  if b_closeness:
    diff_list = []
    for closeness_gt, gt_dt_indices, closeness_dt_image in zip(closeness_gt_list, gt_dt_index_list, closeness_dt_list):
      ap_list = []
      for gt, gt_dt_index in zip(closeness_gt, gt_dt_indices):
        closeness_dt = _sigmoid(closeness_dt_image[gt_dt_index])
        closeness_gt = np.asarray([float(val_str) for val_str in gt.split(' ')], dtype=np.float32)
        num_non_zeros = int(np.sum(closeness_gt != 0))
        if num_non_zeros == 0:
          continue

        argmax_dt = np.argmax(closeness_dt[1:])
        argmax_gt = np.argmax(closeness_gt[1:])
        ap_list.append(float(argmax_dt==argmax_gt))

      if ap_list:
        diff_list.append(float(np.mean(ap_list)))
    if diff_list:
      closeness_diff = float(np.mean(diff_list))
    else:
      closeness_diff = 0.0
    mtl_metrics['mtl/closeness_diff'] = closeness_diff

  if b_edgemask:
    ap_list = []
    for edgemask_gt, edgemask_dt in zip(edgemask_gt_list, edgemask_dt_list):
      edgemask_gt = edgemask_gt[0]
      edgemask_dt = edgemask_dt[0]
      shape_gt = edgemask_gt.shape
      edgemask_dt_resize = resize(edgemask_dt, list(shape_gt) + [2]).astype(np.float32)
      edgemask_dt_resize = (edgemask_dt_resize[:,:,0] < edgemask_dt_resize[:,:,1]).astype(np.float32)

      edgemask_precision = np.mean(edgemask_dt_resize == edgemask_gt)
      ap_list.append(edgemask_precision)

    if ap_list:
      mtl_metrics['mtl/edgemask_ap'] = float(np.mean(ap_list))
    else:
      mtl_metrics['mtl/edgemask_ap'] = float(0)
  return mtl_metrics

