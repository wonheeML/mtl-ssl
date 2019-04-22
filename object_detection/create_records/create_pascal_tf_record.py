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

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    ./create_pascal_tf_record --data_dir=/data/common_datasets/voc/VOCdevkit \
        --year=VOC2012 \
        --output_path=/data/common_datasets/voc/voc2012.record \
        --label_map_path=../data/pascal_label_map.pbtxt
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
# import logging
from global_utils.custom_utils import log
import os
import re
import math
from numpy import inf

from lxml import etree
import tensorflow as tf
from PIL import Image

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import numpy as np
from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops
import copy
import math
import random

join = os.path.join

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or merged set.')
flags.DEFINE_string('exclude', '', 'list of dataset to be excluded e.g VOC2007_test, VOC2012_test')
flags.DEFINE_string('annotations_dir', 'Annotations', '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '../data/pascal_label_map.pbtxt', 'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore difficult instances')
flags.DEFINE_boolean('random_multi_object', True, '')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test', 'all']
YEARS = ['VOC2007', 'VOC2012', 'merged']

subsets = {
  'all':     ([0, inf]), # area area_range
}

def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       class_indices,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  if 'size' in data:
    width = int(data['size']['width'])
    height = int(data['size']['height'])
  else:
    width = image.width
    height = image.height
    data_object_bndbox = dict()
    data_object_bndbox['xmin'] = 0
    data_object_bndbox['ymin'] = 0
    data_object_bndbox['xmax'] = width
    data_object_bndbox['ymax'] = height
    data['object'][0]['bndbox'] = data_object_bndbox

  def get_string_label(val_list, n_round=3):
    str_list_label = [str(round(val, n_round)) for val in val_list]
    str_label = " ".join(str_list_label) + ' '
    str_label = str_label.replace('.0 ', ' ')[:-1]
    return str_label


  def get_box_list(obj_list):
    n_boxes = len(obj_list)
    boxes = np.zeros([n_boxes, 4], dtype=float)
    for idx, obj in enumerate(obj_list):
      bndbox = obj['bndbox']
      boxes[idx, 0] = bndbox['ymin']
      boxes[idx, 1] = bndbox['xmin']
      boxes[idx, 2] = bndbox['ymax']
      boxes[idx, 3] = bndbox['xmax']
    boxlist = np_box_list.BoxList(boxes)
    return boxlist

  def get_rect_area_total(obj_list, window):
    if not obj_list:
      return 0.0

    area_total = 0
    box_list = get_box_list(obj_list)

    box_list_clip = np_box_list_ops.clip_to_window(box_list, window)
    box_list_norm = np_box_list_ops.change_coordinate_frame(box_list_clip, window)
    n_boxes = box_list_norm.num_boxes()

    index = [str(i) for i in range(n_boxes)]
    box_list_norm.add_field('index', np.asarray(index))
    box_list_norm_org = copy.deepcopy(box_list_norm)

    sign = 1
    while box_list_norm:
      area = np.sum(np_box_list_ops.area(box_list_norm))
      area_total += area * sign
      sign *= -1
      box_list_norm = np_box_list_ops.intersection_boxes(box_list_norm_org, box_list_norm)
    return area_total


  def normalization(x, norm_option):
    if norm_option == 0:    # softmax
      val = np.exp(x) / np.sum(np.exp(x))
    elif norm_option == 1:  # div by sum
      val = np.array(x) / np.sum(x)
    return val


  def label_with_option(rect_area_total, label_option):
    if label_option == 0:
      val = rect_area_total
    elif label_option == 1:
      val = math.sqrt(rect_area_total)
    elif label_option == 2:
      val = rect_area_total + 1
    return val


  def create_multi_object(obj_list, width, height):
    # parameters
    label_option = 1  # 0(area), 1(sqrt(area)), 2(area+1)
    normalize_option = 1 # 0(softmax), 1(div by sum)
    window_expand_ratio = 2.0

    multi_obj_list = []  # len of n_sub_window * n_object + 1(full image)
    max_num_classes = max(class_indices)

    # classwise_obj_list
    classwise_obj_list = [[] for i in range(max_num_classes + 1)]
    for obj in obj_list:
      class_id = label_map_dict[obj['name']]
      classwise_obj_list[class_id].append(obj)

    def get_multi_label(obj_list, window, width, height):
      multi_obj = dict()
      multi_obj['ymin'] = window[0] / height
      multi_obj['xmin'] = window[1] / width
      multi_obj['ymax'] = window[2] / height
      multi_obj['xmax'] = window[3] / width

      label_with_bg = [0.0] * (max_num_classes + 1)

      # for bg
      rect_area_total = max(0.0, 1 - get_rect_area_total(obj_list, window))
      val = label_with_option(rect_area_total, label_option)
      label_with_bg[0] = val
      multi_obj['bg'] = val

      # for classwise boxes
      for class_id in class_indices:
        if classwise_obj_list[class_id]:
          rect_area_total = get_rect_area_total(classwise_obj_list[class_id], window)
          val = label_with_option(rect_area_total, label_option)
          label_with_bg[class_id] = val

      label_with_bg = normalization(label_with_bg, normalize_option)
      str_label = get_string_label(label_with_bg)
      multi_obj['labels'] = str_label
      return multi_obj

    # ===================== window image =====================
    if FLAGS.random_multi_object:
      min_obj_size = 32.0
      num_windows = 64

      while len(multi_obj_list) < num_windows:
        box_height = random.random() * (height - min_obj_size) + min_obj_size
        box_width = random.random() * (width - min_obj_size) + min_obj_size
        cy = random.random() * height
        cx = random.random() * width

        ymin = max(0.0, cy - box_height / 2)
        xmin = max(0.0, cx - box_width / 2)
        ymax = min(height, cy + box_height / 2)
        xmax = min(width, cx + box_width / 2)

        if xmax - xmin < min_obj_size:
          if xmin == 0.0:
            xmax = min_obj_size
          elif xmax == width:
            xmin = width - min_obj_size
        if ymax - ymin < min_obj_size:
          if ymin == 0.0:
            ymax = min_obj_size
          elif ymax == height:
            ymin = height - min_obj_size

        window = [ymin, xmin, ymax, xmax]
        multi_obj = get_multi_label(obj_list, window, width, height)

        if not obj_list:
          multi_obj_list.append(multi_obj)
          break
        if multi_obj['bg'] == 1.0:
          continue
        else:
          multi_obj_list.append(multi_obj)

    else:
      window = [0, 0, height, width]
      multi_obj_list.append(get_multi_label(obj_list, window, width, height)) # full-image

      for obj in obj_list:
        box = obj['bndbox']
        ymin, xmin, ymax, xmax = [
          float(box['ymin']), float(box['xmin']), float(box['ymax']), float(box['xmax'])]
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        w2 = (xmax - xmin) / 2
        h2 = (ymax - ymin) / 2

        ratio = window_expand_ratio
        while True:
          ymin = max(cy - h2 * ratio, 0)
          xmin = max(cx - w2 * ratio, 0)
          ymax = min(cy + h2 * ratio, height)
          xmax = min(cx + w2 * ratio, width)

          if ymin <= 0 and xmin <= 0 and ymax >= height and xmax >= width:
            break

          window = [ymin, xmin, ymax, xmax]
          multi_obj_list.append(get_multi_label(obj_list, window, width, height))  # sub-image
          ratio *= window_expand_ratio
    # ===================== window image =====================
    return multi_obj_list

  def split_multi_obj_list(multi_obj_list):
    ymin = []
    xmin = []
    ymax = []
    xmax = []
    labels = []
    for multi_obj in multi_obj_list:
      ymin.append(multi_obj['ymin'])
      xmin.append(multi_obj['xmin'])
      ymax.append(multi_obj['ymax'])
      xmax.append(multi_obj['xmax'])
      labels.append(multi_obj['labels'])
    return ymin, xmin, ymax, xmax, labels

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  subset_list = []
  label_type_list = []
  closeness = []

  def get_box_coord(obj):
    ymin = float(obj['bndbox']['ymin'])
    xmin = float(obj['bndbox']['xmin'])
    ymax = float(obj['bndbox']['ymax'])
    xmax = float(obj['bndbox']['xmax'])
    return ymin, xmin, ymax, xmax

  def get_closeness(obj, obj_list):
    '''
    :param obj:
    :param obj_list:
    :return: closeness label in text, no bg class, Nearest(1) ~ Farthest(0)
    '''
    max_num_classes = max(class_indices)
    closeness_list = [0] * (max_num_classes + 1) # with background
    diag_dist = np.sqrt(width*width + height*height)

    if len(obj_list) == 1:
      closeness_list[0] = 1
      str_label = get_string_label(closeness_list)
      return str_label

    for obj2 in obj_list:
      if obj == obj2:
        continue
      if obj['name'] == obj2['name']:
        continue

      dist = max(0.0, get_center_distance(obj, obj2)) / diag_dist  # Nearest(0) ~ Farthest(1)
      closeness = 1.0 - dist # Nearest(1) ~ Farthest(0.5)
      class_index = label_map_dict[obj2['name']] # with background
      closeness_list[class_index] = max(closeness_list[class_index], closeness)

    if sum(closeness_list[1:]) == 0:
      closeness_list[0] = 1

    norm_val = sum(closeness_list)
    closeness_list = [val / norm_val for val in closeness_list]

    str_label = get_string_label(closeness_list)
    return str_label

  def get_center_distance(obj1, obj2):
    ymin, xmin, ymax, xmax = get_box_coord(obj1)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    ymin2, xmin2, ymax2, xmax2 = get_box_coord(obj2)
    cx2 = (xmin2 + xmax2) / 2
    cy2 = (ymin2 + ymax2) / 2

    distx = abs(cx - cx2)
    disty = abs(cy - cy2)
    dist = math.sqrt(distx * distx + disty * disty)

    return dist

  def create_edgemask(image, obj_list):
    mask_size = 64
    width = float(image.width)
    height = float(image.height)

    box_mask = np.zeros([mask_size, mask_size], np.float32)
    box_weight = np.ones([mask_size, mask_size], np.float32) / mask_size / mask_size

    if width==0 or height==0:
      print('width==0 or height==0')

    for obj in obj_list:
      ymin, xmin, ymax, xmax = get_box_coord(obj)
      ymin = int(ymin / height * mask_size)
      xmin = int(xmin / width * mask_size)
      ymax = min(mask_size-1, int(ymax / height * mask_size + 0.99))
      xmax = min(mask_size-1, int(xmax / width * mask_size + 0.99))
      box_width = xmax - xmin + 1
      box_height = ymax - ymin + 1

      if box_width == 0:
        if xmin+xmax > mask_size:
          xmin -= 1
        else:
          xmax += 1
        box_width = 1

      if box_height == 0:
        if ymin+ymax > mask_size:
          ymin -= 1
        else:
          ymax += 1
        box_height = 1

      box_mask[ymin:(ymax+1), xmin:(xmax+1)] = 1.0
      weight = np.ones([box_height, box_width], np.float32) / (box_width) / (box_height)

      if box_width==0 or box_height==0:
        print('box_width==0 or box_height==0')

      weight_cur = box_weight[ymin:(ymax+1), xmin:(xmax+1)]
      box_weight[ymin:(ymax+1), xmin:(xmax+1)] = np.maximum(weight, weight_cur)

    box_weight /= np.mean(box_weight)
    box_mask_ret = np.array([box_mask, box_weight])

    return box_mask_ret

  num_objects = 0
  for obj in data['object']:
    if obj['difficult'] == '0':
      num_objects += 1

  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue

    difficult_obj.append(int(difficult))

    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

    # determine which subsets the GT belongs to
    box_height = float(obj['bndbox']['ymax']) - float(obj['bndbox']['ymin'])
    box_width = float(obj['bndbox']['xmax']) - float(obj['bndbox']['xmin'])
    box_area = box_height*box_width

    anno_subset = []
    for subset_name in subsets:
      area_range = subsets[subset_name]

      if box_area < area_range[0] or box_area >= area_range[1]:
        continue

      anno_subset.append(subset_name)
    subset_list.append('|'.join(anno_subset))

    label_type_list.append('')
    closeness.append(get_closeness(obj, data['object']))

  multi_obj_list = create_multi_object(data['object'], width, height)
  window_ymin, window_xmin, window_ymax, window_xmax, window_labels = split_multi_obj_list(multi_obj_list)

  edgemask_masks = create_edgemask(image, data['object'])
  edgemask_masks_flat = np.reshape(edgemask_masks, [-1])

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
      'image/object/subset': dataset_util.bytes_list_feature(subset_list),
      'image/object/label_type': dataset_util.bytes_list_feature(label_type_list),
      'image/window/bbox/ymin': dataset_util.float_list_feature(window_ymin),
      'image/window/bbox/xmin': dataset_util.float_list_feature(window_xmin),
      'image/window/bbox/ymax': dataset_util.float_list_feature(window_ymax),
      'image/window/bbox/xmax': dataset_util.float_list_feature(window_xmax),
      'image/window/labels/text': dataset_util.bytes_list_feature(window_labels),
      'image/object/closeness/text': dataset_util.bytes_list_feature(closeness),
      'image/edgemask/masks': dataset_util.float_list_feature(edgemask_masks_flat),
      'image/edgemask/height': dataset_util.int64_feature(edgemask_masks.shape[1]),
      'image/edgemask/width': dataset_util.int64_feature(edgemask_masks.shape[2])
  }))
  return example


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
  if FLAGS.year not in YEARS:
    raise ValueError('year must be in : {}'.format(YEARS))

  if FLAGS.set == 'all':
    datasets = ['trainval', 'test']
  else:
    datasets = [FLAGS.set]

  exclude = [dataset_name.strip() for dataset_name in FLAGS.exclude.split(',')]

  data_dir = FLAGS.data_dir
  years = ['VOC2007', 'VOC2012']
  if FLAGS.year != 'merged':
    years = [FLAGS.year]

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  class_indices = label_map_util.get_class_indices(label_map_dict)

  if not 'bg' in  label_map_dict.keys():
    label_map_dict['bg'] = 0

  for dataset in datasets:
    for year in years:
      dataset_name = year + '_' + dataset
      if dataset_name in exclude:
        continue
      log.info('Reading from PASCAL %s dataset.', dataset_name)
      examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
                                   'aeroplane_' + dataset + '.txt')
      annotations_dir = os.path.join(data_dir, year, FLAGS.annotations_dir)
      examples_list = dataset_util.read_examples_list(examples_path)
      for idx, example in enumerate(examples_list):
        if idx % 100 == 0:
          log.info('On image %d of %d', idx, len(examples_list))
        path = os.path.join(annotations_dir, example + '.xml')
        if os.path.exists(path):
          with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        else:
          data = dict()
          data_object = dict()
          data_object['difficult'] = '0'

          data['folder'] = year
          data['filename'] = example + '.jpg'
          data_object['name'] = 'cat'
          data_object['pose'] = 'Left'
          data_object['truncated'] = '0'
          data['object'] = [data_object]

        tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict, class_indices,
                                        FLAGS.ignore_difficult_instances)
        if tf_example == None:
          continue

        writer.write(tf_example.SerializeToString())

  writer.close()

if __name__ == '__main__':
  tf.app.run()
