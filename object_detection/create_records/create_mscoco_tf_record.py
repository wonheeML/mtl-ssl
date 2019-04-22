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

r"""Convert raw MS-COCO dataset to TFRecord for object_detection.

Example usage:
    ./create_mscoco_tf_record --data_dir=/data/common_datasets/coco \
        --output_path=/data/common_datasets/coco/coco2014.record \
        --label_map_path=../data/mscoco_label_map.pbtxt
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pycocotools.coco import COCO

import hashlib
import io
# import logging
import os
import re
import sys

from lxml import etree
import tensorflow as tf
from PIL import Image

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from global_utils.custom_utils import log
import json

import numpy as np
from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops
import copy
import math
import random


join = os.path.join

flags = tf.app.flags
flags.DEFINE_string('data_dir', '../data/mscoco', 'Root directory to raw MS-COCO dataset.')
flags.DEFINE_string('set', 'trainval', 'Convert training set, validation set or merged set.')
flags.DEFINE_string('annotations_dir', 'annotations', '(Relative) path to annotations directory.')
flags.DEFINE_string('year', '2017', 'Desired challenge year.')
flags.DEFINE_string('output_name', 'coco', 'The name of output records'
                    ' : {output_name}_train.record / {output_name}_val.record')
flags.DEFINE_string('label_map_path', '../data/mscoco_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
flags.DEFINE_boolean('random_multi_object', True, '')

FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test', 'debug', 'test-dev']
YEARS = ['2014', '2017']

def boundary_check(bbx, width, height):
  l = bbx[0]
  t = bbx[1]
  w = bbx[2]
  h = bbx[3]

  l = max(0, min(width, l))
  t = max(0, min(height, t))
  w = max(0, min(width - l, w))
  h = max(0, min(height - t, h))

  return l, t, w, h


def dict_to_tf_example(label_map_dict, image_name, coco, class_indices, index_map_dict, set_name):

  try:
    imgId = getImageId(image_name)
    annIds = coco.getAnnIds(imgIds=imgId)
    anns = coco.loadAnns(annIds)

    with tf.gfile.GFile(image_name, 'rb') as fid:
      encoded_img = fid.read()
    encoded_img_io = io.BytesIO(encoded_img)
    image = Image.open(encoded_img_io)
    if image.format != 'JPEG':
      log.info('imgId(%d)(%s): Image format not JPEG'%(imgId, image.format))
      raise
    key = hashlib.sha256(encoded_img).hexdigest()
  except OSError as err:
    print("OS error: {0}".format(err))
    raise
    return None
  except ValueError as err:
    print("Value error: {0}".format(err))
    raise
    return None
  except:
    print("error")
    raise
    return None

  def get_string_label(val_list, n_round=3):
    str_list_label = [str(round(val, n_round)) for val in val_list]
    str_label = " ".join(str_list_label) + ' '
    str_label = str_label.replace('.0 ', ' ')[:-1]
    return str_label


  def get_box_list(obj_list):
    n_boxes = len(obj_list)
    boxes = np.zeros([n_boxes, 4], dtype=float)
    for idx, obj in enumerate(obj_list):
      ymin, xmin, ymax, xmax = get_box_coord(obj)
      boxes[idx, 0] = ymin
      boxes[idx, 1] = xmin
      boxes[idx, 2] = ymax
      boxes[idx, 3] = xmax
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
    if not obj_list:
      return []

    # parameters
    label_option = 1  # 0(area), 1(sqrt(area)), 2(area+1)
    normalize_option = 1 # 0(softmax), 1(div by sum)
    window_expand_ratio = 2.0

    multi_obj_list = []  # len of n_sub_window * n_object + 1(full image)
    max_num_classes = max(class_indices)

    # classwise_obj_list
    classwise_obj_list = [[] for i in range(max_num_classes + 1)]
    for obj in obj_list:
      class_id = obj['category_id']
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
        ymin, xmin, ymax, xmax = get_box_coord(obj)
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

  def get_box_coord(obj):
    xmin, ymin, obj_w, obj_h = obj['bbox']
    xmax = xmin + obj_w
    ymax = ymin + obj_h
    return ymin, xmin, ymax, xmax

  def get_closeness(obj, obj_list):
    '''
    :param obj:
    :param obj_list:
    :return: closeness label in text, no bg class, Nearest(1) ~ Farthest(0)
    '''
    max_num_classes = max(class_indices)
    closeness_list = [0] * (max_num_classes + 1)
    diag_dist = np.sqrt(width * width + height * height)

    if len(obj_list) == 1:
      closeness_list[0] = 1
      str_label = get_string_label(closeness_list)
      return str_label

    for obj2 in obj_list:
      if obj == obj2:
        continue
      if obj['category_id'] == obj2['category_id']:
        continue

      dist = max(0.0, get_center_distance(obj, obj2)) / diag_dist  # Nearest(0) ~ Farthest(1)
      closeness = 1.0 - dist  # Nearest(1) ~ Farthest(0.5)
      class_index = obj2['category_id']# with background
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

  width = image.width
  height = image.height

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  is_crowd_list = []
  closeness = []

  for ann in anns:
    bbox        = ann['bbox']
    category_id = ann['category_id']
    is_crowd    = ann['iscrowd']
    cat = coco.loadCats(category_id)[0]
    lbl = cat['name']

    bbox = boundary_check(bbox, width, height)

    if not (width > 0 and height > 0 and bbox[2] > 0 and bbox[3] > 0):
      log.info('imgId(%d): bbox(%.1f,%.1f,%.1f,%.1f)'
               % (imgId,
                  ann['bbox'][0], ann['bbox'][1],
                  ann['bbox'][2], ann['bbox'][3]))
      continue

    xmin.append(float(bbox[0]) / width)
    ymin.append(float(bbox[1]) / height)
    xmax.append(float(bbox[2]+bbox[0]) / width)
    ymax.append(float(bbox[3]+bbox[1]) / height)

    classes_text.append(lbl.encode('utf8'))
    classes.append(label_map_dict[lbl.encode('utf8')])

    is_crowd_list.append(is_crowd)
    closeness.append(get_closeness(ann, anns))

  multi_obj_list = create_multi_object(anns, width, height)
  window_ymin, window_xmin, window_ymax, window_xmax, window_labels = split_multi_obj_list(multi_obj_list)

  edgemask_masks = create_edgemask(image, anns)
  edgemask_masks_flat = np.reshape(edgemask_masks, [-1])

  example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(image_name.encode('utf8')),
    'image/source_id': dataset_util.bytes_feature(str(imgId).encode('utf8')),
    'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_img),
    'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
    'image/object/is_crowd': dataset_util.int64_list_feature(is_crowd_list),
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


def get_filename_list(filename):
  try:
    f = open(filename, 'r')
    img_filename_list = []
    for line in f.read().split('\n'):
      if len(line) != 0:
        img_filename_list.append(line)
    f.close()
  except:
    raise ValueError('file not exists : ' + filename)
  return img_filename_list


def getImageId(filename):
  id_str = re.split('_|/|\\.', filename)[-2]
  return int(id_str)

def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))

  if FLAGS.year == '2014':
    data_dir = FLAGS.data_dir
    annFileTrain = join(data_dir, FLAGS.annotations_dir, 'instances' + '_train' + FLAGS.year + '.json')
    annFileVal = join(data_dir, FLAGS.annotations_dir, 'instances' + '_val' + FLAGS.year + '.json')
    cocoTrain = COCO(annFileTrain)
    cocoVal = COCO(annFileVal)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    class_indices = label_map_util.get_class_indices(label_map_dict)
    index_map_dict = label_map_util.get_index_map_dict(label_map_dict)

    if not 'bg' in label_map_dict.keys():
      label_map_dict['bg'] = 0

    train_output_path = join(data_dir, FLAGS.output_name + '_' + FLAGS.year + '_train.record')
    val_output_path = join(data_dir, FLAGS.output_name + '_' + FLAGS.year + '_val.record')

    writer_train = tf.python_io.TFRecordWriter(train_output_path)
    writer_val = tf.python_io.TFRecordWriter(val_output_path)

    log.info('Reading from MS-COCO dataset.')

    images_dir = '/images/train' + FLAGS.year + '/'
    train_filename_list = [images_dir + item[1]['file_name'] for item in cocoTrain.imgs.items()]

    images_dir = '/images/val' + FLAGS.year + '/'
    val_filename_list = [images_dir + item[1]['file_name'] for item in cocoVal.imgs.items()]

    # train
    idx = 1
    for filename in train_filename_list:
      filepath = data_dir + filename
      if idx % 1000 == 0:
        log.info('(train)On image %d of %d', idx, len(train_filename_list))
      idx += 1

      try:
        tf_example = dict_to_tf_example(label_map_dict, filepath, cocoTrain, class_indices, index_map_dict)
      except:
        continue

      if not tf_example == None:
        writer_train.write(tf_example.SerializeToString())

    # val
    idx = 1
    for filename in val_filename_list:
      filepath = data_dir + filename
      if idx % 1000 == 0:
        log.info('(val)On image %d of %d', idx, len(val_filename_list))
      idx += 1

      try:
        tf_example = dict_to_tf_example(label_map_dict, filepath, cocoVal, class_indices, index_map_dict)
      except:
        continue

      if not tf_example == None:
        writer_val.write(tf_example.SerializeToString())

    writer_train.close()
    writer_val.close()

  elif FLAGS.year == '2017':
    set_list = [FLAGS.set]
    if set_list[0] == 'trainval':
      set_list = ['train', 'val']

    num_none = 0
    for set_name in set_list:
      data_dir = FLAGS.data_dir

      if 'test' in set_name:
        annFile = join(data_dir, FLAGS.annotations_dir, 'image_info' + '_' + set_name + FLAGS.year + '.json')
      else:
        annFile = join(data_dir, FLAGS.annotations_dir, 'instances' + '_' + set_name + FLAGS.year + '.json')
      coco = COCO(annFile)
      label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

      class_indices = label_map_util.get_class_indices(label_map_dict)
      index_map_dict = label_map_util.get_index_map_dict(label_map_dict)
      if not 'bg' in label_map_dict.keys():
        label_map_dict['bg'] = 0

      output_path = join(data_dir, FLAGS.output_name + '_' + FLAGS.year + '_' + set_name + '.record')
      writer = tf.python_io.TFRecordWriter(output_path)

      log.info('Reading from MS-COCO dataset: %s'%(set_name))

      if set_name == 'test-dev':
        set_name = 'test'

      images_dir = '/images/' + set_name + FLAGS.year + '/'
      filename_list = [images_dir + item[1]['file_name'] for item in coco.imgs.items()]

      for idx, filename in enumerate(filename_list):
        filepath = data_dir + filename
        if idx % 100 == 0:
          log.info('(train)On image %d of %d', idx, len(filename_list))

        try:
          tf_example = dict_to_tf_example(label_map_dict, filepath, coco, class_indices, index_map_dict, set_name)
        except:
          continue
        if tf_example == None:
          num_none += 1
          print(print("idx(%d), filename(%s), num_none(%d) : annIds == []"%(idx, filename, num_none)))
        else:
          writer.write(tf_example.SerializeToString())

      writer.close()
  else:
    raise ValueError('year must be in : {}'.format(YEARS))

  log.info('End')

if __name__ == '__main__':
  tf.app.run()
