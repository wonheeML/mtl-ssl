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

"""Detection model trainer.

This file provides a generic training method that can be used to train a
DetectionModel.
"""
import functools

import google.protobuf.text_format as text_format
import tensorflow as tf

from object_detection.builders import optimizer_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import batcher
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.utils import ops as util_ops
from object_detection.utils import variables_helper
from deployment import model_deploy
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from slim import learning as custom_learning
slim = tf.contrib.slim


def _create_input_queue(batch_size_per_clone, create_tensor_dict_fn,
                        batch_queue_capacity, num_batch_queue_threads,
                        prefetch_queue_capacity, data_augmentation_options,
                        ignore_options=None, mtl_window=False, mtl_edgemask=False):
  """Sets up reader, prefetcher and returns input queue.

  Args:
    batch_size_per_clone: batch size to use per clone.
    create_tensor_dict_fn: function to create tensor dictionary.
    batch_queue_capacity: maximum number of elements to store within a queue.
    num_batch_queue_threads: number of threads to use for batching.
    prefetch_queue_capacity: maximum capacity of the queue used to prefetch
                             assembled batches.
    data_augmentation_options: a list of tuples, where each tuple contains a
      data augmentation function and a dictionary containing arguments and their
      values (see preprocessor.py).
    ignore_options: exception condition of training loss

  Returns:
    input queue: a batcher.BatchQueue object holding enqueued tensor_dicts
      (which hold images, boxes and targets).  To get a batch of tensor_dicts,
      call input_queue.Dequeue().
  """
  tensor_dict = create_tensor_dict_fn()

  tensor_dict[fields.InputDataFields.image] = tf.expand_dims(
      tensor_dict[fields.InputDataFields.image], 0)

  images = tensor_dict[fields.InputDataFields.image]
  float_images = tf.to_float(images)
  tensor_dict[fields.InputDataFields.image] = float_images

  preprocessor.make_ignore_list(tensor_dict, ignore_options)

  if mtl_window:
    for option in data_augmentation_options:
      if 'random_horizontal_flip' in option[0].func_name:
        option[1][fields.InputDataFields.window_boxes] = tensor_dict[fields.InputDataFields.window_boxes]

  if mtl_edgemask:
    for option in data_augmentation_options:
      if 'random_horizontal_flip' in option[0].func_name:
        option[1][fields.InputDataFields.groundtruth_edgemask_masks] = tensor_dict[fields.InputDataFields.groundtruth_edgemask_masks]

  if data_augmentation_options:
    tensor_dict = preprocessor.preprocess(tensor_dict, data_augmentation_options,
                                          mtl_window=mtl_window, mtl_edgemask=mtl_edgemask)

  input_queue = batcher.BatchQueue(
      tensor_dict,
      batch_size=batch_size_per_clone,
      batch_queue_capacity=batch_queue_capacity,
      num_batch_queue_threads=num_batch_queue_threads,
      prefetch_queue_capacity=prefetch_queue_capacity)
  return input_queue


def _get_inputs(input_queue, num_classes, with_filename=False):
  """Dequeue batch and construct inputs to object detection model.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    num_classes: Number of classes.

  Returns:
    images: a list of 3-D float tensor of images.
    locations_list: a list of tensors of shape [num_boxes, 4]
      containing the corners of the groundtruth boxes.
    classes_list: a list of padded one-hot tensors containing target classes.
    masks_list: a list of 3-D float tensors of shape [num_boxes, image_height,
      image_width] containing instance masks for objects if present in the
      input_queue. Else returns None.
  """
  read_data_list = input_queue.dequeue()
  label_id_offset = 1
  def extract_images_and_targets(read_data):
    image = read_data[fields.InputDataFields.image]
    location_gt = read_data[fields.InputDataFields.groundtruth_boxes]
    classes_gt = tf.cast(read_data[fields.InputDataFields.groundtruth_classes], tf.int32)
    edgemask_gt = tf.cast(read_data[fields.InputDataFields.groundtruth_edgemask_masks], tf.float32)

    ignore_gt = read_data.get(fields.InputDataFields.groundtruth_ignore)
    if ignore_gt.get_shape() is not classes_gt.get_shape():
      ignore_gt = tf.zeros_like(classes_gt, dtype=tf.bool)

    masks_gt = read_data.get(fields.InputDataFields.groundtruth_instance_masks)

    classes_gt -= label_id_offset
    classes_gt = util_ops.padded_one_hot_encoding(indices=classes_gt, depth=num_classes, left_pad=0)

    filename = None
    if with_filename:
      filename = read_data[fields.InputDataFields.filename]

    # window box gt
    window_location_gt = read_data[fields.InputDataFields.window_boxes]
    window_classes_gt_string = read_data[fields.InputDataFields.window_classes]
    st = tf.string_split(window_classes_gt_string)
    st_values_float = tf.string_to_number(st.values)
    window_classes_gt = tf.sparse_to_dense(st.indices, st.dense_shape, st_values_float)
    window_classes_gt = tf.reshape(window_classes_gt, [-1, num_classes + 1])

    # closeness gt
    object_closeness_gt_string = read_data[fields.InputDataFields.groundtruth_closeness]
    st = tf.string_split(object_closeness_gt_string)
    st_values_float = tf.string_to_number(st.values)
    closeness_classes_gt = tf.sparse_to_dense(st.indices, st.dense_shape, st_values_float)
    closeness_classes_gt = tf.reshape(closeness_classes_gt, [-1, num_classes + 1])

    return image, location_gt, ignore_gt, classes_gt, masks_gt, filename, \
           window_location_gt, window_classes_gt, closeness_classes_gt, edgemask_gt
  return zip(*map(extract_images_and_targets, read_data_list))


def _create_losses(input_queue, create_model_fn, show_image_summary,
                   update_schedule, **kwargs):
  """Creates loss function for a DetectionModel.

  Args:
    input_queue: BatchQueue object holding enqueued tensor_dicts.
    create_model_fn: A function to create the DetectionModel.
    kwargs: Additional arguments to make model.
  """
  if kwargs.has_key('mtl'):
    mtl = kwargs['mtl']
    del kwargs['mtl']

  detection_model = create_model_fn()
  (images, groundtruth_boxes_list, groundtruth_ignore_list,
   groundtruth_classes_list, groundtruth_masks_list,
   filenames, window_boxes_list, window_classes_list, groundtruth_closeness_list, groundtruth_edgemask_list) \
      = _get_inputs(input_queue, detection_model.num_classes, with_filename=True)
  if show_image_summary:
    detection_model.provide_image_infos(images, filenames)

  images = [detection_model.preprocess(image) for image in images]
  images = tf.concat(images, 0)
  if any(mask is None for mask in groundtruth_masks_list):
    groundtruth_masks_list = None

  detection_model.provide_groundtruth(groundtruth_boxes_list,
                                      groundtruth_classes_list,
                                      groundtruth_closeness_list,
                                      groundtruth_ignore_list,
                                      groundtruth_masks_list)
  detection_model.provide_window(window_boxes_list, window_classes_list)
  detection_model.provide_edgemask(groundtruth_edgemask_list)

  prediction_dict = detection_model.predict(images)

  # TODO: implement joint training

  if mtl.window:
    prediction_dict = detection_model.predict_with_window(prediction_dict)

  if mtl.edgemask:
    prediction_dict = detection_model.predict_edgemask(prediction_dict)

  if mtl.refine:
    prediction_dict = detection_model.predict_with_mtl_results(prediction_dict)

  losses_dict = detection_model.loss(prediction_dict, **kwargs)

  for loss_name, loss_tensor in losses_dict.iteritems():
    loss_tensor = tf.check_numerics(loss_tensor,
                                    '%s is inf or nan.' % loss_name,
                                    name='Loss/' + loss_name)
    tf.losses.add_loss(loss_tensor)
    if update_schedule is not None:
      for name, _, losses in update_schedule:
        if loss_name in losses:
          tf.losses.add_loss(loss_tensor, loss_collection=name)


def train(create_tensor_dict_fn, create_model_fn, train_config, master, task,
          num_clones, worker_replicas, clone_on_cpu, ps_tasks, worker_job_name,
          is_chief, train_dir, num_examples, total_configs, model_config, is_first_training=True):
  """Training function for detection models.

  Args:
    create_tensor_dict_fn: a function to create a tensor input dictionary.
    create_model_fn: a function that creates a DetectionModel and generates
                     losses.
    train_config: a train_pb2.TrainConfig protobuf.
    master: BNS name of the TensorFlow master to use.
    task: The task id of this training instance.
    num_clones: The number of clones to run per machine.
    worker_replicas: The number of work replicas to train with.
    clone_on_cpu: True if clones should be forced to run on CPU.
    ps_tasks: Number of parameter server tasks.
    worker_job_name: Name of the worker job.
    is_chief: Whether this replica is the chief replica.
    train_dir: Directory to write checkpoints and training summaries to.
    num_examples: The number of examples in dataset for training.
    total_configs: config list
  """

  detection_model = create_model_fn()
  data_augmentation_options = [
      preprocessor_builder.build(step)
      for step in train_config.data_augmentation_options]

  with tf.Graph().as_default():
    # Build a configuration specifying multi-GPU and multi-replicas.
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=clone_on_cpu,
        replica_id=task,
        num_replicas=worker_replicas,
        num_ps_tasks=ps_tasks,
        worker_job_name=worker_job_name)

    # Place the global step on the device storing the variables.
    with tf.device(deploy_config.variables_device()):
      if is_first_training:
        global_step = slim.create_global_step()
      else:
        prev_global_step = int(train_config.fine_tune_checkpoint.split('-')[-1])
        global_step = variable_scope.get_variable(
            ops.GraphKeys.GLOBAL_STEP,
            dtype=dtypes.int64,
            initializer=tf.constant(prev_global_step, dtype=dtypes.int64),
            trainable=False,
            collections=[ops.GraphKeys.GLOBAL_VARIABLES,
                         ops.GraphKeys.GLOBAL_STEP])

    with tf.device(deploy_config.inputs_device()):
      input_queue = _create_input_queue(train_config.batch_size // num_clones,
                                        create_tensor_dict_fn,
                                        train_config.batch_queue_capacity,
                                        train_config.num_batch_queue_threads,
                                        train_config.prefetch_queue_capacity,
                                        data_augmentation_options,
                                        ignore_options=train_config.ignore_options,
                                        mtl_window=model_config.mtl.window,
                                        mtl_edgemask=model_config.mtl.edgemask
                                        )

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    global_summaries = set([])

    kwargs = {}
    kwargs['mtl'] = model_config.mtl

    update_schedule = None
    model_fn = functools.partial(_create_losses,
                                 create_model_fn=create_model_fn,
                                 show_image_summary=train_config.show_image_summary,
                                 update_schedule=update_schedule,
                                 **kwargs)
    clones = model_deploy.create_clones(deploy_config, model_fn, [input_queue])
    first_clone_scope = clones[0].scope
    with tf.device(deploy_config.optimizer_device()):
      training_optimizer = optimizer_builder.build(train_config.optimizer,
                                                     global_summaries)

    sync_optimizer = None
    if train_config.sync_replicas:
      # TODO: support syncrhonous update for manual loss update
      training_optimizer = tf.SyncReplicasOptimizer(
          training_optimizer,
          replicas_to_aggregate=train_config.replicas_to_aggregate,
          total_num_replicas=train_config.worker_replicas)
      sync_optimizer = training_optimizer

    # Create ops required to initialize the model from a given checkpoint.
    init_fn = None
    if train_config.fine_tune_checkpoint:
      var_map = detection_model.restore_map(
          from_detection_checkpoint=train_config.from_detection_checkpoint,
          restore_box_predictor=train_config.restore_box_predictor,
          restore_window=train_config.restore_window,
          restore_edgemask=train_config.restore_edgemask,
          restore_closeness=train_config.restore_closeness,
          restore_mtl_refine=train_config.restore_mtl_refine,
      )
      available_var_map = (variables_helper.
                           get_variables_available_in_checkpoint(
                               var_map, train_config.fine_tune_checkpoint))
      init_saver = tf.train.Saver(available_var_map)

      mtl = model_config.mtl
      mtl_init_saver_list = []
      def _get_mtl_init_saver(scope_name):
        _var_map = detection_model._feature_extractor.mtl_restore_from_classification_checkpoint_fn(scope_name)
        if train_config.from_detection_checkpoint:
          _var_map_new = dict()
          for name, val in _var_map.iteritems():
            _var_map_new[detection_model.second_stage_feature_extractor_scope + '/' + name] = val
          _var_map = _var_map_new
        _available_var_map = (variables_helper.get_variables_available_in_checkpoint(
          _var_map, train_config.fine_tune_checkpoint))
        if _available_var_map:
          return tf.train.Saver(_available_var_map)
        else:
          return None

      # if mtl.share_second_stage_init and mtl.shared_feature == 'proposal_feature_maps':
      if mtl.share_second_stage_init and train_config.from_detection_checkpoint == False:
        if mtl.window:
          mtl_init_saver_list.append(_get_mtl_init_saver(detection_model.window_box_predictor_scope))
        if mtl.closeness:
          mtl_init_saver_list.append(_get_mtl_init_saver(detection_model.closeness_box_predictor_scope))
        if mtl.edgemask:
          mtl_init_saver_list.append(_get_mtl_init_saver(detection_model.edgemask_predictor_scope))

      def initializer_fn(sess):
        init_saver.restore(sess, train_config.fine_tune_checkpoint)
        for mtl_init_saver in mtl_init_saver_list:
          if not mtl_init_saver == None:
            mtl_init_saver.restore(sess, train_config.fine_tune_checkpoint)

      init_fn = initializer_fn

    def _get_trainable_variables(except_scopes=None):
      trainable_variables = tf.trainable_variables()
      if except_scopes is None:
        return trainable_variables
      for var in tf.trainable_variables():
        if any([scope in var.name for scope in except_scopes]):
          trainable_variables.remove(var)
      return trainable_variables

    def _get_update_ops(except_scopes=None):
      # Gather update_ops from the first clone. These contain, for example,
      # the updates for the batch_norm variables created by model_fn.
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
      if except_scopes is None:
        return update_ops
      for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope):
        if any([scope in var.name for scope in except_scopes]):
          update_ops.remove(var)
      return update_ops

    with tf.device(deploy_config.optimizer_device()):
      def _single_update():
        kwargs = {}
        _training_optimizer = training_optimizer
        kwargs['var_list'] = None
        update_ops = _get_update_ops()
        total_loss, grads_and_vars = model_deploy.optimize_clones(
            clones, _training_optimizer, regularization_losses=None, **kwargs)

        # Optionaly multiply gradients by train_config.{grad_multiplier,
        # divide_grad_by_batch}.
        if train_config.grad_multiplier or train_config.divide_grad_by_batch:
          base_multiplier = train_config.grad_multiplier \
              if train_config.grad_multiplier else 1.0
          batch_divider = float(train_config.batch_size) \
              if train_config.divide_grad_by_batch else 1.0
          total_multiplier = base_multiplier / batch_divider
          grads_and_vars = variables_helper.multiply_gradients_by_scalar_multiplier(
              grads_and_vars,
              multiplier=total_multiplier)

        # Optionally multiply bias gradients by train_config.bias_grad_multiplier.
        if train_config.bias_grad_multiplier:
          biases_regex_list = ['.*/biases']
          grads_and_vars = variables_helper.multiply_gradients_matching_regex(
              grads_and_vars,
              biases_regex_list,
              multiplier=train_config.bias_grad_multiplier)

        # Optionally freeze some layers by setting their gradients to be zero.
        if train_config.freeze_variables:
          grads_and_vars = variables_helper.freeze_gradients_matching_regex(
              grads_and_vars, train_config.freeze_variables)

        # Optionally clip gradients
        if train_config.gradient_clipping_by_norm > 0:
          with tf.name_scope('clip_grads'):
            grads_and_vars = slim.learning.clip_gradient_norms(
                grads_and_vars, train_config.gradient_clipping_by_norm)

        # Create gradient updates.
        grad_updates = _training_optimizer.apply_gradients(grads_and_vars,
                                                           global_step=global_step)
        # update_ops.append(grad_updates)
        total_update_ops = update_ops + [grad_updates]

        update_op = tf.group(*total_update_ops)
        with tf.control_dependencies([update_op]):
          train_tensor = tf.identity(total_loss, name=('train_op'))
        return train_tensor

      train_tensor = _single_update()

    # Add summaries.
    def _get_total_loss_with_collection(collection,
                                        add_regularization_losses=True,
                                        name="total_loss"):
      losses = tf.losses.get_losses(loss_collection=collection)
      if add_regularization_losses:
        losses += tf.losses.get_regularization_losses()
      return math_ops.add_n(losses, name=name)

    for model_var in slim.get_model_variables():
      global_summaries.add(tf.summary.histogram(model_var.op.name, model_var))
    for loss_tensor in tf.losses.get_losses():
      global_summaries.add(tf.summary.scalar(loss_tensor.op.name, loss_tensor))
    global_summaries.add(
        tf.summary.scalar('TotalLoss', tf.losses.get_total_loss()))

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))
    summaries |= global_summaries

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # not contained in global_summaries
    config_summary_list = select_config_summary_list(total_configs, as_matrix=False)

    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)

    # Save checkpoints regularly.
    keep_checkpoint_every_n_hours = train_config.keep_checkpoint_every_n_hours
    saver = tf.train.Saver(
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    custom_learning.train(
        train_tensor,
        logdir=train_dir,
        master=master,
        is_chief=is_chief,
        global_step=(None if is_first_training else global_step),
        session_config=session_config,
        startup_delay_steps=train_config.startup_delay_steps,
        init_fn=init_fn,
        summary_op=summary_op,
        number_of_steps=(
            train_config.num_steps if train_config.num_steps else None),
        log_every_n_steps=(
            train_config.log_every_n_steps if train_config.log_every_n_steps else None),
        save_summaries_secs=train_config.save_summaries_secs,
        save_interval_secs=train_config.save_interval_secs,
        sync_optimizer=sync_optimizer,
        saver=saver,
        batch_size=train_config.batch_size,
        num_examples=num_examples,
        config_summary_list=config_summary_list)

def select_config_summary_list(total_configs, as_matrix=True):
  def as_text_matrix(dic):
    return [[k, str(w)] for k, w in sorted(dic.items())]

  def config_to_md_text(config, indent=2):
    if config is None: return ""
    text = text_format.MessageToString(config, indent=indent, float_format='.2g')
    text = text.replace("\n", "<br>").replace(" ", "&nbsp;")
    return text

  if len(total_configs) == 5: # pipeline_config_path
    model_config, train_config, input_config, eval_config, eval_input_config = total_configs
  else:
    model_config, train_config, input_config = total_configs
    eval_config = None
    eval_input_config = None

  model_name = model_config.WhichOneof('model').lower()
  if model_name == 'faster_rcnn':
    model = model_config.faster_rcnn
  elif model_name == 'ssd':
    model = model_config.ssd
  else:
    raise ValueError('unknown model: %s'%(model_config.WhichOneof('model')))

  if as_matrix:
    resizer_name = model.image_resizer.WhichOneof('image_resizer_oneof')
    if resizer_name == 'keep_aspect_ratio_resizer':
      resizer = model.image_resizer.keep_aspect_ratio_resizer
      val_resizer = 'min(%d), max(%d)'%(resizer.min_dimension, resizer.max_dimension)
    elif resizer_name == 'fixed_shape_resizer':
      resizer = model.image_resizer.fixed_shape_resizer
      val_resizer = '(%d, %d)' % (resizer.width, resizer.height)

    # model_config
    model_dict = dict()
    model_dict['feature_extractor'] = str(model.feature_extractor.type)
    model_dict[resizer_name] = str(val_resizer)
    model_dict['num_classes'] = str(model.num_classes)
    model_config_text = as_text_matrix(model_dict)

    # train_config
    train_dict = dict()
    train_dict['batch_size'] = str(train_config.batch_size)
    train_dict['optimizer'] = str(train_config.optimizer.WhichOneof('optimizer'))
    if train_config.gradient_clipping_by_norm > 0:
      train_dict['grad_clip_norm'] = str(train_config.gradient_clipping_by_norm)
      train_dict['data_augmentation'] = (', ').join([str(step.WhichOneof('preprocessing_step'))
                                                     for step
                                                     in train_config.data_augmentation_options])
    train_config_text = as_text_matrix(train_dict)

    # input_config
    input_dict = dict()
    input_dict['input_path'] = str(input_config.tf_record_input_reader.input_path)
    train_input_config_text = as_text_matrix(input_dict)

    # eval_config
    eval_dict = dict()
    if eval_config is not None:
      eval_dict['num_examples'] = str(eval_config.num_examples)
      eval_dict['eval_interval_secs'] = str(eval_config.eval_interval_secs)
      eval_dict['nms_type'] = str(eval_config.nms_type)
      eval_dict['nms_threshold'] = str(eval_config.nms_threshold)
      eval_dict['soft_nms_sigma'] = str(eval_config.soft_nms_sigma)
    eval_config_text = as_text_matrix(eval_dict)

    # eval_input_config
    eval_input_dict = dict()
    if eval_input_config is not None:
      eval_input_dict['input_path'] = str(eval_input_config.tf_record_input_reader.input_path)
    eval_input_config_text = as_text_matrix(eval_input_dict)

  else: # print all as json format
    model_config_text = config_to_md_text(model_config)
    train_config_text = config_to_md_text(train_config)
    train_input_config_text = config_to_md_text(input_config)
    eval_config_text = config_to_md_text(eval_config)
    eval_input_config_text = config_to_md_text(eval_input_config)

  model_config_summary = tf.summary.text('ModelConfig', tf.convert_to_tensor(model_config_text), collections=[])
  train_config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(train_config_text), collections=[])
  train_input_config_summary = tf.summary.text('TrainInputConfig', tf.convert_to_tensor(train_input_config_text), collections=[])
  eval_config_summary = tf.summary.text('EvalConfig', tf.convert_to_tensor(eval_config_text), collections=[])
  eval_input_config_summary = tf.summary.text('EvalInputConfig', tf.convert_to_tensor(eval_input_config_text), collections=[])
  return model_config_summary, train_config_summary, train_input_config_summary, eval_config_summary, eval_input_config_summary
