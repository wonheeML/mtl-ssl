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

r"""Training executable for detection models.

This executable is used to train DetectionModels. There are two ways of
configuring the training job:

1) A single pipeline_pb2.TrainEvalPipelineConfig configuration file
can be specified by --pipeline_config_path.

Example usage:
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --pipeline_config_path=pipeline_config.pbtxt

2) Three configuration files can be provided: a model_pb2.DetectionModel
configuration file to define what type of DetectionModel is being trained, an
input_reader_pb2.InputReader file to specify what training data will be used and
a train_pb2.TrainConfig file to configure training parameters.

Example usage:
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --model_config_path=model_config.pbtxt \
        --train_config_path=train_config.pbtxt \
        --input_config_path=train_input_config.pbtxt
"""

import functools
import json
import os
import time
import tempfile
import tensorflow as tf
import glob

from google.protobuf import text_format

from object_detection import trainer
from object_detection.builders import input_reader_builder
from object_detection.builders import model_builder
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import pipeline_pb2
from object_detection.protos import train_pb2
from global_utils import custom_utils as utils

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')
flags.DEFINE_integer('task', 0, 'task id')
flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy per worker.')
flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                     'replicas.')
flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')
flags.DEFINE_string('train_dir', '',
                    'Directory to save the checkpoints and training summaries.')
flags.DEFINE_string("train_tag", "",
                    """A simple string for discerning a training instance.""")

flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')
flags.DEFINE_string('pipeline_config_dir_path', '',
                    'Path to a directory of pipeline_pb2.TrainEvalPipelineConfig '
                    'config files. pipeline_config_path has priority.')

flags.DEFINE_string('train_config_path', '',
                    'Path to a train_pb2.TrainConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')
flags.DEFINE_string('train_label', '',
                    'Testing Label')
FLAGS = flags.FLAGS


def get_configs_from_pipeline_file():
  """Reads training configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Reads training config from file specified by pipeline_config_path flag.

  Returns:
    model_config: model_pb2.DetectionModel
    train_config: train_pb2.TrainConfig
    input_config: input_reader_pb2.InputReader
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

  model_config = pipeline_config.model
  train_config = pipeline_config.train_config
  input_config = pipeline_config.train_input_reader

  # for post evaluation
  eval_config = pipeline_config.eval_config
  eval_input_config = pipeline_config.eval_input_reader

  return model_config, train_config, input_config, eval_config, eval_input_config


def get_configs_from_multiple_files():
  """Reads training configuration from multiple config files.

  Reads the training config from the following files:
    model_config: Read from --model_config_path
    train_config: Read from --train_config_path
    input_config: Read from --input_config_path

  Returns:
    model_config: model_pb2.DetectionModel
    train_config: train_pb2.TrainConfig
    input_config: input_reader_pb2.InputReader
  """
  train_config = train_pb2.TrainConfig()
  with tf.gfile.GFile(FLAGS.train_config_path, 'r') as f:
    text_format.Merge(f.read(), train_config)

  model_config = model_pb2.DetectionModel()
  with tf.gfile.GFile(FLAGS.model_config_path, 'r') as f:
    text_format.Merge(f.read(), model_config)

  input_config = input_reader_pb2.InputReader()
  with tf.gfile.GFile(FLAGS.input_config_path, 'r') as f:
    text_format.Merge(f.read(), input_config)

  return model_config, train_config, input_config

def get_configs_from_dir():
  """Reads training configurations from the given directory path.

  Reads mutilple training configs from the given directory path.
  Training configs are stored as a list in alphanumeric order.

  Returns:
      model_configs: a list of model_pb2.DetectionModel
      train_configs: a list of train_pb2.TrainConfig
      input_configs: a list of input_reader_pb2.InputReader
  """
  pipeline_config_paths = sorted(glob.glob(os.path.join(FLAGS.pipeline_config_dir_path, '*.config')))
  if not pipeline_config_paths:
      raise ValueError('No config is in %s'%(FLAGS.pipeline_config_dir_path))

  model_configs = []
  train_configs = []
  input_configs = []
  eval_configs = []
  eval_input_configs = []

  for pipeline_config_path in pipeline_config_paths:
      pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

      with tf.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

      model_configs.append(pipeline_config.model)
      train_configs.append(pipeline_config.train_config)
      input_configs.append(pipeline_config.train_input_reader)

      # for post evaluation
      eval_configs.append(pipeline_config.eval_config)
      eval_input_configs.append(pipeline_config.eval_input_reader)

  return model_configs, train_configs, input_configs, eval_configs, eval_input_configs


def main(_):
  if FLAGS.train_label:
    FLAGS.pipeline_config_path = '../configs/test/' + FLAGS.train_label + '.config'
    FLAGS.train_dir = '../checkpoints/train/' + FLAGS.train_label
    FLAGS.train_tag = FLAGS.train_label

  if FLAGS.pipeline_config_dir_path:
    model_configs, train_configs, input_configs, eval_configs, eval_input_configs = get_configs_from_dir()
  else:
    total_configs = get_configs_from_pipeline_file()
    if FLAGS.pipeline_config_path:
      model_config, train_config, input_config, eval_config, eval_input_config = total_configs
    else:
      model_config, train_config, input_config = total_configs

  if not FLAGS.train_dir:
    root_dir = utils.get_tempdir()
    dataset = os.path.basename(input_config.label_map_path).split('_')[0].upper()
    tempfile.tempdir = utils.mkdir_p(os.path.join(root_dir, dataset))
    meta_architecture = model_config.WhichOneof('model')
    model_name = meta_architecture.upper()
    tempfile.tempdir = utils.mkdir_p(os.path.join(tempfile.tempdir, model_name))
    if meta_architecture == 'ssd':
      meta_config = model_config.ssd
    elif meta_architecture == 'faster_rcnn':
      meta_config = model_config.faster_rcnn
    else:
      raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))
    feature_extractor = meta_config.feature_extractor.type
    backbone_name = feature_extractor.replace(meta_architecture, '').lstrip('_').upper()
    tempfile.tempdir = utils.mkdir_p(os.path.join(tempfile.tempdir, backbone_name))

    train_prefix = "small-%s-" % time.strftime("%Y%m%d-%H%M%S")
    FLAGS.train_dir = tempfile.mkdtemp(suffix="-" + FLAGS.train_tag,
                                       prefix=train_prefix)
  if not os.path.exists(FLAGS.train_dir):
    os.makedirs(FLAGS.train_dir)

  # Save configuration
  def _save_config(config, prefix):
    config_str = text_format.MessageToString(config)
    save_path = os.path.join(FLAGS.train_dir, prefix + '.config')
    with open(save_path, 'w') as f:
      f.write(config_str)


  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  cluster_data = env.get('cluster', None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
  task_data = env.get('task', None) or {'type': 'master', 'index': 0}
  task_info = type('TaskSpec', (object,), task_data)

  # Parameters for a single worker.
  ps_tasks = 0
  worker_replicas = 1
  worker_job_name = 'lonely_worker'
  task = 0
  is_chief = True
  master = ''

  if cluster_data and 'worker' in cluster_data:
    # Number of total worker replicas include "worker"s and the "master".
    worker_replicas = len(cluster_data['worker']) + 1
  if cluster_data and 'ps' in cluster_data:
    ps_tasks = len(cluster_data['ps'])

  if worker_replicas > 1 and ps_tasks < 1:
    raise ValueError('At least 1 ps task is needed for distributed training.')

  if worker_replicas >= 1 and ps_tasks > 0:
    # Set up distributed training.
    server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                            job_name=task_info.type,
                            task_index=task_info.index)
    if task_info.type == 'ps':
      server.join()
      return

    worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
    task = task_info.index
    is_chief = (task_info.type == 'master')
    master = server.target

  if not FLAGS.pipeline_config_dir_path:
    # Not consecutive training
    _save_config(model_config, 'model')
    _save_config(train_config, 'train')
    _save_config(input_config, 'train_input')
    if FLAGS.pipeline_config_path:
      _save_config(eval_config, 'eval')
      _save_config(eval_input_config, 'eval_input')

    model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=True)

    create_input_dict_fn = functools.partial(
        input_reader_builder.build, input_config)
    num_examples = sum(1 for _ in tf.python_io.tf_record_iterator(
        input_config.tf_record_input_reader.input_path))

    trainer.train(create_input_dict_fn, model_fn, train_config, master, task,
                  FLAGS.num_clones, worker_replicas, FLAGS.clone_on_cpu, ps_tasks,
                  worker_job_name, is_chief, FLAGS.train_dir, num_examples,
                  total_configs=total_configs, model_config=model_config)
  else:
    # Consecutive training
    num_of_configs = len(model_configs)

    for config_index in range(num_of_configs) :
      model_config = model_configs[config_index]
      train_config = train_configs[config_index]
      input_config = input_configs[config_index]
      eval_config = eval_configs[config_index]
      eval_input_config = eval_input_configs[config_index]
      total_configs = (model_config, train_config, input_config, eval_config, eval_input_config)

      _save_config(model_config, 'model')
      _save_config(train_config, 'train')
      _save_config(input_config, 'train_input')
      _save_config(eval_config, 'eval')
      _save_config(eval_input_config, 'eval_input')

      model_fn = functools.partial(
          model_builder.build,
          model_config=model_config,
          is_training=True)

      create_input_dict_fn = functools.partial(
          input_reader_builder.build, input_config)
      num_examples = sum(1 for _ in tf.python_io.tf_record_iterator(
          input_config.tf_record_input_reader.input_path))

      trainer.train(create_input_dict_fn, model_fn, train_config, master, task,
                    FLAGS.num_clones, worker_replicas, FLAGS.clone_on_cpu, ps_tasks,
                    worker_job_name, is_chief, FLAGS.train_dir, num_examples,
                    total_configs=total_configs, is_first_training=(True if config_index==0 else False))

      def _is_last_training():
          return config_index == num_of_configs-1

      if _is_last_training():
          break

      # Remove all the files except events files in train_dir for the next training.
      for f in os.listdir(FLAGS.train_dir):
        path_to_file = os.path.join(FLAGS.train_dir, f)
        if os.path.isfile(path_to_file) and not f.startswith('events'):
          os.remove(path_to_file)

if __name__ == '__main__':
  tf.app.run()
