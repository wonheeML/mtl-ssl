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

"""Function to build box predictor from configuration."""

from object_detection.builders import hyperparams_builder
from object_detection.core import mask_predictor
from object_detection.protos import box_predictor_pb2


def build(argscope_fn, mask_predictor_config, is_training, num_classes, reuse_weights=None, channels=1):
  """Builds box predictor based on the configuration.

  Builds box predictor based on the configuration. See box_predictor.proto for
  configurable options. Also, see box_predictor.py for more details.

  Args:
    argscope_fn: A function that takes the following inputs:
        * hyperparams_pb2.Hyperparams proto
        * a boolean indicating if the model is in training mode.
      and returns a tf slim argscope for Conv and FC hyperparameters.
    box_predictor_config: box_predictor_pb2.BoxPredictor proto containing
      configuration.
    is_training: Whether the models is in training mode.
    num_classes: Number of classes to predict.

  Returns:
    box_predictor: box_predictor.BoxPredictor object.

  Raises:
    ValueError: On unknown box predictor.
  """

  conv_hyperparams = argscope_fn(mask_predictor_config.conv_hyperparams, is_training)

  box_predictor_object = mask_predictor.MaskPredictor(
      conv_hyperparams=conv_hyperparams,
      is_training=is_training,
      num_classes=num_classes,
      kernel_size=mask_predictor_config.kernel_size,
      reuse_weights=reuse_weights,
      channels=channels)
  return box_predictor_object

