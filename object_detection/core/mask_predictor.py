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

"""Mask predictor for object detectors.

Mask predictors are classes that take a high level
image feature map as input and produce one prediction,
(1) a tensor encoding classwise mask.

These components are passed directly to loss functions in our detection models.
"""
import functools
from abc import abstractmethod
import tensorflow as tf
from tensorflow.python.framework import ops as framework_ops
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import static_shape
from object_detection.utils import kwargs_util


slim = tf.contrib.slim

MASK_PREDICTIONS = 'mask_predictions'


class MaskPredictor(object):
  """MaskPredictor."""

  def __init__(self, conv_hyperparams, is_training, num_classes, kernel_size, reuse_weights=None, channels=1):
    """Constructor.

    Args:
      is_training: Indicates whether the MaskPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
    """
    self._conv_hyperparams = conv_hyperparams
    self._is_training = is_training
    self._num_classes = num_classes
    self._scope = None
    self._reuse_weights = reuse_weights
    self._kernel_size = kernel_size
    self._channels = channels

  @property
  def num_classes(self):
    return self._num_classes

  def set_scope(self, scope):
    self._scope = scope

  def predict(self, image_features, scope, **params):
    """Computes encoded object locations and corresponding confidences.

    Takes a high level image feature map as input and produce one prediction,
    (1) a tensor encoding classwise mask.

    Args:
      image_features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.
      scope: Variable and Op scope name.
      **params: Additional keyword arguments for specific implementations of
              MaskPredictor.

    Returns:
      A dictionary containing at least the following tensor.

        mask_predictions: A float tensor of shape
          [batch_size, height, width, num_classes] representing the classwise mask prediction.
    """
    with tf.variable_scope(scope, reuse=self._reuse_weights) as var_scope:
      self.set_scope(var_scope)
      return self._predict(image_features, **params)

  def _predict(self, image_features):
    """Implementations must override this method.

    Args:
      image_features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.
      **params: Additional keyword arguments for specific implementations of
              MaskPredictor.

    Returns:
      A dictionary containing at least the following tensor.
        mask_predictions: A float tensor of shape
          [batch_size, height, width, num_classes] representing the classwise mask prediction.
    """

    output_depth = self._num_classes * self._channels # channels 2 is for x, y
    net = image_features
    end_points_collection = self._scope.name + '_end_points'
    with slim.arg_scope(self._conv_hyperparams), \
      slim.arg_scope([slim.conv2d],
                     trainable=self._is_training,
                     activation_fn=tf.nn.tanh,
                     normalizer_fn=None,
                     normalizer_params=None,
                     outputs_collections=end_points_collection):

      mask_predictions = slim.conv2d(net, output_depth, [self._kernel_size, self._kernel_size],
            scope='BoxEncodingPredictor')

    return {MASK_PREDICTIONS: mask_predictions}
