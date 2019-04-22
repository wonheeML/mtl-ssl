import tensorflow as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch
from nets import vgg

slim = tf.contrib.slim

class FasterRCNNVggFeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Faster R-CNN Vgg feature extractor implementation."""

  def __init__(self,
               architecture,
               vgg_model,
               is_training,
               first_stage_features_stride,
               reuse_weights=None,
               weight_decay=0.0005,
               base_features='conv5',
               freeze_layer=''):
    """Constructor.

    Args:
      architecture: Architecture name of the Vgg model.
      vgg_model: Definition of the Vgg model.
      is_training: See base class.
      first_stage_features_stride: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If 'first_stage_features_stride' is not 8 or 16.
    """
    if first_stage_features_stride != 8 and first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 8 or 16.')
    self._architecture = architecture
    self._vgg_model = vgg_model
    self._base_features = base_features
    self._freeze_layer = freeze_layer
    super(FasterRCNNVggFeatureExtractor, self).__init__(
        is_training, first_stage_features_stride, reuse_weights, weight_decay)

  def preprocess(self, resized_inputs):
    # channel_means = [123.68, 116.779, 103.939]
    # channel_means = [122.7717, 115.9465, 102.9801]
    resized_inputs = tf.reverse(resized_inputs, [-1]) # RGB to BGR
    channel_means = [102.9801, 115.9465, 122.7717] # BGR mean
    centered_inputs = resized_inputs - [[channel_means]]
    return centered_inputs

  def _extract_proposal_features(self, preprocessed_inputs, scope):
    if len(preprocessed_inputs.get_shape().as_list()) != 4:
      raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                       'tensor of shape %s' % preprocessed_inputs.get_shape())
    shape_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
            tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    with tf.control_dependencies([shape_assert]):
      with slim.arg_scope(
          vgg.vgg_arg_scope(weight_decay=self._weight_decay)):
        with tf.variable_scope(
            self._architecture, reuse=self._reuse_weights) as var_scope:
          _, endpoints = self._vgg_model(
              preprocessed_inputs,
              final_endpoint='conv5',
              trainable=self._is_training,
              freeze_layer=self._freeze_layer,
              scope=var_scope)

    handle = self._base_features
    return endpoints[handle]

  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    with tf.variable_scope(self._architecture, reuse=self._reuse_weights):
      with slim.arg_scope(
          vgg.vgg_arg_scope(weight_decay=self._weight_decay)):
          proposal_classifier_features = tf.identity(proposal_feature_maps)
    return proposal_classifier_features

class FasterRCNNVgg16FeatureExtractor(FasterRCNNVggFeatureExtractor):
  """Faster R-CNN Vgg 16 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               reuse_weights=None,
               weight_decay=0.0005,
               base_features='conv5',
               freeze_layer=''):
    super(FasterRCNNVgg16FeatureExtractor, self).__init__(
          'vgg_16', vgg.vgg_16_base, is_training,
          first_stage_features_stride, reuse_weights, weight_decay,
          base_features, freeze_layer)
