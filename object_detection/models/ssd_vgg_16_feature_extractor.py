"""SSDFeatureExtractor for VGG16 features."""
import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from nets import vgg

slim = tf.contrib.slim


class SSDVgg16FeatureExtractor(
  ssd_meta_arch.SSDFeatureExtractor):

  def __init__(self,
               depth_multiplier,
               min_depth,
               conv_hyperparams,
               reuse_weights=None):
    """VGG16 Feature Extractor for SSD Models."""
    super(SSDVgg16FeatureExtractor, self).__init__(
      depth_multiplier, min_depth, conv_hyperparams, reuse_weights)

  def preprocess(self, resized_inputs):
    """SSD preprocessing"""
    # TODO: Subtract RGB mean instead
    return (2.0 / 255.0) * resized_inputs - 1.0

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                       tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    feature_map_layout = {
        'from_layer': ['conv4', '', '', '', '', '', ''],
        'layer_depth': [-1, 1024, 1024, 512, 256, 256, 256],
    }

    with tf.control_dependencies([shape_assert]):
      with slim.arg_scope(self._conv_hyperparams):
        with tf.variable_scope('vgg_16',
                               reuse=self._reuse_weights) as scope:
          net, image_features = vgg.vgg_16_base(
              preprocessed_inputs,
              final_endpoint='pool5',
              trainable=False,
              scope=scope)
          feature_maps = feature_map_generators.multi_resolution_feature_maps(
              feature_map_layout=feature_map_layout,
              depth_multiplier=self._depth_multiplier,
              min_depth=self._min_depth,
              insert_1x1_conv=True,
              image_features=image_features)

    return feature_maps.values()
