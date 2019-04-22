import joblib
import numpy as np
import tensorflow as tf
from global_utils.custom_utils import log

def build(init_file):
  """Build initializers based on the init file"""
  initial_values = joblib.load(init_file)

  initializers = {}
  for name, values_dict in initial_values.iteritems():
    log.infov('Build initializer for layer [%s].', name)
    initializers[name] = build_layer_initializers(values_dict)
  return initializers

def build_layer_initializers(values_dict):
  layer_initializers = {}
  for k, v in values_dict.iteritems():
    if isinstance(v, np.ndarray):
      layer_initializers[k] = tf.constant_initializer(v)
    elif isinstance(v, dict):
      layer_initializers[k] = build_layer_initializers(v)
    else:
      raise ValueError('Cannot change type of [%s]: %s.' %
                       (k, type(v)))
  return layer_initializers
