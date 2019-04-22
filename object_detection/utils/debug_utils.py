from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

LOG = 'log_collection'
IMAGE = 'image_collection'


def make_print_tensor(t, message=None, first_n=None, summarize=None):
  print_tensor = tf.Print(t, [t], message=message, first_n=first_n, summarize=summarize)
  tf.add_to_collection(LOG, print_tensor)
