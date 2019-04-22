from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_layer_kwargs(scope, is_training=True, freeze_layer='', batch_norm=True,
                     initializers=None):
  kwargs = {'scope': scope}
  if is_training and freeze_layer:
    # currently assume all freezed layers are 'conv' with 'bn'.
    freeze_no = int(freeze_layer[4:])
    trainable = (int(scope[4:]) > freeze_no)
    kwargs['trainable'] = trainable # conv
    if batch_norm:
      kwargs['normalizer_params'] = { # bn
          'trainable': trainable,
          'is_training': trainable
      }
  if initializers is not None:
    if 'weights' in initializers[scope]:
      kwargs['weights_initializer'] = initializers[scope]['weights']
    if 'biases' in initializers[scope]:
      kwargs['biases_initializer'] = initializers[scope]['biases']
    if 'BatchNorm' in initializers[scope]:
      if 'normalizer_params' in kwargs:
        kwargs['normalizer_params']['param_initializers'] = initializers[scope]['BatchNorm']
      else:
        kwargs['normalizer_params'] = {
            'param_initializers': initializers[scope]['BatchNorm']
        }
  return kwargs
