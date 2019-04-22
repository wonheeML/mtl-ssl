from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def get_kwargs_with_initializers(scope, initializers=None):
  kwargs = {'scope': scope}
  if initializers is not None:
    if 'weights' in initializers[scope]:
      kwargs['weights_initializer'] = initializers[scope]['weights']
    if 'biases' in initializers[scope]:
      kwargs['biases_initializer'] = initializers[scope]['biases']
    if 'BatchNorm' in initializers[scope]:
      kwargs['normalizer_params'] = {
          'param_initializers': initializers[scope]['BatchNorm']
      }
  return kwargs
