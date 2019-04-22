"""Operations for [N, 4] or [B, N, 4] float tensor representing bounding boxes."""
import tensorflow as tf

from object_detection.utils import shape_utils


def area(boxes, scope=None):
  """Computes area of boxes.

  Args:
    boxes: [N, 4] float tensor.
    scope: name scope.

  Returns:
    a tensor with shape [N] representing box areas.
  """
  with tf.name_scope(scope, 'Area'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


def height_width(boxes, scope=None):
  """Computes height and width of boxes in boxlist.

  Args:
    boxes: [N, 4] float tensor.
    scope: name scope.

  Returns:
    Height: A tensor with shape [N] representing box heights.
    Width: A tensor with shape [N] representing box widths.
  """
  with tf.name_scope(scope, 'HeightWidth'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=-1)
    return tf.squeeze(y_max - y_min, [-1]), tf.squeeze(x_max - x_min, [-1])


def scale(boxes, y_scale, x_scale, scope=None):
  """scale box coordinates in x and y dimensions.

  Args:
    boxes: [N, 4] float tensor.
    y_scale: (float) scalar tensor
    x_scale: (float) scalar tensor
    scope: name scope.

  Returns:
    scaled_boxes: N boxes.
  """
  with tf.name_scope(scope, 'Scale'):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    y_min, x_min, y_max, x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxes = tf.concat([y_min, x_min, y_max, x_max], 1)
    return scaled_boxes

def get_center_coordinates_and_sizes(boxes, except_center=False, scope=None):
  """Computes the center coordinates, height and width of the boxes.

  Args:
    boxes: [N, 4] float tensor.
    scope: name scope of the function.

  Returns:
    a list of 4 1-D tensors [ycenter, xcenter, height, width].
  """
  with tf.name_scope(scope, 'get_center_coordinates_and_sizes'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=1)
    # ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(boxes))
    width = x_max - x_min
    height = y_max - y_min
    y_center = y_min + height / 2.
    x_center = x_min + width / 2.
    if except_center:
      return tf.concat([height, width], 1)
    return tf.concat([y_center, x_center, height, width], 1)

def to_absolute_coordinates(boxes, height, width,
                            check_range=True, scope=None):
  """Converts normalized box coordinates to absolute pixel coordinates.

  This function raises an assertion failed error when the maximum box coordinate
  value is larger than 1.01 (in which case coordinates are already absolute).

  Args:
    boxes: [N, 4] float tensor with coordinates in range [0, 1].
    height: Maximum value for height of absolute box coordinates.
    width: Maximum value for width of absolute box coordinates.
    check_range: If True, checks if the coordinates are normalized or not.
    scope: name scope.

  Returns:
    boxes with absolute coordinates in terms of the image size.
  """
  with tf.name_scope(scope, 'ToAbsoluteCoordinates'):
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

    # Ensure range of input boxes is correct.
    if check_range:
      box_maximum = tf.reduce_max(boxes)
      max_assert = tf.Assert(tf.greater_equal(1.01, box_maximum),
                             ['maximum box coordinate value is larger '
                              'than 1.01: ', box_maximum])
      with tf.control_dependencies([max_assert]):
        width = tf.identity(width)

    return scale(boxes, height, width)


def get_small_box_indices(boxes, max_threshold, min_threshold=None, image_shape=None, by_area=True,
                          as_boolean_mask=True, scope=None):
  """Get indices of small boxes.

  Args:
    boxes: [N, 4] float tensor which representing boxes.
    max_threshold: Maximum (width AND height) or (area) of box to get.
    image_shape: if given, convert box coordinates to absolute values.
    by_area: by area or length.
    as_boolean_mask: return as boolean mask.
    scope: name scope.

  Returns:
    Small boxes' indices or boolean mask.
  """
  with tf.name_scope(scope, 'IndicesSmallBoxes'):
    if image_shape is not None:
      boxes = to_absolute_coordinates(boxes, image_shape[1], image_shape[2])
    if by_area:
      areas = area(boxes)
      if min_threshold is None:
        is_valid = tf.less_equal(areas, max_threshold)
      else:
        is_valid = tf.logical_and(tf.less_equal(areas, max_threshold),
                                  tf.greater_equal(areas, min_threshold))
    else:
      heights, widths = height_width(boxes)
      if min_threshold is None:
        is_valid = tf.logical_and(tf.less_equal(widths, max_threshold),
                                  tf.less_equal(heights, max_threshold))
      else:
        cond_less = tf.logical_and(tf.less_equal(widths, max_threshold),
                                  tf.less_equal(heights, max_threshold))
        cond_greater = tf.logical_and(tf.greater_equal(widths, min_threshold),
                                  tf.greater_equal(heights, min_threshold))
        is_valid = tf.logical_and(cond_less, cond_greater)

    if as_boolean_mask:
      return is_valid
    else:
      return tf.to_int32(tf.reshape(tf.where(is_valid), [-1]))


def get_large_box_indices(boxes, max_threshold, min_threshold=None, image_shape=None, by_area=True,
                          as_boolean_mask=True, scope=None):
  """Get indices of large boxes.

  Args:
    boxes: [N, 4] float tensor which representing boxes.
    min_threshold: Minimum (width OR height) or (area) of box to get.
    image_shape: if given, convert box coordinates to absolute values.
    by_area: by area or length.
    as_boolean_mask: return as boolean mask.
    scope: name scope.

  Returns:
    Large boxes' indices or boolean mask.
  """
  with tf.name_scope(scope, 'IndicesLargeBoxes'):
    if image_shape is not None:
      boxes = to_absolute_coordinates(boxes, image_shape[1], image_shape[2])
    if by_area:
      areas = area(boxes)
      if min_threshold is None:
        is_valid = tf.greater(areas, max_threshold)
      else:
        is_valid = tf.logical_or(tf.less(areas, min_threshold),
                                 tf.greater(areas, max_threshold))
    else:
      heights, widths = height_width(boxes)
      if min_threshold is None:
        is_valid = tf.logical_or(tf.greater(widths, max_threshold),
                                 tf.greater(heights, max_threshold))
      else:
        cond1 = tf.logical_and(tf.greater(widths, max_threshold),
                                 tf.greater(heights, max_threshold))
        cond2 = tf.logical_and(tf.less(widths, min_threshold),
                                 tf.less(heights, min_threshold))
        is_valid = tf.logical_or(cond1, cond2)

    if as_boolean_mask:
      return is_valid
    else:
      return tf.to_int32(tf.reshape(tf.where(is_valid), [-1]))


def prune_small_boxes(boxes, min_threshold, image_shape=None, by_area=True, scope=None):
  """Prunes small boxes.

  Args:
    boxes: [N, 4] float tensor which representing boxes.
    min_threshold: Minimum (width OR height) or (area) of box to get.
    image_shape: if given, convert box coordinates to absolute values.
    by_area: by area or length.
    scope: name scope.

  Returns:
    A pruned boxes.
  """
  with tf.name_scope(scope, 'PruneSmallBoxes'):
    large_box_indices = get_large_box_indices(boxes, min_threshold,
                                              image_shape=image_shape, by_area=by_area,
                                              as_boolean_mask=False)
    return gather(boxes, large_box_indices)


def prune_large_boxes(boxes, max_threshold, image_shape=None, by_area=True, scope=None):
  """Prunes large boxes.

  Args:
    boxes: [N, 4] float tensor which representing boxes.
    max_threshold: Maximum (width AND height) or (area) of box to get.
    image_shape: if given, convert box coordinates to absolute values.
    by_area: by area or length.
    scope: name scope.

  Returns:
    A pruned boxes.
  """
  with tf.name_scope(scope, 'PruneLargeBoxes'):
    small_box_indices = get_small_box_indices(boxes, max_threshold,
                                              image_shape, by_area,
                                              as_boolean_mask=False)
    return gather(boxes, small_box_indices)


def gather(boxes, indices, scope=None):
  """Gather boxes according to indices and return new subboxes.

  Args:
    boxes: [N, 4] float tensor.
    indices: a rank-1 tensor of type int32 / int64
    scope: name scope.

  Returns:
    subboxes: [N, 4] float tensor corresponding to the subset of the input boxes
    specified by indices
  Raises:
    ValueError: if the indices are not of type int32
  """
  with tf.name_scope(scope, 'Gather'):
    if len(indices.shape.as_list()) != 1:
      raise ValueError('indices should have rank 1')
    if indices.dtype != tf.int32 and indices.dtype != tf.int64:
      raise ValueError('indices should be an int32 / int64 tensor')
    subboxes = tf.gather(boxes, indices)
    return subboxes
