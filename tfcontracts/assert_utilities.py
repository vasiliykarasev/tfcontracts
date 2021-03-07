"""This file contains simple addons to functions in tf.debugging namespace.

tf.debugging.* namespace provides a number of useful convenience functions, and
this package simply provides several basic and natural extensions.
"""

import tensorflow as tf

from typing import Any, Optional, Sequence, Union

Number = Union[int, float, complex]


def assert_shapes_same(inputs: Sequence[tf.Tensor],
                       data=None,
                       summarize=None,
                       message: Optional[str] = None,
                       name: Optional[str] = None) -> tf.Operation:
  """Asserts tensor shapes and dimensions are identical for all tensors.

  This op checks that a collection of tensors has the same shape.
  Internally, it simply calls tf.debugging.assert_shapes() and serves as a
  concise replacement for explicitly listing tensors and writing out their
  dimensions. 

  As an example, the following two perform the same check (and in fact, the
  latter is a bit more general).
  
  >>> tf.debugging.assert_shapes([(x, ['batch', 'height', 'width', 'ch']),
                                  (y, ['batch', 'height', 'width', 'ch']),
                                  (z, ['batch', 'height', 'width', 'ch'])])
  >>> assert_shapes_same([x, y, z]) 
  
  See tf.debugging.assert_shapes() docstring for the explanation of extra
  arguments.
  """
  if not inputs:
    return
  if not name:
    name = 'assert_shapes_same'
  symbolic_shape = [f'dim_{i}' for i in range(len(inputs[0].shape))]
  tensor_and_symbolic_shape_tuples = [(x, symbolic_shape) for x in inputs]
  return tf.debugging.assert_shapes([(x, symbolic_shape) for x in inputs],
                                    data, summarize, message, name)


def assert_in_interval(x: tf.Tensor,
                       low: Union[tf.Tensor, Number],
                       high: Union[tf.Tensor, Number],
                       message: Optional[str] = None,
                       summarize=None,
                       name: Optional[str] = None) -> tf.Operation:
  """Asserts that x is in closed [low, high] interval elementwise.

  This Op checks that `low <= x[i]` and `x[i] <= high` hold elementwise. Note
  that the interval is closed on both sides.
  """
  if not name:
    name = 'assert_in_interval'
  # This is a bit suboptimal: the error message won't include the interval and
  # will instead only describe a single failing side.
  return tf.group([
      tf.debugging.assert_greater_equal(x, low, message, summarize,
                                        name + 'low'),
      tf.debugging.assert_less_equal(x, high, message, summarize,
                                     name + 'high'),
  ])
