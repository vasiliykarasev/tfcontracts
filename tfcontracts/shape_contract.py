import typing
import tensorflow as tf

from typing import Any, Callable, Dict, Sequence, Mapping, Tuple, Union

from . import common
from . import contract
from . import errors

_ShapeSpec = Sequence[Union[str, int]]
_AnyDict = Dict[str, Any]


class ShapeContract(contract.FunctionContract):
  """Contract that ensures that all arguments match the specified shape.

  The shape can be specified as a list of ints (e.g. [1,2,3,4]), or
  symbolically (e.g. ['batch', 'height', 'width', 'channels']), or mixed.

  If an argument is listed and its shape doesn't match the one specified, an
  exception is raised. Omitting an argument is equivalent to stating that the
  "shape can be anything".

  Note that we only support static shapes here: errors in dynamic shapes are
  not caught.

  Example:
    >>> @ShapeContract(values=[
            ('x', ['b', 64, 128, 3]),
            ('y', ['b', 64, 128, 3]),
            ('return': ['b', 32, 64, 3])])
    >>> def my_func(x:tf.Tensor, y:tf.Tensor) -> tf.Tensor:
    >>>   # do stuff...
  """
  def __init__(self, values: Sequence[Tuple[str, _ShapeSpec]]) -> None:
    """
    Args:
      values: List of argument name-shape pairs. Return value is identified
        by 'return' keyword. Alternatively, could be specified as a dict from
        argument names to shapes.
    """
    self._requested_shapes_by_name = dict(values)

  def check_precondition(self, func: Callable[..., Any], *args,
                         **kwargs) -> None:
    func_args_as_dict = common.get_function_args_as_dict(func, *args, **kwargs)
    common.check_contract_args_match_function_args(
        contract_arg_names=list(self._requested_shapes_by_name.keys()),
        function_arg_names=list(func_args_as_dict.keys()),
        function_name=func.__name__)
    check_argument_shapes(func_args_as_dict, self._requested_shapes_by_name,
                          func.__name__)

  def check_postcondition(self, func_results: Any,
                          func: Callable[..., Any]) -> None:
    if 'return' not in self._requested_shapes_by_name:
      return
    check_argument_shapes({'return': func_results},
                          self._requested_shapes_by_name, func.__name__)


def check_argument_shapes(func_args: _AnyDict,
                          requested_shapes_by_name: Dict[str, _ShapeSpec],
                          func_name: str) -> None:
  tensors_and_shapes = concat_tensor_and_shape_pairs(func_args,
                                                     requested_shapes_by_name)
  try:
    tf.debugging.assert_shapes(tensors_and_shapes)
  except ValueError as e:
    requested_names_and_shapes = [(k, v)
                                  for k, v in requested_shapes_by_name.items()]
    raise errors.InvalidArgumentError(
        f'You called "{func_name}()" with values whose shapes did not match '
        f'those requested during contract creation. Requested shapes were '
        f'{requested_names_and_shapes} and actual shapes were '
        f'{tensors_and_shapes}. Details: {str(e)}.')


def concat_tensor_and_shape_pairs(
    func_args_by_name: _AnyDict, shapes_by_name: Dict[str, _ShapeSpec]
) -> Sequence[Tuple[tf.Tensor, _ShapeSpec]]:
  """Returns a list of tensor-shape pairs.

  Given a dict with function arguments and a dict with shape constraint
  specifications, returns a list of tensor-shape pairs that can be fed into
  tf.debugging.assert_shapes.
  """
  tensors_and_shapes = []
  tensors_by_arg_name = common.flatten_tensor_func_args(func_args_by_name)
  for name, tensor_list in tensors_by_arg_name.items():
    if name not in shapes_by_name:
      continue
    shape = shapes_by_name[name]
    tensors_and_shapes += [(tensor, shape) for tensor in tensor_list]
  return tensors_and_shapes
