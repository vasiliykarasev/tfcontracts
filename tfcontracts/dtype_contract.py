import typing
import tensorflow as tf

from typing import Any, Callable, Dict, Sequence, Mapping, Union

from . import common
from . import contract
from . import errors


class SimpleDTypeContract(contract.FunctionContract):
  """Contract that ensures that all arguments match the given dtype.

  Raises an exception if an argument or the return value is a tf.Tensor and
  doesn't match the specified type.
  
  Example:
    >>> @SimpleDTypeContract(value=[tf.float32, tf.float64])
    >>> def my_func(x:tf.Tensor, y:tf.Tensor) -> tf.Tensor:
    >>>   # do stuff...

  TODO: Support something like: SameDTypeContract() (specific dtype is
  unimportant, but all arguments must match it).
  TODO: Support something like: SimpleDTypeContract(x=tf.float32, y=tf.float32),
  i.e. specify which arguments the contract applies to.
  """
  def __init__(self,
               value: Union[tf.DType, Sequence[tf.DType]],
               check_inputs=True,
               check_outputs=True) -> None:
    """
    Args:
      value: Desired dtype(s). A set of dtypes represents an "any-of" condition.
      check_inputs: if true, will check function input values.
      check_outputs: If true, will check function output values.
    """
    self._value = value
    self._check_inputs = check_inputs
    self._check_outputs = check_outputs

  def check_precondition(self, func: Callable[..., Any], *args,
                         **kwargs) -> None:
    if not self._check_inputs:
      return
    func_args_as_dict = common.get_function_args_as_dict(func, *args, **kwargs)
    check_argument_dtypes(func_args_as_dict, self._value, func.__name__)

  def check_postcondition(self, func_results: Any,
                          func: Callable[..., Any]) -> None:
    if not self._check_outputs:
      return
    check_argument_dtypes({'return': func_results}, self._value, func.__name__)


def check_argument_dtypes(func_args: Dict[str, Any],
                          desired_dtype: Union[tf.DType, Sequence[tf.DType]],
                          func_name: str) -> None:
  for name, value in func_args.items():
    if not check_argument_dtype_recursive(value, desired_dtype):
      raise errors.InvalidArgumentError(
          f'You called "{func_name}()" with an argument type that did not '
          f'match the requested data type for "{name}". '
          f'Actual dtype of "{value}" is not consisted '
          f'with the expected dtype "{desired_dtype}".')


def check_argument_dtype_recursive(
    value: Any, desired_dtype: Union[tf.DType, Sequence[tf.DType]]) -> bool:
  """Returns true if argument dtype matches the desired.

  Applies is_matching_dtype() to the input if it is a tf.Tensor. If the input
  is a "sequence" or a "dict" type, recursively applies itself.
  Returns true for any other inputs (e.g. strings, numbers, etc).
  """
  if isinstance(value, tf.Tensor):
    return is_matching_dtype(value.dtype, desired_dtype)
  elif isinstance(value, Sequence) and not isinstance(value, str):
    return all(
        [check_argument_dtype_recursive(x, desired_dtype) for x in value])
  elif isinstance(value, Mapping):
    return all([
        check_argument_dtype_recursive(x, desired_dtype)
        for x in value.values()
    ])
  else:
    # Object is not a Mapping, Sequence, or a Tensor, so skip checking.
    return True


def is_matching_dtype(
    actual_dtype: tf.DType, desired_dtype: Union[tf.DType,
                                                 Sequence[tf.DType]]) -> bool:
  """Returns true if actual type matches the desired type."""
  if isinstance(desired_dtype, Sequence):
    return any([is_matching_dtype(actual_dtype, x) for x in desired_dtype])
  else:
    return actual_dtype == desired_dtype


DTypeContract = SimpleDTypeContract
