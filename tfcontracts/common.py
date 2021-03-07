import inspect
from typing import Any, Dict, List, Mapping, Sequence
import tensorflow as tf


def get_function_args_as_dict(func, *args, **kwargs):
  """Retruns a dict with function arguments."""
  return bind_function_args(inspect.getfullargspec(func).args, *args, **kwargs)


def bind_function_args(argument_names: Sequence[str], *args,
                       **kwargs) -> Dict[str, Any]:
  """Returns a dict with function arguments."""
  outputs = {}
  for k, val in zip(argument_names, args):
    outputs[k] = val
  outputs.update(**kwargs)
  return outputs


def flatten_list_of_lists(list_of_lists: List[List[Any]]) -> List[Any]:
  """Flattens a list of lists."""
  output = []
  for one_list in list_of_lists:
    output += one_list
  return output


def flatten_tensor_func_args(
    func_args_by_name: Dict[str, Any]) -> Dict[str, Sequence[tf.Tensor]]:
  """Returns a dict from argument name to a sequence of tensors.

  Given arbitrary function arguments, returns a dict from argument name to a
  sequence of tensors. If a function argument doesn't contain any tensors,
  this sequence will be empty.
  This function effectively strips out any non tf.Tensor arguments, and
  also flattens the structure for easy inspection.
  As an example:

    >>> flatten_tensor_func_args(
        {'single': tensor_1, 'list': [tensor_2, tensor_3], 'scalar': 0.0})

  will return:

    >>> {'single': [tensor_1], 'list': [tensor_2, tensor_3], 'scalar': []}
  """
  output = {}
  for name, value in func_args_by_name.items():
    output[name] = _flatten_func_args_recursively(value)
  return output


def _flatten_func_args_recursively(value: Any) -> Sequence[tf.Tensor]:
  """Returns a list of tensors, given arbitrarily nested input."""
  if isinstance(value, tf.Tensor):
    # Input is a tensor, so just return a single-item list pair.
    return [value]
  elif isinstance(value, Sequence) and not isinstance(value, str):
    return flatten_list_of_lists(
        [_flatten_func_args_recursively(x) for x in value])
  elif isinstance(value, Mapping):
    return flatten_list_of_lists(
        [_flatten_func_args_recursively(x) for x in value.values()])
  else:
    # Object is not a Mapping, Sequence, or a Tensor, so skip checking.
    return []


def check_contract_args_match_function_args(
    contract_arg_names: Sequence[str],
    function_arg_names: Sequence[str],
    function_name: str) -> None:
  """Ensures that contract_arg_names is a subset of function_arg_names.

  Args:
    contract_arg_names: List of function argument names used by the contract.
      This list may include the special 'return' argument name.
    function_arg_names: List of function argument names.
    function_name: Name of the function as a string.

  Raises:
    InvalidArgumentError if one of the argument names in contract_arg_names
      isn't in function_arg_names.
  """
  if not set(contract_arg_names).issubset(
      set(function_arg_names).union({'return'})):
    raise errors.InvalidArgumentError(
        f'Your contract included assertions on arguments that arent a part of '
        f'function signature. Did you maybe make a typo? '
        f'Contract included [{", ".join(contract_arg_names)}], while the '
        f'function signature for "{function_name}()" consisted of '
        f'[{", ".join(function_arg_names)}] '
        f'(The latter should have been a superset of the former).')
