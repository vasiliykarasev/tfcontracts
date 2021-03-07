import inspect

import typing
from typing import Any, Callable, Dict, Sequence, Type
from . import errors
from . import contract
from . import common


class TypeCheckingContract(contract.FunctionContract):
  """A contract that type-checks annotations against values.

  Checks that passed values satisfy constraints imposed by type annotations.
  """
  def check_precondition(self, func: Callable[..., Any], *args,
                         **kwargs) -> None:
    argspec = inspect.getfullargspec(func)
    # Map function arguments to a dict.
    func_args_as_dict = common.bind_function_args(argspec.args, *args,
                                                  **kwargs)
    check_argument_types(func_args_as_dict, argspec.annotations, func.__name__)

  def check_postcondition(self, func_results: Any,
                          func: Callable[..., Any]) -> None:
    annotations = inspect.getfullargspec(func).annotations
    # 'return' is used as an identifier of the function return value. This works
    # since the keyword is already reserved in python.
    if 'return' in annotations:
      check_argument_types({'return': func_results},
                           {'return': annotations['return']}, func.__name__)


def check_argument_types(args_values: Dict[str, Any],
                         args_annotations: Dict[str,
                                                Any], func_name: str) -> None:
  for arg_name, arg_annotation in args_annotations.items():
    if arg_name not in args_values:
      continue
    arg_value = args_values[arg_name]
    if not satisfies_type_annotation(arg_value, arg_annotation):
      # Should be invalid argument error.
      raise errors.InvalidArgumentError(
          f'You called "{func_name}()" with an argument type that did not '
          f'match type annotation for "{arg_name}". '
          f'Value "{arg_value}" is not consistent with the expected type '
          f'"{arg_annotation}".')


def satisfies_type_annotation(value: Any, annotation: Any) -> bool:
  """Returns true if the value satisfies the type annotation."""
  # TODO(vasiliy): this does not work for many situations; see:
  # https://stackoverflow.com/questions/55503673
  return isinstance(value, annotation)
