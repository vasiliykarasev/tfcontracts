from . import contract

from typing import Any, Callable, Sequence


class CombinedContract(contract.FunctionContract):
  """A contract that internally represents and enforces a contract collection.

  Example:
    >>> @CombinedContract(
    >>>    [ShapeContract(), DTypeContract(), ValueContract()])
    >>> def my_func(x, y):
    >>>   # function body
  """
  def __init__(self, contracts: Sequence[contract.FunctionContract]) -> None:
    self._contracts = contracts

  def check_precondition(self, func: Callable[..., Any], *args,
                         **kwargs) -> None:
    """Checks that function arguments satisfy preconditions."""
    for contract in self._contracts:
      contract.check_precondition(func, *args, **kwargs)

  def check_postcondition(self, func_results: Any,
                          func: Callable[..., Any]) -> None:
    """Checks that function arguments satisfy postconditions."""
    for contract in self._contracts:
      contract.check_precondition(func_results, func)
