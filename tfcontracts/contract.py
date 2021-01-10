import abc
from typing import Any, Callable


class FunctionContract(abc.ABC):
  """Base class for contracts that operate on functions.

  Contracts that operate on functions should derive from this class.
  A contract is enforced by verifying preconditions and postconditions.
  When verifying pre-conditions one typically checks input arguments.
  While verifying post-conditions, one typically verifies function return
  values. Derived class may omit either of the two steps.
  The contract is applied to a function as a decorator; for example to start
  checking function "my_func()" using a contract "MySafeContract", simply add:

  >>> @MySafeContract()
  >>> def my_func(x, y):
  >>>   # function body
  """
  def __init__(self) -> None:
    pass

  def check_precondition(self, func: Callable[..., Any], *args,
                         **kwargs) -> None:
    """Checks that function arguments satisfy preconditions."""
    pass

  def check_postcondition(self, func_results: Any,
                          func: Callable[..., Any]) -> None:
    """Checks that function arguments satisfy postconditions."""
    pass

  def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args, **kwargs):
      self.check_precondition(func, *args, **kwargs)
      results = func(*args, **kwargs)
      self.check_postcondition(results, func)
      return results

    return wrapper
