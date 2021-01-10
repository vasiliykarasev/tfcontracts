import inspect
from typing import Any, Dict, Sequence


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
