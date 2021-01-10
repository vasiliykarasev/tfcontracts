from . import type_checking_contract
from . import combined_contract
from . import dtype_contract
from . import contract
from . import errors

# Any externally usable contract should be derived here.
TypeCheckingContract = type_checking_contract.TypeCheckingContract
DTypeContract = dtype_contract.DTypeContract
CombinedContract = combined_contract.CombinedContract

# Cannot be used directly, but users may wish to derive from this.
FunctionContract = contract.FunctionContract
