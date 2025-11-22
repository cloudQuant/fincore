__version__ = "0.1"

# NumPy 2.0 dropped the ``np.unicode_`` alias which some optional
# dependencies (for example, older versions of ``numexpr`` or PyMC) still
# import. Provide a minimal forward-compat shim so importing fincore does
# not fail when those packages are present in the environment.
import numpy as _np

if not hasattr(_np, "unicode_"):
    _np.unicode_ = _np.str_

from .empyrical import Empyrical
from .pyfolio import Pyfolio

__all__ = ["Empyrical", "Pyfolio"]

