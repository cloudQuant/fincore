__version__ = "0.1"

# NumPy 2.0 dropped the ``np.unicode_`` alias which some optional
# dependencies (for example, older versions of ``numexpr`` or PyMC) still
# import. Provide a minimal forward-compat shim so importing fincore does
# not fail when those packages are present in the environment.
import numpy as _np

if not hasattr(_np, "unicode_"):
    _np.unicode_ = _np.str_

__all__ = ["Empyrical", "Pyfolio", "analyze"]

# ---------------------------------------------------------------------------
# Lazy imports — defer heavy submodules until first attribute access.
# ``from fincore import empyrical`` still works because Python resolves
# sub-module names before calling ``__getattr__``.
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    if name == "Empyrical":
        from .empyrical import Empyrical
        globals()["Empyrical"] = Empyrical
        return Empyrical
    if name == "Pyfolio":
        from .pyfolio import Pyfolio
        globals()["Pyfolio"] = Pyfolio
        return Pyfolio
    if name == "analyze":
        from .core.context import analyze
        globals()["analyze"] = analyze
        return analyze
    raise AttributeError(f"module 'fincore' has no attribute {name!r}")

