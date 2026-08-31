"""Dense NaN-aware numeric primitives owned by metric kernels."""

from __future__ import annotations

from functools import wraps

import numpy as np

try:
    import bottleneck as _bottleneck
except ImportError:  # pragma: no cover - exercised by dependency-specific environments.
    _bottleneck = None


def _with_out(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        out = kwargs.pop("out", None)
        result = function(*args, **kwargs)
        if out is None:
            return result
        out[()] = result
        return out

    return wrapped


nanmean = _with_out(_bottleneck.nanmean) if _bottleneck is not None else np.nanmean
nanstd = _with_out(_bottleneck.nanstd) if _bottleneck is not None else np.nanstd
nansum = _with_out(_bottleneck.nansum) if _bottleneck is not None else np.nansum
nanmax = _with_out(_bottleneck.nanmax) if _bottleneck is not None else np.nanmax
nanmin = _with_out(_bottleneck.nanmin) if _bottleneck is not None else np.nanmin
nanargmax = _with_out(_bottleneck.nanargmax) if _bottleneck is not None else np.nanargmax
nanargmin = _with_out(_bottleneck.nanargmin) if _bottleneck is not None else np.nanargmin
