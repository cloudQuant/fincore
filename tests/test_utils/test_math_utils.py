from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

import numpy as np

import fincore.utils.math_utils as mu


def test_nanmean_supports_out_param() -> None:
    arr = np.array([1.0, np.nan, 3.0])
    out = np.array(0.0)
    mu.nanmean(arr, out=out)
    assert float(out) == 2.0


def test_math_utils_fallback_without_bottleneck(monkeypatch) -> None:
    # Execute the module in an isolated namespace while forcing an ImportError
    # for bottleneck. Coverage still maps by filename, so this covers the
    # ImportError fallback branch.
    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "bottleneck":
            raise ImportError("no bottleneck")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)

    path = Path(mu.__file__)
    spec = importlib.util.spec_from_file_location("_tmp_math_utils_no_bn", path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]

    assert m.nanmean is np.nanmean
    assert float(m.nanmean(np.array([1.0, np.nan, 3.0]))) == 2.0
