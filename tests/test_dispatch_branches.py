"""Branch-completion tests for fincore._dispatch projection and legacy gates."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from fincore import _dispatch
from fincore._dispatch import (
    _apply_projection,
    invoke_metric,
    invoke_prevalidated_metric,
    invoke_prevalidated_projections,
)


def test_apply_projection_frame_success() -> None:
    spec = _dispatch.get_metric_spec("metrics", "sharpe_ratio", "enhanced")
    frame = pd.DataFrame({"a": [1.0]})
    result = _apply_projection(replace(spec, result_projection="frame"), frame)
    assert result is frame


def test_apply_projection_legacy_tuple_success() -> None:
    spec = _dispatch.get_metric_spec("metrics", "sharpe_ratio", "enhanced")
    result = _apply_projection(replace(spec, result_projection="legacy_tuple"), (1.0, 2.0))
    assert result == (1.0, 2.0)


def test_invoke_metric_rejects_legacy_empyrical() -> None:
    with pytest.raises(ValueError, match="frozen facade"):
        invoke_metric("empyrical_module", "sharpe_ratio", "strict-0.6.0")


def test_invoke_prevalidated_metric_rejects_legacy() -> None:
    with pytest.raises(ValueError, match="strict compatibility"):
        invoke_prevalidated_metric("empyrical_module", "sharpe_ratio", "strict-0.6.0")


def test_invoke_prevalidated_projections_empty_names() -> None:
    assert invoke_prevalidated_projections("metrics", (), "enhanced") == {}


def test_invoke_prevalidated_projections_rejects_mixed_kernels() -> None:
    with pytest.raises(ValueError, match="share one kernel"):
        invoke_prevalidated_projections("metrics", ("sharpe_ratio", "max_drawdown"), "enhanced")


def test_invoke_prevalidated_projections_rejects_legacy() -> None:
    with pytest.raises(ValueError, match="strict compatibility"):
        invoke_prevalidated_projections("empyrical_module", ("sharpe_ratio",), "strict-0.6.0")
