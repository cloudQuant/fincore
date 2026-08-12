from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import fincore.empyrical as ep
import fincore.metrics.alpha_beta as alpha_beta_metrics
import fincore.metrics.ratios as ratios_metrics
import fincore.metrics.timing as timing_metrics
from fincore.exceptions import DataAlignmentError
from fincore.metrics.alpha_beta import (
    alpha,
    alpha_beta,
    annual_alpha,
    annual_beta,
    beta,
    down_alpha_beta,
    up_alpha_beta,
)
from fincore.metrics.basic import aligned_series as legacy_aligned_series
from fincore.metrics.ratios import (
    cal_treynor_ratio,
    capture,
    down_capture,
    down_capture_return,
    information_ratio,
    m_squared,
    treynor_ratio,
    up_capture,
    up_capture_return,
    up_down_capture,
)
from fincore.metrics.risk import beta_fragility_heuristic, residual_risk, tracking_error
from fincore.metrics.rolling import (
    roll_alpha,
    roll_alpha_beta,
    roll_beta,
    roll_down_capture,
    roll_up_capture,
    roll_up_down_capture,
    rolling_regression,
)
from fincore.metrics.stats import capm_r_squared, r_cubed, relative_win_rate, tracking_difference
from fincore.metrics.timing import cornell_timing, henriksson_merton_timing, market_timing_return, treynor_mazuy_timing
from fincore.metrics.yearly import annual_active_return, annual_active_return_by_year, information_ratio_by_year

if TYPE_CHECKING:
    from collections.abc import Callable


ENHANCED_BINARY_METRICS: tuple[Callable[..., object], ...] = (
    beta,
    alpha,
    alpha_beta,
    annual_alpha,
    annual_beta,
    information_ratio,
    cal_treynor_ratio,
    m_squared,
    capture,
    up_capture,
    down_capture,
    up_down_capture,
    up_capture_return,
    down_capture_return,
    tracking_error,
    residual_risk,
    beta_fragility_heuristic,
    roll_alpha,
    roll_beta,
    roll_alpha_beta,
    roll_up_capture,
    roll_down_capture,
    rolling_regression,
    relative_win_rate,
    r_cubed,
    capm_r_squared,
    tracking_difference,
    treynor_mazuy_timing,
    henriksson_merton_timing,
    market_timing_return,
    cornell_timing,
    annual_active_return,
    information_ratio_by_year,
)

DEPENDENT_BINARY_METRICS: tuple[Callable[..., object], ...] = (
    up_alpha_beta,
    down_alpha_beta,
    treynor_ratio,
    roll_up_down_capture,
    annual_active_return_by_year,
)

EMPYRICAL_DUAL_BINARY_METHODS: tuple[str, ...] = (
    "r_cubed",
    "relative_win_rate",
    "capm_r_squared",
    "up_capture_return",
    "down_capture_return",
    "tracking_difference",
    "treynor_ratio",
    "m_squared",
    "residual_risk",
    "annual_active_return",
    "annual_active_risk",
    "annual_active_return_by_year",
    "information_ratio_by_year",
    "regression_annual_return",
)


@pytest.mark.parametrize("function", ENHANCED_BINARY_METRICS, ids=lambda function: function.__name__)
def test_enhanced_binary_metric_exposes_keyword_only_alignment_contract(function: Callable[..., object]) -> None:
    signature = inspect.signature(function)

    alignment = signature.parameters["alignment"]
    normalize_tz = signature.parameters["normalize_tz"]

    assert alignment.kind is inspect.Parameter.KEYWORD_ONLY
    assert alignment.default == "inner"
    assert normalize_tz.kind is inspect.Parameter.KEYWORD_ONLY
    assert normalize_tz.default is None


@pytest.mark.parametrize("name", EMPYRICAL_DUAL_BINARY_METHODS)
def test_empyrical_dual_binary_method_exposes_alignment_on_class_and_instance(name: str) -> None:
    for surface in (ep.Empyrical, ep.Empyrical()):
        signature = inspect.signature(getattr(surface, name))
        alignment = signature.parameters["alignment"]
        normalize_tz = signature.parameters["normalize_tz"]

        assert alignment.kind is inspect.Parameter.KEYWORD_ONLY
        assert alignment.default == "inner"
        assert normalize_tz.kind is inspect.Parameter.KEYWORD_ONLY
        assert normalize_tz.default is None


@pytest.mark.parametrize("surface", ["class", "stored_instance"])
def test_empyrical_dual_binary_method_forwards_strict_alignment(surface: str) -> None:
    left, right = _partial_series_pair()

    with pytest.raises(DataAlignmentError, match="strict"):
        if surface == "class":
            ep.Empyrical.r_cubed(left, right, alignment="strict")
        else:
            ep.Empyrical(left, factor_returns=right).r_cubed(alignment="strict")


@pytest.mark.parametrize("surface", ["class", "stored_instance"])
def test_empyrical_dual_binary_method_forwards_timezone_policy(surface: str) -> None:
    utc_index = pd.date_range("2024-01-01", periods=3, tz="UTC")
    returns = pd.Series([0.03, 0.01, 0.02], index=utc_index.tz_localize(None))
    factor_returns = pd.Series([0.02, 0.015, 0.025], index=utc_index.tz_convert("Asia/Shanghai"))
    instance = ep.Empyrical(returns, factor_returns=factor_returns)
    if surface == "class":
        with pytest.raises(DataAlignmentError, match="timezone"):
            ep.Empyrical.relative_win_rate(returns, factor_returns)
        result = ep.Empyrical.relative_win_rate(
            returns,
            factor_returns,
            alignment="strict",
            normalize_tz="UTC",
        )
    else:
        with pytest.raises(DataAlignmentError, match="timezone"):
            instance.relative_win_rate()
        result = instance.relative_win_rate(alignment="strict", normalize_tz="UTC")
    assert result == pytest.approx(1 / 3)


def test_regression_annual_return_forwards_policy_once_to_alpha_and_beta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = np.array([0.01, 0.02, 0.03])
    factor_returns = np.array([0.005, 0.01, 0.015])
    observed: list[tuple[str, str, str | None]] = []

    def fake_alpha(
        returns_arg: np.ndarray,
        factor_arg: np.ndarray,
        risk_free: float,
        period: str,
        annualization: float | None,
        *,
        alignment: str,
        normalize_tz: str | None,
    ) -> float:
        assert returns_arg is returns
        assert factor_arg is factor_returns
        observed.append(("alpha", alignment, normalize_tz))
        return 0.1

    def fake_beta(
        returns_arg: np.ndarray,
        factor_arg: np.ndarray,
        risk_free: float,
        _period: str,
        _annualization: float | None,
        *,
        alignment: str,
        normalize_tz: str | None,
    ) -> float:
        assert returns_arg is returns
        assert factor_arg is factor_returns
        observed.append(("beta", alignment, normalize_tz))
        return 0.5

    monkeypatch.setattr(alpha_beta_metrics, "alpha", fake_alpha)
    monkeypatch.setattr(alpha_beta_metrics, "beta", fake_beta)

    result = ep.Empyrical.regression_annual_return(
        returns,
        factor_returns,
        alignment="outer_dropna",
        normalize_tz="UTC",
    )

    assert np.isfinite(result)
    assert observed == [
        ("alpha", "outer_dropna", "UTC"),
        ("beta", "outer_dropna", "UTC"),
    ]


@pytest.mark.parametrize("function", ENHANCED_BINARY_METRICS, ids=lambda function: function.__name__)
def test_all_direct_binary_metrics_route_duplicate_labels_through_contract(
    function: Callable[..., object],
) -> None:
    duplicate_index = pd.to_datetime(["2024-01-01", "2024-01-01"])
    left = pd.Series([0.01, 0.02], index=duplicate_index)
    right = pd.Series([0.03, 0.04], index=duplicate_index)

    with pytest.raises(DataAlignmentError, match="duplicate"):
        function(left, right)


@pytest.mark.parametrize("function", ENHANCED_BINARY_METRICS, ids=lambda function: function.__name__)
def test_all_direct_binary_metrics_reject_mixed_positional_and_labelled_inputs(
    function: Callable[..., object],
) -> None:
    left = np.array([0.01, 0.02])
    right = pd.Series([0.03, 0.04])

    with pytest.raises(DataAlignmentError, match="mix"):
        function(left, right)


@pytest.mark.parametrize("function", DEPENDENT_BINARY_METRICS, ids=lambda function: function.__name__)
def test_dependent_binary_metric_exposes_and_forwards_strict_alignment(
    function: Callable[..., object],
) -> None:
    signature = inspect.signature(function)
    assert signature.parameters["alignment"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["normalize_tz"].kind is inspect.Parameter.KEYWORD_ONLY
    left, right = _partial_series_pair()

    with pytest.raises(DataAlignmentError, match="strict"):
        function(left, right, alignment="strict")


@pytest.mark.parametrize("name", ["roll_up_capture", "roll_down_capture", "roll_up_down_capture"])
def test_strict_capture_roll_unknown_keyword_uses_pinned_capture_message(name: str) -> None:
    returns = np.array([0.01, -0.02, 0.03])
    factor_returns = np.array([0.02, -0.01, 0.01])

    with pytest.raises(TypeError, match=r"^capture\(\) got an unexpected keyword argument 'bad'$"):
        getattr(ep, name)(returns, factor_returns, window=2, bad=True)


def _align_binary(left: object, right: object, **kwargs: object):
    from fincore.contracts.time_series import align_binary_metric_inputs

    return align_binary_metric_inputs(left, right, **kwargs)


def _partial_series_pair() -> tuple[pd.Series, pd.Series]:
    left = pd.Series(
        [0.01, np.nan, 0.03],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    right = pd.Series(
        [0.02, 0.04, 0.05],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    return left, right


@pytest.mark.parametrize(
    ("alignment", "expected_index"),
    [
        ("inner", pd.to_datetime(["2024-01-02", "2024-01-03"])),
        ("outer_dropna", pd.to_datetime(["2024-01-03"])),
    ],
)
def test_binary_helper_applies_partial_label_policy_without_mutation(
    alignment: str,
    expected_index: pd.DatetimeIndex,
) -> None:
    left, right = _partial_series_pair()
    left_before = left.copy()
    right_before = right.copy()

    left_aligned, right_aligned = _align_binary(left, right, alignment=alignment)

    pd.testing.assert_index_equal(left_aligned.index, expected_index)
    pd.testing.assert_index_equal(right_aligned.index, expected_index)
    pd.testing.assert_series_equal(left, left_before)
    pd.testing.assert_series_equal(right, right_before)


def test_binary_helper_strict_rejects_partial_and_disjoint_labels() -> None:
    left, right = _partial_series_pair()
    disjoint = right.set_axis(pd.date_range("2025-01-01", periods=3))

    with pytest.raises(DataAlignmentError, match="strict"):
        _align_binary(left, right, alignment="strict")
    with pytest.raises(DataAlignmentError, match="strict"):
        _align_binary(left, disjoint, alignment="strict")


@pytest.mark.parametrize("alignment", ["inner", "outer_dropna"])
def test_binary_helper_disjoint_labels_return_empty_pandas_objects(alignment: str) -> None:
    left = pd.Series([0.01], index=pd.to_datetime(["2024-01-01"]))
    right = pd.Series([0.02], index=pd.to_datetime(["2025-01-01"]))

    left_aligned, right_aligned = _align_binary(left, right, alignment=alignment)

    assert left_aligned.empty
    assert right_aligned.empty


def test_binary_helper_rejects_duplicate_labels() -> None:
    duplicate_index = pd.to_datetime(["2024-01-01", "2024-01-01"])
    left = pd.Series([0.01, 0.02], index=duplicate_index)
    right = pd.Series([0.03, 0.04], index=duplicate_index)

    for alignment in ("strict", "inner", "outer_dropna"):
        with pytest.raises(DataAlignmentError, match="duplicate"):
            _align_binary(left, right, alignment=alignment)


def test_binary_helper_strict_success_keeps_unsorted_order_and_returns_copies() -> None:
    index = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
    left = pd.Series([0.03, 0.01, 0.02], index=index)
    right = pd.Series([0.04, 0.02, 0.03], index=index)

    left_aligned, right_aligned = _align_binary(left, right, alignment="strict")

    pd.testing.assert_index_equal(left_aligned.index, index)
    pd.testing.assert_index_equal(right_aligned.index, index)
    pd.testing.assert_series_equal(left_aligned, left)
    pd.testing.assert_series_equal(right_aligned, right)
    assert left_aligned is not left
    assert right_aligned is not right


def test_binary_helper_rejects_mixed_timezone_by_default_and_normalizes_explicitly() -> None:
    utc_index = pd.date_range("2024-01-01", periods=2, tz="UTC")
    naive = pd.Series([0.01, 0.02], index=utc_index.tz_localize(None))
    shanghai = pd.Series([0.03, 0.04], index=utc_index.tz_convert("Asia/Shanghai"))
    naive_index_before = naive.index.copy()
    shanghai_index_before = shanghai.index.copy()

    with pytest.raises(DataAlignmentError, match="timezone"):
        _align_binary(naive, shanghai, alignment="inner")

    naive_aligned, shanghai_aligned = _align_binary(
        naive,
        shanghai,
        alignment="strict",
        normalize_tz="UTC",
    )
    pd.testing.assert_index_equal(naive_aligned.index, utc_index)
    pd.testing.assert_index_equal(shanghai_aligned.index, utc_index)
    pd.testing.assert_index_equal(naive.index, naive_index_before)
    pd.testing.assert_index_equal(shanghai.index, shanghai_index_before)


def test_binary_helper_validates_timezone_option_before_range_index_handling() -> None:
    left = pd.Series([0.01, 0.02])
    right = pd.Series([0.03, 0.04])

    with pytest.raises(ValueError, match="only 'UTC'"):
        _align_binary(left, right, alignment="inner", normalize_tz="Asia/Shanghai")


def test_binary_helper_keeps_equal_ndarrays_positional_and_rejects_ambiguous_inputs() -> None:
    left = np.array([0.01, 0.02])
    right = np.array([0.03, 0.04])

    for alignment in ("strict", "inner", "outer_dropna"):
        left_aligned, right_aligned = _align_binary(left, right, alignment=alignment)

        assert left_aligned is left
        assert right_aligned is right
        with pytest.raises(DataAlignmentError, match="same length"):
            _align_binary(left, right[:1], alignment=alignment)
        with pytest.raises(DataAlignmentError, match="mix"):
            _align_binary(left, pd.Series(right), alignment=alignment)


def test_metric_ndarray_inputs_remain_positional() -> None:
    assert relative_win_rate(
        np.array([0.1, 0.0]),
        np.array([0.0, 0.1]),
        alignment="strict",
    ) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "function",
    [beta, information_ratio, tracking_error, roll_beta, r_cubed, treynor_mazuy_timing, annual_active_return],
    ids=lambda function: function.__name__,
)
def test_each_metric_family_rejects_partial_labels_under_strict(function: Callable[..., object]) -> None:
    left, right = _partial_series_pair()

    with pytest.raises(DataAlignmentError, match="strict"):
        function(left, right, alignment="strict")


def test_dependent_market_timing_return_forwards_alignment_policy() -> None:
    index = pd.date_range("2024-01-01", periods=12)
    returns = pd.Series(np.linspace(-0.03, 0.04, 12), index=index)
    factor_returns = pd.Series(np.linspace(-0.02, 0.03, 12), index=index.shift(1))

    with pytest.raises(DataAlignmentError, match="strict"):
        market_timing_return(returns, factor_returns, alignment="strict")


def test_market_timing_return_uses_strict_policy_after_timezone_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    utc_index = pd.date_range("2024-01-01", periods=3, tz="UTC")
    returns = pd.Series([0.01, 0.02, 0.03], index=utc_index.tz_localize(None))
    factor_returns = pd.Series([0.005, 0.01, 0.015], index=utc_index)
    observed: dict[str, object] = {}

    def fake_timing(
        aligned_returns: pd.Series,
        aligned_factor: pd.Series,
        risk_free: float = 0.0,
        *,
        alignment: str = "inner",
        normalize_tz: str | None = None,
    ) -> float:
        observed.update(
            returns=aligned_returns,
            factor_returns=aligned_factor,
            risk_free=risk_free,
            alignment=alignment,
            normalize_tz=normalize_tz,
        )
        return 1.0

    monkeypatch.setattr(timing_metrics, "treynor_mazuy_timing", fake_timing)

    market_timing_return(returns, factor_returns, normalize_tz="UTC")

    assert observed["alignment"] == "strict"
    assert observed["normalize_tz"] is None
    pd.testing.assert_index_equal(observed["returns"].index, utc_index)  # type: ignore[union-attr]
    pd.testing.assert_index_equal(observed["factor_returns"].index, utc_index)  # type: ignore[union-attr]


def test_dependent_information_ratio_by_year_forwards_timezone_policy() -> None:
    utc_index = pd.date_range("2024-01-01", periods=3, tz="UTC")
    returns = pd.Series([0.01, 0.02, 0.03], index=utc_index.tz_localize(None))
    factor_returns = pd.Series([0.005, 0.01, 0.015], index=utc_index)

    with pytest.raises(DataAlignmentError, match="timezone"):
        information_ratio_by_year(returns, factor_returns)
    result = information_ratio_by_year(returns, factor_returns, normalize_tz="UTC")
    assert result.index.tolist() == [2024]


def test_information_ratio_by_year_uses_strict_policy_after_timezone_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    utc_index = pd.date_range("2024-01-01", periods=3, tz="UTC")
    returns = pd.Series([0.01, 0.02, 0.03], index=utc_index.tz_localize(None))
    factor_returns = pd.Series([0.005, 0.01, 0.015], index=utc_index)
    observed: dict[str, object] = {}

    def fake_information_ratio(
        aligned_returns: pd.Series,
        aligned_factor: pd.Series,
        period: str,
        annualization: float | None,
        *,
        alignment: str = "inner",
        normalize_tz: str | None = None,
    ) -> float:
        observed.update(
            returns=aligned_returns,
            factor_returns=aligned_factor,
            period=period,
            annualization=annualization,
            alignment=alignment,
            normalize_tz=normalize_tz,
        )
        return 7.0

    monkeypatch.setattr(ratios_metrics, "information_ratio", fake_information_ratio)

    result = information_ratio_by_year(returns, factor_returns, normalize_tz="UTC")

    assert result.loc[2024] == 7.0
    assert observed["alignment"] == "strict"
    assert observed["normalize_tz"] is None
    pd.testing.assert_index_equal(observed["returns"].index, utc_index)  # type: ignore[union-attr]
    pd.testing.assert_index_equal(observed["factor_returns"].index, utc_index)  # type: ignore[union-attr]


def test_strict_facade_signatures_and_outer_alignment_values_stay_frozen() -> None:
    assert str(inspect.signature(ep.alpha)) == (
        "(returns, factor_returns, risk_free=0.0, period='daily', annualization=None, out=None, _beta=None)"
    )
    assert str(inspect.signature(ep.beta)) == "(returns, factor_returns, risk_free=0.0, out=None)"
    left = pd.Series([0.01, 0.02, -0.01], index=pd.date_range("2024-01-01", periods=3))
    right = pd.Series([0.005, -0.005, 0.01], index=pd.date_range("2024-01-02", periods=3))

    assert ep.beta(left, right) == pytest.approx(3.0)
    assert ep.alpha(left, right) == pytest.approx(2.51437064469923)
    np.testing.assert_allclose(ep.alpha_beta(left, right), [2.51437064469923, 3.0])
    assert ep.beta_fragility_heuristic(left, right) == pytest.approx(-0.03)
    with pytest.raises(TypeError, match="unexpected keyword"):
        ep.beta(left, right, alignment="inner")


def test_strict_capture_keeps_independent_series_and_filtering_semantics() -> None:
    left = pd.Series([0.01, 0.02, -0.01], index=pd.date_range("2024-01-01", periods=3))
    right = pd.Series([0.005, -0.005, 0.01], index=pd.date_range("2024-01-02", periods=3))

    assert ep.capture(left, right) == pytest.approx(3.2515855139580485)
    with pytest.raises(pd.errors.IndexingError, match="Unalignable"):
        ep.up_capture(left, right)


def test_legacy_basic_aligned_series_keeps_outer_join_and_nan_rows() -> None:
    left, right = _partial_series_pair()

    left_aligned, right_aligned = legacy_aligned_series(left, right)

    expected_index = pd.date_range("2024-01-01", periods=4)
    pd.testing.assert_index_equal(left_aligned.index, expected_index)
    pd.testing.assert_index_equal(right_aligned.index, expected_index)
    assert left_aligned.isna().sum() == 2
    assert right_aligned.isna().sum() == 1
