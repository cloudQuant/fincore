"""Strict C3 migration coverage for pinned Alphalens forward-return cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _factor_from_prices(prices: pd.DataFrame) -> pd.Series:
    """Create an explicit modern stacked factor series without pandas defaults."""

    factor = prices.stack(future_stack=True).astype(float)
    factor.index = factor.index.set_names(("date", "asset"))
    return factor.rename("factor")


def _daily_prices() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=3, name="date")
    return pd.DataFrame([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0]], index=dates, columns=("A", "B"))


def _expected_daily_forward_returns(*, non_cumulative: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=3, name="date")
    index = pd.MultiIndex.from_product((dates, ("A", "B")), names=("date", "asset"))
    second_period = (
        [1.0, -0.5, np.nan, np.nan, np.nan, np.nan] if non_cumulative else [1.0, 0.0, np.nan, np.nan, np.nan, np.nan]
    )
    return pd.DataFrame(
        {
            "1D": [0.0, 1.0, 1.0, -0.5, np.nan, np.nan],
            "2D": second_period,
        },
        index=index,
    )


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_compute_forward_returns#00",
            id="tests/test_utils.py::UtilsTestCase::test_compute_forward_returns#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_utils.py::UtilsTestCase::test_compute_forward_returns#00"
            ),
        )
    ],
)
def test_compute_forward_returns_upstream_case(source_case_id: str) -> None:
    """Pinned daily cumulative returns retain values, labels, and MultiIndex order."""

    from fincore.alphalens import utils

    prices = _daily_prices()
    actual = utils.compute_forward_returns(_factor_from_prices(prices), prices, periods=(1, 2))

    pd.testing.assert_frame_equal(actual, _expected_daily_forward_returns())


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_compute_forward_returns_index_out_of_bound#00",
            id="tests/test_utils.py::UtilsTestCase::test_compute_forward_returns_index_out_of_bound#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_utils.py::UtilsTestCase::test_compute_forward_returns_index_out_of_bound#00"
            ),
        )
    ],
)
def test_compute_forward_returns_index_out_of_bound_upstream_case(source_case_id: str) -> None:
    """Extra leading price dates do not shift factor-date forward-return alignment."""

    from fincore.alphalens import utils

    prices = _daily_prices()
    leading_dates = pd.date_range("2014-12-29", periods=3, name="date")
    leading = pd.DataFrame(np.nan, index=leading_dates, columns=prices.columns)
    expanded_prices = pd.concat((leading, prices))
    actual = utils.compute_forward_returns(_factor_from_prices(prices), expanded_prices, periods=(1, 2))

    pd.testing.assert_frame_equal(actual, _expected_daily_forward_returns())


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_compute_forward_returns_non_cum#00",
            id="tests/test_utils.py::UtilsTestCase::test_compute_forward_returns_non_cum#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_utils.py::UtilsTestCase::test_compute_forward_returns_non_cum#00"
            ),
        )
    ],
)
def test_compute_forward_returns_non_cum_upstream_case(source_case_id: str) -> None:
    """Non-cumulative windows return the terminal single-period return."""

    from fincore.alphalens import utils

    prices = _daily_prices()
    actual = utils.compute_forward_returns(
        _factor_from_prices(prices), prices, periods=(1, 2), cumulative_returns=False
    )

    pd.testing.assert_frame_equal(actual, _expected_daily_forward_returns(non_cumulative=True))


def test_forward_returns_reject_timezone_mismatch_and_preserve_inputs() -> None:
    """The strict adapter projects a timezone mismatch without mutating either input."""

    from fincore.alphalens import utils

    prices = _daily_prices()
    factor = _factor_from_prices(prices)
    factor_before = factor.copy(deep=True)
    prices_before = prices.copy(deep=True)
    utc_prices = prices.copy()
    utc_prices.index = utc_prices.index.tz_localize("UTC")

    with pytest.raises(utils.NonMatchingTimezoneError, match="timezone of 'factor'"):
        utils.compute_forward_returns(factor, utc_prices)

    pd.testing.assert_series_equal(factor, factor_before)
    pd.testing.assert_frame_equal(prices, prices_before)


def test_strict_forward_returns_pad_missing_factor_assets_like_pinned_prices_filter() -> None:
    """Missing price columns become aligned all-NaN rows on the strict source-visible path."""

    from fincore.alphalens import utils

    dates = pd.date_range("2024-01-02", periods=4, name="date")
    index = pd.MultiIndex.from_product((dates[:2], ("A", "MISSING")), names=("date", "asset"))
    factor = pd.Series((1.0, 2.0, 3.0, 4.0), index=index, name="factor")
    prices = pd.DataFrame({"A": (10.0, 11.0, 12.0, 13.0)}, index=dates)
    factor_before = factor.copy(deep=True)
    prices_before = prices.copy(deep=True)

    expected = pd.DataFrame(
        {"1D": (0.1, np.nan, 1.0 / 11.0, np.nan)},
        index=index,
    )
    actual = utils.compute_forward_returns(factor, prices, periods=(1,))
    pd.testing.assert_frame_equal(actual, expected)

    cleaned = utils.get_clean_factor_and_forward_returns(
        factor,
        prices,
        periods=(1,),
        quantiles=None,
        bins=1,
        max_loss=1,
    )
    pd.testing.assert_index_equal(
        cleaned.index, pd.MultiIndex.from_tuples(((dates[0], "A"), (dates[1], "A")), names=("date", "asset"))
    )
    pd.testing.assert_series_equal(cleaned["1D"], expected.loc[(slice(None), "A"), "1D"], check_names=False)
    pd.testing.assert_series_equal(factor, factor_before)
    pd.testing.assert_frame_equal(prices, prices_before)


def test_forward_returns_support_intraday_labels_and_zscore_filtering() -> None:
    """Intraday session spacing produces canonical labels and optional filtering."""

    from fincore.alphalens import utils

    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-02 09:30"),
            pd.Timestamp("2024-01-02 10:30"),
            pd.Timestamp("2024-01-02 12:30"),
            pd.Timestamp("2024-01-03 09:30"),
        ],
        name="date",
    )
    prices = pd.DataFrame({"A": [100.0, 101.0, 99.0, 110.0], "B": [100.0, 100.0, 100.0, 100.0]}, index=dates)
    factor = _factor_from_prices(prices.iloc[:2])
    actual = utils.compute_forward_returns(factor, prices, periods=(1, 2, 3), filter_zscore=0.5)

    assert tuple(actual.columns) == ("1h", "3h", "1D")
    # The small z-score threshold deliberately filters the non-zero A return
    # while retaining the canonical intraday label calculation.
    assert pd.isna(actual.loc[(dates[0], "A"), "1h"])
    assert actual.loc[(dates[0], "A"), "1D"] == pytest.approx(0.1)


def test_forward_returns_cover_long_periods_oob_factor_dates_and_matching_timezones() -> None:
    """The kernel preserves 1/5/10-session arithmetic for naive and UTC inputs."""

    from fincore.alphalens import utils

    price_dates = pd.bdate_range("2024-01-02", periods=12, name="date")
    prices = pd.DataFrame({"A": np.arange(100.0, 112.0)}, index=price_dates)
    factor_dates = pd.DatetimeIndex((price_dates[0], price_dates[1], price_dates[-1] + pd.offsets.BusinessDay()))
    factor_index = pd.MultiIndex.from_product((factor_dates, ("A",)), names=("date", "asset"))
    factor = pd.Series((1.0, 2.0, 3.0), index=factor_index, name="factor")

    actual = utils.compute_forward_returns(factor, prices, periods=(1, 5, 10))
    assert tuple(actual.columns) == ("1D", "5D", "10D")
    assert actual.loc[(price_dates[0], "A"), "1D"] == pytest.approx(0.01)
    assert actual.loc[(price_dates[0], "A"), "5D"] == pytest.approx(0.05)
    assert actual.loc[(price_dates[0], "A"), "10D"] == pytest.approx(0.10)
    assert actual.loc[(factor_dates[-1], "A")].isna().all()

    utc_prices = prices.copy()
    utc_prices.index = utc_prices.index.tz_localize("UTC")
    utc_factor_index = pd.MultiIndex.from_product(
        (factor_dates[:2].tz_localize("UTC"), ("A",)), names=("date", "asset")
    )
    utc_factor = pd.Series((1.0, 2.0), index=utc_factor_index, name="factor")
    utc_actual = utils.compute_forward_returns(utc_factor, utc_prices, periods=(1, 5, 10))

    assert utc_actual.index.get_level_values("date").tz is not None
    assert utc_actual.loc[(price_dates[0].tz_localize("UTC"), "A"), "10D"] == pytest.approx(0.10)
