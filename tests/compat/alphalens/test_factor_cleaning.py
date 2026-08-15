"""Strict C3 migration coverage for pinned Alphalens quantization and cleaning cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _factor_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fresh factor/group frames matching the pinned utility source rows."""

    dates = pd.date_range("2015-01-01", periods=2, name="date")
    assets = ("A", "B", "C", "D")
    index = pd.MultiIndex.from_product((dates, assets), names=("date", "asset"))
    factor = pd.DataFrame({"factor": [1, 2, 3, 4, 4, 3, 2, 1]}, index=index)
    factor["group"] = pd.Categorical([1, 1, 2, 2, 1, 1, 2, 2])

    biased_assets = (*assets, "E", "F", "G", "H")
    biased_index = pd.MultiIndex.from_product((dates, biased_assets), names=("date", "asset"))
    biased = pd.DataFrame(
        {"factor": [-1, 3, -2, 4, -5, 7, -6, 8, -4, 2, -3, 1, -8, 6, -7, 5]},
        index=biased_index,
    )
    biased["group"] = pd.Categorical([1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2])
    return factor, biased


_QUANTIZE_CASES = (
    ("regular", 4, None, False, False, [1, 2, 3, 4, 4, 3, 2, 1]),
    ("regular", 2, None, False, False, [1, 1, 2, 2, 2, 2, 1, 1]),
    ("regular", 2, None, True, False, [1, 2, 1, 2, 2, 1, 2, 1]),
    ("biased", 4, None, False, True, [2, 3, 2, 3, 1, 4, 1, 4, 2, 3, 2, 3, 1, 4, 1, 4]),
    ("biased", 2, None, False, True, [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]),
    ("biased", 2, None, True, True, [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]),
    ("biased", None, 4, False, True, [2, 3, 2, 3, 1, 4, 1, 4, 2, 3, 2, 3, 1, 4, 1, 4]),
    ("biased", None, 2, False, True, [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]),
    ("biased", None, 2, True, True, [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]),
    ("regular", [0, 0.25, 0.5, 0.75, 1.0], None, False, False, [1, 2, 3, 4, 4, 3, 2, 1]),
    ("regular", [0, 0.5, 0.75, 1.0], None, False, False, [1, 1, 2, 3, 3, 2, 1, 1]),
    ("regular", [0, 0.25, 0.5, 1.0], None, False, False, [1, 2, 3, 3, 3, 3, 2, 1]),
    ("regular", [0, 0.5, 1.0], None, False, False, [1, 1, 2, 2, 2, 2, 1, 1]),
    ("regular", [0.25, 0.5, 0.75], None, False, False, [np.nan, 1, 2, np.nan, np.nan, 2, 1, np.nan]),
    ("regular", [0, 0.5, 1.0], None, True, False, [1, 2, 1, 2, 2, 1, 2, 1]),
    ("regular", [0.5, 1.0], None, True, False, [np.nan, 1, np.nan, 1, 1, np.nan, 1, np.nan]),
    ("regular", [0, 1.0], None, True, False, [1, 1, 1, 1, 1, 1, 1, 1]),
    ("regular", None, 4, False, False, [1, 2, 3, 4, 4, 3, 2, 1]),
    ("regular", None, 2, False, False, [1, 1, 2, 2, 2, 2, 1, 1]),
    ("regular", None, 3, False, False, [1, 1, 2, 3, 3, 2, 1, 1]),
    ("regular", None, 8, False, False, [1, 3, 6, 8, 8, 6, 3, 1]),
    ("regular", None, [0, 1, 2, 3, 5], False, False, [1, 2, 3, 4, 4, 3, 2, 1]),
    ("regular", None, [1, 2, 3], False, False, [np.nan, 1, 2, np.nan, np.nan, 2, 1, np.nan]),
    ("regular", None, [0, 2, 5], False, False, [1, 1, 2, 2, 2, 2, 1, 1]),
    ("regular", None, [0.5, 2.5, 4.5], False, False, [1, 1, 2, 2, 2, 2, 1, 1]),
    ("regular", None, [0.5, 2.5], True, False, [1, 1, np.nan, np.nan, np.nan, np.nan, 1, 1]),
    ("regular", None, 2, True, False, [1, 2, 1, 2, 2, 1, 2, 1]),
)


@pytest.mark.parametrize(
    "source_case_id,frame_name,quantiles,bins,by_group,zero_aware,expected_values",
    [
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#00",
            *_QUANTIZE_CASES[0],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#00",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#00"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#01",
            *_QUANTIZE_CASES[1],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#01",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#01"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#02",
            *_QUANTIZE_CASES[2],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#02",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#02"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#03",
            *_QUANTIZE_CASES[3],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#03",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#03"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#04",
            *_QUANTIZE_CASES[4],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#04",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#04"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#05",
            *_QUANTIZE_CASES[5],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#05",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#05"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#06",
            *_QUANTIZE_CASES[6],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#06",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#06"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#07",
            *_QUANTIZE_CASES[7],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#07",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#07"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#08",
            *_QUANTIZE_CASES[8],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#08",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#08"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#09",
            *_QUANTIZE_CASES[9],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#09",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#09"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#10",
            *_QUANTIZE_CASES[10],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#10",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#10"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#11",
            *_QUANTIZE_CASES[11],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#11",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#11"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#12",
            *_QUANTIZE_CASES[12],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#12",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#12"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#13",
            *_QUANTIZE_CASES[13],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#13",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#13"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#14",
            *_QUANTIZE_CASES[14],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#14",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#14"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#15",
            *_QUANTIZE_CASES[15],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#15",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#15"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#16",
            *_QUANTIZE_CASES[16],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#16",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#16"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#17",
            *_QUANTIZE_CASES[17],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#17",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#17"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#18",
            *_QUANTIZE_CASES[18],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#18",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#18"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#19",
            *_QUANTIZE_CASES[19],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#19",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#19"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#20",
            *_QUANTIZE_CASES[20],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#20",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#20"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#21",
            *_QUANTIZE_CASES[21],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#21",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#21"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#22",
            *_QUANTIZE_CASES[22],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#22",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#22"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#23",
            *_QUANTIZE_CASES[23],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#23",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#23"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#24",
            *_QUANTIZE_CASES[24],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#24",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#24"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#25",
            *_QUANTIZE_CASES[25],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#25",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#25"),
        ),
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_quantize_factor#26",
            *_QUANTIZE_CASES[26],
            id="tests/test_utils.py::UtilsTestCase::test_quantize_factor#26",
            marks=pytest.mark.alphalens_upstream_case("tests/test_utils.py::UtilsTestCase::test_quantize_factor#26"),
        ),
    ],
)
def test_quantize_factor_upstream_case(
    source_case_id: str,
    frame_name: str,
    quantiles: int | list[float] | None,
    bins: int | list[float] | None,
    by_group: bool,
    zero_aware: bool,
    expected_values: list[float],
) -> None:
    """Every pinned qcut/cut input/expected row remains a discrete C3 case."""

    from fincore.alphalens import utils

    regular, biased = _factor_frames()
    frame = regular if frame_name == "regular" else biased
    actual = utils.quantize_factor(frame, quantiles=quantiles, bins=bins, by_group=by_group, zero_aware=zero_aware)
    expected = pd.Series(expected_values, index=frame.index, name="factor_quantile").dropna()

    pd.testing.assert_series_equal(actual, expected)


def _clean_case(case_number: int) -> tuple[pd.Series, pd.DataFrame, dict[str, int], tuple[int, ...]]:
    """Build each pinned clean-factor source shape from fresh mutable inputs."""

    assets = ("A", "B", "C", "D", "E", "F")
    groups = {asset: 1 if ordinal % 2 == 0 else 2 for ordinal, asset in enumerate(assets)}
    values = [[3, 4, 2, 1, np.nan, np.nan], [3, np.nan, np.nan, 1, 4, 2], [3, 4, 2, 1, np.nan, np.nan]]
    if case_number in {1, 2}:
        index = pd.date_range(
            "2015-01-11" if case_number == 1 else "2017-01-12",
            periods=6,
            freq=None if case_number == 1 else "B",
            name="date",
        )
        prices = pd.DataFrame(
            [[1.1**row, 0.5**row, 3.0**row, 0.9**row, 0.5**row, 1.0**row] for row in range(1, 7)],
            index=index,
            columns=assets,
        )
        factor_dates = index[:3]
        factor = pd.DataFrame(values, index=factor_dates, columns=assets).stack(future_stack=True).rename("factor")
        factor.index = factor.index.set_names(("date", "asset"))
        return factor, prices, groups, (1, 2, 3)
    if case_number == 3:
        price_dates = pd.bdate_range("2017-01-12", periods=4, name="date")
        price_data = [[1.1**row, 0.5**row, 3.0**row, 0.9**row, 0.5**row, 1.0**row] for row in range(1, 5)]
        today_open = pd.DataFrame(price_data, index=price_dates + pd.Timedelta("9h30m"), columns=assets)
        one_hour = today_open.copy()
        one_hour.index = one_hour.index + pd.Timedelta("1h")
        one_hour *= 1.001
        three_hours = today_open.copy()
        three_hours.index = three_hours.index + pd.Timedelta("3h")
        three_hours *= 0.998
        prices = pd.concat((today_open, one_hour, three_hours)).sort_index()
        factor = (
            pd.DataFrame(
                values,
                index=price_dates[:3] + pd.Timedelta("9h30m"),
                columns=assets,
            )
            .stack(future_stack=True)
            .rename("factor")
        )
        factor.index = factor.index.set_names(("date", "asset"))
        return factor, prices, groups, (1, 2, 3)
    if case_number == 4:
        price_dates = pd.date_range("2017-01-12", periods=8, freq="B", name="date")
        prices = pd.DataFrame(
            [[1.1**row, 0.5**row, 3.0**row, 0.9**row, 0.5**row, 1.0**row] for row in range(1, 9)],
            index=price_dates,
            columns=assets,
        )
        factor_values = [
            [1, np.nan, np.nan, np.nan, np.nan, 6],
            [4, np.nan, np.nan, 7, np.nan, np.nan],
            [np.nan] * 6,
            [np.nan, 3, np.nan, 2, np.nan, np.nan],
            [np.nan, np.nan, 1, np.nan, 3, np.nan],
        ]
        factor = (
            pd.DataFrame(factor_values, index=price_dates[:5], columns=assets)
            .stack(future_stack=True)
            .dropna()
            .rename("factor")
        )
        factor.index = factor.index.set_names(("date", "asset"))
        return factor, prices, groups, (1, 2, 3)

    holidays = pd.to_datetime(("2017-01-13", "2017-01-18", "2017-01-30", "2017-02-07"))
    price_dates = pd.bdate_range(
        "2017-01-12", "2017-02-15" if case_number == 6 else "2017-02-13", name="date"
    ).difference(holidays)
    days = 21 if case_number == 6 else 19
    prices = pd.DataFrame(
        [[1.1**row, 0.5**row, 3.0**row, 0.9**row, 0.5**row, 1.0**row] for row in range(1, days + 1)],
        index=price_dates,
        columns=assets,
    )
    factor_dates = pd.bdate_range("2017-01-12", "2017-02-10", name="date").difference(holidays)
    factor_values = values * 6
    if case_number == 5:
        open_prices = prices.copy()
        open_prices.index = open_prices.index + pd.Timedelta("9h30m")
        one_hour = open_prices.copy()
        one_hour.index = one_hour.index + pd.Timedelta("1h")
        one_hour *= 1.001
        three_hours = open_prices.copy()
        three_hours.index = three_hours.index + pd.Timedelta("3h")
        three_hours *= 0.998
        prices = pd.concat((open_prices, one_hour, three_hours)).sort_index()
        factor_dates = factor_dates + pd.Timedelta("9h30m")
    factor = pd.DataFrame(factor_values, index=factor_dates, columns=assets).stack(future_stack=True).rename("factor")
    factor.index = factor.index.set_names(("date", "asset"))
    return factor, prices, groups, (1, 2, 3)


def _clean_output(case_number: int) -> pd.DataFrame:
    """Execute one pinned cleaning shape and return its strict result."""

    from fincore.alphalens import utils

    factor, prices, groups, periods = _clean_case(case_number)
    return utils.get_clean_factor_and_forward_returns(
        factor, prices, groupby=groups, quantiles=4, periods=periods, max_loss=1
    )


_DAILY_EXPECTED_ROWS = [
    [0.1, 0.21, 0.331, 3.0, 1, 3],
    [-0.5, -0.75, -0.875, 4.0, 2, 4],
    [2.0, 8.0, 26.0, 2.0, 1, 2],
    [-0.1, -0.19, -0.271, 1.0, 2, 1],
    [0.1, 0.21, 0.331, 3.0, 1, 3],
    [-0.1, -0.19, -0.271, 1.0, 2, 1],
    [-0.5, -0.75, -0.875, 4.0, 1, 4],
    [0.0, 0.0, 0.0, 2.0, 2, 2],
    [0.1, 0.21, 0.331, 3.0, 1, 3],
    [-0.5, -0.75, -0.875, 4.0, 2, 4],
    [2.0, 8.0, 26.0, 2.0, 1, 2],
    [-0.1, -0.19, -0.271, 1.0, 2, 1],
]
_INTRADAY_EXPECTED_ROWS = [
    [0.001, -0.002, 0.1, 3.0, 1, 3],
    [0.001, -0.002, -0.5, 4.0, 2, 4],
    [0.001, -0.002, 2.0, 2.0, 1, 2],
    [0.001, -0.002, -0.1, 1.0, 2, 1],
    [0.001, -0.002, 0.1, 3.0, 1, 3],
    [0.001, -0.002, -0.1, 1.0, 2, 1],
    [0.001, -0.002, -0.5, 4.0, 1, 4],
    [0.001, -0.002, 0.0, 2.0, 2, 2],
    [0.001, -0.002, 0.1, 3.0, 1, 3],
    [0.001, -0.002, -0.5, 4.0, 2, 4],
    [0.001, -0.002, 2.0, 2.0, 1, 2],
    [0.001, -0.002, -0.1, 1.0, 2, 1],
]
_EVENT_EXPECTED_ROWS = [
    [0.1, 0.21, 0.331, 1.0, 1, 1],
    [0.0, 0.0, 0.0, 6.0, 2, 4],
    [0.1, 0.21, 0.331, 4.0, 1, 1],
    [-0.1, -0.19, -0.271, 7.0, 2, 4],
    [-0.5, -0.75, -0.875, 3.0, 2, 4],
    [-0.1, -0.19, -0.271, 2.0, 2, 1],
    [2.0, 8.0, 26.0, 1.0, 1, 1],
    [-0.5, -0.75, -0.875, 3.0, 1, 4],
]


def _expected_clean_case(case_number: int) -> pd.DataFrame:
    """Rebuild the pinned expected row matrix without old pandas stack defaults."""

    factor, _, _, _ = _clean_case(case_number)
    rows = (
        _DAILY_EXPECTED_ROWS
        if case_number in {1, 2}
        else _INTRADAY_EXPECTED_ROWS
        if case_number == 3
        else _EVENT_EXPECTED_ROWS
        if case_number == 4
        else _INTRADAY_EXPECTED_ROWS * 6
        if case_number == 5
        else _DAILY_EXPECTED_ROWS * 6
    )
    columns = (
        ("1h", "3h", "1D", "factor", "group", "factor_quantile")
        if case_number in {3, 5}
        else (
            "1D",
            "2D",
            "3D",
            "factor",
            "group",
            "factor_quantile",
        )
    )
    expected = pd.DataFrame(rows, index=factor.dropna().index, columns=columns)
    expected["group"] = expected["group"].astype("category")
    return expected


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_1#00",
            id="tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_1#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_1#00"
            ),
        )
    ],
)
def test_get_clean_factor_and_forward_returns_1_upstream_case(source_case_id: str) -> None:
    """Daily pin 1 produces three day-labeled forward-return columns."""

    pd.testing.assert_frame_equal(_clean_output(1), _expected_clean_case(1))


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_2#00",
            id="tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_2#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_2#00"
            ),
        )
    ],
)
def test_get_clean_factor_and_forward_returns_2_upstream_case(source_case_id: str) -> None:
    """Business-day pin 2 retains the same source-normalized day labels."""

    pd.testing.assert_frame_equal(_clean_output(2), _expected_clean_case(2))


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_3#00",
            id="tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_3#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_3#00"
            ),
        )
    ],
)
def test_get_clean_factor_and_forward_returns_3_upstream_case(source_case_id: str) -> None:
    """Intraday pin 3 exposes the 1h, 3h, and session-day return windows."""

    pd.testing.assert_frame_equal(_clean_output(3), _expected_clean_case(3))


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_4#00",
            id="tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_4#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_4#00"
            ),
        )
    ],
)
def test_get_clean_factor_and_forward_returns_4_upstream_case(source_case_id: str) -> None:
    """Sparse event pin 4 retains only finite factor/return/quantile rows."""

    pd.testing.assert_frame_equal(_clean_output(4), _expected_clean_case(4))


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_5#00",
            id="tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_5#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_5#00"
            ),
        )
    ],
)
def test_get_clean_factor_and_forward_returns_5_upstream_case(source_case_id: str) -> None:
    """Holiday intraday pin 5 infers a custom calendar without losing observations."""

    pd.testing.assert_frame_equal(_clean_output(5), _expected_clean_case(5))


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_6#00",
            id="tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_6#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_utils.py::UtilsTestCase::test_get_clean_factor_and_forward_returns_6#00"
            ),
        )
    ],
)
def test_get_clean_factor_and_forward_returns_6_upstream_case(source_case_id: str) -> None:
    """Holiday daily pin 6 preserves the custom session calendar across periods."""

    pd.testing.assert_frame_equal(_clean_output(6), _expected_clean_case(6))
