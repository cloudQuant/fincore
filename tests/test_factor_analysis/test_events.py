"""Task 4 RED skeleton for event analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fincore.alphalens import performance as strict_performance
from fincore.factor_analysis.performance import average_cumulative_return_by_quantile, common_start_returns


def _common_inputs() -> tuple[pd.Series, pd.DataFrame]:
    """Rebuild the full nine-date pinned common-start source fixture."""

    dates = pd.date_range("2015-01-17", "2015-02-02", freq="D", name="date")
    assets = pd.Index(["A", "B", "C", "D"], name="asset")
    returns = pd.DataFrame(
        [[base**power for base in (1.20, 1.40, 0.90, 0.80)] for power in range(1, 18)],
        index=dates,
        columns=assets,
    )
    factor_dates = pd.date_range("2015-01-21", "2015-01-29", freq="D", name="date")
    factor = pd.Series(
        [value for _ in factor_dates for value in (3, 4, 2, 1)],
        index=pd.MultiIndex.from_product((factor_dates, assets), names=("date", "asset")),
        name="factor",
        dtype=float,
    )
    return factor, returns


_COMMON_START_RETURN_CASES = (
    (
        2,
        3,
        False,
        False,
        [
            [4.93048307, 8.68843922],
            [6.60404312, 12.22369139],
            [8.92068367, 17.1794088],
            [12.1275523, 24.12861778],
            [16.5694159, 33.8740100],
            [22.7273233, 47.53995233],
        ],
    ),
    (
        3,
        2,
        False,
        True,
        [
            [0.0, 5.63219176],
            [0.0, 7.96515233],
            [0.0, 11.2420646],
            [0.0, 15.8458720],
            [0.0, 22.3134160],
            [0.0, 31.3970961],
        ],
    ),
    (
        3,
        5,
        True,
        False,
        [
            [3.7228318, 2.6210478],
            [4.9304831, 3.6296796],
            [6.6040431, 5.0193734],
            [8.9206837, 6.9404046],
            [12.127552, 9.6023405],
            [16.569416, 13.297652],
            [22.727323, 18.434747],
            [31.272682, 25.584180],
            [34.358565, 25.497254],
        ],
    ),
    (1, 4, True, True, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
    (
        6,
        6,
        False,
        False,
        [
            [2.02679565, 2.38468223],
            [2.38769454, 3.22602748],
            [2.85413029, 4.36044469],
            [3.72283181, 6.16462715],
            [4.93048307, 8.68843922],
            [6.60404312, 12.2236914],
            [8.92068367, 17.1794088],
            [12.1275523, 24.1286178],
            [16.5694159, 33.8740100],
            [22.7273233, 47.5399523],
            [31.2726821, 66.7013483],
            [34.3585654, 70.1828776],
            [37.9964585, 74.3294620],
        ],
    ),
    (
        6,
        6,
        False,
        True,
        [
            [0.0, 2.20770299],
            [0.0, 2.95942924],
            [0.0, 3.97022414],
            [0.0, 5.63219176],
            [0.0, 7.96515233],
            [0.0, 11.2420646],
            [0.0, 15.8458720],
            [0.0, 22.3134160],
            [0.0, 31.3970962],
            [0.0, 44.1512888],
            [0.0, 62.0533954],
            [0.0, 65.8668371],
            [0.0, 70.4306483],
        ],
    ),
    (
        6,
        6,
        True,
        False,
        [
            [2.0267957, 0.9562173],
            [2.3876945, 1.3511898],
            [2.8541303, 1.8856194],
            [3.7228318, 2.6210478],
            [4.9304831, 3.6296796],
            [6.6040431, 5.0193734],
            [8.9206837, 6.9404046],
            [12.127552, 9.6023405],
            [16.569416, 13.297652],
            [22.727323, 18.434747],
            [31.272682, 25.584180],
            [34.358565, 25.497254],
            [37.996459, 25.198051],
        ],
    ),
    (
        6,
        6,
        True,
        True,
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    ),
)


def _ordinary_event_factor_data(quantiles: int, after: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the source-cleaned event projection used by the target function.

    The upstream setup used ``periods=range(0, after + 1)`` solely to obtain
    this quantile-labelled factor index; event aggregation consumes neither
    those forward-return columns nor the raw factor values.  The Task 3
    forward-return kernel deliberately rejects a zero period, so this fixture
    records the exact source-cleaned projection instead of pretending to run a
    different production path.
    """

    dates = pd.date_range("2015-01-15", "2015-02-01", freq="D", name="date")
    assets = pd.Index(["A", "B", "C", "D"], name="asset")
    prices = pd.DataFrame(
        [[base**power for base in (1.25, 1.50, 1.00, 0.50)] for power in range(1, 19)],
        index=dates,
        columns=assets,
    )
    factor_dates = pd.date_range("2015-01-21", "2015-01-26", freq="D", name="date")
    quantile_row = [3, 4, 2, 1] if quantiles == 4 else [2, 2, 1, 1]
    factor_data = pd.DataFrame(
        {"factor_quantile": quantile_row * len(factor_dates)},
        index=pd.MultiIndex.from_product((factor_dates, assets), names=("date", "asset")),
    )
    assert quantiles in {2, 4}
    assert after >= 0
    return factor_data, prices


def _varying_event_factor_data(quantiles: int, after: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the exact relevant source-cleaned varying-universe projection."""

    dates = pd.date_range("2015-01-15", "2015-01-25", freq="D", name="date")
    assets = pd.Index(["A", "B", "C", "D", "E", "F"], name="asset")
    prices = pd.DataFrame(
        [[base**power for base in (1.25, 1.50, 1.00, 0.50, 1.50, 1.00)] for power in range(1, 12)],
        index=dates,
        columns=assets,
    )
    factor_dates = pd.date_range("2015-01-18", "2015-01-21", freq="D", name="date")
    factor_values = [
        [3, 4, 2, 1, np.nan, np.nan],
        [3, 4, 2, 1, np.nan, np.nan],
        [3, np.nan, np.nan, 1, 4, 2],
        [3, np.nan, np.nan, 1, 4, 2],
    ]
    factor = pd.Series(
        [value for row in factor_values for value in row],
        index=pd.MultiIndex.from_product((factor_dates, assets), names=("date", "asset")),
        dtype=float,
    ).dropna()
    quantile_projection = factor.astype(int) if quantiles == 4 else factor.map({1.0: 1, 2.0: 1, 3.0: 2, 4.0: 2})
    factor_data = pd.DataFrame(
        {"factor_quantile": quantile_projection},
        index=factor.index,
    )
    assert quantiles in {2, 4}
    assert after >= 0
    return factor_data, prices


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#00",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#00",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#01",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#01",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#02",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#02",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#03",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#03",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#04",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#04",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#05",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#05",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#06",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#06",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#06",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#07",
            id="tests/test_performance.py::PerformanceTestCase::test_common_start_returns#07",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_common_start_returns#07",
            ),
        ),
    ],
)
def test_common_start_returns_upstream_case(source_case_id: str) -> None:
    """Assert the complete pinned event-window summary for every source row."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    before, after, mean_by_date, demeaned, expected_values = _COMMON_START_RETURN_CASES[ordinal]
    source, returns = _common_inputs()
    original_source = source.copy(deep=True)
    original_returns = returns.copy(deep=True)
    actual = common_start_returns(
        source,
        returns,
        before,
        after,
        cumulative=True,
        mean_by_date=mean_by_date,
        demean_by=source if demeaned else None,
    )
    expected = pd.DataFrame(expected_values, index=pd.RangeIndex(-before, after + 1), columns=["mean", "std"])
    actual_summary = pd.DataFrame({"mean": actual.mean(axis=1), "std": actual.std(axis=1)})
    pd.testing.assert_frame_equal(actual_summary, expected, rtol=1e-5, atol=1e-7)
    strict_actual = strict_performance.common_start_returns(
        source,
        returns,
        before,
        after,
        cumulative=True,
        mean_by_date=mean_by_date,
        demean_by=source if demeaned else None,
    )
    strict_summary = pd.DataFrame({"mean": strict_actual.mean(axis=1), "std": strict_actual.std(axis=1)})
    pd.testing.assert_frame_equal(strict_summary, expected, rtol=1e-5, atol=1e-7)
    pd.testing.assert_series_equal(source, original_source)
    pd.testing.assert_frame_equal(returns, original_returns)


_AVERAGE_EVENT_CASES = (
    (
        1,
        2,
        False,
        4,
        [
            [0.00512695, 0.00256348, 0.00128174, 6.40869e-4],
            [0.00579185, 0.00289592, 0.00144796, 7.23981e-4],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [7.15814531, 8.94768164, 11.1846020, 13.9807526],
            [2.93784787, 3.67230984, 4.59038730, 5.73798413],
            [39.4519043, 59.1778564, 88.7667847, 133.150177],
            [28.3717330, 42.5575995, 63.8363992, 95.7545989],
        ],
    ),
    (
        1,
        2,
        True,
        4,
        [
            [-11.898667, -17.279462, -25.236885, -37.032252],
            [7.82587034, 11.5529583, 17.0996881, 25.3636472],
            [-10.903794, -16.282025, -24.238167, -36.032893],
            [7.82140124, 11.5507268, 17.0985737, 25.3630906],
            [-4.7456488, -8.3343438, -14.053565, -23.052140],
            [4.91184665, 7.91180853, 12.5481552, 19.6734224],
            [27.5481102, 41.8958311, 63.5286176, 96.1172844],
            [20.5510133, 31.0075980, 46.7385910, 70.3923129],
        ],
    ),
    (
        3,
        0,
        False,
        4,
        # The pinned source's dormant literal here disagrees with its own
        # executable period-0 setup.  These cells were independently captured
        # from the pinned commit's actual function path (the source fixture
        # itself retains all six event dates), rather than freezing dead data.
        [
            [0.0205078125, 0.01025390625, 0.005126953125, 0.0025634765625],
            [0.0231673888, 0.0115836944, 0.0057918479, 0.0028959239],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [4.5812129974, 5.7265162468, 7.1581453085, 8.9476816356],
            [1.8802226383, 2.3502782979, 2.9378478667, 3.6723098334],
            [17.5341796875, 26.3012695312, 39.4519042969, 59.1778564453],
            [12.6096591166, 18.9144886749, 28.3717330187, 42.5575995280],
        ],
    ),
    (
        0,
        3,
        True,
        4,
        [
            [-17.279462, -25.236885, -37.032252, -54.550061],
            [11.5529583, 17.0996881, 25.3636472, 37.6887906],
            [-16.282025, -24.238167, -36.032893, -53.550382],
            [11.5507268, 17.0985737, 25.3630906, 37.6885125],
            [-8.3343438, -14.053565, -23.052140, -37.074441],
            [7.91180853, 12.5481552, 19.6734224, 30.5748605],
            [41.8958311, 63.5286176, 96.1172844, 145.174884],
            [31.0075980, 46.7385910, 70.3923129, 105.944230],
        ],
    ),
    (
        3,
        3,
        False,
        2,
        [
            [0.5102539, 0.50512695, 0.50256348, 0.50128174, 0.50064087, 0.50032043, 0.50016022],
            [0.0115837, 0.00579185, 0.00289592, 1.44796e-3, 7.23981e-4, 3.61990e-4, 1.80995e-4],
            [11.057696, 16.0138929, 23.3050248, 34.0627690, 49.9756934, 73.5654648, 108.600603],
            [7.2389454, 10.6247239, 15.6450367, 23.1025693, 34.1977045, 50.7264595, 75.3771641],
        ],
    ),
    (
        3,
        3,
        True,
        2,
        [
            [-5.273721, -7.754383, -11.40123, -16.78074, -24.73753, -36.53257, -54.05022],
            [3.6239580, 5.3146000, 7.8236356, 11.551843, 17.099131, 25.363369, 37.688652],
            [5.2737212, 7.7543830, 11.401231, 16.780744, 24.737526, 36.532572, 54.050221],
            [3.6239580, 5.3146000, 7.8236356, 11.551843, 17.099131, 25.363369, 37.688652],
        ],
    ),
)


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#00",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#00",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#01",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#01",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#02",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#02",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#03",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#03",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#04",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#04",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#04",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#05",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#05",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile#05",
            ),
        ),
    ],
)
def test_average_cumulative_return_by_quantile_upstream_case(source_case_id: str) -> None:
    """Reconstruct each ordinary source row and assert every numeric cell."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    before, after, demeaned, quantiles, expected_values = _AVERAGE_EVENT_CASES[ordinal]
    source, returns = _ordinary_event_factor_data(quantiles, after)
    original_source = source.copy(deep=True)
    original_returns = returns.copy(deep=True)
    actual = average_cumulative_return_by_quantile(source, returns, before, after, demeaned)
    expected = pd.DataFrame(
        expected_values,
        index=pd.MultiIndex.from_product((range(1, quantiles + 1), ["mean", "std"]), names=("factor_quantile", None)),
        columns=pd.Index(range(-before, after + 1)),
    )
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-5, atol=1e-7)
    pd.testing.assert_frame_equal(
        strict_performance.average_cumulative_return_by_quantile(source, returns, before, after, demeaned),
        expected,
        rtol=1e-5,
        atol=1e-7,
    )
    pd.testing.assert_frame_equal(source, original_source)
    pd.testing.assert_frame_equal(returns, original_returns)


_AVERAGE_EVENT_VARYING_CASES = (
    (
        0,
        2,
        False,
        4,
        [
            [0.0292969, 0.0146484, 7.32422e-3],
            [0.0241851, 0.0120926, 6.04628e-3],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [3.5190582, 4.3988228, 5.49852848],
            [1.0046375, 1.2557969, 1.56974616],
            [10.283203, 15.424805, 23.1372070],
            [5.2278892, 7.8418338, 11.7627508],
        ],
    ),
    (
        0,
        3,
        True,
        4,
        [
            [-3.6785927, -5.1949205, -7.4034407, -10.641996],
            [1.57386873, 2.28176590, 3.33616491, 4.90228915],
            [-2.7078896, -4.2095690, -6.4107649, -9.6456583],
            [1.55205002, 2.27087143, 3.33072273, 4.89956999],
            [-0.1888313, -0.8107462, -1.9122365, -3.7724977],
            [0.55371389, 1.02143924, 1.76795263, 2.94536298],
            [6.57531357, 10.2152357, 15.7264421, 24.0601522],
            [3.67596914, 5.57112656, 8.43221341, 12.7447568],
        ],
    ),
    (
        0,
        3,
        False,
        2,
        [
            [0.51464844, 0.50732422, 0.50366211, 0.50183105],
            [0.01209256, 0.00604628, 0.00302314, 0.00151157],
            [6.90113068, 9.91181374, 14.3178678, 20.7894856],
            [3.11499629, 4.54718783, 6.66416616, 9.80049950],
        ],
    ),
    (
        0,
        3,
        True,
        2,
        [
            [-3.1932411, -4.7022448, -6.9071028, -10.143827],
            [1.56295067, 2.27631715, 3.33344356, 4.90092953],
            [3.19324112, 4.70224476, 6.90710282, 10.1438273],
            [1.56295067, 2.27631715, 3.33344356, 4.90092953],
        ],
    ),
)


@pytest.mark.parametrize(
    "source_case_id",
    [
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#00",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#00",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#00",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#01",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#01",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#01",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#02",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#02",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#02",
            ),
        ),
        pytest.param(
            "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#03",
            id="tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#03",
            marks=pytest.mark.alphalens_upstream_case(
                "tests/test_performance.py::PerformanceTestCase::test_average_cumulative_return_by_quantile_2#03",
            ),
        ),
    ],
)
def test_average_cumulative_return_by_quantile_2_upstream_case(source_case_id: str) -> None:
    """Keep the generated source-name collision and all literal rows live."""

    ordinal = int(source_case_id.rsplit("#", 1)[1])
    before, after, demeaned, quantiles, expected_values = _AVERAGE_EVENT_VARYING_CASES[ordinal]
    source, returns = _varying_event_factor_data(quantiles, after)
    original_source = source.copy(deep=True)
    original_returns = returns.copy(deep=True)
    actual = average_cumulative_return_by_quantile(source, returns, before, after, demeaned)
    expected = pd.DataFrame(
        expected_values,
        index=pd.MultiIndex.from_product((range(1, quantiles + 1), ["mean", "std"]), names=("factor_quantile", None)),
        columns=pd.Index(range(-before, after + 1)),
    )
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-5, atol=1e-7)
    pd.testing.assert_frame_equal(
        strict_performance.average_cumulative_return_by_quantile(source, returns, before, after, demeaned),
        expected,
        rtol=1e-5,
        atol=1e-7,
    )
    pd.testing.assert_frame_equal(source, original_source)
    pd.testing.assert_frame_equal(returns, original_returns)


def test_average_event_returns_by_group_enhanced_contract() -> None:
    """Enhanced event output has a group level when group-wise analytics are requested."""

    source, returns = _varying_event_factor_data(2, 1)
    source["group"] = pd.Categorical(["g1", "g2", "g2", "g1"] * 4)
    actual = average_cumulative_return_by_quantile(source, returns, 1, 1, by_group=True)
    assert actual.index.nlevels == 3
    pd.testing.assert_index_equal(actual.columns, pd.Index([-1, 0, 1]))
