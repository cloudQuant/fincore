"""Deterministic enhanced factor-analysis benchmark workloads.

This module deliberately imports only ``fincore.factor_analysis`` kernels and
models.  The strict Alphalens facade is outside the benchmark boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

from fincore.factor_analysis.analysis import analyze_factor
from fincore.factor_analysis.data import prepare_factor_data, quantize_factor
from fincore.factor_analysis.models import fingerprint_value
from fincore.factor_analysis.performance import (
    average_cumulative_return_by_quantile,
    common_start_returns,
    factor_information_coefficient,
    factor_returns,
    factor_weights,
)

SEED = 20260815


@dataclass(frozen=True)
class Scenario:
    """One fixed-size reproducible scenario and its enhanced workloads."""

    name: str
    dates: int
    assets: int
    kernels: tuple[str, ...]

    @property
    def input_shape(self) -> dict[str, int]:
        return {"dates": self.dates, "assets": self.assets, "rows": self.dates * self.assets}


SCENARIOS = {
    "small-ci": Scenario(
        name="small-ci",
        dates=252,
        assets=100,
        kernels=("prepare", "quantize", "information-coefficient", "weights"),
    ),
    "medium-artifact": Scenario(
        name="medium-artifact",
        dates=1_260,
        assets=500,
        kernels=("prepare", "factor-returns", "full-model"),
    ),
    "event": Scenario(
        name="event",
        dates=756,
        assets=200,
        kernels=("common-start", "event-average"),
    ),
}


def _raw_inputs(scenario: Scenario) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2010-01-04", periods=scenario.dates + 6, name="date")
    assets = pd.Index([f"A{number:04d}" for number in range(scenario.assets)], name="asset")
    factor_index = pd.MultiIndex.from_product((dates[: scenario.dates], assets), names=("date", "asset"))
    factor = pd.Series(rng.standard_normal(len(factor_index)), index=factor_index, name="factor")
    innovations = rng.normal(0.0002, 0.01, (len(dates), len(assets)))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(innovations, axis=0)), index=dates, columns=assets)
    groups = pd.Series({asset: f"G{number % 10:02d}" for number, asset in enumerate(assets)}, name="group")
    return factor, prices, groups


def _clean_inputs(scenario: Scenario) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor, prices, groups = _raw_inputs(scenario)
    prepared = prepare_factor_data(
        factor,
        prices,
        groupby=groups,
        periods=(1, 5),
        quantiles=5,
        max_loss=1,
    )
    return prepared.data, prices.pct_change(fill_method=None).fillna(0.0)


def _event_inputs(scenario: Scenario) -> tuple[pd.DataFrame, pd.DataFrame]:
    _factor, prices, groups = _raw_inputs(scenario)
    returns = prices.iloc[: scenario.dates].pct_change(fill_method=None).fillna(0.0)
    event_dates = returns.index[10:-10:21]
    event_assets = returns.columns[::5]
    event_index = pd.MultiIndex.from_product((event_dates, event_assets), names=("date", "asset"))
    rng = np.random.default_rng(SEED + 1)
    factor_data = pd.DataFrame(
        {
            "factor": rng.standard_normal(len(event_index)),
            "factor_quantile": np.tile(np.arange(1, 6), int(np.ceil(len(event_index) / 5)))[: len(event_index)],
            "group": groups.reindex(event_index.get_level_values("asset")).to_numpy(),
        },
        index=event_index,
    )
    return factor_data, returns


def build_workload(scenario_name: str, kernel: str) -> Callable[[], object]:
    """Construct inputs outside the measurement window and return one workload."""

    scenario = SCENARIOS[scenario_name]
    if kernel not in scenario.kernels:
        raise ValueError(f"kernel {kernel!r} is not part of scenario {scenario_name!r}")

    if kernel == "prepare":
        factor, prices, groups = _raw_inputs(scenario)
        return lambda: prepare_factor_data(
            factor,
            prices,
            groupby=groups,
            periods=(1, 5),
            quantiles=5,
            max_loss=1,
        ).data

    if scenario_name == "event":
        factor_data, returns = _event_inputs(scenario)
        if kernel == "common-start":
            return lambda: common_start_returns(
                factor_data["factor_quantile"],
                returns,
                before=10,
                after=15,
                cumulative=True,
                mean_by_date=True,
            )
        return lambda: average_cumulative_return_by_quantile(
            factor_data,
            returns,
            periods_before=10,
            periods_after=15,
            demeaned=True,
        )

    clean, _returns = _clean_inputs(scenario)
    if kernel == "quantize":
        source = clean.drop(columns="factor_quantile")
        return lambda: quantize_factor(source, quantiles=5)
    if kernel == "information-coefficient":
        return lambda: factor_information_coefficient(clean)
    if kernel == "weights":
        return lambda: factor_weights(clean)
    if kernel == "factor-returns":
        return lambda: factor_returns(clean)
    if kernel == "full-model":
        return lambda: analyze_factor(clean, include_pyfolio=True)
    raise AssertionError(f"unhandled benchmark kernel {kernel!r}")


def output_metadata(value: object) -> tuple[object, str]:
    """Return a JSON-compatible output shape and deterministic SHA-256 digest."""

    result_fingerprint = getattr(value, "result_fingerprint", None)
    if isinstance(result_fingerprint, str) and len(result_fingerprint) == 64:
        forward_periods = tuple(getattr(value, "forward_periods", ()))
        shape: object = {
            "factor_data": list(cast("Any", value).factor_data.shape),
            "forward_periods": list(forward_periods),
        }
        return shape, result_fingerprint
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return list(value.shape), fingerprint_value(value)
    raise TypeError(f"unsupported benchmark output {type(value).__name__}")


__all__ = ["SCENARIOS", "SEED", "Scenario", "build_workload", "output_metadata"]
