"""Focused contracts for the public research-evaluation entry point."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fincore.factor_analysis import research

ASSETS = tuple(f"A{index:02d}" for index in range(12))
DATES = pd.bdate_range("2024-01-02", periods=8, tz="UTC")


def _panel_values(n_dates: int = 8, n_assets: int = 12) -> np.ndarray:
    assets = np.linspace(-0.05, 0.05, n_assets)
    sessions = np.linspace(-0.01, 0.01, n_dates)
    return assets[None, :] + sessions[:, None]


def _panel(n_dates: int = 8, n_assets: int = 12) -> dict[str, object]:
    dates = pd.bdate_range("2024-01-02", periods=n_dates, tz="UTC")
    assets = tuple(f"A{index:02d}" for index in range(n_assets))
    grid = pd.bdate_range(dates[0], periods=n_dates + 2, tz="UTC")
    values = _panel_values(n_dates, n_assets)
    opens = pd.DataFrame(index=grid, columns=list(assets), dtype=float)
    for position, asset in enumerate(assets):
        path = np.empty(len(grid), dtype=float)
        path[0] = 100.0
        path[1] = 100.0
        for session in range(n_dates):
            path[session + 2] = path[session + 1] * (1.0 + values[session, position])
        opens[asset] = path
    closes = opens * 0.97
    index = pd.MultiIndex.from_product((dates, assets), names=("time", "asset"))
    return {
        "dates": dates,
        "assets": assets,
        "grid": grid,
        "grid_frame": pd.DataFrame({"open_time": grid}, index=grid),
        "open": opens,
        "close": closes,
        "index": index,
        "values": values,
    }


def _factor(panel: dict[str, object], name: str = "momentum", direction: int = 1) -> pd.DataFrame:
    values = np.asarray(panel["values"], dtype=float) * float(direction)
    series = pd.Series(values.reshape(-1), index=panel["index"], name=name)
    return series.to_frame()


def _spec(panel: dict[str, object], **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "targets": panel["assets"],
        "start": panel["dates"][0],
        "end": panel["dates"][-1],
        "periods": (1,),
        "preprocess": "none",
        "min_assets": 10,
        "quantiles": 5,
        "stability_lags": (1,),
    }
    values.update(overrides)
    return values


def _inputs(
    panel: dict[str, object],
    spec: dict[str, object],
    *,
    universe: pd.DataFrame | None = None,
    prices: dict[str, pd.DataFrame] | None = None,
    precomputed_returns: pd.DataFrame | None = None,
    groups: dict[object, object] | None = None,
) -> dict[str, object]:
    return {
        "grid": panel["grid_frame"],
        "prices": prices if prices is not None else {"open": panel["open"], "close": panel["close"]},
        "universe": universe,
        "groups": groups,
        "precomputed_returns": precomputed_returns,
        "spec": spec,
    }


def _mean_ic(result: research.ResearchEvaluationResult, factor: str = "momentum") -> float:
    rows = result.ic_summary[result.ic_summary["factor"] == factor]
    return float(rows["mean_ic"].iloc[0])


def test_capabilities_expose_the_required_contract() -> None:
    capabilities = research.research_capabilities()

    assert capabilities["api_version"] == "1.0"
    assert capabilities["result_schema_version"] == 1
    assert capabilities["evaluation_config_schema_version"] == 1
    assert capabilities["python_min"] == "3.11"
    assert capabilities["entrypoint"] == "fincore.factor_analysis.research.evaluate_factor_research"
    assert capabilities["spec"] == "fincore.factor_analysis.research.ResearchEvaluationSpec"
    assert capabilities["normalizer"] == "fincore.factor_analysis.research.normalize_research_evaluation_spec"
    assert capabilities["labels"] == ("close_to_close", "next_open_to_open", "precomputed")
    assert capabilities["preprocess"] == ("cross_section_rank", "cross_section_zscore", "none")
    assert capabilities["metrics"] == (
        "pearson_ic",
        "spearman_ic",
        "icir",
        "quantile_returns",
        "spread",
        "rank_autocorrelation",
        "quantile_turnover",
        "factor_correlation",
        "paired_comparison",
    )
    assert capabilities["result_tables"] == (
        "signals",
        "labels",
        "ic",
        "ic_summary",
        "quantile_returns",
        "turnover",
        "spreads",
        "stability",
        "correlation",
        "comparisons",
        "coverage",
        "diagnostics",
    )
    assert capabilities["export_formats"] == ("json", "csv")
    assert isinstance(capabilities["charts_supported"], bool)


def test_normalize_spec_applies_documented_defaults() -> None:
    spec = research.normalize_research_evaluation_spec(
        {"targets": ["AAA", "BBB"], "start": "2024-01-02", "end": "2024-03-01"}
    )

    assert isinstance(spec, research.ResearchEvaluationSpec)
    assert spec.targets == ("AAA", "BBB")
    assert spec.start == pd.Timestamp("2024-01-02", tz="UTC")
    assert spec.end == pd.Timestamp("2024-03-01", tz="UTC")
    assert spec.snapshot_id == "unspecified"
    assert spec.price_basis == "unspecified"
    assert spec.calendar_id == "unspecified"
    assert spec.frequency == "unspecified"
    assert spec.source_timezone == "UTC"
    assert spec.feed_time_kind == "close_time"
    assert spec.periods == (1, 5, 20)
    assert spec.label == "next_open_to_open"
    assert spec.precomputed_entry == "next_open"
    assert spec.quantiles == 5
    assert spec.min_assets == 10
    assert spec.min_quantile_assets == 1
    assert spec.preprocess == "cross_section_rank"
    assert spec.max_loss == 1.0
    assert spec.stability_lags == (1, 5, 20)
    assert dict(spec.direction_overrides) == {}
    assert dict(spec.combinations) == {}
    assert dict(spec.comparisons) == {}
    assert dict(spec.segments) == {}


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"label": "forward_return"}, "unsupported_label"),
        ({"preprocess": "winsorize"}, "unsupported_preprocess"),
        ({"targets": ()}, "spec_field_empty"),
        ({"targets": ("AAA", "AAA")}, "spec_field_duplicate"),
        ({"start": "2024-04-01"}, "spec_window"),
        ({"unexpected": 1}, "unknown_spec_field"),
        ({"direction_overrides": {"momentum": 0}}, "spec_field_value"),
        ({"periods": (1, 1)}, "spec_field_duplicate"),
    ],
)
def test_normalize_spec_rejects_invalid_values(overrides: dict[str, object], code: str) -> None:
    payload: dict[str, object] = {
        "targets": ("AAA", "BBB"),
        "start": "2024-01-02",
        "end": "2024-03-01",
    }
    payload.update(overrides)

    with pytest.raises(research.ConfigError) as excinfo:
        research.normalize_research_evaluation_spec(payload)

    assert excinfo.value.code == code
    assert excinfo.value.report is not None


def test_forward_return_labels_match_hand_computed_values() -> None:
    dates = pd.bdate_range("2024-01-02", periods=4, tz="UTC")
    grid = pd.DataFrame({"open_time": dates}, index=dates)
    opens = pd.DataFrame({"X": [100.0, 100.0, 110.0, 121.0]}, index=dates)
    closes = pd.DataFrame({"X": [10.0, 11.0, 12.1, 13.31]}, index=dates)
    index = pd.MultiIndex.from_product((dates, ["X"]), names=("time", "asset"))
    factors = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0]}, index=index)

    def labels_for(label: str) -> dict[tuple[pd.Timestamp, str], float]:
        result = research.evaluate_factor_research(
            factors,
            grid=grid,
            prices={"open": opens, "close": closes},
            universe=None,
            groups=None,
            precomputed_returns=None,
            spec={
                "targets": ("X",),
                "start": dates[0],
                "end": dates[-1],
                "periods": (1, 2),
                "label": label,
                "preprocess": "none",
                "quantiles": 2,
                "min_assets": 1,
                "min_quantile_assets": 1,
                "stability_lags": (),
            },
        )
        rows = result.labels.set_index(["time", "period"])["value"]
        return {(time, period): float(value) for (time, period), value in rows.items()}

    next_open = labels_for("next_open_to_open")
    close_to_close = labels_for("close_to_close")

    # h=1 enters at the next open: t0 uses open[2]/open[1]-1.
    assert next_open[(dates[0], "1D")] == pytest.approx(0.10)
    assert next_open[(dates[0], "2D")] == pytest.approx(0.21)
    assert next_open[(dates[1], "1D")] == pytest.approx(0.10)
    assert np.isnan(next_open[(dates[1], "2D")])
    # h=2 closes at t+2: t0 = 110/100-1 and t1 = 121/110-1.
    assert close_to_close[(dates[0], "1D")] == pytest.approx(0.10)
    assert close_to_close[(dates[2], "1D")] == pytest.approx(0.10)
    assert close_to_close[(dates[1], "2D")] == pytest.approx(0.21)
    assert np.isnan(close_to_close[(dates[3], "1D")])


def test_perfect_predictor_scores_unit_ic_with_monotone_quantiles() -> None:
    panel = _panel()
    result = research.evaluate_factor_research(_factor(panel), **_inputs(panel, _spec(panel)))

    assert result.status == "OK"
    assert np.allclose(result.ic["ic"].to_numpy(dtype=float), 1.0, atol=1e-9)
    assert np.allclose(result.ic["rank_ic"].to_numpy(dtype=float), 1.0, atol=1e-9)
    assert (result.ic["n_assets"] == 12).all()
    assert float(result.ic_summary["mean_ic"].iloc[0]) == pytest.approx(1.0, abs=1e-9)
    assert int(result.ic_summary["n_dates"].iloc[0]) == 8
    assert float(result.stability["mean_rank_autocorrelation"].iloc[0]) == pytest.approx(1.0, abs=1e-9)

    coverage = result.coverage.iloc[0]
    assert coverage["factor"] == "momentum"
    assert coverage["kind"] == "factor"
    assert int(coverage["input_cells"]) == 8 * 12
    assert int(coverage["finite_cells"]) == 8 * 12
    assert int(coverage["label_cells"]) == 8 * 12
    assert int(coverage["evaluable_cells"]) == 8 * 12
    assert coverage["status"] == "OK"

    assert (result.spreads["spread"] > 0.0).all()
    profile = result.quantile_returns.groupby("quantile", observed=False)["mean_return"].mean()
    assert profile.index.is_monotonic_increasing
    assert profile.is_monotonic_increasing
    assert float(profile.iloc[-1] - profile.iloc[0]) == pytest.approx(float(result.spreads["spread"].mean()), abs=1e-9)


def test_negative_direction_flips_the_information_coefficient() -> None:
    panel = _panel()
    factors = _factor(panel)
    factors.attrs["factor_directions"] = {"momentum": -1}

    result = research.evaluate_factor_research(factors, **_inputs(panel, _spec(panel)))

    assert _mean_ic(result) == pytest.approx(-1.0, abs=1e-9)
    assert set(result.signals["direction"]) == {-1}
    assert result.signals["value"].to_numpy(dtype=float) == pytest.approx(
        -np.asarray(panel["values"], dtype=float).reshape(-1)
    )


def test_direction_override_takes_precedence() -> None:
    panel = _panel()
    factors = _factor(panel)
    factors.attrs["factor_directions"] = {"momentum": -1}
    spec = _spec(panel, direction_overrides={"momentum": 1})

    result = research.evaluate_factor_research(factors, **_inputs(panel, spec))

    assert _mean_ic(result) == pytest.approx(1.0, abs=1e-9)
    assert set(result.signals["direction"]) == {1}


def test_universe_false_rows_are_excluded_from_ic_and_counts() -> None:
    panel = _panel()
    universe = pd.DataFrame(True, index=panel["dates"], columns=list(panel["assets"]))
    universe[["A00", "A01"]] = False

    result = research.evaluate_factor_research(_factor(panel), **_inputs(panel, _spec(panel), universe=universe))

    assert result.status == "OK"
    assert (result.ic["n_assets"] == 10).all()
    assert _mean_ic(result) == pytest.approx(1.0, abs=1e-9)
    coverage = result.coverage.iloc[0]
    assert int(coverage["input_cells"]) == 8 * 10
    assert int(coverage["finite_cells"]) == 8 * 10
    assert int(coverage["evaluable_cells"]) == 8 * 10
    assert len(result.signals) == 8 * 10


def test_all_nan_factor_is_unevaluable_with_diagnostics_not_an_exception() -> None:
    panel = _panel()
    factors = pd.DataFrame(
        {"broken": np.nan},
        index=panel["index"],
    )

    result = research.evaluate_factor_research(factors, **_inputs(panel, _spec(panel)))

    assert result.status == "UNEVALUABLE"
    assert result.ic.empty
    assert int(result.ic_summary["n_dates"].iloc[0]) == 0
    assert np.isnan(float(result.ic_summary["mean_ic"].iloc[0]))
    assert result.coverage["status"].tolist() == ["UNEVALUABLE"]
    assert int(result.coverage["finite_cells"].iloc[0]) == 0
    assert "no_finite_factor" in set(result.diagnostics["code"])


def test_mixed_factors_report_partial_status() -> None:
    panel = _panel()
    good = _factor(panel)
    broken = pd.DataFrame({"broken": np.nan}, index=panel["index"])
    factors = pd.concat([good, broken], axis=1)

    result = research.evaluate_factor_research(factors, **_inputs(panel, _spec(panel)))

    assert result.status == "PARTIAL"
    assert dict(zip(result.coverage["factor"], result.coverage["status"], strict=True)) == {
        "momentum": "OK",
        "broken": "UNEVALUABLE",
    }


def test_rejects_duplicate_keys_and_non_utc_indexes() -> None:
    panel = _panel()
    factors = _factor(panel)
    duplicated = pd.concat([factors, factors.iloc[[0]]])
    with pytest.raises(research.DataQualityError) as excinfo:
        research.evaluate_factor_research(duplicated, **_inputs(panel, _spec(panel)))
    assert excinfo.value.code == "factors_duplicate_keys"

    naive = factors.copy()
    naive.index = pd.MultiIndex.from_arrays(
        (pd.DatetimeIndex(naive.index.get_level_values(0)).tz_localize(None), naive.index.get_level_values(1)),
        names=("time", "asset"),
    )
    with pytest.raises(research.DataQualityError) as excinfo:
        research.evaluate_factor_research(naive, **_inputs(panel, _spec(panel)))
    assert excinfo.value.code == "factors_timezone"


def test_rejects_missing_prices_and_missing_targets() -> None:
    panel = _panel()
    factors = _factor(panel)
    absent_prices = _inputs(panel, _spec(panel))
    absent_prices["prices"] = None
    with pytest.raises(research.ConfigError) as excinfo:
        research.evaluate_factor_research(factors, **absent_prices)
    assert excinfo.value.code == "missing_prices"

    with pytest.raises(research.DatasetError) as excinfo:
        research.evaluate_factor_research(
            factors,
            **_inputs(panel, _spec(panel, targets=(*panel["assets"], "MISSING"))),
        )
    assert excinfo.value.code == "factors_missing_targets"


def test_precomputed_returns_are_used_only_when_selected() -> None:
    panel = _panel()
    precomputed = pd.DataFrame(
        {"1D": np.asarray(panel["values"], dtype=float).reshape(-1)},
        index=panel["index"],
    )

    result = research.evaluate_factor_research(
        _factor(panel),
        **_inputs(panel, _spec(panel, label="precomputed"), prices=None, precomputed_returns=precomputed),
    )

    assert result.status == "OK"
    assert result.manifest["label_mode"] == "precomputed"
    assert _mean_ic(result) == pytest.approx(1.0, abs=1e-9)

    with pytest.raises(research.ConfigError) as excinfo:
        research.evaluate_factor_research(
            _factor(panel),
            **_inputs(panel, _spec(panel), precomputed_returns=precomputed),
        )
    assert excinfo.value.code == "precomputed_returns_unused"


def test_combination_and_comparison_tables() -> None:
    panel = _panel()
    values = np.asarray(panel["values"], dtype=float)
    factors = pd.concat(
        [
            _factor(panel, "momentum"),
            pd.Series((values * 2.0).reshape(-1), index=panel["index"], name="value").to_frame(),
        ],
        axis=1,
    )
    spec = _spec(
        panel,
        combinations={"blend": {"momentum": 0.5, "value": 0.5}},
        comparisons={"head_to_head": ("momentum", "value")},
    )

    result = research.evaluate_factor_research(factors, **_inputs(panel, spec))

    assert set(result.coverage["kind"]) == {"factor", "combination"}
    assert "blend" in set(result.ic_summary["factor"])
    assert _mean_ic(result, "blend") == pytest.approx(1.0, abs=1e-9)
    assert result.comparisons["comparison"].unique().tolist() == ["head_to_head"]
    comparison = result.comparisons.iloc[0]
    assert comparison["left"] == "momentum"
    assert comparison["right"] == "value"
    assert float(comparison["mean_ic"]) == pytest.approx(0.0, abs=1e-9)
    assert len(result.correlation) == 3
    assert np.allclose(result.correlation["mean_correlation"], 1.0, atol=1e-9)


def test_combination_errors_are_rejected() -> None:
    panel = _panel()
    factors = _factor(panel)

    with pytest.raises(research.ConfigError) as excinfo:
        research.evaluate_factor_research(
            factors,
            **_inputs(panel, _spec(panel, combinations={"blend": {"missing": 1.0}})),
        )
    assert excinfo.value.code == "combination_unknown_factor"

    with pytest.raises(research.ConfigError) as excinfo:
        research.evaluate_factor_research(
            factors,
            **_inputs(panel, _spec(panel, combinations={"momentum": {"momentum": 1.0}})),
        )
    assert excinfo.value.code == "combination_name_conflict"


def test_export_writes_tables_and_manifest_and_refuses_overwrite(tmp_path) -> None:
    panel = _panel()
    result = research.evaluate_factor_research(_factor(panel), **_inputs(panel, _spec(panel)))

    target = result.export(tmp_path, charts=False)

    assert target.name == result.manifest["evaluation_id"]
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evaluation_id"] == result.manifest["evaluation_id"]
    assert manifest["result_fingerprint"] == result.manifest["result_fingerprint"]
    for table in research.research_capabilities()["result_tables"]:
        assert (target / f"{table}.csv").exists(), table
    assert "signals.csv" in manifest["exports"]

    with pytest.raises(research.DatasetError) as excinfo:
        result.export(tmp_path, charts=False)
    assert excinfo.value.code == "export_target_exists"

    if research.research_capabilities()["charts_supported"]:
        chart_target = result.export(tmp_path / "charts")
        assert (chart_target / "ic_series.png").exists()
        assert (chart_target / "quantile_returns.png").exists()


def test_result_fingerprint_is_stable_and_input_sensitive() -> None:
    panel = _panel()
    factors = _factor(panel)
    inputs = _inputs(panel, _spec(panel))

    first = research.evaluate_factor_research(factors, **inputs)
    second = research.evaluate_factor_research(factors, **inputs)

    fingerprint = first.manifest["result_fingerprint"]
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64
    assert int(fingerprint, 16) >= 0
    assert fingerprint == second.manifest["result_fingerprint"]
    assert first.manifest["evaluation_id"] == second.manifest["evaluation_id"]

    changed = factors.copy()
    changed.iloc[0, 0] = float(changed.iloc[0, 0]) + 0.25
    third = research.evaluate_factor_research(changed, **inputs)
    assert third.manifest["result_fingerprint"] != fingerprint
    assert third.manifest["evaluation_id"] != first.manifest["evaluation_id"]

    payload = first.to_serializable()
    assert payload["status"] == "OK"
    assert set(payload["tables"]) == set(research.research_capabilities()["result_tables"])
    json.dumps(payload, allow_nan=False)


def test_insufficient_assets_are_reported_as_diagnostics() -> None:
    panel = _panel(n_dates=8, n_assets=4)
    spec = _spec(panel, min_assets=10, quantiles=2)

    result = research.evaluate_factor_research(_factor(panel), **_inputs(panel, spec))

    assert result.status == "UNEVALUABLE"
    assert int(result.ic_summary["n_dates"].iloc[0]) == 0
    assert "insufficient_assets" in set(result.diagnostics["code"])
