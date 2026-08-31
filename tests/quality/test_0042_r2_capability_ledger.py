"""Fail-closed contracts for the scoped 0042-R2 capability ledger.

The scoped ledger covers the metrics and performance families only.  It binds
each capability to real collected source nodeids, planned wheel nodeids, and
one independent-authority scenario.  It is explicitly not D0 evidence: the
capture tool rejects it while it remains scoped.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
FIXTURES = REPOSITORY_ROOT / "tests" / "parity" / "fixtures"
LEDGER = FIXTURES / "capability-ledger-0042-r2.json"
INVENTORY = FIXTURES / "legacy-surface-inventory-0042-r2.json"
NODE_FACTS = FIXTURES / "test-node-facts-discovery-0042-r2.json"
UPSTREAM_MANIFEST = REPOSITORY_ROOT / "tests" / "compat" / "fixtures" / "empyrical-0.6.0-api.json"
ALPHALENS_MANIFEST = REPOSITORY_ROOT / "tests" / "compat" / "fixtures" / "alphalens-0.4.0-cloudquant-api.json"
CAPTURE_SCRIPT = REPOSITORY_ROOT / "scripts" / "capture_capability_baseline.py"

# Packaging inventory rows are distribution configuration, not analytical
# capabilities; they remain outside the ledger by design and are gated by the
# package/installed lanes instead.
_COVERED_OWNERS = (
    "metrics",
    "performance",
    "factor",
    "risk",
    "attribution",
    "simulation",
    "optimization",
    "portfolio",
    "report",
    "viz",
    "data",
    "extensions",
    "runtime",
)
_REQUIRED_NON_ASSERTIONS = frozenset({"D0", "D-TECH", "installed_wheel_behavior", "legacy_zero"})


def _load_capture_module():
    specification = importlib.util.spec_from_file_location("capture_capability_baseline_ledger_test", CAPTURE_SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    original = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = original
    return module


def _load() -> dict:
    assert LEDGER.is_file(), "committed capability ledger fixture is missing"
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def _inventory_targets() -> dict[str, str]:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    targets: dict[str, str] = {}
    for entry in inventory["entries"]:
        target = entry["target_operation_id"]
        if entry["owner"] in _COVERED_OWNERS and ".surface." not in target:
            targets[target] = entry["owner"]
    return targets


def test_ledger_header_is_scoped_and_fail_closed() -> None:
    ledger = _load()

    assert ledger["schema_version"] == 1
    assert ledger["artifact_type"] == "capability_ledger"
    assert ledger["scope"] == "all_capability_families_except_packaging"
    assert ledger["covered_families"] == sorted(_COVERED_OWNERS)
    assert ledger["decision_status"] == "scoped"
    assert ledger["not_for_d0"] is True
    assert set(ledger["does_not_assert"]) >= _REQUIRED_NON_ASSERTIONS


def test_ledger_passes_the_frozen_capture_schema() -> None:
    capture = _load_capture_module()
    ledger = _load()

    entries = capture.validate_ledger(ledger)

    assert len(entries) == len(ledger["entries"])
    assert entries


def test_ledger_source_nodeids_are_real_collected_nodes() -> None:
    ledger = _load()
    facts = json.loads(NODE_FACTS.read_text(encoding="utf-8"))
    collected = {node["nodeid"] for node in facts["nodes"]}

    for entry in ledger["entries"]:
        unknown = [nodeid for nodeid in entry["source_nodeids"] if nodeid not in collected]
        assert unknown == [], entry["capability_id"]


def test_ledger_covers_every_covered_inventory_target_exactly_once() -> None:
    ledger = _load()
    targets = _inventory_targets()

    ledger_ids = {entry["capability_id"] for entry in ledger["entries"]}
    gap_ids = {gap["capability_id"] for gap in ledger["coverage_gaps"]}

    assert ledger_ids.isdisjoint(gap_ids)
    assert ledger_ids | gap_ids == set(targets)
    for entry in ledger["entries"]:
        assert targets[entry["capability_id"]] == entry["owner"]


def test_ledger_source_contract_binds_the_committed_input_bytes() -> None:
    ledger = _load()
    contract = ledger["source_contract"]

    assert contract["inventory_sha256"] == hashlib.sha256(INVENTORY.read_bytes()).hexdigest()
    assert contract["node_facts_sha256"] == hashlib.sha256(NODE_FACTS.read_bytes()).hexdigest()
    assert contract["upstream_manifest_sha256"] == hashlib.sha256(UPSTREAM_MANIFEST.read_bytes()).hexdigest()
    assert contract["alphalens_manifest_sha256"] == hashlib.sha256(ALPHALENS_MANIFEST.read_bytes()).hexdigest()


def test_upstream_capabilities_use_the_pinned_empyrical_oracle() -> None:
    ledger = _load()
    manifest = json.loads(UPSTREAM_MANIFEST.read_text(encoding="utf-8"))
    pinned_commit = manifest["commit"]
    upstream_symbols = {item["symbol"] for item in manifest["callables"]}

    for entry in ledger["entries"]:
        scenario = entry["scenarios"][0]
        authority = scenario["authority"]
        if authority["kind"] == "pinned_upstream_oracle" and authority["source_project"] == "empyrical":
            reference_symbol = authority["reference"].split(".", 1)[1]
            assert reference_symbol in upstream_symbols, entry["capability_id"]
            assert authority["artifact_digest"] == f"git-commit:{pinned_commit}", entry["capability_id"]
            assert authority["version"] == "0.6.0", entry["capability_id"]


def test_factor_capabilities_use_the_pinned_alphalens_oracle() -> None:
    ledger = _load()
    manifest = json.loads(ALPHALENS_MANIFEST.read_text(encoding="utf-8"))
    pinned_commit = manifest["identity"]["value"]
    upstream_symbols = {item["symbol"] for item in manifest["entries"] if item.get("kind") in {"function", "class"}}

    alphalens_references = 0
    for entry in ledger["entries"]:
        if entry["owner"] != "factor":
            continue
        scenario = entry["scenarios"][0]
        authority = scenario["authority"]
        if authority["kind"] == "pinned_upstream_oracle" and authority["source_project"] == "alphalens":
            reference_symbol = authority["reference"].rsplit(".", 1)[1]
            assert reference_symbol in upstream_symbols, entry["capability_id"]
            assert authority["artifact_digest"] == f"git-commit:{pinned_commit}", entry["capability_id"]
            assert authority["version"] == "0.4.0", entry["capability_id"]
            alphalens_references += 1

    assert alphalens_references > 0, "factor tranche must bind alphalens-derived capabilities to the pinned oracle"


def test_coverage_gaps_declare_only_missing_source_evidence() -> None:
    ledger = _load()

    assert ledger["coverage_gaps"], "a scoped ledger must declare its remaining coverage gaps"
    for gap in ledger["coverage_gaps"]:
        assert set(gap) == {"capability_id", "reason"}
        assert gap["reason"] == "no_source_nodeid"


def test_metrics_drawdown_and_leverage_coverage_is_backed_by_real_scenarios() -> None:
    """Prevent these implemented metrics from regressing to unsupported gaps.

    The three functions have stable, domain-native oracle cases.  They must
    therefore be represented by executable source and installed-wheel
    scenarios instead of indefinitely remaining in the generic coverage-gap
    queue.
    """
    ledger = _load()
    expected = {
        "metrics.gross_leverage": {
            "tests/parity/test_metrics.py::test_gross_leverage",
        },
        "metrics.second_max_drawdown": {
            "tests/parity/test_metrics.py::test_second_max_drawdown",
        },
        "metrics.third_max_drawdown": {
            "tests/parity/test_metrics.py::test_third_max_drawdown",
        },
    }

    gap_ids = {gap["capability_id"] for gap in ledger["coverage_gaps"]}
    entries = {entry["capability_id"]: entry for entry in ledger["entries"]}

    assert not expected.keys() & gap_ids
    for capability_id, wheel_nodeids in expected.items():
        entry = entries[capability_id]
        assert entry["owner"] == "metrics"
        assert entry["disposition"] == "required"
        assert entry["source_nodeids"]
        assert set(entry["wheel_nodeids"]) == wheel_nodeids


def test_optimization_and_simulation_gaps_are_backed_by_canonical_scenarios() -> None:
    """Keep direct domain behaviors out of the generic coverage-gap queue."""
    ledger = _load()
    expected = {
        "optimization.optimization_error": {
            "tests/parity/test_optimization.py::test_optimization_error",
        },
        "optimization.optimize": {
            "tests/parity/test_optimization.py::test_optimize",
        },
        "optimization.risk_parity": {
            "tests/parity/test_optimization.py::test_risk_parity",
        },
        "simulation.monte_carlo": {
            "tests/parity/test_simulation.py::test_monte_carlo",
        },
    }

    gap_ids = {gap["capability_id"] for gap in ledger["coverage_gaps"]}
    entries = {entry["capability_id"]: entry for entry in ledger["entries"]}

    assert not expected.keys() & gap_ids
    for capability_id, wheel_nodeids in expected.items():
        entry = entries[capability_id]
        assert entry["disposition"] == "required"
        assert entry["source_nodeids"]
        assert set(entry["wheel_nodeids"]) == wheel_nodeids


def test_performance_cashflow_disclosure_and_inference_have_canonical_scenarios() -> None:
    """Enhanced performance primitives must not remain unreviewed coverage gaps."""
    ledger = _load()
    expected = {
        "performance.cashflow.cashflow_adjusted_returns": {
            "tests/parity/test_performance.py::test_cashflow_adjusted_returns",
        },
        "performance.cashflow.cashflow_adjusted_twr": {
            "tests/parity/test_performance.py::test_cashflow_adjusted_twr",
        },
        "performance.cashflow.cashflow_timing": {
            "tests/parity/test_performance.py::test_cashflow_timing_and_fee_treatment",
        },
        "performance.cashflow.fee_treatment": {
            "tests/parity/test_performance.py::test_cashflow_timing_and_fee_treatment",
        },
        "performance.disclosure.disclosure_context": {
            "tests/parity/test_performance.py::test_disclosure_context",
        },
        "performance.disclosure.render_disclosure": {
            "tests/parity/test_performance.py::test_render_disclosure",
        },
        "performance.inference.sharpe_confidence_interval": {
            "tests/parity/test_performance.py::test_sharpe_inference",
        },
        "performance.inference.sharpe_standard_error": {
            "tests/parity/test_performance.py::test_sharpe_inference",
        },
        "performance.inference.standard_error_of_mean": {
            "tests/parity/test_performance.py::test_sharpe_inference",
        },
    }

    gap_ids = {gap["capability_id"] for gap in ledger["coverage_gaps"]}
    entries = {entry["capability_id"]: entry for entry in ledger["entries"]}

    assert not expected.keys() & gap_ids
    for capability_id, wheel_nodeids in expected.items():
        entry = entries[capability_id]
        assert entry["owner"] == "performance"
        assert entry["disposition"] == "required"
        assert entry["source_nodeids"]
        assert set(entry["wheel_nodeids"]) == wheel_nodeids


def test_risk_contracts_are_backed_by_executable_reference_scenarios() -> None:
    """Risk models, diagnostics, and audit reports cannot remain generic gaps."""
    ledger = _load()
    expected = {
        "risk.basel_reference_disclosure": {
            "tests/numerical/test_risk_validation_report.py::test_report_reconstructs_every_forecast_exception_and_refit",
        },
        "risk.build_risk_validation_report": {
            "tests/numerical/test_risk_validation_report.py::test_report_reconstructs_every_forecast_exception_and_refit",
        },
        "risk.evt": {
            "tests/numerical/test_risk_reference_oracles.py::TestEVTSemantics::test_gpd_pwm_matches_independent_l_moment_estimator",
        },
        "risk.risk_backtest_result": {
            "tests/test_risk/test_backtesting.py::test_var_backtest_keeps_time_alignment_and_exception_count",
        },
        "risk.risk_model_spec": {
            "tests/numerical/test_risk_model_validation.py::TestRiskModelSpec::test_defaults",
        },
        "risk.risk_validation_report": {
            "tests/numerical/test_risk_validation_report.py::test_report_reconstructs_every_forecast_exception_and_refit",
        },
        "risk.risk_validation_report_schema_version": {
            "tests/numerical/test_risk_validation_report.py::test_report_reconstructs_every_forecast_exception_and_refit",
        },
        "risk.walk_forward_va_r_result": {
            "tests/numerical/test_risk_model_validation.py::TestWalkForward::test_walk_forward_result_enforces_status_specific_state_invariants",
        },
        "risk.walk_forward_validation": {
            "tests/numerical/test_risk_model_validation.py::TestWalkForward::test_public_historical_forecast_is_strictly_out_of_sample_and_backtestable",
        },
        "risk.walk_forward_var": {
            "tests/numerical/test_risk_model_validation.py::TestWalkForward::test_public_historical_forecast_is_strictly_out_of_sample_and_backtestable",
        },
    }

    gap_ids = {gap["capability_id"] for gap in ledger["coverage_gaps"]}
    entries = {entry["capability_id"]: entry for entry in ledger["entries"]}

    assert not expected.keys() & gap_ids
    for capability_id, wheel_nodeids in expected.items():
        entry = entries[capability_id]
        assert entry["owner"] == "risk"
        assert entry["disposition"] == "required"
        assert entry["source_nodeids"]
        assert set(entry["wheel_nodeids"]) == wheel_nodeids


def test_data_provider_contracts_have_offline_executable_scenarios() -> None:
    """Network integrations need hermetic provider and error-path evidence."""
    ledger = _load()
    expected = {
        "data.provider.ak_share_provider": {
            "tests/test_data/test_providers_offline_fetch.py::test_akshare_provider_offline_fetch_and_info_via_stub_module",
        },
        "data.provider.alpha_vantage_provider": {
            "tests/test_data/test_providers_offline_fetch.py::test_alpha_vantage_fetch_success_and_error_are_offline",
        },
        "data.provider.alphavantage": {
            "tests/test_data/providers_unit/test_convenience.py::TestConvenienceFunctionsUnit::test_get_provider_av",
        },
        "data.provider.data_provider": {
            "tests/test_data/test_providers_offline_fetch.py::test_fetch_price_data_and_multiple_prices_default_date_logic",
        },
    }

    gap_ids = {gap["capability_id"] for gap in ledger["coverage_gaps"]}
    entries = {entry["capability_id"]: entry for entry in ledger["entries"]}

    assert not expected.keys() & gap_ids
    for capability_id, wheel_nodeids in expected.items():
        entry = entries[capability_id]
        assert entry["owner"] == "data"
        assert entry["disposition"] == "required"
        assert entry["source_nodeids"]
        assert set(entry["wheel_nodeids"]) == wheel_nodeids


def test_extension_snapshot_capabilities_have_canonical_scenarios() -> None:
    """Every extension registry behavior needs an executable isolation contract."""
    ledger = _load()
    expected = {
        "extensions.snapshot.clear_registry": {
            "tests/parity/test_extensions.py::test_registry_lookup_isolation_and_clear_policy",
        },
        "extensions.snapshot.default_metric_family": {
            "tests/parity/test_extensions.py::test_extension_types_and_default_metric_families",
        },
        "extensions.snapshot.duplicate_policy": {
            "tests/parity/test_extensions.py::test_metric_registration_lookup_and_duplicate_policy",
        },
        "extensions.snapshot.duplicate_registration_error": {
            "tests/parity/test_extensions.py::test_metric_registration_lookup_and_duplicate_policy",
        },
        "extensions.snapshot.extension_kind": {
            "tests/parity/test_extensions.py::test_extension_types_and_default_metric_families",
        },
        "extensions.snapshot.extension_registry": {
            "tests/parity/test_extensions.py::test_registry_lookup_isolation_and_clear_policy",
        },
        "extensions.snapshot.get_metric": {
            "tests/parity/test_extensions.py::test_metric_registration_lookup_and_duplicate_policy",
        },
        "extensions.snapshot.get_registry": {
            "tests/parity/test_extensions.py::test_registry_lookup_isolation_and_clear_policy",
        },
        "extensions.snapshot.get_viz_backend": {
            "tests/parity/test_extensions.py::test_viz_backend_registration_lookup_and_listing",
        },
        "extensions.snapshot.isolated_registry": {
            "tests/parity/test_extensions.py::test_registry_lookup_isolation_and_clear_policy",
        },
        "extensions.snapshot.list_hooks": {
            "tests/parity/test_extensions.py::test_hook_registration_listing_and_execution",
        },
        "extensions.snapshot.list_metrics": {
            "tests/parity/test_extensions.py::test_metric_registration_lookup_and_duplicate_policy",
        },
        "extensions.snapshot.list_viz_backends": {
            "tests/parity/test_extensions.py::test_viz_backend_registration_lookup_and_listing",
        },
        "extensions.snapshot.register_hook": {
            "tests/parity/test_extensions.py::test_hook_registration_listing_and_execution",
        },
        "extensions.snapshot.register_metric": {
            "tests/parity/test_extensions.py::test_metric_registration_lookup_and_duplicate_policy",
        },
        "extensions.snapshot.register_viz_backend": {
            "tests/parity/test_extensions.py::test_viz_backend_registration_lookup_and_listing",
        },
        "extensions.snapshot.registration": {
            "tests/parity/test_extensions.py::test_extension_types_and_default_metric_families",
        },
        "extensions.snapshot.rolling_family": {
            "tests/parity/test_extensions.py::test_extension_types_and_default_metric_families",
        },
    }

    gap_ids = {gap["capability_id"] for gap in ledger["coverage_gaps"]}
    entries = {entry["capability_id"]: entry for entry in ledger["entries"]}

    assert not expected.keys() & gap_ids
    for capability_id, wheel_nodeids in expected.items():
        entry = entries[capability_id]
        assert entry["owner"] == "extensions"
        assert entry["disposition"] == "required"
        assert entry["source_nodeids"]
        assert set(entry["wheel_nodeids"]) == wheel_nodeids


def test_factor_models_costs_pit_and_inference_have_canonical_scenarios() -> None:
    """Bind the factor-analysis kernel to direct numerical and model scenarios.

    The retained source cases exercise the enhanced domain implementation; the
    strict profile tests referenced by the two legacy error identities are
    temporary migration-oracle evidence and do not authorize a final facade.
    """
    ledger = _load()
    expected = {
        "factor.analysis.apply_factor_costs": {
            "tests/parity/test_factor_analysis.py::test_apply_factor_costs_and_capacity",
        },
        "factor.analysis.estimate_factor_capacity": {
            "tests/parity/test_factor_analysis.py::test_apply_factor_costs_and_capacity",
        },
        "factor.analysis.event_analysis_model": {
            "tests/parity/test_factor_analysis.py::test_factor_analysis_models",
        },
        "factor.analysis.factor_capacity_result": {
            "tests/parity/test_factor_analysis.py::test_apply_factor_costs_and_capacity",
        },
        "factor.analysis.factor_cost_model": {
            "tests/parity/test_factor_analysis.py::test_apply_factor_costs_and_capacity",
        },
        "factor.analysis.factor_cost_result": {
            "tests/parity/test_factor_analysis.py::test_apply_factor_costs_and_capacity",
        },
        "factor.analysis.factor_function_spec": {
            "tests/parity/test_factor_analysis.py::test_factor_spec_disposition",
        },
        "factor.analysis.factor_group_analysis": {
            "tests/parity/test_factor_analysis.py::test_factor_analysis_models",
        },
        "factor.analysis.factor_tear_sheet_artifacts": {
            "tests/parity/test_factor_analysis.py::test_factor_workflow_artifacts",
        },
        "factor.analysis.factor_workflow_spec": {
            "tests/parity/test_factor_analysis.py::test_factor_spec_disposition",
        },
        "factor.analysis.fama_macbeth": {
            "tests/parity/test_factor_analysis.py::test_fama_macbeth",
        },
        "factor.analysis.ic_inference_result": {
            "tests/parity/test_factor_analysis.py::test_factor_inference",
        },
        "factor.analysis.pyfolio_factor_inputs": {
            "tests/parity/test_factor_analysis.py::test_pyfolio_factor_inputs",
        },
        "factor.fama_macbeth": {
            "tests/parity/test_factor_analysis.py::test_fama_macbeth",
        },
        "factor.inference": {
            "tests/parity/test_factor_analysis.py::test_factor_inference",
        },
        "factor.pit_prepare": {
            "tests/parity/test_factor_analysis.py::test_pit_preparation",
        },
        "factor.prepare.enhanced_non_matching_timezone_error": {
            "tests/parity/test_factor_analysis.py::test_preparation_error_categories",
        },
        "factor.prepare.factor_analysis_config": {
            "tests/parity/test_factor_analysis.py::test_factor_analysis_models",
        },
        "factor.prepare.factor_data_error": {
            "tests/parity/test_factor_analysis.py::test_preparation_error_categories",
        },
        "factor.prepare.factor_loss_exceeded_error": {
            "tests/parity/test_factor_analysis.py::test_preparation_error_categories",
        },
        "factor.prepare.factor_loss_report": {
            "tests/parity/test_factor_analysis.py::test_preparation_error_categories",
        },
        "factor.prepare.materialize_pit_factor": {
            "tests/parity/test_factor_analysis.py::test_pit_preparation",
        },
        "factor.prepare.max_loss_exceeded_error": {
            "tests/parity/test_factor_analysis.py::test_preparation_error_categories",
        },
        "factor.prepare.multi_horizon_prepared_factor_data": {
            "tests/parity/test_factor_analysis.py::test_prepare_factor_data_by_horizon",
        },
        "factor.prepare.non_matching_timezone_error": {
            "tests/parity/test_factor_analysis.py::test_preparation_error_categories",
        },
        "factor.prepare.prepare_factor_data_by_horizon": {
            "tests/parity/test_factor_analysis.py::test_prepare_factor_data_by_horizon",
        },
        "factor.prepare.prepare_factor_data_from_forward_returns": {
            "tests/parity/test_factor_analysis.py::test_prepare_factor_data_from_forward_returns",
        },
        "factor.prepare.validate_pit_alignment": {
            "tests/parity/test_factor_analysis.py::test_pit_preparation",
        },
        "factor.prepare_by_horizon": {
            "tests/parity/test_factor_analysis.py::test_prepare_factor_data_by_horizon",
        },
    }

    gap_ids = {gap["capability_id"] for gap in ledger["coverage_gaps"]}
    entries = {entry["capability_id"]: entry for entry in ledger["entries"]}

    assert not expected.keys() & gap_ids
    for capability_id, wheel_nodeids in expected.items():
        entry = entries[capability_id]
        assert entry["owner"] == "factor"
        assert entry["disposition"] == "required"
        assert entry["source_nodeids"]
        assert set(entry["wheel_nodeids"]) == wheel_nodeids


def test_reporting_portfolio_and_attribution_migration_scenarios_are_bound() -> None:
    """Display behaviors retain real source scenarios while APIs are redesigned."""
    ledger = _load()
    expected = {
        "attribution": {
            "attribution.brinson",
            "attribution.ff_factor_provider",
            "attribution.performance.plot_factor_contribution_to_perf",
            "attribution.performance.show_profit_attribution",
            "attribution.style_factor_provider",
        },
        "portfolio": {
            "portfolio.perf_stats.plot_perf_stats",
            "portfolio.perf_stats.show_worst_drawdown_periods",
            "portfolio.positions.show_and_plot_top_positions",
            "portfolio.round_trips.print_round_trip_stats",
        },
        "report": {
            "report.portfolio.plot_alpha_returns",
            "report.portfolio.plot_annual_returns",
            "report.portfolio.plot_drawdown_periods",
            "report.portfolio.plot_drawdown_underwater",
            "report.portfolio.plot_exposures",
            "report.portfolio.plot_gross_leverage",
            "report.portfolio.plot_holdings",
            "report.portfolio.plot_long_short_holdings",
            "report.portfolio.plot_max_median_position_concentration",
            "report.portfolio.plot_monthly_returns_dist",
            "report.portfolio.plot_monthly_returns_heatmap",
            "report.portfolio.plot_monthly_returns_timeseries",
            "report.portfolio.plot_perf_attrib_returns",
            "report.portfolio.plot_return_quantiles",
            "report.portfolio.plot_risk_exposures",
            "report.portfolio.plot_rolling_beta",
            "report.portfolio.plot_rolling_returns",
            "report.portfolio.plot_rolling_volatility",
            "report.portfolio.plot_sector_allocations",
            "report.portfolio.strategy_report",
        },
        "viz": {"viz.resource.close_owned_figures"},
    }

    gap_ids = {gap["capability_id"] for gap in ledger["coverage_gaps"]}
    entries = {entry["capability_id"]: entry for entry in ledger["entries"]}

    assert "viz.resource.show_owned_figures" in gap_ids
    for owner, capability_ids in expected.items():
        assert not capability_ids & gap_ids
        for capability_id in capability_ids:
            entry = entries[capability_id]
            assert entry["owner"] == owner
            assert entry["disposition"] == "required"
            assert entry["source_nodeids"]
            assert entry["wheel_nodeids"]
