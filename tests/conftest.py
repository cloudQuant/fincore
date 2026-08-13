"""Shared pytest configuration and fixtures.

This file contains:
- Priority markers configuration (p0, p1, p2, p3)
- Shared fixtures for test data
- Custom pytest hooks

Priority Levels:
- P0: Critical - core metrics (sharpe_ratio, max_drawdown, etc.), security, compliance
- P1: High - frequently used features, important edge cases
- P2: Medium - secondary features, admin functions, edge cases
- P3: Low - rarely used, cosmetic, deprecation tests
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_UPSTREAM_RESULT_PHASES = ("setup", "call", "teardown")
_UPSTREAM_RERUN_MARKERS = ("flaky", "rerun", "reruns", "rerunfailures")
_UPSTREAM_RESULTS_SCHEMA = "alphalens-upstream-case-results-v2"

# ==============================================================================
# Priority Markers - Apply to test classes and methods
# ==============================================================================
# See pyproject.toml [tool.pytest.ini_options].markers for marker definitions
#
# Usage examples:
#
# @pytest.mark.p0  # Critical: core financial metric
# def test_sharpe_ratio():
#     ...
#
# @pytest.mark.p1  # High: important edge case
# def test_sharpe_ratio_with_nan():
#     ...
#
# @pytest.mark.p2  # Medium: nice-to-have validation
# def test_sharpe_ratio_boundary_conditions():
#     ...
#
# Run selective tests:
#   pytest -m p0                    # Only critical tests
#   pytest -m "p0 or p1"            # Critical + high priority
#   pytest -m "not slow"            # Skip slow tests
# ==============================================================================


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the non-xdist proof file used by future Alphalens migrations."""
    parser.addoption(
        "--alphalens-upstream-result-json",
        action="store",
        default=None,
        metavar="PATH",
        help="write marked Alphalens upstream-case outcomes to a JSON file under build/ (non-xdist only)",
    )


def _is_xdist_run(config: pytest.Config) -> bool:
    """Return whether this process is part of an xdist run."""
    return hasattr(config, "workerinput") or bool(getattr(config.option, "numprocesses", 0))


def _result_path(config: pytest.Config) -> Path | None:
    """Resolve and constrain the optional marker proof output path to build/."""
    configured = config.getoption("alphalens_upstream_result_json")
    if configured is None:
        return None
    if _is_xdist_run(config):
        raise pytest.UsageError("--alphalens-upstream-result-json is supported only in a non-xdist pytest run")
    root = Path(str(config.rootpath)).resolve()
    build_root = (root / "build").resolve()
    configured_path = Path(configured)
    path = (configured_path if configured_path.is_absolute() else root / configured_path).resolve()
    try:
        path.relative_to(build_root)
    except ValueError as exc:
        raise pytest.UsageError("--alphalens-upstream-result-json must point under build/") from exc
    return path


def _global_reruns_requested(config: pytest.Config) -> bool:
    """Return whether the rerun plugin was asked to retry tests in this run."""
    reruns = getattr(config.option, "reruns", None)
    if reruns is None:
        return False
    try:
        return int(reruns) > 0
    except (TypeError, ValueError):
        return bool(reruns)


def _new_upstream_attempt() -> dict[str, object]:
    """Create one append-only per-attempt phase record for marker proof output."""
    return {"outcomes": dict.fromkeys(_UPSTREAM_RESULT_PHASES, "not-run")}


def _record_upstream_phase(record: dict[str, object], phase: str, outcome: str) -> None:
    """Append an attempt phase without allowing later reruns to erase prior truth."""
    attempts = record["attempts"]
    assert isinstance(attempts, list)
    if not attempts:
        attempts.append(_new_upstream_attempt())
    current = attempts[-1]
    assert isinstance(current, dict)
    current_outcomes = current["outcomes"]
    assert isinstance(current_outcomes, dict)
    if current_outcomes.get(phase) != "not-run":
        current = _new_upstream_attempt()
        attempts.append(current)
        current_outcomes = current["outcomes"]
        assert isinstance(current_outcomes, dict)
    current_outcomes[phase] = outcome

    aggregate = record["outcomes"]
    assert isinstance(aggregate, dict)
    prior = aggregate.get(phase)
    if prior == "not-run" or (prior == "passed" and outcome != "passed"):
        aggregate[phase] = outcome


def _has_nonpassing_upstream_attempt(results: dict[str, dict[str, object]]) -> bool:
    """Return whether a marked item ever emitted an actual non-passing phase."""
    for record in results.values():
        attempts = record.get("attempts")
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, dict):
                return True
            outcomes = attempt.get("outcomes")
            if not isinstance(outcomes, dict):
                return True
            if any(outcomes.get(phase) != "passed" for phase in _UPSTREAM_RESULT_PHASES):
                return True
    return False


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers.

    This hook ensures markers are registered even if not in pyproject.toml.
    """
    config.addinivalue_line("markers", "p0: Critical priority tests (core metrics, security)")
    config.addinivalue_line("markers", "p1: High priority tests (frequently used)")
    config.addinivalue_line("markers", "p2: Medium priority tests (secondary features)")
    config.addinivalue_line("markers", "p3: Low priority tests (rarely used)")
    config.addinivalue_line(
        "markers",
        "alphalens_upstream_case(case_id): pinned Alphalens upstream case or invocation migration proof",
    )
    config._alphalens_upstream_result_path = _result_path(config)  # type: ignore[attr-defined]
    config._alphalens_upstream_results = {}  # type: ignore[attr-defined]


# ==============================================================================
# P0: Critical Metrics - These are the core financial metrics
# ==============================================================================
P0_METRICS = [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "annual_return",
    "volatility",
    "alpha",
    "beta",
    "cum_returns",
    "cum_returns_final",
    "value_at_risk",
    "conditional_value_at_risk",
]

P0_FEATURES = [
    "returns_calculation",
    "drawdown_analysis",
    "risk_adjusted_returns",
]


# ==============================================================================
# Test collection hooks for automatic priority assignment
# ==============================================================================


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Automatically assign P0 markers to critical metric tests.

    This hook looks for test functions that test P0 metrics and marks them
    automatically if not already marked.
    """
    marker_errors: list[str] = []
    reruns_requested = _global_reruns_requested(config)
    for item in items:
        markers = list(item.iter_markers(name="alphalens_upstream_case"))
        if not markers:
            continue
        if len(markers) != 1 or len(markers[0].args) != 1 or not isinstance(markers[0].args[0], str):
            marker_errors.append(f"{item.nodeid}: alphalens_upstream_case requires exactly one string case ID")
            continue
        if reruns_requested:
            marker_errors.append(f"{item.nodeid}: upstream case cannot enable reruns via --reruns")
            continue
        blocked = [name for name in ("skip", "skipif", "xfail") if item.get_closest_marker(name) is not None]
        if blocked:
            marker_errors.append(f"{item.nodeid}: upstream case cannot carry {', '.join(blocked)}")
            continue
        rerun_markers = [
            name for name in _UPSTREAM_RERUN_MARKERS if item.get_closest_marker(name) is not None
        ]
        if rerun_markers:
            marker_errors.append(f"{item.nodeid}: upstream case cannot carry {', '.join(rerun_markers)}")
            continue
        results: dict[str, dict[str, object]] = config._alphalens_upstream_results  # type: ignore[attr-defined]
        results[item.nodeid] = {
            "nodeid": item.nodeid,
            "case_id": markers[0].args[0],
            "outcomes": dict.fromkeys(_UPSTREAM_RESULT_PHASES, "not-run"),
            "attempts": [],
        }
    if marker_errors:
        raise pytest.UsageError("invalid alphalens upstream-case markers:\n  " + "\n  ".join(marker_errors))

    for item in items:
        # Skip if already has a priority marker
        if any(marker in item.keywords for marker in ["p0", "p1", "p2", "p3"]):
            continue

        # Check if test name contains a P0 metric
        test_name = item.name.lower()
        for metric in P0_METRICS:
            if metric in test_name:
                item.add_marker(pytest.mark.p0)
                break


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[object]):
    """Turn every non-passing marked phase into a failure and record its truth."""
    outcome = yield
    report = outcome.get_result()
    marker = item.get_closest_marker("alphalens_upstream_case")
    if marker is None:
        return
    results: dict[str, dict[str, object]] = item.config._alphalens_upstream_results  # type: ignore[attr-defined]
    record = results.get(item.nodeid)
    if record is None:
        return
    actual_outcome = "xfailed" if getattr(report, "wasxfail", None) else report.outcome
    _record_upstream_phase(record, report.when, actual_outcome)
    if actual_outcome == "passed":
        return
    report.outcome = "failed"
    report.longrepr = (
        f"Alphalens upstream case {record['case_id']!r} did not pass during {report.when}: {actual_outcome}"
    )
    if hasattr(report, "wasxfail"):
        delattr(report, "wasxfail")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write the non-xdist marked-case proof consumed only by the migration checker."""
    config = session.config
    results: dict[str, dict[str, object]] = config._alphalens_upstream_results  # type: ignore[attr-defined]
    final_exitstatus = exitstatus
    if _has_nonpassing_upstream_attempt(results) and exitstatus == pytest.ExitCode.OK:
        final_exitstatus = pytest.ExitCode.TESTS_FAILED
        session.exitstatus = final_exitstatus
    path: Path | None = config._alphalens_upstream_result_path  # type: ignore[attr-defined]
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": _UPSTREAM_RESULTS_SCHEMA,
        "xdist": False,
        "pytest_exitstatus": final_exitstatus,
        "results": [results[nodeid] for nodeid in sorted(results)],
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
