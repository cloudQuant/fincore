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
    for item in items:
        markers = list(item.iter_markers(name="alphalens_upstream_case"))
        if not markers:
            continue
        if len(markers) != 1 or len(markers[0].args) != 1 or not isinstance(markers[0].args[0], str):
            marker_errors.append(f"{item.nodeid}: alphalens_upstream_case requires exactly one string case ID")
            continue
        blocked = [name for name in ("skip", "skipif", "xfail") if item.get_closest_marker(name) is not None]
        if blocked:
            marker_errors.append(f"{item.nodeid}: upstream case cannot carry {', '.join(blocked)}")
            continue
        results: dict[str, dict[str, object]] = config._alphalens_upstream_results  # type: ignore[attr-defined]
        results[item.nodeid] = {
            "nodeid": item.nodeid,
            "case_id": markers[0].args[0],
            "outcomes": {"setup": "not-run", "call": "not-run", "teardown": "not-run"},
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
    outcomes = record["outcomes"]
    assert isinstance(outcomes, dict)
    actual_outcome = "xfailed" if getattr(report, "wasxfail", None) else report.outcome
    outcomes[report.when] = actual_outcome
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
    path: Path | None = config._alphalens_upstream_result_path  # type: ignore[attr-defined]
    if path is None:
        return
    results: dict[str, dict[str, object]] = config._alphalens_upstream_results  # type: ignore[attr-defined]
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": "alphalens-upstream-case-results-v1",
        "xdist": False,
        "pytest_exitstatus": exitstatus,
        "results": [results[nodeid] for nodeid in sorted(results)],
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
