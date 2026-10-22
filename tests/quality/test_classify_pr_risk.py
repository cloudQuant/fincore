"""Contract tests for the PR risk classifier (iteration 0041 / M3).

The classifier maps changed file paths to the R0-R3 levels defined in
``docs/governance/branch-governance.md``. Paths give a *floor*: multi-path
diffs take the highest level, maintainers may upgrade with a structured
reason and must never downgrade.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_PATH = REPO_ROOT / "scripts" / "ci" / "classify_pr_risk.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("classify_pr_risk", CLASSIFIER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {CLASSIFIER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("paths", "expected_level"),
    [
        (["README.md"], "R0"),
        (["docs/guide.md", "mkdocs_docs/index.md"], "R0"),
        (["examples/demo.py", ".github/ISSUE_TEMPLATE/config.yml"], "R0"),
        (["tests/test_core/test_engine.py"], "R1"),
        (["tests/test_empyrical/test_sharpe.py", "fincore/utils/helpers.py"], "R1"),
        pytest.param(
            ["fincore/metrics/ratios.py"],
            "R2",
            id="numerical-metrics-change-is-r2",
        ),
        pytest.param(["fincore/core/engine.py"], "R2", id="rolling-engine-is-r2"),
        pytest.param(["fincore/empyrical.py"], "R2", id="strict-compat-facade-is-r2"),
        pytest.param(["fincore/pyfolio.py"], "R2", id="pyfolio-facade-is-r2"),
        pytest.param(["fincore/contracts/api.py"], "R2", id="contracts-are-r2"),
        pytest.param(["fincore/_registry.py"], "R2", id="registry-is-r2"),
        pytest.param(["fincore/_dispatch.py"], "R2", id="dispatch-is-r2"),
        pytest.param(
            ["tests/compat/fixtures/public-api-0.4.0.dev0.json"],
            "R2",
            id="frozen-compat-manifest-is-r2",
        ),
        pytest.param(["fincore/alphalens/model.py"], "R2", id="alphalens-is-r2"),
        pytest.param(["fincore/factor_analysis/prepare.py"], "R2", id="factor-analysis-is-r2"),
        pytest.param([".github/workflows/ci.yml"], "R3", id="workflow-is-r3"),
        pytest.param([".github/CODEOWNERS"], "R3", id="codeowners-is-r3"),
        pytest.param([".github/governance/rulesets/dev.json"], "R3", id="ruleset-manifest-is-r3"),
        pytest.param(["scripts/ci/verify_github_governance.py"], "R3", id="ci-verifier-is-r3"),
        pytest.param(["pyproject.toml"], "R3", id="packaging-metadata-is-r3"),
        pytest.param(["setup.py"], "R3", id="setup-py-is-r3"),
        pytest.param(["requirements-test.txt"], "R3", id="requirements-are-r3"),
        pytest.param(["constraints/constraints.txt"], "R3", id="constraints-are-r3"),
        pytest.param(["NOTICE"], "R3", id="notice-is-r3"),
        pytest.param(["THIRD_PARTY_NOTICES.md"], "R3", id="third-party-notices-are-r3"),
        pytest.param(["SECURITY.md"], "R3", id="security-policy-is-r3"),
        pytest.param(["GOVERNANCE.md"], "R3", id="governance-policy-is-r3"),
    ],
)
def test_single_path_classification(paths: list[str], expected_level: str) -> None:
    mod = _load_module()
    result = mod.classify(paths)
    assert result["risk_level"] == expected_level
    assert result["upgraded"] is False


def test_multi_path_takes_highest_level() -> None:
    mod = _load_module()
    result = mod.classify(["docs/x.md", "fincore/core/context.py", "pyproject.toml"])
    assert result["risk_level"] == "R3"
    r3_hits = [m for m in result["matched_rules"] if m["level"] == "R3"]
    assert r3_hits


def test_upgrade_allowed_with_reason() -> None:
    mod = _load_module()
    result = mod.classify(["docs/guide.md"], upgrade_to="R2", reason="doc changes published API contract wording")
    assert result["risk_level"] == "R2"
    assert result["upgraded"] is True
    assert result["reason"]


def test_downgrade_is_rejected() -> None:
    mod = _load_module()
    result = mod.classify(["fincore/core/engine.py"], upgrade_to="R0", reason="small diff")
    assert result["risk_level"] == "R2"
    assert result["upgraded"] is False
    assert "downgrade" in result["warning"].lower()


def test_invalid_upgrade_target_rejected() -> None:
    mod = _load_module()
    result = mod.classify(["README.md"], upgrade_to="R9", reason="typo")
    assert result["risk_level"] == "R0"
    assert result["upgraded"] is False


def test_unknown_path_defaults_to_r1() -> None:
    mod = _load_module()
    result = mod.classify(["some/new/dir/file.xyz"])
    assert result["risk_level"] == "R1"


def test_cli_reports_json_and_exit_zero(tmp_path: Path) -> None:
    mod = _load_module()
    out_file = tmp_path / "classification.json"
    rc = mod.main(
        [
            "--paths",
            "fincore/metrics/sharpe.py",
            "docs/note.md",
            "--output",
            str(out_file),
        ]
    )
    assert rc == 0
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["risk_level"] == "R2"


def test_cli_accepts_paths_file(tmp_path: Path) -> None:
    mod = _load_module()
    list_file = tmp_path / "changed.txt"
    list_file.write_text("fincore/metrics/risk.py\n\nREADME.md\n", encoding="utf-8")
    rc = mod.main(["--paths-file", str(list_file)])
    assert rc == 0
