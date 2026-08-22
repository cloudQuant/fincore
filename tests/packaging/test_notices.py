"""Third-party notice inventory tests."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = (
    ROOT / ".github" / "workflows" / "ci.yml",
    ROOT / ".github" / "workflows" / "publish.yml",
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_notices import check_notices, load_notices


def _fincore_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as file:
        return str(tomllib.load(file)["project"]["version"])


def test_fincore_project_identity_uses_one_project_license_and_version() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")

    assert notices["schema_version"] == 2
    assert notices["project"] == {
        "name": "fincore",
        "version": _fincore_version(),
        "license": "MIT",
    }


def test_copied_or_adapted_component_has_notice_and_license_status() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")

    assert notices["alphalens"]["review_status"] in {"pending-human-review", "approved"}
    assert notices["alphalens"]["source_commit"]
    assert notices["pyfolio"]["license_header"] == "mixed"
    assert notices["alphalens"]["license_header"] == "mixed"


def test_empyrical_notice_records_pinned_commit() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")

    assert notices["empyrical"]["source_commit"] == "74655e974ed2935563820c548c339731f1fe0621"


def test_all_adapted_components_have_pinned_commits() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")

    for name in ("empyrical", "pyfolio", "alphalens"):
        record = notices[name]
        assert len(record["source_commit"]) == 40
        assert record["adapted"] is True
        assert "upstream_version" in record


def test_vendored_echarts_is_recorded_with_its_artifact_digest() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")

    record = notices["echarts"]
    assert record["adapted"] is False
    assert record["license"] == "Apache-2.0"
    assert len(record["source_sha256"]) == 64
    assert record["embedded_attributions"] == ["Copyright (c) Microsoft Corporation."]


def test_notice_checker_passes_the_checked_in_inventory() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")

    assert check_notices(notices) == []


def test_strict_notice_policy_audit_remains_available_but_is_not_the_default() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")

    violations = check_notices(notices, require_approved=True)

    assert {violation.split(":", 1)[0] for violation in violations} == {
        "alphalens",
        "echarts",
        "empyrical",
        "pyfolio",
    }
    assert all("review_status must be approved" in violation for violation in violations)


def test_ci_and_publish_validate_notice_integrity_without_claiming_human_approval() -> None:
    for workflow_path in WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")

        assert "run: python scripts/check_notices.py" in workflow
        assert "python scripts/check_notices.py --require-approved" not in workflow


def test_notice_checker_rejects_missing_commit() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")
    notices["empyrical"]["source_commit"] = "not-a-commit"

    violations = check_notices(notices)

    assert any("source_commit" in v for v in violations)


def test_notice_checker_rejects_project_identity_drift() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")
    notices["project"]["version"] = "0.0.0"

    violations = check_notices(notices)

    assert "project.version must equal pyproject.toml" in violations


def test_notice_checker_rejects_vendored_asset_digest_drift() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")
    notices["echarts"]["source_sha256"] = "0" * 64

    violations = check_notices(notices)

    assert any("vendored asset digest" in violation for violation in violations)


def test_notice_checker_rejects_missing_vendored_embedded_attribution() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")
    del notices["echarts"]["embedded_attributions"]

    violations = check_notices(notices)

    assert "echarts: embedded_attributions must be a list of non-empty strings" in violations
