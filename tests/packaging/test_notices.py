"""Third-party notice inventory tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_notices import check_notices, load_notices


def test_copied_or_adapted_component_has_notice_and_license_status() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")

    assert notices["alphalens"]["review_status"] in {"pending-human-review", "approved"}
    assert notices["alphalens"]["source_commit"]


def test_empyrical_notice_records_pinned_commit() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")

    assert notices["empyrical"]["source_commit"] == "74655e974ed2935563820c548c339731f1fe0621"


def test_all_adapted_components_have_pinned_commits() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")

    for name in ("empyrical", "pyfolio", "alphalens"):
        record = notices[name]
        assert len(record["source_commit"]) == 40
        assert record["adapted"] is True


def test_notice_checker_passes_the_checked_in_inventory() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")

    assert check_notices(notices) == []


def test_notice_checker_rejects_missing_commit() -> None:
    notices = load_notices(ROOT / "THIRD_PARTY_NOTICES.md")
    notices["empyrical"]["source_commit"] = "not-a-commit"

    violations = check_notices(notices)

    assert any("source_commit" in v for v in violations)
