"""Community governance file contract tests (iteration 0041 / M3).

Guards the contribution entry points against drift: Issue Forms stay
well-formed, the PR template cites the current API snapshot, security and
maintainer channels cross-link, CODEOWNERS only names real owners, and no
maintained doc or workflow routes work to the nonexistent ``main`` branch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GITHUB_DIR = REPO_ROOT / ".github"
ISSUE_TEMPLATE_DIR = GITHUB_DIR / "ISSUE_TEMPLATE"

MAINTENANCE_DOC_SCOPE = [
    "CONTRIBUTING.md",
    "README.md",
    "GOVERNANCE.md",
    "SECURITY.md",
    "SUPPORT.md",
    "MAINTAINERS.md",
    ".github/pull_request_template.md",
]
WORKFLOW_SCOPE = ["ci.yml", "test-priority.yml", "pr-governance.yml"]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Issue Forms
# ---------------------------------------------------------------------------


def test_issue_forms_exist_and_parse() -> None:
    for name in ("bug_report.yml", "feature_request.yml", "config.yml"):
        path = ISSUE_TEMPLATE_DIR / name
        assert path.is_file(), f"missing {path}"
        data = yaml.safe_load(_read(path))
        assert isinstance(data, dict), f"{name} must be a mapping"


def test_bug_form_requires_environment_repro_and_expected_actual() -> None:
    form = yaml.safe_load(_read(ISSUE_TEMPLATE_DIR / "bug_report.yml"))
    body = form["body"]
    by_id = {item.get("id"): item for item in body if isinstance(item, dict)}
    for required_id in ("environment", "api_surface", "repro", "expected_actual"):
        field = by_id[required_id]
        assert field.get("validations", {}).get("required") is True, required_id


def test_feature_form_requires_problem_and_surface() -> None:
    form = yaml.safe_load(_read(ISSUE_TEMPLATE_DIR / "feature_request.yml"))
    body = form["body"]
    by_id = {item.get("id"): item for item in body if isinstance(item, dict)}
    for required_id in ("problem", "surface_impact"):
        field = by_id[required_id]
        assert field.get("validations", {}).get("required") is True, required_id


def test_config_routes_questions_to_discussions_and_security_privately() -> None:
    config = yaml.safe_load(_read(ISSUE_TEMPLATE_DIR / "config.yml"))
    assert config["blank_issues_enabled"] is False
    contacts = {link["name"]: link for link in config["contact_links"]}
    assert any("Discussions" in name for name in contacts), contacts
    security_links = [link for name, link in contacts.items() if "security" in name.lower() or "安全" in name]
    assert security_links, "security contact link is mandatory"
    assert "/security/advisories/new" in security_links[0]["url"]


# ---------------------------------------------------------------------------
# Pull request template
# ---------------------------------------------------------------------------


def test_pr_template_has_governance_sections() -> None:
    template = _read(GITHUB_DIR / "pull_request_template.md")
    for section in (
        "## Target branch",
        "## Risk level",
        "## Compatibility surface impact",
        "## Verification",
        "## Related issue / ADR",
    ):
        assert section in template, section


def test_pr_template_cites_current_api_snapshot_only() -> None:
    template = _read(GITHUB_DIR / "pull_request_template.md")
    assert "public-api-0.4.0.dev0.json" in template
    assert "public-api-0.3.x" not in template


# ---------------------------------------------------------------------------
# Cross-references between community documents
# ---------------------------------------------------------------------------


def test_security_doc_points_to_private_channel_and_maintainers() -> None:
    security = _read(REPO_ROOT / "SECURITY.md")
    assert "security/advisories/new" in security
    assert "MAINTAINERS.md" in security
    assert "Pending enablement confirmation" in security


def test_support_doc_links_all_channels() -> None:
    support = _read(REPO_ROOT / "SUPPORT.md")
    assert "discussions" in support.lower()
    assert "SECURITY.md" in support
    assert "CONTRIBUTING.md" in support
    assert "branch-governance.md" in support


def test_maintainers_doc_declares_real_owner_and_known_gaps() -> None:
    maintainers = _read(REPO_ROOT / "MAINTAINERS.md")
    assert "@cloudQuant" in maintainers
    assert "第二维护者缺口" in maintainers
    assert "governance-exception" in maintainers


def test_contributing_targets_dev_and_references_governance() -> None:
    contributing = _read(REPO_ROOT / "CONTRIBUTING.md")
    assert "`dev` branch" in contributing
    assert "branch-governance.md" in contributing
    assert "MAINTAINERS.md" in contributing


# ---------------------------------------------------------------------------
# Branch routing hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("doc", MAINTENANCE_DOC_SCOPE)
def test_no_main_branch_routing_in_maintained_docs(doc: str) -> None:
    path = REPO_ROOT / doc
    if not path.is_file():
        pytest.skip(f"{doc} not present")
    text = _read(path).lower()
    assert "against the `main`" not in text
    assert "target: main" not in text
    assert "public-api-0.3.x" not in text


@pytest.mark.parametrize("workflow", WORKFLOW_SCOPE)
def test_workflows_pull_request_covers_master_and_dev_only(workflow: str) -> None:
    path = GITHUB_DIR / "workflows" / workflow
    data = yaml.safe_load(_read(path))
    triggers = data.get(True) or data.get("on") or {}
    assert "pull_request" in triggers, workflow
    assert set(triggers["pull_request"]["branches"]) == {"master", "dev"}, workflow


@pytest.mark.parametrize("workflow", ["ci.yml", "test-priority.yml"])
@pytest.mark.parametrize("trigger_key", ["push", "pull_request"])
def test_push_and_pr_workflows_trigger_on_master_and_dev(workflow: str, trigger_key: str) -> None:
    path = GITHUB_DIR / "workflows" / workflow
    data = yaml.safe_load(_read(path))
    triggers = data.get(True) or data.get("on") or {}
    assert trigger_key in triggers, workflow
    assert set(triggers[trigger_key]["branches"]) == {"master", "dev"}, workflow


# ---------------------------------------------------------------------------
# CODEOWNERS
# ---------------------------------------------------------------------------

REQUIRED_CODEOWNERS_PATHS = (
    "/.github/",
    "/scripts/",
    "/docs/",
    "/fincore/empyrical.py",
    "/fincore/pyfolio.py",
    "/fincore/_dispatch.py",
    "/fincore/_registry.py",
    "/fincore/core/",
    "/fincore/contracts/",
    "/fincore/metrics/",
    "/fincore/alphalens/",
    "/fincore/factor_analysis/",
    "/tests/compat/",
    "/pyproject.toml",
    "/setup.py",
    "/requirements-test.txt",
)


def test_codeowners_covers_required_paths_with_real_owner() -> None:
    lines = _read(GITHUB_DIR / "CODEOWNERS").splitlines()
    rules = []
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        rules.append((parts[0], parts[1:]))

    owned = {pattern for pattern, _owners in rules}
    for required in REQUIRED_CODEOWNERS_PATHS:
        assert required in owned, f"CODEOWNERS misses {required}"

    for pattern, owners in rules:
        assert owners, f"rule without owner: {pattern}"
        for owner in owners:
            assert owner.startswith("@"), f"invalid owner token {owner!r}"
            assert owner == "@cloudQuant", f"owner {owner!r} is not a confirmed maintainer (decision D2)"
