"""Optional-extras union contract tests.

``all`` must be the exact normalized union of the functional extras
(pyfolio / factor-analysis / alphalens / interactive / report-pdf / report-xlsx / bayesian / data-*) —
never a self-reference such as ``fincore[...]`` and never dev-only tooling.
The compatibility aliases (``viz``, ``datareader``) declare exactly which
functional extras they cover, so a hand-edited alias list cannot drift past
this test unnoticed.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CONTRIBUTOR_REQUIREMENT_FILES = (REPO_ROOT / "requirements.txt", REPO_ROOT / "requirements-test.txt")
PROHIBITED_EXTERNAL_REQUIREMENTS = {"alphalens", "empyrical"}

# Functional extras: every extra that installs runtime capability.
FUNCTIONAL_EXTRAS = {
    "pyfolio",
    "factor-analysis",
    "alphalens",
    "interactive",
    "report-pdf",
    "report-xlsx",
    "bayesian",
    "data-yahoo",
    "data-alphavantage",
    "data-pandas-datareader",
    "data-cn",
}

# 0.3.x compatibility aliases (kept for at least one documented minor cycle).
ALIAS_EXTRAS = {"viz", "datareader"}

# Dev-only tooling must never leak into the ``all`` union.
DEV_ONLY_TOOLS = {
    "pytest",
    "pytest-xdist",
    "pytest-cov",
    "pytest-benchmark",
    "pytest-sugar",
    "parameterized",
    "ruff",
    "mypy",
    "types-requests",
    "pre-commit",
    "bandit",
}


def _extras() -> dict[str, list[str]]:
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["optional-dependencies"]


def _normalized(reqs: list[str]) -> set[str]:
    """Canonicalize requirement strings (PEP 503 name + SpecifierSet)."""
    return {str(Requirement(req)) for req in reqs}


def _functional_union(extras: dict[str, list[str]]) -> set[str]:
    union: set[str] = set()
    for name in FUNCTIONAL_EXTRAS:
        assert name in extras, f"functional extra {name!r} missing from pyproject.toml"
        union |= _normalized(extras[name])
    return union


def _supported_source_requirements() -> list[tuple[str, str]]:
    """Return every requirement used by contributor or PEP 517/621 metadata."""
    with PYPROJECT.open("rb") as fh:
        metadata = tomllib.load(fh)
    requirements: list[tuple[str, str]] = [
        ("pyproject build-system", raw) for raw in metadata["build-system"]["requires"]
    ]
    project = metadata["project"]
    requirements.extend(("pyproject dependencies", raw) for raw in project.get("dependencies", []))
    requirements.extend(
        (f"pyproject extra {name}", raw)
        for name, values in project.get("optional-dependencies", {}).items()
        for raw in values
    )
    for path in CONTRIBUTOR_REQUIREMENT_FILES:
        requirements.extend(
            (path.name, line.strip())
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return requirements


def _assert_supported_requirement_is_integrated(source: str, raw: str) -> None:
    """Reject external compatibility distributions in a supported input."""
    assert "://" not in raw and not raw.lower().startswith("git+"), f"{source} installs from URL: {raw!r}"
    requirement = Requirement(raw)
    assert canonicalize_name(requirement.name) not in PROHIBITED_EXTERNAL_REQUIREMENTS, (
        f"{source} installs an external compatibility package: {raw!r}"
    )
    assert requirement.url is None, f"{source} installs from URL: {raw!r}"


def test_all_is_exact_normalized_union_of_functional_extras() -> None:
    """``all`` must equal the union of the functional extras, requirement-for-requirement."""
    extras = _extras()
    assert _normalized(extras["all"]) == _functional_union(extras)


def test_no_self_reference_in_any_extra() -> None:
    """PEP 621 extras must not reference ``fincore[...]`` (no self-dependency)."""
    for extra_name, reqs in _extras().items():
        for req in reqs:
            assert not req.strip().lower().startswith("fincore"), (
                f"extra {extra_name!r} contains self-reference: {req!r}"
            )


def test_supported_dependency_inputs_do_not_install_external_compatibility_packages_or_urls() -> None:
    """Contributor and distribution metadata never re-install integrated code."""
    for source, raw in _supported_source_requirements():
        _assert_supported_requirement_is_integrated(source, raw)


@pytest.mark.parametrize("raw", ("Empyrical>=1", "AlphaLens>=1"))
def test_source_requirement_guard_rejects_mixed_case_external_names(raw: str) -> None:
    """PEP 503 name variants cannot bypass the contributor/metadata guard."""
    with pytest.raises(AssertionError, match="external compatibility package"):
        _assert_supported_requirement_is_integrated("mixed-case fixture", raw)


def test_all_excludes_dev_only_tools() -> None:
    """Dev-only tooling belongs to ``dev``, never to the runtime ``all`` union."""
    names_in_all = {Requirement(req).name for req in _extras()["all"]}
    assert not (names_in_all & DEV_ONLY_TOOLS), f"dev-only tools leaked into all: {names_in_all & DEV_ONLY_TOOLS}"


def test_alphalens_extras_cover_the_declared_runtime_boundaries() -> None:
    """The recovery commands emitted by strict adapters name installable extras."""

    extras = _extras()
    assert _normalized(extras["factor-analysis"]) == {"statsmodels>=0.14"}
    assert _normalized(extras["alphalens"]) == {
        "statsmodels>=0.14",
        "matplotlib>=3.3",
        "seaborn>=0.11",
        "ipython>=7",
    }
    assert _normalized(extras["alphalens"]) <= _normalized(extras["dev"])


def test_datareader_alias_maps_exactly_to_data_pandas_datareader() -> None:
    """The 0.3.x ``datareader`` alias must stay identical to the renamed extra."""
    extras = _extras()
    assert _normalized(extras["datareader"]) == _normalized(extras["data-pandas-datareader"])


def test_viz_alias_covers_pyfolio_plus_interactive_plus_pypdf2() -> None:
    """The 0.3.x ``viz`` alias covers pyfolio + interactive + the PyPDF2 report dep."""
    extras = _extras()
    expected = _normalized(extras["pyfolio"]) | _normalized(extras["interactive"]) | _normalized({"PyPDF2>=3"})
    assert _normalized(extras["viz"]) == expected


def test_viz_alias_is_subset_of_functional_union() -> None:
    """Everything ``viz`` installs must be covered by the functional extras."""
    extras = _extras()
    assert _normalized(extras["viz"]) <= _functional_union(extras)
