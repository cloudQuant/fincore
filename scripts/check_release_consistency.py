#!/usr/bin/env python
"""Verify that every release fact agrees with pyproject.toml.

pyproject.toml is the single project-metadata source.  This script cross-checks:

1. ``fincore.__version__`` at runtime (fresh subprocess importing the checkout,
   falling back to the source-tree path when no distribution metadata exists);
2. CHANGELOG.md's version statement (and that no future release section
   exists);
3. every wheel in ``--dist``: METADATA ``Version``, the ``Provides-Extra``
   set, the wheel filename, and the absence of a ``Requires-Dist: fincore[...]``
   self-dependency;
4. source-level extras: no extra may reference ``fincore[...]``;
5. git tag agreement, enforced only when tags are present in the checkout
   (shallow CI clones carry none; release checkouts do).

Exit code is 0 when every available fact is consistent, 1 otherwise.
"""

from __future__ import annotations

import argparse
import email
import os
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

# Changelog versions follow PEP 440.  Development candidates (for example
# ``0.5.0.dev0``) intentionally have no release tag, but they must still be
# cross-checked against project metadata before a candidate can be built.
_VERSION_RE = re.compile(r"reports version \*\*([^*\s]+)\*\*")
_RELEASE_SECTION_RE = re.compile(r"^## \[([^\]]+)\]", re.MULTILINE)
_SELF_DEP_RE = re.compile(r"^\s*fincore(\[|\s*$|$)", re.IGNORECASE)
_BANNED_ARTIFACT_FRAGMENTS = (
    "versioneer",
    "requirements-alphalens",
    "requirements-empyrical",
    "requirements-pyfolio",
)
_BANNED_ARTIFACT_SUFFIXES = (".ipynb", ".png")
_CONTRIBUTOR_REQUIREMENT_ARTIFACTS = {"requirements.txt", "requirements-test.txt"}
_PROHIBITED_EXTERNAL_REQUIREMENTS = {"alphalens", "empyrical"}
_REQUIRED_LEGAL_FILES = {
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "THIRD_PARTY_LICENSES/Apache-2.0.txt",
}
_REQUIRED_RUNTIME_MODULES = {
    "fincore/__init__.py",
    "fincore/attribution/performance.py",
    "fincore/factor_analysis/__init__.py",
    "fincore/factor_analysis/data.py",
    "fincore/factor_analysis/performance.py",
    "fincore/factor_analysis/portfolio.py",
    "fincore/report/models.py",
    "fincore/report/portfolio/compute.py",
    "fincore/report/renderers/html.py",
    "fincore/runtime/catalog.py",
    "fincore/py.typed",
}
_REMOVED_RUNTIME_PREFIXES = (
    "fincore/alphalens/",
    "fincore/empyrical.py",
    "fincore/pyfolio.py",
    "fincore/tearsheets/",
    "fincore/core/",
    "fincore/plugin/",
)


def _project() -> dict[str, Any]:
    with PYPROJECT.open("rb") as fh:
        project = tomllib.load(fh)["project"]
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml [project] must be a table")
    return cast("dict[str, Any]", project)


def _scrubbed_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}


def _gt(a: str, b: str) -> bool:
    return Version(a) > Version(b)


def _valid_version(value: str) -> bool:
    """Return whether a changelog label is a PEP 440 version."""
    try:
        Version(value)
    except InvalidVersion:
        return False
    return True


def _check_artifact_layout(
    check: Callable[[bool, str], None],
    names: set[str],
    read_text: Callable[[str], str],
    *,
    label: str,
    prefix: str = "",
) -> None:
    """Check source-free runtime layout and retained legal material."""
    relative_names = {name.removeprefix(prefix) for name in names if name.startswith(prefix)}
    required = {prefix + module for module in _REQUIRED_RUNTIME_MODULES}
    check(required <= names, f"{label} contains the required canonical runtime modules")
    removed = tuple(prefix + path for path in _REMOVED_RUNTIME_PREFIXES)
    check(
        not any(name.startswith(removed) for name in names),
        f"{label} excludes removed compatibility runtime paths",
    )
    forbidden_prefixes = tuple(prefix + directory for directory in ("tests/", "examples/", "docs/", "benchmarks/"))
    check(
        not any(name.startswith(forbidden_prefixes) for name in names),
        f"{label} excludes tests, examples, docs, and benchmarks",
    )
    check(
        not any(
            name.startswith(("/", "../"))
            or "/../" in name
            or name.lower().endswith(_BANNED_ARTIFACT_SUFFIXES)
            or Path(name).name in _CONTRIBUTOR_REQUIREMENT_ARTIFACTS
            or any(fragment in name.lower() for fragment in _BANNED_ARTIFACT_FRAGMENTS)
            for name in names
        ),
        f"{label} excludes sibling paths, contributor requirements, Versioneer, notebooks, PNGs, and oracle requirements",
    )

    def legal_relative_path(name: str) -> str | None:
        marker = ".dist-info/licenses/"
        if marker in name:
            return name.rsplit(marker, 1)[1]
        if name in _REQUIRED_LEGAL_FILES:
            return name
        return None

    legal_names = {path for name in relative_names if (path := legal_relative_path(name)) is not None}
    check(legal_names == _REQUIRED_LEGAL_FILES, f"{label} includes the MIT license and third-party legal files")

    def artifact_name(relative_path: str) -> str | None:
        if prefix:
            candidate = prefix + relative_path
            return candidate if candidate in names else None
        suffix = f".dist-info/licenses/{relative_path}"
        return next((name for name in names if name.endswith(suffix)), None)

    mit_license = artifact_name("LICENSE")
    notice = artifact_name("NOTICE")
    third_party_inventory = artifact_name("THIRD_PARTY_NOTICES.md")
    apache_license = artifact_name("THIRD_PARTY_LICENSES/Apache-2.0.txt")
    if mit_license is not None:
        check(read_text(mit_license).startswith("MIT License"), f"{label} LICENSE is MIT")
    if notice is not None:
        check("Apache-2.0" in read_text(notice), f"{label} NOTICE retains Apache-2.0 attribution")
    if third_party_inventory is not None:
        check("upstream_version" in read_text(third_party_inventory), f"{label} ships third-party provenance inventory")
    if apache_license is not None:
        check(
            read_text(apache_license).startswith("                                 Apache License"),
            f"{label} ships Apache-2.0 text",
        )


def _is_allowed_runtime_requirement(requirement: Requirement) -> bool:
    """Return whether a requirement remains a direct, reproducible dependency."""
    return canonicalize_name(requirement.name) not in _PROHIBITED_EXTERNAL_REQUIREMENTS and requirement.url is None


def _check_artifact_requirements(check: Callable[[bool, str], None], requirements: list[str], *, label: str) -> None:
    """Reject retired upstream packages and direct URLs in built metadata."""
    malformed: list[str] = []
    prohibited: list[str] = []
    for raw in requirements:
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            malformed.append(raw)
            continue
        if not _is_allowed_runtime_requirement(requirement):
            prohibited.append(raw)
    check(
        not malformed,
        f"{label} Requires-Dist contains only valid requirements ({malformed or 'valid'})",
    )
    check(
        not prohibited,
        f"{label} Requires-Dist uses no retired upstream package or direct URL ({prohibited or 'clean'})",
    )


def _failures(dist_dir: Path | None) -> list[str]:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if condition:
            print(f"PASS: {message}")
        else:
            failures.append(message)
            print(f"FAIL: {message}")

    project = _project()
    version = str(project["version"])
    print(f"pyproject.toml version: {version}")

    # ------------------------------------------------------------------
    # 1. Runtime __version__ (checkout import in a fresh subprocess).
    # ------------------------------------------------------------------
    probe = (
        "import sys, sysconfig; from pathlib import Path; "
        "root = Path(sys.argv[1]).resolve(); "
        "sys.path[:0] = [str(root), sysconfig.get_paths()['purelib']]; "
        "import fincore; print(fincore.__version__); print(Path(fincore.__file__).resolve())"
    )
    proc = subprocess.run(
        [sys.executable, "-S", "-E", "-c", probe, str(REPO_ROOT)],
        cwd=REPO_ROOT,
        env=_scrubbed_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        check(False, f"runtime version probe failed: {proc.stderr.strip()}")
    else:
        lines = proc.stdout.strip().splitlines()
        runtime_version = lines[0]
        check(
            runtime_version == version,
            f"runtime fincore.__version__ ({runtime_version}) equals pyproject ({version})",
        )
        runtime_file = Path(lines[1])
        check(
            runtime_file.is_relative_to(REPO_ROOT),
            f"runtime probe imported the checkout ({runtime_file})",
        )

    # ------------------------------------------------------------------
    # 2. CHANGELOG.
    # ------------------------------------------------------------------
    if CHANGELOG.is_file():
        changelog = CHANGELOG.read_text(encoding="utf-8")
        stated = _VERSION_RE.findall(changelog)
        check(bool(stated), "CHANGELOG states a package version")
        if stated:
            check(stated[-1] == version, f"CHANGELOG version statement ({stated[-1]}) equals pyproject ({version})")
        future = [
            candidate
            for candidate in _RELEASE_SECTION_RE.findall(changelog)
            if _valid_version(candidate) and _gt(candidate, version)
        ]
        check(not future, f"CHANGELOG has no release section newer than {version} (found {future})")
    else:
        failures.append("CHANGELOG.md missing")

    # ------------------------------------------------------------------
    # 3. Built artifacts (wheels + sdist).
    # ------------------------------------------------------------------
    if dist_dir is not None and dist_dir.is_dir():
        wheels = sorted(dist_dir.glob("fincore-*.whl"))
        check(bool(wheels), f"found wheels in {dist_dir}")
        for wheel in wheels:
            check(
                wheel.name.startswith(f"fincore-{version}-"),
                f"wheel filename {wheel.name} embeds version {version}",
            )
            with zipfile.ZipFile(wheel) as zf:
                metadata_name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
                metadata = email.message_from_bytes(zf.read(metadata_name))
                _check_artifact_layout(
                    check,
                    set(zf.namelist()),
                    lambda name: zf.read(name).decode("utf-8", "replace"),
                    label=wheel.name,
                )
            check(metadata["Version"] == version, f"wheel METADATA version for {wheel.name}")
            check(metadata["License-Expression"] == "MIT", f"wheel METADATA license expression for {wheel.name}")
            check(
                set(metadata.get_all("License-File", [])) == _REQUIRED_LEGAL_FILES,
                f"wheel METADATA lists all legal files for {wheel.name}",
            )
            requires = metadata.get_all("Requires-Dist", [])
            self_deps = [req for req in requires if _SELF_DEP_RE.match(req)]
            check(not self_deps, f"no self-dependency in {wheel.name} ({self_deps or 'clean'})")
            _check_artifact_requirements(check, requires, label=wheel.name)
            provides = set(metadata.get_all("Provides-Extra", []))
            expected_extras = set(project.get("optional-dependencies", {}))
            check(
                provides == expected_extras,
                f"{wheel.name} Provides-Extra equals pyproject extras ({provides ^ expected_extras})",
            )
        sdists = sorted(dist_dir.glob("fincore-*.tar.gz"))
        for sdist in sdists:
            check(
                sdist.name.startswith(f"fincore-{version}"),
                f"sdist filename {sdist.name} embeds version {version}",
            )
            with tarfile.open(sdist) as tf:
                names = set(tf.getnames())
                package_root = next(name.split("/", 1)[0] for name in names if name.endswith("/PKG-INFO")) + "/"
                pkg_info = next(n for n in names if n.endswith("/PKG-INFO"))
                pkg_info_metadata = email.message_from_bytes(tf.extractfile(pkg_info).read())  # type: ignore[union-attr]
                _check_artifact_layout(
                    check,
                    names,
                    lambda name: tf.extractfile(name).read().decode("utf-8", "replace"),  # type: ignore[union-attr]
                    label=sdist.name,
                    prefix=package_root,
                )
            check(pkg_info_metadata["Version"] == version, f"sdist PKG-INFO version for {sdist.name}")
            check(
                pkg_info_metadata["License-Expression"] == "MIT", f"sdist PKG-INFO license expression for {sdist.name}"
            )
            check(
                set(pkg_info_metadata.get_all("License-File", [])) == _REQUIRED_LEGAL_FILES,
                f"sdist PKG-INFO lists all legal files for {sdist.name}",
            )
            _check_artifact_requirements(check, pkg_info_metadata.get_all("Requires-Dist", []), label=sdist.name)
    else:
        print("NOTE: no --dist directory given; skipping built-artifact checks")

    # ------------------------------------------------------------------
    # 4. Source-level self-reference guard.
    # ------------------------------------------------------------------
    for extra_name, reqs in project.get("optional-dependencies", {}).items():
        for req in reqs:
            check(
                not _SELF_DEP_RE.match(req),
                f"extra {extra_name!r} contains no self-reference ({req!r})",
            )
            parsed = Requirement(req)
            check(
                _is_allowed_runtime_requirement(parsed),
                f"extra {extra_name!r} uses no retired upstream package or direct URL ({req!r})",
            )

    # ------------------------------------------------------------------
    # 5. Git tag.
    # ------------------------------------------------------------------
    try:
        tags_proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "tag", "--list"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"NOTE: git unavailable ({exc}); skipping tag check")
    else:
        tags = {t.strip() for t in tags_proc.stdout.splitlines() if t.strip()}
        if Version(version).is_devrelease:
            print(f"NOTE: {version} is a development version; skipping release-tag check")
        elif not tags:
            print("NOTE: no git tags present in checkout (shallow clone); skipping tag check")
        else:
            check(version in tags or f"v{version}" in tags, f"git tag for version {version} exists")

    return failures


def main(dist_dir: Path | None) -> int:
    failures = _failures(dist_dir)
    print()
    if failures:
        print(f"{len(failures)} release-consistency failure(s):")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("Release consistency: OK")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", help="directory containing built wheels/sdist (e.g. dist/ or /tmp/fincore-dist)")
    args = parser.parse_args()
    raise SystemExit(main(Path(args.dist).resolve() if args.dist else None))
