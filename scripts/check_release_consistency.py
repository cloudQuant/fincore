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

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

_VERSION_RE = re.compile(r"reports version \*\*(\d+\.\d+\.\d+)\*\*")
_RELEASE_SECTION_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.MULTILINE)
_SELF_DEP_RE = re.compile(r"^\s*fincore(\[|\s*$|$)", re.IGNORECASE)


def _project() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]


def _scrubbed_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}


def _gt(a: str, b: str) -> bool:
    from packaging.version import Version

    return Version(a) > Version(b)


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
    probe = "import fincore, pathlib; print(fincore.__version__); print(pathlib.Path(fincore.__file__).resolve())"
    proc = subprocess.run(
        [sys.executable, "-c", probe],
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
        future = [v for v in _RELEASE_SECTION_RE.findall(changelog) if _gt(v, version)]
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
            check(metadata["Version"] == version, f"wheel METADATA version for {wheel.name}")
            requires = metadata.get_all("Requires-Dist", [])
            self_deps = [req for req in requires if _SELF_DEP_RE.match(req)]
            check(not self_deps, f"no self-dependency in {wheel.name} ({self_deps or 'clean'})")
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
                pkg_info = next(n for n in tf.getnames() if n.endswith("/PKG-INFO"))
                text = tf.extractfile(pkg_info).read().decode("utf-8", "replace")  # type: ignore[union-attr]
            version_line = next(
                (ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("Version:")), None
            )
            check(version_line == version, f"sdist PKG-INFO version for {sdist.name}")
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
        if not tags:
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
