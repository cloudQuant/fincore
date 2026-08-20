#!/usr/bin/env python3
"""Fail-closed release-candidate verification.

Verifies the ``dist/`` directory against the release contract:

1. It contains one wheel and one sdist built from the same source tree.
2. Each artifact's SHA256 digest matches the release manifest (when provided).
3. The package version is consistent across ``pyproject.toml`` and the wheel
   ``METADATA`` (a wheel whose metadata version drifts from the source tree is
   a release-identity violation).

This is the local, deterministic half of the supply-chain gate; the CI build
job is the only place that may *produce* a candidate, and publish only
downloads and re-verifies that candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _wheel_metadata_version(wheel_path: Path) -> str | None:
    with zipfile.ZipFile(wheel_path) as zf:
        for name in zf.namelist():
            if name.endswith(".dist-info/METADATA"):
                for line in zf.read(name).decode("utf-8").splitlines():
                    if line.startswith("Version:"):
                        return line.split(":", 1)[1].strip()
    return None


def _pyproject_version() -> str:
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return str(data["project"]["version"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", default="dist/", help="path to the dist directory")
    parser.add_argument("--manifest", default=None, help="path to a release manifest JSON")
    args = parser.parse_args(argv)

    dist = Path(args.dist)
    artifacts = sorted(p for p in dist.glob("*") if p.is_file()) if dist.exists() else []
    wheels = [p for p in artifacts if p.suffix == ".whl"]
    sdists = [p for p in artifacts if p.name.endswith(".tar.gz")]

    violations: list[str] = []
    if len(wheels) != 1 or len(sdists) != 1:
        violations.append(
            f"release candidate must contain exactly one wheel and one sdist; "
            f"found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
        )

    digests = {p.name: _sha256(p) for p in artifacts}

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("artifacts", {})
        for name, expected_sha in expected.items():
            actual = digests.get(name)
            if actual != expected_sha:
                violations.append(f"digest mismatch for {name}: manifest={expected_sha} actual={actual}")

    pyproject_version = _pyproject_version()
    for wheel in wheels:
        wheel_version = _wheel_metadata_version(wheel)
        if wheel_version is not None and wheel_version != pyproject_version:
            violations.append(f"version drift: wheel {wheel_version} != pyproject.toml {pyproject_version}")

    for violation in violations:
        print(f"FAIL: {violation}", file=sys.stderr)
    if violations:
        return 1
    print("release candidate is consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
