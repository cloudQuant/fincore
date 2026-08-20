"""Release-candidate verification tests for scripts/check_release_candidate.py."""

from __future__ import annotations

import json
import zipfile
from typing import TYPE_CHECKING

from scripts.check_release_candidate import _pyproject_version, _sha256, _wheel_metadata_version, main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_pyproject_version_is_next_dev() -> None:
    assert _pyproject_version() == "0.4.0.dev0"


def test_sha256_is_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "artifact.txt"
    p.write_bytes(b"payload")

    assert _sha256(p) == _sha256(p)
    assert len(_sha256(p)) == 64


def _make_wheel(path: Path, version: str) -> Path:
    wheel = path / f"fincore-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr(
            "fincore-0.4.0.dev0.dist-info/METADATA", f"Metadata-Version: 2.1\nName: fincore\nVersion: {version}\n"
        )
    return wheel


def test_empty_dist_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    empty = tmp_path / "empty-dist"
    empty.mkdir()

    rc = main(["--dist", str(empty)])

    assert rc == 1
    assert "exactly one wheel and one sdist" in capsys.readouterr().err


def test_version_drift_is_detected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    _make_wheel(dist, "0.3.0")
    (dist / "fincore-0.3.0.tar.gz").write_bytes(b"not a real sdist")

    rc = main(["--dist", str(dist)])

    assert rc == 1
    assert "version drift" in capsys.readouterr().err


def test_matching_version_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    _make_wheel(dist, "0.4.0.dev0")
    (dist / "fincore-0.4.0.dev0.tar.gz").write_bytes(b"not a real sdist")

    rc = main(["--dist", str(dist)])

    assert rc == 0
    assert "consistent" in capsys.readouterr().out


def test_manifest_digest_mismatch_detected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    _make_wheel(dist, "0.4.0.dev0")
    sdist = dist / "fincore-0.4.0.dev0.tar.gz"
    sdist.write_bytes(b"payload")

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": {sdist.name: "0" * 64}}))

    rc = main(["--dist", str(dist), "--manifest", str(manifest)])

    assert rc == 1
    assert "digest mismatch" in capsys.readouterr().err
