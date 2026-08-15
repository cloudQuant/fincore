"""Release-consistency checks for built artifact dependency metadata."""

from __future__ import annotations

import copy
import io
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_CHECK = REPO_ROOT / "scripts" / "check_release_consistency.py"


def _build_dist(out_dir: Path) -> Path:
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(out_dir), str(REPO_ROOT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"build failed:\n{proc.stdout}\n{proc.stderr}"
    assert list(out_dir.glob("fincore-*.whl"))
    assert list(out_dir.glob("fincore-*.tar.gz"))
    return out_dir


@pytest.fixture(scope="module")
def clean_dist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real clean wheel/sdist pair used as mutation input."""
    return _build_dist(tmp_path_factory.mktemp("release-consistency-dist"))


def _append_wheel_requirement(wheel: Path, requirement: str) -> None:
    replacement = wheel.with_name(f"rewritten-{wheel.name}")
    metadata_files = 0
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(replacement, "w") as destination:
        for info in source.infolist():
            content = source.read(info.filename)
            if info.filename.endswith(".dist-info/METADATA"):
                metadata_files += 1
                content = _add_requirement_header(content, requirement)
            destination.writestr(info, content)
    assert metadata_files == 1
    replacement.replace(wheel)


def _add_requirement_header(metadata: bytes, requirement: str) -> bytes:
    """Insert a valid ``Requires-Dist`` header before the metadata body."""
    separator = b"\r\n\r\n" if b"\r\n\r\n" in metadata else b"\n\n"
    headers, found, body = metadata.partition(separator)
    assert found, "metadata has no header/body separator"
    newline = b"\r\n" if separator.startswith(b"\r\n") else b"\n"
    return headers + newline + f"Requires-Dist: {requirement}".encode() + separator + body


def _append_sdist_requirement(sdist: Path, requirement: str) -> None:
    replacement = sdist.with_name(f"rewritten-{sdist.name}")
    metadata_files = 0
    with tarfile.open(sdist) as source, tarfile.open(replacement, "w:gz") as destination:
        for member in source.getmembers():
            content = source.extractfile(member).read() if member.isfile() else None
            if member.name.endswith("/PKG-INFO"):
                metadata_files += 1
                content = _add_requirement_header(content or b"", requirement)
                member = copy.copy(member)
                member.size = len(content)
            destination.addfile(member, io.BytesIO(content) if content is not None else None)
    assert metadata_files == 1
    replacement.replace(sdist)


def _release_check(dist_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RELEASE_CHECK), "--dist", str(dist_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.p3  # packaging metadata guard: not a core-metric smoke test; keeps it out of the p0/p1 priority jobs
@pytest.mark.parametrize(
    ("artifact", "requirement"),
    (
        ("wheel", "Empyrical>=1"),
        ("sdist", "AlphaLens>=1"),
        ("wheel", "safe-package @ https://example.invalid/safe-package-1.0.whl"),
        ("wheel", "not a valid @ requirement"),
        ("sdist", "not a valid @ requirement"),
    ),
)
def test_release_consistency_rejects_prohibited_artifact_requirements(
    clean_dist: Path, tmp_path: Path, artifact: str, requirement: str
) -> None:
    """Real release checks reject mixed-case compatibility names and URLs."""
    dist_dir = tmp_path / "dist"
    shutil.copytree(clean_dist, dist_dir)
    if artifact == "wheel":
        _append_wheel_requirement(next(dist_dir.glob("fincore-*.whl")), requirement)
    else:
        _append_sdist_requirement(next(dist_dir.glob("fincore-*.tar.gz")), requirement)

    result = _release_check(dist_dir)

    assert result.returncode == 1, f"release consistency accepted {artifact} metadata: {requirement!r}"
    assert requirement in result.stdout
    assert "Traceback" not in result.stdout + result.stderr
