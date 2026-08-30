"""Regression tests for the public-API snapshot entry point."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path


def test_direct_snapshot_script_prefers_its_checkout_source(tmp_path: Path) -> None:
    """Runtime ``__all__`` discovery must come from the checkout being scanned."""

    repository = tmp_path / "checkout"
    script_dir = repository / "scripts"
    package_dir = repository / "fincore"
    script_dir.mkdir(parents=True)
    package_dir.mkdir()
    shadow_package_dir = tmp_path / "shadow" / "fincore"
    shadow_package_dir.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    shutil.copy2(root / "scripts" / "snapshot_public_api.py", script_dir / "snapshot_public_api.py")
    (package_dir / "__init__.py").write_text("__all__ = ['checkout_sentinel']\n", encoding="utf-8")
    (shadow_package_dir / "__init__.py").write_text(
        "__all__ = [f'shadow_{number}' for number in range(500)]\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path / "shadow"), str(repository)))

    output = repository / "snapshot.json"
    subprocess.run(
        [sys.executable, str(script_dir / "snapshot_public_api.py"), "--output", str(output)],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["surfaces"]["fincore"]["public_symbols"] == ["checkout_sentinel"]


def _snapshot_script() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "snapshot_public_api.py"


def _write_package(root: Path, source: str) -> None:
    package = root / "fincore"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(source, encoding="utf-8")


def _write_wheel(wheel: Path, source: str, *extra_members: tuple[str, str]) -> None:
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fincore/__init__.py", source)
        for name, content in extra_members:
            archive.writestr(name, content)


def _run_snapshot(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_snapshot_script()), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_source_and_wheel_snapshots_share_a_static_callable_contract(tmp_path: Path) -> None:
    """Source and wheel scans must produce byte-identical schema-v2 snapshots."""

    source = """\
import unavailable_heavy_dependency

__all__ = ["sync", "asynchronous", "Widget"]

def sync(a, /, b=3, *items, threshold="high", required, **options):
    return a

async def asynchronous(*, delay=0):
    return delay

class Widget:
    def __init__(self, name, /, *, enabled=True):
        self.name = name
"""
    source_root = tmp_path / "source"
    _write_package(source_root, source)
    wheel = tmp_path / "fincore-0.0.0-py3-none-any.whl"
    _write_wheel(wheel, source)
    source_output = tmp_path / "source.json"
    wheel_output = tmp_path / "wheel.json"

    source_run = _run_snapshot(
        "--source-root",
        str(source_root),
        "--surface",
        "fincore",
        "--output",
        str(source_output),
        cwd=tmp_path,
    )
    wheel_run = _run_snapshot(
        "--wheel",
        str(wheel),
        "--surface",
        "fincore",
        "--output",
        str(wheel_output),
        cwd=tmp_path,
    )
    compare_run = _run_snapshot(
        "--source-root",
        str(source_root),
        "--wheel",
        str(wheel),
        "--surface",
        "fincore",
        "--compare",
        cwd=tmp_path,
    )

    assert source_run.returncode == 0, source_run.stderr
    assert wheel_run.returncode == 0, wheel_run.stderr
    assert compare_run.returncode == 0, compare_run.stderr
    source_snapshot = json.loads(source_output.read_text(encoding="utf-8"))
    wheel_snapshot = json.loads(wheel_output.read_text(encoding="utf-8"))
    assert source_snapshot == wheel_snapshot
    assert source_snapshot["schema_version"] == 2
    entry = source_snapshot["surfaces"]["fincore"]["entries"]["sync"]
    assert entry == {
        "kind": "function",
        "public_path": "fincore.sync",
        "signature": {
            "parameters": [
                {"annotation": None, "default": None, "kind": "POSITIONAL_ONLY", "name": "a"},
                {"annotation": None, "default": "3", "kind": "POSITIONAL_OR_KEYWORD", "name": "b"},
                {"annotation": None, "default": None, "kind": "VAR_POSITIONAL", "name": "items"},
                {"annotation": None, "default": "'high'", "kind": "KEYWORD_ONLY", "name": "threshold"},
                {"annotation": None, "default": None, "kind": "KEYWORD_ONLY", "name": "required"},
                {"annotation": None, "default": None, "kind": "VAR_KEYWORD", "name": "options"},
            ],
            "return_annotation": None,
        },
    }
    assert source_snapshot["surfaces"]["fincore"]["entries"]["asynchronous"]["kind"] == "async_function"
    assert source_snapshot["surfaces"]["fincore"]["entries"]["Widget"]["signature"]["parameters"] == [
        {"annotation": None, "default": None, "kind": "POSITIONAL_ONLY", "name": "name"},
        {"annotation": None, "default": "True", "kind": "KEYWORD_ONLY", "name": "enabled"},
    ]


def test_static_snapshot_does_not_import_optional_or_heavy_dependencies(tmp_path: Path) -> None:
    """A top-level unavailable import is data to scan, never code to execute."""

    source_root = tmp_path / "source"
    _write_package(
        source_root,
        "import unavailable_heavy_dependency\n__all__ = ['available']\ndef available(): pass\n",
    )

    result = _run_snapshot("--source-root", str(source_root), "--surface", "fincore", cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert '"available"' in result.stdout


def test_selected_empty_public_surface_fails_closed(tmp_path: Path) -> None:
    """An explicitly selected surface cannot disappear behind an empty export list."""

    source_root = tmp_path / "source"
    _write_package(source_root, "__all__ = []\n")

    result = _run_snapshot("--source-root", str(source_root), "--surface", "fincore", cwd=tmp_path)

    assert result.returncode != 0
    assert "has no public exports" in result.stderr


def test_snapshot_rejects_source_package_symlink_that_escapes_root(tmp_path: Path) -> None:
    """The selected package must be physically contained in the requested source root."""

    source_root = tmp_path / "source"
    source_root.mkdir()
    outside = tmp_path / "outside"
    _write_package(outside, "__all__ = ['escape']\ndef escape(): pass\n")
    (source_root / "fincore").symlink_to(outside / "fincore", target_is_directory=True)

    result = _run_snapshot("--source-root", str(source_root), "--surface", "fincore", cwd=tmp_path)

    assert result.returncode != 0
    assert "symlink" in result.stderr.lower() or "inside source root" in result.stderr.lower()


def test_snapshot_rejects_wheel_members_that_escape_the_archive_root(tmp_path: Path) -> None:
    """Archive paths are validated before any member is parsed."""

    source = "__all__ = ['safe']\ndef safe(): pass\n"
    wheel = tmp_path / "fincore-0.0.0-py3-none-any.whl"
    _write_wheel(wheel, source, ("../outside.py", "raise RuntimeError"))

    result = _run_snapshot("--wheel", str(wheel), "--surface", "fincore", cwd=tmp_path)

    assert result.returncode != 0
    assert "unsafe wheel member" in result.stderr.lower()


def test_snapshot_rejects_wheel_source_member_marked_as_unix_symlink(tmp_path: Path) -> None:
    """A wheel source module must be a regular file even when it is never extracted."""

    wheel = tmp_path / "fincore-0.0.0-py3-none-any.whl"
    source_member = zipfile.ZipInfo("fincore/__init__.py")
    source_member.create_system = 3
    source_member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(source_member, "__all__ = ['safe']\ndef safe(): pass\n")

    result = _run_snapshot("--wheel", str(wheel), "--surface", "fincore", cwd=tmp_path)

    assert result.returncode != 0
    assert "not a regular file" in result.stderr.lower()


def test_snapshot_rejects_multiple_or_damaged_wheels(tmp_path: Path) -> None:
    """The wheel mode has one unambiguous, readable package input."""

    source = "__all__ = ['safe']\ndef safe(): pass\n"
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    _write_wheel(first, source)
    _write_wheel(second, source)
    damaged = tmp_path / "damaged.whl"
    damaged.write_text("not a zip archive", encoding="utf-8")

    duplicate = _run_snapshot(
        "--wheel",
        str(first),
        "--wheel",
        str(second),
        "--surface",
        "fincore",
        cwd=tmp_path,
    )
    corrupt = _run_snapshot("--wheel", str(damaged), "--surface", "fincore", cwd=tmp_path)

    assert duplicate.returncode != 0
    assert "exactly one wheel" in duplicate.stderr.lower()
    assert corrupt.returncode != 0
    assert "invalid wheel" in corrupt.stderr.lower()


def test_compare_fails_when_wheel_public_contract_differs_from_source(tmp_path: Path) -> None:
    """A source/wheel compare is a real equality gate, not two independent scans."""

    source_root = tmp_path / "source"
    _write_package(source_root, "__all__ = ['same']\ndef same(value=1): pass\n")
    wheel = tmp_path / "fincore-0.0.0-py3-none-any.whl"
    _write_wheel(wheel, "__all__ = ['same']\ndef same(value=2): pass\n")

    result = _run_snapshot(
        "--source-root",
        str(source_root),
        "--wheel",
        str(wheel),
        "--surface",
        "fincore",
        "--compare",
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "source/wheel public API snapshot mismatch" in result.stderr
