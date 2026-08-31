"""Contract tests for the complete-input 0042-R2 surface-union collector.

The collector is still a raw fact producer, not a D0 verdict.  It must make
every inventory source explicit before a later reviewed inventory can decide
what is retained, rewritten, or removed.
"""

from __future__ import annotations

import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts" / "collect_0042_r2_surface_union.py"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _source_root(tmp_path: Path) -> Path:
    source_root = tmp_path / "source"
    source_root.mkdir()
    _write(source_root / "fincore" / "__init__.py", "from .api import annual_return\n")
    _write(
        source_root / "fincore" / "api.py",
        """def annual_return(values):
    return sum(values)


def _private_helper(values):
    return values


class Portfolio:
    def analyze(self):
        return 1

    def _private_method(self):
        return 0
""",
    )
    _write(source_root / "fincore" / "metric_registry.py", "METRIC_REGISTRY = {}\n")
    _write(source_root / "docs" / "guide.md", "# Guide\n")
    _write(source_root / "mkdocs_docs" / "index.md", "# Documentation\n")
    _write(source_root / "examples" / "demo.py", "print('demo')\n")
    _write(source_root / "tests" / "benchmarks" / "test_profile.py", "def test_profile():\n    pass\n")
    _write(source_root / "tests" / "compat" / "fixtures" / "legacy-api.json", "{}\n")
    _write(
        source_root / "pyproject.toml",
        """[project]
name = "fincore"
version = "0.5.0"

[project.optional-dependencies]
viz = ["matplotlib"]
""",
    )
    _write(source_root / "requirements-optional.txt", "matplotlib\n")
    _write(source_root / "MANIFEST.in", "include README.md\n")
    _write(source_root / "README.md", "# Fincore\n")

    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(["git", "add", "."], cwd=source_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=0042-R2 test",
            "-c",
            "user.email=0042-r2@example.invalid",
            "commit",
            "-qm",
            "surface union input",
        ],
        cwd=source_root,
        check=True,
    )
    return source_root


def _wheel(tmp_path: Path, *, unsafe_member: str | None = None, symlink_member: bool = False) -> Path:
    wheel = tmp_path / "fincore-0.5.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fincore/__init__.py", "")
        archive.writestr("fincore/api.py", "def annual_return(values): return sum(values)\n")
        archive.writestr("fincore-0.5.0.dist-info/METADATA", "Name: fincore\nVersion: 0.5.0\n")
        if unsafe_member is not None:
            archive.writestr(unsafe_member, "unsafe")
        if symlink_member:
            member = zipfile.ZipInfo("fincore/linked.py")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(member, "api.py")
    return wheel


def _collect(source_root: Path, wheel: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--wheel", str(wheel), "--output", str(output)],
        cwd=source_root,
        capture_output=True,
        check=False,
        text=True,
    )


def test_collects_all_required_raw_surface_kinds_deterministically(tmp_path: Path) -> None:
    source_root = _source_root(tmp_path)
    wheel = _wheel(tmp_path)
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"

    first = _collect(source_root, wheel, first_output)
    second = _collect(source_root, wheel, second_output)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first_output.read_bytes() == second_output.read_bytes()

    artifact = json.loads(first_output.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 1
    assert artifact["artifact_type"] == "surface_union_facts_discovery"
    assert artifact["discovery_status"] == "complete"
    assert artifact["not_for_d0"] is True
    assert artifact["entry_count"] == len(artifact["entries"])
    assert artifact["entries"] == sorted(artifact["entries"], key=lambda entry: entry["entry_id"])
    assert artifact["source_provenance"]["clean"] is True
    assert artifact["wheel"]["sha256"]
    assert artifact["wheel"]["member_count"] == 3

    kinds = {entry["source_kind"] for entry in artifact["entries"]}
    assert {
        "public_definition",
        "registry",
        "manifest",
        "documentation",
        "example",
        "benchmark",
        "extra",
        "wheel_content",
    } <= kinds

    entries_by_id = {entry["entry_id"]: entry for entry in artifact["entries"]}
    assert "public_definition:fincore/api.py:function:annual_return:1" in entries_by_id
    assert "public_definition:fincore/api.py:class:Portfolio:9" in entries_by_id
    assert "public_definition:fincore/api.py:method:Portfolio.analyze:10" in entries_by_id
    assert all("_private_helper" not in entry_id for entry_id in entries_by_id)
    assert all("_private_method" not in entry_id for entry_id in entries_by_id)
    assert all(set(entry) == {"entry_id", "source", "source_kind"} for entry in artifact["entries"]), (
        "raw discovery must not infer owners, dispositions, targets, or oracles"
    )


def test_rejects_dirty_source_without_replacing_previous_output(tmp_path: Path) -> None:
    source_root = _source_root(tmp_path)
    wheel = _wheel(tmp_path)
    output = tmp_path / "union.json"
    output.write_text('{"previous": true}\n', encoding="utf-8")
    (source_root / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    result = _collect(source_root, wheel, output)

    assert result.returncode != 0
    assert "clean" in result.stderr.lower()
    assert output.read_text(encoding="utf-8") == '{"previous": true}\n'


def test_rejects_wheel_member_path_traversal(tmp_path: Path) -> None:
    source_root = _source_root(tmp_path)
    wheel = _wheel(tmp_path, unsafe_member="../escape.py")

    result = _collect(source_root, wheel, tmp_path / "union.json")

    assert result.returncode != 0
    assert "wheel" in result.stderr.lower()
    assert "path" in result.stderr.lower()


def test_rejects_wheel_symbolic_link_member(tmp_path: Path) -> None:
    source_root = _source_root(tmp_path)
    wheel = _wheel(tmp_path, symlink_member=True)

    result = _collect(source_root, wheel, tmp_path / "union.json")

    assert result.returncode != 0
    assert "wheel" in result.stderr.lower()
    assert "regular" in result.stderr.lower()
