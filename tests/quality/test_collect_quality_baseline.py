from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest


def _collector_module():
    path = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).parents[2])).resolve() / "scripts" / "collect_quality_baseline.py"
    spec = importlib.util.spec_from_file_location("collect_quality_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_test_counts_distinguishes_discovered_from_marker_selected() -> None:
    collector = _collector_module()

    serial = collector._parse_test_counts("collected 2299 items / 2293 deselected / 6 selected")
    xdist = collector._parse_test_counts("created: 8/8 workers\n8 workers [2278 items]")

    assert serial == {"discovered": 2299, "selected": 6}
    assert xdist == {"discovered": None, "selected": 2278}


def test_render_incomplete_failure_report_without_coverage_run() -> None:
    collector = _collector_module()
    data = {
        "generated_at": "2026-08-12T00:00:00+00:00",
        "source": {
            "commit": "abc",
            "dirty": True,
            "tracked_diff_sha256": "diff",
            "untracked_manifest_sha256": "untracked",
        },
        "copy_manifest_sha256": "manifest",
        "environment": {"python": "test"},
        "outcome": "fail",
        "failure": "trusted-baseline modified disposable-copy files",
        "runs": [
            {
                "label": "trusted-baseline",
                "selector": "not slow",
                "discovered": 10,
                "selected": 9,
                "passed": 0,
                "skipped": 0,
                "warnings": 0,
                "duration_seconds": 0.1,
                "returncode": 1,
                "integrity_ok": False,
            }
        ],
    }

    markdown = collector._render_markdown(data)

    assert "## Incomplete Baseline" in markdown
    assert "trusted-baseline modified disposable-copy files" in markdown
    assert "N/A" in markdown


def test_package_write_failure_is_appended_before_rendering() -> None:
    collector = _collector_module()
    data = {"runs": []}
    record = {"label": "trusted-baseline", "integrity_ok": False}

    collector._append_failure_run(data, collector.PackageWriteError("copy write detected", record))

    assert data["runs"] == [record]
    assert data["failure"] == "copy write detected"


def test_write_artifacts_replaces_stale_success_with_failure_atomically(tmp_path) -> None:
    collector = _collector_module()
    json_path = tmp_path / "baseline.json"
    markdown_path = tmp_path / "baseline.md"
    json_path.write_text('{"outcome": "pass"}\n')
    markdown_path.write_text("# Current Quality Baseline\n\npass\n")
    failure = {
        "generated_at": "2026-08-12T00:00:00+00:00",
        "source": {
            "commit": "abc",
            "dirty": True,
            "tracked_diff_sha256": "diff",
            "untracked_manifest_sha256": "untracked",
        },
        "copy_manifest_sha256": "manifest",
        "environment": {"python": "test"},
        "outcome": "fail",
        "failure": "copy write detected",
        "runs": [],
    }

    collector._write_artifacts(failure, json_path, markdown_path)

    assert '"outcome": "fail"' in json_path.read_text()
    assert "copy write detected" in markdown_path.read_text()


def test_manifest_excludes_output_artifacts_and_dirty_provenance_is_hashed(tmp_path) -> None:
    collector = _collector_module()
    (tmp_path / "fincore.py").write_text("value = 1\n")
    output = tmp_path / "docs" / "quality" / "baseline.json"
    output.parent.mkdir(parents=True)
    output.write_text("first\n")
    excluded = {Path("docs/quality/baseline.json")}

    first = collector._copy_manifest_sha256(collector._inventory(tmp_path, excluded))
    output.write_text("second\n")
    second = collector._copy_manifest_sha256(collector._inventory(tmp_path, excluded))

    assert first == second


def test_nonserial_xdist_inherits_discovery_count_with_explicit_source() -> None:
    collector = _collector_module()
    single = {"discovered": 2299, "selected": 2278, "passed": 2264, "skipped": 14}
    xdist = {"discovered": None, "selected": 2278, "passed": 2264, "skipped": 14}

    assert collector._normalize_nonserial_counts(single, xdist) is True
    assert xdist["discovered"] == 2299
    assert xdist["collection_source"] == "non-serial-single pytest collection"


def test_git_inclusion_manifest_and_copy_exclude_ignored_files(tmp_path) -> None:
    collector = _collector_module()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(".superpowers/\n")
    (tmp_path / "tracked.py").write_text("tracked = True\n")
    subprocess.run(["git", "add", ".gitignore", "tracked.py"], cwd=tmp_path, check=True)
    (tmp_path / "untracked.py").write_text("untracked = True\n")
    ignored = tmp_path / ".superpowers" / "sdd" / "artifact.md"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("ignored first\n")
    outputs = {Path("output.json")}

    first = collector._source_file_manifest(tmp_path, outputs)
    ignored.write_text("ignored second\n")
    second = collector._source_file_manifest(tmp_path, outputs)
    copy_root = tmp_path / "copy"
    collector._copy_source_tree(tmp_path, copy_root, second)

    assert first == second
    assert ".superpowers/sdd/artifact.md" not in second
    assert (copy_root / "tracked.py").is_file()
    assert (copy_root / "untracked.py").is_file()
    assert not (copy_root / ".superpowers" / "sdd" / "artifact.md").exists()


def test_artifact_transaction_rolls_back_first_replace_when_second_fails(tmp_path, monkeypatch) -> None:
    collector = _collector_module()
    json_path = tmp_path / "baseline.json"
    markdown_path = tmp_path / "baseline.md"
    json_path.write_text('{"outcome": "pass"}\n')
    markdown_path.write_text("# Current Quality Baseline\n\npass\n")
    failure = {
        "generated_at": "2026-08-12T00:00:00+00:00",
        "source": {"commit": "abc", "dirty": True},
        "copy_manifest_sha256": "manifest",
        "environment": {"python": "test"},
        "outcome": "fail",
        "failure": "copy write detected",
        "runs": [],
    }
    original_replace = collector._replace_file
    calls = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("second replace failed")
        original_replace(source, destination)

    monkeypatch.setattr(collector, "_replace_file", fail_second_replace)

    with pytest.raises(OSError, match="second replace failed"):
        collector._write_artifacts(failure, json_path, markdown_path)

    assert json_path.read_text() == '{"outcome": "pass"}\n'
    assert markdown_path.read_text() == "# Current Quality Baseline\n\npass\n"


def test_timeout_output_normalizes_bytes() -> None:
    collector = _collector_module()
    error = subprocess.TimeoutExpired(["pytest"], 1, output=b"partial stdout", stderr=b"partial stderr")

    assert collector._timeout_output(error) == "partial stdoutpartial stderr"


def test_disposable_baseline_copy_allows_generated_egg_info_but_not_source_files() -> None:
    collector = _collector_module()

    assert collector._is_excluded(Path("fincore.egg-info/PKG-INFO"))
    assert not collector._is_excluded(Path("fincore/alphalens/utils.py"))


def test_disposable_baseline_copy_has_an_isolated_git_head(tmp_path) -> None:
    collector = _collector_module()
    (tmp_path / "tracked.py").write_text("value = 1\n", encoding="utf-8")

    collector._initialize_copy_git_repository(tmp_path)

    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert revision.returncode == 0, revision.stdout + revision.stderr
    assert len(revision.stdout.strip()) == 40


def test_disposable_copy_preserves_source_history_for_reproducibility_fixtures(tmp_path) -> None:
    collector = _collector_module()
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    (source / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (source / "ignored").mkdir()
    (source / "ignored" / "provenance.json").write_text('{"historical": true}\n', encoding="utf-8")
    (source / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", ".gitignore", "tracked.py", "ignored/provenance.json"], cwd=source, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "first"],
        cwd=source,
        check=True,
    )
    first_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True, check=True
    ).stdout.strip()
    (source / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=source, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "second"],
        cwd=source,
        check=True,
    )
    source_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest = collector._source_file_manifest(source, set())
    copy_root = tmp_path / "copy"

    collector._prepare_disposable_copy(source, copy_root, manifest)

    copied_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=copy_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    history = subprocess.run(
        ["git", "cat-file", "-e", f"{first_head}^{{commit}}"], cwd=copy_root, capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=copy_root, capture_output=True, text=True, check=True
    ).stdout
    tracked_ignored = subprocess.run(
        ["git", "ls-tree", "--name-only", "HEAD", "ignored/provenance.json"],
        cwd=copy_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert copied_head == source_head
    assert history.returncode == 0, history.stderr
    assert status == ""
    assert tracked_ignored == "ignored/provenance.json\n"


def test_disposable_copies_are_fresh_between_quality_runs(tmp_path) -> None:
    """One baseline run cannot leave source state for the next run to inherit."""

    collector = _collector_module()
    source = tmp_path / "source"
    source.mkdir()
    tracked = source / "project.toml"
    tracked.write_text("name = 'fincore'\n", encoding="utf-8")
    manifest = {"project.toml": "test-hash"}

    first = tmp_path / "first-run"
    collector._prepare_disposable_copy(source, first, manifest)
    (first / "project.toml").write_text("poisoned = true\n", encoding="utf-8")

    second = tmp_path / "second-run"
    collector._prepare_disposable_copy(source, second, manifest)

    assert (second / "project.toml").read_text(encoding="utf-8") == "name = 'fincore'\n"


def test_baseline_environment_forces_the_headless_matplotlib_backend(monkeypatch) -> None:
    """Quality collection cannot inherit a macOS GUI backend from the shell."""

    collector = _collector_module()
    monkeypatch.setenv("MPLBACKEND", "macosx")

    environment = collector._baseline_environment()

    assert environment["MPLBACKEND"] == "Agg"


def test_branch_coverage_binds_to_the_disposable_copy_not_a_relative_child_cwd(tmp_path) -> None:
    collector = _collector_module()
    copy_root = tmp_path / "copy"
    (copy_root / "fincore").mkdir(parents=True)

    args = collector._coverage_pytest_args(
        copy_root,
        ["--ignore=tests/benchmarks", "--cov=fincore", "--cov-branch"],
        coverage=True,
    )

    assert "--cov=fincore" not in args
    assert f"--cov={(copy_root / 'fincore').resolve()}" in args


def test_non_coverage_runs_do_not_rewrite_pytest_arguments(tmp_path) -> None:
    collector = _collector_module()
    original = ["--ignore=tests/benchmarks", "-m", "not slow"]

    assert collector._coverage_pytest_args(tmp_path, original, coverage=False) == original


def test_branch_coverage_requires_its_single_explicit_target(tmp_path) -> None:
    collector = _collector_module()

    with pytest.raises(RuntimeError, match="exactly one --cov=fincore"):
        collector._coverage_pytest_args(tmp_path, ["--cov=elsewhere"], coverage=True)


def test_full_quality_runs_have_a_thirty_minute_timeout_budget() -> None:
    """Coverage has a larger, explicit budget than normal subprocess probes."""

    collector = _collector_module()

    assert collector.COMMAND_TIMEOUT_SECONDS >= 30 * 60
