from __future__ import annotations

import importlib.util
from pathlib import Path


def _collector_module():
    path = Path(__file__).parents[2] / "scripts" / "collect_quality_baseline.py"
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
