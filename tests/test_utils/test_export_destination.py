"""Task 8 export-destination contract for print_table.

Exports happen only when the caller supplies an explicit ``ExportConfig``;
the destination is always caller-owned.  Default (and legacy
``run_flask_app``) invocations never write files.
"""

from __future__ import annotations

import pandas as pd

from fincore.utils import ExportConfig, print_table


def test_table_export_requires_explicit_destination(tmp_path) -> None:
    result = print_table(
        pd.DataFrame({"x": [1]}),
        name="test",
        export=ExportConfig(output_dir=tmp_path),
    )

    assert result is not None
    assert result.files == [tmp_path / "strategy_performance_test.xlsx"]
    assert (tmp_path / "strategy_performance_test.xlsx").is_file()

    content = pd.read_excel(tmp_path / "strategy_performance_test.xlsx", index_col=0)
    assert list(content.columns) == ["x"]


def test_table_export_honors_custom_filename(tmp_path) -> None:
    result = print_table(
        pd.DataFrame({"x": [1]}),
        name="ignored",
        export=ExportConfig(output_dir=tmp_path, filename="custom.xlsx"),
    )

    assert result.files == [tmp_path / "custom.xlsx"]
    assert (tmp_path / "custom.xlsx").is_file()


def test_table_export_creates_missing_destination_dir(tmp_path) -> None:
    target = tmp_path / "nested" / "exports"
    result = print_table(
        pd.DataFrame({"x": [1]}),
        name="deep",
        export=ExportConfig(output_dir=target),
    )

    assert result.files == [target / "strategy_performance_deep.xlsx"]
    assert (target / "strategy_performance_deep.xlsx").is_file()


def test_default_print_table_writes_nothing_anywhere(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = print_table(pd.DataFrame({"x": [1]}), name="silent", run_flask_app=True)

    assert result is None
    assert list(tmp_path.iterdir()) == []
