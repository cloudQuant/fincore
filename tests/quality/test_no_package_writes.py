import pandas as pd
import pytest

from fincore.utils import common_utils


def test_run_flask_display_does_not_implicitly_export(monkeypatch) -> None:
    def forbidden_export(*_args, **_kwargs):
        pytest.fail("run_flask_app attempted to write an XLSX into the package")

    monkeypatch.setattr(pd.DataFrame, "to_excel", forbidden_export)
    monkeypatch.setattr(common_utils, "display", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(common_utils, "HTML", lambda html: html)

    common_utils.print_table(
        pd.DataFrame({"x": [1.0]}),
        name="Stress Events",
        run_flask_app=True,
    )
