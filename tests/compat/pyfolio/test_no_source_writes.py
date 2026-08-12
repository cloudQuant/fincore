from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from .conftest import hash_tracked_package_files

if TYPE_CHECKING:
    import pandas as pd


def test_compat_workflow_never_writes_inside_package(workflow_returns: pd.Series) -> None:
    from fincore import pyfolio

    before = hash_tracked_package_files()
    figure = pyfolio.create_returns_tear_sheet(workflow_returns, run_flask_app=True)
    after = hash_tracked_package_files()
    plt.close(figure)
    assert after == before
