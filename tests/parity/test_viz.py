"""Resource-ownership behavior retained for the future unified viz package."""

from __future__ import annotations

import matplotlib.pyplot as plt


def test_owned_figures_are_shown_and_closed_without_touching_caller_figures(monkeypatch) -> None:
    """Only the workflow's artifact figures are displayed and released."""
    from fincore.factor_analysis.tears import FactorTearSheetArtifacts, close_owned_figures, show_owned_figures

    caller_figure = plt.figure()
    first_owned = plt.figure()
    second_owned = plt.figure()
    artifacts = FactorTearSheetArtifacts(
        model=None,  # type: ignore[arg-type]
        figures=(first_owned, second_owned),
        tables={},
    )
    show_calls: list[None] = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(None))

    show_owned_figures(artifacts)
    close_owned_figures(artifacts)

    assert len(show_calls) == 2
    assert plt.fignum_exists(caller_figure.number)
    assert not plt.fignum_exists(first_owned.number)
    assert not plt.fignum_exists(second_owned.number)
    plt.close(caller_figure)
