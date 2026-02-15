import fincore.tearsheets.utils as tu


def test_plotting_context_returns_context() -> None:
    ctx = tu.plotting_context(context="notebook", font_scale=1.0, rc={"lines.linewidth": 2.0})
    # seaborn returns a dict-like context object.
    assert isinstance(ctx, dict)
    assert ctx["lines.linewidth"] == 2.0


def test_axes_style_returns_style() -> None:
    style = tu.axes_style(style="darkgrid", rc={"axes.grid": False})
    assert isinstance(style, dict)
    assert style["axes.grid"] is False
