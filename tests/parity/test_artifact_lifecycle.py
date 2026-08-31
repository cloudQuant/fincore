"""The runtime owns the only artifact lifecycle contract used by reports."""

from __future__ import annotations

import pytest


def test_named_artifacts_keep_explicit_ownership_and_metadata_immutable() -> None:
    from fincore.runtime import ArtifactBundle

    class Resource:
        def __init__(self) -> None:
            self.calls = 0

        def close(self) -> None:
            self.calls += 1

    resource = Resource()
    bundle = ArtifactBundle(metadata={"backend": "test"})
    bundle.add("<html/>", owned=False, name="html")
    bundle.add(resource, owned=True, name="figure")

    assert bundle.named_artifacts == {"html": "<html/>", "figure": resource}
    assert bundle.metadata["backend"] == "test"
    with pytest.raises(TypeError):
        bundle.metadata["backend"] = "mutated"  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicate artifact name"):
        bundle.add("duplicate", owned=False, name="html")

    bundle.close()
    bundle.close()
    assert resource.calls == 1


def test_report_rendering_returns_the_runtime_artifact_bundle_only(tmp_path) -> None:
    import pandas as pd

    from fincore.report.portfolio.compute import build_portfolio_report
    from fincore.report.renderers.html import write_html
    from fincore.runtime import ArtifactBundle

    returns = pd.Series([0.01, -0.002, 0.003], index=pd.date_range("2024-01-02", periods=3, freq="B"))
    bundle = write_html(build_portfolio_report(returns, rolling_window=2), tmp_path / "report.html")

    assert isinstance(bundle, ArtifactBundle)
    assert bundle.named_artifacts["file"].exists()
