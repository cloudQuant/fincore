from __future__ import annotations

import pandas as pd
import pytest

from fincore.core.context import AnalysisContext
from fincore.report.artifacts import ReportArtifacts


def test_close_deduplicates_axes_owned_by_the_same_figure(monkeypatch) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2)
    closed = []
    monkeypatch.setattr(plt, "close", lambda item: closed.append(item))

    ReportArtifacts(backend="matplotlib", figures=[axes[0], axes[1], axes[0]]).close()

    assert closed == [figure]


def test_close_attempts_all_resources_then_raises_the_first_error() -> None:
    events = []

    class Broken:
        def close(self):
            events.append("broken")
            raise RuntimeError("cannot close")

    class Healthy:
        def close(self):
            events.append("healthy")

    with pytest.raises(RuntimeError, match="cannot close"):
        ReportArtifacts(backend="test", figures=[Broken(), Healthy()]).close()

    assert events == ["broken", "healthy"]


def test_close_is_idempotent_after_all_resources_are_released() -> None:
    events = []

    class Resource:
        def close(self):
            events.append("closed")

    artifacts = ReportArtifacts(backend="test", figures=[Resource()])

    artifacts.close()
    artifacts.close()

    assert events == ["closed"]
    assert artifacts.closed is True


def test_close_retries_only_resources_that_failed() -> None:
    events = []

    class Flaky:
        attempts = 0

        def close(self):
            self.attempts += 1
            events.append(f"flaky-{self.attempts}")
            if self.attempts == 1:
                raise RuntimeError("try again")

    class Healthy:
        def close(self):
            events.append("healthy")

    artifacts = ReportArtifacts(backend="test", figures=[Flaky(), Healthy()])

    with pytest.raises(RuntimeError, match="try again"):
        artifacts.close()
    assert artifacts.closed is False

    artifacts.close()

    assert events == ["flaky-1", "healthy", "flaky-2"]
    assert artifacts.closed is True


def test_artifacts_context_manager_closes_owned_resources() -> None:
    events = []

    class Resource:
        def close(self):
            events.append("closed")

    with ReportArtifacts(backend="test", figures=[Resource()]) as artifacts:
        assert artifacts.backend == "test"

    assert events == ["closed"]


def test_html_context_artifacts_have_stable_lifecycle() -> None:
    returns = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2024-01-01", periods=3))

    artifacts = AnalysisContext(returns).plot(backend="html")

    assert artifacts.html is not None
    assert "Cumulative Returns" in artifacts.html
    artifacts.close()
