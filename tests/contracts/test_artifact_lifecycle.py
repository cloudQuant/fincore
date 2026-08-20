"""Artifact lifecycle tests."""

from __future__ import annotations

from fincore.results.artifacts import IdempotentCloseMixin


class FakeArtifacts(IdempotentCloseMixin):
    def __init__(self) -> None:
        self.release_count = 0

    def _close_resources(self) -> None:
        self.release_count += 1


def test_close_is_idempotent() -> None:
    artifacts = FakeArtifacts()
    artifacts.close()
    artifacts.close()
    assert artifacts.release_count == 1
    assert artifacts.closed


def test_context_manager_closes() -> None:
    with FakeArtifacts() as artifacts:
        assert not artifacts.closed
    assert artifacts.closed
    assert artifacts.release_count == 1


def test_context_manager_closes_on_exception() -> None:
    artifacts = FakeArtifacts()
    try:
        with artifacts:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert artifacts.closed
    assert artifacts.release_count == 1
