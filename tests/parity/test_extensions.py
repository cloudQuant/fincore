"""Canonical extension-snapshot scenarios for source and wheel evidence."""

from __future__ import annotations

import pytest

from fincore.extensions.snapshot import ExtensionHook, ExtensionSnapshot, RendererRegistration
from fincore.viz.base import get_backend


class _RecordingBackend:
    def __init__(self, theme: str = "light") -> None:
        self.theme = theme


def _increment(value: int) -> int:
    return value + 1


def _double(value: int) -> int:
    return value * 2


def test_renderer_registration_is_snapshot_local_and_resolved_explicitly() -> None:
    initial = ExtensionSnapshot()
    snapshot = initial.with_renderer(RendererRegistration(name="recording", renderer=_RecordingBackend))

    assert initial.renderers == ()
    assert initial.renderer("recording") is None
    assert snapshot.renderer("recording") is _RecordingBackend
    assert isinstance(get_backend("recording", extension_snapshot=snapshot), _RecordingBackend)


def test_extension_hooks_are_immutable_priority_ordered_transforms() -> None:
    snapshot = ExtensionSnapshot(
        hooks=(
            ExtensionHook(event="normalize", callable=_increment, priority=200),
            ExtensionHook(event="normalize", callable=_double, priority=10),
        )
    )

    assert [hook.fingerprint for hook in snapshot.hooks_for("normalize")] == [
        f"{__name__}:_double",
        f"{__name__}:_increment",
    ]
    assert snapshot.execute_hooks("normalize", 3) == 7
    assert snapshot.execute_hooks("missing", 3) == 3


def test_extension_snapshot_rejects_duplicate_renderer_names_and_invalid_viz_snapshot() -> None:
    with pytest.raises(ValueError, match="duplicate renderer"):
        ExtensionSnapshot(
            renderers=(
                RendererRegistration(name="recording", renderer=_RecordingBackend),
                RendererRegistration(name="recording", renderer=_RecordingBackend),
            )
        )

    with pytest.raises(TypeError, match="renderer"):
        get_backend("matplotlib", extension_snapshot=object())
