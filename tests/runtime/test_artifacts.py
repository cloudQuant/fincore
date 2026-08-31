"""Contracts for explicit artifact resource ownership."""

from __future__ import annotations

import pytest


def test_bundle_closes_only_fincore_owned_resources_and_is_idempotent() -> None:
    from fincore.runtime.artifacts import ArtifactBundle

    class Resource:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    owned = Resource()
    caller_owned = Resource()
    bundle = ArtifactBundle()
    bundle.add(owned, owned=True)
    bundle.add(caller_owned, owned=False)
    bundle.add(owned, owned=True)

    bundle.close()
    bundle.close()

    assert bundle.closed is True
    assert owned.close_calls == 1
    assert caller_owned.close_calls == 0
    assert bundle.artifacts == (owned, caller_owned, owned)


def test_bundle_records_close_failure_once_and_rejects_new_resources_after_close() -> None:
    from fincore.runtime.artifacts import ArtifactBundle

    class FailingResource:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise OSError("renderer failed to release")

    resource = FailingResource()
    bundle = ArtifactBundle()
    bundle.add(resource, owned=True)

    with pytest.raises(OSError, match="renderer failed"):
        bundle.close()

    bundle.close()
    assert resource.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        bundle.add(object(), owned=False)
