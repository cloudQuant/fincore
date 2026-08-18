"""Data snapshot contract tests."""

from __future__ import annotations

import pandas as pd

from fincore.data.snapshots import SCHEMA_VERSION, DataSnapshot


def test_snapshot_hash_is_stable_and_excludes_secret_configuration() -> None:
    snapshot = DataSnapshot.from_frame(
        frame=pd.DataFrame({"close": [10.0]}),
        provider="fixture",
        requested_start="2024-01-01",
        requested_end="2024-01-02",
        as_of="2024-01-03T00:00:00Z",
    )

    rebuilt = DataSnapshot.from_frame(snapshot.data, **snapshot.identity_kwargs())

    assert snapshot.content_sha256 == rebuilt.content_sha256
    assert "api_key" not in snapshot.to_manifest()


def test_snapshot_data_is_a_defensive_copy() -> None:
    frame = pd.DataFrame({"close": [10.0, 11.0]})
    snapshot = DataSnapshot.from_frame(
        frame=frame,
        provider="fixture",
        requested_start="2024-01-01",
        requested_end="2024-01-02",
        as_of="2024-01-03T00:00:00Z",
    )
    frame.loc[0, "close"] = 999.0

    assert snapshot.data.loc[0, "close"] == 10.0


def test_snapshot_hash_changes_with_data_content() -> None:
    a = DataSnapshot.from_frame(
        frame=pd.DataFrame({"close": [10.0]}),
        provider="fixture",
        requested_start="2024-01-01",
        requested_end="2024-01-02",
        as_of="2024-01-03T00:00:00Z",
    )
    b = DataSnapshot.from_frame(
        frame=pd.DataFrame({"close": [11.0]}),
        provider="fixture",
        requested_start="2024-01-01",
        requested_end="2024-01-02",
        as_of="2024-01-03T00:00:00Z",
    )

    assert a.content_sha256 != b.content_sha256


def test_snapshot_manifest_carries_provenance_not_data() -> None:
    snapshot = DataSnapshot.from_frame(
        frame=pd.DataFrame({"close": [10.0]}),
        provider="fixture",
        requested_start="2024-01-01",
        requested_end="2024-01-02",
        as_of="2024-01-03T00:00:00Z",
        price_adjustment="qfq",
    )

    manifest = snapshot.to_manifest()

    assert manifest["provider"] == "fixture"
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["price_adjustment"] == "qfq"
    assert len(manifest["content_sha256"]) == 64
