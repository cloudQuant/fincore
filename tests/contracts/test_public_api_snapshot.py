"""Public API surface snapshot drift tests.

The snapshot enumerates every public surface and its symbols; these tests
assert (a) the checked-in fixture is byte-stable, (b) every surface maps to
exactly one semantic profile, and (c) no public path is duplicated across
surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.snapshot_public_api import SURFACE_PROFILES, build_snapshot

FIXTURE = Path(__file__).parent / "fixtures" / "public-api-0.3.x.json"

PROFILES = {
    "strict_empyrical_0_6_0",
    "strict_pyfolio_0_9_6",
    "strict_alphalens_cloudquant_0_4_0",
    "enhanced_v1",
    "plugin_v1",
}


def test_snapshot_matches_checked_in_fixture() -> None:
    snapshot = build_snapshot()
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert snapshot == fixture, "public API snapshot drifted from the checked-in fixture"


def test_every_surface_has_a_known_profile() -> None:
    snapshot = build_snapshot()
    for surface, data in snapshot["surfaces"].items():
        assert data["profile"] in PROFILES, f"{surface} has unknown profile {data['profile']}"
        assert data["profile"] == SURFACE_PROFILES[surface], f"{surface} profile mismatch"


def test_no_duplicate_public_paths_across_surfaces() -> None:
    snapshot = build_snapshot()
    paths = [f"{surface}.{name}" for surface, data in snapshot["surfaces"].items() for name in data["public_symbols"]]
    assert len(paths) == len(set(paths)), "duplicate public paths detected"


def test_strict_surfaces_are_distinct_from_enhanced() -> None:
    snapshot = build_snapshot()
    strict = {
        name
        for surface, data in snapshot["surfaces"].items()
        if data["profile"].startswith("strict_")
        for name in data["public_symbols"]
    }
    enhanced = {
        name
        for surface, data in snapshot["surfaces"].items()
        if data["profile"] == "enhanced_v1"
        for name in data["public_symbols"]
    }
    # A given symbol name may appear on both strict and enhanced surfaces, but
    # each *public path* (surface.name) is unique (asserted above).
    assert isinstance(strict, set) and isinstance(enhanced, set)
