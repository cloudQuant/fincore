"""Contracts for explicit extension discovery without a global plugin registry."""

from __future__ import annotations

import sys

from fincore.extensions.discovery import ENTRY_POINT_GROUPS, discover_extensions


def test_discovery_returns_immutable_records() -> None:
    extensions = discover_extensions()

    assert isinstance(extensions, tuple)
    for extension in extensions:
        assert extension.name
        assert extension.group in ENTRY_POINT_GROUPS
        assert extension.distribution
        assert extension.value


def test_discovery_does_not_import_extension_code_or_heavy_dependencies() -> None:
    before = set(sys.modules)
    discover_extensions()
    after = set(sys.modules)

    assert not ({"matplotlib", "yfinance", "akshare", "tushare"} & (after - before))


def test_entry_point_group_has_one_canonical_extension_namespace() -> None:
    assert ENTRY_POINT_GROUPS == ("fincore.extensions",)
