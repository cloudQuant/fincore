"""Plugin discovery and overlay tests."""

from __future__ import annotations

from fincore.plugin.discovery import ENTRY_POINT_GROUPS, discover_plugins


def test_discover_plugins_returns_immutable_records() -> None:
    plugins = discover_plugins()
    assert isinstance(plugins, tuple)
    for plugin in plugins:
        assert plugin.name
        assert plugin.group in ENTRY_POINT_GROUPS
        assert plugin.distribution
        assert plugin.value


def test_discovery_does_not_import_third_party_code() -> None:
    import sys

    before = set(sys.modules)
    discover_plugins()
    after = set(sys.modules)
    # Discovery must not pull in any new heavy module (matplotlib, yfinance, ...).
    heavy = {"matplotlib", "yfinance", "akshare", "tushare"}
    assert not (heavy & (after - before))


def test_entry_point_groups_are_documented() -> None:
    assert ENTRY_POINT_GROUPS == ("fincore.metrics", "fincore.providers", "fincore.renderers", "fincore.exporters")
