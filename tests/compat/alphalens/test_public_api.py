"""C0 public-surface checks against the pinned Alphalens manifest."""

from __future__ import annotations

import importlib
import inspect
import json
import typing
from copy import deepcopy
from dataclasses import FrozenInstanceError
from typing import Any

import pandas as pd
import pytest

from . import conftest as fixture_contract
from .conftest import load_pinned_manifest, manifest_entries


def _entry_id(entry: dict[str, Any]) -> str:
    return f"{entry['module']}:{entry['symbol']}"


@pytest.mark.parametrize("entry", manifest_entries(), ids=_entry_id)
def test_frozen_definition_resolves(entry: dict[str, Any]) -> None:
    """Every C0 manifest definition resolves from its declared module."""

    module = importlib.import_module(f"fincore.alphalens.{entry['module']}")
    value = getattr(module, str(entry["symbol"]))
    assert callable(value)


def test_manifest_definition_count_resolves_without_flattening_root_api() -> None:
    """The 64 frozen definitions remain module-scoped rather than root aliases."""

    import fincore
    from fincore import alphalens

    manifest = manifest_entries()
    assert len(manifest) == 64
    assert sum(entry["kind"] == "function" for entry in manifest) == 61
    assert sum(entry["kind"] == "class" for entry in manifest) == 3
    assert fincore.alphalens is alphalens
    assert "alphalens" in fincore.__all__
    for module_name in ("performance", "plotting", "tears", "utils"):
        assert getattr(alphalens, module_name) is importlib.import_module(f"fincore.alphalens.{module_name}")
    assert not hasattr(fincore, "quantize_factor")
    assert not hasattr(fincore, "plot_ic_ts")


def test_facade_modules_export_only_their_pinned_public_definitions() -> None:
    """Implementation helpers do not become accidental compatibility names."""

    for module_name in ("performance", "plotting", "tears", "utils"):
        module = importlib.import_module(f"fincore.alphalens.{module_name}")
        expected_names = tuple(entry["symbol"] for entry in manifest_entries() if entry["module"] == module_name)
        assert tuple(module.__all__) == expected_names
        assert not hasattr(module, "export_deferred_functions")


def test_static_contract_registry_keeps_source_and_introspection_facts_separate() -> None:
    """The facade registry is checked-in code, not a runtime fixture reader."""

    from fincore.contracts.factor_analysis import ALPHALENS_FUNCTION_SPECS, FactorFunctionSpec
    from fincore.contracts.factor_workflows import ALPHALENS_WORKFLOW_SPECS, FactorWorkflowSpec

    assert len(ALPHALENS_FUNCTION_SPECS) == 61
    assert len(ALPHALENS_WORKFLOW_SPECS) == 7
    quantize = ALPHALENS_FUNCTION_SPECS[("utils", "quantize_factor")]
    assert isinstance(quantize, FactorFunctionSpec)
    assert str(quantize.introspection_signature) == "(*args, **kwargs)"
    assert str(quantize.source_signature) == (
        "(factor_data, quantiles=5, bins=None, by_group=False, no_raise=False, zero_aware=False)"
    )
    assert all(isinstance(spec, FactorWorkflowSpec) for spec in ALPHALENS_WORKFLOW_SPECS.values())
    with pytest.raises(FrozenInstanceError):
        quantize.module = "changed"  # type: ignore[misc]


def test_factor_contract_annotations_resolve_at_runtime() -> None:
    """Public frozen contract annotations are usable through runtime reflection."""

    from fincore.contracts.factor_analysis import FactorFunctionSpec
    from fincore.contracts.factor_workflows import FactorWorkflowSpec

    function_hints = typing.get_type_hints(FactorFunctionSpec)
    workflow_hints = typing.get_type_hints(FactorWorkflowSpec)

    assert function_hints["source_signature"] is inspect.Signature
    assert workflow_hints["source_signature"] is inspect.Signature


def test_workflow_specs_match_the_pinned_tear_sheet_manifest() -> None:
    """The seven lifecycle contracts retain the exact frozen public workflow shape."""

    from fincore.contracts.factor_workflows import ALPHALENS_WORKFLOW_SPECS

    entries = tuple(entry for entry in manifest_entries() if entry["module"] == "tears" and entry["kind"] == "function")
    expected_names = tuple(str(entry["symbol"]) for entry in entries)

    assert tuple(ALPHALENS_WORKFLOW_SPECS) == expected_names
    for entry in entries:
        name = str(entry["symbol"])
        spec = ALPHALENS_WORKFLOW_SPECS[name]
        assert str(spec.source_signature) == entry["source_signature"]
        assert str(spec.introspection_signature) == entry["introspection_signature"]
        assert spec.model_ref == f"fincore.factor_analysis.tears:{name}"
        assert spec.renderer_ref == f"fincore.factor_analysis.render_matplotlib:{name}"
        assert spec.optional_extra == "alphalens"
        expected_variants = (
            ("by_group=False:show-close", "by_group=True:show-close")
            if "by_group" in spec.source_signature.parameters
            else ()
        )
        assert spec.by_group_variants == expected_variants


def test_optional_extras_are_named_for_the_alphalens_surface() -> None:
    """Strict plotting/tear contracts point at their own optional-extra boundary."""

    from fincore.contracts.factor_analysis import ALPHALENS_FUNCTION_SPECS
    from fincore.contracts.factor_workflows import ALPHALENS_WORKFLOW_SPECS

    for spec in ALPHALENS_FUNCTION_SPECS.values():
        expected_extra = (
            "factor-analysis"
            if (spec.module, spec.public_name) == ("performance", "factor_alpha_beta")
            else "alphalens"
            if spec.module in {"plotting", "tears"}
            else None
        )
        assert spec.optional_extra == expected_extra
    assert {spec.optional_extra for spec in ALPHALENS_WORKFLOW_SPECS.values()} == {"alphalens"}


def test_manifest_helpers_do_not_share_mutable_cached_entries() -> None:
    """Nested manifest mutation cannot corrupt later C0/C1 expectations."""

    first_manifest = load_pinned_manifest()
    first_entries = manifest_entries()
    first_manifest["counts"]["functions"] = -1
    first_entries[0]["parameters"][0]["name"] = "corrupted_parameter"
    case_entry = next(entry for entry in first_entries if entry["accepted_call_cases"])
    case_entry["accepted_call_cases"][0]["hidden_kwargs"]["synthetic_hidden_key"] = "corrupted_hidden_value"

    second_manifest = load_pinned_manifest()
    second_entries = manifest_entries()
    second_case_entry = next(entry for entry in second_entries if entry["symbol"] == case_entry["symbol"])

    assert second_manifest["counts"]["functions"] == 61
    assert second_entries[0]["parameters"][0]["name"] == "factor_data"
    assert second_case_entry["accepted_call_cases"][0]["hidden_kwargs"] == {}


def test_grid_figure_and_legacy_exceptions_resolve() -> None:
    """The non-function C0 names have stable, import-safe definitions."""

    from fincore.alphalens.tears import GridFigure
    from fincore.alphalens.utils import MaxLossExceededError, NonMatchingTimezoneError

    assert str(inspect.signature(GridFigure)) == "(rows, cols)"
    assert isinstance(MaxLossExceededError(), MaxLossExceededError)
    assert isinstance(NonMatchingTimezoneError(), NonMatchingTimezoneError)


def test_shared_synthetic_fixture_contract(
    raw_factor: pd.Series,
    prices: pd.DataFrame,
    tz_aware_prices: pd.DataFrame,
    groups: pd.Series,
) -> None:
    """The future Tasks 3–8 fixture inputs stay deterministic and non-overlapping."""

    assert raw_factor.name == "factor"
    assert raw_factor.index.names == ["date", "asset"]
    assert len(raw_factor) == 120 * 10
    pd.testing.assert_index_equal(
        raw_factor.index.get_level_values("date").unique(),
        pd.bdate_range("2024-01-02", periods=120, name="date"),
    )
    assert prices.shape == (120, 12)
    assert prices.index.tz is None
    assert tz_aware_prices.shape == prices.shape
    assert str(tz_aware_prices.index.tz) == "UTC"
    assert set(raw_factor.index.get_level_values("asset")) < set(prices.columns)
    assert groups.index.name == "asset"
    assert groups.to_dict() == {
        f"asset_{ordinal:02d}": "sector_a" if ordinal % 2 == 0 else "sector_b" for ordinal in range(10)
    }


@pytest.mark.parametrize("fixture_name", ("raw_factor", "prices", "tz_aware_prices", "groups"))
def test_shared_fixture_tables_round_trip_through_portable_json(
    request: pytest.FixtureRequest, fixture_name: str
) -> None:
    """Every shared input survives a JSON-safe table/metadata round trip exactly."""

    original = request.getfixturevalue(fixture_name)
    payload = fixture_contract.serialize_factor_fixture_table(original)
    restored = fixture_contract.deserialize_factor_fixture_table(
        json.loads(json.dumps(payload, allow_nan=False, sort_keys=True))
    )

    assert payload["schema_version"] == "fincore-factor-fixture-table-v1"
    assert payload["kind"] in {"series", "dataframe"}
    assert "index" in payload
    assert "data" in payload
    if isinstance(original, pd.Series):
        assert isinstance(restored, pd.Series)
        pd.testing.assert_series_equal(original, restored)
    else:
        assert isinstance(original, pd.DataFrame)
        assert isinstance(restored, pd.DataFrame)
        pd.testing.assert_frame_equal(original, restored)


def test_shared_fixture_table_round_trip_preserves_explicit_nan_mask(raw_factor: pd.Series) -> None:
    """Missing values use a portable mask instead of non-standard JSON NaN values."""

    original = raw_factor.copy()
    original.iloc[0] = float("nan")
    payload = fixture_contract.serialize_factor_fixture_table(original)
    restored = fixture_contract.deserialize_factor_fixture_table(json.loads(json.dumps(payload, allow_nan=False)))

    assert payload["data"]["nan_mask"][0] is True
    assert isinstance(restored, pd.Series)
    pd.testing.assert_series_equal(original, restored)


@pytest.mark.parametrize(
    "original",
    (
        pd.Series(
            [float("nan"), float("inf"), float("-inf")],
            index=pd.Index([float("nan"), float("inf"), float("-inf")], name="numeric_index"),
            name="nonfinite_series",
        ),
        pd.DataFrame(
            {
                "left": [float("nan"), float("inf"), float("-inf")],
                "right": [float("inf"), float("-inf"), float("nan")],
            },
            index=pd.Index(["nan", "positive", "negative"], name="row"),
        ),
        pd.Series(
            [float("nan"), float("inf"), float("-inf")],
            index=pd.MultiIndex.from_tuples(
                [(float("nan"), "nan"), (float("inf"), "positive"), (float("-inf"), "negative")],
                names=("numeric_level", "label"),
            ),
            name="nonfinite_multiindex_series",
        ),
    ),
    ids=("series", "dataframe", "multiindex"),
)
def test_shared_fixture_table_round_trip_preserves_nonfinite_values(
    original: pd.Series | pd.DataFrame,
) -> None:
    """NaN, positive infinity, and negative infinity have distinct portable encodings."""

    payload = fixture_contract.serialize_factor_fixture_table(original)
    restored = fixture_contract.deserialize_factor_fixture_table(json.loads(json.dumps(payload, allow_nan=False)))

    assert "nonfinite" in payload["data"]
    if isinstance(original, pd.Series):
        assert isinstance(restored, pd.Series)
        assert payload["data"]["nonfinite"] == [None, "positive_infinity", "negative_infinity"]
        pd.testing.assert_series_equal(original, restored)
    else:
        assert isinstance(restored, pd.DataFrame)
        assert payload["data"]["nonfinite"] == [
            [None, "positive_infinity"],
            ["positive_infinity", "negative_infinity"],
            ["negative_infinity", None],
        ]
        pd.testing.assert_frame_equal(original, restored)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload["data"].__setitem__("nonfinite", ["positive_infinity"]),
        lambda payload: payload["data"]["nonfinite"].__setitem__(1, "unknown_nonfinite_tag"),
        lambda payload: payload["data"]["nan_mask"].__setitem__(1, True),
        lambda payload: payload["data"]["values"].__setitem__(1, 1.0),
    ),
    ids=("wrong_length", "unknown_tag", "nan_and_nonfinite", "tagged_value_present"),
)
def test_shared_fixture_table_rejects_malformed_nonfinite_metadata(mutate: Any) -> None:
    """The v1 decoder fails closed for nonfinite metadata corruption."""

    original = pd.Series([0.0, float("inf"), 1.0], name="nonfinite_validation")
    payload = deepcopy(fixture_contract.serialize_factor_fixture_table(original))
    mutate(payload)

    with pytest.raises(ValueError, match="nonfinite"):
        fixture_contract.deserialize_factor_fixture_table(payload)


def test_shared_fixture_table_rejects_nonfinite_tag_coerced_to_string_series() -> None:
    """A signed-infinity tag cannot silently become a string extension scalar."""

    payload = deepcopy(fixture_contract.serialize_factor_fixture_table(pd.Series(["finite", "text"], dtype="string")))
    payload["data"]["values"][0] = None
    payload["data"]["nonfinite"][0] = "positive_infinity"

    with pytest.raises(ValueError, match="nonfinite.*numeric"):
        fixture_contract.deserialize_factor_fixture_table(payload)


def test_shared_fixture_table_rejects_nonfinite_tag_coerced_to_string_dataframe() -> None:
    """Matrix restoration applies the same signed-infinity dtype validation."""

    payload = deepcopy(
        fixture_contract.serialize_factor_fixture_table(
            pd.DataFrame({"label": pd.Series(["finite", "text"], dtype="string")})
        )
    )
    payload["data"]["values"][0][0] = None
    payload["data"]["nonfinite"][0][0] = "negative_infinity"

    with pytest.raises(ValueError, match="nonfinite.*numeric"):
        fixture_contract.deserialize_factor_fixture_table(payload)


def test_shared_fixture_table_rejects_nonfinite_tag_coerced_to_string_index() -> None:
    """Index reconstruction rejects a tag that a string dtype would rewrite."""

    payload = deepcopy(
        fixture_contract.serialize_factor_fixture_table(
            pd.Series([1.0, 2.0], index=pd.Index(["first", "second"], dtype="string", name="labels"))
        )
    )
    payload["index"]["values"][0] = None
    payload["index"]["nonfinite"][0] = "positive_infinity"

    with pytest.raises(ValueError, match="nonfinite.*numeric"):
        fixture_contract.deserialize_factor_fixture_table(payload)


def test_enhanced_fixture_conftest_reexports_portable_table_helpers(raw_factor: pd.Series) -> None:
    """Future enhanced tests consume the exact same portable fixture contract."""

    from tests.test_factor_analysis import conftest as enhanced_conftest

    payload = enhanced_conftest.serialize_factor_fixture_table(raw_factor)
    restored = enhanced_conftest.deserialize_factor_fixture_table(json.loads(json.dumps(payload, allow_nan=False)))

    assert callable(enhanced_conftest.serialize_factor_fixture_table)
    assert callable(enhanced_conftest.deserialize_factor_fixture_table)
    assert isinstance(restored, pd.Series)
    pd.testing.assert_series_equal(raw_factor, restored)


def test_clean_factor_data_fixture_is_an_explicit_task_3_boundary(request: pytest.FixtureRequest) -> None:
    """Task 2 does not pretend to have a cleaned factor-data implementation."""

    with pytest.raises(RuntimeError, match="deferred until Task 3"):
        request.getfixturevalue("clean_factor_data")
