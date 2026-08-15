"""Compute-once model contracts for the enhanced factor-analysis surface."""

from __future__ import annotations

import datetime as dt
import json
import os
import struct
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, is_dataclass
from decimal import Decimal
from typing import Any, get_type_hints
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import BDay, CustomBusinessDay

from fincore.factor_analysis import performance, portfolio
from fincore.factor_analysis.calendar import get_forward_returns_columns


def _require_named_zone(zone: str) -> None:
    """Make ``zoneinfo`` resolve named zones without a system IANA database.

    Windows has no ``/usr/share/zoneinfo``; when the ``tzdata`` wheel is
    installed, point ``zoneinfo`` at it explicitly. Skipped only when neither
    the system database nor ``tzdata`` is available.
    """
    import zoneinfo

    try:
        zoneinfo.ZoneInfo(zone)
        return
    except zoneinfo.ZoneInfoNotFoundError:
        pass
    try:
        from importlib import resources

        import tzdata

        zoneinfo.reset_tzpath([str(resources.files(tzdata) / "zoneinfo")])
        zoneinfo.ZoneInfo(zone)
    except (ImportError, zoneinfo.ZoneInfoNotFoundError) as error:
        pytest.skip(f"named IANA zone {zone!r} unavailable: {error}")


class _MutableGroupLabel:
    """Pickleable, hashable object used to prove object cells are owned."""

    def __init__(self, labels: list[str]) -> None:
        self.labels = labels

    def __hash__(self) -> int:
        return hash(_MutableGroupLabel)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _MutableGroupLabel) and self.labels == other.labels

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _MutableGroupLabel):
            return NotImplemented
        return tuple(self.labels) < tuple(other.labels)


class _BaseSlottedLabel:
    """Base state must participate in a subclass's deterministic fingerprint."""

    __slots__ = ("base_label",)

    def __init__(self, base_label: str) -> None:
        self.base_label = base_label


class _InheritedSlottedLabel(_BaseSlottedLabel):
    """Hashable group value with semantic state split across two slot classes."""

    __slots__ = ("label",)

    def __init__(self, base_label: str, label: str) -> None:
        super().__init__(base_label)
        self.label = label

    def __hash__(self) -> int:
        return hash(_InheritedSlottedLabel)


class _DictAndInheritedSlotLabel(_BaseSlottedLabel):
    """A subclass with a dict must not hide inherited slot state."""

    def __init__(self, base_label: str, label: str) -> None:
        super().__init__(base_label)
        self.label = label

    def __hash__(self) -> int:
        return hash(_DictAndInheritedSlotLabel)


class _IdentityOnlyKey:
    """A pickle clone cannot preserve this key's identity-only lookup behavior."""


def _only_periods(factor_data: pd.DataFrame, periods: tuple[str, ...]) -> pd.DataFrame:
    """Return a copied clean table with exactly the requested forward columns."""

    copied = factor_data.copy(deep=True)
    forward = get_forward_returns_columns(copied.columns)
    return copied.drop(columns=[column for column in forward if column not in periods])


def _event_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a real return matrix suitable for the event kernel."""

    return prices.pct_change(fill_method=None).fillna(0.0)


def _assert_serializable_data_only(value: object) -> None:
    """Reject renderer objects and executable cache payloads recursively."""

    assert not callable(value)
    qualified_name = f"{type(value).__module__}.{type(value).__qualname__}"
    assert "matplotlib" not in qualified_name
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _assert_serializable_data_only(getattr(value, item.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_serializable_data_only(key)
            _assert_serializable_data_only(item)
    elif isinstance(value, tuple):
        for item in value:
            _assert_serializable_data_only(item)


def test_factor_analysis_model_declares_every_renderer_required_field() -> None:
    """The model contract is explicit rather than a loose ``Mapping[str, Any]``."""

    from fincore.factor_analysis.models import FactorAnalysisModel

    names = {item.name for item in fields(FactorAnalysisModel)}
    assert {
        "config",
        "factor_data",
        "forward_periods",
        "quantile_statistics",
        "factor_weights",
        "factor_returns",
        "factor_cumulative_returns",
        "factor_positions",
        "alpha_beta",
        "mean_returns_by_quantile",
        "std_error_by_quantile",
        "mean_returns_by_date",
        "mean_return_spread",
        "mean_return_spread_std",
        "information_coefficient",
        "mean_information_coefficient",
        "quantile_turnover",
        "rank_autocorrelation",
        "grouped_results",
        "time_aggregated_results",
        "pyfolio_inputs",
        "event_returns",
        "result_fingerprint",
    } <= names

    from fincore.factor_analysis.models import EventAnalysisModel, FactorGroupAnalysis

    assert {"group", "quantile_statistics", "factor_returns", "information_coefficient", "quantile_turnover"} <= {
        item.name for item in fields(FactorGroupAnalysis)
    }
    assert {"event_windows", "mean_returns", "return_distribution", "quantile_average_returns"} <= {
        item.name for item in fields(EventAnalysisModel)
    }


def test_public_model_and_entrypoint_annotations_resolve_at_runtime() -> None:
    """Renderer integrations can reflect the checked-in typed public contract."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.models import FactorAnalysisModel

    assert "periods" in get_type_hints(analyze_factor)
    assert "pyfolio_inputs" in get_type_hints(FactorAnalysisModel)


def test_analyze_factor_computes_ic_once_and_owns_input_snapshot(
    clean_factor_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analysis performs computation once and render consumers read its snapshot."""

    from fincore.factor_analysis.analysis import analyze_factor

    calls = {"ic": 0}
    original = performance.factor_information_coefficient

    def counted(*args: object, **kwargs: object) -> pd.DataFrame:
        calls["ic"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(performance, "factor_information_coefficient", counted)
    model = analyze_factor(
        clean_factor_data,
        periods=("1D", "5D"),
        turnover_periods=(1,),
        include_pyfolio=False,
    )
    # Pandas 3 rejects NaN assignment into the integer quantile column; a
    # factor-only mutation still proves the model owns its input snapshot.
    clean_factor_data.loc[:, "factor"] = np.nan

    assert calls["ic"] == 1
    assert not model.factor_data.isna().all().all()
    assert model.forward_periods == ("1D", "5D")

    calls_before_consumption = calls["ic"]
    first_renderer_input = model.information_coefficient.copy(deep=True)
    second_renderer_input = model.mean_returns_by_quantile.copy(deep=True)
    assert not first_renderer_input.empty
    assert not second_renderer_input.empty
    assert calls["ic"] == calls_before_consumption


def test_model_fields_match_the_existing_enhanced_kernel_outputs(clean_factor_data: pd.DataFrame) -> None:
    """Model fields are snapshots of audited kernel output, not renderer placeholders."""

    from fincore.factor_analysis.analysis import analyze_factor

    source = _only_periods(clean_factor_data, ("1D", "5D"))
    model = analyze_factor(source, periods=("1D", "5D"), turnover_periods=(1,), include_pyfolio=False)

    expected_weights = performance.factor_weights(source).to_frame("factor")
    expected_returns = performance.factor_returns(source)
    expected_alpha_beta = performance.factor_alpha_beta(source, returns=expected_returns)
    expected_mean, expected_std = performance.mean_return_by_quantile(source)
    expected_by_date, _ = performance.mean_return_by_quantile(source, by_date=True)
    expected_ic = performance.factor_information_coefficient(source)

    pd.testing.assert_frame_equal(model.factor_weights, expected_weights)
    pd.testing.assert_frame_equal(model.factor_returns, expected_returns)
    pd.testing.assert_frame_equal(model.alpha_beta, expected_alpha_beta)
    pd.testing.assert_frame_equal(model.mean_returns_by_quantile, expected_mean)
    pd.testing.assert_frame_equal(model.std_error_by_quantile, expected_std)
    pd.testing.assert_frame_equal(model.mean_returns_by_date, expected_by_date)
    pd.testing.assert_frame_equal(model.information_coefficient, expected_ic)
    pd.testing.assert_series_equal(model.mean_information_coefficient, expected_ic.mean())
    for period in model.forward_periods:
        pd.testing.assert_series_equal(
            model.factor_cumulative_returns[period],
            portfolio.factor_cumulative_returns(source, period),
        )
        pd.testing.assert_frame_equal(
            model.factor_positions[period],
            portfolio.factor_positions(source, period),
        )

    expected_statistics = source.groupby("factor_quantile", observed=False, sort=True)["factor"].agg(
        ["min", "max", "mean", "std", "count"]
    )
    expected_statistics["count %"] = expected_statistics["count"] / expected_statistics["count"].sum() * 100.0
    pd.testing.assert_frame_equal(model.quantile_statistics, expected_statistics)
    assert tuple(model.quantile_turnover) == (1,)
    assert list(model.rank_autocorrelation.columns) == [1]
    assert model.pyfolio_inputs is None


def test_config_and_result_fingerprints_cover_options_and_input(
    clean_factor_data: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    """All compute-affecting options and the owned input snapshot change fingerprints."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.models import FactorAnalysisConfig

    base = FactorAnalysisConfig(periods=("1D", "5D"))
    variants = (
        FactorAnalysisConfig(long_short=False, periods=("1D", "5D")),
        FactorAnalysisConfig(group_neutral=True, periods=("1D", "5D")),
        FactorAnalysisConfig(equal_weight=True, periods=("1D", "5D")),
        FactorAnalysisConfig(by_group=True, periods=("1D", "5D")),
        FactorAnalysisConfig(periods=("5D",)),
        FactorAnalysisConfig(periods=("1D", "5D"), event_before=1, event_after=2),
        FactorAnalysisConfig(periods=("1D", "5D"), turnover_periods=(2,)),
        FactorAnalysisConfig(periods=("1D", "5D"), time_aggregation=("W",)),
        FactorAnalysisConfig(periods=("1D", "5D"), include_pyfolio=False),
        FactorAnalysisConfig(periods=("1D", "5D"), pyfolio_capital=100_000.0),
        FactorAnalysisConfig(periods=("1D", "5D"), pyfolio_benchmark_period="5D"),
    )
    assert len({base.fingerprint, *(item.fingerprint for item in variants)}) == len(variants) + 1

    model = analyze_factor(clean_factor_data, periods=("1D",), turnover_periods=(1,), include_pyfolio=False)
    equivalent = analyze_factor(clean_factor_data, periods=("1D",), turnover_periods=(1,), include_pyfolio=False)
    changed_input = clean_factor_data.copy(deep=True)
    changed_input.iloc[0, changed_input.columns.get_loc("factor")] += 0.25
    changed = analyze_factor(changed_input, periods=("1D",), turnover_periods=(1,), include_pyfolio=False)

    assert (
        model.config.fingerprint != changed.config.fingerprint or model.result_fingerprint != changed.result_fingerprint
    )
    assert model.result_fingerprint != changed.result_fingerprint
    assert model.result_fingerprint == equivalent.result_fingerprint

    next_representable = clean_factor_data.copy(deep=True)
    factor_column = next_representable.columns.get_loc("factor")
    next_representable.iloc[0, factor_column] = np.nextafter(float(next_representable.iloc[0, factor_column]), np.inf)
    next_model = analyze_factor(
        next_representable,
        periods=("1D",),
        turnover_periods=(1,),
        include_pyfolio=False,
    )

    assert model.result_fingerprint != next_model.result_fingerprint
    assert len(model.config.fingerprint) == 64
    assert len(model.result_fingerprint) == 64

    event_input = _event_returns(prices)
    incomplete_event = analyze_factor(
        clean_factor_data,
        periods=("1D",),
        turnover_periods=(1,),
        include_pyfolio=False,
        event_returns=event_input,
    )
    changed_event_input = event_input.copy(deep=True)
    changed_event_input.iloc[1, 0] += 0.01
    changed_incomplete_event = analyze_factor(
        clean_factor_data,
        periods=("1D",),
        turnover_periods=(1,),
        include_pyfolio=False,
        event_returns=changed_event_input,
    )
    assert incomplete_event.result_fingerprint != changed_incomplete_event.result_fingerprint


def test_fingerprint_is_stable_across_hash_seeds_for_unordered_object_values() -> None:
    """Hashable object cells cannot make provenance depend on process hash randomization."""

    script = """
import pandas as pd
from fincore.factor_analysis.models import fingerprint_value

print(fingerprint_value(pd.DataFrame({'group': [frozenset({'aa', 'bb', 'cc'})]})))
"""

    def fingerprint_for_seed(seed: str) -> str:
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            cwd=".",
            env=environment,
            text=True,
        )
        return completed.stdout.strip()

    assert fingerprint_for_seed("1") == fingerprint_for_seed("2")


def test_fingerprint_includes_inherited_slot_state() -> None:
    """Changing an inherited slot is as material as changing a subclass slot."""

    from fincore.factor_analysis.models import fingerprint_value

    left = pd.DataFrame({"group": [_InheritedSlottedLabel("base-left", "shared")]})
    right = pd.DataFrame({"group": [_InheritedSlottedLabel("base-right", "shared")]})
    dict_backed_left = pd.DataFrame({"group": [_DictAndInheritedSlotLabel("base-left", "shared")]})
    dict_backed_right = pd.DataFrame({"group": [_DictAndInheritedSlotLabel("base-right", "shared")]})

    assert fingerprint_value(left) != fingerprint_value(right)
    assert fingerprint_value(dict_backed_left) != fingerprint_value(dict_backed_right)


def test_fingerprint_retains_extension_dtype_parameters() -> None:
    """Two extension dtypes with the same display name must not alias."""

    pyarrow = pytest.importorskip("pyarrow")

    from fincore.factor_analysis.models import deserialize_serializable_value, fingerprint_value, serializable_value

    python_storage = pd.Series(["label", None], dtype=pd.StringDtype(storage="python"))
    arrow_storage = pd.Series(["label", None], dtype=pd.ArrowDtype(pyarrow.string()))
    arrow_timestamp = pd.Series(
        [pd.Timestamp("2024-01-01 09:30", tz="America/New_York"), None],
        dtype=pd.ArrowDtype(pyarrow.timestamp("us", tz="America/New_York")),
    )

    assert str(python_storage.dtype) == "string"
    assert fingerprint_value(python_storage) != fingerprint_value(arrow_storage)
    restored = deserialize_serializable_value(
        json.loads(
            json.dumps(
                serializable_value(
                    {"python": python_storage, "arrow": arrow_storage, "arrow_timestamp": arrow_timestamp}
                ),
                allow_nan=False,
            )
        )
    )

    assert isinstance(restored, Mapping)
    pd.testing.assert_series_equal(restored["python"], python_storage)
    pd.testing.assert_series_equal(restored["arrow"], arrow_storage)
    pd.testing.assert_series_equal(restored["arrow_timestamp"], arrow_timestamp)

    python_index = pd.Index(["label", None], dtype=pd.StringDtype(storage="python"), name="asset")
    arrow_index = pd.Index(["label", None], dtype=pd.ArrowDtype(pyarrow.string()), name="asset")
    restored_indexes = deserialize_serializable_value(
        json.loads(
            json.dumps(
                serializable_value(
                    {
                        "python": pd.DataFrame({"value": [1, 2]}, index=python_index),
                        "arrow": pd.DataFrame({"value": [1, 2]}, index=arrow_index),
                    }
                ),
                allow_nan=False,
            )
        )
    )

    assert isinstance(restored_indexes, Mapping)
    pd.testing.assert_index_equal(restored_indexes["python"].index, python_index, exact=True)
    pd.testing.assert_index_equal(restored_indexes["arrow"].index, arrow_index, exact=True)


def test_fingerprint_supports_standard_extension_dtype_metadata(clean_factor_data: pd.DataFrame) -> None:
    """Timezone, period, interval, and sparse dtypes stay fingerprintable."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.models import deserialize_serializable_value, fingerprint_value, serializable_value

    observed = clean_factor_data.copy(deep=True)
    observed["observed_at"] = pd.date_range("2024-01-01", periods=len(observed), tz="America/New_York")
    model = analyze_factor(observed, periods=("1D",), turnover_periods=(1,), include_pyfolio=False)
    extension_frame = pd.DataFrame(
        {
            "period": pd.Series(pd.period_range("2024-01", periods=2, freq="M")),
            "interval": pd.Series(pd.arrays.IntervalArray.from_breaks([0, 1, 2])),
            "sparse": pd.Series([0, 1], dtype=pd.SparseDtype("int64", fill_value=0)),
        }
    )

    assert len(model.result_fingerprint) == 64
    assert len(fingerprint_value(extension_frame)) == 64
    restored = deserialize_serializable_value(
        json.loads(json.dumps(serializable_value(extension_frame), allow_nan=False))
    )
    assert isinstance(restored, pd.DataFrame)
    pd.testing.assert_frame_equal(restored, extension_frame)


def test_fingerprint_retains_structured_numpy_dtype_layout() -> None:
    """Raw bytes alone cannot identify structured ndarray field metadata."""

    from fincore.factor_analysis.models import fingerprint_value

    left = np.array([(1, 2)], dtype=[("left", "i4"), ("right", "i4")])
    right = np.array([(1, 2)], dtype=[("bid", "i4"), ("ask", "i4")])

    assert left.dtype.str == right.dtype.str == "|V8"
    assert left.tobytes() == right.tobytes()
    assert fingerprint_value(left) != fingerprint_value(right)
    assert fingerprint_value(pd.DataFrame({"metadata": [left]})) != fingerprint_value(
        pd.DataFrame({"metadata": [right]})
    )


def test_fingerprint_retains_numpy_dtype_metadata() -> None:
    """Public NumPy dtype metadata is part of the model input contract."""

    from fincore.factor_analysis.models import fingerprint_value

    left = np.array([1], dtype=np.dtype("i4", metadata={"unit": "USD"}))
    right = np.array([1], dtype=np.dtype("i4", metadata={"unit": "EUR"}))

    assert left.tobytes() == right.tobytes()
    assert fingerprint_value(left) != fingerprint_value(right)
    assert fingerprint_value(pd.DataFrame({"metadata": [left]})) != fingerprint_value(
        pd.DataFrame({"metadata": [right]})
    )


def test_json_handoff_retains_ieee_float_bits_in_scalars_and_tables() -> None:
    """NaN payloads and signs cannot be collapsed by a textual float codec."""

    from fincore.factor_analysis.models import deserialize_serializable_value, fingerprint_value, serializable_value

    first_nan = struct.unpack(">d", bytes.fromhex("7ff8000000000001"))[0]
    second_nan = struct.unpack(">d", bytes.fromhex("7ff8000000000002"))[0]
    source = {"first": first_nan, "second": second_nan, "frame": pd.DataFrame({"value": [first_nan]})}
    restored = deserialize_serializable_value(json.loads(json.dumps(serializable_value(source), allow_nan=False)))

    assert isinstance(restored, Mapping)
    assert struct.pack(">d", restored["first"]) == struct.pack(">d", first_nan)
    assert struct.pack(">d", restored["second"]) == struct.pack(">d", second_nan)
    assert struct.pack(">d", restored["frame"].iloc[0, 0]) == struct.pack(">d", first_nan)
    assert fingerprint_value(first_nan) != fingerprint_value(second_nan)


def test_model_handoff_retains_numpy_dtype_attrs(clean_factor_data: pd.DataFrame) -> None:
    """NumPy dtype state in renderer metadata remains part of the owned model."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.models import deserialize_serializable_value

    source = clean_factor_data.copy(deep=True)
    source.attrs["metadata_dtype"] = np.dtype("i4", metadata={"unit": "USD"})
    source.attrs["typed_metadata_key_dtype"] = np.dtype("i4", metadata={1: "typed-key"})
    source.attrs["structured_dtype"] = np.dtype([("left", "i4"), ("right", "i4")], metadata={"unit": "USD"})
    source.attrs["aligned_dtype"] = np.dtype(
        [(("title_a", "a"), "i4"), ("b", "f8")], align=True, metadata={"unit": "USD"}
    )
    source.attrs["subarray_dtype"] = np.dtype((np.dtype("i4"), (2,)), metadata={"unit": "USD"})
    model = analyze_factor(source, periods=("1D",), turnover_periods=(1,), include_pyfolio=False)
    restored = deserialize_serializable_value(json.loads(json.dumps(model.to_serializable(), allow_nan=False)))

    assert isinstance(restored, Mapping)
    restored_dtype = restored["factor_data"].attrs["metadata_dtype"]
    assert isinstance(restored_dtype, np.dtype)
    assert restored_dtype.metadata == {"unit": "USD"}
    for name in ("typed_metadata_key_dtype", "structured_dtype", "aligned_dtype", "subarray_dtype"):
        restored_dtype = restored["factor_data"].attrs[name]
        source_dtype = source.attrs[name]
        assert isinstance(restored_dtype, np.dtype)
        assert restored_dtype.descr == source_dtype.descr
        assert restored_dtype.metadata == source_dtype.metadata
        assert restored_dtype.isalignedstruct == source_dtype.isalignedstruct
    assert (
        model.result_fingerprint
        != analyze_factor(
            clean_factor_data, periods=("1D",), turnover_periods=(1,), include_pyfolio=False
        ).result_fingerprint
    )


def test_model_fingerprint_and_json_handoff_support_stdlib_asset_labels(
    clean_factor_data: pd.DataFrame,
) -> None:
    """Valid non-string asset labels remain usable in provenance and handoff."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.models import deserialize_serializable_value, serializable_value

    asset_dates = {
        asset: dt.date(2024, 1, position + 1)
        for position, asset in enumerate(clean_factor_data.index.get_level_values("asset").unique())
    }
    relabelled = clean_factor_data.copy(deep=True)
    relabelled.index = pd.MultiIndex.from_arrays(
        [
            clean_factor_data.index.get_level_values("date"),
            clean_factor_data.index.get_level_values("asset").map(asset_dates),
        ],
        names=clean_factor_data.index.names,
    )

    model = analyze_factor(relabelled, periods=("1D",), turnover_periods=(1,), include_pyfolio=False)
    restored = deserialize_serializable_value(json.loads(json.dumps(model.to_serializable(), allow_nan=False)))
    scalar_labels = pd.Index(
        [
            dt.date(2024, 1, 1),
            dt.datetime(2024, 1, 1, 9, 30, 15, 123456),
            dt.timedelta(days=2, seconds=3, microseconds=4),
        ],
        dtype=object,
        name="asset",
    )

    assert len(model.result_fingerprint) == 64
    assert isinstance(restored, Mapping)
    pd.testing.assert_frame_equal(restored["factor_data"], model.factor_data)
    label_frame = pd.DataFrame({"value": [1, 2, 3]}, index=scalar_labels)
    round_tripped_labels = deserialize_serializable_value(
        json.loads(json.dumps(serializable_value(label_frame), allow_nan=False))
    )
    assert isinstance(round_tripped_labels, pd.DataFrame)
    pd.testing.assert_frame_equal(round_tripped_labels, label_frame)


def test_pandas_timezone_payloads_resolve_to_resolved_zoneinfo_objects() -> None:
    """The pandas-timezone restore path never hands a bare name to ``tz_convert``."""

    import zoneinfo as zoneinfo_module

    from fincore.factor_analysis.models import _pandas_timezone_from_payload

    _require_named_zone("America/New_York")
    resolved = _pandas_timezone_from_payload({"kind": "pandas-timezone", "name": "America/New_York"})

    assert isinstance(resolved, zoneinfo_module.ZoneInfo)
    assert resolved.key == "America/New_York"


def test_dateutil_tzfile_text_form_is_normalized_to_the_iana_envelope() -> None:
    """dateutil's Windows ``tzfile(...)`` string maps to the shared iana-zone kind."""

    import zoneinfo as zoneinfo_module

    from fincore.factor_analysis.models import _pandas_timezone_from_payload, _pandas_timezone_payload

    _require_named_zone("America/New_York")
    payload = _pandas_timezone_payload("tzfile('America/New_York')")
    assert payload == {"kind": "iana-zone", "name": "America/New_York"}

    resolved = _pandas_timezone_from_payload({"kind": "pandas-timezone", "name": "tzfile('America/New_York')"})
    assert isinstance(resolved, zoneinfo_module.ZoneInfo)
    assert resolved.key == "America/New_York"


def test_tzfile_text_forms_without_an_iana_name_are_left_untouched() -> None:
    """Only genuine ``tzfile(...)`` IANA text is normalized; paths and archives are not."""

    from fincore.factor_analysis.models import _iana_name_from_tzfile_text

    assert _iana_name_from_tzfile_text("tzfile('/usr/share/zoneinfo/America/New_York')") == "America/New_York"
    assert _iana_name_from_tzfile_text("tzfile('C:\\zones\\dateutil-zoneinfo.tar.gz')") is None
    assert _iana_name_from_tzfile_text("tzfile('dateutil-zoneinfo.tar.gz')") is None
    assert _iana_name_from_tzfile_text("America/New_York") is None
    assert _iana_name_from_tzfile_text(42) is None


def test_named_timezone_restore_returns_the_raw_name_without_any_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither system data nor the tzdata wheel, the raw name goes back to pandas."""

    import builtins
    import zoneinfo as zoneinfo_module

    from fincore.factor_analysis.models import _resolve_named_timezone

    real_import = builtins.__import__

    def failing_zoneinfo(key: str) -> object:
        raise zoneinfo_module.ZoneInfoNotFoundError(key)

    def no_tzdata(name: str, globals_: object = None, locals_: object = None, fromlist: object = (), level: int = 0):
        if name == "tzdata":
            raise ImportError("tzdata deliberately unavailable")
        return real_import(name, globals_, locals_, fromlist, level)  # type: ignore[arg-type]

    monkeypatch.setattr(zoneinfo_module, "ZoneInfo", failing_zoneinfo)
    monkeypatch.setattr(builtins, "__import__", no_tzdata)
    resolved = _resolve_named_timezone("America/New_York")

    assert resolved == "America/New_York"


def test_named_timezone_restore_falls_back_to_the_tzdata_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without system IANA data, the ``tzdata`` wheel is wired into ``zoneinfo`` explicitly."""

    import zoneinfo as zoneinfo_module

    from fincore.factor_analysis.models import _resolve_named_timezone

    pytest.importorskip("tzdata")  # the wheel is the fallback under test
    _require_named_zone("America/New_York")
    real_zoneinfo = zoneinfo_module.ZoneInfo
    attempts = {"count": 0}

    def failing_zoneinfo(key: str) -> Any:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise zoneinfo_module.ZoneInfoNotFoundError(key)
        return real_zoneinfo(key)

    before = zoneinfo_module.TZPATH
    try:
        monkeypatch.setattr(zoneinfo_module, "ZoneInfo", failing_zoneinfo)
        resolved = _resolve_named_timezone("America/New_York")
    finally:
        zoneinfo_module.reset_tzpath(list(before))

    assert isinstance(resolved, real_zoneinfo)
    assert resolved.key == "America/New_York"
    assert attempts["count"] == 2


def test_stdlib_datetime_timezone_identity_is_retained_in_handoff_and_fingerprint() -> None:
    """A named zone cannot alias its same-offset fixed-timezone counterpart."""

    _require_named_zone("America/New_York")
    from fincore.factor_analysis.models import deserialize_serializable_value, fingerprint_value, serializable_value

    named_zone = dt.datetime(2024, 6, 1, 9, 30, tzinfo=ZoneInfo("America/New_York"))
    fixed_offset = dt.datetime(2024, 6, 1, 9, 30, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
    source = pd.DataFrame({"value": [1]}, index=pd.Index([named_zone], dtype=object, name="asset"))
    restored = deserialize_serializable_value(json.loads(json.dumps(serializable_value(source), allow_nan=False)))
    restored_scalars = deserialize_serializable_value(
        json.loads(json.dumps(serializable_value({"time": named_zone}), allow_nan=False))
    )

    assert isinstance(restored, pd.DataFrame)
    restored_label = restored.index[0]
    assert isinstance(restored_label, dt.datetime)
    assert isinstance(restored_label.tzinfo, ZoneInfo)
    assert restored_label.tzinfo.key == "America/New_York"
    assert isinstance(restored_scalars, Mapping)
    assert isinstance(restored_scalars["time"], dt.datetime)
    assert isinstance(restored_scalars["time"].tzinfo, ZoneInfo)
    assert restored_scalars["time"].tzinfo.key == "America/New_York"
    assert fingerprint_value({"time": named_zone}) != fingerprint_value({"time": fixed_offset})


def test_datetime_fixed_offset_microseconds_are_lossless_in_handoff_and_fingerprint() -> None:
    """Fixed offsets must not be rounded to whole seconds during JSON handoff."""

    from fincore.factor_analysis.models import deserialize_serializable_value, fingerprint_value, serializable_value

    one_microsecond = dt.datetime(2024, 6, 1, 9, 30, tzinfo=dt.timezone(dt.timedelta(microseconds=1)))
    mixed_offset = dt.datetime(
        2024,
        6,
        1,
        9,
        30,
        tzinfo=dt.timezone(dt.timedelta(seconds=30, microseconds=1)),
    )
    restored = deserialize_serializable_value(
        json.loads(json.dumps(serializable_value({"one": one_microsecond, "mixed": mixed_offset}), allow_nan=False))
    )

    assert isinstance(restored, Mapping)
    assert restored["one"] == one_microsecond
    assert restored["mixed"] == mixed_offset
    assert restored["one"].utcoffset() == dt.timedelta(microseconds=1)
    assert restored["mixed"].utcoffset() == dt.timedelta(seconds=30, microseconds=1)
    assert fingerprint_value(one_microsecond) != fingerprint_value(mixed_offset)


def test_pandas_fixed_offset_microsecond_timezones_are_lossless_in_handoff_and_fingerprint() -> None:
    """Pandas timestamps and indexes must retain local time and exact fixed offsets."""

    from fincore.factor_analysis.models import deserialize_serializable_value, fingerprint_value, serializable_value

    timezone = dt.timezone(dt.timedelta(microseconds=1))
    timestamp = pd.Timestamp(dt.datetime(2024, 1, 1, 9, 30, tzinfo=timezone))
    nanosecond_timestamp = pd.Timestamp("2024-01-01 09:30:00.123456789", tz=timezone)
    index = pd.DatetimeIndex([timestamp], name="date")
    empty_index = pd.DatetimeIndex([], tz=timezone, name="date")
    source = {
        "timestamp": timestamp,
        "nanosecond_timestamp": nanosecond_timestamp,
        "indexed": pd.DataFrame({"value": [1.0]}, index=index),
        "empty": pd.DataFrame({"value": []}, index=empty_index),
    }
    restored = deserialize_serializable_value(json.loads(json.dumps(serializable_value(source), allow_nan=False)))

    assert isinstance(restored, Mapping)
    restored_timestamp = restored["timestamp"]
    assert isinstance(restored_timestamp, pd.Timestamp)
    assert restored_timestamp.isoformat() == timestamp.isoformat()
    assert fingerprint_value(restored_timestamp) == fingerprint_value(timestamp)
    restored_nanosecond_timestamp = restored["nanosecond_timestamp"]
    assert isinstance(restored_nanosecond_timestamp, pd.Timestamp)
    assert restored_nanosecond_timestamp.value == nanosecond_timestamp.value
    assert restored_nanosecond_timestamp.tz == nanosecond_timestamp.tz
    assert fingerprint_value(restored_nanosecond_timestamp) == fingerprint_value(nanosecond_timestamp)
    for unit, value in (("s", 1_704_101_400), ("ms", 1_704_101_400_123)):
        unit_source = pd.Timestamp(value, unit=unit, tz=timezone)
        unit_target = deserialize_serializable_value(json.loads(json.dumps(serializable_value(unit_source))))
        assert isinstance(unit_target, pd.Timestamp)
        assert unit_target.unit == unit_source.unit
        assert fingerprint_value(unit_target) == fingerprint_value(unit_source)
    pd.testing.assert_frame_equal(restored["indexed"], source["indexed"])
    pd.testing.assert_frame_equal(restored["empty"], source["empty"])


def test_pandas_dateutil_and_pytz_fixed_offsets_use_the_lossless_timezone_envelope() -> None:
    """Fixed offsets need not pass through a non-reversible timezone display string."""

    import pytz
    from dateutil import tz

    from fincore.factor_analysis.models import deserialize_serializable_value, fingerprint_value, serializable_value

    values = {
        "dateutil": pd.Timestamp("2024-01-01 09:30:00.123456789", tz=tz.tzoffset("custom", 30)),
        "pytz": pd.Timestamp("2024-01-01 09:30:00.123456789", tz=pytz.FixedOffset(30)),
    }
    restored = deserialize_serializable_value(json.loads(json.dumps(serializable_value(values), allow_nan=False)))

    assert isinstance(restored, Mapping)
    for name, source in values.items():
        target = restored[name]
        assert isinstance(target, pd.Timestamp)
        assert target.value == source.value
        assert target.utcoffset() == source.utcoffset()
        assert fingerprint_value(target) == fingerprint_value(source)
        source_index = pd.DatetimeIndex([source], name="date")
        restored_frame = deserialize_serializable_value(
            json.loads(
                json.dumps(serializable_value(pd.DataFrame({"value": [1.0]}, index=source_index)), allow_nan=False)
            )
        )
        assert isinstance(restored_frame, pd.DataFrame)
        restored_index = restored_frame.index
        assert isinstance(restored_index, pd.DatetimeIndex)
        assert tuple(restored_index.asi8) == tuple(source_index.asi8)
        assert restored_index[0].utcoffset() == source_index[0].utcoffset()
        assert fingerprint_value(restored_index) == fingerprint_value(source_index)


def test_pandas_named_timezone_providers_use_one_lossless_iana_envelope() -> None:
    """ZoneInfo, dateutil and pytz names share one reversible timezone identity."""

    _require_named_zone("America/New_York")
    import pytz
    from dateutil import tz

    from fincore.factor_analysis.models import deserialize_serializable_value, fingerprint_value, serializable_value

    providers = {
        "zoneinfo": ZoneInfo("America/New_York"),
        "dateutil": tz.gettz("America/New_York"),
        "pytz": pytz.timezone("America/New_York"),
    }
    for month in ("2024-01", "2024-07"):
        source = {
            name: pd.Timestamp(f"{month}-01 09:30:00.123456789", tz=timezone) for name, timezone in providers.items()
        }
        restored = deserialize_serializable_value(json.loads(json.dumps(serializable_value(source), allow_nan=False)))

        assert isinstance(restored, Mapping)
        for name, timestamp in source.items():
            target = restored[name]
            assert isinstance(target, pd.Timestamp)
            assert target.value == timestamp.value
            assert target.utcoffset() == timestamp.utcoffset()
            assert str(target.tz) == "America/New_York"
            assert fingerprint_value(target) == fingerprint_value(timestamp)
            source_index = pd.DatetimeIndex([timestamp], name="date")
            restored_frame = deserialize_serializable_value(
                json.loads(
                    json.dumps(serializable_value(pd.DataFrame({"value": [1.0]}, index=source_index)), allow_nan=False)
                )
            )
            assert isinstance(restored_frame, pd.DataFrame)
            assert fingerprint_value(restored_frame.index) == fingerprint_value(source_index)


def test_model_handoff_supports_hashable_unordered_object_cells(clean_factor_data: pd.DataFrame) -> None:
    """Frozen metadata remains serializable in both table cells and mapping keys."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.models import deserialize_serializable_value, serializable_value

    metadata = frozenset({"a", "b"})
    source = clean_factor_data.copy(deep=True)
    source["metadata"] = [metadata] * len(source)
    model = analyze_factor(source, periods=("1D",), turnover_periods=(1,), include_pyfolio=False)
    restored_model = deserialize_serializable_value(json.loads(json.dumps(model.to_serializable(), allow_nan=False)))
    restored_mapping = deserialize_serializable_value(
        json.loads(json.dumps(serializable_value({metadata: {"values": metadata}}), allow_nan=False))
    )

    assert isinstance(restored_model, Mapping)
    pd.testing.assert_frame_equal(restored_model["factor_data"], model.factor_data)
    assert isinstance(restored_mapping, Mapping)
    assert restored_mapping[metadata] == {"values": metadata}


def test_model_handoff_supports_accepted_binary_numeric_and_arrow_object_values(
    clean_factor_data: pd.DataFrame,
) -> None:
    """Lossless handoff covers supported object cells and parameterized Arrow scalars."""

    pyarrow = pytest.importorskip("pyarrow")

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.models import deserialize_serializable_value, serializable_value

    array_value = np.array([1, 2], dtype=np.int16)
    source = clean_factor_data.copy(deep=True)
    source["metadata_array"] = [array_value] * len(source)
    source["metadata_bytes"] = [b"asset-bytes"] * len(source)
    source["metadata_complex"] = [complex(1.5, -2.25)] * len(source)
    source["metadata_decimal"] = [Decimal("1.2300")] * len(source)
    model = analyze_factor(source, periods=("1D",), turnover_periods=(1,), include_pyfolio=False)
    restored_model = deserialize_serializable_value(json.loads(json.dumps(model.to_serializable(), allow_nan=False)))
    arrow_frame = pd.DataFrame(
        {
            "decimal": pd.Series([Decimal("1.23"), None], dtype=pd.ArrowDtype(pyarrow.decimal128(10, 2))),
            "list": pd.Series([[1, 2], None], dtype=pd.ArrowDtype(pyarrow.list_(pyarrow.int16()))),
        }
    )
    restored_arrow = deserialize_serializable_value(
        json.loads(json.dumps(serializable_value(arrow_frame), allow_nan=False))
    )

    assert isinstance(restored_model, Mapping)
    restored_factor_data = restored_model["factor_data"]
    restored_array = restored_factor_data["metadata_array"].iloc[0]
    assert isinstance(restored_array, np.ndarray)
    assert restored_array.dtype == np.dtype("int16")
    np.testing.assert_array_equal(restored_array, array_value)
    assert restored_factor_data["metadata_bytes"].iloc[0] == b"asset-bytes"
    assert restored_factor_data["metadata_complex"].iloc[0] == complex(1.5, -2.25)
    assert restored_factor_data["metadata_decimal"].iloc[0].as_tuple() == Decimal("1.2300").as_tuple()
    assert isinstance(restored_arrow, pd.DataFrame)
    pd.testing.assert_frame_equal(restored_arrow, arrow_frame)


def test_json_handoff_retains_numpy_scalars_and_nested_object_arrays(
    clean_factor_data: pd.DataFrame,
) -> None:
    """Object cells cannot collapse NumPy dtypes or nested array values to Python."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.models import deserialize_serializable_value, fingerprint_value, serializable_value

    nested = np.empty(2, dtype=object)
    nested[0] = np.array([1, 2], dtype=np.int16)
    nested[1] = {"value": np.array([3], dtype=np.int8)}
    object_scalar = np.array([({"value": "x"},)], dtype=[("payload", object)])[0]
    scalar_values = {
        "float32": np.float32(1.25),
        "int16": np.int16(7),
        "datetime64": np.datetime64("2024-01-02T03:04:05.678", "ms"),
        "timedelta64": np.timedelta64(1234, "ms"),
        "longdouble": np.longdouble("1.234567890123456789"),
        "clongdouble": np.clongdouble("1.25-2.5j"),
        "nested": nested,
        "object_scalar": object_scalar,
    }
    restored_scalars = deserialize_serializable_value(
        json.loads(json.dumps(serializable_value(scalar_values), allow_nan=False))
    )
    source = clean_factor_data.copy(deep=True)
    source["metadata_scalar"] = pd.Series([np.float32(1.25)] * len(source), index=source.index, dtype=object)
    model = analyze_factor(source, periods=("1D",), turnover_periods=(1,), include_pyfolio=False)
    restored_model = deserialize_serializable_value(json.loads(json.dumps(model.to_serializable(), allow_nan=False)))

    assert isinstance(restored_scalars, Mapping)
    for name, scalar in scalar_values.items():
        if name == "nested":
            restored_nested = restored_scalars[name]
            assert isinstance(restored_nested, np.ndarray)
            assert restored_nested.dtype == nested.dtype
            np.testing.assert_array_equal(restored_nested[0], nested[0])
            np.testing.assert_array_equal(restored_nested[1]["value"], nested[1]["value"])
            continue
        if name == "object_scalar":
            assert isinstance(restored_scalars[name], np.void)
            assert restored_scalars[name]["payload"] == {"value": "x"}
            assert fingerprint_value(restored_scalars[name]) == fingerprint_value(scalar)
            continue
        restored_scalar = restored_scalars[name]
        assert isinstance(restored_scalar, np.generic)
        assert restored_scalar.dtype == scalar.dtype
        assert restored_scalar.tobytes() == scalar.tobytes()
        assert fingerprint_value(restored_scalar) == fingerprint_value(scalar)
    assert isinstance(restored_model, Mapping)
    restored_scalar = restored_model["factor_data"]["metadata_scalar"].iloc[0]
    assert isinstance(restored_scalar, np.float32)
    assert restored_scalar.tobytes() == np.float32(1.25).tobytes()


def test_config_owns_sequence_options_and_rejects_lossy_integer_capital() -> None:
    """A frozen config cannot retain caller-owned lists or rounded capital."""

    from fincore.factor_analysis.models import FactorAnalysisConfig

    periods = ["1D"]
    turnover_periods = [1]
    aggregations = ["M"]
    config = FactorAnalysisConfig(
        periods=periods,  # type: ignore[arg-type]
        turnover_periods=turnover_periods,  # type: ignore[arg-type]
        time_aggregation=aggregations,  # type: ignore[arg-type]
        pyfolio_capital=2**53,
    )
    fingerprint = config.fingerprint
    periods.append("5D")
    turnover_periods.append(2)
    aggregations.append("W")

    assert config.periods == ("1D",)
    assert config.turnover_periods == (1,)
    assert config.time_aggregation == ("M",)
    assert config.pyfolio_capital == 2**53
    assert config.fingerprint == fingerprint

    with pytest.raises(ValueError, match="exactly"):
        FactorAnalysisConfig(pyfolio_capital=2**53 + 1)
    assert FactorAnalysisConfig(pyfolio_capital=2**54).pyfolio_capital == 2**54


def test_config_rejects_nondeterministic_sequences_and_invalid_typed_options(
    clean_factor_data: pd.DataFrame,
) -> None:
    """Typed config inputs must not silently accept unordered or invalid values."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.models import FactorAnalysisConfig

    with pytest.raises(TypeError, match="periods"):
        FactorAnalysisConfig(periods="1D")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="periods"):
        FactorAnalysisConfig(periods={"1D", "5D"})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="turnover_periods"):
        FactorAnalysisConfig(turnover_periods={1, 2})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="time_aggregation"):
        FactorAnalysisConfig(time_aggregation="M")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="long_short"):
        FactorAnalysisConfig(long_short="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="event_before"):
        FactorAnalysisConfig(event_before=True)
    with pytest.raises(TypeError, match="event_before"):
        FactorAnalysisConfig(event_before="bad", event_after=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        FactorAnalysisConfig(event_before=-1)

    with pytest.raises(TypeError, match="periods"):
        analyze_factor(clean_factor_data, periods="1D", include_pyfolio=False)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="event_before"):
        analyze_factor(
            clean_factor_data,
            event_before="bad",  # type: ignore[arg-type]
            event_after=None,
            include_pyfolio=False,
        )


def test_frozen_mapping_owns_and_releases_independent_mutable_values() -> None:
    """Mapping keys and nested mutable values cannot mutate a stored snapshot."""

    from fincore.factor_analysis.models import frozen_mapping

    source = {"items": [{"value": 1}], "labels": {1, 2}}
    frozen = frozen_mapping(source)
    source["items"][0]["value"] = 99
    source["labels"].add(3)

    assert frozen["items"] == [{"value": 1}]
    assert frozen["labels"] == {1, 2}
    exposed_items = frozen["items"]
    exposed_labels = frozen["labels"]
    exposed_items.append({"value": 4})
    exposed_labels.add(4)
    assert frozen["items"] == [{"value": 1}]
    assert frozen["labels"] == {1, 2}

    equal_key = _MutableGroupLabel(["group"])
    equal_key_mapping = frozen_mapping({equal_key: "value"})
    released_key = next(iter(equal_key_mapping))
    assert equal_key_mapping[equal_key] == "value"
    assert equal_key_mapping[released_key] == "value"
    assert dict(equal_key_mapping)[released_key] == "value"
    with pytest.raises(TypeError, match="preserve equality"):
        frozen_mapping({_IdentityOnlyKey(): "value"})


def test_snapshot_value_pickles_generic_owned_objects() -> None:
    """Calendar-like objects outside the typed fast paths use the pickle fallback."""

    from fincore.factor_analysis.models import _snapshot_value

    original = BDay(2)
    snapshot = _snapshot_value(original)

    assert snapshot == original
    assert snapshot is not original
    assert isinstance(snapshot, BDay)


def test_model_exposes_defensive_snapshots_for_all_renderer_data(
    clean_factor_data: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    """Public table access cannot mutate the frozen model or its provenance."""

    from fincore.factor_analysis.analysis import analyze_factor

    model = analyze_factor(
        clean_factor_data,
        periods=("1D",),
        turnover_periods=(1,),
        by_group=True,
        include_pyfolio=True,
        event_returns=_event_returns(prices),
        event_before=1,
        event_after=2,
    )
    fingerprint = model.result_fingerprint
    factor_snapshot = model.factor_data.copy(deep=True)
    cumulative_snapshot = model.factor_cumulative_returns["1D"].copy(deep=True)
    group_key = next(iter(model.grouped_results))
    group_snapshot = model.grouped_results[group_key].factor_returns.copy(deep=True)
    assert model.event_returns is not None
    event_snapshot = model.event_returns.event_windows.copy(deep=True)
    assert model.pyfolio_inputs is not None
    positions_snapshot = model.pyfolio_inputs.positions.copy(deep=True)

    changed_factor_data = model.factor_data
    changed_factor_data.iloc[0, changed_factor_data.columns.get_loc("factor")] = 999.0
    changed_cumulative = model.factor_cumulative_returns["1D"]
    changed_cumulative.iloc[0] = 999.0
    changed_group = model.grouped_results[group_key].factor_returns
    changed_group.iloc[0, 0] = 999.0
    assert model.event_returns is not None
    changed_event = model.event_returns.event_windows
    changed_event.iloc[0, 0] = 999.0
    assert model.pyfolio_inputs is not None
    changed_positions = model.pyfolio_inputs.positions
    changed_positions.iloc[0, 0] = 999.0

    pd.testing.assert_frame_equal(model.factor_data, factor_snapshot)
    pd.testing.assert_series_equal(model.factor_cumulative_returns["1D"], cumulative_snapshot)
    pd.testing.assert_frame_equal(model.grouped_results[group_key].factor_returns, group_snapshot)
    assert model.event_returns is not None
    pd.testing.assert_frame_equal(model.event_returns.event_windows, event_snapshot)
    assert model.pyfolio_inputs is not None
    pd.testing.assert_frame_equal(model.pyfolio_inputs.positions, positions_snapshot)
    assert model.result_fingerprint == fingerprint
    with pytest.raises(TypeError):
        model.factor_positions["new"] = pd.DataFrame()  # type: ignore[index]
    with pytest.raises(AttributeError):
        _ = model.__dict__


def test_model_owns_mutable_object_cells_before_computation(clean_factor_data: pd.DataFrame) -> None:
    """Pandas object cells cannot retain aliases into the caller's input frame."""

    from fincore.factor_analysis.analysis import analyze_factor

    mutable_group = _MutableGroupLabel(["initial"])
    source = clean_factor_data.astype({"group": object})
    source.iloc[0, source.columns.get_loc("group")] = mutable_group
    model = analyze_factor(source, periods=("1D",), turnover_periods=(1,), include_pyfolio=False)
    fingerprint = model.result_fingerprint

    mutable_group.labels.append("caller-mutation")

    stored_group = model.factor_data.iloc[0, model.factor_data.columns.get_loc("group")]
    assert isinstance(stored_group, _MutableGroupLabel)
    assert stored_group.labels == ["initial"]
    assert model.result_fingerprint == fingerprint


def test_grouped_model_exposes_copied_mapping_keys(clean_factor_data: pd.DataFrame) -> None:
    """Mutable group labels from a public mapping cannot alter canonical provenance."""

    from fincore.factor_analysis.analysis import analyze_factor

    first_group = _MutableGroupLabel(["one"])
    second_group = _MutableGroupLabel(["two"])
    source = clean_factor_data.astype({"group": object})
    source.loc[:, "group"] = [first_group if row % 2 else second_group for row in range(len(source))]
    model = analyze_factor(source, periods=("1D",), turnover_periods=(1,), by_group=True, include_pyfolio=False)
    fingerprint = model.result_fingerprint

    exposed_key = next(key for key in model.grouped_results if isinstance(key, _MutableGroupLabel))
    exposed_key.labels.append("public-mutation")

    canonical_key = next(key for key in model.grouped_results if isinstance(key, _MutableGroupLabel))
    assert canonical_key.labels in (["one"], ["two"])
    assert model.result_fingerprint == fingerprint


def test_serializable_handoff_round_trips_exact_values_keys_and_pandas_metadata() -> None:
    """JSON handoff retains adjacent floats, typed keys, categories, and timezone."""

    from fincore.factor_analysis.models import deserialize_serializable_value, serializable_value

    index = pd.date_range("2024-03-08 09:30", periods=2, freq="D", tz="America/New_York", name="when")
    frame = pd.DataFrame(
        {
            "factor": [np.nextafter(1.0, np.inf), 2.0],
            "group": pd.Categorical(["b", "a"], categories=["unused", "b", "a"], ordered=True),
        },
        index=index,
    )
    frame.attrs["renderer-hint"] = {"periods": ("1D",)}
    frame.flags.allows_duplicate_labels = False
    series = pd.Series([1.0, 2.0], index=index, name="alpha")
    series.attrs["source"] = "frozen"
    series.flags.allows_duplicate_labels = False
    source = {1: "integer-key", "1": "text-key", "frame": frame, "series": series}

    payload = serializable_value(source)
    restored = deserialize_serializable_value(json.loads(json.dumps(payload, allow_nan=False)))

    assert isinstance(restored, Mapping)
    assert restored[1] == "integer-key"
    assert restored["1"] == "text-key"
    pd.testing.assert_frame_equal(restored["frame"], frame)
    pd.testing.assert_series_equal(restored["series"], series)
    assert restored["frame"].attrs == frame.attrs
    assert restored["frame"].flags.allows_duplicate_labels is False
    assert restored["series"].attrs == series.attrs
    assert restored["series"].flags.allows_duplicate_labels is False


def test_serializable_handoff_round_trips_distant_and_empty_timezone_indexes() -> None:
    """Datetime handoff keeps the native unit and timezone without values to infer it."""

    from fincore.factor_analysis.models import deserialize_serializable_value, serializable_value

    far_index = pd.DatetimeIndex([pd.Timestamp("2500-01-01 09:30")], name="date")
    far_timezone_index = pd.DatetimeIndex([pd.Timestamp("2500-06-01 09:30", tz="America/New_York")], name="date")
    empty_index = pd.DatetimeIndex([], tz="America/New_York", name="date", freq="D")
    custom_business_day = CustomBusinessDay(weekmask="Mon Tue Wed Thu", holidays=["2024-01-02"])
    custom_index = pd.date_range("2024-01-01", periods=3, freq=custom_business_day, name="date")
    normalized_business_index = pd.date_range("2024-01-01 12:00", periods=3, freq=BDay(normalize=True), name="date")
    source = {
        "far": pd.DataFrame({"value": [1.0]}, index=far_index),
        "far_timezone": pd.DataFrame({"value": [1.0]}, index=far_timezone_index),
        "empty": pd.DataFrame({"value": pd.Series(dtype=float)}, index=empty_index),
        "custom": pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=custom_index),
        "normalized_business": pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=normalized_business_index),
    }

    restored = deserialize_serializable_value(json.loads(json.dumps(serializable_value(source), allow_nan=False)))

    assert isinstance(restored, Mapping)
    pd.testing.assert_frame_equal(restored["far"], source["far"])
    pd.testing.assert_frame_equal(restored["far_timezone"], source["far_timezone"])
    pd.testing.assert_frame_equal(restored["empty"], source["empty"])
    pd.testing.assert_frame_equal(restored["custom"], source["custom"])
    pd.testing.assert_frame_equal(restored["normalized_business"], source["normalized_business"])


def test_group_and_event_sections_are_optional_typed_models(
    clean_factor_data: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    """Missing optional inputs omit their sections without leaking untyped dictionaries."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.models import EventAnalysisModel, FactorGroupAnalysis

    without_group = clean_factor_data.drop(columns="group")
    no_group = analyze_factor(
        without_group,
        periods=("1D",),
        by_group=True,
        turnover_periods=(1,),
        include_pyfolio=False,
    )
    assert no_group.grouped_results == {}
    assert no_group.event_returns is None

    grouped = analyze_factor(
        clean_factor_data,
        periods=("1D",),
        by_group=True,
        turnover_periods=(1,),
        include_pyfolio=False,
    )
    assert set(grouped.grouped_results) == set(clean_factor_data["group"].unique())
    assert all(isinstance(item, FactorGroupAnalysis) for item in grouped.grouped_results.values())

    event = analyze_factor(
        clean_factor_data,
        periods=("1D",),
        turnover_periods=(1,),
        include_pyfolio=False,
        event_returns=_event_returns(prices),
        event_before=1,
        event_after=2,
    )
    assert isinstance(event.event_returns, EventAnalysisModel)
    assert not event.event_returns.event_windows.empty
    assert not event.event_returns.mean_returns.empty
    assert isinstance(event.event_returns.return_distribution, pd.Series)
    assert not event.event_returns.return_distribution.empty


def test_model_is_frozen_json_serializable_and_contains_no_render_objects(clean_factor_data: pd.DataFrame) -> None:
    """Task 6 ends in renderer-ready data, never figures, axes, or executable cache state."""

    from fincore.factor_analysis.analysis import analyze_factor

    model = analyze_factor(clean_factor_data, periods=("1D",), turnover_periods=(1,), include_pyfolio=False)

    with pytest.raises(FrozenInstanceError):
        model.config = model.config  # type: ignore[misc]
    assert not hasattr(model, "cache")
    _assert_serializable_data_only(model)
    payload = model.to_serializable()
    assert (
        json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))["result_fingerprint"]
        == model.result_fingerprint
    )


def test_pyfolio_bridge_is_optional_typed_data_not_a_renderer(clean_factor_data: pd.DataFrame) -> None:
    """The model may include the Task 5 bridge without importing external Pyfolio."""

    from fincore.factor_analysis.analysis import analyze_factor
    from fincore.factor_analysis.portfolio import PyfolioFactorInputs

    model = analyze_factor(
        clean_factor_data,
        periods=("1D",),
        turnover_periods=(1,),
        include_pyfolio=True,
        pyfolio_capital=100_000.0,
        pyfolio_benchmark_period="5D",
    )

    assert isinstance(model.pyfolio_inputs, PyfolioFactorInputs)
    _assert_serializable_data_only(model.pyfolio_inputs)
    bridge_payload = json.loads(json.dumps(model.to_serializable(), sort_keys=True, allow_nan=False))["pyfolio_inputs"]
    assert set(bridge_payload) == {"benchmark_rets", "positions", "returns"}
