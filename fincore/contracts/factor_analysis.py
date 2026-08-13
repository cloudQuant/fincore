"""Static contracts for the pinned cloudQuant Alphalens facade.

The rows below were transcribed from the reviewed, pinned API manifest during
development.  They are deliberately normal Python data: importing fincore
never opens a test fixture or relies on a sibling Alphalens checkout.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

FactorProfile = Literal["legacy_alphalens_cloudquant_0_4_0", "enhanced_factor_analysis"]


@dataclass(frozen=True)
class FactorFunctionSpec:
    """One named function contract on a factor-analysis compatibility surface."""

    module: str
    public_name: str
    introspection_signature: inspect.Signature
    source_signature: inspect.Signature
    implementation: str
    profile: FactorProfile
    optional_extra: str | None = None
    adapter: str | None = None
    result_projection: str | None = None


class _SignatureSymbol:
    """A symbolic static default that keeps a legacy repr without an optional import."""

    def __init__(self, text: str) -> None:
        self._text = text

    def __repr__(self) -> str:
        return self._text


class _StaticStats:
    """Namespace sufficient to parse the pinned ``stats.norm`` default text."""

    norm = _SignatureSymbol("stats.norm")


# Each row is (module, public name, source-visible signature,
# introspection signature, adapter).  ``None`` introspection in the original
# AST manifest is represented by its symbolic source text, not an imported
# statsmodels distribution or an invented numerical object.
_RAW_FACTOR_FUNCTION_SPECS: tuple[tuple[str, str, str, str | None, str | None], ...] = (
    (
        "performance",
        "average_cumulative_return_by_quantile",
        "(factor_data, returns, periods_before=10, periods_after=15, demeaned=True, group_adjust=False, by_group=False)",
        "(factor_data, returns, periods_before=10, periods_after=15, demeaned=True, group_adjust=False, by_group=False)",
        None,
    ),
    (
        "performance",
        "common_start_returns",
        "(factor, returns, before, after, cumulative=False, mean_by_date=False, demean_by=None)",
        "(factor, returns, before, after, cumulative=False, mean_by_date=False, demean_by=None)",
        None,
    ),
    (
        "performance",
        "compute_mean_returns_spread",
        "(mean_returns, upper_quant, lower_quant, std_err=None)",
        "(mean_returns, upper_quant, lower_quant, std_err=None)",
        None,
    ),
    (
        "performance",
        "create_pyfolio_input",
        "(factor_data, period, capital=None, long_short=True, group_neutral=False, equal_weight=False, quantiles=None, groups=None, benchmark_period='1D')",
        "(factor_data, period, capital=None, long_short=True, group_neutral=False, equal_weight=False, quantiles=None, groups=None, benchmark_period='1D')",
        None,
    ),
    ("performance", "cumulative_returns", "(returns)", "(returns)", None),
    (
        "performance",
        "factor_alpha_beta",
        "(factor_data, returns=None, demeaned=True, group_adjust=False, equal_weight=False)",
        "(factor_data, returns=None, demeaned=True, group_adjust=False, equal_weight=False)",
        None,
    ),
    (
        "performance",
        "factor_cumulative_returns",
        "(factor_data, period, long_short=True, group_neutral=False, equal_weight=False, quantiles=None, groups=None)",
        "(factor_data, period, long_short=True, group_neutral=False, equal_weight=False, quantiles=None, groups=None)",
        None,
    ),
    (
        "performance",
        "factor_information_coefficient",
        "(factor_data, group_adjust=False, by_group=False)",
        "(factor_data, group_adjust=False, by_group=False)",
        None,
    ),
    (
        "performance",
        "factor_positions",
        "(factor_data, period, long_short=True, group_neutral=False, equal_weight=False, quantiles=None, groups=None)",
        "(factor_data, period, long_short=True, group_neutral=False, equal_weight=False, quantiles=None, groups=None)",
        None,
    ),
    ("performance", "factor_rank_autocorrelation", "(factor_data, period=1)", "(factor_data, period=1)", None),
    (
        "performance",
        "factor_returns",
        "(factor_data, demeaned=True, group_adjust=False, equal_weight=False, by_asset=False)",
        "(factor_data, demeaned=True, group_adjust=False, equal_weight=False, by_asset=False)",
        None,
    ),
    (
        "performance",
        "factor_weights",
        "(factor_data, demeaned=True, group_adjust=False, equal_weight=False)",
        "(factor_data, demeaned=True, group_adjust=False, equal_weight=False)",
        None,
    ),
    (
        "performance",
        "mean_information_coefficient",
        "(factor_data, group_adjust=False, by_group=False, by_time=None)",
        "(factor_data, group_adjust=False, by_group=False, by_time=None)",
        None,
    ),
    (
        "performance",
        "mean_return_by_quantile",
        "(factor_data, by_date=False, by_group=False, demeaned=True, group_adjust=False)",
        "(factor_data, by_date=False, by_group=False, demeaned=True, group_adjust=False)",
        None,
    ),
    ("performance", "positions", "(weights, period, freq=None)", "(weights, period, freq=None)", None),
    (
        "performance",
        "quantile_turnover",
        "(quantile_factor, quantile, period=1)",
        "(quantile_factor, quantile, period=1)",
        None,
    ),
    ("plotting", "axes_style", "(style='darkgrid', rc=None)", "(style='darkgrid', rc=None)", None),
    ("plotting", "customize", "(func)", "(func)", None),
    (
        "plotting",
        "plot_cumulative_returns",
        "(factor_returns, period, freq=None, title=None, ax=None)",
        "(factor_returns, period, freq=None, title=None, ax=None)",
        None,
    ),
    (
        "plotting",
        "plot_cumulative_returns_by_quantile",
        "(quantile_returns, period, freq=None, ax=None)",
        "(quantile_returns, period, freq=None, ax=None)",
        None,
    ),
    ("plotting", "plot_events_distribution", "(events, num_bars=50, ax=None)", "(events, num_bars=50, ax=None)", None),
    (
        "plotting",
        "plot_factor_rank_auto_correlation",
        "(factor_autocorrelation, period=1, ax=None)",
        "(factor_autocorrelation, period=1, ax=None)",
        None,
    ),
    ("plotting", "plot_ic_by_group", "(ic_group, ax=None)", "(ic_group, ax=None)", None),
    ("plotting", "plot_ic_hist", "(ic, ax=None)", "(ic, ax=None)", None),
    ("plotting", "plot_ic_qq", "(ic, theoretical_dist=stats.norm, ax=None)", None, None),
    ("plotting", "plot_ic_ts", "(ic, ax=None)", "(ic, ax=None)", None),
    ("plotting", "plot_information_table", "(ic_data)", "(ic_data)", None),
    (
        "plotting",
        "plot_mean_quantile_returns_spread_time_series",
        "(mean_returns_spread, std_err=None, bandwidth=1, ax=None)",
        "(mean_returns_spread, std_err=None, bandwidth=1, ax=None)",
        None,
    ),
    ("plotting", "plot_monthly_ic_heatmap", "(mean_monthly_ic, ax=None)", "(mean_monthly_ic, ax=None)", None),
    (
        "plotting",
        "plot_quantile_average_cumulative_return",
        "(avg_cumulative_returns, by_quantile=False, std_bar=False, title=None, ax=None)",
        "(avg_cumulative_returns, by_quantile=False, std_bar=False, title=None, ax=None)",
        None,
    ),
    (
        "plotting",
        "plot_quantile_returns_bar",
        "(mean_ret_by_q, by_group=False, ylim_percentiles=None, ax=None)",
        "(mean_ret_by_q, by_group=False, ylim_percentiles=None, ax=None)",
        None,
    ),
    (
        "plotting",
        "plot_quantile_returns_violin",
        "(return_by_q, ylim_percentiles=None, ax=None)",
        "(return_by_q, ylim_percentiles=None, ax=None)",
        None,
    ),
    ("plotting", "plot_quantile_statistics_table", "(factor_data)", "(factor_data)", None),
    (
        "plotting",
        "plot_returns_table",
        "(alpha_beta, mean_ret_quantile, mean_ret_spread_quantile)",
        "(alpha_beta, mean_ret_quantile, mean_ret_spread_quantile)",
        None,
    ),
    (
        "plotting",
        "plot_top_bottom_quantile_turnover",
        "(quantile_turnover, period=1, ax=None)",
        "(quantile_turnover, period=1, ax=None)",
        None,
    ),
    (
        "plotting",
        "plot_turnover_table",
        "(autocorrelation_data, quantile_turnover)",
        "(autocorrelation_data, quantile_turnover)",
        None,
    ),
    (
        "plotting",
        "plotting_context",
        "(context='notebook', font_scale=1.5, rc=None)",
        "(context='notebook', font_scale=1.5, rc=None)",
        None,
    ),
    (
        "tears",
        "create_event_returns_tear_sheet",
        "(factor_data, returns, avgretplot=(5, 15), long_short=True, group_neutral=False, std_bar=True, by_group=False)",
        "(factor_data, returns, avgretplot=(5, 15), long_short=True, group_neutral=False, std_bar=True, by_group=False)",
        "plotting.customize",
    ),
    (
        "tears",
        "create_event_study_tear_sheet",
        "(factor_data, returns, avgretplot=(5, 15), rate_of_ret=True, n_bars=50)",
        "(factor_data, returns, avgretplot=(5, 15), rate_of_ret=True, n_bars=50)",
        "plotting.customize",
    ),
    (
        "tears",
        "create_full_tear_sheet",
        "(factor_data, long_short=True, group_neutral=False, by_group=False)",
        "(factor_data, long_short=True, group_neutral=False, by_group=False)",
        "plotting.customize",
    ),
    (
        "tears",
        "create_information_tear_sheet",
        "(factor_data, group_neutral=False, by_group=False)",
        "(factor_data, group_neutral=False, by_group=False)",
        "plotting.customize",
    ),
    (
        "tears",
        "create_returns_tear_sheet",
        "(factor_data, long_short=True, group_neutral=False, by_group=False)",
        "(factor_data, long_short=True, group_neutral=False, by_group=False)",
        "plotting.customize",
    ),
    (
        "tears",
        "create_summary_tear_sheet",
        "(factor_data, long_short=True, group_neutral=False)",
        "(factor_data, long_short=True, group_neutral=False)",
        "plotting.customize",
    ),
    (
        "tears",
        "create_turnover_tear_sheet",
        "(factor_data, turnover_periods=None)",
        "(factor_data, turnover_periods=None)",
        "plotting.customize",
    ),
    ("utils", "add_custom_calendar_timedelta", "(input, timedelta, freq)", "(input, timedelta, freq)", None),
    ("utils", "backshift_returns_series", "(series, N)", "(series, N)", None),
    (
        "utils",
        "compute_forward_returns",
        "(factor, prices, periods=(1, 5, 10), filter_zscore=None, cumulative_returns=True)",
        "(factor, prices, periods=(1, 5, 10), filter_zscore=None, cumulative_returns=True)",
        None,
    ),
    ("utils", "demean_forward_returns", "(factor_data, grouper=None)", "(factor_data, grouper=None)", None),
    ("utils", "diff_custom_calendar_timedeltas", "(start, end, freq)", "(start, end, freq)", None),
    (
        "utils",
        "get_clean_factor",
        "(factor, forward_returns, groupby=None, binning_by_group=False, quantiles=5, bins=None, groupby_labels=None, max_loss=0.35, zero_aware=False)",
        "(factor, forward_returns, groupby=None, binning_by_group=False, quantiles=5, bins=None, groupby_labels=None, max_loss=0.35, zero_aware=False)",
        None,
    ),
    (
        "utils",
        "get_clean_factor_and_forward_returns",
        "(factor, prices, groupby=None, binning_by_group=False, quantiles=5, bins=None, periods=(1, 5, 10), filter_zscore=20, groupby_labels=None, max_loss=0.35, zero_aware=False, cumulative_returns=True)",
        "(factor, prices, groupby=None, binning_by_group=False, quantiles=5, bins=None, periods=(1, 5, 10), filter_zscore=20, groupby_labels=None, max_loss=0.35, zero_aware=False, cumulative_returns=True)",
        None,
    ),
    (
        "utils",
        "get_forward_returns_columns",
        "(columns, require_exact_day_multiple=False)",
        "(columns, require_exact_day_multiple=False)",
        None,
    ),
    ("utils", "infer_trading_calendar", "(factor_idx, prices_idx)", "(factor_idx, prices_idx)", None),
    ("utils", "non_unique_bin_edges_error", "(func)", "(func)", None),
    ("utils", "print_table", "(table, name=None, fmt=None)", "(table, name=None, fmt=None)", None),
    (
        "utils",
        "quantize_factor",
        "(factor_data, quantiles=5, bins=None, by_group=False, no_raise=False, zero_aware=False)",
        "(*args, **kwargs)",
        "non_unique_bin_edges_error",
    ),
    ("utils", "rate_of_return", "(period_ret, base_period)", "(period_ret, base_period)", None),
    ("utils", "rethrow", "(exception, additional_message)", "(exception, additional_message)", None),
    ("utils", "std_conversion", "(period_std, base_period)", "(period_std, base_period)", None),
    ("utils", "timedelta_strings_to_integers", "(sequence)", "(sequence)", None),
    ("utils", "timedelta_to_string", "(timedelta)", "(timedelta)", None),
)

_STATIC_SIGNATURE_TEXTS = frozenset(
    text
    for _, _, source_text, introspection_text, _ in _RAW_FACTOR_FUNCTION_SPECS
    for text in (source_text, introspection_text)
    if text
)


def _signature_from_static_text(signature_text: str) -> inspect.Signature:
    """Parse one checked-in signature row in a deliberately closed namespace."""

    if signature_text not in _STATIC_SIGNATURE_TEXTS:
        raise ValueError("Alphalens signatures must originate in the checked-in static registry")
    namespace: dict[str, Any] = {}
    exec(  # nosec B102: inputs are verified static literals above; only a stub definition is evaluated.
        f"def _static_signature{signature_text}:\n    pass",
        {"__builtins__": {}, "stats": _StaticStats},
        namespace,
    )
    return inspect.signature(namespace["_static_signature"])


def _make_function_specs() -> Mapping[tuple[str, str], FactorFunctionSpec]:
    specs: dict[tuple[str, str], FactorFunctionSpec] = {}
    for module, name, source_text, introspection_text, adapter in _RAW_FACTOR_FUNCTION_SPECS:
        key = (module, name)
        if key in specs:
            raise RuntimeError(f"duplicate Alphalens compatibility entry: {module}.{name}")
        specs[key] = FactorFunctionSpec(
            module=module,
            public_name=name,
            source_signature=_signature_from_static_text(source_text),
            introspection_signature=_signature_from_static_text(introspection_text or source_text),
            implementation="deferred_task_2_kernel",
            profile="legacy_alphalens_cloudquant_0_4_0",
            optional_extra="pyfolio" if module in {"plotting", "tears"} else None,
            adapter=adapter,
            result_projection="not_implemented_until_task_3_4_or_8",
        )
    if len(specs) != 61:
        raise RuntimeError(f"expected 61 pinned Alphalens functions, found {len(specs)}")
    return MappingProxyType(specs)


ALPHALENS_FUNCTION_SPECS: Mapping[tuple[str, str], FactorFunctionSpec] = _make_function_specs()


def function_specs_for_module(module: str) -> tuple[FactorFunctionSpec, ...]:
    """Return the stable, source-manifest order for one public facade module."""

    return tuple(spec for (spec_module, _), spec in ALPHALENS_FUNCTION_SPECS.items() if spec_module == module)


__all__ = ["ALPHALENS_FUNCTION_SPECS", "FactorFunctionSpec", "FactorProfile", "function_specs_for_module"]
