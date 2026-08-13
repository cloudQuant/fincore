"""Frozen lifecycle contracts for Alphalens tear-sheet entry points."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Mapping

from fincore.contracts.factor_analysis import ALPHALENS_FUNCTION_SPECS

if TYPE_CHECKING:
    import inspect


@dataclass(frozen=True)
class FactorWorkflowSpec:
    """A deferred tear-sheet workflow, including its future renderer lifecycle."""

    public_name: str
    introspection_signature: inspect.Signature
    source_signature: inspect.Signature
    model_ref: str
    renderer_ref: str
    optional_extra: str | None = None
    result_projection: Literal["legacy_none_show", "artifacts"] = "legacy_none_show"
    by_group_variants: tuple[str, ...] = ()


_TEAR_SHEET_NAMES = (
    "create_event_returns_tear_sheet",
    "create_event_study_tear_sheet",
    "create_full_tear_sheet",
    "create_information_tear_sheet",
    "create_returns_tear_sheet",
    "create_summary_tear_sheet",
    "create_turnover_tear_sheet",
)


def _workflow_variants(source_signature: inspect.Signature) -> tuple[str, ...]:
    """Freeze only source-visible by-group branches for Task 8 to implement."""

    if "by_group" not in source_signature.parameters:
        return ()
    return ("by_group=False:show-close", "by_group=True:show-close")


def _make_workflow_specs() -> Mapping[str, FactorWorkflowSpec]:
    specs: dict[str, FactorWorkflowSpec] = {}
    for name in _TEAR_SHEET_NAMES:
        function_spec = ALPHALENS_FUNCTION_SPECS[("tears", name)]
        specs[name] = FactorWorkflowSpec(
            public_name=name,
            introspection_signature=function_spec.introspection_signature,
            source_signature=function_spec.source_signature,
            model_ref=f"fincore.factor_analysis.tears:{name}",
            renderer_ref=f"fincore.factor_analysis.render_matplotlib:{name}",
            optional_extra="pyfolio",
            by_group_variants=_workflow_variants(function_spec.source_signature),
        )
    if len(specs) != 7:
        raise RuntimeError(f"expected 7 pinned Alphalens tear-sheet workflows, found {len(specs)}")
    return MappingProxyType(specs)


ALPHALENS_WORKFLOW_SPECS: Mapping[str, FactorWorkflowSpec] = _make_workflow_specs()


__all__ = ["ALPHALENS_WORKFLOW_SPECS", "FactorWorkflowSpec"]
