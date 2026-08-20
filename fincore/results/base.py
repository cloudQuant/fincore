"""Unified result metadata and the discriminant AnalysisResult.

Every enhanced high-level result is ``Success | Unsupported | Failed``.  The
metadata records *what* was computed, under which semantics, when, and with
what diagnostics — so a result can always answer "what did you compute, from
what, and is it trustworthy".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from fincore.exceptions import FincoreError

STATUS_SUCCESS = "success"
STATUS_UNSUPPORTED = "unsupported"
STATUS_FAILED = "failed"

__all__ = [
    "STATUS_FAILED",
    "STATUS_SUCCESS",
    "STATUS_UNSUPPORTED",
    "AnalysisResult",
    "ResultMetadata",
]


@dataclass(frozen=True)
class ResultMetadata:
    """Composable metadata for an enhanced result."""

    operation: str
    profile: str
    schema_version: str
    status: str = STATUS_SUCCESS
    units: str | None = None
    frequency: str | None = None
    sign: str | None = None
    input_digest: str = ""
    config_digest: str = ""
    software: dict[str, str] = field(default_factory=dict)
    dependency_provenance: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] | None = None


T = TypeVar("T")


@dataclass(frozen=True)
class AnalysisResult(Generic[T]):
    """A discriminant result: Success (typed value), Unsupported, or Failed."""

    status: str
    value: T | None = None
    metadata: ResultMetadata | None = None
    error: FincoreError | None = None

    @classmethod
    def success(cls, value: T, metadata: ResultMetadata) -> AnalysisResult[T]:
        return cls(status=STATUS_SUCCESS, value=value, metadata=metadata)

    @classmethod
    def unsupported(cls, metadata: ResultMetadata, reason: str) -> AnalysisResult[T]:
        metadata = ResultMetadata(**{**metadata.__dict__, "status": STATUS_UNSUPPORTED, "diagnostics": {**metadata.diagnostics, "reason": reason}})
        return cls(status=STATUS_UNSUPPORTED, metadata=metadata)

    @classmethod
    def failure(cls, error: FincoreError, metadata: ResultMetadata) -> AnalysisResult[T]:
        metadata = ResultMetadata(**{**metadata.__dict__, "status": STATUS_FAILED})
        return cls(status=STATUS_FAILED, metadata=metadata, error=error)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_SUCCESS
