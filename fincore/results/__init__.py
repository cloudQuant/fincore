"""Unified result metadata, artifact lifecycle, and serialization."""

from __future__ import annotations

from fincore.results.artifacts import ArtifactBundle, IdempotentCloseMixin
from fincore.results.base import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    STATUS_UNSUPPORTED,
    AnalysisResult,
    ResultMetadata,
)
from fincore.results.serialization import SCHEMA_VERSION, from_json, to_json

__all__ = [
    "SCHEMA_VERSION",
    "STATUS_FAILED",
    "STATUS_SUCCESS",
    "STATUS_UNSUPPORTED",
    "AnalysisResult",
    "ArtifactBundle",
    "IdempotentCloseMixin",
    "ResultMetadata",
    "from_json",
    "to_json",
]
