"""Audit provenance for enhanced strategy reports.

``ReportProvenance`` records *what* a report was computed from — code commit and
version, dependency versions, normalized configuration, and per-input shape,
time bounds and content hash — without ever copying the raw input data,
credentials, or absolute local paths into the manifest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any, Mapping
from urllib.parse import urlsplit

import pandas as pd

SCHEMA_VERSION = 1

# Keys that must never be recorded in an audit manifest, defensively stripped
# even if a caller accidentally passes them through configuration.
_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "authorization",
)
_OMIT = object()
_REDACTED = "[redacted]"


def _sha256_pandas(obj: pd.Series | pd.DataFrame) -> str:
    payload = obj.to_csv(index=True, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contains_secret_marker(value: str) -> bool:
    return any(marker in value.lower() for marker in _SECRET_KEY_MARKERS)


def _is_absolute_local_path(value: str) -> bool:
    """Recognize local absolute paths independently of the host platform."""

    text = value.strip()
    if text.startswith(("/", "\\", "~")):
        return True
    if PureWindowsPath(text).is_absolute():
        return True
    return urlsplit(text).scheme.lower() == "file"


def _sanitize_configuration_value(value: Any) -> Any:
    """Return a JSON-safe configuration value or ``_OMIT``.

    Configuration now includes structured report disclosures.  Sanitize every
    nested level so a note or field cannot bypass the manifest's credential and
    local-path guarantees.
    """

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested_value in value.items():
            name = str(key)
            if _contains_secret_marker(name):
                continue
            sanitized = _sanitize_configuration_value(nested_value)
            if sanitized is not _OMIT:
                result[name] = sanitized
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitized for item in value if (sanitized := _sanitize_configuration_value(item)) is not _OMIT]
    if isinstance(value, PurePath):
        return _OMIT
    if isinstance(value, str):
        if _is_absolute_local_path(value):
            return _OMIT
        if _contains_secret_marker(value):
            return _REDACTED
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value

    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is not value:
            return _sanitize_configuration_value(scalar)
    return _sanitize_configuration_value(str(value))


def _sanitize_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in configuration.items():
        name = str(key)
        if _contains_secret_marker(name):
            continue
        sanitized = _sanitize_configuration_value(value)
        if sanitized is not _OMIT:
            result[name] = sanitized
    return result


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _dependency_versions() -> dict[str, str]:
    import numpy
    import pandas
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
    }


def _input_record(obj: Any) -> dict[str, Any]:
    if isinstance(obj, pd.Series):
        return {
            "kind": "series",
            "length": len(obj),
            "sha256": _sha256_pandas(obj),
            "start": str(obj.index.min()) if len(obj) and isinstance(obj.index, pd.DatetimeIndex) else None,
            "end": str(obj.index.max()) if len(obj) and isinstance(obj.index, pd.DatetimeIndex) else None,
        }
    if isinstance(obj, pd.DataFrame):
        return {
            "kind": "dataframe",
            "rows": len(obj),
            "columns": len(obj.columns),
            "sha256": _sha256_pandas(obj),
        }
    return {"kind": type(obj).__name__}


@dataclass(frozen=True)
class ReportProvenance:
    """An audit manifest for a computed report."""

    schema_version: int = SCHEMA_VERSION
    code_commit: str = ""
    code_version: str = ""
    dependencies: dict[str, str] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        code_version: str,
        configuration: Mapping[str, Any],
        inputs: Mapping[str, Any],
    ) -> ReportProvenance:
        return cls(
            code_commit=_git_commit(),
            code_version=code_version,
            dependencies=_dependency_versions(),
            configuration=_sanitize_configuration(configuration),
            inputs={name: _input_record(value) for name, value in inputs.items() if value is not None},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "code_commit": self.code_commit,
            "code_version": self.code_version,
            "dependencies": self.dependencies,
            "configuration": self.configuration,
            "inputs": self.inputs,
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


__all__ = ["SCHEMA_VERSION", "ReportProvenance"]
