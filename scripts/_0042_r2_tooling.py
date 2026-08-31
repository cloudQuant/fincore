"""Shared candidate-source resolution for frozen 0042-R2 tooling.

The acceptance runner executes scripts from its immutable tooling checkout but
must measure the separate candidate checkout.  A task-specific environment
variable makes that boundary explicit without allowing a candidate to replace
the runner or its expected values.
"""

from __future__ import annotations

import os
from pathlib import Path

SOURCE_ROOT_ENV = "FINCORE_0042R2_SOURCE_ROOT"


def resolve_source_root(tooling_root: Path) -> Path:
    """Return the explicitly selected candidate root or the tooling root."""

    raw = os.environ.get(SOURCE_ROOT_ENV)
    if not raw:
        return tooling_root.resolve()
    candidate = Path(raw).expanduser().resolve()
    if not candidate.is_dir():
        raise ValueError(f"{SOURCE_ROOT_ENV} must identify an existing source directory: {candidate}")
    return candidate
