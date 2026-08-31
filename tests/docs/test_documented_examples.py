"""Executable named examples embedded in the MkDocs guide pages."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(os.environ.get("FINCORE_0042R2_SOURCE_ROOT", Path(__file__).resolve().parents[2])).resolve()
DOCS = _ROOT / "mkdocs_docs"

_CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def run_markdown_example(path: str | Path, name: str) -> dict[str, Any]:
    """Extract and execute a named code block from a Markdown guide page.

    A named block is a Python fenced block containing a ``# -- <name>`` marker.
    The whole block runs in a fresh namespace preloaded with ``pd`` and ``np``
    (so its imports take effect), and the namespace is returned for assertions.
    """

    text = Path(path).read_text(encoding="utf-8")
    marker = f"# -- {name}"
    namespace: dict[str, Any] = {"pd": pd, "np": np}
    for match in _CODE_BLOCK_RE.finditer(text):
        block = match.group(1)
        if marker not in block:
            continue
        exec(block, namespace)
        return namespace
    raise ValueError(f"named example {name!r} not found in {path}")


def test_risk_validation_quickstart_runs_without_network() -> None:
    namespace = run_markdown_example(DOCS / "guide" / "risk-validation.md", name="minimal-backtest")

    result = namespace["result"]
    assert result.observations > 0
    assert result.exceptions == 1


def test_reproducible_research_example_runs_offline() -> None:
    namespace = run_markdown_example(DOCS / "guide" / "reproducible-research.md", name="snapshot")

    snapshot = namespace["snapshot"]
    assert snapshot.content_sha256
    manifest = snapshot.to_manifest()
    assert manifest["provider"] == "fixture"
