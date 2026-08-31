"""Runtime-owned type aliases shared by domain and orchestration code."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

ArrayLike: TypeAlias = np.ndarray | pd.Series | pd.DataFrame | Sequence[float]
ReturnOrDataFrame: TypeAlias = np.ndarray | pd.Series | pd.DataFrame
Schema: TypeAlias = Mapping[str, Any]
Scalar: TypeAlias = bool | int | float | str | None
