"""Explicit test ownership for numerical kernel and upstream-oracle suites."""

from __future__ import annotations

import pytest

from fincore._dispatch import _raw_kernel_execution


@pytest.fixture(autouse=True)
def raw_metric_kernel_suite():
    """Bypass enhanced public validation while exercising numerical kernels.

    Public enhanced validation is covered by ``tests/contracts``.  Suites that
    import this fixture own the upstream NaN/empty oracle and low-level branch
    behavior, so their calls deliberately enter the same raw-composition guard
    used by strict compatibility adapters.
    """

    with _raw_kernel_execution():
        yield
