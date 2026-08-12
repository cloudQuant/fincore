from __future__ import annotations

import subprocess
import sys

SUBPROCESS_TIMEOUT_SECONDS = 60


def test_import_benchmark_does_not_break_later_monkeypatches() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "tests/test_import_time.py",
            "tests/test_risk/evt/test_evt_cvar.py::TestEVTCVArEdgeCases::test_evt_cvar_gpd_xi_ge_1_raises_line_425",
            "-q",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stdout + result.stderr
