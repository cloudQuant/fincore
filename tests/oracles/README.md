# Independent oracles

Property tests in `tests/property/` assert invariants for classes of inputs
rather than hand-selected examples. For critical families they cross-check
against a small, checked-in **independent oracle** that never calls the
function under test.

## What lives here

| Oracle | Used by | Independence rule |
| --- | --- | --- |
| `tests/property/test_time_series_contracts.py::_numpy_cumulative` | cumulative-return tests | Pure NumPy `cumprod(1 + r) - 1`; does not call `fincore.metrics.returns.cum_returns` |
| `tests/test_risk/fixtures/risk_backtest_cases.json` | risk backtest tests | Fixed, hand-verified exception counts and alignments |
| `tests/test_attribution/fixtures/` | attribution provider contracts | Injected fake transports; no network, no SDK import |

## Rules

1. An oracle must be **transparent**: a plain NumPy/pandas reference or a
   small fixed fixture, not another call into the function under test.
2. An oracle must be **stable**: no random values, no clock-dependent output,
   no network access.
3. Property tests run serially (`-n 0`) so shrinking and the global Hypothesis
   seed remain reproducible. Bounded examples (`max_examples`) keep CI time
   predictable.

To re-run the property suite:

```sh
python -m pytest -o addopts='' tests/property -q -n 0
```
