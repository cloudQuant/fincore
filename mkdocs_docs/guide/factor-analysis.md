# Factor-analysis migration

fincore offers two factor-analysis routes for the pinned cloudQuant-local
Alphalens source snapshot at commit
`3fa17ad4c3edb025d1410de7aeba9673cba7791c`:

- `fincore.alphalens` is the strict, source-shaped namespace for an existing
  Alphalens migration.
- `fincore.factor_analysis` is the enhanced namespace for new code. It
  separates preparation, immutable analysis, typed Pyfolio inputs, and
  caller-owned rendering artifacts.

The source contains conflicting version evidence (`v0.4.0` in Versioneer and
`1.0.0+dev` in `setup.py`), so use the pinned commit rather than either string
as the upstream identity. This is a Beta integration, not a full standalone
Alphalens compatibility claim. In particular, `import alphalens` is not
supported, and the first release has no notebook, HTML, or interactive-backend
workflow.

## Install and run offline

```bash
pip install "fincore[alphalens]"
MPLBACKEND=Agg python examples/factor_analysis_quickstart.py
```

The executable example uses fixed-seed synthetic data only, makes no network
requests, writes no default files, renders under Agg, and closes figures after
inspection. `fincore[factor-analysis]` is sufficient for compute-only enhanced
work; rendering and strict migration workflows require `fincore[alphalens]`.
If the rendering dependencies are absent, the actionable error instructs:

```text
pip install fincore[alphalens]
```

## Choose a route

| Goal | Strict route | Enhanced route |
| --- | --- | --- |
| Clean factor and forward returns | `fincore.alphalens.utils.get_clean_factor_and_forward_returns` | `prepare_factor_data` |
| Information coefficient | `fincore.alphalens.performance.factor_information_coefficient` | `factor_information_coefficient` |
| Full tear sheet | `fincore.alphalens.tears.create_full_tear_sheet` | `analyze_factor`, then `create_full_tear_sheet(model)` |
| Pyfolio handoff | legacy tuple + `fincore.pyfolio` | typed `PyfolioFactorInputs` |

For enhanced code, retain the `PreparedFactorData.loss_report`, pass its
`data` to `analyze_factor`, then explicitly manage `FactorTearSheetArtifacts`.
Artifacts are not shown or closed automatically.

## Research safeguards

- Strict cleanup retains `filter_zscore=20` as a source-shaped default. It can
  use future return information and introduce look-ahead bias; prefer
  `filter_zscore=None` unless the protocol explicitly permits it.
- Keep factor and price timezones compatible, retain the intended exchange
  calendar, and validate holidays/session frequency instead of assuming daily
  observations.
- Treat `max_loss` as a data-quality threshold. Inspect the loss report and
  justify any increase rather than accepting dropped observations silently.
- For new enhanced research with multiple return horizons, use
  `prepare_factor_data_by_horizon`. It creates one prepared table and one loss
  report per horizon, so missing long-horizon returns do not silently delete
  usable short-horizon observations. The strict Alphalens route intentionally
  keeps its source-shaped all-horizon cleanup behavior.

## Causal PIT inputs for enhanced research

New enhanced research should materialize factor values from a point-in-time
ledger instead of passing a prebuilt series whose availability cannot be
audited. `materialize_pit_factor` requires `asset`, `as_of`, `known_at`,
`effective_from`, `value`, and `in_universe` columns. On each evaluation date
it selects only the latest revision that was both known and effective on that
date; a later `in_universe=False` observation removes the asset rather than
leaking a stale value forward.

```python
import pandas as pd

from fincore.factor_analysis import materialize_pit_factor

observations = pd.DataFrame(
    {
        "asset": ["A", "B"],
        "as_of": pd.to_datetime(["2024-01-01", "2024-01-01"], utc=True),
        "known_at": pd.to_datetime(["2024-01-02", "2024-01-02"], utc=True),
        "effective_from": pd.to_datetime(["2024-01-02", "2024-01-02"], utc=True),
        "value": [1.0, -1.0],
        "in_universe": [True, True],
    }
)
factor = materialize_pit_factor(observations, pd.date_range("2024-01-02", periods=2, tz="UTC"))
```

Use `prepare_pit_factor_data` to materialize that ledger before forward-return
preparation. It deliberately rejects `filter_zscore`: a full-sample return
filter is not causal. The strict `fincore.alphalens` facade is unchanged and
retains its source-shaped options. See the [factor research protocol](../concepts/factor-research-protocol.md)
for the event-time contract, validation boundaries, and remaining scope.

The compatibility status is limited to current executable strict-path,
signature, kernel, and workflow tests. The human license/NOTICE review remains
a release blocker; this page makes no legal conclusion. See the full
[migration guide](../getting-started/migration.md) and the repository
[`docs/MIGRATION.md`](https://github.com/cloudQuant/fincore/blob/master/docs/MIGRATION.md)
for the detailed map.
