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

The compatibility status is limited to current executable strict-path,
signature, kernel, and workflow tests. The human license/NOTICE review remains
a release blocker; this page makes no legal conclusion. See the full
[migration guide](../getting-started/migration.md) and the repository
[`docs/MIGRATION.md`](https://github.com/cloudQuant/fincore/blob/master/docs/MIGRATION.md)
for the detailed map.
