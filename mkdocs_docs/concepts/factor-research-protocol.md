# Factor Research Protocol

## Scope and API boundary

`fincore.factor_analysis` offers an enhanced point-in-time (PIT) input route
for new research. It is additive: `fincore.alphalens` remains the strict,
source-shaped compatibility facade and is not silently reinterpreted as a PIT
workflow.

The current PIT route establishes causal factor materialization and causal
factor-data preparation. It does not by itself certify a research result,
replace a versioned corporate-action source, model transaction costs or
capacity, or record every research trial. Those controls must be supplied by
the calling research protocol until their dedicated workflows are available.

## Event-time ledger

Call `materialize_pit_factor(observations, evaluation_dates)` with one or more
revisions per asset. The ledger has these required columns:

| Column | Meaning | Contract |
| --- | --- | --- |
| `asset` | Security identifier | Non-missing and hashable. |
| `as_of` | Time the underlying fact describes | Must not be later than `known_at`. |
| `known_at` | Time the research system learned the fact | Must not be later than `effective_from`. |
| `effective_from` | First time the factor is permitted for use | Must be on or before the evaluation timestamp to be selected. |
| `value` | Numeric factor observation | Finite only. |
| `in_universe` | Membership revision | Boolean; a selected `False` removes the asset. |

All timestamp columns must share the evaluation-date timezone (or all be
naive), and evaluation dates must be sorted and duplicate-free. The causal
ordering is:

```text
as_of <= known_at <= effective_from <= evaluation_date
```

For each `(evaluation_date, asset)`, fincore selects the latest eligible
revision ordered by `effective_from`, `known_at`, then `as_of`. It rejects
duplicate revisions with the same asset and event-time tuple, non-finite
values, time-order violations, and timezone mismatches rather than guessing.

## Prepare data without full-sample filtering

```python
from fincore.factor_analysis import prepare_pit_factor_data

prepared = prepare_pit_factor_data(
    observations,
    prices,
    evaluation_dates,
    periods=(1, 5),
    quantiles=5,
    max_loss=0.35,
)
```

`prepare_pit_factor_data` first materializes eligible values and then uses the
enhanced factor preparation kernel. It rejects `filter_zscore`; using a
full-sample forward-return distribution as a filter would permit future data
to affect historical eligibility. The returned `PreparedFactorData` still has
ordinary forward-return availability loss, so callers must inspect
`loss_report` and document the selection protocol.

## Post-analysis IC inference and FDR

After enhanced analysis, run the explicit post-analysis step rather than
reading a raw IC average as a discovery claim:

```python
from fincore.factor_analysis import analyze_factor, factor_model_inference

model = analyze_factor(prepared.data, periods=("1D", "5D"), include_pyfolio=False)
inference = factor_model_inference(model, alpha=0.05)
audit_table = inference.hypotheses
```

`factor_model_inference` consumes the model's stored aggregate date-by-period
IC snapshot; it does not recompute returns or weights. For each forward
period, the audit table records finite sample count, mean IC, a two-sided
Student-t statistic and p-value, Benjamini-Hochberg adjusted p-value, and the
rejection decision. The BH family includes only rows with at least two finite
IC observations. Untestable rows remain visible with `testable=False`, `NaN`
p/q values, and `rejected=False`; they must not be reported as non-findings.

This inference path assumes independent IC observations. It is not HAC or
clustered inference, does not pre-register a hypothesis family, and does not
replace a research-trial register. Callers must define the tested horizons and
factor family before viewing results, retain the returned audit table, and
state any dependence correction or trial policy that is not yet supplied by
the platform.

## Required research evidence

For each strategy or factor study, retain the source snapshot identity,
corporate-action and calendar versions, universe construction rule, evaluation
timestamps, and the exact factor ledger used. Add a future-perturbation test:
changing an observation that is not yet known/effective must not change any
earlier materialized factor value. The repository keeps an executable
adversarial fixture for this property in
`tests/numerical/test_factor_pit_materialization.py`.

PIT materialization does not remove the need for out-of-sample validation,
multiple-testing control, cost/slippage/borrow assumptions, capacity analysis,
or an explicit trial register. Treat an omitted control as undisclosed, not as
passing by default.
