# Empyrical 0.6.0 compatibility baseline

This page defines the target; it is not a claim that fincore 0.3.0 already
implements it. The machine-readable source of truth is
[`tests/compat/fixtures/empyrical-0.6.0-api.json`](../../tests/compat/fixtures/empyrical-0.6.0-api.json).

## Pinned target

| Item | Frozen value |
| --- | --- |
| Upstream version | `0.6.0` |
| Upstream commit | `74655e974ed2935563820c548c339731f1fe0621` |
| Public symbols | 54 |
| Callable symbols | 49 |
| Constants | `DAILY`, `WEEKLY`, `MONTHLY`, `QUARTERLY`, `YEARLY` |
| Extraction | Static AST; sibling package is not imported |

Every symbol in the JSON has an explicit C0–C4 status. At this baseline all
implementation levels are `not-verified` (C1 is `not-applicable` to constants).
The `target_evidence` field separately records which upstream source and
signature facts were frozen. This distinction prevents an upstream manifest
from being mistaken for fincore compatibility proof.

## Compatibility levels

| Level | Required evidence | Baseline status |
| --- | --- | --- |
| C0 | Public path resolves in fincore | Not verified |
| C1 | Parameter name, order, kind, and default match | Not verified |
| C2 | Input immutability, type/shape/index/dtype, and exceptions match | Not verified |
| C3 | Numeric, NaN/Inf, timezone, and boundary behavior match | Not verified |
| C4 | Cross-layer workflow and output contract match | Not verified |

Nine rolling callables (`roll_alpha`, `roll_alpha_aligned`,
`roll_alpha_beta_aligned`, `roll_annual_volatility`, `roll_beta`,
`roll_beta_aligned`, `roll_max_drawdown`, `roll_sharpe_ratio`, and
`roll_sortino_ratio`) are created by upstream factories. Their template
signatures are statically frozen, but `needs_dynamic_review=true` and
`reviewed=false` remain until an isolated oracle run is reviewed by a person.

## Reproduction

Run the generator against checkouts at the pinned commits:

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python \
  scripts/generate_compat_manifest.py \
  --empyrical-root "$EMPYRICAL_ROOT" \
  --pyfolio-root "$PYFOLIO_ROOT" \
  --output tests/compat/fixtures
```

CI consumes only the frozen JSON. It does not need either sibling checkout,
network access, or an oracle environment. The optional oracle requirements are
in `tests/compat/oracle/requirements-empyrical-0.6.0.txt`; oracle output is
unreviewed evidence until a reviewer deliberately sets `reviewed=true`.
