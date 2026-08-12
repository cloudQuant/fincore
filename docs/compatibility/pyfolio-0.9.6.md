# Pyfolio 0.9.6 compatibility profile

This is a bounded compatibility target, not a claim of complete pyfolio
replacement. The frozen source of truth is
[`tests/compat/fixtures/pyfolio-0.9.6-api.json`](../../tests/compat/fixtures/pyfolio-0.9.6-api.json).

## Pinned target

| Item | Frozen value |
| --- | --- |
| Upstream version | `0.9.6` |
| Upstream commit | `724bbd7dbed9a88bb47e1057f2ca29b3409d8e7a` |
| Profile size | 11 callable workflows |
| Extraction | Static AST from `pyfolio/tears.py`; sibling package is not imported |

## Profile and current status

All entries start as unverified against fincore 0.3.0. Static signatures here
describe the pinned upstream target only.

| Public symbol | C0 | C1 | C2 | C3 | C4 |
| --- | --- | --- | --- | --- | --- |
| `create_full_tear_sheet` | Not verified | Not verified | Not verified | Not verified | Not verified |
| `create_simple_tear_sheet` | Not verified | Not verified | Not verified | Not verified | Not verified |
| `create_returns_tear_sheet` | Not verified | Not verified | Not verified | Not verified | Not verified |
| `create_position_tear_sheet` | Not verified | Not verified | Not verified | Not verified | Not verified |
| `create_txn_tear_sheet` | Not verified | Not verified | Not verified | Not verified | Not verified |
| `create_round_trip_tear_sheet` | Not verified | Not verified | Not verified | Not verified | Not verified |
| `create_interesting_times_tear_sheet` | Not verified | Not verified | Not verified | Not verified | Not verified |
| `create_capacity_tear_sheet` | Not verified | Not verified | Not verified | Not verified | Not verified |
| `create_bayesian_tear_sheet` | Not verified | Not verified | Not verified | Not verified | Not verified |
| `create_risk_tear_sheet` | Not verified | Not verified | Not verified | Not verified | Not verified |
| `create_perf_attrib_tear_sheet` | Not verified | Not verified | Not verified | Not verified | Not verified |

The compatibility façade may intentionally prevent package-directory writes.
That safety difference must remain documented and tested; it must not be
presented as exact side-effect compatibility. Enhanced report APIs and the
`Pyfolio` class are outside this strict 11-workflow surface.

## Review boundaries

The local pyfolio root `LICENSE` contains MIT text while inspected source files
carry Apache-2.0 headers. This inconsistency requires human/license review; no
license conclusion is made here and no `THIRD_PARTY_NOTICES.md` has been
created. See [upstream provenance](../upstream-provenance.md) for file hashes
and audit scope.

The optional isolated environment is described by
`tests/compat/oracle/requirements-pyfolio-0.9.6.txt`. CI does not create it or
import pyfolio; it reads the frozen JSON only.
