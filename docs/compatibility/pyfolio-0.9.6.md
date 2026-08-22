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
| Extraction | Static AST from pinned Git blobs for `__init__.py`, `tears.py`, and `utils.py` |

## Profile and current status

All entries start as unverified against fincore 0.4.0.dev0. Static signatures here
describe the pinned upstream target only. Restricted AST resolution evaluates
known constants and safe arithmetic (`last_n_days=126`) and the portable
`FACTOR_PARTITIONS` dictionary while retaining each `default_expression`.
Parsing is resource-bounded; Git and optional oracle processes are
noninteractive, time-limited, and report the timed-out operation by name.

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
final license conclusion is made here. The distribution now carries
`THIRD_PARTY_NOTICES.md`, `NOTICE`, and the Apache-2.0 text while that review
remains pending. See [upstream provenance](../upstream-provenance.md) for file
hashes and audit scope.

The optional isolated environment is described by
`tests/compat/oracle/requirements-pyfolio-0.9.6.txt`. Oracle mode imports from a
temporary checkout of the pinned commit and rejects an installed package with
the same name. CI does not create this environment or import pyfolio; it reads
the frozen JSON only. Human `reviewed=true` attestations survive regeneration
only while the exact evidence key is unchanged.
