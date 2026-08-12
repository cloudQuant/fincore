# Upstream provenance and review register

This register records concrete source inputs and likely adapted destinations.
It supports engineering traceability but is not legal advice or a license
determination.

## Frozen upstream inputs

| Upstream | Commit | File | SHA256 | Observed notice |
| --- | --- | --- | --- | --- |
| empyrical 0.6.0 | `74655e974ed2935563820c548c339731f1fe0621` | `empyrical/__init__.py` | `c0f115ede515fbc1216f5a34d45a9d8d4c813f6e1d73871415eb246b4ade4127` | Apache-2.0 header |
| empyrical 0.6.0 | same | `empyrical/stats.py` | `c8edee822c26efc0b6b52eb853e0cc39d828a48afa876f6b6d51709504cb7311` | Apache-2.0 header |
| empyrical 0.6.0 | same | `empyrical/periods.py` | `c1650af00a46001d89dae4c8aaf01e92bda573f4b2999ccb46ebd06868956a3c` | Apache-2.0 header |
| empyrical 0.6.0 | same | `empyrical/perf_attrib.py` | `859ce666e3160e84ce9fb409a854d1e6c01cf3df979246967825129909d72243` | Apache-2.0 header |
| empyrical 0.6.0 | same | root `LICENSE` | `2b651c2d29c644d1c73417cd96e6cfa506d6de102ed96fc3de96d676e089cf29` | Apache License 2.0 text |
| pyfolio 0.9.6 | `724bbd7dbed9a88bb47e1057f2ca29b3409d8e7a` | `pyfolio/__init__.py` | `92d4418efc129627e49526b1665f50bbccea9661f4d42908b98dd29f71a818be` | no license header in inspected file |
| pyfolio 0.9.6 | same | `pyfolio/tears.py` | `14e75d15c933c022d698c6cad454e0d4b5187fff5c5ae129eeae08e53297995c` | Apache-2.0 header |
| pyfolio 0.9.6 | same | `pyfolio/utils.py` | `b4423661845a1ece0e6f59dcade2f87d255b696088cdc71e8cc8db8caeff4b6b` | no license header in inspected file |
| pyfolio 0.9.6 | same | `pyfolio/plotting.py` | `b15a1ed427417ba53811e9c561da6143130f5585c0e9a132f32bb99132e7cd0e` | Apache-2.0 header |
| pyfolio 0.9.6 | same | `pyfolio/perf_attrib.py` | `a4436cc02a9f345ea2e238e914b255c72f5fca57305cd965228309a0f25a04c6` | Apache-2.0 header |
| pyfolio 0.9.6 | same | root `LICENSE` | `4391163aa82bbe18b3d5c9670d3b08e0f249d966d689874c3306bcfc91e51941` | MIT text |

Paths above are checkout-relative. Manifest source bytes and hashes come from
the pinned Git blobs, not potentially dirty worktree files. Absolute sibling
locations are deliberately absent from frozen fixtures.

## Adaptation inventory

These are code-lineage review targets, not assertions that every destination
line is copied. A human reviewer should compare the pinned inputs before a
release notice is finalized.

| Source family | Fincore copied/modified candidates | Review state |
| --- | --- | --- |
| `empyrical/stats.py`, `periods.py` | `fincore/metrics/{alpha_beta,basic,drawdown,ratios,returns,risk,rolling,stats,yearly}.py`, `fincore/constants/` | Engineering provenance recorded; line-level/license review pending |
| `empyrical/perf_attrib.py` | `fincore/metrics/perf_attrib.py`, `fincore/empyrical.py` | Engineering provenance recorded; line-level/license review pending |
| `pyfolio/tears.py`, `plotting.py` | `fincore/tearsheets/`, `fincore/pyfolio.py` | Engineering provenance recorded; line-level/license review pending |
| pyfolio portfolio helpers | `fincore/metrics/{positions,transactions,round_trips}.py` | Engineering provenance recorded; line-level/license review pending |

Several fincore metric files retain Quantopian and Apache-2.0 headers. Other
candidate destination files do not carry equivalent per-file headers, so the
repository-level license alone is not treated as sufficient provenance proof.

## Required human/license decision

Before a release claim or notice file is finalized, a qualified reviewer must:

1. resolve the pyfolio root-license/source-header inconsistency;
2. decide the required attribution, notice, and SPDX treatment for modified or
   copied files;
3. review the adaptation inventory at file or line level; and
4. decide whether `THIRD_PARTY_NOTICES.md` is required and approve its content.

That review has not occurred in this task. Consequently no optional notice file
is generated and no legal conclusion is implied.
