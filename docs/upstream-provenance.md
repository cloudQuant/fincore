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
| Alphalens cloudQuant local | `3fa17ad4c3edb025d1410de7aeba9673cba7791c` | `alphalens/__init__.py` | `4a93dbb33d372a5ed52232426056f16a9c297bff09db7216c94da87770824a07` | imports four public modules; no license header in inspected file |
| Alphalens cloudQuant local | same | `alphalens/performance.py` | `0c5ff3f6dc6a23c81a5f2c2bfa9b9860b3f9a8657fafe06be429629074b6a6ac` | Copyright 2017 Quantopian, Apache-2.0 header |
| Alphalens cloudQuant local | same | `alphalens/utils.py` | `adce74b07070f890e2567b1aedf67841ec69b9beeebc9aabee272ea850001871` | Copyright 2018 Quantopian, Apache-2.0 header |
| Alphalens cloudQuant local | same | `alphalens/plotting.py` | `56a793a44f975fe0f9a650f3a92648cdb8adeb8067557aff24dc9e6e3ce3bc15` | Copyright 2017 Quantopian, Apache-2.0 header |
| Alphalens cloudQuant local | same | `alphalens/tears.py` | `2d56f6c4f6545bac21d40378f8f63fcee0bfcc666877beb27a3cee6cf338ba05` | Copyright 2017 Quantopian, Apache-2.0 header |
| Alphalens cloudQuant local | same | root `LICENSE` | `c880f680840331b0c9b2b8968cd08faf26914b6efcd1a7a4afaba105248718d6` | MIT text, copyright line names 云金杞 |
| Alphalens cloudQuant local | same | root `README.md` | `397ec98f88157234e6a425b21657f74db711e4ee6aa246d11bed9d21665ac621` | project README advertises Apache-2.0 |
| Alphalens cloudQuant local | same | `setup.py` | `9a0192f4d1189524f568fd1d0a076e5c77cdd0b6bf12621724f76095bc0a0a81` | static fallback version `1.0.0+dev` |
| Alphalens cloudQuant local | same | `alphalens/_version.py` | `485407a5fb66fd94a9e8e4ff6a86c7c3346182b484ce16cdca11b931af1cf0dc` | Versioneer source embeds `v0.4.0` and older revision `77084f1...` |
| Alphalens cloudQuant local | same | `tests/test_utils.py` | `0f476933684b1eae8f86c3ce9dcf3806b840cc69a1005e19f43a52d4bdf31334` | upstream test source, Git blob `22480c305a07b8ccd83e15ed7b6d1b06be08307e` |
| Alphalens cloudQuant local | same | `tests/test_performance.py` | `278ecc858a228e686edd6e8aa4ef30d42fe7258a9af5da14263de61607474917` | upstream test source, Git blob `5f38d92b936f3b7f0afb0b4d63a84edd347766a1`; one parameterized row is source-shadowed |
| Alphalens cloudQuant local | same | `tests/test_tears.py` | `227d23e8eebb3585b29f5f953e67f817517d802148f3e72c0cf8b27087853b86` | commented upstream tear workflows, Git blob `8c1b74705e89ae3fe090049120c06d34fe7f13fd` |

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
| Alphalens `performance.py`, `utils.py` | planned `fincore/alphalens/{performance,utils}.py`, `fincore/factor_analysis/{data,performance,portfolio}.py` | Snapshot only; no destination implementation or license decision in this task |
| Alphalens `plotting.py`, `tears.py` | planned `fincore/alphalens/{plotting,tears}.py`, `fincore/factor_analysis/{render_matplotlib,tears}.py` | Snapshot only; no destination implementation or license decision in this task |
| Alphalens upstream test sources | planned Task 3/4/8 tests recorded in `tests/compat/fixtures/alphalens-0.4.0-cloudquant-upstream-test-migration.json` | Static source-to-target review map only; no target execution, copy, adaptation, or license decision in this task |

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

For Alphalens specifically, the reviewer must also resolve the root-MIT versus
file-level-Quantopian-Apache notices, inspect the historical `ff4d582` commit
message indicating a copy from the official site, and decide which headers or
notices apply to any future clean-room reimplementation versus adaptation.

## 2026-08-22 Fincore project-license decision

Fincore's own project license is MIT. This decision applies to Fincore-owned
contributions and project metadata only; it does not relicense retained
third-party source or vendored assets. Every source file that already carries
a Quantopian Apache-2.0 header retains that header. The distributable package
contains one Fincore `LICENSE` (MIT), plus `NOTICE`, this provenance inventory,
and `THIRD_PARTY_LICENSES/Apache-2.0.txt` for the independent Apache-2.0
obligations.

The version in `pyproject.toml` is the only Fincore product version. Entries
such as `empyrical 0.6.0` and `pyfolio 0.9.6` above are immutable upstream
source identifiers and must remain available for traceability.

The pinned upstream test inventory is engineering evidence, not a legal
conclusion about test text, derived fixtures, or target rewrites. Its 141-row
migration map is intentionally a deferred handoff to future test tasks; it
does not assert that a target suite has been copied, collected, executed, or
approved. When those target tasks begin, the migration checker will reject
direct or dynamic upstream/source-test imports, sibling-upstream absolute
paths assembled for a finite set of AST-visible `runpy`/builtins execution
APIs (including bounded named first-operand forms), direct assignment aliases
of those recognized module namespaces, imported `os.path.join` aliases, and
`sys.path` mutation. Its later collection gate accepts only the checker's
versioned controlled-collector proof (scope, command identity, zero exit
status, exact target paths/nodeids, and no collection errors), never a plain
`pytest --collect-only` transcript; its writer is restricted to a relative,
non-traversing file under the repository `build/` directory. Its C2/C3/C4 target checks use bounded
reachable-AST evidence rather than treating a nested or demonstrably dead
assertion as proof. These are bounded engineering safeguards, not evidence of
a legal review.

That review has not occurred in this task. The distributed NOTICE and Apache
license copy preserve observed attribution and terms. Pending review does not
block CI/CD, but it also does not create a release-approval claim or legal
conclusion.
