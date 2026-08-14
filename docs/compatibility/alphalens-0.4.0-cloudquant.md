# Alphalens cloudQuant local compatibility profile

This is the frozen source target for the Beta `fincore.alphalens` strict
façade and its separate `fincore.factor_analysis` enhanced workflow. The
machine-readable source of truth is
[`tests/compat/fixtures/alphalens-0.4.0-cloudquant-api.json`](../../tests/compat/fixtures/alphalens-0.4.0-cloudquant-api.json).
Only the strict-path/signature and enhanced kernel/workflow behavior exercised
by the repository's executable tests is claimed; this page is not a statement
of full standalone Alphalens compatibility.

## Pinned identity

| Item | Frozen value |
| --- | --- |
| Profile | `cloudquant-local-3fa17ad` |
| Authoritative identity | Git commit `3fa17ad4c3edb025d1410de7aeba9673cba7791c` |
| Static extraction | Bounded AST parsing of `git show <commit>:<path>` bytes |
| Source files | `alphalens/__init__.py` plus `performance.py`, `utils.py`, `plotting.py`, and `tears.py` |
| Evidence files | root `LICENSE`, `README.md`, `setup.py`, and `alphalens/_version.py` |
| Surface | 61 functions + 3 classes = 64 definitions |

The fixture stores both each pinned Git blob object ID and SHA256 of the blob
contents. It deliberately stores only repository-relative paths: the sibling
checkout is an optional generation input, never a runtime or CI dependency.

## Version ambiguity

The file name includes `0.4.0` because the pinned `_version.py` embeds the
Versioneer tag `v0.4.0`. It is not a reliable release identity for this
snapshot:

- `setup.py` has a static fallback of `1.0.0+dev`.
- `_version.py` embeds an older revision (`77084f1...`) that is not the pinned
  snapshot identity.
- the local checkout has no tag that identifies the pinned commit itself.

For all compatibility, provenance, and acceptance decisions, the full commit
is the identity. The two version strings are source facts recorded with their
own blob evidence; neither overrides the commit.

## Static compatibility contract

| Module | Functions | Classes | Executable-test boundary |
| --- | ---: | ---: | --- |
| `performance` | 16 | 0 | strict public paths/signatures plus targeted numerical kernels |
| `utils` | 17 | 2 | strict public paths/signatures plus targeted cleanup/data kernels |
| `plotting` | 21 | 0 | strict public paths/signatures; rendering only where a targeted workflow exercises it |
| `tears` | 7 | 1 | strict public paths/signatures plus targeted figure/workflow tests |

Every entry records its module, symbol, kind, source line and SHA256,
source-visible signature, predicted `inspect.signature` form, accepted-call
grammar, and C0–C4 status. Freezing a target alone is not compatibility proof:
read each executable test scope for the actual level and behavior it covers.

The restricted AST logic recognizes the pinned decorators without importing
the sibling package. `quantize_factor` preserves its source signature while
recording the legacy wrapper form `(*args, **kwargs)`. The seven
`@plotting.customize` tear sheets preserve their source-visible signatures and
record the hidden accepted `set_context=True` and `set_context=False` calls:

- `create_summary_tear_sheet`
- `create_returns_tear_sheet`
- `create_information_tear_sheet`
- `create_turnover_tear_sheet`
- `create_full_tear_sheet`
- `create_event_returns_tear_sheet`
- `create_event_study_tear_sheet`

Dynamic defaults such as `stats.norm` are marked for dynamic review instead of
being fabricated as canonical runtime values.

## Deterministic case inputs

[`alphalens-0.4.0-cloudquant-cases.json`](../../tests/compat/fixtures/alphalens-0.4.0-cloudquant-cases.json)
contains deliberately small JSON-serializable inputs for later C2–C4 work:
daily and business-day calendars, intraday and UTC-aware indexes, ties/NaNs/
zeros, group-neutral weights, bins and quantiles, `max_loss`, pre-cleaned
performance data, event windows, and Pyfolio inputs.

Each table records index values and names, timezone, columns, dtypes, values,
and a separate NaN mask using `fincore-compat-json-table-v1`. The inputs have
received a schema-only review. They intentionally contain no expected output
or numerical golden result, so they cannot be mistaken for a reviewed oracle.

## Pinned upstream-test migration inventory

[`alphalens-0.4.0-cloudquant-upstream-test-inventory.json`](../../tests/compat/fixtures/alphalens-0.4.0-cloudquant-upstream-test-inventory.json)
is a static AST-only inventory of the three pinned upstream test blobs. Its
separate, human-reviewable handoff map is
[`alphalens-0.4.0-cloudquant-upstream-test-migration.json`](../../tests/compat/fixtures/alphalens-0.4.0-cloudquant-upstream-test-migration.json).
Neither file imports or executes Alphalens, `parameterized`, Matplotlib, or a
source-side test module.

| Frozen upstream test source | Git blob | SHA256 |
| --- | --- | --- |
| `tests/test_utils.py` | `22480c305a07b8ccd83e15ed7b6d1b06be08307e` | `0f476933684b1eae8f86c3ce9dcf3806b840cc69a1005e19f43a52d4bdf31334` |
| `tests/test_performance.py` | `5f38d92b936f3b7f0afb0b4d63a84edd347766a1` | `278ecc858a228e686edd6e8aa4ef30d42fe7258a9af5da14263de61607474917` |
| `tests/test_tears.py` | `8c1b74705e89ae3fe090049120c06d34fe7f13fd` | `227d23e8eebb3585b29f5f953e67f817517d802148f3e72c0cf8b27087853b86` |

The inventory has 117 declared active source rows across 22 methods; one
Performance row is explicitly marked
`shadowed_by_generated_method_name`, leaving 116 diagnostic-collectible rows.
The otherwise commented `TearsTestCase` is parsed only after removing the
comment prefix inside that class block: it contributes 24 dormant rows, seven
workflows, and 96 individually named internal invocations. Its source outcome
is `smoke_only`; it is not treated as an exception from migration.

The mapping covers all 141 source row IDs. Utils and Performance entries point
to the future Task 3/4 target suites with C2/C3 target assertions. Tear rows
point to the future Task 8 C4 suites, and each of their 96 invocation IDs has
one unique exact future pytest nodeid. The map records the upstream discarded
`.equals()` assertions as source evidence, not as an accepted target
assertion.

Task 1.5 freezes this contract but does not create placeholder target tests or
claim that any future target node was collected or passed. Once Tasks 3, 4,
and 8 create those tests, first use the checker's
`--write-collection-proof PATH` wrapper for the selected scope. It runs the
exact scope-owned `pytest -o addopts= --collect-only -q` command and writes a
versioned JSON envelope containing its command identity, scope, target paths,
exit status, nodeids, and collection errors. Pass that envelope back with
`--collection-proof PATH` and the non-xdist result file written by
`--alphalens-upstream-result-json`; a plain collection transcript is not
accepted. The writer accepts only a non-traversing relative `build/...` output
path and resolves it against the repository before creating a directory or
writing proof bytes. The checker accepts only the frozen `inventory-v1` and `migration-v1`
schemas, `cloudquant-local-3fa17ad` profile, exact static Git-blob extraction
record, and deferred-review envelope. Unknown, missing, or malformed envelope
fields fail closed. Each v1 inventory row also has an exact path-specific
shape and source class/state/assertion contract, per-row blob/SHA binding, and
canonical record-sequence digest; source method, line, ordinal, shadow, and
dormant tear invocation substitutions therefore fail closed. The dynamic
`alphalens_upstream_case(case_id)` marker rejects `skip`, `skipif`, `xfail`,
global `--reruns`, and per-item `flaky`/`rerun`/`rerunfailures` markers at
collection. It
records append-only setup/call/teardown attempts in the version-2 result JSON:
a non-passing phase in any attempt forces the pytest session to fail even if a
later rerun reports passed. The checker requires every phase in every recorded
attempt to be passed, so a later plugin ordering cannot replace earlier
failure evidence.

The deferred target AST audit also rejects direct or dynamic imports of
`alphalens` and the three upstream test modules, `sys.path` mutation, and
absolute sibling-upstream/source-test paths. Dynamic coverage includes direct
or imported-name aliases of `builtins.__import__`/`__builtins__.__import__`
and `importlib`, literal `getattr` or `builtins.__dict__` access to those import
APIs, and known `runpy`/`exec`/`eval` execution calls. It additionally follows
direct assignment aliases of already recognized `builtins`, `importlib`,
`runpy`, `pathlib`, `os`, and `sys` module names; this is not a general alias
or data-flow analysis. For those known sinks only, it examines their positional
first operand or the bounded named forms `name`, `mod_name`, `path_name`, or
`source`. It resolves a literal relative `importlib.import_module` name only
with a literal `package` context. It folds literal `Path`, `resolve()` or
`resolve(strict=False)`, `absolute()`, `joinpath`, `os.path.join`, direct
`from os.path import join` aliases, and string joins only when they are fed to
one of those sinks. These are deliberately finite AST patterns, not a general
dynamic-execution detector. They permit ordinary relative or fincore-local
imports, including other repository test packages that are not one of the
three pinned upstream source modules (`test_utils`, `test_performance`, and
`test_tears`, with their `tests.*` aliases).

Every Task 3/4 C2/C3 target must expose a direct `assert` or recognized
`pandas.testing`/`numpy.testing` assertion call in its reachable outer test
body. The same deliberately bounded walker used for C4 ignores nested
functions, classes, lambdas, comprehensions, and generator expressions; literal
false branches; code after an unconditional return/raise; unreachable sides of
literal short-circuit Boolean expressions; empty literal `for` bodies; and code
following a definitely nonempty literal `for` whose body unconditionally
returns/raises. This prevents an ordinary target from satisfying its assertion
contract only through demonstrably dead code; it is not a claim of full control
flow or symbolic analysis.

Every Task 8 C4 invocation target must present all of these statically
auditable signals in its own test function:

- Figure/Axes return or inspection, through a Figure/Axes property,
  `gca`/`gcf`, or one of
  `assert_figure_axes`, `assert_figure_artifacts`, `assert_axes_artifacts`,
  `assert_rendered_figure`, or `assert_tear_sheet_figures`.
- Both show and close handling, through `.show()`/`.close()` or the recognized
  `show_figure`, `show_owned_figures`, `assert_show_called`, `close_figure`,
  `close_owned_figures`, `assert_figures_closed`, or
  `assert_no_open_figures` helpers.
- Artifact/resource ownership or cleanup, through
  `assert_artifact_ownership`, `assert_owned_artifacts`,
  `assert_figure_ownership`, `assert_no_figure_leaks`,
  `assert_no_open_figures`, or `close_owned_figures`.

A bare `assert True` or numeric-only assertion cannot satisfy C4. Signals in
statically unreachable code (`if False`, literal-false loop/conditional
branches, after an unconditional return/raise, or after a literal-true loop
whose body unconditionally returns/raises) do not count. It also excludes the
unreached side of literal short-circuit Boolean expressions (`False and …`,
`True or …`), empty literal `for` bodies, and code following a definitely
nonempty literal `for` whose body unconditionally returns/raises. When a
helper or method exposes a concrete receiver/argument, the Figure/Axes,
show/close, and ownership signals must bind to the same resource; global pyplot
state is the limited recognized exception. Comprehensions and generator
expressions never contribute C4 evidence, and the literal-iterable check also
recognizes empty numeric `range(...)` forms such as `range(0)` and
`range(1, 1)`. This is a deliberately bounded AST rule, not a claim of full
symbolic execution. The checker then requires the exact mapped nodeids, literal
marker IDs, allowed provenance, the required target assertion contract, and
all-attempt-passed results.

To reproduce the static inventory or its map audit locally, use:

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python \
  scripts/generate_alphalens_upstream_test_inventory.py \
  --source /Users/yunjinqi/Documents/new_projects/alphalens \
  --commit 3fa17ad4c3edb025d1410de7aeba9673cba7791c \
  --check tests/compat/fixtures/alphalens-0.4.0-cloudquant-upstream-test-inventory.json

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python \
  scripts/check_alphalens_upstream_test_migration.py \
  --inventory tests/compat/fixtures/alphalens-0.4.0-cloudquant-upstream-test-inventory.json \
  --migration tests/compat/fixtures/alphalens-0.4.0-cloudquant-upstream-test-migration.json \
  --scope all
```

After the deferred target files exist, create collection evidence only through
the controlled wrapper, then use that same scope's passing marker-hook result:

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python \
  scripts/check_alphalens_upstream_test_migration.py \
  --inventory tests/compat/fixtures/alphalens-0.4.0-cloudquant-upstream-test-inventory.json \
  --migration tests/compat/fixtures/alphalens-0.4.0-cloudquant-upstream-test-migration.json \
  --scope all \
  --write-collection-proof build/alphalens-upstream-collection.json

# Run the same scope's future target tests non-xdist with
# --alphalens-upstream-result-json build/alphalens-upstream-results.json.

/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python \
  scripts/check_alphalens_upstream_test_migration.py \
  --inventory tests/compat/fixtures/alphalens-0.4.0-cloudquant-upstream-test-inventory.json \
  --migration tests/compat/fixtures/alphalens-0.4.0-cloudquant-upstream-test-migration.json \
  --scope all \
  --collection-proof build/alphalens-upstream-collection.json \
  --results build/alphalens-upstream-results.json
```

## Oracle boundary

The checked-in tuple is executable on the captured Darwin/arm64 platform:

- the Conda explicit lock has 19 exact package URLs with MD5 fragments;
- the pip lock has 41 exact requirements and wheel SHA256 values under
  `--require-hashes`; and
- the environment record stores all 59 Conda/pip distribution name, version,
  build, channel, and platform records plus raw and normalized runtime facts.

`scripts/generate_alphalens_oracle.py` creates a temporary prefix from those
locks, installs only the hashed pip tuple, clones a clean detached checkout of
the pinned commit, and executes the cases with that prefix. It removes inherited
`CONDA_*`, virtual-environment, and Python-path variables before executing the
prefix interpreter. The worker is launched as prefix Python with `-I`, with the
clean detached checkout as its working directory; the case path is resolved
before that directory change. Thus caller CWD, `PYTHONPATH`, and
`sitecustomize` cannot select an unintended upstream package. Every Conda, pip,
Git, and worker command has its own process session; on timeout the runner
terminates and reaps only that session's process group before temporary-prefix
cleanup. It rejects dirty sources, commit/blob/lock mismatches and raw or
normalized runtime-fingerprint mismatches; it never installs into the Anaconda
base environment.

The table serializer restores the stored IANA timezone onto the timestamp level
of a single or multi-index, preserving named zones and the distinct instants on
DST transitions. It preserves index names, row order, and non-time indexes;
timezone is `null` for a non-time index.

Darwin is retained in `runtime.raw.platform.system`; the corresponding portable
name is `macOS` in `runtime.normalized.platform.os`, with `raw_system=Darwin`
to make that translation auditable. The BLAS record is deliberately semantic
(name/version/configuration), avoiding captured filesystem build paths.

The tuple is still **unreviewed**. Running the command writes a transient
candidate marked `reviewed=false`; it does not copy a candidate result into the
fixture. A human may mark the three matching review attestations reviewed only
after inspecting the candidate output and recording reviewer, date, candidate
digest, environment digest, and the combined evidence key. Any source/API,
case, environment, Conda lock, pip lock, candidate digest, or environment
digest change invalidates that attestation.

The true tuple execution is also available as an opt-in integration test (it
creates a temporary prefix and can take several minutes):

```bash
FINCORE_RUN_ALPHALENS_ORACLE_E2E=1 \
  /Users/yunjinqi/opt/anaconda3/bin/conda run -n base python -m pytest \
  -o addopts='' tests/compat/test_manifest_integrity.py::test_alphalens_oracle_executed_tuple_end_to_end -q
```

## Reproduction

Only this target is generated by the following command; legacy Empyrical and
Pyfolio fixture bytes are not opened or rewritten.

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python \
  scripts/generate_compat_manifest.py \
  --alphalens-root "$ALPHALENS_ROOT" \
  --target alphalens \
  --output tests/compat/fixtures
```

The integrity test also regenerates into a temporary directory twice and
checks byte idempotence. It does not import Alphalens and can remain an
offline, static CI check once the frozen fixture is committed.

To create an unreviewed candidate from the complete tuple, use the plan's
command with an output outside the source checkout, for example:

```bash
/Users/yunjinqi/opt/anaconda3/bin/conda run -n base python \
  scripts/generate_alphalens_oracle.py \
  --source "$ALPHALENS_ROOT" \
  --commit 3fa17ad4c3edb025d1410de7aeba9673cba7791c \
  --environment tests/compat/oracle/alphalens-0.4.0-cloudquant-environment.json \
  --explicit-lock tests/compat/oracle/alphalens-0.4.0-cloudquant-conda-explicit.txt \
  --cases tests/compat/fixtures/alphalens-0.4.0-cloudquant-cases.json \
  --output /tmp/alphalens-oracle-candidate.json
```

## Provenance and license boundary

The root `LICENSE` observed at this commit is MIT text, while each of the four
core source modules carries a Quantopian Apache-2.0 header. Repository history
also contains a commit described as copying code from the official site. These
are engineering provenance facts, not a legal determination. Human/license
review remains pending before copying, adapting, notices, SPDX text, or a
release claim. It is a release blocker: no Alphalens NOTICE decision is
invented here. See [upstream provenance](../upstream-provenance.md).
