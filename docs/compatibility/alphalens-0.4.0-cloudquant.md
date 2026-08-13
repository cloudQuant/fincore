# Alphalens cloudQuant local compatibility profile

This is a frozen source target for the planned `fincore.alphalens` façade. It
is not a claim that the façade, factor-analysis kernels, numerical behavior,
or tear sheets have been implemented. The machine-readable source of truth is
[`tests/compat/fixtures/alphalens-0.4.0-cloudquant-api.json`](../../tests/compat/fixtures/alphalens-0.4.0-cloudquant-api.json).

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

| Module | Functions | Classes | Intended levels |
| --- | ---: | ---: | --- |
| `performance` | 16 | 0 | C0–C3 |
| `utils` | 17 | 2 | C0–C3 |
| `plotting` | 21 | 0 | C0–C2 plus figure semantics |
| `tears` | 7 | 1 | C0–C2 plus C4 |

Every entry records its module, symbol, kind, source line and SHA256,
source-visible signature, predicted `inspect.signature` form, accepted-call
grammar, and C0–C4 status. All current levels are `not-verified`: freezing a
target is not compatibility proof.

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
release claim; see [upstream provenance](../upstream-provenance.md).
