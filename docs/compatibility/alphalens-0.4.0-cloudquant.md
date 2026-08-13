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

The checked-in Conda explicit lock is an exact, hash-bearing observation of
the current Darwin/arm64 base environment. It is **not** an executable oracle:
the observed environment includes PyPI-installed dependencies with no reviewed
wheel hashes, including `alphalens`, `empyrical`, and `pandas-datareader`.
The corresponding pip requirements file is intentionally comment-only rather
than pretending to support `pip --require-hashes`.

`scripts/generate_alphalens_oracle.py` validates a supplied source checkout's
clean state, exact HEAD, pinned Git blobs, fixture schema, environment/lock
digests, and (when approved) portable runtime fingerprint. It never checks out
source, creates an environment, installs packages, or imports the sibling
package. With the current metadata it must refuse to create a candidate because
the environment is explicitly `unreviewed-current-base-observation`.

A future review may mark an isolated execution tuple reviewed only after it
has a complete Conda lock, a complete pip `--require-hashes` lock, matching
source blobs/fingerprint, and separately reviewed candidate output digest.
Changing the manifest's static evidence invalidates the retained oracle review
attestation.

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

## Provenance and license boundary

The root `LICENSE` observed at this commit is MIT text, while each of the four
core source modules carries a Quantopian Apache-2.0 header. Repository history
also contains a commit described as copying code from the official site. These
are engineering provenance facts, not a legal determination. Human/license
review remains pending before copying, adapting, notices, SPDX text, or a
release claim; see [upstream provenance](../upstream-provenance.md).
