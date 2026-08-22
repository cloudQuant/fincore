# API Stability

Stability is claimed **only** for surfaces whose compatibility level has been
verified by the executable gates in `tests/compat/`. A surface not listed
below carries no broader guarantee.

## Stable surfaces

- **`fincore.empyrical`** — the frozen empyrical 0.6.0 surface: 54/54 public
  symbols (C0), 49/49 callable signatures (C1), core callables numerically
  verified (C3). C2/C3 is not claimed for symbols the contract suites do not
  exercise; the nine upstream-factory rolling callables remain
  `needs_dynamic_review=true` until an oracle run is human-reviewed.
- **`fincore.pyfolio`** — the frozen pyfolio 0.9.6 profile of 11 workflows:
  all entries C1; risk/returns/perf-attrib/full-sheet main chains C4. The
  `Pyfolio` class (requires `fincore[pyfolio]`) is enhanced OO convenience
  over the same workflows.
- **Flat API** (`from fincore import ...`) — stable within the current pre-1.0 series as an
  **enhanced** surface bound to `fincore.metrics`, not as empyrical equality.
- **`AnalysisContext`** — `fincore.analyze(...)`, metric properties,
  `perf_stats()`, `to_dict()`, `to_json()`, `to_json(path=...)`,
  `to_html(path=...)`, `plot()` (returns `ReportArtifacts`), and
  `replace_data()`.
- **`RollingEngine`** — `compute([...])` with the documented metric names.

## Not covered

### Alphalens migration surfaces

`fincore.alphalens` and `fincore.factor_analysis` are Beta migration APIs,
not Stable surfaces. Their source identity is pinned to commit
`3fa17ad4c3edb025d1410de7aeba9673cba7791c`; `v0.4.0` and `1.0.0+dev` are
ambiguous historical version strings. Claims are limited to the strict-path,
signature, kernel, and workflow behavior exercised by current executable
tests, not full standalone Alphalens parity. The unresolved human
third-party license/NOTICE decision is an advisory follow-up, not a CI/CD gate
or automated legal approval.

- `Empyrical`/`Pyfolio` methods beyond the frozen verified surface;
- enhanced-vs-legacy equality (documented divergences exist by design);
- modules prefixed with `_` (e.g. `fincore._registry`) — internal.

## Python support

fincore requires **Python 3.11+** — a documented breaking change relative to
empyrical. Exercised versions: 3.11, 3.12, 3.13.

## Versioning

- **Major (X.0.0)**: Breaking changes to stable APIs
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, backward compatible

## Deprecation Process

1. Mark as deprecated in docs + add warning
2. Keep functional for at least one minor version
3. Remove in next major version

See also: [Compatibility](compatibility.md), [Changelog](changelog.md).
