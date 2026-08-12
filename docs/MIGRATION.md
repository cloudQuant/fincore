# Migrating from empyrical to fincore 0.3.0

Fincore's current package version is **0.3.0**. Do not add
`fincore>=1.0.0`: that release does not exist in the repository's current
version history.

## Compatibility status

Migration is currently symbol-by-symbol, not a certified drop-in replacement.
The empyrical 0.6.0 target contains 54 public symbols (49 callables), while the
pyfolio 0.9.6 target is a bounded profile of 11 tear-sheet workflows. Task 2
freezes their upstream source and signatures; it does not prove that fincore
implements C0–C4 compatibility.

- [Empyrical 0.6.0 matrix](compatibility/empyrical-0.6.0.md)
- [Pyfolio 0.9.6 profile](compatibility/pyfolio-0.9.6.md)
- [Upstream provenance and license-review register](upstream-provenance.md)

Compatibility levels are C0 (public path), C1 (signature), C2 (structural and
exception behavior), C3 (numeric behavior), and C4 (cross-layer workflows).
Every frozen entry currently says `not-verified` for implementation levels;
constants use `not-applicable` for C1.

## Installation

```bash
pip install fincore

# Optional visualization dependencies
pip install "fincore[viz]"
```

For repository development, install the checked-out 0.3.0 source:

```bash
pip install -e ".[dev,viz]"
```

## 0.3.x flat API policy

The existing `from fincore import ...` mappings remain unchanged throughout
0.3.x. They point to enhanced `fincore.metrics` implementations and are not
automatically identical to empyrical signatures or edge-case behavior. The
complete generated mapping, including `current_target`, `recommended_target`,
`deprecate_in`, and `remove_or_switch_in`, is in
[`tests/compat/fixtures/fincore-flat-api-migrations.json`](../tests/compat/fixtures/fincore-flat-api-migrations.json).

No flat API deprecation is scheduled. A switch to strict-compatible targets is
only a candidate for an unscheduled future major release and requires an
explicit deprecation period first.

## Safe migration workflow

1. Pin your current empyrical version and record the symbols you use.
2. Find each symbol in the empyrical compatibility matrix and frozen JSON.
3. Do not migrate a production call until its required C-level has executable
   evidence (normally C3 for a metric and C4 for a report workflow).
4. Run differential tests on your own NaN, Inf, timezone, empty-input, and
   alignment cases.
5. Keep enhanced fincore APIs explicit; do not assume that an equal function
   name means equal signature or behavior.

For code that intentionally uses the current enhanced flat API, this example
executes on fincore 0.3.0:

```python
import pandas as pd

from fincore import max_drawdown, sharpe_ratio

returns = pd.Series([0.01, -0.005, 0.002, 0.004])
print(sharpe_ratio(returns))
print(max_drawdown(returns))
```

## Known migration boundaries

- `fincore.empyrical` is the strict-compatibility target surface, but Task 2
  does not certify its symbols or signatures. Follow later matrix updates.
- `fincore.metrics` and the 0.3.x flat API are enhanced surfaces with
  semver-managed differences.
- The pyfolio profile covers only the 11 listed functional workflows, not the
  entire upstream package or the enhanced `Pyfolio` class.
- Compatibility workflows intentionally must not write into the installed
  package directory. This safety difference is tracked rather than hidden.
- Upstream pyfolio license metadata is inconsistent: its root `LICENSE` has MIT
  text while inspected source files have Apache-2.0 headers. Human/license
  review remains required; this guide makes no legal determination.

## Existing applications

Do not perform a blind replacement such as `import empyrical` to
`import fincore`. First add differential tests, then migrate individual imports
only after the relevant matrix rows reach the required level. `AnalysisContext`,
`RollingEngine`, optimization, and visualization APIs are fincore features, not
empyrical compatibility guarantees.
