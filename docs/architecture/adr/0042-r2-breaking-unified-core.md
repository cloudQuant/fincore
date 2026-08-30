# ADR-0042-R2: Breaking Unified Core for Fincore 0.5+

- **Status:** `D-ID` / `D-BREAK` are `PASSED (local decision)` under the
  user's explicit breaking-policy direction.  This is not a D0, D-TECH, or
  release approval.
- **Date:** 2026-08-30
- **Iteration:** 0042-R2
- **Target version:** `0.5.0.dev0`
- **Supersedes:** ADR-0042 for Fincore 0.5+
- **Decision authority:** User direction to retain analysis capabilities while
  ending legacy API compatibility.  That explicit direction passes the local
  D-ID/D-BREAK product decision only; it does not imply D0, D-TECH, release,
  merge, publish, or remote-action approval.

## Context and lineage

[ADR-0042](0042-unified-operation-model.md) remains an accepted, historical
decision for Fincore 0.4: it allowed compatibility façades and their
multi-surface bindings during the prior migration.  It is not edited or
reinterpreted by this ADR.  For Fincore 0.5+, this ADR supersedes that
compatibility-preserving decision with a breaking product contract.

The corresponding implementation proposal is the
[0042-R2 plan](../../plans/2026-08-30-fincore-0042-r2-breaking-unified-core.md).
This ADR records product direction and the local decision that opens Task 0.
It does not assert that D0, D-TECH, or release work has passed, and it
authorizes no remote action.

## Decision

Fincore 0.5 preserves required financial-analysis capabilities and their
observable financial semantics; it does **not** preserve the way those
capabilities were previously presented through Empyrical, Pyfolio, or
Alphalens compatibility surfaces.

The `0.5.0.dev0` contract is therefore a breaking contract with these rules:

1. Public functionality is organized through domain namespaces.  The root
   package exposes versioning, errors, and approved domain namespaces only; it
   does not retain flat functions, classes, dynamic aliases, or lazy legacy
   exports.
2. Every public required leaf capability or workflow is identified through a
   canonical `operation_id` and is reached through its one canonical domain
   implementation.  A legacy path must not be recreated as an alias, shim,
   hidden `sys.modules` entry, transition wheel, or compatibility wrapper.
3. Public failures use structured, documented error categories.  Exact legacy
   exception text, class identity, parameter ordering, MRO, descriptor shape,
   and state-binding behavior are not compatibility obligations.
4. Optional dependencies use capability-oriented extras.  Legacy extra and
   profile names are not aliases for new extras; new names are approved by
   capability rather than by the former Empyrical/Pyfolio/Alphalens product
   families.
5. The final source tree, wheel, Catalog, registries, profiles, root exports,
   maintained executable documentation, and package metadata must contain no
   legacy Empyrical/Pyfolio/Alphalens APIs, import paths, façades, registries,
   profiles, root aliases, or old extras.

The exact namespace list, `operation_id` registry, error-category taxonomy,
capability-extra mapping, and migration policy must be captured as immutable
Task 0/D0 inputs.  The local D-BREAK decision fixes the breaking product
direction and target version; it does not pretend those D0 artifacts, or the
later technical evidence that verifies them, already exist.

## Local decision and technical boundary

The user's explicit direction marks `D-ID` and `D-BREAK` as
`PASSED (local decision)`.  The development-readiness record must still prove
the clean Task 0 entry identity: caller-supplied worktree root, `dev` branch,
clean status, full expected HEAD, plan SHA256, and recorded baseline ancestry.

`D0` remains `NOT STARTED` until Task 0 captures a fresh clean exact-SHA
bundle with the capability ledger, independent oracle/golden inputs, quality,
architecture, performance, and provenance evidence required by the plan.
`D-TECH` and `D-RELEASE` likewise remain `NOT STARTED`.

The current local-decision status and exact preflight commands are recorded in
[0042-R2 development readiness](../../quality/0042-r2-development-readiness.md).
No completed technical or release result is inferred by this document.

## Consequences and authorization boundary

The historical ADR and the historical 0042 acceptance record remain unchanged,
including the latter's `BLOCKED` conclusion.  README and historical-plan status
pointers remain unchanged until the relevant gates actually pass.

This documentation record authorizes neither merge, push, tag, publish, nor
remote-configuration changes.  This commit itself changes no production code,
CI, package, or test files.  Any Task 0 execution must use the clean proven
worktree identity and satisfy the R2 plan's D0, parity, cutover, and exact-SHA
acceptance gates before a technical or release conclusion.

## See also

- [Historical ADR-0042](0042-unified-operation-model.md)
- [Historical 0042 acceptance](../../quality/2026-08-21-unified-platform-acceptance.md)
- [0042-R2 implementation plan](../../plans/2026-08-30-fincore-0042-r2-breaking-unified-core.md)
- [0042-R2 development readiness](../../quality/0042-r2-development-readiness.md)
