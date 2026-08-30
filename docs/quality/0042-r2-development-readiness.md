# 0042-R2 Development Readiness

**Status: `READY FOR TASK 0`**

This is a fail-closed, docs-only preflight record.  The user's explicit
breaking-policy direction marks `D-ID` and `D-BREAK` as `PASSED (local
decision)`.  It does not mark D0, D-TECH, or release as passed.

## Evidence snapshot

| Field | Recorded value |
| --- | --- |
| Local decision | User explicitly directs retention of analysis capabilities without legacy API compatibility; `D-ID` / `D-BREAK` are `PASSED (local decision)` |
| Required worktree branch | `dev` |
| Recorded pre-document baseline | `2bcb65773f01dd836b5fb4d928741ff1b072179e` |
| Required preflight identity | caller supplies `FINCORE_R2_ROOT` and a full-SHA `FINCORE_R2_EXPECTED_HEAD`; the actual clean `dev` HEAD must equal that SHA |
| R2 plan | [`docs/plans/2026-08-30-fincore-0042-r2-breaking-unified-core.md`](../plans/2026-08-30-fincore-0042-r2-breaking-unified-core.md) |
| R2 plan SHA256 | `8806e8b1a02b5985b5cc539270862f7b00cf6b29ded23f6ccfcb0a90a4d9e1fd` |
| R2 plan-containing commit | `b84148d585df0cd6b87d93e0c59d2f302b6072f9` |
| Target version | `0.5.0.dev0` |
| Decision ADR | [`ADR-0042-R2`](../architecture/adr/0042-r2-breaking-unified-core.md) |

The recorded commit contains the byte-for-byte plan blob identified by the
SHA256 above.  The preflight proves that this commit is an ancestor of the
caller-supplied expected HEAD and that both its blob and the checked-out plan
match the recorded digest.  This identity proof is not a D0 baseline or a
substitute for later provenance evidence.

## Local decision and technical state

| Gate or boundary | Current result |
| --- | --- | --- |
| D-ID lineage decision | `PASSED (local decision)` — the R2 ADR/plan distinguish this breaking route from historical 0042 records without rewriting them. |
| D-BREAK policy decision | `PASSED (local decision)` — the user's explicit direction ends legacy API compatibility while preserving required analysis capabilities. |
| Task 0 entry | `READY` — the preflight below must exit `0` for the caller-supplied clean `dev` worktree and full expected HEAD. |
| D0 evidence reset | `NOT STARTED` — it requires the fresh clean exact-SHA bundle, ledger/oracles, quality, architecture, performance, and provenance evidence specified by Task 0. |
| D-TECH / D-RELEASE | `NOT STARTED` — no technical or release conclusion is implied by this record. |

## Worktree isolation

The current root `master` worktree contains user-owned, dirty governance, CI,
and documentation changes.  They remain excluded from this clean `dev`
worktree and from this R2 preflight.  This record neither copies, stages,
reviews as baseline, takes over, nor changes those root-worktree files.

The only intended paths for this local documentation commit are the R2 plan
copy and these two R2 documents.  README/status pointers, the structural
consolidation plan, the historical ADR, and historical acceptance records
remain untouched in this docs-only scope.

## Exact preflight command

From a clean shell, supply `FINCORE_R2_ROOT` and a full commit SHA in
`FINCORE_R2_EXPECTED_HEAD`.  A correct clean `dev` worktree exits `0`; any
missing or mismatched identity fact exits nonzero.

```bash
set -uo pipefail

FINCORE_R2_PREFLIGHT_RESULT=0
FINCORE_R2_EXPECTED_BRANCH=dev
FINCORE_R2_PRE_DOC_BASE=2bcb65773f01dd836b5fb4d928741ff1b072179e
FINCORE_R2_PLAN=docs/plans/2026-08-30-fincore-0042-r2-breaking-unified-core.md
FINCORE_R2_PLAN_SHA256=8806e8b1a02b5985b5cc539270862f7b00cf6b29ded23f6ccfcb0a90a4d9e1fd
FINCORE_R2_PLAN_COMMIT=b84148d585df0cd6b87d93e0c59d2f302b6072f9

FINCORE_R2_ROOT_VALID=0
if test -z "${FINCORE_R2_ROOT:-}"; then
  printf '%s\n' 'BLOCKED: FINCORE_R2_ROOT was not supplied' >&2
  FINCORE_R2_PREFLIGHT_RESULT=1
elif ! test -d "$FINCORE_R2_ROOT"; then
  printf '%s\n' 'BLOCKED: FINCORE_R2_ROOT is not an existing directory' >&2
  FINCORE_R2_PREFLIGHT_RESULT=1
elif ! FINCORE_R2_TOPLEVEL=$(git -C "$FINCORE_R2_ROOT" rev-parse --show-toplevel 2>/dev/null); then
  printf '%s\n' 'BLOCKED: FINCORE_R2_ROOT is not a Git worktree' >&2
  FINCORE_R2_PREFLIGHT_RESULT=1
elif test "$(cd "$FINCORE_R2_ROOT" && pwd -P)" != "$(cd "$FINCORE_R2_TOPLEVEL" && pwd -P)"; then
  printf '%s\n' 'BLOCKED: FINCORE_R2_ROOT must be the Git worktree root' >&2
  FINCORE_R2_PREFLIGHT_RESULT=1
else
  FINCORE_R2_ROOT_VALID=1
fi

if test "$FINCORE_R2_ROOT_VALID" -eq 1; then
  FINCORE_R2_ACTUAL_BRANCH=$(git -C "$FINCORE_R2_ROOT" branch --show-current)
  FINCORE_R2_ACTUAL_HEAD=$(git -C "$FINCORE_R2_ROOT" rev-parse HEAD)
  if test "$FINCORE_R2_ACTUAL_BRANCH" != "$FINCORE_R2_EXPECTED_BRANCH"; then
    printf 'BLOCKED: expected branch %s, got %s\n' "$FINCORE_R2_EXPECTED_BRANCH" "${FINCORE_R2_ACTUAL_BRANCH:-detached}" >&2
    FINCORE_R2_PREFLIGHT_RESULT=1
  fi
  if test -n "$(git -C "$FINCORE_R2_ROOT" status --porcelain)"; then
    printf '%s\n' 'BLOCKED: R2 worktree is dirty' >&2
    FINCORE_R2_PREFLIGHT_RESULT=1
  fi
  if ! test -f "$FINCORE_R2_ROOT/$FINCORE_R2_PLAN"; then
    printf '%s\n' 'BLOCKED: R2 plan is missing from FINCORE_R2_ROOT' >&2
    FINCORE_R2_PREFLIGHT_RESULT=1
  elif test "$(shasum -a 256 "$FINCORE_R2_ROOT/$FINCORE_R2_PLAN" | awk '{print $1}')" != "$FINCORE_R2_PLAN_SHA256"; then
    printf '%s\n' 'BLOCKED: R2 plan SHA256 does not match the recorded preflight plan' >&2
    FINCORE_R2_PREFLIGHT_RESULT=1
  fi
  if ! FINCORE_R2_RESOLVED_PLAN_COMMIT=$(git -C "$FINCORE_R2_ROOT" rev-parse --verify "$FINCORE_R2_PLAN_COMMIT^{commit}" 2>/dev/null); then
    printf '%s\n' 'BLOCKED: R2 plan-containing commit does not resolve to a commit' >&2
    FINCORE_R2_PREFLIGHT_RESULT=1
  elif ! git -C "$FINCORE_R2_ROOT" merge-base --is-ancestor "$FINCORE_R2_RESOLVED_PLAN_COMMIT" "$FINCORE_R2_ACTUAL_HEAD"; then
    printf '%s\n' 'BLOCKED: R2 plan-containing commit is not an ancestor of actual HEAD' >&2
    FINCORE_R2_PREFLIGHT_RESULT=1
  elif ! FINCORE_R2_RECORDED_PLAN_SHA=$(git -C "$FINCORE_R2_ROOT" show "$FINCORE_R2_RESOLVED_PLAN_COMMIT:$FINCORE_R2_PLAN" 2>/dev/null | shasum -a 256 | awk '{print $1}'); then
    printf '%s\n' 'BLOCKED: R2 plan-containing commit does not contain the recorded plan path' >&2
    FINCORE_R2_PREFLIGHT_RESULT=1
  elif test "$FINCORE_R2_RECORDED_PLAN_SHA" != "$FINCORE_R2_PLAN_SHA256"; then
    printf '%s\n' 'BLOCKED: recorded plan blob SHA256 does not match the preflight plan' >&2
    FINCORE_R2_PREFLIGHT_RESULT=1
  fi
  if test -z "${FINCORE_R2_EXPECTED_HEAD:-}"; then
    printf '%s\n' 'BLOCKED: FINCORE_R2_EXPECTED_HEAD was not supplied' >&2
    FINCORE_R2_PREFLIGHT_RESULT=1
  elif ! test "${#FINCORE_R2_EXPECTED_HEAD}" -eq 40 || ! printf '%s' "$FINCORE_R2_EXPECTED_HEAD" | grep -Eq '^[0-9a-fA-F]{40}$'; then
    printf '%s\n' 'BLOCKED: FINCORE_R2_EXPECTED_HEAD must be a full commit SHA' >&2
    FINCORE_R2_PREFLIGHT_RESULT=1
  elif ! FINCORE_R2_RESOLVED_EXPECTED_HEAD=$(git -C "$FINCORE_R2_ROOT" rev-parse --verify "$FINCORE_R2_EXPECTED_HEAD^{commit}" 2>/dev/null); then
    printf '%s\n' 'BLOCKED: FINCORE_R2_EXPECTED_HEAD does not resolve to a commit' >&2
    FINCORE_R2_PREFLIGHT_RESULT=1
  elif test "$FINCORE_R2_ACTUAL_HEAD" != "$FINCORE_R2_RESOLVED_EXPECTED_HEAD"; then
    printf 'BLOCKED: actual HEAD %s does not equal FINCORE_R2_EXPECTED_HEAD %s\n' "$FINCORE_R2_ACTUAL_HEAD" "$FINCORE_R2_RESOLVED_EXPECTED_HEAD" >&2
    FINCORE_R2_PREFLIGHT_RESULT=1
  fi
  if ! git -C "$FINCORE_R2_ROOT" merge-base --is-ancestor "$FINCORE_R2_PRE_DOC_BASE" "$FINCORE_R2_ACTUAL_HEAD"; then
    printf '%s\n' 'BLOCKED: recorded pre-document baseline is not an ancestor of actual HEAD' >&2
    FINCORE_R2_PREFLIGHT_RESULT=1
  fi
fi

exit "$FINCORE_R2_PREFLIGHT_RESULT"
```

## Task 0 entry conditions

1. Supply a clean R2 `dev` worktree root and its exact full expected HEAD to
   the preflight above; retain the successful identity output proving the
   plan-containing commit, plan blob, checkout plan, and baseline ancestry
   with the Task 0 evidence.
2. Capture D0 only from that clean exact-SHA worktree.  The Task 0 ledger,
   independent oracle/golden inputs, same-wheel checks, quality, performance,
   architecture, and provenance gates remain mandatory.
3. A successful preflight starts Task 0 only.  It does not pass D0, D-TECH,
   D-RELEASE, or authorize merge, push, tag, publish, or remote configuration.

## Related records

- [ADR-0042-R2](../architecture/adr/0042-r2-breaking-unified-core.md)
- [R2 implementation plan](../plans/2026-08-30-fincore-0042-r2-breaking-unified-core.md)
- [Historical ADR-0042](../architecture/adr/0042-unified-operation-model.md)
- [Historical 0042 acceptance](2026-08-21-unified-platform-acceptance.md)
