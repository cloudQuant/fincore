# Financial Semantics

This document freezes the financial semantics that the **enhanced** profiles
(`enhanced_v1`) must follow.  Strict compatibility façades keep their upstream
behavior; these definitions only apply where a computation is exposed as an
enhanced operation.

## Return conventions

| Term | Definition |
| --- | --- |
| simple return | `r_t = P_t / P_{t-1} - 1` (fractional, not percent) |
| log return | `r_t = ln(P_t / P_{t-1})` |
| price/return boundary | enhanced kernels accept returns unless the operation name says `price`; a `price` input is converted via `pct_change` after a documented convention |
| cumulative return | geometric compound: `(1+r_1)···(1+r_T) - 1` |
| annualization | explicit `period` (`daily`/`weekly`/`monthly`/`yearly`); no implicit 252 |

## Alignment and calendar

- **timezone:** naive timestamps are not silently treated as UTC; the profile
  or caller must localize, convert, or reject.
- **frequency/calendar:** inferred from a DatetimeIndex or declared explicitly;
  no implicit trading-day assumption.
- **alignment:** default inner-join on index; every dropped row is recorded in
  diagnostics with a reason.
- **duplicate/order:** duplicate timestamps are rejected; input must be sorted.
- **missing/NaN:** NaN in the return series is dropped with a diagnostic;
  NaN in the *result* is an explicit state, never a silent number.
- **finite:** non-finite inputs are rejected or recorded, never propagated.

## Sign and unit conventions

| Term | Convention |
| --- | --- |
| VaR / ES | `losses_negative`: negative values are losses, so `ES <= VaR <= 0` |
| excess return | `returns - risk_free`, where `risk_free` shares the period unit |
| currency | operations are currency-agnostic unless a `currency` argument is present; currency is metadata, never a number |
| benchmark / risk-free unit | per-period rate matching the return frequency |

## Portfolio and factor semantics

| Term | Definition |
| --- | --- |
| weights | row-aligned with returns; must sum to 1 within tolerance unless a `gross` flag is set |
| weight timestamp | weights apply from their timestamp until the next one (`as-of`) |
| as-of / known-at | enhanced factor inputs use `as_of`/`known_at`/`effective_from` and a universe; no full-sample z-score filtering |
| cashflow / fees | explicit gross/net-of-fees and timing, never folded into a bare number |
| ddof | sample statistics use `ddof=1` |

## Discriminant result state

Every enhanced high-level result is `Success | Unsupported | Failed`.  `Success`
carries a typed value; `Unsupported` means the operation cannot run under the
requested profile/input (not an error); `Failed` carries an error and
diagnostics.  Direct scalar APIs keep their frozen return shape.
