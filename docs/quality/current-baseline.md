# Current Quality Baseline

Generated: `2026-08-15T02:47:44.626600+00:00`

## Provenance

- Source commit: `58a4c089bf569aeb83a1f89ef64481a4394cec4b`
- Dirty state: `True`
- Tracked diff SHA256: `3192944bf4d585ce07a8abc8ae855a4096fb47abc6e5197b32fd7cd79ece71b9`
- Untracked manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- Disposable-copy manifest SHA256: `8764661d74610c3fa9f21e16db1b83cec34e948ee1b03ed0951cf7837ca14eb3`
- Manifest exclusions: `docs/quality/current-baseline.json, docs/quality/current-baseline.md`

## Environment

- python: `3.11.8 | packaged by conda-forge | (main, Feb 16 2024, 20:49:36) [Clang 16.0.6 ]`
- numpy: `1.26.4`
- pandas: `3.0.3`
- scipy: `1.17.1`
- matplotlib: `3.10.9`

## Test Runs

| Run | Selector | Discovered | Selected | Passed | Skipped | Warnings | Duration | Exit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trusted-baseline | `not slow and not integration` | 4411 | 4395 | 4374 | 21 | 140 | 968.004s | 0 |
| serial | `serial` | 4411 | 7 | 6 | 1 | 1 | 7.322s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 4411 | 4389 | 4368 | 21 | 140 | 604.625s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 4411 | 4389 | 4368 | 21 | 147 | 238.965s | 0 |
| branch-coverage | `not slow and not integration` | 4411 | 4395 | 4374 | 21 | 140 | 973.746s | 0 |

## Branch Coverage

- Total: `55.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
