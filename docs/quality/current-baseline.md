# Current Quality Baseline

Generated: `2026-08-12T15:25:35.725964+00:00`

## Provenance

- Source commit: `257b7fe67aa8a0f3e435b470f7609de8278bd9e1`
- Dirty state: `True`
- Disposable-copy manifest SHA256: `a1738ac561cc5911b1a5d3577496a73fbad7e2687d99055b1c7a7179e5f3d382`

## Environment

- matplotlib: `3.10.9`
- numpy: `1.26.4`
- pandas: `3.0.3`
- python: `3.11.8 | packaged by conda-forge | (main, Feb 16 2024, 20:49:36) [Clang 16.0.6 ]`
- scipy: `1.17.1`

## Test Runs

| Run | Selector | Collected | Passed | Skipped | Warnings | Duration | Exit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trusted-baseline | `not slow and not integration` | 2284 | 2270 | 14 | 11 | 93.992s | 0 |
| serial | `serial` | 6 | 6 | 0 | 0 | 6.216s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 2278 | 2264 | 14 | 11 | 114.095s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 2278 | 2264 | 14 | 11 | 128.822s | 0 |
| branch-coverage | `not slow and not integration` | 2284 | 2270 | 14 | 11 | 156.727s | 0 |

## Branch Coverage

- Total: `94.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
