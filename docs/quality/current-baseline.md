# Current Quality Baseline

Generated: `2026-08-12T15:53:56.625864+00:00`

## Provenance

- Source commit: `60a13272ad399f22e2eee3371beb44690cc118b5`
- Dirty state: `True`
- Tracked diff SHA256: `2f35ccad7c79426f8ef2ede4754309a2d0f3a24fdce2ade97f48a5d172cf94d2`
- Untracked manifest SHA256: `3cb8edc269156acf718fe34e40dcd722d2b11a4df3af8765e3d0e22cc0c66deb`
- Disposable-copy manifest SHA256: `40850410a386d6ed6e85f19b41d9e849f17f14b780db6831b0a39234b4dbb165`
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
| trusted-baseline | `not slow and not integration` | 2305 | 2290 | 2276 | 14 | 11 | 95.316s | 0 |
| serial | `serial` | 2305 | 6 | 6 | 0 | 0 | 7.101s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 2305 | 2284 | 2270 | 14 | 11 | 161.509s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 2305 | 2284 | 2270 | 14 | 11 | 129.154s | 0 |
| branch-coverage | `not slow and not integration` | 2305 | 2290 | 2276 | 14 | 11 | 176.920s | 0 |

## Branch Coverage

- Total: `94.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
