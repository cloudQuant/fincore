# Current Quality Baseline

Generated: `2026-09-01T14:00:47.471138+00:00`

## Provenance

- Source commit: `fb3c9289409ac73e0fe89bdf2abff5db1cf6a4e0`
- Dirty state: `False`
- Tracked diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Untracked manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- Disposable-copy manifest SHA256: `f44c243bb0503a3cfb68649037d5283e2582608ec1070a21ce5e611c24f18706`
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
| trusted-baseline | `not slow and not integration` | 1854 | 1853 | 1838 | 15 | 1 | 104.972s | 0 |
| serial | `serial` | 1854 | 3 | 3 | 0 | 0 | 6.476s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 1854 | 1850 | 1835 | 15 | 1 | 103.155s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 1854 | 1850 | 1835 | 15 | 1 | 54.304s | 0 |
| branch-coverage | `not slow and not integration` | 1854 | 1853 | 1838 | 15 | 1 | 134.966s | 0 |

## Branch Coverage

- Total: `76.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
