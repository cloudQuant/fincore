# Current Quality Baseline

Generated: `2026-08-21T16:36:45.336564+00:00`

## Provenance

- Source commit: `f8174aeb262187ed87a5e3d2ede8c86ca8e6db00`
- Dirty state: `False`
- Tracked diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Untracked manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- Disposable-copy manifest SHA256: `899527a92987d3934d8d948f72aa4a790640e23575bd3aed8f36556535caa5a2`
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
| trusted-baseline | `not slow and not integration` | 5264 | 5248 | 5226 | 22 | 98 | 546.625s | 0 |
| serial | `serial` | 5264 | 7 | 6 | 1 | 1 | 9.562s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 5264 | 5242 | 5220 | 22 | 98 | 447.514s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 5264 | 5242 | 5220 | 22 | 105 | 224.727s | 0 |
| branch-coverage | `not slow and not integration` | 5264 | 5248 | 5226 | 22 | 98 | 715.319s | 0 |

## Branch Coverage

- Total: `45.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
