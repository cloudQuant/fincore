# Current Quality Baseline

Generated: `2026-08-12T16:13:45.424101+00:00`

## Provenance

- Source commit: `53af92151f69b990415d865d5d1f3885f6ac3d8e`
- Dirty state: `True`
- Tracked diff SHA256: `333e01b958e76384e55a2c558641f819c44e75d66ede39528a4c124d42e5e575`
- Untracked manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- Disposable-copy manifest SHA256: `aaae7a6629a493f12f96286b3523115f855f5cd6ccc64e0ff49433ca0934dc15`
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
| trusted-baseline | `not slow and not integration` | 2308 | 2293 | 2279 | 14 | 11 | 125.885s | 0 |
| serial | `serial` | 2308 | 6 | 6 | 0 | 0 | 8.102s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 2308 | 2287 | 2273 | 14 | 11 | 117.125s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 2308 | 2287 | 2273 | 14 | 11 | 108.924s | 0 |
| branch-coverage | `not slow and not integration` | 2308 | 2293 | 2279 | 14 | 11 | 139.318s | 0 |

## Branch Coverage

- Total: `94.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
