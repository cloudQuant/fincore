# Current Quality Baseline

Generated: `2026-09-01T11:51:29.563275+00:00`

## Provenance

- Source commit: `9cf0dc22eb95444496b2768258a91581d5ebf66f`
- Dirty state: `False`
- Tracked diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Untracked manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- Disposable-copy manifest SHA256: `fb7b858565effa3ec89249b0fc0d8e75b26a1aad2a0260711dfbb1c217984fd9`
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
| trusted-baseline | `not slow and not integration` | 1853 | 1852 | 1837 | 15 | 1 | 99.774s | 0 |
| serial | `serial` | 1853 | 3 | 3 | 0 | 0 | 6.032s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 1853 | 1849 | 1834 | 15 | 1 | 98.251s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 1853 | 1849 | 1834 | 15 | 1 | 56.483s | 0 |
| branch-coverage | `not slow and not integration` | 1853 | 1852 | 1837 | 15 | 1 | 136.243s | 0 |

## Branch Coverage

- Total: `76.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
