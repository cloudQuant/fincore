# Current Quality Baseline

Generated: `2026-09-20T02:06:36.731364+00:00`

## Provenance

- Source commit: `75de146098de6a1fe81245f9c345ba8f9ba9f8a3`
- Dirty state: `False`
- Tracked diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Untracked manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- Disposable-copy manifest SHA256: `aae6ff55dde15aa706dcb6f49ba30a15b0d5519f8f6c3ed2eecb15187c4316ff`
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
| trusted-baseline | `not slow and not integration` | 1857 | 1856 | 1841 | 15 | 1 | 204.036s | 0 |
| serial | `serial` | 1857 | 3 | 3 | 0 | 0 | 9.555s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 1857 | 1853 | 1838 | 15 | 1 | 215.449s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 1857 | 1853 | 1838 | 15 | 1 | 90.980s | 0 |
| branch-coverage | `not slow and not integration` | 1857 | 1856 | 1841 | 15 | 1 | 302.923s | 0 |

## Branch Coverage

- Total: `76.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
