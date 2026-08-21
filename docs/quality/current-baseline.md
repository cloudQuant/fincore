# Current Quality Baseline

Generated: `2026-08-21T03:09:00.549970+00:00`

## Provenance

- Source commit: `93fbf54d9a03b9f14b37bfb25ce5c3b821ef5710`
- Dirty state: `False`
- Tracked diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Untracked manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- Disposable-copy manifest SHA256: `79f428ced76dd7a625cc1f1cdaa9431ac11e13004211a6af0f4454f04506b93c`
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
| trusted-baseline | `not slow and not integration` | 5263 | 5247 | 5225 | 22 | 98 | 432.017s | 0 |
| serial | `serial` | 5263 | 7 | 6 | 1 | 2 | 6.500s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 5263 | 5241 | 5219 | 22 | 99 | 422.302s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 5263 | 5241 | 5219 | 22 | 113 | 388.258s | 0 |
| branch-coverage | `not slow and not integration` | 5263 | 5247 | 5208 | 22 | 99 | 1455.693s | 1 |

## Branch Coverage

- Total: `45.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`

## Incomplete Baseline

baseline did not complete
