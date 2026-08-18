# Current Quality Baseline

Generated: `2026-08-18T20:01:50.006178+00:00`

## Provenance

- Source commit: `6cb26ab8328299a8f1a55aa977da4d8119e952a8`
- Dirty state: `False`
- Tracked diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Untracked manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- Disposable-copy manifest SHA256: `c805bb98ea789e7d6a1916b3bbb75e0b5a71a5af042d021afa522c435ac8f1a9`
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
| trusted-baseline | `not slow and not integration` | 4495 | 4479 | 4457 | 22 | 97 | 441.418s | 0 |
| serial | `serial` | 4495 | 7 | 6 | 1 | 2 | 6.660s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 4495 | 4473 | 4451 | 22 | 98 | 438.296s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 4495 | 4473 | 4451 | 22 | 112 | 206.492s | 0 |
| branch-coverage | `not slow and not integration` | 4495 | 4479 | 4457 | 22 | 98 | 775.343s | 0 |

## Branch Coverage

- Total: `55.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
