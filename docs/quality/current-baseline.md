# Current Quality Baseline

Generated: `2026-08-20T16:46:38.002233+00:00`

## Provenance

- Source commit: `4f2b6c65ec346b3a60c7eefed6cc1b994f1db687`
- Dirty state: `False`
- Tracked diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Untracked manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- Disposable-copy manifest SHA256: `7fe2327c3ac35a695864179d90d792221cc19306e92d9c88cfbbbedc34a034f7`
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
| trusted-baseline | `not slow and not integration` | 5092 | 5076 | 5054 | 22 | 98 | 445.195s | 0 |
| serial | `serial` | 5092 | 7 | 6 | 1 | 2 | 7.296s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 5092 | 5070 | 5048 | 22 | 99 | 440.891s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 5092 | 5070 | 5048 | 22 | 113 | 217.458s | 0 |
| branch-coverage | `not slow and not integration` | 5092 | 5076 | 5054 | 22 | 99 | 779.345s | 0 |

## Branch Coverage

- Total: `97.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
