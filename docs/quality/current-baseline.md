# Current Quality Baseline

Generated: `2026-09-01T11:29:52.441784+00:00`

## Provenance

- Source commit: `f8d8473407564d32cd51f142983a22209efe5456`
- Dirty state: `False`
- Tracked diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Untracked manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- Disposable-copy manifest SHA256: `f3077cc005cb3194e39e32569836dea894c8f348796dc635b4f5f80d4a188651`
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
| trusted-baseline | `not slow and not integration` | 1853 | 1852 | 1837 | 15 | 1 | 95.792s | 0 |
| serial | `serial` | 1853 | 3 | 3 | 0 | 0 | 5.497s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 1853 | 1849 | 1834 | 15 | 1 | 96.547s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 1853 | 1849 | 1834 | 15 | 1 | 49.697s | 0 |
| branch-coverage | `not slow and not integration` | 1853 | 1852 | 1837 | 15 | 1 | 128.252s | 0 |

## Branch Coverage

- Total: `76.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
