# Current Quality Baseline

Generated: `2026-09-01T11:38:30.938312+00:00`

## Provenance

- Source commit: `698852c3a999f761665ca47a4b45e9f38d9cc3eb`
- Dirty state: `False`
- Tracked diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Untracked manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- Disposable-copy manifest SHA256: `bc0a1edbd0171cbcb4cedf3ec326f8548720e0cff690904c347e94cf0173a26b`
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
| trusted-baseline | `not slow and not integration` | 1853 | 1852 | 1837 | 15 | 1 | 94.691s | 0 |
| serial | `serial` | 1853 | 3 | 3 | 0 | 0 | 5.369s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 1853 | 1849 | 1834 | 15 | 1 | 103.694s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 1853 | 1849 | 1834 | 15 | 1 | 55.369s | 0 |
| branch-coverage | `not slow and not integration` | 1853 | 1852 | 1837 | 15 | 1 | 129.254s | 0 |

## Branch Coverage

- Total: `76.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
