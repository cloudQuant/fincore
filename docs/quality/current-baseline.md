# Current Quality Baseline

Generated: `2026-08-20T21:48:01.268584+00:00`

## Provenance

- Source commit: `bd94bfa3c0624953ca10bbac46e74c78cbfcea03`
- Dirty state: `False`
- Tracked diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Untracked manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- Disposable-copy manifest SHA256: `b4e31bc659f91768b903f16481ebb3e462c02de4b20a27cf9f61b62c157614fb`
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
| trusted-baseline | `not slow and not integration` | 5187 | 5171 | 5149 | 22 | 98 | 449.001s | 0 |
| serial | `serial` | 5187 | 7 | 6 | 1 | 2 | 7.487s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 5187 | 5165 | 5143 | 22 | 99 | 444.122s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 5187 | 5165 | 5143 | 22 | 113 | 224.525s | 0 |
| branch-coverage | `not slow and not integration` | 5187 | 5171 | 5149 | 22 | 99 | 587.632s | 0 |

## Branch Coverage

- Total: `46.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
