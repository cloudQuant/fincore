# Current Quality Baseline

Generated: `2026-08-31T23:02:09.906000+00:00`

## Provenance

- Source commit: `129cb26327e7caa2f8e4adf0654b69797766653f`
- Dirty state: `False`
- Tracked diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Untracked manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- Disposable-copy manifest SHA256: `d0c99b8363d7459fe75e91a993fb09c1914f3cedec17d2984c667a00c4a623cc`
- Manifest exclusions: ``

## Environment

- python: `3.11.8 | packaged by conda-forge | (main, Feb 16 2024, 20:49:36) [Clang 16.0.6 ]`
- numpy: `1.26.4`
- pandas: `3.0.3`
- scipy: `1.17.1`
- matplotlib: `3.10.9`

## Test Runs

| Run | Selector | Discovered | Selected | Passed | Skipped | Warnings | Duration | Exit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trusted-baseline | `not slow and not integration` | 5657 | 5641 | 5620 | 21 | 99 | 769.376s | 0 |
| serial | `serial` | 5657 | 7 | 6 | 1 | 1 | 12.037s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 5657 | 5635 | 5614 | 21 | 99 | 809.004s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 5657 | 5635 | 5614 | 21 | 106 | 434.919s | 0 |
| branch-coverage | `not slow and not integration` | 5657 | 5641 | 5620 | 21 | 99 | 881.574s | 0 |

## Branch Coverage

- Total: `42.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
