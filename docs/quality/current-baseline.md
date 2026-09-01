# Current Quality Baseline

Generated: `2026-09-01T12:28:31.193619+00:00`

## Provenance

- Source commit: `34a5a0f1f1db88e6ac369efb27c164c242c227b9`
- Dirty state: `False`
- Tracked diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Untracked manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- Disposable-copy manifest SHA256: `0043ac423e0f2186413ab36da0b1f8616185f0b13263930a18a5f03872d8830e`
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
| trusted-baseline | `not slow and not integration` | 1854 | 1853 | 1838 | 15 | 1 | 101.800s | 0 |
| serial | `serial` | 1854 | 3 | 3 | 0 | 0 | 6.005s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 1854 | 1850 | 1835 | 15 | 1 | 99.927s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 1854 | 1850 | 1835 | 15 | 1 | 51.056s | 0 |
| branch-coverage | `not slow and not integration` | 1854 | 1853 | 1838 | 15 | 1 | 130.234s | 0 |

## Branch Coverage

- Total: `76.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
