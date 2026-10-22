# Current Quality Baseline

Generated: `2026-10-22T06:47:26.251738+00:00`

## Provenance

- Source commit: `d7a30d0179301c7588537ab23b3ffc98081d7ab5`
- Dirty state: `False`
- Tracked diff SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Untracked manifest SHA256: `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`
- Disposable-copy manifest SHA256: `710388b08a7a23aaf6e2614997cdd51d0e785ddff7ca8bc3175a873d37ddebc5`
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
| trusted-baseline | `not slow and not integration` | 1929 | 1928 | 1913 | 15 | 1 | 108.647s | 0 |
| serial | `serial` | 1929 | 3 | 3 | 0 | 0 | 6.041s | 0 |
| non-serial-single | `not serial and not slow and not integration` | 1929 | 1925 | 1910 | 15 | 1 | 111.167s | 0 |
| non-serial-xdist | `not serial and not slow and not integration` | 1929 | 1925 | 1910 | 15 | 1 | 69.164s | 0 |
| branch-coverage | `not slow and not integration` | 1929 | 1928 | 1913 | 15 | 1 | 150.688s | 0 |

## Branch Coverage

- Total: `76.0%`

## Integrity

- trusted-baseline: `True`
- serial: `True`
- non-serial-single: `True`
- non-serial-xdist: `True`
- branch-coverage: `True`
