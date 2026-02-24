# API Stability

## Stable APIs

- All functions in `fincore.__all__`
- `AnalysisContext` and its public methods
- `RollingEngine` and its `compute()` method
- `Empyrical` class methods
- `Pyfolio` class methods

## Internal APIs

Modules prefixed with `_` (e.g., `fincore._registry`) are internal and may change without notice.

## Versioning

- **Major (X.0.0)**: Breaking changes to stable APIs
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, backward compatible

## Deprecation Process

1. Mark as deprecated in docs + add warning
2. Keep functional for at least one minor version
3. Remove in next major version
