# Reproducible research

fincore's enhanced layer makes external-data analysis reproducible: a fetched
frame is wrapped in a `DataSnapshot` that freezes its source, request interval,
as-of timestamp, price-adjustment convention, and a SHA256 of the data — without
recording secret configuration.

```python
import pandas as pd

from fincore.data.snapshots import DataSnapshot

# -- snapshot
snapshot = DataSnapshot.from_frame(
    frame=pd.DataFrame({"close": [10.0]}),
    provider="fixture",
    requested_start="2024-01-01",
    requested_end="2024-01-02",
    as_of="2024-01-03T00:00:00Z",
)
# -- snapshot

manifest = snapshot.to_manifest()
print(manifest["content_sha256"])  # 64-hex SHA256 of the data
```

The manifest carries provenance only — no API keys, tokens, raw returns, or
absolute local paths. Reports add a second layer: with `return_result=True` and
`audit_manifest=True`, `create_strategy_report` writes a sidecar JSON recording
the code commit, dependency versions, per-input shapes and hashes, and the
sanitized resolved structured performance disclosure (calculation convention, units,
frequency, sample period, data quality, fee/cashflow treatment, benchmark,
risk-free convention and annualization). It still does not copy raw input
values into the manifest; free-form report HTML is rendered separately (see
[Risk validation](risk-validation.md) and the report API).

Provider access is `provider_required`: inject a client for offline tests, and
a broken optional SDK surfaces as a controlled `DependencyError` that names the
required extra. See [Core Concepts](concepts.md) for the provider contract.
