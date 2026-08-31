# `fincore.runtime` and `fincore.extensions`

The runtime resolves direct operations, validates inputs, records artifacts,
and maintains session-local state. Extensions are discovered as immutable
metadata and applied through an `ExtensionSnapshot`; no process-global plugin
registry mutates a running analysis.

```python
from fincore.extensions.discovery import discover_extensions
from fincore.extensions.snapshot import ExtensionSnapshot
from fincore.runtime.catalog import OperationCatalog
```

Use these APIs when embedding fincore in a larger platform. Domain functions do
not require extension registration for ordinary direct use.
