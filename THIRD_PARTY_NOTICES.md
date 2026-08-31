# Third-Party Notices

Fincore has one project identity: `fincore` and the version declared in
`pyproject.toml`. Its project license is MIT. The upstream versions in this
file are source-provenance identifiers, not additional Fincore versions.

This file records the provenance of source code imported, adapted, or vendored
from upstream projects. It is evidence, not a legal conclusion: the human
review status remains visible for follow-up but is not a CI/CD release gate.
The distributed [`NOTICE`](NOTICE) and
[`THIRD_PARTY_LICENSES/Apache-2.0.txt`](THIRD_PARTY_LICENSES/Apache-2.0.txt)
preserve Apache-2.0 attribution and terms without changing Fincore's MIT
project license.

The machine-readable inventory below is parsed by
`scripts/check_notices.py` and asserted by `tests/packaging/test_notices.py`.
Do not hand-edit the JSON block without updating the paired evidence in
`docs/upstream-provenance.md`.

| Component | Upstream source identifier | License observed | Review status |
| --- | --- | --- | --- |
| empyrical 0.6.0 | `74655e974ed2935563820c548c339731f1fe0621` | Apache-2.0 header | pending-human-review |
| pyfolio 0.9.6 | `724bbd7dbed9a88bb47e1057f2ca29b3409d8e7a` | root MIT text; inspected Apache-2.0 headers | pending-human-review |
| Alphalens cloudQuant local 0.4.0 | `3fa17ad4c3edb025d1410de7aeba9673cba7791c` | root MIT text; inspected Apache-2.0 headers | pending-human-review |
| Apache ECharts 5.5.0 | `npm:echarts@5.5.0/dist/echarts.min.js` | Apache-2.0 header | pending-human-review |

## Machine-readable inventory

```json
{
  "schema_version": 2,
  "project": {
    "name": "fincore",
    "version": "0.5.0.dev0",
    "license": "MIT"
  },
  "empyrical": {
    "name": "empyrical",
    "upstream_version": "0.6.0",
    "source_commit": "74655e974ed2935563820c548c339731f1fe0621",
    "license": "Apache-2.0",
    "license_header": "present",
    "adapted": true,
    "review_status": "pending-human-review",
    "reviewer": null,
    "reviewed_at": null
  },
  "pyfolio": {
    "name": "pyfolio",
    "upstream_version": "0.9.6",
    "source_commit": "724bbd7dbed9a88bb47e1057f2ca29b3409d8e7a",
    "license": "unresolved",
    "license_header": "mixed",
    "adapted": true,
    "review_status": "pending-human-review",
    "reviewer": null,
    "reviewed_at": null
  },
  "alphalens": {
    "name": "alphalens",
    "upstream_version": "0.4.0",
    "source_commit": "3fa17ad4c3edb025d1410de7aeba9673cba7791c",
    "license": "unresolved",
    "license_header": "mixed",
    "adapted": true,
    "review_status": "pending-human-review",
    "reviewer": null,
    "reviewed_at": null
  },
  "echarts": {
    "name": "Apache ECharts",
    "upstream_version": "5.5.0",
    "source_reference": "npm:echarts@5.5.0/dist/echarts.min.js",
    "source_sha256": "42f8329d989b6f6539dd2b15bbdf0d82025762ac112fbb60dc57b27d7bcf3946",
    "vendored_path": "fincore/report/assets/echarts.min.js",
    "embedded_attributions": [
      "Copyright (c) Microsoft Corporation."
    ],
    "license": "Apache-2.0",
    "license_header": "present",
    "adapted": false,
    "review_status": "pending-human-review",
    "reviewer": null,
    "reviewed_at": null
  }
}
```
