# Third-Party Notices

This file records the provenance of source code imported or adapted from
upstream projects. It is evidence, not a legal conclusion: the human
review status is an explicit release blocker until a reviewer records their
approval.

The machine-readable inventory below is parsed by
`scripts/check_notices.py` and asserted by `tests/packaging/test_notices.py`.
Do not hand-edit the JSON block without updating the paired evidence in
`docs/upstream-provenance.md`.

| Component | Upstream commit | License observed | Review status |
| --- | --- | --- | --- |
| empyrical 0.6.0 | `74655e974ed2935563820c548c339731f1fe0621` | Apache-2.0 header | pending-human-review |
| pyfolio 0.9.6 | `724bbd7dbed9a88bb47e1057f2ca29b3409d8e7a` | no license header in inspected file | pending-human-review |
| Alphalens cloudQuant local | `3fa17ad4c3edb025d1410de7aeba9673cba7791c` | no license header in inspected file | pending-human-review |

## Machine-readable inventory

```json
{
  "schema_version": 1,
  "empyrical": {
    "name": "empyrical",
    "version": "0.6.0",
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
    "version": "0.9.6",
    "source_commit": "724bbd7dbed9a88bb47e1057f2ca29b3409d8e7a",
    "license": "unresolved",
    "license_header": "absent",
    "adapted": true,
    "review_status": "pending-human-review",
    "reviewer": null,
    "reviewed_at": null
  },
  "alphalens": {
    "name": "alphalens",
    "version": "0.4.0",
    "source_commit": "3fa17ad4c3edb025d1410de7aeba9673cba7791c",
    "license": "unresolved",
    "license_header": "absent",
    "adapted": true,
    "review_status": "pending-human-review",
    "reviewer": null,
    "reviewed_at": null
  }
}
```
