# License review

This page records the status of the human license/NOTICE decisions for the
source code adapted from upstream projects. Code does not self-certify legal
compliance; every entry stays `pending-human-review` until a human reviewer
records their name and approval date.

## Adapted components

| Component | Upstream commit | License observed | Status |
| --- | --- | --- | --- |
| empyrical 0.6.0 | `74655e974ed2935563820c548c339731f1fe0621` | Apache-2.0 header | pending-human-review |
| pyfolio 0.9.6 | `724bbd7dbed9a88bb47e1057f2ca29b3409d8e7a` | no license header in inspected file | pending-human-review |
| Alphalens cloudQuant local | `3fa17ad4c3edb025d1410de7aeba9673cba7791c` | no license header in inspected file | pending-human-review |

## Review protocol

1. A reviewer compares the pinned upstream blobs (see
   `docs/upstream-provenance.md`) against the adapted source in this
   repository.
2. The reviewer records the NOTICE/SPDX/header decision — whether the
   destination files need a header, a NOTICE entry, or both.
3. The reviewer updates `THIRD_PARTY_NOTICES.md` (the JSON inventory) and this
   page with `reviewer`, `reviewed_at`, and `review_status="approved"`.

Until every entry is `approved`, the release candidate stays Beta and the
third-party notice gate remains an explicit release blocker. The machine
checker (`scripts/check_notices.py`) validates the inventory shape; it never
substitutes for the human decision.
