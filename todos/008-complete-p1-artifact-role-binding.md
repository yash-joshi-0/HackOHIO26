---
status: complete
priority: p1
issue_id: "008"
tags: [placement, artifacts, integration]
dependencies: []
---

# Reject incompatible artifact roles during ring preparation

The PR46 review identified that an older suffix can have the expected range but
lack terminal tensors. Bind optional catalog role assertions to exact ranges;
retain unknown roles for old registries without changing their serialized hashes.
Actual tensor verification remains a destination staging requirement.

- [x] Add optional first/middle/final/full role and reject range mismatch.
- [x] Preserve omitted-role JSON and preparation IDs.
- [x] Exercise real prepare/validate and unchanged CLI configuration path.
- [x] Formatting, Clippy and 43 deployment/CLI checks pass.

2026-10-25: two new role/compatibility tests; preparation still reports
activation_ready=false and does not read destination artifacts. Handoff uses a
single follow-up commit on the PR46 source; no runtime/worker files changed.
