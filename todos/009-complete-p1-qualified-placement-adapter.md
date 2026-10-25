---
status: complete
priority: p1
issue_id: "009"
tags: [placement, evidence, integration]
dependencies: ["005", "008"]
---

# Qualify separate evidence inputs before measured ring planning

Aarush confirmed no overlapping implementation on 2026-10-25 (Discord
1563836230586728489). This adapter implements the proposed evidence boundary
without changing the existing planner search, worker protocols or downloader.

- [x] Typed reusable range records, current peer availability, directed links.
- [x] Explicit exclusions for incomplete, failed, mismatched or stale evidence.
- [x] Whole-range p95 service costs; valid state histories and sustained sessions.
- [x] Separate evidence/capacity/link expiry; no timestamp rewriting.
- [x] Exact selected artifact/build/config binding before bundle preparation.
- [x] Revalidation, CLI, meaningful integration tests and documentation.
- [x] Review/publication artifacts prepared; no hardware attestation/activation claim.

The adapter consumes trusted local observations and provenance claims; it does
not establish their authenticity. Current candidate diagnostics remain missing
whole-contribution physical memory, sampled conditions and effective-backend
evidence. Do not manufacture these fields from GGML/RSS or requested settings.

2026-10-25: 68 focused Rust checks pass (32 CLI, 8 original planner, 11 deployment,
17 new qualification integration). Formatting and all-target Clippy pass.
Trial acceptance policy is predeclared and hash-bound independently of decision
TTLs. Complete requested repetitions, full context histories, backend evidence,
per-pool peaks and sensor scope are required. Devices without qualified ranges
remain in the identity snapshot but do not consume the exact-search peer limit.
No changes to measured_placement.rs, worker/ring protocols or downloader code.
