---
status: complete
priority: p1
issue_id: "005"
tags: [runtime, placement, ring]
dependencies: []
---

# Prepare executable ring configurations from measured placement

## Problem Statement

The measured planner returns layer assignments, but the core needs a leader manifest
and each worker's exact artifact, runner configuration and adjacent identities.
Users currently supply these independently, risking drift from the measured ranges.

## Findings

- Main 8846193 includes experimental M-27 ring execution.
- Ring stage zero is leader-local; all remaining stages are remote, ending at layers.
- Existing measured v1 search can pick unsupported topology; Aarush owns an additive
  ring-constrained search and its validation.
- Diagnostic profiles remain ineligible; preparing configs must not claim reservation,
  staging, evidence qualification or activation.

## Proposed Solutions

1. Build validated deployment bundles from a constrained plan and explicit local registry.
   Small, reviewable bridge to actual core types; preserves evidence boundaries.
2. Build automatic distributed staging/activation first. Much larger coupling and no
   current ownership/epoch protocol; defer until the bundle contract is concrete.

## Recommended Action

Implement option 1 with a callable Rust API, CLI, exact identity checks, a new
output directory with a final completion marker, and real core manifest/CLI compatibility tests. Keep paths
as arguments and never execute generated commands through a shell.

## Acceptance Criteria

- [x] Constrained measured search produces the exact plan recorded in the bundle.
- [x] Selected ranges map uniquely to pinned per-peer runtime and artifact bindings.
- [x] Leader manifest passes the real ShardManifest validator; worker argv carries
      source/artifact hashes, range, context, budget and correct upstream/next/origin.
- [x] Missing/stale/mismatched evidence, topology, duplicate identities or bindings fail.
- [x] CLI emits complete bundle only into a new directory and never launches workers.
- [x] Integration tests prove planner -> bundle -> runtime parser compatibility and failures.
- [x] Formatting, Clippy and focused tests pass; publication handoff records limitations.

## Work Log

### 2026-10-25 — Implementation started

David explicitly authorized continuing with six hours remaining. Created isolated
checkout measured-deployment from main8846193; preserve all other sessions' files.
Coordinated planner ownership with Aarush and ring admission hooks with Yash.
After no planner ACK, explicitly took the minimal additive ring search to unblock
the bridge; Aarush's v2 evidence qualification remains separate.

### 2026-10-25 — Implementation and local validation complete

- Added plan_ring/validate_ring_output without changing the original plan behavior.
- Added prepare/validate/write_bundle, prepare-ring/validate-ring CLI, real manifest
  validation and worker configs/argv. Inputs remain advisory; no workers launched.
- 49 focused Rust tests pass: 32 CLI/parser, 8 measured placement, 9 deployment.
  This includes 12 new tests and real subprocess CLI/file lifecycle coverage.
- cargo fmt --check, all-target offline Clippy -D warnings and git diff --check pass.
- Publication is handed to runtime/publication; network-enabled integration and
  actual hardware qualification remain separate and are not claimed here.
