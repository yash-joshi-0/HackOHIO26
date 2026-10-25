---
status: ready
priority: p1
issue_id: "004"
tags: [runtime, profiling, admission]
dependencies: ["003"]
---

# Integrate managed profiling and native compute admission

## Problem Statement

The standalone profiler can overlap native inference. HTTP job permits do not
cover workers, persistent RPC backends or the interval between cancellation and
native process exit.

## Findings

Native subprocesses enter through process supervision, pipeline local stages and
shard serving. M-27 is separately owned and has additional spawn sites. A shared
in-process owner matches the single Rust host direction; external workers remain
outside this gate until their launch/control path is integrated.

## Proposed Solutions

1. Gate JobQueue only: small, but unsafe around asynchronous native cleanup.
2. Carry inference leases with native children; exclusive diagnostic admission,
   managed start/status/cancel, and fence compute if cleanup is unconfirmed.

## Recommended Action

Implement option 2. Preserve existing inference concurrency and reject profiling
while managed compute is occupied. Configure named candidates at startup; HTTP
callers must not supply executable paths. Keep automatic join activation disabled.

## Acceptance Criteria

- [x] Shared native leases cover core supervisor, pipeline and shard worker paths.
- [x] Leases survive cancellation until child exit/reap; failed cleanup fences admission.
- [x] Managed profiler start/status/cancel uses real helper and retains evidence.
- [x] Loopback control accepts configured candidate names only and requires compute opt-in.
- [x] Tests cover busy admission, queued inference, cancellation and failure fencing.
- [x] Documentation and owner handoff state process scope and pending M-27/mobile integration.
- [ ] Relevant Rust/Python tests, formatting and Clippy pass.

## Work Log

### 2026-10-25 — Start

User explicitly asked to integrate findings. Branch david/profile-runtime-admission
builds on source profiler f13096c (published separately as draft PR38/c85ab90).
Announced touch points to runtime/apps/Yash before changes. No live workers touched.

### 2026-10-25 — Implementation and local validation

- Added admission.rs and profiling.rs. Integrated native ownership into process
  supervision, pipeline local stages and shard serving; added optional API routes
  and an explicit embedded-host entry point. Kept mesh/ALPN and native kernels unchanged.
- Python helper now correlates the runtime run ID and confirms started/reaped
  native child counts. Unconfirmed cleanup fences future managed compute.
- All 29 Python tests pass. Twelve new Rust integration tests pass, including
  real helper subprocesses, actual shard-stage ownership, cancellation, fencing
  and in-process HTTP handlers. The retained thirteenth test cannot bind its loopback
  socket in this sandbox (EPERM before application execution).
- Full all-target regression run performed once; socket-dependent existing tests
  likewise fail on sandbox permission. Logs are retained at
  /Users/stephenson.439/llama-rpc-demo/results/runtime-profile-integration/rust-all-targets.log.
  Independent socket-enabled validation remains required; do not report a full pass.
- That run also exposed two subsequent stage tests blocked by a spurious fence:
  runtime teardown cancelled the async reaper before it observed an already
  killed child. Added bounded synchronous exit confirmation during reaper teardown
  (100 ms maximum, never time-only release) and a dedicated runtime-shutdown
  regression. All twelve socket-free integration tests pass after this fix.
- Formatting and all-target Clippy pass. No live device benchmarking or workers changed.
- Yash identified M-27's NativeRunner/Shard::spawn_runner/reap hooks; sent the exact
  AdmittedChild interface and async spawn change for his integration. That unpushed
  ring branch is not present here and is not claimed covered.
- Other-session Android/iOS styling and design-language edits appeared in this
  checkout. They are preserved and excluded from this runtime commit.

Task remains ready pending independent network-enabled regression validation and
publication review. Automatic device placement, distributed reservations and
mobile in-process engine admission remain later milestones.
