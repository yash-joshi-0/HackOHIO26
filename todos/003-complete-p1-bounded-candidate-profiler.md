---
status: complete
priority: p1
issue_id: "003"
tags: [profiling, runtime, placement]
dependencies: []
---

# Bounded native candidate profiler

## Problem Statement

The measured planner accepts observations, but the existing native diagnostic
runner has no bounded, cancellable correctness-checked profiling command.

## Findings

The native MSH1 runner supports exact ranges, ordered input/state history,
compute timings and scoped buffer/RSS metrics. Existing Python helpers can block
on writes, lack a total deadline, and do not retain cancellation as an outcome.
Aarush owns advisory planning; Yash owns the fixed-topology M-27 runtime.

## Proposed Solutions

1. Extend the old trial helper: small diff, but risks changing existing experiments
   and lacks a strict evidence/fixture contract.
2. Add a standalone local profiler using the same native protocol: isolates the
   lifecycle and offers reproducible fixtures; managed admission remains separate.

## Recommended Action

Implement option 2 under experiments/model-validation. No device runs, remote
launches, automatic membership triggers or changes to existing workers. User
authorized continuing implementation without waiting for Connor on October 25.

## Acceptance Criteria

- [x] Pin model, artifact, runner/dependencies, workload and reference fixture.
- [x] Replay complete ordered prefill/decode state, checking every output.
- [x] Bound loading, step reads/writes, total wall time, native wire and logs.
- [x] Cancel and clean up owned processes; preserve first failures and partial data.
- [x] Report actual range costs, fresh-process loads and scoped memory without claiming eligibility.
- [x] Test real subprocess lifecycle, stalls, cancellation, malformed replies and mismatches.
- [x] Document command/schema, admission limitations and hardware validation requirements.
- [x] Notify owners with tested results and remaining integration boundaries.

## Work Log

### 2026-10-25 — Start

- Created isolated branch david/bounded-candidate-profiler from main snapshot 3c3cdd4.
- Reviewed native protocol and existing fixed-input CPU capture harness.
- Claimed this bounded diagnostic milestone in Discord; coordinator ownership remains open.

### 2026-10-25 — Implemented and validated

- Added profile_candidate.py, its process-level tests, usage/schema documentation
  and a POSIX Python test step in CI.
- Fixed an initial early-exit cleanup failure: a sandbox may deny a signal to an
  already vanished process group. A live-child signal denial still fails; any
  cleanup failure is retained without erasing the original report.
- Investigated a one-second success-fixture startup timeout. A reproduction
  recorded a loading timeout; later observed loads were 713 ms and 371 ms.
  Normal-success fixture bounds now allow five-second startup/steps; dedicated
  150–200 ms timeout tests retain their original bounds. Runtime limits remain
  explicit request fields, not silently relaxed defaults.
- Final functional run: 29 tests pass on Python 3.13.5/macOS (20 profiler and
  9 existing benchmark tests). Annotation-only edits then passed py_compile;
  git diff --check passes. Added CI coverage has not run remotely.
- Runtime/apps owner received implementation, test results and remaining gates
  through Discord MCP. No hardware profiling or worker deployment was performed.
- GitHub connector returned 404 and HTTPS git failed DNS resolution. Preparing
  an isolated local commit/patch for publication through the existing owner.

This completes the bounded local diagnostic milestone only. Shared admission,
sensor hooks, complete pool accounting, link profiles and deployment remain open.

## Resources

- [Implementation scope](https://discord.com/channels/1563598920800542840/1563603695688622092/1563807551789994006)
- docs/measured-placement.md
- docs/contributor-profiling.md
