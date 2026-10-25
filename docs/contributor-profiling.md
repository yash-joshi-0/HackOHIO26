# Contributor profiling and evidence record

Use this record before considering a device for model placement. Fill each field
from the actual device and executed runtime. Leave unknown values as **not
measured**; a device name or installed RAM figure does not establish a safe
contribution budget.

For a bounded local native range test with pinned correctness fixtures and
cancellation, see [candidate profiling](candidate-profiling.md). Its diagnostic
output does not establish complete capacity accounting or placement eligibility.

## One record per device, model, runtime and operating mode

| Field | Observation to record |
| --- | --- |
| Device identity | Owner's device name, stable peer ID, hardware model and architecture |
| System | Exact OS/build, physical RAM, available memory or app limit at test time, and storage free at test time |
| Backend | Backend actually selected by the runtime; GPU model and dedicated/shared VRAM where applicable |
| Workload | Model file hash, runtime build/patch ID, context, batch/slot count, shard range, prompt and output length |
| Policy | Owner authorization, foreground/background rules, power source, battery state, thermal state and throttling observed over a sustained run |
| Memory | Opt-in `memory_budget_bytes`, conservative `reserved_bytes`, measured process/driver/bridge scopes, sampling method and peak window covering load **and** inference |
| Compute | Fresh per-layer or per-shard timings, compute-event source, cold/warm distinction and repeated-run spread |
| Network | Peer IDs, Iroh path (direct/relay), directed transfer cost for this workload, reconnect and suspension behavior |
| Outcome | Exact commands/configuration, raw log paths, observed layer/output placement, errors and whether an interrupted session needed re-prefill |

Keep separate measurements separate. GGML buffer totals, process footprint,
RSS, GPU allocations and shared-memory counters can overlap or omit other
processes; do not sum them into a purported total. Record the accounting
boundary and any unmeasured allocations. A nonzero `peak_bytes` value is not
trustworthy merely because the verifier accepts it.

### Devices awaiting Aarush's measurements

Create a separate record for each of these device names as supplied. No RAM,
OS, backend or budget has been inferred for them.

| Device | First missing facts |
| --- | --- |
| First-generation iPad | Exact model and OS/build, app memory ceiling, supported backend, foreground/suspension behavior |
| iPhone 17 Pro | Exact OS/build, available app budget, supported backend, foreground/suspension behavior |
| G14 (2024) | Exact OS/build, CPU/GPU identity, usable VRAM/RAM, power and thermal behavior |

These records are diagnostic candidates. Enroll one as a compute worker only
after its native shard execution and Iroh path pass on that physical device.
For mobile, treat app suspension or network loss as peer loss; restarting a
worker requires explicit state restoration and re-prefill, not preserved KV.

## Translate observations into planner inputs

- `ModelProfile` needs measured weight and KV costs for the exact model,
  runtime and context. Include architecture-specific non-KV state in reserves.
- `PeerProfile` needs the same identity/context, explicit authorization and
  current health, an owner-approved application budget, conservative reserve,
  and measured decode cost for every layer. An unmeasured candidate is not
  eligible for a live plan.
- `LinkProfile` needs the directed transfer overhead observed for this workload.
  RPC traffic can traverse the leader even when workers are connected by Iroh.
- After one fresh run, `RunEvidence` needs one entry for every selected peer:
  exact model/runtime/context identity, peak **total contribution** bytes over
  load and inference, and fresh compute events. Keep the raw source alongside
  the JSON supplied to `llm-cluster verify` or `--evidence-file`.

`planning::validate_output` checks the saved plan before launch.
`planning::verify_output` checks supplied evidence after the run. Neither
collects telemetry or enforces a memory cap. The leader currently accepts an
externally written evidence file; the operator must associate it with the exact
run and avoid reusing a stale file. A passing verifier result does not establish
that excluded allocations fit the budget.

## Current evidence boundary (2026-10-24)

The [two-Mac shard-runner result](../experiments/model-validation/SHARD_RUNNER.md)
shows Qwen2.5-32B Q4_K_M weights larger than either Mac's physical RAM and
both exported partitions executing one generation over Iroh. Its GGML buffer
samples and native process footprint peaks are scoped diagnostics. The bridge,
endpoint, coordinator and some driver allocations are outside those counters;
hard cap enforcement and a planner-driven leader/job API run remain open.

David also [reported a three-device Qwen3.8-27B run](https://discord.com/channels/1563598920800542840/1563603695688622092/1563688552880021649)
on 2026-10-24 over Iroh
(M2 Pro, M4 iPad, RTX 5060 Ti). Treat that as a reported result until its
model/runtime manifest, commands and raw evidence are published and reviewed.
It does not by itself establish mobile background operation, recovery or the
M-5 total-contribution budget gate.
