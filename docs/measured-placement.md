# M-5 measured placement path

`llm-cluster measured-plan PROFILE.json` reads version 1 range and link
observations and prints an **advisory** sequential decode placement. It does
not launch a worker, write a shard manifest, reserve memory, or verify a run.
The existing `plan` command and llama RPC launch path remain separate.

`prepare-ring PROFILE.json REGISTRY.json --output-dir NEW_DIRECTORY` adds a
ring-constrained search and generates validated leader/worker configurations.
See [measured ring deployment](measured-ring-deployment.md). It preserves the
advisory/evidence boundary and does not reserve or start devices.

Run the synthetic fixture:

```sh
cargo run -- measured-plan examples/measured-placement-profile.json
```

Save the output, refresh the profile's capacity and time observations, then
recheck it before any launch:

```sh
cargo run -- validate-measured saved-plan.json current-profile.json
```

That fixture chooses peer `b` for layers `0..2`, then peer `a` for `2..4`.
Its invented times and capacities are for testing only. Replace every
observation and identity before drawing a hardware conclusion.

## Input contract

The [fixture](../examples/measured-placement-profile.json) shows every field.
`workload` pins the full model source hash, exact runtime/build identity,
context, batch, layer count and payload sizes. Every range and link observation
repeats that identity; a mismatch makes that observation ineligible. Give each
observation a stable `profile_id` and its measurement time. `observed_at_unix_ms`
and `max_profile_age_ms` reject old or future observations. A real request
should use the current time and an age limit appropriate to capacity, thermal
and network changes.

Each peer supplies authorization and health, a timestamped observation of
`available_bytes` after other workloads, an owner-approved `budget_bytes`,
conservative `reserved_bytes` and exact measured `[begin,end)` ranges. A
selected range must fit both the approved limit and currently available
capacity. Stale capacity makes the peer
ineligible. Each range supplies measured
decode time and peak bytes for **that range** over load and inference. A whole
range observation stays whole; the planner never divides it into per-layer
costs. Missing ranges make those assignments impossible.

Memory values have three distinct physical pools: `ram`, `vram` and `shared`.
Put unified-memory allocations in `shared` once. Do not put the same bytes in
both host and device pools. Charge bridge, process, driver, model staging,
KV/recurrent state, scratch, output tensors and OS headroom according to their
actual accounting boundaries. Each selected range's peak plus its peer's
reserve must fit all three budgets. If a scope cannot be measured or bounded,
leave that peer out of a live placement request.

Links are directed observations keyed by `from`, `to` and exact payload size.
The planner requires measurements for origin-to-first control, each adjacent
activation transfer, and final-to-origin control when those peers differ. It
adds those measured transfer times to range decode times. Profile `path`
records direct or relay transport; use observations from the actual expected
path and workload. Do not insert a raw RTT, a whole-ring residual, or a
different wire protocol as a directed transfer measurement. The input allows
one eligible observation per range or directed payload edge so there is no
ambiguous choice between trials.

The search uses each peer at most once and covers every layer with contiguous
ranges. It jointly chooses order and range boundaries for up to eight eligible
peers, respecting `min_peers`. An unmeasured transition or an over-budget
range makes that route impossible. The output includes ordered assignments,
charged memory, contributing profile IDs, planning time and predicted decode
time. It is a prediction for one sequential request, not a throughput or
prefill estimate. `measured_placement::validate_output` recomputes a saved plan
against refreshed inputs and rejects changed or expired profiles.
To bound validation and exact-search memory, the planner accepts at most 4,096
range observations per peer and 4,096 links, rejects labels longer than 256
UTF-8 bytes, and stops if it would retain more than 16,384 search states.

## Version 2 evidence contract (proposal)

The current planner consumes schema v1 and does not consume profiler output.
Before extending it, keep these inputs separately versioned and refreshed:

- **Reusable range evidence:** profile ID and schema version; model source and
  exported-artifact hashes; backend, runtime, build and configuration identity;
  exact `[begin,end)`; fixture ID/hash and correctness result; separate prefill
  and decode timings with token counts; context occupancy/limit, batch and
  sequence count; sample count; scoped peak memory during load and inference
  by `ram`, `vram` and `shared`; thermal/power state; and terminal outcome with
  failure/cancel reason. Retries retain earlier failures rather than replacing
  them with successful evidence. Timings are local to the measured host.
- **Current peer availability:** authorization/health, owner-approved
  per-pool budget, fresh usable capacity and existing reservations. This is
  refreshed for each placement decision and is not reusable range evidence.
- **Directed transfer evidence:** source/destination, transport/protocol,
  actual direct/relay path, exact payload size, workload identity, sample
  distribution including tail, profile ID and freshness. Do not infer link
  cost from RTT or whole-chain residuals.

The planner remains advisory. Its result should identify the selected ordered
ranges, all contributing profile IDs, objective and predicted prefill/decode
costs. Reservation, worker readiness and manifest activation remain runtime
responsibilities. The existing v1 JSON contract is unchanged until a compatible
input schema and migration are implemented.

### Native candidate diagnostic input boundary

The current producer is `native_candidate_diagnostic` schema version 1, emitted
by `experiments/model-validation/profile_candidate.py`. Its `report.json`
includes a pinned request, reference provenance, per-session initialization and
completion state, and summaries; `samples.jsonl` retains each checked step and
its buffer/RSS/VM observations. The report deliberately sets
`eligible_for_placement: false` for every outcome. This flag remains false even
when `status` is `passed`: it means the pinned local fixture completed its
diagnostic checks, not that the range is qualified or admitted.

An M-5 adapter may read this format only as raw source evidence. It must check
both schema version and `kind`, retain the report and sample-file identities,
and preserve every session and terminal outcome. A non-passing report, missing
or incomplete session, missing required sample, or failed correctness result
cannot be converted into a successful range observation. A retry is a separate
profile; it cannot overwrite or repair the earlier outcome. The request pins
the model source, artifact, runner and listed dependencies, exact range,
context, fixture and requested offload settings, but the current report does
not attest the effective backend or unlisted dynamic dependencies. Its buffer,
RSS and VM measurements do not establish complete physical-pool contribution
peaks, and the operator-supplied thermal/power descriptions are not sensor
measurements. It also has no shared compute admission, current availability,
reservation, or directed-link evidence.

Keep the producer's timing fields separate as well. `compute_ms` is the
runner's synchronized native interval; `service_ms` includes local request
write and response-pipe handling. Neither is mesh transfer time, and service
time is not end-to-end request latency. A qualification record must preserve
both distributions and identify the exact versioned cost objective used by a
planner. For example, whole-range service p95 plus directed-link p95 is a
predicted additive objective; it must not be labeled measured end-to-end p95.

Therefore this diagnostic format is not a serialization of measured-planner
v1 `RangeObservation`, and a passing report cannot be copied into that input.
Promotion requires a separately versioned qualification record that links the
raw report and samples to independently attested backend identity, complete
per-pool load/inference peaks, an explicit versioned threshold policy, full
successful repetitions, and the required reference/workload provenance. The
qualified record remains a whole-range observation. Placement admission then
joins it with fresh capacity/reservation and directed-link evidence; neither
step mutates or upgrades the diagnostic report itself.

### Qualification and admission gates

Keep four states distinct; they are not a single `passed` or `eligible` flag:

1. **Diagnostic only.** Preserve every raw report and sample file with its
   outcome and provenance. The diagnostic runner's `eligible_for_placement`
   value remains false. Neither a passed fixture comparison nor a valid hash
   makes its range eligible.
2. **Qualified range observation.** A versioned M-5 qualification record may
   refer to a whole native range only when source and exported-artifact hashes,
   exact range, runner/dependency/runtime identity, effective backend and how
   it was observed, context/batch/sequence, and independently validated
   fixture/reference provenance match. Qualification thresholds and required
   sample/session counts must be declared in a versioned policy before the
   run. Required prefill and decode histories must be complete, and every
   repetition used must complete and pass all required output checks. Keep
   failed or excluded repetitions with reasons; do not combine successful
   steps from an incomplete repetition. Require complete physical-pool peaks
   over both load and inference, separated into `ram`, `vram`, and `shared`
   with shared memory counted once, plus condition measurements and timing
   distributions. Requested offload settings, buffer totals, RSS, or an
   operator-entered thermal label alone cannot satisfy these requirements.
   A missing or ambiguous identity, scope, backend, condition, or output keeps
   the range diagnostic-only. Do not invent universal memory, thermal, or
   timing cutoffs.
3. **Admissible for a placement decision.** Join a qualified range with fresh
   peer authorization/health, owner-approved per-pool budgets, current usable
   capacity and reservations, and directed link samples for the exact peers,
   path/protocol, payload size, and workload. Any stale, mismatched, unknown,
   or missing input excludes that candidate route. Refresh capacity and link
   observations independently from reusable range qualification.
4. **Activation ready.** The runtime/deployment owner sets this only after
   holding local compute admission and destination memory reservations,
   staging and hash-verifying the required artifacts and runtime builds on
   each host, checking device identities and the expected contiguous worker
   chain, and completing an end-to-end generation check. A process-local
   compute lease prevents conflicting compute in that process; it is not a
   fleet-wide memory reservation. A plan, manifest, artifact download, or
   diagnostic pass alone is not activation readiness.

Changing model or shard bytes, runner/dependencies/backend, workload, fixture
or reference, memory-accounting scope, or qualification policy invalidates the
affected range record and requires new evidence. Availability and transfer
observations expire on their own freshness limits. A qualified range remains
whole; do not derive per-layer times by dividing it or edge costs from RTT or
whole-chain residuals.

## Runtime acceptance still required

### Available Qwen3.8 ring evidence (partial)

The two fixed-route runs in
[`QWEN38_RING.md`](../experiments/model-validation/QWEN38_RING.md) provide
actual contiguous-range compute samples for source model
`2de73110cb254cbf09b54b717578dadff12ef1194e7271527e68202f39ba4bfd`, context
512 and the same 0–28 / 28–32 / 32–64 route. Each run contains 44 decode
samples per range (`phase: decode`); the values below summarize each range's
reported native `compute_ms`, not end-to-end transfer or request latency.

| Range and device | Run 01 median / p95 (ms) | Run 02 median / p95 (ms) | Decode-sample max buffer (bytes) | Decode-sample max RSS (bytes) |
|---|---:|---:|---:|---:|
| `[0,28)` M2 Pro Metal | 52.756 / 59.271 | 53.830 / 58.838 | 8,253,922,752 | 7,788,724,224 |
| `[28,32)` M4 iPad Metal | 20.126 / 22.509 | 19.835 / 22.268 | 1,175,917,248 | 1,424,426,496 |
| `[32,64)` RTX 5060 Ti CUDA | 25.909 / 26.043 | 25.928 / 26.015 | 10,391,506,464 | 727,334,912 |

These numbers are measured compute evidence for those exact ranges and that
fixed workload. They do not make an eligible M-5 input: the runs do not
provide matching directed per-payload transfer measurements, current usable
capacity or reservations, a complete contribution peak on every host, or
per-artifact hashes for each exported stage. Buffer, RSS and physical-footprint
metrics have different accounting scopes and must not be added together. Keep
these observations provisional until those missing inputs are collected.

The bounded profiling lifecycle, cancellation behavior, admission requirements,
and ownership boundaries are documented in
[`runtime-profiling.md`](runtime-profiling.md). That contract does not itself
implement a candidate-range executor, fleet memory reservation, or deployment
activation.

Pipeline mode accepts one remote prefix followed by a local final stage. The
experimental M-27 ring now accepts a leader-local prefix followed by contiguous
remote stages ending at the final layer. `plan_ring` searches under that
constraint and requires at least two stages; `plan` retains its unconstrained
behavior. The `prepare-ring` adapter binds exact ranges to the static ring
manifest and worker arguments. Neither planning nor preparation qualifies raw
diagnostics, reserves capacity, stages artifacts, or activates a deployment.

To make this evidence eligible, collect exact exported-artifact hashes, fresh
usable capacity and reservations, complete contribution peaks across load and
inference, and directed costs for every exact transfer size and path. Re-run
the planner-selected route through the leader/job API with the same artifacts,
runtime and workload; verify exact-run evidence and compare repeated measured
decode times with predictions. Record failures and exclusions. M-5 remains in
progress until those gates pass.
