# Evidence checks before measured ring placement

The version 2 adapter checks trusted local evidence for completeness, provenance
references and consistency, then calls the existing exact whole-range planner.
Reusable range trials, current availability and directed transfers stay separate.
It does not authenticate a remote measurement, read referenced evidence files,
reserve capacity, download artifacts or start workers. Every result remains
advisory with hardware_attested and activation_ready both false.

## Commands and host API

~~~sh
llm-cluster qualified-plan evidence-v2.json
llm-cluster prepare-qualified-ring evidence-v2.json registry.json > qualified-ring.json
llm-cluster validate-qualified-ring qualified-ring.json evidence-v2.json registry.json
~~~

All commands read bounded JSON (8 MiB per file), use current system time and do
not bind network listeners. Preparation prints an envelope containing the real
leader manifest, worker configurations and argument vectors. It does not write
or activate those files. Keep the entire envelope: extracting only its inner v1
bundle discards the stricter v2 expiration checks.

The qualified_placement module exposes plan_ring, prepare_ring and
validate_deployment with explicit current-time arguments, plus
trial_policy_sha256 for pinning acceptance thresholds before profiling.
QualifiedPlan.plan is null when no route fits. Exclusions identify peer, range
and link records with rejection reasons. Malformed schemas, duplicate
identities/evidence, ambiguous eligible ranges/links and unsupported settings
fail with an error. An unsuccessful trial remains excluded.

## Input contract

EvidenceRequest has schema_version 2, the existing workload, origin_peer_id,
min_peers, policy, peers, ranges and links. Structs in
[the source](../src/qualified_placement.rs) define exact JSON fields and reject
unknown fields. [Integration fixtures](../tests/qualified_placement.rs) show
complete synthetic inputs; their hashes, timings and capacities are not hardware
evidence.

| Input | Required evidence |
| --- | --- |
| PeerAvailability | Stable peer/resource IDs, current Iroh endpoint, authorization, explicit compute opt-in, health, observation time, approved/available/reserved physical-pool bytes, effective execution identity and current power/thermal state. |
| RangeEvidence | Profile/resource IDs, full source workload, exact range, artifact/report hashes, pinned trial-policy hash, measurement start time, outcome, cleanup confirmation, requested repetition count and all sessions. |
| ExecutionIdentity | Runner/dependency/configuration hashes; effective backend and separate backend-evidence hash; GPU setting, native buffer limit, effective context and sequence count. Requested offload or buffer device labels cannot establish the backend. |
| CorrectnessEvidence | Passing result for every step, exact checked-step count, fixture hash, independently retained reference-validation hash, reference runtime/backend. A file hash alone does not validate numerical correctness. |
| MemoryEvidence | Whole-contribution scope, raw evidence hash, accounting explanation, separate load/inference peaks by RAM/VRAM/shared physical pools. GGML-buffer and process-RSS scopes are rejected. |
| ConditionEvidence | Sampled power source, worst thermal state, throttling status, complete monitoring and retained evidence hash. Current power must match the trial. Operator descriptions are insufficient. |
| TransferEvidence | Profile/report IDs, directed peers and exact endpoint IDs, workload/payload, measurement time, transport/protocol, actual direct/relay path, passing outcome and transfer samples. |

Missing execution, correctness, memory or conditions remain unknown and exclude
the observation. Changing a physical resource invalidates its old range evidence;
changing an endpoint invalidates transfers measured against its old identity.

The adapter consumes declarations from a trusted integration owner. Hashes
identify retained evidence; valid hash syntax does not prove that a file exists
or that a sensor measured the claimed quantity. The owner must verify evidence
and its origin before constructing inputs. This is not an admission interface
for arbitrary peer-supplied JSON.

## Policy and complete histories

Policy separates range, capacity and link maximum ages. It sets minimum session
and prefill/decode/transfer sample counts, minimum sustained service duration,
thermal limits and battery permission. Serious, critical and unknown thermal
states are rejected. This adapter supports one-sequence M-27 rings, contexts
32..4096, single-token decode, Iroh and meshthing/shard-ring/1.

Declare trial_policy_version and trial_policy_declared_at_unix_ms before taking
measurements. Pin trial_policy_sha256 into each trial. Its digest covers those
fields, session/prefill/decode counts, sustained duration, thermal limit and
battery permission; decision freshness and link policy are separate. Changed
thresholds, a mismatched digest or a policy declared after measurement began
exclude the trial.

Each session starts at position zero, contains prefill followed by single-token
decode, preserves every position and fills the requested context occupancy.
All requested repetitions must finish with the same batch/position history and
every output checked. A short-context fixture cannot qualify a larger context.
The sustained duration threshold applies to one uninterrupted session; separate
short reloads are not added together. Load and per-step compute/service timings
remain in the original evidence.

Decode cost is nearest-rank p95 service time for the whole exact range. Link cost
is nearest-rank p95 for the whole directed payload transfer, excluding native
execution. The search minimizes their sum. **The sum of component p95 values is
not an end-to-end latency p95 or a statistical guarantee.** Prefill samples are
required, but this objective does not predict prefill latency or throughput.

For each physical pool, the greater of load and inference peaks must fit the
fresh usable capacity (which is already net of existing reservations), and that
peak plus the separately reported reservations must fit the owner-approved
contribution budget. Do not subtract reservations from usable capacity again.
Count unified memory once in shared. Do not add RSS to GGML buffers. Accounting
evidence must cover load temporaries, staging, native/host processes, drivers,
state, scratch and outputs within an explicit attribution boundary.

## Binding and revalidation

Preparation matches selected evidence against registry artifact and runner
hashes, GPU settings, native buffer budget, physical resource and endpoint.
Existing ring topology and real manifest validation remain in force. Native
buffer limits are distinct from total physical-pool charges.

qualification.valid_until_unix_ms is the earliest selected range, capacity or
link expiry under its own policy. Original timestamps are preserved. Validation
recomputes the entire envelope and rejects changed evidence, availability,
policy, registry or generated configuration. Refresh availability and links
externally first: validation cannot discover an unreported departure.

Activation still requires runtime-owned reservations and cache retention, local
file/build/dependency verification, current authorization/health/path checks,
worker readiness and deployment-version fencing. A downloaded artifact is
staged; a prepared configuration remains advisory. No retries, process launch
or fleet epochs are added here.

## Existing native diagnostics

Native candidate diagnostic v1 is a different producer schema. Preserve it with
the bounded importer before reviewing or transforming observations:

```sh
llm-cluster import-native-diagnostic --report report.json --samples samples.jsonl \
  > diagnostic-import.json
```

The importer checks the exact kind/version and false eligibility flag, computes
hashes of the supplied report and sample bytes, and retains the complete report
plus every sample row in order. These hashes identify the supplied files; they
do not verify that the files belong to the same profile run. The CLI accepts
separate paths, and matching the samples to the report is the caller's
responsibility. It preserves failed, canceled, running and incomplete
outcomes; malformed or oversized input fails without a partial import. The
versioned result is diagnostic-only and is not an `EvidenceRequest` or
`RangeEvidence`. A passed status or renaming the schema cannot grant
eligibility.
The current producer lacks independently established effective-backend evidence,
complete physical-pool load/inference accounting and sampled power/thermal data.
Leave those fields unknown until collected and verified. Native configuration
and buffer snapshots do not satisfy the missing scopes.

The integration tests exercise actual planner/configuration code and all three
CLI commands using synthetic observations. They cover incomplete trials,
histories, sustained duration, identity changes, freshness, budgets, departures,
tail costs and envelope tampering. They are not hardware benchmarks.
