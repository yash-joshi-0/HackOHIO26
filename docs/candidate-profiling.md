# Bounded native candidate profiling

`experiments/model-validation/profile_candidate.py` runs one local native range
against a pinned, teacher-forced reference fixture. It uses the existing MSH1
native runner protocol on POSIX hosts. It creates a new process/context for each
repetition and replays every prefill/decode input in order from position zero.
Each output tensor is checked against the reference before its timing enters the
successful-sample summary. No device names or fixed layer assignments are built in.

This is the executable diagnostic milestone for
[adaptive placement](https://discord.com/channels/1563598920800542840/1563603695688622092/1563801650936684609).
It does not implement automatic compute admission. Aarush's measured planner
remains advisory, and Yash's M-27 ring remains static per launch with leader-local
stage zero. Connor's response is not required to exercise this local profiler.

## Run and cancel

After the owner enables compute and a local idle test window is coordinated:

```sh
python3 experiments/model-validation/profile_candidate.py request.json \
  --output-dir results/candidate-trial-001 \
  --cancel-file /tmp/cancel-candidate-001 --confirm-idle
```

Create the cancellation file or send SIGINT/SIGTERM to the profiler to cancel.
Cancellation is polled during hashing and pipe operations. Owned native processes
receive TERM, then KILL if needed; partial results are retained. Use a fresh
cancellation path and output directory for a new run. An existing output directory
is never overwritten. The exit code is zero only for `passed`.

The explicit idle confirmation is an operator check, **not a shared runtime
admission gate**. It cannot prevent another application or mesh worker from
starting. Do not connect this command to membership events or expose automatic
app profiling until the runtime supplies shared inference/profiling admission.
It launches only the explicitly supplied local executable; it has no SSH, device
deployment, artifact download or network probe behavior.

## Request version 1

All fields shown below are required. Replace descriptive hash placeholders with
actual lowercase SHA-256 values. Paths resolve relative to the request file.

```json
{
  "schema_version": 1,
  "device_id": "owner-supplied-peer-identity",
  "model": {
    "source_sha256": "FULL_MODEL_SHA256",
    "layers": 64,
    "embd": 5120,
    "vocab": 248320
  },
  "range": {"begin": 20, "end": 24},
  "context_tokens": 512,
  "artifact": {"path": "range.gguf", "sha256": "EXPORTED_ARTIFACT_SHA256"},
  "runner": {
    "path": "mesh-shard-runner",
    "sha256": "EXECUTABLE_SHA256",
    "runtime_id": "exact-build-and-backend-configuration",
    "gpu_layers": 0,
    "dependencies": [
      {"path": "libllama.dylib", "sha256": "LIBRARY_SHA256"}
    ]
  },
  "fixture": {"path": "fixture/fixture.json", "sha256": "FIXTURE_MANIFEST_SHA256"},
  "repetitions": 2,
  "min_session_seconds": 60,
  "conditions": {"power": null, "thermal": null},
  "limits": {
    "total_seconds": 600,
    "load_seconds": 120,
    "step_seconds": 10,
    "max_wire_bytes": 268435456,
    "max_fixture_bytes": 67108864,
    "max_log_bytes": 1048576,
    "max_observed_buffer_bytes": 1610612736,
    "max_reported_rss_bytes": 2147483648
  }
}
```

The dimensions, ranges and limits are an **illustration**, not a measured device
budget or a reusable hardware recommendation. Derive budgets from current usable
capacity and the owner's contribution policy before running. Model dimensions
must match the actual artifact; do not infer them from its name.

Artifact, executable and listed dependency hashes are checked before launch.
Their file identity, size and modification time must remain unchanged through
the run. The runner's source hash, dimensions and range must match the request.
An unexported full model with empty source metadata is accepted only when its
file hash equals the requested source hash. Immutable artifact staging and full
runtime dependency attestation remain the integrating runtime's responsibilities.

The native runner currently fixes one sequence, four CPU threads, a batch ceiling
of 128 and disabled flash attention. `gpu_layers` is a **requested** offload
configuration. This protocol does not attest the backend actually selected.
Record the exact build/configuration in `runtime_id` and retain the native log;
do not invent an effective backend from the request or a device label.

## Reference fixture version 1

The fixture repeats the request's `model`, `range` and `context_tokens` exactly:

```json
{
  "schema_version": 1,
  "model": {"source_sha256": "FULL_MODEL_SHA256", "layers": 64, "embd": 5120, "vocab": 248320},
  "range": {"begin": 20, "end": 24},
  "context_tokens": 512,
  "reference": {
    "runtime_id": "validated-reference-build",
    "runner_sha256": "REFERENCE_EXECUTABLE_SHA256",
    "backend": "actual-reference-backend"
  },
  "atol": 0.001,
  "rtol": 0.001,
  "steps": [
    {"phase": "prefill", "n": 15,
     "input": {"path": "prefill-input.bin", "sha256": "INPUT_SHA256"},
     "expected": {"path": "prefill-output.bin", "sha256": "OUTPUT_SHA256"}},
    {"phase": "decode", "n": 1,
     "input": {"path": "decode-0-input.bin", "sha256": "INPUT_SHA256"},
     "expected": {"path": "decode-0-output.bin", "sha256": "OUTPUT_SHA256"}}
  ]
}
```

Fixture payload paths resolve relative to the fixture manifest. There must be
prefill followed by one or more single-token decode steps, without missing state
positions or a return to prefill. Multiple prefill batches are allowed. The total
token count must fit the requested context. A middle/final range receives captured
inputs from the preceding layers under the same token history; do not repeat a
single activation vector as if it represented later positions.

Layer-zero inputs are big-endian u32 token IDs. Other ranges receive little-endian
f32 activations of shape `n * embd`. Expected outputs are little-endian f32:
`n * embd` values for nonterminal ranges, or `vocab` logits for terminal ranges.
The profiler always requests full terminal logits. All tensors must be finite.
The per-value check is `abs(actual-reference) <= atol + rtol*abs(reference)`.
Choose tolerances from validated model/backend evidence; the example tolerances
are not an acceptance policy for Qwen or another model.

Capture reference outputs with an independently validated execution path and
retain that provenance. Hashing a reference identifies its bytes; it does not
prove the reference is correct. The existing native CPU capture experiments
show how to retain complete input/output histories, but their device identities,
ports and paths must not become automatic scheduling rules.

## Outcomes and measurement scopes

The output contains:

- `report.json`: profile/run ID, pinned request, provenance, current/terminal
  phase and status, per-repetition initialization/load time, verified sample
  counts and separate prefill/decode compute/service distributions.
- `samples.jsonl`: completed response observations, repetition, position, batch,
  host-local elapsed time, compute/service duration, native buffer/RSS/VM scopes
  and numerical correctness result. A mismatched tensor is retained with a
  failed correctness result and excluded from timing summaries.
- `runner.log`: bounded native stderr across repetitions, including first errors.

`passed` means the fixture checks and declared diagnostic thresholds completed.
Other terminal outcomes are `failed`, `timed_out`, `cancelled`, `limit_exceeded`,
`insufficient_duration` and `cleanup_failed`. Never retry into the same output
directory or silently drop an unsuccessful repetition.

Every report sets `eligible_for_placement: false`. This diagnostic schema is
separate from M-5 planner schema v1 and the proposed v2 evidence contract. It can
supply raw whole-range evidence to a later adapter; it must not be fed directly
to `measured-plan` or converted into invented per-layer timings.

Compute time is the runner's synchronized native decode interval. Local service
time includes writing the request and receiving metadata/tensor over pipes; it
is not network latency. Load time includes process startup and model/context
initialization. A fresh process resets state, but does not imply a cold filesystem
cache, cold GPU cache or cooled hardware. p95 uses the nearest-rank method and is
computed separately for each phase across the fixture's recorded positions.

`min_session_seconds` requires the sum of native request service intervals in
at least one uninterrupted process session to reach the requested duration.
Independent reloads are never added together to satisfy it. If the fixture ends
too soon, the outcome is `insufficient_duration`; capture a longer valid history
or use a context that supports the intended test. Numerical checking between
requests adds host overhead, so even a passing 60-second run does not establish
thermal equilibrium or production workload utilization. Per-step records permit
analysis against context occupancy and elapsed time without attributing drift
to heat automatically.

Time limits cover hashing, loading, request writes/reads and normal shutdown.
Forced cleanup adds up to 2.5 seconds of bounded waiting. The native wire budget
includes requests plus stdout metadata/tensors; stderr has a separate disk bound.
Fixture bytes are bounded and verified before native loading.

Buffer and RSS limits are **observational stop thresholds** at reported init/step
boundaries. They do not prevent an allocation before the next observation.
GGML buffers, RSS and Darwin physical footprint can overlap; do not add them.
The parent process, driver, staging and other processes are not fully accounted.
Thermal/power fields are operator descriptions or null, not sampled sensors.

## Validation and next integration gate

```sh
python3 -m unittest discover -s experiments/model-validation -p 'test_*.py' -v
```

The subprocess tests use explicitly synthetic data to exercise state reset,
identity/correctness checks, read/write stalls, early exit, oversized metadata,
wire/log/memory thresholds, cancellation and cleanup. They establish no hardware
throughput, capacity, thermal or backend correctness result.

Before app-driven profiling and placement admission: integrate shared runtime
admission, device power/thermal callbacks, effective backend identity, full scoped
capacity/loading peaks, current availability and matched Iroh links. Then perform
a coordinated real-device baseline/candidate trial using the same model, fixture,
context and runtime. Reservation and versioned fleet activation remain separate
work; no running deployment is changed by this command.
