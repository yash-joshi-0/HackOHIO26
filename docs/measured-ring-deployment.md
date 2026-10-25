# Prepare a ring deployment from measured ranges

The core can now generate the existing experimental ring runtime's configuration
from measured placement. It searches with the actual backend constraint: stage
zero runs on the origin, followed by at least one remote stage. It keeps exact
whole-range measurements and uses each peer at most once.

```sh
llm-cluster prepare-ring current-profile.json registry.json --output-dir prepared
llm-cluster validate-ring prepared/bundle.json current-profile.json registry.json
```

`current-profile.json` uses the [measured placement v1 schema](measured-placement.md).
The preparation command checks observations against the current time, even if the
file's `observed_at_unix_ms` is old. It requires single-token decode observations.
The original `measured-plan` command retains its unconstrained behavior.

The output directory must not exist. It contains:

- `leader-manifest.json`, accepted by the real `ShardManifest` loader.
- `worker-1.json`, `worker-2.json`, etc., with each selected artifact, native
  runner, profile identity, upstream/next/origin and existing CLI arguments.
- `bundle.json`, written last as the completion marker. It retains the measured
  plan, selected registry bindings, input digests, expiration and leader arguments.

The tool does not start processes or change devices. `activation_ready` remains
false. Configuration validity does not qualify profiler results, reserve memory,
or prove that files exist on another device.

## Registry

The registry is trusted operator configuration mapping measured peer/profile IDs
to actual endpoint identities and already exported artifacts. This illustrative
fragment uses placeholders, not usable hardware measurements or hashes:

```json
{
  "schema_version": 1,
  "model_sha256": "FULL_MODEL_SHA256",
  "runtime_id": "EXACT_WORKLOAD_RUNTIME_ID",
  "chat_template": {
    "jinja": {"path": "/models/model-template.jinja", "sha256": "TEMPLATE_SHA256"},
    "enable_thinking": false
  },
  "max_tokens": 128,
  "leader": {
    "cluster": "my-mesh",
    "manifest_path": "/models/prepared/leader-manifest.json",
    "api": "127.0.0.1:8080",
    "upstream_port": 8081,
    "worker_wait_seconds": 120
  },
  "devices": [{
    "peer_id": "MEASURED_PEER_ID",
    "resource_id": "PHYSICAL_DEVICE_ID",
    "endpoint_id": "EXISTING_IROH_PUBLIC_ID",
    "key_path": "/private/meshthing.key",
    "runner": "/opt/layercake/mesh-shard-runner",
    "runner_sha256": "RUNNER_SHA256",
    "artifacts": [{
      "range_profile_id": "EXACT_RANGE_PROFILE_ID",
      "begin": 0,
      "end": 8,
      "model": "/models/range-0-8.gguf",
      "artifact_sha256": "EXPORTED_ARTIFACT_SHA256",
      "gpu_layers": 0,
      "budget_bytes": 1073741824
    }]
  }]
}
```

Supply a device entry for every selected peer and an artifact entry for its exact
profile ID and range. Paths belong to the destination host and must be absolute;
Windows drive/UNC paths are accepted when preparing on another OS. The key file
must already identify the recorded endpoint. Do not copy private keys into the
bundle. A registry entry supplies the key's path only.

Artifact entries may declare `role` as `first`, `middle`, `final` or `full`.
Preparation checks that the role matches the exact range and model layer count.
For example, a `middle` export cannot be bound to the terminal stage just because
its layer range ends at the last layer. Omitted roles preserve legacy registries
and bundle hashes, but remain unknown. The declaration is an early compatibility
check: staging must still verify the GGUF metadata and actual required tensors
(first-stage embeddings, final output/output_norm or tied embeddings, and shared
tensors). A matching role never changes `activation_ready` to true.

The registry's model/runtime must equal the measured workload. Range and link
profile IDs must be unambiguous. Selected endpoints must have distinct physical
resource IDs, so multiple identities cannot multiply one device's capacity.
The registry is an operator assertion of these associations, not independent
attestation. A changed native build, dependency, backend or configuration requires
matching new measurements and bindings. The v1 workload's shared runtime ID does
not replace per-device native build provenance.

`budget_bytes` is the runner's GGML buffer limit. Set it from that measured scope;
it is not derived by summing RAM/VRAM/shared pools and is not a total allocation
cap. The planner's per-pool contribution charges remain separately recorded.

## Use the generated configuration

After qualification, current opt-in/capacity checks and deployment ownership are
established, stage and verify each selected artifact and native build on its
destination. Verify the existing key's public identity too. Copy the generated
leader manifest to `leader.manifest_path` on the leader. Verify the pinned chat
template against the actual model.

The `argv` arrays target the existing `llm-cluster` executable. Pass each element
as a process argument; never join the array into a shell command. Worker arguments
include the exported artifact hash and original model hash, exact range/context,
native buffer limit, allowed upstream, origin and next peer. The terminal has no
`--next`. The leader uses `leader --backend shard --shard-manifest ...`.

The existing ring runtime checks its ready chain against the manifest on each
job; the leader also runs its generation readiness probe before opening the job
API. Preparation does not replace these checks. Partial startup must be stopped
and cleaned up by the runtime owner before a different deployment starts.

Changing available peers or measured costs produces a new bundle, including new
range boundaries when a measured alternative wins. This is configuration
preparation, not live migration: keep the active request on its current plan and
activate replacements only through the owning runtime's drain/readiness process.

Expiration is the earliest selected capacity/range/link observation plus the
request's maximum age. `validate-ring` checks the exact generated contents against
the original inputs and rejects expired/future bundles. It cannot observe current
remote health, capacity or path changes: refresh those inputs and prepare again.
Input digests detect mismatched files; they are not signatures or authentication.

## Remaining activation gates

The standalone candidate diagnostic remains ineligible for direct planner input.
Complete correctness, effective backend, memory scopes, thermal/power and matching
directed-link evidence are still required. Remote reservations, versioned fleet
activation, mobile embedding and recovery from a departed worker are separate work.
This tool never sets a diagnostic's eligibility to true or invents missing costs.

Tests cover real planner-to-manifest/worker-config conversion, the existing CLI
parser, expiration, modified inputs, ambiguous identities/artifacts, and replanning
when a peer becomes unavailable. Their measurements and keys are synthetic; they
establish integration behavior, not hardware performance.
