# Seed catalogs and verified shard staging

One seed keeps the full pinned model. Compute devices fetch **whole exported GGUF
shards**, first a small candidate for profiling and then the exact range selected
by placement. They do not need the full model. A completed download is **staged**;
worker admission, profiling, numerical checks, and activation are separate gates.

## Core contract

```rust,ignore
use llm_cluster::artifact::{ArtifactCache, CancellationToken, Seed};

// In the existing runtime process, using its existing authenticated mesh Node:
let cancel = CancellationToken::new();
let seed = Seed::verify(seed_spec, node.address(), &cancel).await?;
let catalog = seed.catalog().clone();
node.set_artifact_seed(Some(std::sync::Arc::new(seed))).await;

let cache = ArtifactCache::open(cache_dir, catalog.model.clone(), quota_bytes,
                                Some(node.clone())).await?;
let progress = cache.progress(); // watch receiver: phase, completed/total bytes
let candidate = catalog.smallest_candidate()?;
cache.retain(&candidate.artifact_sha256, true, &cancel).await?;
let verified_local_path = cache.ensure_artifact(candidate, &cancel).await?;
// Register the candidate with the worker-local profiler. Nothing is activated.
// After profiling and placement select a supported exported range:
let assigned = catalog.exact_range(begin, end)?;
cache.retain(&assigned.artifact_sha256, true, &cancel).await?;
let assigned_path = cache.ensure_artifact(assigned, &cancel).await?;
// Release candidate retention when no profiler/worker uses it. If candidate and
// assignment are the same artifact, keep the retention until inference releases it.
```

`retain(hash, true, cancel)` must happen **before** staging an artifact that will
be used by a profiler or worker. Retentions persist across restarts. The owner
calls `retain(hash, false, cancel)` after the last consumer releases it. Retention
is a boolean ownership contract, not a reference counter: the runtime should
coordinate overlapping consumers. Unretained verified artifacts and partials
remain reusable until the disk quota/free-space policy evicts them, oldest first.

The selected catalog must come from the mesh's chosen, trusted seed. The device's
`ModelPin` is passed separately to the cache and every manifest must match it.
A manifest carries version, immutable source revision, source SHA-256/bytes,
architecture, decoder layer count, half-open layer range, first/middle/final/full
role, artifact SHA-256/bytes, export report hash, and an Iroh or HTTPS location.
Catalog selection does not create exports or choose an inference placement.
An unavailable exact range returns an error requiring a proper export.

## Seed provenance

Use `experiments/model-validation/export_prefix.py` to produce new exports when
needed. It preserves source metadata, original layer numbering and quantized
tensor bytes, and includes the embeddings/output tensors required by the role.
Do not split the full file into arbitrary byte chunks. The experimental
`artifact_range_fetch.py` reconstructs exports from source ranges and is not used
by this downloader.

A private `SeedSpec` JSON contains `model`, `source` (full GGUF path), and `shards`
(an explicit list of already-exported paths). Each shard needs its adjacent
`.gguf.json` report. `Seed::verify`:

1. Hashes the full source and checks pinned size/SHA-256.
2. Checks each report's source identity, architecture and decoder count.
3. Hashes each shard, checking its reported size/SHA-256.
4. Reads the physical little-endian GGUF v3 header: checks embedded source/range,
   decoder count excluding MTP, actual tensor names against the report, every
   advertised layer, and required first/final role tensors.
5. Publishes only those verified exports in the catalog, without local paths.

This is structural and byte verification, **not** numerical/model qualification.
The seed's export files must remain immutable while served. The receiver always
rehashes bytes, including cache hits, before returning a path.

For the current Qwen3.8-27B UD-Q5_K_M source, the upstream repository API identifies:

- Repository: `unsloth/Qwen3.8-27B-GGUF`
- Revision: `4ca720788d1e01f1bff70c033e0d0028fd02e502`
- Source SHA-256: `2de73110cb254cbf09b54b717578dadff12ef1194e7271527e68202f39ba4bfd`
- Source size: `19771509664` bytes
- Architecture: `qwen35`; decoder layers: **64** (stored block count includes MTP).

[Upstream pinned file metadata](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-GGUF/revision/4ca720788d1e01f1bff70c033e0d0028fd02e502?blobs=true).
The current smallest existing compatible export is `Qwen3.8-27B-middle-16-20.gguf`
(965,577,440 bytes). Use `Qwen3.8-27B-suffix-32-64-final.gguf` for a final stage.
Older reports declaring 65 decoder layers and truncated fixtures are deliberately
excluded. Do not relabel their sidecars to bypass provenance checks.

## Operator CLI

These commands are standalone diagnostics. Applications should attach the seed
and downloader to their existing Node in the same runtime process, as above;
never open the same identity directory in two simultaneous processes.
The CLI caps each seed spec, catalog, and manifest JSON file at 16 MiB; it
enforces the limit during the read as well as before reading the file.

For large models, use release builds (`cargo run --release`) so hashing is optimized.
Create a seed spec using absolute local paths, for example:

```json
{
  "model": {
    "revision": "4ca720788d1e01f1bff70c033e0d0028fd02e502",
    "source_sha256": "2de73110cb254cbf09b54b717578dadff12ef1194e7271527e68202f39ba4bfd",
    "source_bytes": 19771509664,
    "architecture": "qwen35",
    "layers": 64
  },
  "source": "/absolute/models/Qwen3.8-27B-UD-Q5_K_M.gguf",
  "shards": [
    "/absolute/models/Qwen3.8-27B-middle-16-20.gguf",
    "/absolute/models/Qwen3.8-27B-suffix-32-64-final.gguf"
  ]
}
```

```sh
# Verify without starting a server. SEED_ID is the seed's real mesh endpoint ID.
cargo run --bin llm-cluster -- artifacts catalog \
  --spec seed.json --seed-id "$SEED_ID" --output catalog.json

# Serve from an already enrolled identity (requires fresh output path).
cargo run --bin llm-cluster -- artifacts seed \
  --spec seed.json --state-dir /private/seed-state --catalog live-catalog.json

# Smallest candidate from a catalog, using the device's enrolled identity.
cargo run --bin llm-cluster -- artifacts fetch \
  --catalog live-catalog.json --state-dir /private/device-state \
  --cache /private/shard-cache --quota-bytes 12884901888 --retain

# After placement, request the exact range; no implicit model/range substitution.
cargo run --bin llm-cluster -- artifacts fetch \
  --catalog live-catalog.json --begin 32 --end 64 \
  --state-dir /private/device-state --cache /private/shard-cache \
  --quota-bytes 12884901888 --retain

# Reuse an already-exported local candidate; verify and atomically cache it.
cargo run --bin llm-cluster -- artifacts fetch \
  --catalog catalog.json --import /absolute/models/Qwen3.8-27B-middle-16-20.gguf \
  --cache /private/shard-cache --quota-bytes 12884901888
```

`fetch --manifest shard.json` accepts a single manifest instead of a catalog.
`fetch --catalog` without a range selects the smallest export; the caller remains
responsible for choosing a suitable profiling role and device memory budget.
Progress is JSON on stderr; the successful stdout record contains `status: staged`,
verified path/hash/range, and `inference_ready: false`. Ctrl-C preserves a partial;
rerunning the command resumes it. A local import copies into the managed cache;
it never renames or modifies the operator's original file.

## Transfer and failure behavior

- Native Iroh `layercake/artifact/1` shares the existing Node endpoint/accept loop.
  Both peers require trusted membership. Requests address advertised artifact
  hashes, never filesystem paths. At most two seed streams run concurrently.
  Leaving the mesh interrupts live transfers. No central lookup service is added.
- HTTPS supports whole-export range resume. Responses must have exact 206 bounds
  and identity encoding. A server returning 200 to a resume request causes a safe
  truncate/restart. Redirects are refused: provide the final HTTPS artifact URL.
- Cache paths derive only from validated SHA-256. A process-shared lock serializes
  transfers/eviction. The actual `.part` file length is the resume offset; the
  content hash names that partial across restarts and mirror changes. A final
  full-file hash catches damaged or mixed partial bytes. There is no blind trust
  in a completion marker, filename, or prior successful download.
- Quota includes completed and partial artifacts. Free disk space is checked with
  64 MiB headroom before writing. In-use retentions are never quota-evicted. Disk
  exhaustion during a transfer returns an error and leaves a resumable partial.
- Cancellation/network failure retains partial bytes. Size/hash failure never
  returns a staged path; a completed bad hash removes the partial for a clean
  retry. Sync file data, verify, rename atomically, then sync the parent directory
  on Unix. The cache must be private to the runtime, not writable by other users.
- Timeouts bound connection setup and stalled reads/writes; retry is caller-owned.
  There is no background retry loop, automatic deployment, mobile background
  policy, or UI download flow in this core change.

## Local validation (2026-10-25)

Fresh Rust verification passed for the full pinned source and these existing
exports; original files were read without modification:

| Export | Range | Bytes |
| --- | --- | ---: |
| `Qwen3.8-27B-middle-16-20.gguf` | 16..20 | 965577440 |
| `Qwen3.8-27B-middle-20-24.gguf` | 20..24 | 1165830880 |
| `Qwen3.8-27B-middle-24-28.gguf` | 24..28 | 1146620640 |
| `Qwen3.8-27B-middle-16-28.gguf` | 16..28 | 3256137856 |
| `Qwen3.8-27B-prefix-0-16.gguf` | 0..16 | 4886036224 |
| `Qwen3.8-27B-prefix-0-20.gguf` | 0..20 | 5840668096 |
| `Qwen3.8-27B-prefix-0-24.gguf` | 0..24 | 6995553408 |
| `Qwen3.8-27B-suffix-32-64-final.gguf` | 32..64 | 10173553344 |

The real 16..20 candidate import was cancelled after 68,157,440 bytes, resumed
in a new process, verified to
`a5106cff46da97261263548893270df5323b7b3e501b394c8db1b8f1df576100`,
and then reused by another process without a mesh identity or source connection.
Core tests exercise actual local Iroh endpoints, server/client membership gates,
partial resume, corruption, quota/retention, lock cancellation, HTTP range and
truncation behavior, GGUF role/report mismatches, and offline CLI cache reuse.
This does not establish mobile deployment, public-relay transfer performance,
model correctness, or inference readiness.
