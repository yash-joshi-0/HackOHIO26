# Layer balancing (M-28)

A balanced shard manifest lets the leader decide **at every prompt start** how many layers each
node serves, so a Raspberry Pi gets a handful of layers and a 48 GB workstation gets most of
them. It applies to the `shard` backend only ([shard-backend.md](shard-backend.md)); the
`llama` backend keeps its planner-driven `--tensor-split` ([model-layers.md](model-layers.md)).

## When it runs

Balancing runs only when a prompt starts:

- when a job reaches the front end (the user submitted a prompt, or the leader's readiness
  probe), and
- when a retry starts after a failover or another retryable error (M-29's retry loop).

Every run is fresh: probes go out again, free memory is measured again, and the plan is
recomputed. Nothing is cached between prompts, so a node whose memory filled up since the last
prompt gets fewer layers on the next one.

## Why any range can be chosen per session

Every balanced node holds the **full** model file rather than one exported range. The shard
runner already takes `BEGIN END` on its command line, and a fresh runner starts for every
session, so each session can load a different range. The private llama.cpp patch the runner is
built against (`shard-runtime.patch` in
[results/shard-runner-2026-10-24](../experiments/model-validation/results/shard-runner-2026-10-24))
loads selectively from a full GGUF: `create_tensor` marks every tensor of a layer outside
`[begin, end)` as `TENSOR_SKIP`, skips the token embedding when `begin > 0`, and skips the
output norm and head when `end < layers`. Its exported-range check only fires for files that
carry `meshthing.shard_begin`/`shard_end` keys, which a full GGUF does not. The runner loads with
`LLAMA_LOAD_MODE_NONE` (no mmap of the whole file). So only the chosen range's weights become
resident, and the full file costs disk space, not memory.

## Protocol

`shard-serve` gains a second handshake on the existing RPC bridge ALPN, documented in
[`src/shard/wire.rs`](../src/shard/wire.rs). The original `MSH1\n` preface is unchanged.

```text
client -> MSH2\n{"type":"probe"}\n
worker <- {"available_bytes":7996391424,"model_sha256":"2e5f…","layers":64,"begin":0,"end":64}\n   (stream ends)

client -> MSH2\n{"type":"session","begin":0,"end":9}\n
worker <- runner metadata line (begin 0, end 9, stamped with the worker's hashes), then the usual
          request/reply traffic
```

- `available_bytes` is the operating system's available memory (Linux `MemAvailable`, lowered
  to the cgroup's free memory under a cgroup limit; free plus reclaimable pages on macOS;
  `ullAvailPhys` on Windows), measured with the `sysinfo` crate at the moment of the probe and
  capped by `shard-serve --offer-bytes`. It is never the total RAM; 0 means unknown.
- `layers` is set only by a full-model worker (`shard-serve --layers N`); a fixed-range worker
  reports `null` and is skipped by balancing leaders.
- Probes never start a runner and never wait for the worker's session lock.
- A ranged session is checked before its runner starts: a full-model worker accepts any
  non-empty range within its `--layers`; a fixed-range worker accepts only its own range. The
  runner's metadata must then report that range (and, for a full-model worker, the model's layer
  count), exactly as before.
- Probe and session share one Iroh connection.
- A worker older than this handshake treats `MSH2\n` as a bad preface and resets the stream; the
  leader sees a failed probe and skips that worker for the attempt.

## Algorithm

[`src/shard/balance.rs`](../src/shard/balance.rs) holds the pure algorithm, `plan_layers`.

Inputs: for each node, in layout order with the final node (the leader) last, its **budget**
(free memory minus the safety reserve) and its **fixed overhead** (per-stage runner scratch,
plus the output norm and head for the final node); the memory cost of each layer (weights plus
the KV cache for the context); and the token embedding size, which whichever node owns layer 0
must also hold.

1. **Weights.** Each node's weight is `budget − overhead`, where non-final nodes also subtract
   the embedding, since any of them may end up owning layer 0. A node fits
   `max_layers = weight / layer_cost` layers. Each layer is charged the largest per-layer cost:
   exact for the uniform decoder layers of supported architectures, conservative otherwise.
2. **Checks.** The final node must fit at least one layer, and together the nodes must fit
   every layer; otherwise the plan fails with a clear error.
3. **Smooth weighted round robin.** The final node takes one layer up front (it must own the
   last one). Each remaining layer is handed out in turn: every node that still has room adds
   its weight to its `current`; the node with the largest `current` (the earlier node on a tie)
   takes the layer and subtracts the round's total weight. Counts end up proportional to the
   weights, and no node ever gets more than `max_layers`. A node that cannot fit one layer gets
   none.
4. **Embedding fix-up.** If every layer landed on the final node while some other node could
   hold one, one layer moves to the largest such node, so the final node never has to hold the
   embedding as well. If no other node can hold a layer, the final node is charged the
   embedding from the start.
5. **Layout.** The counts become contiguous half-open ranges in node order, skipping nodes with
   none, so the final node's range ends at the last layer.

The result is deterministic: the same inputs always give the same ranges.

### Worked example

A 64-layer Qwen2.5-32B Q4_K_M split over a Raspberry Pi (2 GB free), a laptop (8 GB free) and a
workstation leader (48 GB free). Per-layer cost is 300,000,000 bytes of weights plus
4,096 bytes × 512 tokens of KV = 302,097,152 bytes. Reserve 512 MiB and stage overhead 256 MiB
on every node; 400 MB of token embedding for the first stage and 400 MB of output head for the
final one.

| Node | Free | Budget (− reserve) | Overhead | Weight | Fits | Layers | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Raspberry Pi | 2,000,000,000 | 1,463,129,088 | 668,435,456 (scratch + embedding) | 794,693,632 | 2 | 1 | `0..1` |
| Laptop | 8,000,000,000 | 7,463,129,088 | 668,435,456 (scratch + embedding) | 6,794,693,632 | 22 | 8 | `1..9` |
| Workstation (final) | 48,000,000,000 | 47,463,129,088 | 668,435,456 (scratch + head) | 46,794,693,632 | 154 | 55 | `9..64` |

The weights are 1.5 % / 12.5 % / 86.0 % of the total, which is 0.9 / 8.0 / 55.1 layers of 64.
This three-node plan is what `plan_layers` computes (it is a unit test). Running it needs
chained remote stages, which arrive with ring mode (M-27); see [v1 limits](#v1-limits).

In v1 the leader picks the remote with the most free memory, the laptop, and the plan for
`[laptop, workstation]` is `0..8` / `8..64`. With only the Pi available it is `0..1` / `1..64`.

## Leader flow at each attempt

1. Order the candidates by M-29's failure record (workers that have not failed in this job
   first, then least recently failed) and probe all of them at once, each bounded by
   `PROBE_TIMEOUT` (20 s). A worker whose probe fails is recorded as failed and skipped for
   this attempt. A worker that answers with another model, another layer count or a fixed range
   is skipped with a warning.
2. Measure the leader's own available memory, capped by `balance.local.offer_bytes`.
3. Choose the remote (v1: at most one): among the workers that failed least recently, the one
   with the most free memory; earlier candidates win ties.
4. Plan `[remote, leader]` (or `[leader]` alone) with `plan_layers`. A remote that gets no
   layers is let go, and the leader runs the whole model.
5. Open the remote session with its chosen range on the probe's connection and spawn the local
   runner over the full model file with the final range. Each stage's metadata is checked
   against the **chosen** range, the manifest's layer count and model identity.
6. Resume the generation as M-29 does.

Errors:

- A plan that cannot fit is fatal (HTTP 502, not retried), unless a candidate's probe failed
  during this attempt; then it is retryable, since another attempt may reach that worker.
- A remote session that fails to open is retryable and blames that worker, so the next attempt
  prefers another candidate and recomputes the plan with its memory.

## Manifest

A balanced manifest replaces `stages` with `balance`; a manifest has exactly one of the two.

```json
{
  "runner": "/path/to/mesh-shard-runner",
  "model_sha256": "2e5f6daea180dbc59f65a40641e94d3973b5dbaa32b3c0acf54647fa874e519e",
  "layers": 64,
  "context": 512,
  "max_tokens": 64,
  "chat_template": "chatml",
  "balance": {
    "candidates": ["<laptop node id>", "<pi node id>"],
    "local": {
      "model": "/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf",
      "gpu_layers": 99,
      "budget_bytes": 21474836480
    },
    "layer_bytes": 300000000,
    "kv_bytes_per_token": 4096,
    "stage_overhead_bytes": 268435456,
    "embedding_bytes": 400000000,
    "output_bytes": 400000000,
    "reserve_bytes": 536870912
  }
}
```

| Field | Meaning |
| --- | --- |
| `candidates` | Node IDs of `shard-serve` workers started with the full model and `--layers`, in preference order. May be empty. |
| `local.model` | The leader's full model file. Hashed once at startup; it must equal `model_sha256`. |
| `local.gpu_layers`, `local.budget_bytes`, `local.env` | As for a local stage. |
| `local.offer_bytes` | Optional cap on the memory the leader offers its own stage. |
| `layer_bytes` | Weight bytes of one decoder layer. Required. |
| `kv_bytes_per_token` | KV bytes one layer needs per token of context (default 0). |
| `stage_overhead_bytes` | Fixed bytes per stage: runner scratch and compute buffers (default 256 MiB). |
| `embedding_bytes` | Token embedding bytes, held by the stage that owns layer 0 (default 0). |
| `output_bytes` | Output norm and head bytes, held by the final stage; count the embedding again when the model ties them (default 0). |
| `reserve_bytes` | Memory left free on every node (default 512 MiB). |

Unknown fields are refused, as everywhere in the manifest. The sizes come from the model file
(for example `gguf-dump`), not from measurements; measured inputs are future work.

Each candidate worker runs:

```sh
llm-cluster shard-serve \
  --runner /path/to/mesh-shard-runner \
  --model /path/to/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
  --layers 64 --context 512 --gpu-layers 99 --budget-bytes 11811160064 \
  --expected-sha256 2e5f6daea180dbc59f65a40641e94d3973b5dbaa32b3c0acf54647fa874e519e \
  --allow-peer <leader node id> [--offer-bytes 6000000000]
```

`--range` is optional with `--layers` and defaults to the whole model; it is the range served
to leaders that use the original preface.

## v1 limits

- **Topology.** The shard backend runs at most one remote prefix plus the leader's final stage
  (no hub forwarding). So in v1 balancing chooses which remote candidate serves the prefix and
  where to cut. `plan_layers` is general for N nodes; chaining three or more nodes needs ring
  forwarding (M-27).
- **Free memory is a snapshot.** It is read at probe time; nothing reserves it, so another
  process can take it before the runner loads. The reserve is the safety margin.
- **Sizes are declared.** Per-layer, embedding, head and overhead sizes come from the manifest.
  Layers are charged at their largest cost, which overestimates models with uneven layers
  (for example Qwen hybrids or MTP blocks).
- **Memory only.** Compute speed and link cost are not weighed; the planner in
  [`src/placement.rs`](../src/placement.rs) does that for the `llama` backend.
- **Full model everywhere.** Every candidate and the leader need the full model file on disk.
- **The runner budget is fixed.** `--budget-bytes` and `local.budget_bytes` do not scale with
  the chosen range.

## Tests

- `src/shard/balance.rs`: proportional shares (the worked example), caps never exceeded, a node
  that cannot fit a layer gets none, determinism, insufficient memory, a single node, the final
  node last, the embedding fix-up, uneven layers.
- `src/shard/wire.rs`: golden bytes and round trips for probes, ranged sessions and replies.
- `tests/shard_serve.rs`: a full-model worker answers probes capped by its offer, serves any
  requested range and refuses ranges past the model; a fixed-range worker refuses other ranges.
- `tests/shard_frontend.rs`: two real full-model workers report different free memory and the
  cut follows the weights, a later prompt replans when a worker leaves, a retry replans without
  the worker that failed, everything runs locally when no worker answers, a plan that cannot
  fit fails without retries, and a leader whose full model is not the manifest's model refuses
  to start.
