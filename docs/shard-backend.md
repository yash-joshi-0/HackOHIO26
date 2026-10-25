# Shard backend

The leader can run jobs on one of two execution backends:

- `llama` (the default): a distributed `llama-server` over ggml RPC bridges, launched from a
  placement plan (`llm-cluster plan`).
- `shard`: david-codex's v1 shard runner (`experiments/model-validation/shard_runner.cpp`,
  unchanged). Each machine loads only its own layer range of the model, so a model larger than
  any one device's memory can run. This is how Qwen2.5-32B Q4_K_M ran across two 16 GiB Macs
  (see [SHARD_RUNNER.md](../experiments/model-validation/SHARD_RUNNER.md)).

The job API, `GET /jobs`, the browser client, `llm-cluster prompt`, history and shutdown are the
same on both backends. With `--backend shard` the leader runs an OpenAI-shaped front end
in-process on its internal loopback port (`--llama-port`, default 8081), and the job API
forwards to it just as it forwards to `llama-server`.

## How a job runs

1. The job API takes a job off its queue and forwards the chat request to the front end.
2. The front end renders the messages with the manifest's chat template and opens fresh stage
   links: it dials the remote prefix worker by node ID over Iroh (RPC bridge ALPN) and spawns
   the local final stage's runner. A runner serves exactly one sequence, so every job (and the
   readiness probe) gets new processes.
3. Token IDs go to the prefix, its hidden states come back to the leader, the local final stage
   samples the next token greedily, and the token is fed back to the prefix. Pieces stream back
   to the job API as OpenAI-style SSE.
4. If a stage fails in a way a fresh attempt might fix, the front end resumes the generation on
   new stage links (see [Retries and standby workers](#retries-and-standby-workers)).

## Retries and standby workers

A runner holds one sequence whose cache only grows and cannot be reset, and it insists on
strictly sequential steps and positions, so a broken generation cannot be repaired in place.
Instead the front end keeps the generation's progress outside the stage links (the prompt
tokens, every token generated so far, the text already sent, including a UTF-8 character
that is only partly received) and resumes it on fresh links:

1. It opens new links: local stages get new runner processes, and a remote stage dials its
   next usable worker (below).
2. The new links are checked exactly as on the first attempt. The new final stage must
   tokenize the prompt exactly as the first one did; a difference is fatal.
3. The prompt plus every generated token is prefilled again, in batches of at most 128,
   from step and position 0, and decoding continues from the final reply. No text is sent
   twice, so a stream simply carries on.

Limits count totals over every attempt: `max_tokens` is the total number of completion tokens,
and the context counts every position. A resume that no longer fits finishes with
`finish_reason: "length"` rather than failing. A run without failures behaves exactly as
before.

Errors fall into two classes:

| Class | Errors | What happens |
| --- | --- | --- |
| Retryable | Stage I/O errors, a stream or connection that closed, a step timeout (120 s), a runner that exited, a failed dial, a metadata timeout. | Retried after 250 ms, then 500 ms: at most 2 retries (3 attempts), and no retry starts once 15 minutes have passed since the job began. Each retry is logged as a warning with the attempt number, stage, worker and error. |
| Fatal | A prompt longer than the context (HTTP 400), metadata that does not match the manifest (layer range, model identity, hidden size, final-ness), a prompt tokenized differently on a retry, a sampled token outside the vocabulary, a wrong number of hidden values, any other protocol violation. | Not retried. |

When retries run out, or on a fatal error, the job fails as before: a JSON 5xx (or 400) if
nothing was sent yet, or an SSE `error` event with no `[DONE]` mid-stream, so the job API
fails the job without committing history. A client that disconnects still drops its
generation at once, with no retry.

A remote stage may list standby workers:

```json
{"begin": 0, "end": 32, "remote": {"worker": "<primary node id>", "standby": ["<standby node id>"]}}
```

Each attempt tries the stage's workers in order, primary first, skipping any worker that has
already failed during this job. A worker whose dial or metadata read fails is skipped at once
and the next one is tried within the same attempt; a worker that fails mid-generation is not
dialled again for that job. Once every worker of a stage has failed, the job fails. A remote
stage with no standby therefore gets no retry when its worker fails; only failures of local
stages (or of a stage whose standby remains) are retried.

Limits:

- Each retry pays the cost of re-prefilling the prompt and everything generated so far, plus
  the local runner's model load.
- A standby must already run `shard-serve` with the identical stage artifact (same range, same
  model identity) and allowlist the leader. Nothing is copied or loaded on demand.
- Replanning a stage onto surviving peers is M-28; ring mode (M-27) is not covered.

The frontend buffers at most 16 text events for a slow HTTP reader; generation applies
backpressure instead of accumulating an unbounded queue. A disconnected client signals
cooperative cancellation. Requests still waiting for the generation slot exit without starting
workers; an active pipeline closes remote streams and waits for local runners to be killed and
reaped before releasing the slot. If disconnect happens during stage startup, already-started
links are closed as soon as startup finishes.

## Commands

On the worker that serves the prefix:

```sh
llm-cluster shard-serve \
  --runner /path/to/mesh-shard-runner \
  --model /path/to/prefix-0-32.gguf \
  --range 0:32 --context 512 --gpu-layers 99 --budget-bytes 11811160064 \
  --expected-sha256 <sha256 of the --model file> \
  --source-sha256 <sha256 of the original full model> \
  --allow-peer <leader node id>
```

It prints `node id: <id>`; put that ID in the leader's manifest. `--allow-peer` is required and
repeatable: other peers are refused. `--source-sha256` is for an exported shard, which reports
the original model's identity; leave it out when `--model` is the full model file.

On the leader:

```sh
llm-cluster leader --backend shard --shard-manifest m.json
```

`--plan`, `--profile`, `--llama-server`, `--llama-arg`, `--min-peers`, `--evidence-file`,
`--evidence-wait` and `--llama-ready-timeout` belong to the llama backend and are refused with
`--backend shard`. `--api`, `--llama-port` and `--worker-wait` work on both. Use the global
`--key` flag to keep the leader's identity stable; `llm-cluster --key leader.key listen` prints
the node ID that key gives (stop it with Ctrl-C), which the worker needs for `--allow-peer`.

## Startup

1. The leader loads and validates the manifest before binding anything. An invalid manifest
   (for example two remote stages) fails here with a message naming the rule.
2. It starts the front end and waits until it listens. The front end first hashes every local
   stage file that has an `artifact_sha256` and refuses to start on a mismatch.
3. `shard-serve` workers open no control link to the leader, so there are no worker hellos to
   wait for. Instead the leader sends a one-token readiness probe through the front end and
   retries it with a short backoff (100 ms doubling to 2 s) until it passes or `--worker-wait`
   elapses (default 60 s). The final error carries the last probe failure, which names the
   failing stage, for example `remote stage on <worker>: ... peer not approved`.
4. Only then does the job API start serving.

Post-run evidence verification (`--evidence-file`) checks a run against its planning profile,
so it exists only on the llama backend.

## Manifest

```json
{
  "runner": "/path/to/mesh-shard-runner",
  "model_sha256": "2e5f6daea180dbc59f65a40641e94d3973b5dbaa32b3c0acf54647fa874e519e",
  "layers": 64,
  "context": 512,
  "max_tokens": 64,
  "chat_template": "chatml",
  "stages": [
    {"begin": 0, "end": 32, "remote": {"worker": "<worker node id>"}},
    {"begin": 32, "end": 64, "local": {
      "model": "/path/to/suffix-32-64.gguf",
      "gpu_layers": 99,
      "budget_bytes": 11811160064,
      "artifact_sha256": "91e7c4513c290df2d9546f29250a91db28cba2163c7c8ab812dd65cccef06513"
    }}
  ]
}
```

| Field | Meaning |
| --- | --- |
| `runner` | The `shard_runner` binary the leader runs local stages with. |
| `model_sha256` | SHA-256 of the original, unsharded model (64 hex digits). Every stage must prove it serves a slice of this model: a remote stage through the hash `shard-serve` announces, a local stage through the `source_sha256` its runner reports or a verified `artifact_sha256`. |
| `layers` | Total layers in the model. |
| `context` | Context window passed to every stage, 32 to 4096. A prompt longer than this is refused with HTTP 400. |
| `max_tokens` | Cap on completion tokens (default 512); a request may ask for fewer. |
| `chat_template` | Required: `"chatml"` or a pinned Jinja template object; the template is never guessed. See [Chat templates](#chat-templates). |
| `stages` | Layer ranges in order, half-open, covering `0..layers` with no gaps. Each has `begin`, `end` and exactly one of `remote` or `local`. |
| `remote.worker` | Node ID of the `shard-serve` worker for this range. The leader dials it by ID: through Iroh's n0 discovery in production, or in-memory address lookup in tests. |
| `remote.standby` | Optional list of node IDs of workers serving the identical stage, tried in order when the primary fails (see [Retries and standby workers](#retries-and-standby-workers)). No node ID may appear twice anywhere in the manifest, as primary or standby. |
| `local.model` | Model or exported shard file the local runner loads. |
| `local.gpu_layers` | Layers offloaded to the GPU (default 0). |
| `local.budget_bytes` | Runtime buffer budget passed to the runner; must be positive. |
| `local.artifact_sha256` | Optional SHA-256 of `local.model`, checked once at startup. It establishes the stage's identity when the runner reports no `source_sha256` (a plain GGUF). |
| `local.env` | Optional extra environment variables for the runner, such as `CUDA_VISIBLE_DEVICES`. |

Unknown fields are refused everywhere, so a typo cannot silently drop a setting.

### Chat templates

The front end renders a request's `messages` into the prompt the tokenizing stage tokenizes
(the local final stage, or stage 0 in a ring). `chat_template` takes one of two forms.

`"chatml"` writes `<|im_start|>{role}\n{content}<|im_end|>\n` for each message, then
`<|im_start|>assistant\n`.

A pinned Jinja template renders the model's official chat template:

```json
"chat_template": {
  "jinja": {"path": "qwen3-chat-template.jinja", "sha256": "<64 hex digits: SHA-256 of the file>"},
  "enable_thinking": false
}
```

| Field | Meaning |
| --- | --- |
| `jinja.path` | The template file, for example a GGUF's `tokenizer.chat_template` metadata saved as text. A relative path is resolved against the manifest file's directory. (A manifest built in memory rather than loaded from a file keeps it relative to the working directory, as model paths are.) |
| `jinja.sha256` | SHA-256 of the file's bytes (64 hex digits). |
| `enable_thinking` | Optional, default `false`. Passed to the template as `enable_thinking`. |

When the front end starts, it reads at most 1 MiB of the file once, checks its SHA-256 and
compiles it. A missing or oversized file, a hash mismatch or a template that does not compile
stops startup, as an `artifact_sha256` mismatch does. Each request then renders the template with
[minijinja](https://docs.rs/minijinja) set up the way Hugging Face's `apply_chat_template` and
david-codex's demo (`experiments/model-validation/live_demo.py`) set up Jinja:

- `trim_blocks` and `lstrip_blocks` are on.
- Python string methods such as `.strip()`, `.startswith()` and `.split()` work, through
  `minijinja-contrib`'s pycompat.
- The context is `messages` (each with `role` and `content`; text parts are joined),
  `add_generation_prompt = true`, `enable_thinking` from the manifest, `tools = none`, and a
  `raise_exception(msg)` function.

Requests are validated the same way as with `chatml`. A template error, including a
`raise_exception` call, answers HTTP 400 with the template's message; it never reaches a stage.

The pinned template is what david-codex's experimental ring demo uses, and he asked for it in
place of generic ChatML. His demo renders the GGUF's official Qwen template with
`add_generation_prompt=True, enable_thinking=False, tools=None`. With `enable_thinking: false`,
a system message `You are a helpful assistant.` and a user message `<q>` render exactly as in
his run record (`experiments/model-validation/results/qwen38-three-device-2026-10-24/manifest.json`):

```text
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<q><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

With `enable_thinking: true`, the official Qwen3 template leaves out the empty
`<think>\n\n</think>\n\n` block and the prompt ends at `<|im_start|>assistant\n`.
`tests/fixtures/qwen3-chat-template.jinja` is the official Qwen3 template, copied byte for byte
from `Qwen/Qwen3-8B`'s `tokenizer_config.json` (SHA-256
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`). For a real model, pin the
template embedded in that model's own GGUF.

## v1 limits

- Topology: one remote prefix plus a local final stage, or a single local stage that runs the
  whole model. Chained remote stages and a remote final stage need ring forwarding: see
  [Experimental: ring execution](#experimental-ring-execution-shard-ring1) below. Hidden states
  never take an extra hop through the leader ("hub forwarding"); that design was rejected.
- Greedy decoding only. `temperature`, `top_p`, `seed` and similar fields are accepted and
  ignored.
- Chat templates: `chatml` or a pinned Jinja template (see [Chat templates](#chat-templates)).
  `tools` is always `none`; tool calls are not supported.
- One sequence per runner process and no reset, so every job starts fresh runner processes
  (and a fresh Iroh connection to the worker). Model load time is paid on every job, and again
  on every retry.
- One generation runs at a time, matching the job queue.

## Experimental: ring execution (shard-ring/1)

Ring execution runs chained remote stages and a remote final stage without sending hidden
states back through the leader. It is opt-in (`"execution": "ring"` in the manifest) and
EXPERIMENTAL: it is pending david-codex's production sign-off. Manifests without `execution`,
or with `"execution": "pipeline"`, behave exactly as described above.

### Wire compatibility

The leader and `llm-cluster shard-ring-worker` speak `meshthing/shard-ring/1` byte for byte as
david-codex's diagnostic `experiments/mobile-probe/iroh-core/src/ring.rs` does (see
[QWEN38_RING.md](../experiments/model-validation/QWEN38_RING.md)): the same ALPN, one bounded
JSON line per control message (at most 65,536 bytes), raw little-endian `f32` payloads of at
most 16 MiB, a 32-hex job id consumed once, strict step and position order, batches of 1 to
128 rows, and the same chain checks (contiguous ranges; the same layers, hidden size,
vocabulary and 64-hex `source_sha256` throughout). Our leader and workers are designed to mix
with his gateway and `mesh-ring worker` in one ring; that mix has not been run yet.

### Topology

- The leader is the ring's origin. It renders the prompt, runs stage 0 (`0..k`) in a local
  runner that tokenizes it, and sends stage 0's hidden rows to the entry worker.
- Each worker runs its own range in a fresh runner per session and forwards its rows straight
  to the next worker. Only small progress lines travel back toward the leader, hop by hop.
- The terminal worker (the one owning the last layer) opens a result channel straight back to
  the leader and returns each step's token and text piece on it.
- The leader never receives activations. Every byte it reads from ring streams is counted, and
  the tests check that all of it was JSON control lines.

For each generation the leader registers a fresh job id for the manifest's terminal, starts
the session at the entry, checks the `ready` chain against the manifest (ranges and workers
in order, `source_sha256` equal to `model_sha256`, and layers, hidden size and vocabulary
matching stage 0), and waits for the terminal's result channel. Its endpoint also accepts the
ring ALPN: result channels are accepted only from the manifest's terminal and only for a
pending job, and any other peer is closed with code 3 ("peer not approved").

### Worker commands

Each hop names its neighbours by node ID: `--upstream` is the only peer allowed to start
sessions, `--origin` is the leader, and `--next` is the following worker. A worker without
`--next` is the terminal. Print the leader's node ID first with
`llm-cluster --key leader.key listen`.

Entry worker (layers `16..40`, upstream is the leader):

```sh
llm-cluster --key entry.key shard-ring-worker \
  --runner /path/to/mesh-shard-runner --model /path/to/shard-16-40.gguf \
  --range 16:40 --context 512 --gpu-layers 99 --budget-bytes 11811160064 \
  --expected-sha256 <sha256 of the --model file> \
  --source-sha256 <sha256 of the original full model> \
  --upstream <leader node id> --origin <leader node id> --next <terminal node id>
```

Terminal worker (layers `40..64`, upstream is the entry worker, no `--next`):

```sh
llm-cluster --key terminal.key shard-ring-worker \
  --runner /path/to/mesh-shard-runner --model /path/to/shard-40-64.gguf \
  --range 40:64 --context 512 --gpu-layers 99 --budget-bytes 11811160064 \
  --expected-sha256 <sha256 of the --model file> \
  --source-sha256 <sha256 of the original full model> \
  --upstream <entry node id> --origin <leader node id>
```

A middle worker takes `--upstream <previous worker>` and `--next <following worker>`. Every
ring worker must load an exported shard whose runner reports the 64-hex `source_sha256`, as in
ring.rs. Then start the leader as usual:
`llm-cluster --key leader.key leader --backend shard --shard-manifest ring.json --worker-wait 5m`.

### Manifest

```json
{
  "runner": "/path/to/mesh-shard-runner",
  "model_sha256": "<sha256 of the original full model>",
  "layers": 64,
  "context": 512,
  "max_tokens": 64,
  "chat_template": "chatml",
  "execution": "ring",
  "stages": [
    {"begin": 0, "end": 16, "local": {
      "model": "/path/to/shard-0-16.gguf", "gpu_layers": 99, "budget_bytes": 11811160064
    }},
    {"begin": 16, "end": 40, "remote": {"worker": "<entry node id>"}},
    {"begin": 40, "end": 64, "remote": {"worker": "<terminal node id>"}}
  ]
}
```

Ring rules: `stages[0]` is local, begins at layer 0 and ends before the last layer (it
tokenizes and runs the first layers); every later stage is remote, there is at least one,
they are contiguous up to `layers`, and no worker appears twice. The last remote stage is the
terminal. All the other manifest rules (hashes, context, `max_tokens`, `chat_template`,
unknown fields) apply unchanged.

### Diagnostics

With the global `--debug` flag (or `RUST_LOG=llm_cluster::shard=debug`) the leader logs the
origin side of each ring session, following the conventions in
[runtime-diagnostics.md](runtime-diagnostics.md): `ring session ready` (the entry's and the
terminal's selected Iroh path, and `ready_us` from the start to the result channel), then for
every step `ring activations sent` (`input_bytes`, `write_us`) and `shard ring exchange`
(`input_bytes`, `output_bytes`, `write_us`, `read_us` until the terminal's result arrived,
`total_us`), plus the whole-step `shard pipeline step`. Each worker's relayed progress lines
are logged as `ring progress`. A timeout error says whether sending the activations or waiting
for the result ran out of the step deadline. At `info`, `ring session closed` reports the
session's step count, activation bytes sent and bytes read (with `activation_bytes_read`,
always 0). `read_us` includes every worker's compute time, so it is not a network latency.

### Limits

- Fixed order: the manifest's ranges and workers must match the running ring exactly. There is
  no automatic placement yet.
- One job per worker: a worker runs one session at a time and drops a concurrent one, and the
  leader runs one generation at a time.
- No retries, rerouting or re-prefill: any error (a bad chain, a missing result channel, a
  worker failure) fails the job. A ring must become ready, including the terminal's result
  channel, within 300 s by default (`RingSettings::result_channel_wait`).
- Chat templates work as in pipeline mode: the leader renders the prompt for its local stage
  0. To match david-codex's ring demo, use the pinned official Qwen template with
  `enable_thinking: false` (see [Chat templates](#chat-templates)).
- The manual multi-device run of ring execution through `llm-cluster prompt` is still pending
  on hardware. It has only been tested on loopback with the fake runner.

## Tests

`tests/shard_ring.rs` runs the job API end to end on a three-stage ring: the leader's local
stage `0..1`, then two real `serve_ring_worker`s (entry `1..3`, terminal `3..4`) on the
`fake_shard_runner` example. It checks buffered and streamed prompts, that no activation byte
reached the leader, a `ready` chain that does not match the manifest, workers in the wrong
order, a result channel from a peer that is not the terminal (and a replayed job), and a
terminal that never opens its result channel, which fails within the bounded wait.

`tests/shard_backend.rs` runs the job API end to end on the shard backend: a real
`shard-serve` worker drives the `fake_shard_runner` example as the prefix, and the leader runs
the local final stage. It covers buffered and streamed prompts, `GET /jobs`, fresh runners per
job, a worker that does not allowlist the leader, and an invalid manifest.
`tests/shard_frontend.rs` covers retries and standbys against the front end directly: a stage
that fails once (`FAKE_SHARD_FAIL_AT_STEP` with `FAKE_SHARD_FAIL_ONCE_FILE`) is resumed with
the same text and token count as an uninterrupted run, an unreachable or failing primary hands
over to its standby, exhausted retries end the job, and fatal errors are not retried. Build the
helpers first:

```sh
cargo build --examples
cargo test --test shard_backend
cargo test --test shard_ring
```

## Reproducing the two-Mac 32B run through `llm-cluster prompt`

This manual acceptance step is still pending: it has not yet been run on the hardware. The
values below come from david-codex's capacity run
([results](../experiments/model-validation/results/shard-runner-2026-10-24)).

- Model: [bartowski/Qwen2.5-32B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF)
  Q4_K_M, SHA-256 `2e5f6daea180dbc59f65a40641e94d3973b5dbaa32b3c0acf54647fa874e519e`, 64 layers.
- Shards exported with `export_prefix.py`: the prefix `0..32` (9,828,263,360 bytes, SHA-256
  `8455d69052c1fc055c9a1c1f48f20451c4fc6f1d725f7bf9f43b0a60f66646f6`) on the M1 Air, and the
  suffix `32..64` (10,029,008,480 bytes, SHA-256
  `91e7c4513c290df2d9546f29250a91db28cba2163c7c8ab812dd65cccef06513`) on the M2 Pro.
- Both Macs need `mesh-shard-runner` built as in SHARD_RUNNER.md. Pause the normal demo during
  the run to free memory.

1. On the M2 Pro (leader), print the leader's node ID: `llm-cluster --key leader.key listen`,
   then Ctrl-C.
2. On the M1 Air (worker):

   ```sh
   llm-cluster --key worker.key shard-serve \
     --runner "$PWD/shard-runtime-v3/mesh-shard-runner" \
     --model "$PWD/models/Qwen2.5-32B-prefix-0-32.gguf" \
     --range 0:32 --context 512 --gpu-layers 99 --budget-bytes 11811160064 \
     --expected-sha256 8455d69052c1fc055c9a1c1f48f20451c4fc6f1d725f7bf9f43b0a60f66646f6 \
     --source-sha256 2e5f6daea180dbc59f65a40641e94d3973b5dbaa32b3c0acf54647fa874e519e \
     --allow-peer <leader node id>
   ```

3. On the M2 Pro, write the manifest above as `qwen32.json`, with the M1's node ID as
   `remote.worker`, the M2's runner path as `runner` and its suffix file as `local.model`.
   The 11,811,160,064-byte budget is the one the prefix used; the suffix reported
   10,177,193,984 runtime buffer bytes, so it fits. Then start the leader, allowing time for
   the cold load in the first probe:

   ```sh
   llm-cluster --key leader.key leader --backend shard --shard-manifest qwen32.json --worker-wait 5m
   ```

4. Once it logs `leader ready; serving the job API`:

   ```sh
   llm-cluster prompt "Explain why splitting one language model across two computers allows a larger model to run. Answer in two short sentences."
   ```

Expected: a coherent two-sentence answer, with each Mac's runner loading only its own shard.
David's run reported about 9.96 GB of runtime buffers on the prefix and 10.18 GB on the suffix,
and 2.33 output steps per second after prefill. `llm-cluster prompt` sends a single user
message, while David's run also had the system message `You are a helpful assistant.`, so the
text need not match his byte for byte. To replay his exact prompt, send both messages to the
job API:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain why splitting one language model across two computers allows a larger model to run. Answer in two short sentences."}
  ]
}'
```
