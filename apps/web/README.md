# Layercake web UI

The Rust runtime owns membership, heartbeats, the queued OpenAI-compatible API,
and this SvelteKit + Tailwind dashboard in one process. Node/npm are build tools;
no Node server or mesh subprocess runs in production.

From the repository root:

```sh
npm --prefix apps/web ci
npm --prefix apps/web run build
cargo build --bin llm-cluster
./mesh-ui
```

The runtime opens `http://127.0.0.1:3210`. Use `--no-open` to skip opening the
browser, `--port 0` for a free port, and `--state-dir PATH` for an isolated mesh.
Ctrl+C stops the runtime and its UI. A fresh profile shows Create / Join.
`mesh-ui` execs `llm-cluster ui`; the packaged `meshthing` executable calls the
same Rust library directly.

## API and inference status

The same listener exposes the OpenAI base URL `http://127.0.0.1:3210/v1`,
`POST /v1/chat/completions`, `GET /v1/models`, and `GET /jobs`. The terminal and
device cards display the bound address, including the chosen port. URLs are
local to the device that owns them; a peer's loopback URL is not a remote route.

Attach an existing inference backend with `--upstream 127.0.0.1:PORT`.
Chat requests use the core's FIFO queue and streaming relay. Without membership
and a responding backend, inference returns HTTP 503 with an OpenAI-style error.
The optional top-level `conversation_id` is removed before forwarding and is
limited to 256 UTF-8 bytes because recent job records retain it.
A listening API does not mean a loaded model or an executable distributed plan.
Readiness checks backend `/health`; model IDs come from its `/v1/models`.
The API does not launch workers, choose a model, or download weights.

Authenticated member sync includes API metadata. The mobile apps show the desktop
endpoint on its card, labeled **On that device only**. Devices with no listener
show API not running (self) or API not reported (remote). Missing measurements,
layer assignments and native mobile inference are not invented.

The same listener retains `GET /peers` for authenticated control connections,
with connection/heartbeat timestamps. Dashboard telemetry reads that shared table
directly; membership, connection status and inference readiness are distinct.

## Membership and QR

Core-owned create, signed full invitations, camera scanning, trusted approval,
and Leave mesh work over Iroh without a Layercake lookup server. QR decorations
stay outside the actual code and quiet zone. Scanning fills the invitation for
review; the user requests to join and a trusted member approves.

Membership, private identity and preferences live under
`~/.local/state/layercake/mesh` (`--state-dir` or `LAYERCAKE_MESH_STATE_DIR`).
Leaving disconnects this node and clears its saved membership, retaining device
identity. Other nodes retain their signed trust history. Rejoining needs a new
invitation and approval. Use one runtime per state directory.

## Development and checks

```sh
cargo build --bins --examples
npm --prefix apps/web run check
npm --prefix apps/web test
npm --prefix apps/web run build
npm --prefix apps/web run test:integration
npm --prefix apps/web run test:mesh
```

Integration tests use temporary state and a synthetic backend to verify the same
PID owns the UI/API, serialized requests, streaming, readiness, persistence,
origin checks and shutdown. `test:mesh` uses public Iroh connectivity to verify
real invitation/trust, peer API metadata, leave and restart behavior.

For frontend development only, `npm --prefix apps/web run dev` proxies `/api`
and `/v1` to a runtime on port 3210. For the complete app, use `./mesh-ui`.

The static build follows [SvelteKit's static adapter](https://svelte.dev/docs/kit/adapter-static).
The [Cake JE4 render](https://minecraft.wiki/images/Cake_JE4.png?7c59e) was supplied
for Layercake branding.
