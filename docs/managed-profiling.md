# Managed profiling in the Rust runtime

The core now supplies a shared admission owner and managed start/status/cancel
controls for the [native candidate profiler](candidate-profiling.md). This is
process-local integration: the gate covers native work launched by this Rust
process, not every process or user on the physical device.

## What shares admission

`ComputeAdmission::process()` is the default owner. These core entry points use
`AdmittedChild` and carry inference leases through native process exit/reaping:

- `process::spawn_supervised`: the persistent llama-server/RPC-server supervisor,
  including startup, failures, restarts and shutdown.
- `shard::pipeline::StageLink::spawn_local`: the local native shard stage.
- `shard::serve`: each admitted native worker session.

Inference leases preserve existing inference concurrency. Profiling acquires an
exclusive lease only when no managed native process holds an inference lease.
Busy profiling requests are rejected rather than queued ahead of inference.
New inference arriving during an admitted profile waits until cleanup finishes.
The HTTP JobQueue remains responsible for FIFO chat scheduling; it is not the
memory-lifetime boundary for profiling.

A persistent backend continues to hold its lease while its model remains loaded,
even between chat requests. Profiling therefore stays busy until that backend is
unloaded/stopped. The shard backend uses fresh processes per request and can admit
profiles between requests. Unloading/reloading a persistent backend automatically
is not implemented here.

Dropping an inference future initiates native termination and reaping while the
lease stays owned by cleanup. If that cleanup task itself is aborted or cannot
confirm process exit asynchronously, teardown attempts a bounded synchronous
reap (up to 100 ms). The admission owner becomes fenced if exit remains
unconfirmed; elapsed time alone never releases capacity. A profile helper has
the same fail-closed behavior: an exit code alone does not prove that its native
child stopped.

## Configure the existing loopback API

The existing `serve_api` optionally reads `LAYERCAKE_PROFILE_CONFIG` at startup.
When unset, profile control routes are absent. Set it to a trusted local JSON
configuration; all paths in the configuration must be absolute:

```json
{
  "python": "/absolute/path/to/python3",
  "script": "/absolute/path/to/profile_candidate.py",
  "script_sha256": "SHA256_OF_THE_UPDATED_HELPER",
  "output_root": "/absolute/path/to/private-profile-results",
  "candidates": {
    "candidate-a": "/absolute/path/to/candidate-request.json"
  },
  "max_run_seconds": 600,
  "cleanup_grace_seconds": 10
}
```

The request file uses the standalone profiler schema. Generate its fixture from
valid state history, use actual artifact hashes, and set limits from the owner's
contribution budget. The configured script's hash is verified at manager creation
and again before every launch. The interpreter, candidate registry and filesystem
are trusted operator inputs; this is not a remote script-execution interface.

For a single-process application host, construct `ProfileManager::new(config)`
and use `api::serve_api_with_profiles(..., Some(manager), shutdown)`, or merge
`manager.router()` into the existing **trusted loopback** control listener. If
embedding the router directly, the host must call `manager.shutdown().await`
before tearing down its async runtime. The manager's last-handle drop requests
cancellation, but cannot synchronously wait for native cleanup.

The manager must share the same admission owner as its actual native backend.
`with_admission` supports explicit ownership in an embedding or isolated tests;
creating a separate owner for profiling and inference defeats exclusion.

## Control contract

| Request | Behavior |
| --- | --- |
| `POST /profiles` with `{"candidate":"candidate-a","compute_enabled":true}` | Returns 202 and a run ID after acquiring idle admission. Only configured names are accepted. |
| `GET /profiles/{run_id}` | Returns lifecycle state, diagnostic outcome/error, report directory and cleanup confirmation. |
| `DELETE /profiles/{run_id}` | Requests cooperative cancellation; repeated cancellation is harmless. |

Missing compute opt-in returns 403, an unknown candidate/run returns 404, busy
managed compute returns 409, and a fenced or shutting-down owner returns 503.
The diagnostic executor is POSIX-only and returns 501 on unsupported hosts.
Unknown start fields are rejected, and request bodies are limited to 1 KiB.
HTTP callers cannot submit executable paths, argv, output paths or fixture files.

`compute_enabled` is an explicit local operator action for this diagnostic API;
the app must connect it to its owner's compute preference. Mesh membership alone
must not trigger the endpoint. Routing this loopback management API to untrusted
peers requires a separately reviewed authorization layer.

States are `running`, `cancelling`, `finished` and `fenced`. A numerical/reference
failure can finish cleanly and return capacity while retaining a failed outcome.
It never makes the candidate eligible. Up to 64 recent run records are retained
in memory; each run's files remain on disk under a unique UUID directory. Disk
retention is operator-owned and not automatically pruned.

## Cleanup and recovery boundary

The manager supplies a unique `--run-id` and `--cancel-file` to the pinned helper.
The updated helper reports `cleanup_confirmed`, `children_started` and
`children_reaped` in its final report. Successful admission release requires:

- Matching run ID, report kind/schema and a recognized terminal outcome.
- Native child counts present and equal, and explicit cleanup confirmation.
- `eligible_for_placement` remaining false and helper exit status matching its
  reported success/failure outcome.

An older helper without this confirmation cannot satisfy managed cleanup. The
manager never treats a partial `running` report as a final result.

On cancellation or the manager's execution deadline, the manager writes the
cancellation marker and waits the configured cleanup grace. The helper normally
terminates and reaps its native child, preserves partial results, and exits. If
the marker cannot be written, the helper ignores cancellation, or final evidence
is missing/mismatched, compute is fenced. Forced helper termination is **not**
assumed to kill a native descendant in its own process group.

A fence blocks new managed inference and profiling. There is intentionally no
HTTP clear-fence action. Inspect retained logs and owned processes, stop any
remaining native work, then restart the runtime owner. Restart alone is not proof
that a native process from the previous owner has exited. The filesystem reports
are retained, but the manager does not yet restore its run registry across an
owner crash; startup reconciliation is a separate gate before unattended use.

## Integration still required

- Yash's M-27 ring spawn/reap paths must use the same owner or `AdmittedChild`.
  This patch changes no ALPN dispatch, accept loop or ring wire protocol.
- Standalone workers/diagnostics in other processes do not share this gate.
  A worker's profile controller must live under the same runtime owner; a separate
  profile CLI process cannot lock it through this in-memory gate.
- Mobile in-process engines need leases covering actual native context lifetime,
  suspension/cancellation callbacks and device power/thermal policies. The Python
  subprocess helper is not a mobile embedding solution.
- Complete RAM/VRAM/shared accounting, effective backend identity, thermal
  monitoring, current availability and matching Iroh link measurements remain
  required for planner eligibility.
- Per-device reservations, exact M-27 manifest validation, versioned fleet
  activation and survivor replanning remain separate integration work.

No device is automatically assigned layers by these control routes. All
diagnostic results still report `eligible_for_placement: false`.

## Verification

```sh
cargo build --example fake_llama_server --example fake_rpc_server --example fake_shard_runner
cargo test --test managed_profiling -- --test-threads=1
python3 -m unittest discover -s experiments/model-validation -p 'test_*.py' -v
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

The Rust tests exercise the real helper with synthetic native fixtures, including
busy admission, real stage ownership, cancellation before waiting inference can
start, stale/missing reports, forced-stop fencing, API schema/opt-in checks and
host shutdown. They establish lifecycle behavior, not hardware performance.
One test binds a real loopback listener; environments that deny socket binds
cannot validate that test. The in-process router test covers the same control
handlers without sockets and is not reported as a substitute for listener coverage.
