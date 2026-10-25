//! `shard-serve`: relays Iroh sessions to a local shard-runner process.
//!
//! A worker that holds one layer range of a model runs [`serve_shard`]. Each approved leader
//! session (one bidirectional stream on [`RPC_BRIDGE_ALPN`] opened with
//! [`wire::CLIENT_PREFACE`]) gets a fresh runner process: the runner serves exactly one sequence
//! and has no reset, so every job needs its own. The relay adds the worker's file hashes to the
//! runner's metadata line and then forwards bytes untouched in both directions, exactly like
//! david-codex's `shard_endpoint.py`, which this replaces.
//!
//! Sessions run strictly one at a time, because a runner may claim most of the machine's memory:
//! the previous runner has exited and been reaped before the next one spawns, and later sessions
//! queue in arrival order instead of being refused.
//!
//! Like the RPC bridge, only peers on a [`PeerAllowlist`] are served, and [`serve_shard`] is
//! cancel-safe: dropping its future aborts every session and kills every runner.
//!
//! For balanced manifests (M-28) a worker can hold the **full** model file
//! ([`ShardServeOptions::layers`]) and serve whichever range a leader asks for per session, and
//! it answers probes with its current free memory and model identity; see [`wire`] for the
//! [`wire::REQUEST_PREFACE`] handshake. Probes never wait for the session lock or start a
//! runner, so they take no compute-admission (inference) lease: only a session's runner does,
//! through the same [`crate::admission::AdmittedChild`] path as a fixed-range session. Probe
//! connections still count against the worker's connection limit. A worker with a fixed range keeps refusing every other range, so old manifests run
//! unchanged.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, ensure};
use iroh::endpoint::{Connection, Incoming, RecvStream, SendStream, VarInt};
use iroh::{Endpoint, EndpointId};
use sha2::{Digest, Sha256};
use tokio::io::{
    AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader,
};
use tokio::process::{ChildStderr, ChildStdin, Command};
use tokio::sync::{Mutex, Notify, Semaphore};
use tokio::task::{JoinError, JoinSet};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

use crate::admission::AdmittedChild as Child;

use crate::protocol::RPC_BRIDGE_ALPN;
use crate::rpc_bridge::{PEER_NOT_APPROVED, PEER_NOT_APPROVED_REASON, PeerAllowlist};
use crate::shard::wire::{self, MAX_CONTEXT, Preface, ProbeReply, SessionRequest, StageMetadata};

/// Prompt passed to every runner. A relayed stage receives token IDs or hidden rows from the
/// leader, so the runner's own tokenization of its prompt argument is never used.
pub const PLACEHOLDER_PROMPT: &str = "MeshThing";

/// `RETURN_LOGITS` argument: the protocol's final stage returns the greedy token, not logits.
const RETURN_LOGITS_OFF: &str = "0";

/// How long a new stream may take to send [`wire::CLIENT_PREFACE`].
pub const PREFACE_TIMEOUT: Duration = Duration::from_secs(30);

/// How long the runner may take to load its shard and print the metadata line.
pub const RUNNER_STARTUP_TIMEOUT: Duration = Duration::from_secs(300);

/// A session with no bytes in either direction for this long ends, so a leader that vanished
/// without closing its connection cannot hold the worker forever.
pub const SESSION_IDLE_TIMEOUT: Duration = Duration::from_secs(10 * 60);

/// How long a runner may take to exit on its own after its output ends before it is killed.
const RUNNER_EXIT_GRACE: Duration = Duration::from_secs(10);

/// How long to keep draining a reaped runner's stderr, in case a grandchild still holds it.
pub(crate) const STDERR_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

/// Per-direction copy buffer; QUIC flow control and the pipes bound everything beyond it.
const COPY_BUFFER_LEN: usize = 64 * 1024;

/// Longest stderr chunk logged as one line; longer lines are split.
const MAX_STDERR_LINE: u64 = 8 * 1024;

/// Maximum incoming connections, including those queued behind the one-session runner lock.
const MAX_SHARD_CONNECTIONS: usize = 128;

/// Read size used while hashing files.
const HASH_CHUNK_LEN: usize = 8 * 1024 * 1024;

/// Hex digits in a SHA-256 digest.
const SHA256_HEX_LEN: usize = 64;

/// Application error code used to reset a session stream that failed part-way.
pub(crate) const SESSION_ABORTED: VarInt = VarInt::from_u32(1);

/// Application error code used to close a connection that negotiated another protocol.
const WRONG_ALPN: VarInt = VarInt::from_u32(2);

/// Balanced-mode settings for a worker; the default serves only the configured range.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShardServeOptions {
    /// Layers in the model when the model file is the full model: leaders may then ask for any
    /// range within `0..layers` per session, and the configured range is only the default for
    /// [`wire::CLIENT_PREFACE`] sessions. `None` serves the configured range only.
    pub layers: Option<u32>,
    /// Most memory, in bytes, the worker reports as free to probes, whatever the operating
    /// system says, so a shared machine can keep memory for other work.
    pub offer_bytes: Option<u64>,
}

/// Memory the operating system reports as available right now, in bytes: Linux
/// `MemAvailable` (and the cgroup's free memory when a cgroup limit is lower), the free plus
/// reclaimable pages on macOS, `ullAvailPhys` on Windows. Never the total. 0 when unknown.
pub fn available_memory_bytes() -> u64 {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let available = system.available_memory();
    match system.cgroup_limits() {
        Some(limits) => available.min(limits.free_memory),
        None => available,
    }
}

/// What a worker serves: one layer range of one model, run by one runner binary.
#[derive(Clone, Debug)]
pub struct ShardServeConfig {
    /// The `shard_runner` binary.
    pub runner: PathBuf,
    /// The model (or exported shard) file passed to the runner.
    pub model: PathBuf,
    /// First layer of the stage (inclusive).
    pub begin: u32,
    /// End of the stage's layer range (exclusive).
    pub end: u32,
    /// Context window passed to the runner.
    pub context: u32,
    /// Layers the runner offloads to the GPU.
    pub gpu_layers: u32,
    /// Runtime buffer budget passed to the runner.
    pub budget_bytes: u64,
    /// SHA-256 the model file must have; checked once at startup.
    pub expected_sha256: String,
    /// Original model identity an exported shard must report in its metadata, if any. When set
    /// it is announced as the stage's `model_sha256` instead of the file's own hash.
    pub source_sha256: Option<String>,
}

/// Serves `config`'s stage to peers on `allowed` over [`RPC_BRIDGE_ALPN`] until the endpoint
/// closes, then returns `Ok(())`.
///
/// The model file must hash to `expected_sha256`, or this fails before accepting anything.
/// Connections from other peers are closed with [`PEER_NOT_APPROVED`] before any stream is
/// accepted. Each stream is one session with a fresh runner, and sessions run one at a time.
pub async fn serve_shard(
    endpoint: Endpoint,
    config: ShardServeConfig,
    allowed: PeerAllowlist,
) -> anyhow::Result<()> {
    serve_shard_with(endpoint, config, ShardServeOptions::default(), allowed).await
}

/// [`serve_shard`] with balanced-mode `options`: with [`ShardServeOptions::layers`] set, the
/// configured range must lie within the model and leaders may ask for any range per session.
pub async fn serve_shard_with(
    endpoint: Endpoint,
    config: ShardServeConfig,
    options: ShardServeOptions,
    allowed: PeerAllowlist,
) -> anyhow::Result<()> {
    let shard = Arc::new(Shard::load_with(config, options).await?);
    info!(
        id = %endpoint.id(),
        begin = shard.config.begin,
        end = shard.config.end,
        any_range_of = ?shard.options.layers,
        offer_bytes = ?shard.options.offer_bytes,
        model_sha256 = %shard.model_sha256(),
        runner_sha256 = %shard.runner_sha256,
        "shard worker serving"
    );

    let connection_slots = Arc::new(Semaphore::new(MAX_SHARD_CONNECTIONS));
    let mut connections = JoinSet::new();
    loop {
        tokio::select! {
            incoming = endpoint.accept() => {
                let Some(incoming) = incoming else {
                    info!("endpoint closed; shard worker stopping");
                    return Ok(());
                };
                let Ok(permit) = connection_slots.clone().try_acquire_owned() else {
                    incoming.refuse();
                    debug!(limit = MAX_SHARD_CONNECTIONS, "shard connection limit reached; refusing connection");
                    continue;
                };
                let shard = Arc::clone(&shard);
                let allowed = allowed.clone();
                connections.spawn(async move {
                    let _permit = permit;
                    serve_connection(incoming, shard, allowed).await;
                });
            }
            Some(joined) = connections.join_next(), if !connections.is_empty() => {
                log_task_panic(joined);
            }
        }
    }
}

/// The served stage, its file hashes, and the lock that keeps sessions one at a time.
pub(crate) struct Shard {
    config: ShardServeConfig,
    options: ShardServeOptions,
    artifact_sha256: String,
    runner_sha256: String,
    sessions: Mutex<()>,
}

impl Shard {
    /// Validates `config` and hashes its files, refusing a model that does not match.
    /// Ring workers serve only their configured range ([`ShardServeOptions::default`]).
    pub(crate) async fn load(config: ShardServeConfig) -> anyhow::Result<Self> {
        Self::load_with(config, ShardServeOptions::default()).await
    }

    /// [`Shard::load`] with this worker's balancing options.
    async fn load_with(
        mut config: ShardServeConfig,
        options: ShardServeOptions,
    ) -> anyhow::Result<Self> {
        ensure!(
            config.begin < config.end,
            "invalid layer range {}:{}: begin must be below end",
            config.begin,
            config.end
        );
        if let Some(layers) = options.layers {
            ensure!(
                config.end <= layers,
                "layer range {}:{} does not fit a model of {layers} layers",
                config.begin,
                config.end
            );
        }
        ensure!(
            (1..=MAX_CONTEXT).contains(&config.context),
            "context {} outside 1..={MAX_CONTEXT}",
            config.context
        );
        config.expected_sha256 = normalize_sha256(&config.expected_sha256, "expected model")?;
        config.source_sha256 = config
            .source_sha256
            .as_deref()
            .map(|hash| normalize_sha256(hash, "source model"))
            .transpose()?;

        let artifact_sha256 = sha256_file(config.model.clone()).await?;
        ensure!(
            artifact_sha256 == config.expected_sha256,
            "model checksum mismatch for {}: expected {}, found {artifact_sha256}",
            config.model.display(),
            config.expected_sha256
        );
        let runner_sha256 = sha256_file(config.runner.clone()).await?;
        Ok(Self {
            config,
            options,
            artifact_sha256,
            runner_sha256,
            sessions: Mutex::new(()),
        })
    }

    /// Identity announced to leaders: the original model for an exported shard, else the file.
    pub(crate) fn model_sha256(&self) -> &str {
        self.config
            .source_sha256
            .as_deref()
            .unwrap_or(&self.artifact_sha256)
    }

    /// Checks the runner's metadata against this stage's configured range and adds the
    /// worker's hashes to it.
    pub(crate) fn stamp_metadata(&self, meta: &mut StageMetadata) -> anyhow::Result<()> {
        self.stamp_metadata_for(meta, self.default_range())
    }

    /// The configured range, served to [`wire::CLIENT_PREFACE`] sessions.
    fn default_range(&self) -> LayerRange {
        LayerRange {
            begin: self.config.begin,
            end: self.config.end,
        }
    }

    /// Checks a range a leader asked for: any non-empty range within the model for a
    /// full-model worker, only the configured range otherwise.
    fn check_range(&self, range: LayerRange) -> anyhow::Result<()> {
        let LayerRange { begin, end } = range;
        match self.options.layers {
            Some(layers) => ensure!(
                begin < end && end <= layers,
                "requested layers {begin}:{end} are not a non-empty range within the model's \
                 {layers} layers"
            ),
            None => ensure!(
                range == self.default_range(),
                "requested layers {begin}:{end}, but this worker serves only {}:{}",
                self.config.begin,
                self.config.end
            ),
        }
        Ok(())
    }

    /// What a probe reports: current free memory (capped by the offer) and identity.
    fn probe_reply(&self) -> ProbeReply {
        let measured = available_memory_bytes();
        if measured == 0 {
            warn!("available memory is unknown on this system; reporting 0 bytes free");
        }
        let available_bytes = self
            .options
            .offer_bytes
            .map_or(measured, |offer| measured.min(offer));
        ProbeReply {
            available_bytes,
            model_sha256: self.model_sha256().to_owned(),
            layers: self.options.layers,
            begin: self.config.begin,
            end: self.config.end,
            extra: serde_json::Map::new(),
        }
    }

    /// Checks the runner's metadata against the session's range (and, for a full-model
    /// worker, the model's layer count) and adds the worker's hashes to it.
    fn stamp_metadata_for(
        &self,
        meta: &mut StageMetadata,
        range: LayerRange,
    ) -> anyhow::Result<()> {
        ensure!(
            (meta.begin, meta.end) == (range.begin, range.end),
            "runner reports layers {}:{}, expected {}:{}",
            meta.begin,
            meta.end,
            range.begin,
            range.end
        );
        if let Some(layers) = self.options.layers {
            ensure!(
                meta.layers == layers,
                "runner reports a model of {} layers, but this worker was started with {layers}",
                meta.layers
            );
        }
        if let Some(source) = &self.config.source_sha256 {
            ensure!(
                meta.source_sha256.eq_ignore_ascii_case(source),
                "exported shard source identity mismatch: runner reports {:?}, expected {source}",
                meta.source_sha256
            );
        }
        meta.model_sha256 = Some(self.model_sha256().to_owned());
        meta.artifact_sha256 = Some(self.artifact_sha256.clone());
        meta.runner_sha256 = Some(self.runner_sha256.clone());
        Ok(())
    }

    /// Starts a runner for this stage's configured range; see [`Shard::spawn_runner_for`].
    pub(crate) async fn spawn_runner(&self) -> anyhow::Result<Child> {
        self.spawn_runner_for(self.default_range()).await
    }

    /// Starts a runner for `range` with piped stdio through the process's compute admission
    /// gate (it holds an inference lease until reaped); it is killed if the handle is dropped.
    async fn spawn_runner_for(&self, range: LayerRange) -> anyhow::Result<Child> {
        let config = &self.config;
        Child::spawn(
            Command::new(&config.runner)
                .arg(&config.model)
                .args([
                    range.begin.to_string(),
                    range.end.to_string(),
                    config.context.to_string(),
                    config.gpu_layers.to_string(),
                    config.budget_bytes.to_string(),
                    PLACEHOLDER_PROMPT.to_owned(),
                    RETURN_LOGITS_OFF.to_owned(),
                ])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true),
        )
        .await
        .with_context(|| format!("failed to start shard runner {}", config.runner.display()))
    }
}

/// A half-open layer range one session serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LayerRange {
    begin: u32,
    end: u32,
}

/// Lowercases a configured SHA-256, refusing anything but 64 hex digits.
fn normalize_sha256(hash: &str, what: &str) -> anyhow::Result<String> {
    let valid = hash.len() == SHA256_HEX_LEN && hash.bytes().all(|byte| byte.is_ascii_hexdigit());
    ensure!(
        valid,
        "{what} sha256 {hash:?} is not {SHA256_HEX_LEN} hex digits"
    );
    Ok(hash.to_ascii_lowercase())
}

/// SHA-256 of a file as lowercase hex, computed on a blocking thread.
async fn sha256_file(path: PathBuf) -> anyhow::Result<String> {
    tokio::task::spawn_blocking(move || hash_file_blocking(&path))
        .await
        .context("file hashing task failed")?
}

fn hash_file_blocking(path: &Path) -> anyhow::Result<String> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("failed to open {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; HASH_CHUNK_LEN];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read {} for hashing", path.display()))?;
        if read == 0 {
            return Ok(wire::encode_hex(&hasher.finalize()));
        }
        hasher.update(&buffer[..read]);
    }
}

/// One Iroh connection: refuse unapproved peers, then run each stream as a session in turn.
async fn serve_connection(incoming: Incoming, shard: Arc<Shard>, allowed: PeerAllowlist) {
    let connection = match incoming.await {
        Ok(connection) => connection,
        Err(error) => {
            warn!(%error, "incoming shard connection failed");
            return;
        }
    };
    let peer = connection.remote_id();
    if connection.alpn() != RPC_BRIDGE_ALPN {
        warn!(%peer, alpn = %String::from_utf8_lossy(connection.alpn()), "rejecting connection");
        connection.close(WRONG_ALPN, b"unexpected alpn");
        return;
    }
    // Checked before accepting any stream, so an unapproved peer never starts a runner.
    if !allowed.contains(&peer) {
        warn!(%peer, "rejecting shard peer that is not on the allowlist");
        connection.close(PEER_NOT_APPROVED, PEER_NOT_APPROVED_REASON);
        return;
    }
    info!(%peer, "shard leader connected");

    loop {
        match connection.accept_bi().await {
            Ok(stream) => serve_stream(&connection, &shard, stream).await,
            Err(error) => {
                info!(%peer, %error, "shard connection closed");
                return;
            }
        }
    }
}

/// One stream: read the handshake, then answer a probe, or wait for this worker's turn and
/// run the session over the requested (or configured) range.
async fn serve_stream(
    connection: &Connection,
    shard: &Shard,
    (mut send, recv): (SendStream, RecvStream),
) {
    let peer = connection.remote_id();
    let mut recv = BufReader::new(recv);
    let handshake = timeout(PREFACE_TIMEOUT, read_handshake(&mut recv)).await;
    let request = match handshake
        .context("timed out waiting for the shard client preface")
        .and_then(|read| read)
    {
        Ok(request) => request,
        Err(error) => return refuse(peer, &error, &mut send, recv.get_mut()),
    };
    let range = match request {
        Handshake::Probe => {
            answer_probe(shard, &mut send, recv.get_mut(), peer).await;
            return;
        }
        Handshake::Session(range) => range.unwrap_or_else(|| shard.default_range()),
    };
    if let Err(error) = shard.check_range(range) {
        return refuse(peer, &error, &mut send, recv.get_mut());
    }

    let Some(_turn) = wait_for_turn(connection, shard, &send).await else {
        info!(%peer, "shard leader left while waiting for its session");
        abort_stream(&mut send, recv.get_mut());
        return;
    };
    info!(%peer, begin = range.begin, end = range.end, "shard session started");
    let started = Instant::now();
    let result = run_session(shard, range, &mut send, &mut recv).await;
    let duration_ms = started.elapsed().as_millis();
    match result {
        Ok(status) => info!(%peer, duration_ms, %status, "shard session ended"),
        Err(error) => {
            abort_stream(&mut send, recv.get_mut());
            warn!(%peer, duration_ms, error = %format!("{error:#}"), "shard session failed");
        }
    }
}

/// Logs why a session is refused and aborts its stream.
fn refuse(peer: EndpointId, error: &anyhow::Error, send: &mut SendStream, recv: &mut RecvStream) {
    warn!(%peer, error = %format!("{error:#}"), "refusing shard session");
    abort_stream(send, recv);
}

/// What a client opened its stream for.
enum Handshake {
    /// A probe for free memory and identity.
    Probe,
    /// A session over the given range, or the configured one (`None`, the original preface).
    Session(Option<LayerRange>),
}

/// Reads either preface and, after [`wire::REQUEST_PREFACE`], the request line.
async fn read_handshake<R: AsyncBufRead + Unpin>(recv: &mut R) -> anyhow::Result<Handshake> {
    match wire::read_any_preface(recv).await? {
        Preface::Session => Ok(Handshake::Session(None)),
        Preface::Request => Ok(match wire::read_session_request(recv).await? {
            SessionRequest::Probe => Handshake::Probe,
            SessionRequest::Session { begin, end } => {
                Handshake::Session(Some(LayerRange { begin, end }))
            }
        }),
    }
}

/// Answers a probe and finishes the stream.
async fn answer_probe(
    shard: &Shard,
    send: &mut SendStream,
    recv: &mut RecvStream,
    peer: EndpointId,
) {
    let reply = shard.probe_reply();
    let written = wire::write_probe_reply(send, &reply).await;
    match written.and_then(|()| send.finish().context("failed to finish the probe stream")) {
        Ok(()) => info!(%peer, available_bytes = reply.available_bytes, "answered shard probe"),
        Err(error) => {
            warn!(%peer, error = %format!("{error:#}"), "failed to answer shard probe");
            abort_stream(send, recv);
        }
    }
}

/// Waits until no other session is running, or returns `None` if the leader gives up first.
async fn wait_for_turn<'a>(
    connection: &Connection,
    shard: &'a Shard,
    send: &SendStream,
) -> Option<tokio::sync::MutexGuard<'a, ()>> {
    tokio::select! {
        turn = shard.sessions.lock() => Some(turn),
        _ = connection.closed() => None,
        _ = send.stopped() => None,
    }
}

/// Runs one runner for the session and always reaps it, returning its exit status.
async fn run_session<R: AsyncRead + Unpin>(
    shard: &Shard,
    range: LayerRange,
    send: &mut SendStream,
    recv: &mut R,
) -> anyhow::Result<ExitStatus> {
    let mut runner = shard.spawn_runner_for(range).await?;
    let mut logs = JoinSet::new();
    if let Some(stderr) = runner.stderr.take() {
        logs.spawn(forward_stderr(stderr, runner.id()));
    }
    let relayed = relay_runner(shard, range, &mut runner, send, recv).await;
    let reaped = reap(&mut runner, relayed.is_ok()).await;
    // Collect the runner's last words (often the reason it failed) before reporting.
    let _ = timeout(STDERR_DRAIN_TIMEOUT, logs.join_next()).await;
    relayed?;
    let status = reaped?;
    if !status.success() {
        warn!(%status, "shard runner exited unsuccessfully");
    }
    Ok(status)
}

/// Forwards the runner's stamped metadata, then relays bytes until the runner's output ends.
async fn relay_runner<R: AsyncRead + Unpin>(
    shard: &Shard,
    range: LayerRange,
    runner: &mut Child,
    send: &mut SendStream,
    recv: &mut R,
) -> anyhow::Result<()> {
    let stdin = runner.stdin.take().context("shard runner has no stdin")?;
    let stdout = runner.stdout.take().context("shard runner has no stdout")?;
    let mut stdout = BufReader::new(stdout);
    let mut meta = timeout(RUNNER_STARTUP_TIMEOUT, wire::read_metadata(&mut stdout))
        .await
        .context("timed out waiting for the shard runner to start")??;
    shard.stamp_metadata_for(&mut meta, range)?;
    wire::write_metadata(send, &meta).await?;
    // The buffered reader may already hold the start of the runner's output; it is relayed too.
    relay_bytes(recv, stdin, &mut stdout, send).await
}

/// Copies the leader's requests into the runner and its replies back until the runner's output
/// ends, closing the runner's stdin once the leader finishes sending. Fails on an I/O error or
/// after [`SESSION_IDLE_TIMEOUT`] without traffic.
async fn relay_bytes<I, R, W>(
    recv: &mut I,
    stdin: ChildStdin,
    stdout: &mut R,
    send: &mut W,
) -> anyhow::Result<()>
where
    I: AsyncRead + Unpin,
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let activity = Notify::new();
    let inbound = forward_requests(recv, stdin, &activity);
    let outbound = pump(stdout, send, &activity);
    let idle = idle_watchdog(&activity);
    tokio::pin!(inbound, outbound, idle);
    let mut inbound_done = false;
    loop {
        tokio::select! {
            result = &mut inbound, if !inbound_done => {
                result.context("leader to runner")?;
                inbound_done = true;
            }
            result = &mut outbound => return result.context("runner to leader"),
            error = &mut idle => return Err(error),
        }
    }
}

/// Copies requests into the runner, then drops its stdin so it sees end of input.
async fn forward_requests<I: AsyncRead + Unpin>(
    recv: &mut I,
    mut stdin: ChildStdin,
    activity: &Notify,
) -> std::io::Result<()> {
    pump(recv, &mut stdin, activity).await
}

/// Copies `reader` into `writer` through a fixed buffer, signalling `activity` after every
/// chunk, then shuts `writer` down so the far side sees EOF.
async fn pump<R, W>(reader: &mut R, writer: &mut W, activity: &Notify) -> std::io::Result<()>
where
    R: AsyncRead + Unpin + ?Sized,
    W: AsyncWrite + Unpin + ?Sized,
{
    let mut buffer = vec![0u8; COPY_BUFFER_LEN];
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            return writer.shutdown().await;
        }
        writer.write_all(&buffer[..read]).await?;
        writer.flush().await?;
        activity.notify_one();
    }
}

/// Resolves to an error once [`SESSION_IDLE_TIMEOUT`] passes without a signal on `activity`.
async fn idle_watchdog(activity: &Notify) -> anyhow::Error {
    while timeout(SESSION_IDLE_TIMEOUT, activity.notified())
        .await
        .is_ok()
    {}
    anyhow::anyhow!(
        "no traffic for {}s; ending idle session",
        SESSION_IDLE_TIMEOUT.as_secs()
    )
}

/// Waits for the runner to exit, killing it unless it ended cleanly and exits within
/// [`RUNNER_EXIT_GRACE`]. Returns only once the process has been reaped.
pub(crate) async fn reap(runner: &mut Child, ended_cleanly: bool) -> anyhow::Result<ExitStatus> {
    // Its stdin is already gone on the clean path; dropping it here covers the error paths.
    drop(runner.stdin.take());
    if ended_cleanly && let Ok(status) = timeout(RUNNER_EXIT_GRACE, runner.wait()).await {
        return status.context("failed to wait for the shard runner");
    }
    // Fails only if the runner has already exited, which `wait` then reports.
    let _ = runner.start_kill();
    runner
        .wait()
        .await
        .context("failed to reap the shard runner")
}

/// Logs the runner's stderr line by line until it closes.
pub(crate) async fn forward_stderr(stderr: ChildStderr, pid: Option<u32>) {
    let pid = pid.unwrap_or_default();
    let mut reader = BufReader::new(stderr);
    let mut line = Vec::new();
    loop {
        line.clear();
        match (&mut reader)
            .take(MAX_STDERR_LINE)
            .read_until(b'\n', &mut line)
            .await
        {
            Ok(0) => return,
            Ok(_) => {
                let truncated = line.last() != Some(&b'\n') && line.len() as u64 == MAX_STDERR_LINE;
                let text = String::from_utf8_lossy(&line);
                info!(pid, truncated, "shard runner: {}", text.trim_end());
            }
            Err(error) => {
                warn!(pid, %error, "failed to read shard runner stderr");
                return;
            }
        }
    }
}

/// Resets both halves of a stream so the peer sees an abort instead of a clean EOF.
fn abort_stream(send: &mut SendStream, recv: &mut RecvStream) {
    // Both calls only fail when that half is already closed, which is the state we want.
    let _ = send.reset(SESSION_ABORTED);
    let _ = recv.stop(SESSION_ABORTED);
}

/// Session tasks handle their own errors, so a join error can only be a panic worth reporting.
pub(crate) fn log_task_panic(joined: Result<(), JoinError>) {
    if let Err(join_error) = joined {
        error!(%join_error, "shard worker task panicked");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(model: PathBuf, expected_sha256: &str) -> ShardServeConfig {
        ShardServeConfig {
            runner: model.clone(),
            model,
            begin: 0,
            end: 2,
            context: 64,
            gpu_layers: 0,
            budget_bytes: 1,
            expected_sha256: expected_sha256.to_owned(),
            source_sha256: None,
        }
    }

    fn temp_file(name: &str, contents: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
        std::fs::write(&path, contents).expect("write temp file");
        path
    }

    fn meta(begin: u32, end: u32, source_sha256: &str) -> StageMetadata {
        StageMetadata {
            protocol: 1,
            begin,
            end,
            layers: 4,
            embd: 8,
            vocab: 300,
            allocated_bytes: 0,
            source_sha256: source_sha256.to_owned(),
            tokens: Vec::new(),
            model_sha256: None,
            artifact_sha256: None,
            runner_sha256: None,
            extra: serde_json::Map::new(),
        }
    }

    /// The range [`config`] serves.
    const CONFIGURED: LayerRange = LayerRange { begin: 0, end: 2 };

    async fn load(config: ShardServeConfig) -> anyhow::Result<Shard> {
        Shard::load(config).await
    }

    async fn load_full(
        config: ShardServeConfig,
        layers: u32,
        offer: Option<u64>,
    ) -> anyhow::Result<Shard> {
        let options = ShardServeOptions {
            layers: Some(layers),
            offer_bytes: offer,
        };
        Shard::load_with(config, options).await
    }

    /// SHA-256 of `abc` (FIPS 180-2 test vector).
    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    #[tokio::test]
    async fn hashes_files_as_lowercase_hex() {
        let path = temp_file("shard-serve-hash", b"abc");
        assert_eq!(sha256_file(path.clone()).await.unwrap(), ABC_SHA256);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn load_accepts_matching_model_in_any_case() {
        let path = temp_file("shard-serve-load", b"abc");
        let shard = load(config(path.clone(), &ABC_SHA256.to_ascii_uppercase()))
            .await
            .unwrap();
        assert_eq!(shard.model_sha256(), ABC_SHA256);
        assert_eq!(shard.runner_sha256, ABC_SHA256);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn load_refuses_mismatched_model_and_bad_config() {
        let path = temp_file("shard-serve-mismatch", b"abc");
        let error = load(config(path.clone(), &"0".repeat(64)))
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("checksum mismatch"), "{error:#}");

        let short = load(config(path.clone(), "abc")).await.err().unwrap();
        assert!(short.to_string().contains("hex digits"), "{short:#}");

        let mut empty_range = config(path.clone(), ABC_SHA256);
        empty_range.end = empty_range.begin;
        assert!(load(empty_range).await.is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn stamps_hashes_and_checks_source_identity() {
        let path = temp_file("shard-serve-stamp", b"abc");
        let source = "1".repeat(64);
        let mut exported = config(path.clone(), ABC_SHA256);
        exported.source_sha256 = Some(source.clone());
        let shard = load(exported).await.unwrap();

        let mut good = meta(0, 2, &source);
        shard.stamp_metadata_for(&mut good, CONFIGURED).unwrap();
        assert_eq!(good.model_sha256.as_deref(), Some(source.as_str()));
        assert_eq!(good.artifact_sha256.as_deref(), Some(ABC_SHA256));
        assert_eq!(good.runner_sha256.as_deref(), Some(ABC_SHA256));

        assert!(
            shard
                .stamp_metadata_for(&mut meta(0, 2, ""), CONFIGURED)
                .is_err()
        );
        assert!(
            shard
                .stamp_metadata_for(&mut meta(0, 3, &source), CONFIGURED)
                .is_err()
        );
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn a_fixed_range_worker_serves_only_its_range() {
        let path = temp_file("shard-serve-fixed", b"abc");
        let shard = load(config(path.clone(), ABC_SHA256)).await.unwrap();
        assert!(shard.check_range(CONFIGURED).is_ok());
        let other = LayerRange { begin: 0, end: 3 };
        let error = shard.check_range(other).unwrap_err().to_string();
        assert!(error.contains("serves only 0:2"), "{error}");
        let probe = shard.probe_reply();
        assert_eq!((probe.layers, probe.begin, probe.end), (None, 0, 2));
        assert_eq!(probe.model_sha256, ABC_SHA256);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn a_full_model_worker_serves_any_range_within_the_model() {
        let path = temp_file("shard-serve-full", b"abc");
        let shard = load_full(config(path.clone(), ABC_SHA256), 4, None)
            .await
            .unwrap();
        for (begin, end) in [(0, 4), (1, 3), (3, 4)] {
            assert!(
                shard.check_range(LayerRange { begin, end }).is_ok(),
                "{begin}:{end}"
            );
        }
        for (begin, end) in [(0, 5), (2, 2), (3, 1)] {
            assert!(
                shard.check_range(LayerRange { begin, end }).is_err(),
                "{begin}:{end}"
            );
        }
        let range = LayerRange { begin: 1, end: 3 };
        let mut good = meta(1, 3, "");
        shard.stamp_metadata_for(&mut good, range).unwrap();
        assert_eq!(good.model_sha256.as_deref(), Some(ABC_SHA256));
        assert!(
            shard
                .stamp_metadata_for(&mut meta(0, 2, ""), range)
                .is_err()
        );
        let mut wrong_model = meta(1, 3, "");
        wrong_model.layers = 6;
        let error = shard
            .stamp_metadata_for(&mut wrong_model, range)
            .unwrap_err();
        assert!(error.to_string().contains("6 layers"), "{error:#}");

        let mut too_long = config(path.clone(), ABC_SHA256);
        too_long.end = 5;
        assert!(load_full(too_long, 4, None).await.is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn probes_report_available_memory_capped_by_the_offer() {
        let path = temp_file("shard-serve-probe", b"abc");
        let shard = load_full(config(path.clone(), ABC_SHA256), 4, Some(1234))
            .await
            .unwrap();
        let probe = shard.probe_reply();
        assert_eq!(probe.layers, Some(4));
        assert!(probe.available_bytes <= 1234, "{probe:?}");

        let uncapped = load_full(config(path.clone(), ABC_SHA256), 4, None)
            .await
            .unwrap();
        let total = {
            let mut system = sysinfo::System::new();
            system.refresh_memory();
            system.total_memory()
        };
        let available = uncapped.probe_reply().available_bytes;
        assert!(
            available > 0,
            "available memory should be known on CI platforms"
        );
        assert!(available <= total, "{available} > {total}");
        std::fs::remove_file(path).unwrap();
    }
}
