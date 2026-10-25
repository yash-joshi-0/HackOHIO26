//! The origin (leader) side of `meshthing/shard-ring/1`. EXPERIMENTAL, pending david-codex's
//! production sign-off.
//!
//! In a ring manifest the leader runs stage 0 itself and every later stage is a
//! `shard-ring-worker`. For each generation a [`RingLink`]:
//!
//! 1. generates a job id and registers it with the leader's [`ResultChannels`] for the manifest's
//!    terminal worker;
//! 2. opens the entry session ([`open_ring_session`]) and checks the chain the entry reports
//!    against the manifest ([`check_ready`]);
//! 3. waits, bounded by [`RingSettings::result_channel_wait`], for the terminal's result channel,
//!    which the leader's accept loop hands over through the registry;
//! 4. per step, sends stage 0's hidden rows to the entry as an execute frame and reads the step's
//!    token from the result channel, while a background task drains the entry stream's progress
//!    lines so QUIC flow control never stalls the ring.
//!
//! The origin never receives activations: nothing here reads a payload, and every byte read from
//! a ring stream is counted ([`RingTraffic`]) so tests can check that all of it was JSON control
//! lines. Any error fails the generation; nothing is retried or re-prefilled.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll};
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow, bail, ensure};
use iroh::endpoint::{Connection, RecvStream, SendStream, VarInt};
use iroh::{Endpoint, EndpointId};
use serde_json::Value;
use tokio::io::{AsyncBufRead, AsyncRead, BufReader, ReadBuf};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::{timeout, timeout_at};
use tracing::{debug, info, warn};

use crate::rpc_bridge::{PEER_NOT_APPROVED, PEER_NOT_APPROVED_REASON, path_kind};
use crate::shard::manifest::{ShardManifest, StageLocation, same_sha256};
use crate::shard::ring::{
    ExecuteFrame, NATIVE_STEP_TIMEOUT, RingMessage, RingResult, RingStageInfo,
    STREAM_DRAIN_TIMEOUT, accept_result_channel, encode_activations, new_job_id, open_ring_session,
    read_control, read_result, read_result_channel_hello, validate_chain, write_execute,
};
use crate::shard::wire::{StageMetadata, StepReply};

/// Default bound on a ring becoming ready: the entry's `ready` plus the terminal's result
/// channel. Every worker loads its model first, which can take minutes for a large shard.
pub const DEFAULT_RESULT_CHANNEL_WAIT: Duration = Duration::from_secs(300);

/// Bound on one step's round trip through the whole ring: a worker may take this long for one
/// stage (ring.rs's control-line plus payload timeouts), and the result line is bounded the same.
pub const RING_STEP_TIMEOUT: Duration = NATIVE_STEP_TIMEOUT;

/// Close code for a ring session the origin is done with.
const SESSION_DONE: VarInt = VarInt::from_u32(0);

/// Close reason sent with [`SESSION_DONE`].
const SESSION_DONE_REASON: &[u8] = b"generation finished";

/// Close code (and stream reset code) for a failed ring session, as ring workers use it.
const SESSION_FAILED: VarInt = VarInt::from_u32(1);

/// Close reason sent with [`SESSION_FAILED`].
const SESSION_FAILED_REASON: &[u8] = b"ring session failed";

/// Finished sessions whose traffic the registry keeps; older ones are dropped.
pub const TRAFFIC_HISTORY: usize = 64;

/// Close code for a result channel naming a job that is not pending for that terminal.
const RESULT_CHANNEL_REFUSED: VarInt = VarInt::from_u32(4);

/// Close reason sent with [`RESULT_CHANNEL_REFUSED`].
const RESULT_CHANNEL_REFUSED_REASON: &[u8] = b"unknown or replayed ring job";

/// What the front end needs to run ring manifests: the registry the leader's accept loop feeds,
/// and how long a ring may take to become ready.
#[derive(Clone, Debug)]
pub struct RingSettings {
    /// Result channels the leader's accept loop delivers to waiting generations.
    pub channels: Arc<ResultChannels>,
    /// Bound on the entry's `ready` plus the terminal's result channel.
    pub result_channel_wait: Duration,
}

impl Default for RingSettings {
    fn default() -> Self {
        Self {
            channels: Arc::new(ResultChannels::new()),
            result_channel_wait: DEFAULT_RESULT_CHANNEL_WAIT,
        }
    }
}

/// A job waiting for its terminal's result channel.
struct PendingJob {
    terminal: EndpointId,
    sink: oneshot::Sender<ResultChannel>,
}

/// The leader's registry of ring jobs waiting for a result channel, shared between the accept
/// loop ([`ResultChannels::handle_connection`]) and the generations ([`ResultChannels::register`]).
/// It also keeps the traffic of the last [`TRAFFIC_HISTORY`] finished ring sessions.
#[derive(Default)]
pub struct ResultChannels {
    pending: Mutex<HashMap<String, PendingJob>>,
    finished: Mutex<VecDeque<RingTraffic>>,
}

impl std::fmt::Debug for ResultChannels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pending = self.pending.lock().map_or(0, |pending| pending.len());
        f.debug_struct("ResultChannels")
            .field("pending", &pending)
            .finish_non_exhaustive()
    }
}

impl ResultChannels {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `job` as waiting for a result channel from `terminal`. The entry is removed when
    /// the channel is delivered or the returned guard is dropped, whichever comes first.
    pub fn register(
        self: &Arc<Self>,
        job: &str,
        terminal: EndpointId,
    ) -> anyhow::Result<PendingResult> {
        let (sink, receiver) = oneshot::channel();
        let mut pending = self.lock_pending()?;
        ensure!(
            !pending.contains_key(job),
            "ring job {job} is already pending"
        );
        pending.insert(job.to_owned(), PendingJob { terminal, sink });
        Ok(PendingResult {
            channels: Arc::clone(self),
            job: job.to_owned(),
            receiver: Some(receiver),
        })
    }

    /// Whether `job` still waits for its result channel.
    pub fn is_pending(&self, job: &str) -> bool {
        self.lock_pending()
            .is_ok_and(|pending| pending.contains_key(job))
    }

    /// Traffic of the last [`TRAFFIC_HISTORY`] finished ring sessions, oldest first.
    pub fn traffic(&self) -> Vec<RingTraffic> {
        self.finished
            .lock()
            .map(|finished| finished.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Serves one accepted connection on the ring ALPN. A peer that is not the terminal of any
    /// pending job is closed with [`PEER_NOT_APPROVED`]. Each stream must open with
    /// `result_channel` for a job pending for this peer; it is then accepted, consumed and handed
    /// to the waiting generation. Any other job closes the connection with code 4.
    pub async fn handle_connection(&self, connection: Connection) {
        let peer = connection.remote_id();
        if !self.is_terminal(peer) {
            warn!(%peer, "rejecting ring connection from a peer that is no pending terminal");
            connection.close(PEER_NOT_APPROVED, PEER_NOT_APPROVED_REASON);
            return;
        }
        while let Ok((send, recv)) = connection.accept_bi().await {
            if let Err(error) = self.accept_stream(&connection, send, recv).await {
                warn!(%peer, error = %format!("{error:#}"), "refused a ring result channel");
                connection.close(RESULT_CHANNEL_REFUSED, RESULT_CHANNEL_REFUSED_REASON);
                return;
            }
        }
    }

    /// Reads the hello, consumes the pending job and delivers the accepted channel.
    async fn accept_stream(
        &self,
        connection: &Connection,
        mut send: SendStream,
        recv: RecvStream,
    ) -> anyhow::Result<()> {
        let mut recv = BufReader::new(recv);
        let job = read_result_channel_hello(&mut recv).await?;
        let sink = self.take(&job, connection.remote_id())?;
        accept_result_channel(&mut send, &job).await?;
        info!(%job, terminal = %connection.remote_id(), "ring result channel accepted");
        let channel = ResultChannel {
            connection: connection.clone(),
            recv,
            job,
        };
        sink.send(channel)
            .map_err(|channel| anyhow!("the generation for ring job {} is gone", channel.job))
    }

    /// Removes and returns `job`'s entry if it is pending for `peer`. A job pending for another
    /// terminal stays pending, so an impostor cannot consume it.
    fn take(&self, job: &str, peer: EndpointId) -> anyhow::Result<oneshot::Sender<ResultChannel>> {
        let mut pending = self.lock_pending()?;
        let entry = pending
            .get(job)
            .with_context(|| format!("ring job {job} is unknown or already consumed"))?;
        ensure!(
            entry.terminal == peer,
            "ring job {job} expects its result channel from {}, not {peer}",
            entry.terminal
        );
        let entry = pending
            .remove(job)
            .context("ring job vanished while it was being consumed")?;
        Ok(entry.sink)
    }

    fn is_terminal(&self, peer: EndpointId) -> bool {
        self.lock_pending()
            .is_ok_and(|pending| pending.values().any(|entry| entry.terminal == peer))
    }

    fn forget(&self, job: &str) {
        match self.pending.lock() {
            Ok(mut pending) => {
                pending.remove(job);
            }
            Err(_) => warn!(%job, "ring registry lock poisoned; pending job not removed"),
        }
    }

    fn record(&self, traffic: RingTraffic) {
        match self.finished.lock() {
            Ok(mut finished) => {
                if finished.len() == TRAFFIC_HISTORY {
                    finished.pop_front();
                }
                finished.push_back(traffic);
            }
            Err(_) => {
                warn!(job = %traffic.job, "ring registry lock poisoned; traffic not recorded")
            }
        }
    }

    fn lock_pending(
        &self,
    ) -> anyhow::Result<std::sync::MutexGuard<'_, HashMap<String, PendingJob>>> {
        self.pending
            .lock()
            .map_err(|_| anyhow!("ring registry lock poisoned"))
    }
}

/// A registered job; dropping it removes the job from the registry if it is still pending.
pub struct PendingResult {
    channels: Arc<ResultChannels>,
    job: String,
    receiver: Option<oneshot::Receiver<ResultChannel>>,
}

impl PendingResult {
    /// Waits for the terminal's accepted result channel. Applies no timeout.
    pub async fn wait(&mut self) -> anyhow::Result<ResultChannel> {
        let receiver = self
            .receiver
            .as_mut()
            .context("the result channel was already taken")?;
        let channel = receiver
            .await
            .map_err(|_| anyhow!("the result channel for ring job {} failed", self.job))?;
        self.receiver = None;
        Ok(channel)
    }
}

impl Drop for PendingResult {
    fn drop(&mut self) {
        self.channels.forget(&self.job);
    }
}

/// An accepted result channel: the terminal's connection and the stream results arrive on.
pub struct ResultChannel {
    connection: Connection,
    recv: BufReader<RecvStream>,
    job: String,
}

impl std::fmt::Debug for ResultChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResultChannel")
            .field("job", &self.job)
            .field("terminal", &self.connection.remote_id())
            .finish_non_exhaustive()
    }
}

impl ResultChannel {
    /// The job the channel was accepted for.
    pub fn job(&self) -> &str {
        &self.job
    }

    /// The connection (keep it alive while reading) and the stream results arrive on.
    pub fn into_parts(self) -> (Connection, BufReader<RecvStream>) {
        (self.connection, self.recv)
    }
}

/// What one ring session read and sent, for checking that no activation reached the origin.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RingTraffic {
    /// The session's job id.
    pub job: String,
    /// Steps whose result arrived.
    pub steps: u32,
    /// Whether the generation finished without an error.
    pub succeeded: bool,
    /// Activation payload bytes sent to the entry worker (stage 0's hidden rows).
    pub activation_bytes_sent: u64,
    /// Every byte read from the entry stream after `ready` and from the result channel after
    /// `accepted`.
    pub bytes_read: u64,
    /// The part of `bytes_read` that was whole JSON control lines.
    pub control_line_bytes: u64,
    /// Control lines read (progress lines and results).
    pub control_lines: u64,
}

impl RingTraffic {
    /// Bytes read that were not JSON control lines; the ring design keeps this at zero.
    pub fn activation_bytes_read(&self) -> u64 {
        self.bytes_read.saturating_sub(self.control_line_bytes)
    }
}

/// Live counters behind a [`RingTraffic`], shared with the progress drain task.
#[derive(Default)]
struct TrafficCounter {
    bytes_read: AtomicU64,
    control_line_bytes: AtomicU64,
    control_lines: AtomicU64,
    activation_bytes_sent: AtomicU64,
}

impl TrafficCounter {
    fn add(counter: &AtomicU64, bytes: usize) {
        counter.fetch_add(u64::try_from(bytes).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    fn snapshot(&self, job: &str, steps: u32, succeeded: bool) -> RingTraffic {
        RingTraffic {
            job: job.to_owned(),
            steps,
            succeeded,
            activation_bytes_sent: self.activation_bytes_sent.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            control_line_bytes: self.control_line_bytes.load(Ordering::Relaxed),
            control_lines: self.control_lines.load(Ordering::Relaxed),
        }
    }
}

/// A buffered ring stream that counts every byte consumed from it, and which of those bytes
/// formed control lines.
struct Counted<R> {
    inner: R,
    /// Bytes consumed from this stream; the shared counter sums every stream of the session.
    consumed: u64,
    counter: Arc<TrafficCounter>,
}

impl<R> Counted<R> {
    fn new(inner: R, counter: Arc<TrafficCounter>) -> Self {
        Self {
            inner,
            consumed: 0,
            counter,
        }
    }

    fn record_read(&mut self, bytes: usize) {
        let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
        self.consumed = self.consumed.saturating_add(bytes);
        self.counter.bytes_read.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Records the control line just read, `before` being this stream's count before it.
    fn line_read(&self, before: u64) {
        let line = self.consumed.saturating_sub(before);
        self.counter
            .control_line_bytes
            .fetch_add(line, Ordering::Relaxed);
        self.counter.control_lines.fetch_add(1, Ordering::Relaxed);
    }
}

impl<R: AsyncBufRead + Unpin> Counted<R> {
    /// [`read_control`], counted.
    async fn control(&mut self) -> anyhow::Result<Option<RingMessage>> {
        let before = self.consumed;
        let message = read_control(self).await?;
        if message.is_some() {
            self.line_read(before);
        }
        Ok(message)
    }

    /// [`read_result`], counted.
    async fn result(&mut self, job: &str, step: u32) -> anyhow::Result<RingResult> {
        let before = self.consumed;
        let result = read_result(self, job, step).await?;
        self.line_read(before);
        Ok(result)
    }
}

impl<R: AsyncRead + Unpin> AsyncRead for Counted<R> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let before = buf.filled().len();
        let poll = Pin::new(&mut this.inner).poll_read(cx, buf);
        this.record_read(buf.filled().len() - before);
        poll
    }
}

impl<R: AsyncBufRead + Unpin> AsyncBufRead for Counted<R> {
    fn poll_fill_buf(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<io::Result<&[u8]>> {
        Pin::new(&mut self.get_mut().inner).poll_fill_buf(cx)
    }

    fn consume(self: Pin<&mut Self>, amt: usize) {
        let this = self.get_mut();
        this.record_read(amt);
        Pin::new(&mut this.inner).consume(amt);
    }
}

/// One remote ring stage as the manifest places it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingHop {
    /// First layer (inclusive).
    pub begin: u32,
    /// End of the layer range (exclusive).
    pub end: u32,
    /// The `shard-ring-worker` serving the range.
    pub worker: EndpointId,
}

/// The ring a manifest describes: its remote stages in order and the model they must serve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingPlan {
    /// Remote stages from the entry to the terminal.
    pub hops: Vec<RingHop>,
    /// Total layers in the model.
    pub layers: u32,
    /// SHA-256 of the original model every ring stage must report as `source_sha256`.
    pub model_sha256: String,
}

impl RingPlan {
    /// The remote stages of a validated ring manifest (`stages[1..]`).
    pub fn from_manifest(manifest: &ShardManifest) -> anyhow::Result<Self> {
        ensure!(
            manifest.is_ring(),
            "the manifest does not use ring execution"
        );
        let hops = manifest
            .stages
            .iter()
            .skip(1)
            .map(|stage| match stage.location {
                StageLocation::Remote { worker, .. } => Ok(RingHop {
                    begin: stage.begin,
                    end: stage.end,
                    worker,
                }),
                _ => bail!("ring stage {}..{} must be remote", stage.begin, stage.end),
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        ensure!(!hops.is_empty(), "the ring manifest has no remote stage");
        Ok(Self {
            hops,
            layers: manifest.layers,
            model_sha256: manifest.model_sha256.clone(),
        })
    }

    /// The worker the origin starts sessions at.
    pub fn entry(&self) -> anyhow::Result<EndpointId> {
        self.hops
            .first()
            .map(|hop| hop.worker)
            .context("the ring has no entry")
    }

    /// The worker that returns results to the origin.
    pub fn terminal(&self) -> anyhow::Result<EndpointId> {
        self.hops
            .last()
            .map(|hop| hop.worker)
            .context("the ring has no terminal")
    }
}

/// Checks the entry's `ready.stages` against the manifest: a valid chain ([`validate_chain`])
/// whose ranges and workers are exactly the manifest's remote stages in order, whose layer count
/// is the manifest's, and whose model identity is `model_sha256`. `validate_chain` checks neither
/// where the chain starts nor where it ends; this does, through the ranges.
pub fn check_ready(stages: &[RingStageInfo], plan: &RingPlan) -> anyhow::Result<()> {
    validate_chain(stages).context("invalid ring chain")?;
    ensure!(
        stages.len() == plan.hops.len(),
        "the ring reports {} stages, but the manifest has {}",
        stages.len(),
        plan.hops.len()
    );
    for (index, (stage, hop)) in stages.iter().zip(&plan.hops).enumerate() {
        let meta = &stage.meta;
        ensure!(
            (meta.begin, meta.end) == (hop.begin, hop.end),
            "ring stage {index} serves layers {}..{}, but the manifest expects {}..{}",
            meta.begin,
            meta.end,
            hop.begin,
            hop.end
        );
        let node: EndpointId = stage
            .node_id
            .parse()
            .with_context(|| format!("invalid ring stage node id {:?}", stage.node_id))?;
        ensure!(
            node == hop.worker,
            "ring stage {index} runs on {node}, but the manifest expects {}",
            hop.worker
        );
        ensure!(
            meta.layers == plan.layers,
            "ring stage {index} reports {} layers, but the manifest has {}",
            meta.layers,
            plan.layers
        );
        ensure!(
            same_sha256(&meta.source_sha256, &plan.model_sha256),
            "ring stage {index} serves model {}, but the manifest's model_sha256 is {}",
            meta.source_sha256,
            plan.model_sha256
        );
    }
    Ok(())
}

/// Checks that the ring continues the origin's local stage 0: it starts where stage 0 ends and
/// agrees on layers, hidden size and vocabulary, and on the model when stage 0 reports one.
pub fn check_origin_stage(origin: &StageMetadata, stages: &[RingStageInfo]) -> anyhow::Result<()> {
    let first = &stages.first().context("empty ring chain")?.meta;
    ensure!(
        origin.end == first.begin,
        "the local stage ends at layer {}, but the ring starts at {}",
        origin.end,
        first.begin
    );
    let same = [
        ("layers", origin.layers == first.layers),
        ("embd", origin.embd == first.embd),
        ("vocab", origin.vocab == first.vocab),
    ];
    for (key, equal) in same {
        ensure!(equal, "the local stage and the ring disagree on {key}");
    }
    ensure!(
        origin.source_sha256.is_empty() || same_sha256(&origin.source_sha256, &first.source_sha256),
        "the local stage serves model {}, but the ring serves {}",
        origin.source_sha256,
        first.source_sha256
    );
    Ok(())
}

/// The progress drain task; aborted when dropped.
struct DrainTask(Option<JoinHandle<anyhow::Result<()>>>);

impl DrainTask {
    /// Waits for the drain to end and returns how it ended. Cancel-safe; after it has returned
    /// once, later calls fail at once.
    async fn wait(&mut self) -> anyhow::Result<()> {
        let handle = self
            .0
            .as_mut()
            .context("the progress drain already ended")?;
        let joined = handle.await;
        self.0 = None;
        joined.unwrap_or_else(|join_error| {
            Err(anyhow::Error::new(join_error).context("the progress drain died"))
        })
    }

    /// Resolves only if the entry stream ends, which mid-session means the ring failed.
    async fn failed(&mut self) -> anyhow::Error {
        match self.wait().await {
            Ok(()) => anyhow!("the ring entry ended the session early"),
            Err(error) => error.context("the ring entry stream failed"),
        }
    }
}

impl Drop for DrainTask {
    fn drop(&mut self) {
        if let Some(handle) = &self.0 {
            handle.abort();
        }
    }
}

/// Reads the entry stream's progress lines until it ends cleanly, logging them at debug level.
/// Anything but progress fails it. No timeout: the ring is silent while the origin is.
async fn drain_progress(
    mut recv: Counted<BufReader<RecvStream>>,
    job: String,
) -> anyhow::Result<()> {
    while let Some(message) = recv.control().await? {
        match message {
            RingMessage::Progress(body) => {
                let progress = Value::Object(body);
                debug!(%job, %progress, "ring progress");
            }
            other => bail!("unexpected {} line on the ring entry stream", other.kind()),
        }
    }
    Ok(())
}

/// The ring as one stage of a generation: it takes stage 0's hidden rows and returns the
/// terminal's final reply for each step. Single-use, like every stage link.
pub struct RingLink {
    job: String,
    label: String,
    entry: Connection,
    send: SendStream,
    drain: DrainTask,
    results: Counted<BufReader<RecvStream>>,
    result_connection: Connection,
    stages: Vec<RingStageInfo>,
    counter: Arc<TrafficCounter>,
    steps: u32,
    channels: Arc<ResultChannels>,
    /// Keeps the job registered (and removes it when dropped) for the whole generation.
    _pending: PendingResult,
}

impl RingLink {
    /// Starts a ring session for one generation: registers a fresh job, opens the entry session,
    /// checks its chain against `plan` and waits for the terminal's result channel, all within
    /// `settings.result_channel_wait`.
    pub async fn open(
        endpoint: &Endpoint,
        plan: &RingPlan,
        settings: &RingSettings,
    ) -> anyhow::Result<Self> {
        let job = new_job_id();
        let entry = plan.entry()?;
        let mut pending = settings.channels.register(&job, plan.terminal()?)?;
        let wait = settings.result_channel_wait;
        let label = format!("ring job {job} from {}", entry.fmt_short());
        let started = Instant::now();
        let (stream, channel) = timeout(wait, ready(endpoint, plan, &job, &mut pending))
            .await
            .with_context(|| {
                format!(
                    "{label} did not become ready within {wait:?} (entry readiness plus the \
                     terminal's result channel)"
                )
            })??;
        info!(session = %label, stages = stream.stages.len(), "ring ready");
        debug!(
            session = %label,
            entry_path = path_kind(&stream.connection),
            terminal_path = path_kind(&channel.connection),
            ready_us = started.elapsed().as_micros(),
            "ring session ready"
        );
        let counter = Arc::new(TrafficCounter::default());
        let progress = Counted::new(stream.recv, Arc::clone(&counter));
        let drain = DrainTask(Some(tokio::spawn(drain_progress(progress, job.clone()))));
        let (result_connection, results) = channel.into_parts();
        Ok(Self {
            results: Counted::new(results, Arc::clone(&counter)),
            job,
            label,
            entry: stream.connection,
            send: stream.send,
            drain,
            result_connection,
            stages: stream.stages,
            counter,
            steps: 0,
            channels: Arc::clone(&settings.channels),
            _pending: pending,
        })
    }

    /// The validated chain, from the entry to the terminal.
    pub fn stages(&self) -> &[RingStageInfo] {
        &self.stages
    }

    /// Sends `rows` (`count` rows of stage 0's output at `step` and `position`) to the entry and
    /// returns the terminal's result for that step.
    pub(crate) async fn exchange(
        &mut self,
        step: u32,
        position: u32,
        count: usize,
        rows: &[f32],
    ) -> anyhow::Result<StepReply> {
        let embd = self.stages.first().context("empty ring chain")?.meta.embd;
        let n = u32::try_from(count).context("batch too large")?;
        let frame = ExecuteFrame::new(&self.job, step, position, n, embd);
        let payload = encode_activations(rows);
        let started = Instant::now();
        let deadline = tokio::time::Instant::now() + RING_STEP_TIMEOUT;
        let sent = self.send_step(&frame, &payload, deadline).await?;
        let write_us = started.elapsed().as_micros();
        debug!(
            session = %self.label,
            step,
            position,
            input_bytes = sent,
            write_us,
            "ring activations sent"
        );
        let read_started = Instant::now();
        let before = self.results.consumed;
        let result = self.await_result(step, deadline).await?;
        debug!(
            session = %self.label,
            step,
            position,
            input_bytes = sent,
            output_bytes = self.results.consumed.saturating_sub(before),
            write_us,
            read_us = read_started.elapsed().as_micros(),
            total_us = started.elapsed().as_micros(),
            "shard ring exchange"
        );
        self.steps += 1;
        Ok(result.reply)
    }

    /// Writes one execute frame and its payload to the entry before `deadline`; returns the
    /// payload size.
    async fn send_step(
        &mut self,
        frame: &ExecuteFrame,
        payload: &[u8],
        deadline: tokio::time::Instant,
    ) -> anyhow::Result<usize> {
        let step = frame.step;
        let sent = timeout_at(deadline, write_execute(&mut self.send, frame, payload))
            .await
            .with_context(|| {
                format!(
                    "{} timed out sending step {step} activations within {RING_STEP_TIMEOUT:?}",
                    self.label
                )
            })?
            .with_context(|| format!("{} failed sending step {step}", self.label))?;
        TrafficCounter::add(&self.counter.activation_bytes_sent, sent);
        Ok(sent)
    }

    /// Waits before `deadline` for the terminal's result for `step`, failing at once if the
    /// entry stream ends first (the ring failed).
    async fn await_result(
        &mut self,
        step: u32,
        deadline: tokio::time::Instant,
    ) -> anyhow::Result<RingResult> {
        let Self {
            job,
            drain,
            results,
            ..
        } = self;
        let arrival = async {
            tokio::select! {
                biased;
                result = results.result(job, step) => result,
                error = drain.failed() => Err(error),
            }
        };
        timeout_at(deadline, arrival)
            .await
            .with_context(|| {
                format!(
                    "{} timed out waiting for the step {step} result within {RING_STEP_TIMEOUT:?}",
                    self.label
                )
            })?
            .with_context(|| format!("{} failed at step {step}", self.label))
    }

    /// Ends the session. After a successful generation it finishes the entry stream and waits
    /// (bounded) for the ring to drain: the entry finishes its stream once every worker is done,
    /// and the terminal finishes the result channel. On failure it resets the entry stream.
    /// Either way it closes both connections and records the session's traffic.
    pub(crate) async fn close(mut self, success: bool) {
        let succeeded = success && self.finish().await;
        if !succeeded {
            // Fails only if the stream is already closed, which closing the connection covers.
            let _ = self.send.reset(SESSION_FAILED);
        }
        let (code, reason) = if succeeded {
            (SESSION_DONE, SESSION_DONE_REASON)
        } else {
            (SESSION_FAILED, SESSION_FAILED_REASON)
        };
        self.entry.close(code, reason);
        self.result_connection.close(code, reason);
        let traffic = self.counter.snapshot(&self.job, self.steps, success);
        info!(
            session = %self.label,
            steps = traffic.steps,
            activation_bytes_sent = traffic.activation_bytes_sent,
            bytes_read = traffic.bytes_read,
            activation_bytes_read = traffic.activation_bytes_read(),
            "ring session closed"
        );
        self.channels.record(traffic);
    }

    /// Finishes the entry stream and waits for both ring streams to end cleanly; logs and
    /// returns `false` if they do not within [`STREAM_DRAIN_TIMEOUT`].
    async fn finish(&mut self) -> bool {
        let Self {
            send,
            drain,
            results,
            ..
        } = self;
        let drained = async {
            send.finish().context("failed to finish the entry stream")?;
            tokio::try_join!(drain.wait(), results_end(results)).map(|_| ())
        };
        match timeout(STREAM_DRAIN_TIMEOUT, drained).await {
            Ok(Ok(())) => true,
            Ok(Err(error)) => {
                warn!(session = %self.label, error = %format!("{error:#}"), "ring did not end cleanly");
                false
            }
            Err(_) => {
                warn!(session = %self.label, "ring did not drain within {STREAM_DRAIN_TIMEOUT:?}");
                false
            }
        }
    }
}

/// Opens the entry session, checks its chain and waits for the result channel. On a bad chain
/// the entry connection is closed at once, so the workers see a failure rather than a hang.
async fn ready(
    endpoint: &Endpoint,
    plan: &RingPlan,
    job: &str,
    pending: &mut PendingResult,
) -> anyhow::Result<(crate::shard::ring::RingStream, ResultChannel)> {
    let entry = plan.entry()?;
    let stream = open_ring_session(endpoint, entry, job)
        .await
        .with_context(|| format!("failed to start ring job {job} at entry worker {entry}"))?;
    if let Err(error) = check_ready(&stream.stages, plan) {
        stream
            .connection
            .close(SESSION_FAILED, SESSION_FAILED_REASON);
        return Err(error.context("the ring's readiness does not match the manifest"));
    }
    match pending.wait().await {
        Ok(channel) => Ok((stream, channel)),
        Err(error) => {
            stream
                .connection
                .close(SESSION_FAILED, SESSION_FAILED_REASON);
            Err(error)
        }
    }
}

/// Reads the result channel to its end; nothing may follow the last result.
async fn results_end(results: &mut Counted<BufReader<RecvStream>>) -> anyhow::Result<()> {
    match results.control().await? {
        None => Ok(()),
        Some(other) => bail!("unexpected {} line after the last result", other.kind()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, json};
    use tokio::io::AsyncReadExt;

    use super::*;

    const LAYERS: u32 = 4;

    fn hash() -> String {
        "ab".repeat(32)
    }

    fn id(seed: u8) -> EndpointId {
        iroh::SecretKey::from_bytes(&[seed; 32]).public()
    }

    fn meta(begin: u32, end: u32) -> StageMetadata {
        StageMetadata {
            protocol: 1,
            begin,
            end,
            layers: LAYERS,
            embd: 8,
            vocab: 300,
            allocated_bytes: 0,
            source_sha256: hash(),
            tokens: Vec::new(),
            model_sha256: None,
            artifact_sha256: None,
            runner_sha256: None,
            extra: Map::new(),
        }
    }

    fn info(begin: u32, end: u32, seed: u8) -> RingStageInfo {
        RingStageInfo {
            meta: meta(begin, end),
            node_id: id(seed).to_string(),
        }
    }

    fn plan() -> RingPlan {
        RingPlan {
            hops: vec![
                RingHop {
                    begin: 1,
                    end: 3,
                    worker: id(1),
                },
                RingHop {
                    begin: 3,
                    end: 4,
                    worker: id(2),
                },
            ],
            layers: LAYERS,
            model_sha256: hash().to_uppercase(),
        }
    }

    fn ready_error(stages: &[RingStageInfo]) -> String {
        format!(
            "{:#}",
            check_ready(stages, &plan()).expect_err("chain must be refused")
        )
    }

    #[test]
    fn matching_chain_is_ready() {
        check_ready(&[info(1, 3, 1), info(3, 4, 2)], &plan()).unwrap();
    }

    #[test]
    fn workers_out_of_order_are_refused() {
        let error = ready_error(&[info(1, 3, 2), info(3, 4, 1)]);
        assert!(error.contains("but the manifest expects"), "{error}");
    }

    #[test]
    fn ranges_other_than_the_manifest_are_refused() {
        let error = ready_error(&[info(1, 2, 1), info(2, 4, 2)]);
        assert!(error.contains("serves layers 1..2"), "{error}");
    }

    #[test]
    fn chain_must_start_and_end_where_the_manifest_does() {
        // validate_chain alone accepts both: it checks neither ends of the chain.
        let error = ready_error(&[info(0, 3, 1), info(3, 4, 2)]);
        assert!(error.contains("serves layers 0..3"), "{error}");
        let error = ready_error(&[info(1, 3, 1)]);
        assert!(error.contains("reports 1 stages"), "{error}");
    }

    #[test]
    fn wrong_model_identity_is_refused() {
        let mut stages = vec![info(1, 3, 1), info(3, 4, 2)];
        for stage in &mut stages {
            stage.meta.source_sha256 = "cd".repeat(32);
        }
        let error = ready_error(&stages);
        assert!(error.contains("model_sha256"), "{error}");
    }

    #[test]
    fn invalid_chain_is_refused() {
        let mut stages = vec![info(1, 3, 1), info(3, 4, 2)];
        stages[1].meta.embd = 16;
        assert!(ready_error(&stages).contains("invalid ring chain"));
        let mut layers = vec![info(1, 3, 1), info(3, 4, 2)];
        for stage in &mut layers {
            stage.meta.layers = 5;
        }
        let error = ready_error(&layers);
        assert!(error.contains("5 layers"), "{error}");
    }

    #[test]
    fn origin_stage_must_continue_into_the_ring() {
        let ring = [info(1, 3, 1), info(3, 4, 2)];
        let mut origin = meta(0, 1);
        check_origin_stage(&origin, &ring).unwrap();
        origin.source_sha256.clear();
        check_origin_stage(&origin, &ring).unwrap();
        let mut gap = meta(0, 2);
        gap.end = 2;
        assert!(check_origin_stage(&gap, &ring).is_err());
        let mut embd = meta(0, 1);
        embd.embd = 16;
        let error = format!("{:#}", check_origin_stage(&embd, &ring).unwrap_err());
        assert!(error.contains("embd"), "{error}");
        let mut model = meta(0, 1);
        model.source_sha256 = "cd".repeat(32);
        assert!(check_origin_stage(&model, &ring).is_err());
    }

    #[test]
    fn plan_comes_from_a_ring_manifest_only() {
        let value = json!({
            "runner": "shard_runner",
            "model_sha256": hash(),
            "layers": LAYERS,
            "context": 128,
            "chat_template": "chatml",
            "execution": "ring",
            "stages": [
                {"begin": 0, "end": 1, "local": {"model": "m.gguf", "budget_bytes": 1}},
                {"begin": 1, "end": 3, "remote": {"worker": id(1).to_string()}},
                {"begin": 3, "end": 4, "remote": {"worker": id(2).to_string()}},
            ],
        });
        let mut manifest: ShardManifest = serde_json::from_value(value).unwrap();
        manifest.validate().unwrap();
        let plan = RingPlan::from_manifest(&manifest).unwrap();
        assert_eq!(plan.hops.len(), 2);
        assert_eq!(plan.entry().unwrap(), id(1));
        assert_eq!(plan.terminal().unwrap(), id(2));
        manifest.execution = crate::shard::manifest::Execution::Pipeline;
        assert!(RingPlan::from_manifest(&manifest).is_err());
    }

    #[test]
    fn a_job_is_consumed_once_and_only_by_its_terminal() {
        let channels = Arc::new(ResultChannels::new());
        let job = new_job_id();
        let pending = channels.register(&job, id(2)).unwrap();
        assert!(channels.register(&job, id(2)).is_err(), "duplicate job");
        assert!(channels.is_terminal(id(2)) && !channels.is_terminal(id(1)));

        let error = channels.take(&job, id(1)).unwrap_err().to_string();
        assert!(error.contains("expects its result channel"), "{error}");
        assert!(
            channels.is_pending(&job),
            "an impostor must not consume the job"
        );
        assert!(channels.take(&new_job_id(), id(2)).is_err(), "unknown job");

        assert!(channels.take(&job, id(2)).is_ok());
        assert!(!channels.is_pending(&job));
        let replay = channels.take(&job, id(2)).unwrap_err().to_string();
        assert!(replay.contains("already consumed"), "{replay}");
        drop(pending);
    }

    #[test]
    fn dropping_the_guard_removes_the_pending_job() {
        let channels = Arc::new(ResultChannels::new());
        let job = new_job_id();
        let pending = channels.register(&job, id(2)).unwrap();
        assert!(channels.is_pending(&job));
        drop(pending);
        assert!(!channels.is_pending(&job));
        assert!(!channels.is_terminal(id(2)));
    }

    #[tokio::test]
    async fn waiting_fails_once_the_registry_drops_the_sink() {
        let channels = Arc::new(ResultChannels::new());
        let job = new_job_id();
        let mut pending = channels.register(&job, id(2)).unwrap();
        drop(channels.take(&job, id(2)).unwrap());
        assert!(pending.wait().await.is_err());
    }

    #[tokio::test]
    async fn counted_streams_split_control_lines_from_other_bytes() {
        let counter = Arc::new(TrafficCounter::default());
        let lines = b"{\"kind\":\"progress\",\"event\":\"stage_start\"}\n";
        let mut reader = Counted::new(BufReader::new(&lines[..]), Arc::clone(&counter));
        assert!(matches!(
            reader.control().await.unwrap(),
            Some(RingMessage::Progress(_))
        ));
        assert_eq!(reader.control().await.unwrap(), None);
        let traffic = counter.snapshot("j", 0, true);
        assert_eq!(traffic.bytes_read, lines.len() as u64);
        assert_eq!(traffic.control_line_bytes, lines.len() as u64);
        assert_eq!(
            (traffic.control_lines, traffic.activation_bytes_read()),
            (1, 0)
        );

        let mut raw = Counted::new(BufReader::new(&[0u8; 12][..]), Arc::clone(&counter));
        let mut buffer = Vec::new();
        raw.read_to_end(&mut buffer).await.unwrap();
        assert_eq!(counter.snapshot("j", 0, true).activation_bytes_read(), 12);
    }
}
