//! `meshthing/shard-ring/1`: chained remote stages that forward activations peer to peer.
//!
//! EXPERIMENTAL, pending david-codex's production sign-off. This is the production counterpart of
//! his diagnostic `experiments/mobile-probe/iroh-core/src/ring.rs` and stays byte-compatible
//! with it, so our workers interoperate with his `mesh-ring worker` and gateway in both
//! directions.
//!
//! An origin (the leader) opens an entry stream to the first ring worker and sends
//! [`RingMessage::Start`]. Each worker starts a fresh runner for its layer range, opens the same
//! kind of stream to its configured next worker, checks the chain it reports
//! ([`validate_chain`]), and answers upstream with [`RingMessage::Ready`] listing every stage
//! from itself to the terminal. The terminal (the worker owning the final layer) instead opens a
//! result channel straight back to the origin. After that the origin sends
//! [`RingMessage::Execute`] frames, each followed by raw little-endian `f32` hidden rows; every
//! worker runs its stage and forwards its own rows to the next worker, and the terminal returns
//! a small [`RingMessage::Result`] per step on the result channel. Intermediate activations
//! never travel back to the origin; only small [`RingMessage::Progress`] lines do.
//!
//! Every control message is one JSON line of at most [`MAX_CONTROL_LINE_BYTES`] including its
//! newline. Fields this crate does not interpret (runner metrics, future additions) are kept and
//! forwarded untouched.
//!
//! The worker side is [`serve_ring_worker`]. Unlike david-codex's worker, which connects to a
//! native runner wrapper over loopback TCP, ours spawns a fresh runner per session over stdio
//! like `shard-serve`; peers see the same wire behaviour either way.

use std::path::PathBuf;
use std::process::ExitStatus;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::admission::AdmittedChild as Child;
use anyhow::{Context, bail, ensure};
use iroh::endpoint::{Connection, Incoming, RecvStream, SendStream};
use iroh::{Endpoint, EndpointId};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::io::{
    AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader,
};
use tokio::process::{ChildStdin, ChildStdout};
use tokio::sync::{Mutex, Semaphore};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::{timeout, timeout_at};
use tracing::{debug, info, warn};

use crate::rpc_bridge::{PEER_NOT_APPROVED, PEER_NOT_APPROVED_REASON};
use crate::shard::serve::{
    RUNNER_STARTUP_TIMEOUT, SESSION_ABORTED, STDERR_DRAIN_TIMEOUT, Shard, ShardServeConfig,
    forward_stderr, log_task_panic, reap,
};
use crate::shard::wire::{
    self, MAX_BATCH_TOKENS, MAX_CONTEXT, MAX_EMBD, StageInput, StageMetadata, StepOutput, StepReply,
};

/// ALPN of the ring protocol, separate from the v1 shard relay's.
pub const RING_ALPN: &[u8] = b"meshthing/shard-ring/1";

/// Bound on one control line, newline included.
pub const MAX_CONTROL_LINE_BYTES: usize = 65_536;

/// Bound on the activation payload that follows one execute frame.
pub const MAX_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

/// Most stages a ring may have; also bounds the metrics an execute frame carries.
pub const MAX_CHAIN_STAGES: usize = 16;

/// Maximum inbound connections to a ring worker, including sessions waiting on its runner.
const MAX_RING_CONNECTIONS: usize = 16;

/// How long a peer may take to send its next control line.
pub const CONTROL_LINE_TIMEOUT: Duration = Duration::from_secs(CONTROL_LINE_TIMEOUT_SECS);

/// How long a peer may take to send the activation payload after its execute frame.
pub const PAYLOAD_TIMEOUT: Duration = Duration::from_secs(PAYLOAD_TIMEOUT_SECS);

/// How long a worker waits for its runner to answer one step: the reply line, then its rows.
pub const NATIVE_STEP_TIMEOUT: Duration =
    Duration::from_secs(CONTROL_LINE_TIMEOUT_SECS + PAYLOAD_TIMEOUT_SECS);

/// How long a worker may take to connect to its next peer or to the origin.
pub const PEER_CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

/// After a clean session end, how long the next hop may take to finish its own streams.
pub const STREAM_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

/// Bounds on a ring worker's writes to its peers. A write that misses its deadline fails the
/// session like any other error: the upstream stream is reset, the downstream or result
/// connection is closed with code 1 and the session's runner is killed and reaped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingWriteTimeouts {
    /// One control line: `start` to the next worker, `ready` or a progress line upstream, the
    /// result-channel hello or a result to the origin.
    pub control_line: Duration,
    /// One execute frame plus its activations to the next worker, which reads them within
    /// [`CONTROL_LINE_TIMEOUT`] and then [`PAYLOAD_TIMEOUT`].
    pub execute: Duration,
}

impl Default for RingWriteTimeouts {
    /// The bounds a peer has to read each write: [`CONTROL_LINE_TIMEOUT`] per control line and
    /// [`NATIVE_STEP_TIMEOUT`] (control line plus payload) per execute frame.
    fn default() -> Self {
        Self {
            control_line: CONTROL_LINE_TIMEOUT,
            execute: NATIVE_STEP_TIMEOUT,
        }
    }
}

const CONTROL_LINE_TIMEOUT_SECS: u64 = 240;
const PAYLOAD_TIMEOUT_SECS: u64 = 120;

/// Native runner protocol version the ring accepts.
const NATIVE_PROTOCOL: u32 = 1;

/// Hex digits in a job id (a UUID v4 without dashes).
const JOB_ID_HEX_LEN: usize = 32;

/// Hex digits in a SHA-256 model identity.
const SHA256_HEX_LEN: usize = 64;

/// Size of one encoded `f32`.
const F32_LEN: usize = 4;

/// Size of a native request header, counted in the `request_bytes` metric like ring.rs does.
const NATIVE_HEADER_BYTES: usize = 20;

/// Close reason sent when a ring session fails.
const SESSION_FAILED_REASON: &[u8] = b"ring session failed";

/// Progress event a worker sends before running a step.
const STAGE_START_EVENT: &str = "stage_start";

/// Progress event a worker sends after running a step.
const STAGE_END_EVENT: &str = "stage_end";

/// Progress phase of step 0, which carries the prompt.
const PREFILL_PHASE: &str = "prefill";

/// Progress phase of every later step.
const DECODE_PHASE: &str = "decode";

/// Converts the `wall_ms` metric from seconds.
const MILLIS_PER_SECOND: f64 = 1000.0;

/// One shard-ring control line, tagged by its `"kind"` field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RingMessage {
    /// Origin or worker → next worker: start a session for `job`.
    Start {
        /// The session's job id.
        job: String,
    },
    /// Worker → upstream: the chain from this worker to the terminal is running.
    Ready {
        /// The session's job id.
        job: String,
        /// One entry per stage, from the answering worker to the terminal.
        stages: Vec<RingStageInfo>,
    },
    /// Origin or worker → next worker: run one step; the activation payload follows the line.
    Execute(ExecuteFrame),
    /// Worker → upstream: a small diagnostic event, relayed hop by hop to the origin. Never a
    /// result.
    Progress(Map<String, Value>),
    /// Terminal → origin: open the result channel for `job`.
    ResultChannel {
        /// The session's job id.
        job: String,
    },
    /// Origin → terminal: the result channel for `job` is accepted.
    Accepted {
        /// The session's job id.
        job: String,
    },
    /// Terminal → origin: the final output of one step.
    Result(RingResult),
}

impl RingMessage {
    /// The message's `"kind"` tag, for error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Start { .. } => "start",
            Self::Ready { .. } => "ready",
            Self::Execute(_) => "execute",
            Self::Progress(_) => "progress",
            Self::ResultChannel { .. } => "result_channel",
            Self::Accepted { .. } => "accepted",
            Self::Result(_) => "result",
        }
    }
}

/// One stage in [`RingMessage::Ready`]: the runner's metadata line plus its worker's identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RingStageInfo {
    /// The native metadata, with every extra field the runner or relay added.
    #[serde(flatten)]
    pub meta: StageMetadata,
    /// Endpoint id of the worker running this stage.
    pub node_id: String,
}

/// The JSON line of [`RingMessage::Execute`]; `payload_bytes` of raw `f32` rows follow it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecuteFrame {
    /// The session's job id.
    pub job: String,
    /// Sequence number: 0 first, then one more per frame.
    pub step: u32,
    /// Position of the first row: 0 first, then advanced by `n` per frame.
    pub pos: u32,
    /// Rows in the payload.
    pub n: u32,
    /// Payload size: `n * embd * 4`.
    pub payload_bytes: u64,
    /// Per-stage metrics appended by every worker the frame has passed.
    pub metrics: Vec<Value>,
    /// Every other field, forwarded untouched.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ExecuteFrame {
    /// The frame an origin sends for `n` rows of `embd` values at `step` and `pos`.
    pub fn new(job: &str, step: u32, pos: u32, n: u32, embd: u32) -> Self {
        Self {
            job: job.to_owned(),
            step,
            pos,
            n,
            payload_bytes: u64::from(n) * u64::from(embd) * F32_LEN as u64,
            metrics: Vec::new(),
            extra: Map::new(),
        }
    }
}

/// The JSON line of [`RingMessage::Result`]: the terminal runner's reply plus ring fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RingResult {
    /// The session's job id.
    pub job: String,
    /// The terminal runner's reply (token, piece, metrics, extra fields such as `node_id`).
    #[serde(flatten)]
    pub reply: StepReply,
    /// Every stage's metrics for this step, from the entry to the terminal.
    #[serde(default)]
    pub chain_metrics: Vec<Value>,
}

/// A fresh random job id: 32 lowercase hex digits.
pub fn new_job_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// Refuses a job id that is not 32 hex digits.
pub fn validate_job_id(job: &str) -> anyhow::Result<()> {
    ensure!(
        job.len() == JOB_ID_HEX_LEN && job.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "invalid job id {job:?}: expected {JOB_ID_HEX_LEN} hex digits"
    );
    Ok(())
}

/// Reads one control line, or `None` if the stream ends cleanly before it. Refuses a line
/// longer than [`MAX_CONTROL_LINE_BYTES`] before buffering it. Applies no timeout.
pub async fn read_control<R: AsyncBufRead + Unpin>(
    r: &mut R,
) -> anyhow::Result<Option<RingMessage>> {
    let Some(line) = read_line(r).await? else {
        return Ok(None);
    };
    serde_json::from_slice(&line)
        .context("invalid shard-ring control line")
        .map(Some)
}

/// Reads one control line within [`CONTROL_LINE_TIMEOUT`], treating end of stream as an error.
pub async fn next_control<R: AsyncBufRead + Unpin>(r: &mut R) -> anyhow::Result<RingMessage> {
    timeout(CONTROL_LINE_TIMEOUT, read_control(r))
        .await
        .context("timed out waiting for a shard-ring control line")??
        .context("ring peer closed the stream")
}

/// Writes one control line and returns its size, refusing (before writing anything) one longer
/// than [`MAX_CONTROL_LINE_BYTES`]. Applies no timeout.
pub async fn write_control<W: AsyncWrite + Unpin>(
    w: &mut W,
    msg: &RingMessage,
) -> anyhow::Result<usize> {
    let mut line = serde_json::to_vec(msg).context("failed to encode shard-ring message")?;
    line.push(b'\n');
    ensure!(
        line.len() <= MAX_CONTROL_LINE_BYTES,
        "{} line of {} bytes exceeds {MAX_CONTROL_LINE_BYTES}",
        msg.kind(),
        line.len()
    );
    w.write_all(&line)
        .await
        .with_context(|| format!("failed to send {} line", msg.kind()))?;
    w.flush()
        .await
        .with_context(|| format!("failed to flush {} line", msg.kind()))?;
    Ok(line.len())
}

/// Writes an execute frame followed by its payload, which must be `frame.payload_bytes` long.
/// Returns the payload size. Applies no timeout.
pub async fn write_execute<W: AsyncWrite + Unpin>(
    w: &mut W,
    frame: &ExecuteFrame,
    payload: &[u8],
) -> anyhow::Result<usize> {
    ensure!(
        u64::try_from(payload.len()).ok() == Some(frame.payload_bytes),
        "step {} payload is {} bytes but the frame declares {}",
        frame.step,
        payload.len(),
        frame.payload_bytes
    );
    write_control(w, &RingMessage::Execute(frame.clone())).await?;
    w.write_all(payload)
        .await
        .with_context(|| format!("failed to send step {} activations", frame.step))?;
    w.flush()
        .await
        .with_context(|| format!("failed to flush step {} activations", frame.step))?;
    Ok(payload.len())
}

/// [`write_control`] that must finish by `deadline`; `to` names the peer for the timeout error.
async fn write_control_by<W: AsyncWrite + Unpin>(
    deadline: tokio::time::Instant,
    to: &str,
    w: &mut W,
    msg: &RingMessage,
) -> anyhow::Result<usize> {
    timeout_at(deadline, write_control(w, msg))
        .await
        .with_context(|| format!("timed out sending a {} line to {to}", msg.kind()))?
}

/// [`write_execute`] that must finish by `deadline`.
async fn write_execute_by<W: AsyncWrite + Unpin>(
    deadline: tokio::time::Instant,
    w: &mut W,
    frame: &ExecuteFrame,
    payload: &[u8],
) -> anyhow::Result<usize> {
    timeout_at(deadline, write_execute(w, frame, payload))
        .await
        .with_context(|| {
            format!(
                "timed out sending step {} activations to the next ring worker",
                frame.step
            )
        })?
}

/// The deadline for a write that may take `bound` from now.
fn deadline_in(bound: Duration) -> tokio::time::Instant {
    tokio::time::Instant::now() + bound
}

/// Reads the `len`-byte activation payload of a checked frame within [`PAYLOAD_TIMEOUT`].
pub async fn read_payload<R: AsyncRead + Unpin>(r: &mut R, len: usize) -> anyhow::Result<Vec<u8>> {
    ensure!(
        len <= MAX_PAYLOAD_BYTES,
        "{len}-byte payload exceeds {MAX_PAYLOAD_BYTES}"
    );
    let mut payload = vec![0u8; len];
    timeout(PAYLOAD_TIMEOUT, r.read_exact(&mut payload))
        .await
        .context("timed out waiting for activations")?
        .context("truncated activation payload")?;
    Ok(payload)
}

/// Encodes hidden rows as the raw little-endian `f32` payload of an execute frame.
pub fn encode_activations(rows: &[f32]) -> Vec<u8> {
    rows.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// Decodes an execute frame's payload into hidden values.
pub fn decode_activations(payload: &[u8]) -> anyhow::Result<Vec<f32>> {
    let (words, rest) = payload.as_chunks::<F32_LEN>();
    ensure!(
        rest.is_empty(),
        "{}-byte payload is not whole f32 values",
        payload.len()
    );
    Ok(words.iter().map(|&word| f32::from_le_bytes(word)).collect())
}

/// Checks an execute frame against the session's state and returns its payload size.
///
/// Same rules as ring.rs's `check_frame`: the job must match; `step` and `pos` must be the next
/// expected ones (so a replayed or reordered frame fails); `1 <= n <= 128`; `pos + n <= 4096`;
/// and `payload_bytes` must equal `n * embd * 4`, at most [`MAX_PAYLOAD_BYTES`]. Also refuses a
/// frame that already carries [`MAX_CHAIN_STAGES`] metrics, which ring.rs refuses after running.
pub fn check_execute(
    frame: &ExecuteFrame,
    job: &str,
    step: u32,
    pos: u32,
    embd: u32,
) -> anyhow::Result<usize> {
    ensure!(frame.job == job, "execute frame for the wrong job");
    ensure!(
        frame.step == step && frame.pos == pos,
        "replayed/out-of-order request: got step {} at position {}, expected step {step} at \
         position {pos}",
        frame.step,
        frame.pos
    );
    let n = frame.n;
    ensure!(
        (1..=MAX_BATCH_TOKENS).contains(&n)
            && u64::from(pos) + u64::from(n) <= u64::from(MAX_CONTEXT),
        "invalid batch of {n} rows at position {pos}"
    );
    let bytes = u64::from(n) * u64::from(embd) * F32_LEN as u64;
    ensure!(
        bytes <= MAX_PAYLOAD_BYTES as u64 && frame.payload_bytes == bytes,
        "invalid activation size: {} bytes declared, {bytes} expected",
        frame.payload_bytes
    );
    ensure!(
        frame.metrics.len() < MAX_CHAIN_STAGES,
        "chain too long: frame already carries {} metrics",
        frame.metrics.len()
    );
    usize::try_from(bytes).context("activation size overflows")
}

/// Checks one stage's metadata the way ring.rs's `validate_info` does: native protocol 1, a
/// nonempty range within the model, a supported hidden size and a 64-hex model identity.
pub fn validate_stage(meta: &StageMetadata) -> anyhow::Result<()> {
    ensure!(
        meta.protocol == NATIVE_PROTOCOL,
        "wrong native protocol {}",
        meta.protocol
    );
    ensure!(
        meta.begin < meta.end && meta.end <= meta.layers,
        "invalid range {}:{} of {} layers",
        meta.begin,
        meta.end,
        meta.layers
    );
    ensure!(
        (1..=MAX_EMBD).contains(&meta.embd),
        "invalid hidden dimension {}",
        meta.embd
    );
    meta.validate_vocabulary()?;
    let hash = &meta.source_sha256;
    ensure!(
        hash.len() == SHA256_HEX_LEN && hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "invalid model identity {hash:?}: expected {SHA256_HEX_LEN} hex digits"
    );
    Ok(())
}

/// Checks a ready chain: 1 to [`MAX_CHAIN_STAGES`] stages, each valid ([`validate_stage`]) with
/// a parseable `node_id`, contiguous ranges, and the same `layers`, `embd`, `vocab` and
/// `source_sha256` throughout.
pub fn validate_chain(stages: &[RingStageInfo]) -> anyhow::Result<()> {
    let first = stages.first().context("empty ring chain")?;
    ensure!(
        stages.len() <= MAX_CHAIN_STAGES,
        "ring chain of {} stages exceeds {MAX_CHAIN_STAGES}",
        stages.len()
    );
    for stage in stages {
        validate_stage(&stage.meta)
            .with_context(|| format!("invalid ring stage from {}", stage.node_id))?;
        stage
            .node_id
            .parse::<EndpointId>()
            .with_context(|| format!("invalid ring stage node id {:?}", stage.node_id))?;
        check_same_model(&first.meta, &stage.meta)?;
    }
    for pair in stages.windows(2) {
        let (before, after) = (&pair[0].meta, &pair[1].meta);
        ensure!(
            before.end == after.begin,
            "noncontiguous ranges: {}:{} then {}:{}",
            before.begin,
            before.end,
            after.begin,
            after.end
        );
    }
    Ok(())
}

fn check_same_model(first: &StageMetadata, other: &StageMetadata) -> anyhow::Result<()> {
    let same = [
        ("layers", first.layers == other.layers),
        ("embd", first.embd == other.embd),
        ("vocab", first.vocab == other.vocab),
        ("source_sha256", first.source_sha256 == other.source_sha256),
    ];
    for (key, equal) in same {
        ensure!(equal, "model mismatch: {key}");
    }
    Ok(())
}

/// A started session toward a ring peer: the stream it answered `ready` on and its chain.
pub struct RingStream {
    /// The connection the stream belongs to; dropping every handle closes it.
    pub connection: Connection,
    /// Where execute frames go.
    pub send: SendStream,
    /// Where progress lines come from, after `ready`.
    pub recv: BufReader<RecvStream>,
    /// The peer's `ready.stages`, not yet validated.
    pub stages: Vec<RingStageInfo>,
}

/// Connects to `peer` on [`RING_ALPN`], sends `start` for `job` and waits for its `ready`.
///
/// Used by a worker for its next hop and by an origin for the entry worker; the caller
/// validates the returned stages ([`validate_chain`] plus its own expectations). The `start`
/// line must be sent within [`CONTROL_LINE_TIMEOUT`]. On failure the connection is closed with
/// code 1 ("ring session failed").
pub async fn open_ring_session(
    endpoint: &Endpoint,
    peer: EndpointId,
    job: &str,
) -> anyhow::Result<RingStream> {
    open_ring_session_by(endpoint, peer, job, CONTROL_LINE_TIMEOUT).await
}

/// [`open_ring_session`] whose `start` line must be sent within `write_timeout`.
async fn open_ring_session_by(
    endpoint: &Endpoint,
    peer: EndpointId,
    job: &str,
    write_timeout: Duration,
) -> anyhow::Result<RingStream> {
    let connection = connect(endpoint, peer).await?;
    let started = async {
        let (mut send, recv) = connection
            .open_bi()
            .await
            .with_context(|| format!("failed to open a ring stream to {peer}"))?;
        let start = RingMessage::Start {
            job: job.to_owned(),
        };
        let to = format!("ring peer {peer}");
        write_control_by(deadline_in(write_timeout), &to, &mut send, &start).await?;
        let mut recv = BufReader::new(recv);
        let stages = match next_control(&mut recv)
            .await
            .with_context(|| format!("waiting for ring readiness from {peer}"))?
        {
            RingMessage::Ready { job: ready, stages } if ready == job => stages,
            other => bail!("wrong ring readiness from {peer}: got {}", other.kind()),
        };
        Ok((send, recv, stages))
    }
    .await;
    match started {
        Ok((send, recv, stages)) => Ok(RingStream {
            connection,
            send,
            recv,
            stages,
        }),
        Err(error) => {
            connection.close(SESSION_ABORTED, SESSION_FAILED_REASON);
            Err(error)
        }
    }
}

/// Origin: reads the opening line of a result-channel stream the terminal opened and returns
/// its (validated) job id. The caller checks the terminal's identity and that the job is
/// pending, then answers with [`accept_result_channel`].
pub async fn read_result_channel_hello<R: AsyncBufRead + Unpin>(
    recv: &mut R,
) -> anyhow::Result<String> {
    match next_control(recv).await? {
        RingMessage::ResultChannel { job } => {
            validate_job_id(&job)?;
            Ok(job)
        }
        other => bail!("wrong result channel opening: got {}", other.kind()),
    }
}

/// Origin: accepts a result channel for `job` and finishes the send side, as ring.rs's gateway
/// does. The `accepted` line must be sent within [`CONTROL_LINE_TIMEOUT`]. Results then arrive
/// on the stream's receive side ([`read_result`]).
pub async fn accept_result_channel(send: &mut SendStream, job: &str) -> anyhow::Result<()> {
    let accepted = RingMessage::Accepted {
        job: job.to_owned(),
    };
    let deadline = deadline_in(CONTROL_LINE_TIMEOUT);
    write_control_by(deadline, "the ring terminal", send, &accepted).await?;
    send.finish().context("failed to finish the result channel")
}

/// Origin: reads the result for `step` of `job` within [`CONTROL_LINE_TIMEOUT`], refusing
/// anything but a final result (zero floats) for exactly that step.
pub async fn read_result<R: AsyncBufRead + Unpin>(
    recv: &mut R,
    job: &str,
    step: u32,
) -> anyhow::Result<RingResult> {
    let result = match next_control(recv).await? {
        RingMessage::Result(result) => result,
        other => bail!(
            "expected a result on the result channel, got {}",
            other.kind()
        ),
    };
    ensure!(
        result.job == job && result.reply.step == step && result.reply.n_floats == 0,
        "invalid final result: job {:?} step {} with {} floats, expected step {step}",
        result.job,
        result.reply.step,
        result.reply.n_floats
    );
    result.reply.piece_bytes()?;
    Ok(result)
}

async fn connect(endpoint: &Endpoint, peer: EndpointId) -> anyhow::Result<Connection> {
    timeout(PEER_CONNECT_TIMEOUT, endpoint.connect(peer, RING_ALPN))
        .await
        .with_context(|| format!("timed out connecting to ring peer {peer}"))?
        .with_context(|| format!("failed to connect to ring peer {peer}"))
}

/// Reads one newline-terminated line of at most [`MAX_CONTROL_LINE_BYTES`] (newline included),
/// or `None` at a clean end of stream.
async fn read_line<R: AsyncBufRead + Unpin>(r: &mut R) -> anyhow::Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let available = r
            .fill_buf()
            .await
            .context("failed to read shard-ring control line")?;
        if available.is_empty() {
            ensure!(
                line.is_empty(),
                "stream closed in the middle of a control line"
            );
            return Ok(None);
        }
        let count = available
            .iter()
            .position(|&byte| byte == b'\n')
            .map_or(available.len(), |index| index + 1);
        ensure!(
            line.len() + count <= MAX_CONTROL_LINE_BYTES,
            "control line exceeds {MAX_CONTROL_LINE_BYTES} bytes"
        );
        line.extend_from_slice(&available[..count]);
        r.consume(count);
        if line.last() == Some(&b'\n') {
            return Ok(Some(line));
        }
    }
}

/// What a ring worker runs and where it sits in the ring.
#[derive(Clone, Debug)]
pub struct RingWorkerConfig {
    /// The `shard_runner` binary.
    pub runner: PathBuf,
    /// The exported shard file passed to the runner.
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
    /// Original model identity the runner must report, if any.
    pub source_sha256: Option<String>,
    /// The only peer allowed to start sessions: the origin for the entry worker, else the
    /// previous worker.
    pub upstream: EndpointId,
    /// The origin the terminal opens its result channel to.
    pub origin: EndpointId,
    /// The next worker; `None` makes this worker the terminal.
    pub next: Option<EndpointId>,
}

impl RingWorkerConfig {
    fn shard_config(&self) -> ShardServeConfig {
        ShardServeConfig {
            runner: self.runner.clone(),
            model: self.model.clone(),
            begin: self.begin,
            end: self.end,
            context: self.context,
            gpu_layers: self.gpu_layers,
            budget_bytes: self.budget_bytes,
            expected_sha256: self.expected_sha256.clone(),
            source_sha256: self.source_sha256.clone(),
        }
    }
}

/// Serves `config`'s stage as a shard-ring worker until the endpoint closes, then returns
/// `Ok(())`. The endpoint must accept [`RING_ALPN`].
///
/// The model file must hash to `expected_sha256`, or this fails before accepting anything.
/// Connections from any peer but `upstream` (or on another ALPN) are closed with
/// [`PEER_NOT_APPROVED`]. Each stream is one session with a fresh runner. Like ring.rs, one
/// session runs at a time: streams on one connection are served in turn, and a stream that
/// arrives on another connection while a session is running is dropped unanswered (its peer
/// sees the stream end without `ready`). A runner's metadata must match the configured range
/// and carry a 64-hex model identity, and the terminal must own the model's final layer.
///
/// Every write to a peer is bounded by [`RingWriteTimeouts::default`]; a write that misses its
/// deadline fails the session.
///
/// Cancel-safe: dropping the future aborts every session and kills every runner.
pub async fn serve_ring_worker(endpoint: Endpoint, config: RingWorkerConfig) -> anyhow::Result<()> {
    serve_ring_worker_with(endpoint, config, RingWriteTimeouts::default()).await
}

/// [`serve_ring_worker`] with explicit bounds on the worker's writes to its peers.
pub async fn serve_ring_worker_with(
    endpoint: Endpoint,
    config: RingWorkerConfig,
    timeouts: RingWriteTimeouts,
) -> anyhow::Result<()> {
    let shard = Shard::load(config.shard_config()).await?;
    info!(
        id = %endpoint.id(),
        begin = config.begin,
        end = config.end,
        model_sha256 = %shard.model_sha256(),
        upstream = %config.upstream,
        origin = %config.origin,
        next = ?config.next,
        "shard-ring worker serving (experimental)"
    );
    let worker = Arc::new(RingWorker {
        endpoint: endpoint.clone(),
        shard,
        upstream: config.upstream,
        origin: config.origin,
        next: config.next,
        gate: Semaphore::new(1),
        timeouts,
    });

    let connection_slots = Arc::new(Semaphore::new(MAX_RING_CONNECTIONS));
    let mut connections = JoinSet::new();
    loop {
        tokio::select! {
            incoming = endpoint.accept() => {
                let Some(incoming) = incoming else {
                    info!("endpoint closed; shard-ring worker stopping");
                    return Ok(());
                };
                let Ok(permit) = connection_slots.clone().try_acquire_owned() else {
                    incoming.refuse();
                    debug!(limit = MAX_RING_CONNECTIONS, "shard-ring connection limit reached; refusing connection");
                    continue;
                };
                let worker = Arc::clone(&worker);
                connections.spawn(async move {
                    let _permit = permit;
                    serve_connection(incoming, worker).await;
                });
            }
            Some(joined) = connections.join_next(), if !connections.is_empty() => {
                log_task_panic(joined);
            }
        }
    }
}

/// A serving ring worker: its stage, its place in the ring, the one-session gate and the bounds
/// on its writes.
struct RingWorker {
    endpoint: Endpoint,
    shard: Shard,
    upstream: EndpointId,
    origin: EndpointId,
    next: Option<EndpointId>,
    gate: Semaphore,
    timeouts: RingWriteTimeouts,
}

/// One Iroh connection: refuse anyone but the upstream, then serve its streams in turn.
async fn serve_connection(incoming: Incoming, worker: Arc<RingWorker>) {
    let connection = match incoming.await {
        Ok(connection) => connection,
        Err(error) => {
            warn!(%error, "incoming shard-ring connection failed");
            return;
        }
    };
    let peer = connection.remote_id();
    if peer != worker.upstream || connection.alpn() != RING_ALPN {
        warn!(%peer, alpn = %String::from_utf8_lossy(connection.alpn()), "rejecting ring peer");
        connection.close(PEER_NOT_APPROVED, PEER_NOT_APPROVED_REASON);
        return;
    }
    loop {
        let (send, recv) = match connection.accept_bi().await {
            Ok(stream) => stream,
            Err(error) => {
                info!(%peer, %error, "ring connection closed");
                return;
            }
        };
        let Ok(_permit) = worker.gate.try_acquire() else {
            warn!(%peer, "dropping a concurrent ring session: one session per worker");
            continue;
        };
        worker.run_session(send, recv).await;
    }
}

/// Byte counters for a session's summary line.
#[derive(Default)]
struct SessionStats {
    job: String,
    activation_in: usize,
    activation_out: usize,
    token_reply_bytes: usize,
    steps: u32,
}

/// The stream back to the upstream peer, shared with the task relaying downstream progress.
type Upstream = Arc<Mutex<SendStream>>;

/// Writes one control line upstream by `deadline`, which also bounds waiting for the stream
/// while the other writer (the progress relay or the session) holds it.
async fn write_upstream(
    deadline: tokio::time::Instant,
    writer: &Upstream,
    msg: &RingMessage,
) -> anyhow::Result<usize> {
    let write = async { write_control(&mut *writer.lock().await, msg).await };
    timeout_at(deadline, write)
        .await
        .with_context(|| format!("timed out sending a {} line upstream", msg.kind()))?
}

impl RingWorker {
    /// Runs one session and reports how it ended: finished upstream stream on success, reset
    /// streams on failure, plus a `RING ...` summary line either way.
    async fn run_session(&self, send: SendStream, recv: RecvStream) {
        let mut recv = BufReader::new(recv);
        let upstream: Upstream = Arc::new(Mutex::new(send));
        let mut stats = SessionStats::default();
        let started = Instant::now();
        let result = self.session(&mut recv, &upstream, &mut stats).await;
        let mut send = upstream.lock().await;
        let next = self
            .next
            .map_or_else(|| "-".to_owned(), |next| next.to_string());
        let summary = format!(
            "RING job={} node={} next={next} origin={} activation_in={} activation_out={} \
             token_reply_bytes={} steps={}",
            stats.job,
            self.endpoint.id(),
            self.origin,
            stats.activation_in,
            stats.activation_out,
            stats.token_reply_bytes,
            stats.steps
        );
        let duration_ms = started.elapsed().as_millis();
        match result {
            Ok(()) => {
                // Fails only if the stream is already closed, in which case nothing is left to do.
                let _ = send.finish();
                info!(duration_ms, "{summary}");
            }
            Err(error) => {
                // Both calls only fail when that half is already closed, which is what we want.
                let _ = send.reset(SESSION_ABORTED);
                let _ = recv.get_mut().stop(SESSION_ABORTED);
                warn!(duration_ms, error = %format!("{error:#}"), "{summary}");
            }
        }
    }

    /// Reads `start`, runs a fresh runner for the session and always reaps it.
    async fn session(
        &self,
        upstream: &mut BufReader<RecvStream>,
        writer: &Upstream,
        stats: &mut SessionStats,
    ) -> anyhow::Result<()> {
        let job = match next_control(upstream).await? {
            RingMessage::Start { job } => job,
            other => bail!("missing ring start: got {}", other.kind()),
        };
        validate_job_id(&job)?;
        stats.job.clone_from(&job);
        let mut runner = NativeRunner::spawn(&self.shard).await?;
        let outcome: anyhow::Result<()> = async {
            let meta = runner.start(&self.shard).await?;
            let mut hop = self.ready_hop(job, meta, writer).await?;
            let outcome = hop.execute_all(upstream, &mut runner, stats).await;
            let closed = hop.downstream.close(outcome.is_ok()).await;
            outcome.and(closed)
        }
        .await;
        let status = runner.close(outcome.is_ok()).await;
        outcome?;
        let status = status?;
        if !status.success() {
            warn!(%status, "shard runner exited unsuccessfully");
        }
        Ok(())
    }

    /// Checks our stage, sets up the next hop (or the result channel) and sends `ready`.
    async fn ready_hop(
        &self,
        job: String,
        meta: StageMetadata,
        writer: &Upstream,
    ) -> anyhow::Result<Hop> {
        validate_stage(&meta).context("invalid runner metadata")?;
        let info = RingStageInfo {
            meta,
            node_id: self.endpoint.id().to_string(),
        };
        let (downstream, stages) = self.link_downstream(&job, &info, writer).await?;
        let ready = RingMessage::Ready {
            job: job.clone(),
            stages,
        };
        let deadline = deadline_in(self.timeouts.control_line);
        if let Err(error) = write_upstream(deadline, writer, &ready).await {
            downstream.close(false).await?;
            return Err(error);
        }
        Ok(Hop {
            job,
            info,
            writer: Arc::clone(writer),
            downstream,
            timeouts: self.timeouts,
            step: 0,
            pos: 0,
        })
    }

    /// Links to the next worker and validates the whole chain from us, or (on the terminal)
    /// opens the result channel. Returns the link and the chain to announce as ready.
    async fn link_downstream(
        &self,
        job: &str,
        info: &RingStageInfo,
        writer: &Upstream,
    ) -> anyhow::Result<(Downstream, Vec<RingStageInfo>)> {
        let mut stages = vec![info.clone()];
        let Some(next) = self.next else {
            let meta = &info.meta;
            ensure!(
                meta.is_final(),
                "terminal does not own the final layer: runs {}:{} of {}",
                meta.begin,
                meta.end,
                meta.layers
            );
            let write_timeout = self.timeouts.control_line;
            let downstream =
                Downstream::origin(&self.endpoint, self.origin, job, write_timeout).await?;
            return Ok((downstream, stages));
        };
        let write_timeout = self.timeouts.control_line;
        let (downstream, rest) =
            Downstream::next(&self.endpoint, next, job, writer, write_timeout).await?;
        stages.extend(rest);
        if let Err(error) = validate_chain(&stages) {
            downstream.close(false).await?;
            return Err(error.context("invalid downstream chain"));
        }
        Ok((downstream, stages))
    }
}

/// A runner process started for one session.
struct NativeRunner {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Option<BufReader<ChildStdout>>,
    logs: JoinSet<()>,
}

impl NativeRunner {
    /// Spawns a fresh runner with its stderr forwarded to the log.
    async fn spawn(shard: &Shard) -> anyhow::Result<Self> {
        let mut child = shard.spawn_runner().await?;
        let mut logs = JoinSet::new();
        if let Some(stderr) = child.stderr.take() {
            logs.spawn(forward_stderr(stderr, child.id()));
        }
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().map(BufReader::new);
        Ok(Self {
            child,
            stdin,
            stdout,
            logs,
        })
    }

    /// Reads the runner's metadata line and checks and stamps it like `shard-serve` does.
    async fn start(&mut self, shard: &Shard) -> anyhow::Result<StageMetadata> {
        let stdout = self.stdout.as_mut().context("shard runner has no stdout")?;
        let mut meta = timeout(RUNNER_STARTUP_TIMEOUT, wire::read_metadata(stdout))
            .await
            .context("timed out waiting for the shard runner to start")??;
        shard.stamp_metadata(&mut meta)?;
        Ok(meta)
    }

    /// Runs one opcode-2 step and returns the reply, checking its step and that it carries
    /// exactly `expected_floats` values.
    async fn step(
        &mut self,
        frame: &ExecuteFrame,
        rows: &[f32],
        embd: u32,
        expected_floats: usize,
    ) -> anyhow::Result<StepOutput> {
        let stdin = self.stdin.as_mut().context("shard runner has no stdin")?;
        let stdout = self.stdout.as_mut().context("shard runner has no stdout")?;
        let input = StageInput::Hidden { rows, embd };
        let exchange = async {
            wire::write_request(stdin, frame.step, frame.pos, &input).await?;
            wire::read_step(stdout, expected_floats).await
        };
        let output = timeout(NATIVE_STEP_TIMEOUT, exchange)
            .await
            .with_context(|| format!("timed out waiting for the runner's step {}", frame.step))??;
        ensure!(
            output.reply.step == frame.step,
            "wrong native step: runner answered {} for {}",
            output.reply.step,
            frame.step
        );
        ensure!(
            output.reply.n_floats == expected_floats,
            "wrong native output shape: {} floats, expected {expected_floats}",
            output.reply.n_floats
        );
        Ok(output)
    }

    /// Closes the runner's input, reaps it (killing it unless the session ended cleanly) and
    /// collects its last stderr lines.
    async fn close(mut self, ended_cleanly: bool) -> anyhow::Result<ExitStatus> {
        drop(self.stdin.take());
        let status = reap(&mut self.child, ended_cleanly).await;
        // Best effort: a grandchild may keep stderr open, and the logs are only diagnostics.
        let _ = timeout(STDERR_DRAIN_TIMEOUT, self.logs.join_next()).await;
        status
    }
}

/// Where a worker's output goes: the next worker, or (for the terminal) the origin.
enum Downstream {
    /// A middle worker's link to the next worker, plus the task relaying its progress upstream.
    Next {
        connection: Connection,
        send: SendStream,
        relay: RelayTask,
    },
    /// The terminal's result channel to the origin.
    Origin {
        connection: Connection,
        send: SendStream,
    },
}

impl Downstream {
    /// Starts the session on `next`, checks it reports our next worker first, and starts
    /// relaying its progress upstream. Returns the link and the rest of the chain.
    /// Every control line (`start`, relayed progress) must be sent within `write_timeout`.
    async fn next(
        endpoint: &Endpoint,
        next: EndpointId,
        job: &str,
        writer: &Upstream,
        write_timeout: Duration,
    ) -> anyhow::Result<(Self, Vec<RingStageInfo>)> {
        let stream = open_ring_session_by(endpoint, next, job, write_timeout).await?;
        let first = stream.stages.first().map(|stage| stage.node_id.clone());
        if first != Some(next.to_string()) {
            stream
                .connection
                .close(SESSION_ABORTED, SESSION_FAILED_REASON);
            bail!("downstream chain starts at {first:?}, not the configured next worker {next}");
        }
        let relay = RelayTask(tokio::spawn(relay_progress(
            stream.recv,
            Arc::clone(writer),
            write_timeout,
        )));
        let link = Self::Next {
            connection: stream.connection,
            send: stream.send,
            relay,
        };
        Ok((link, stream.stages))
    }

    /// Opens the result channel to the origin and waits for it to be accepted. The hello must be
    /// sent within `write_timeout`; on failure the connection is closed with code 1.
    async fn origin(
        endpoint: &Endpoint,
        origin: EndpointId,
        job: &str,
        write_timeout: Duration,
    ) -> anyhow::Result<Self> {
        let connection = connect(endpoint, origin).await?;
        let opened = async {
            let (mut send, recv) = connection
                .open_bi()
                .await
                .context("failed to open the result channel")?;
            let hello = RingMessage::ResultChannel {
                job: job.to_owned(),
            };
            let deadline = deadline_in(write_timeout);
            write_control_by(deadline, "the origin", &mut send, &hello).await?;
            let mut recv = BufReader::new(recv);
            match next_control(&mut recv)
                .await
                .context("waiting for the origin to accept the result channel")?
            {
                RingMessage::Accepted { job: accepted } if accepted == job => Ok(send),
                other => bail!("result channel not accepted: got {}", other.kind()),
            }
        }
        .await;
        match opened {
            // Like ring.rs, nothing more is read: the origin finishes its side after `accepted`.
            Ok(send) => Ok(Self::Origin { connection, send }),
            Err(error) => {
                connection.close(SESSION_ABORTED, SESSION_FAILED_REASON);
                Err(error)
            }
        }
    }

    /// Resolves only if the next worker ends its stream to us, which mid-session means the
    /// chain failed; never resolves for the terminal.
    async fn failed(&mut self) -> anyhow::Error {
        match self {
            Self::Next { relay, .. } => match relay.wait().await {
                Ok(()) => anyhow::anyhow!("next ring worker ended the session early"),
                Err(error) => error.context("next ring worker failed"),
            },
            Self::Origin { .. } => std::future::pending().await,
        }
    }

    /// Ends the link: on success, finishes our side and waits (bounded) for the peer to finish
    /// or acknowledge it, so nothing in flight is lost when the connection drops; on failure,
    /// closes the connection so the peer sees an error rather than a clean end.
    async fn close(self, success: bool) -> anyhow::Result<()> {
        let (connection, mut send) = match self {
            Self::Next {
                connection,
                mut send,
                mut relay,
            } => {
                if success {
                    send.finish()
                        .context("failed to finish the downstream stream")?;
                    return timeout(STREAM_DRAIN_TIMEOUT, relay.wait())
                        .await
                        .context("next ring worker did not finish its stream")?
                        .context("next ring worker failed while finishing");
                }
                (connection, send)
            }
            Self::Origin {
                connection,
                mut send,
            } => {
                if success {
                    send.finish()
                        .context("failed to finish the result channel")?;
                    // The origin's acknowledgement or stop both mean nothing is left in flight.
                    let _ = timeout(STREAM_DRAIN_TIMEOUT, send.stopped()).await;
                    return Ok(());
                }
                (connection, send)
            }
        };
        // Fails only if the stream is already closed; closing the connection covers it anyway.
        let _ = send.reset(SESSION_ABORTED);
        connection.close(SESSION_ABORTED, SESSION_FAILED_REASON);
        Ok(())
    }
}

/// The progress relay task; aborted when dropped.
struct RelayTask(JoinHandle<anyhow::Result<()>>);

impl RelayTask {
    /// Waits for the relay to end and returns how it ended.
    async fn wait(&mut self) -> anyhow::Result<()> {
        match (&mut self.0).await {
            Ok(result) => result,
            Err(join_error) => Err(anyhow::Error::new(join_error).context("progress relay died")),
        }
    }
}

impl Drop for RelayTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Copies the next worker's progress lines upstream until its stream ends cleanly. Anything
/// but progress fails the relay, and so does a line not sent upstream within `write_timeout`.
/// No read timeout applies: the next worker is legitimately silent while the origin is.
async fn relay_progress(
    mut recv: BufReader<RecvStream>,
    writer: Upstream,
    write_timeout: Duration,
) -> anyhow::Result<()> {
    while let Some(message) = read_control(&mut recv).await? {
        ensure!(
            matches!(message, RingMessage::Progress(_)),
            "unexpected downstream {} line",
            message.kind()
        );
        write_upstream(deadline_in(write_timeout), &writer, &message)
            .await
            .context("failed to relay progress")?;
    }
    Ok(())
}

/// One running session on a ready worker.
struct Hop {
    job: String,
    info: RingStageInfo,
    writer: Upstream,
    downstream: Downstream,
    timeouts: RingWriteTimeouts,
    step: u32,
    pos: u32,
}

impl Hop {
    /// Serves execute frames until the upstream finishes its stream.
    async fn execute_all(
        &mut self,
        upstream: &mut BufReader<RecvStream>,
        runner: &mut NativeRunner,
        stats: &mut SessionStats,
    ) -> anyhow::Result<()> {
        loop {
            let frame = tokio::select! {
                frame = read_frame(upstream) => frame?,
                error = self.downstream.failed() => return Err(error),
            };
            let Some(frame) = frame else {
                return Ok(());
            };
            let embd = self.info.meta.embd;
            let size = check_execute(&frame, &self.job, self.step, self.pos, embd)?;
            let payload = read_payload(upstream, size).await?;
            stats.activation_in += size;
            let rows = frame.n;
            self.execute(frame, &payload, runner, stats).await?;
            self.step += 1;
            self.pos += rows;
            stats.steps = self.step;
        }
    }

    /// Runs one checked frame through the runner and passes its output on.
    async fn execute(
        &mut self,
        mut frame: ExecuteFrame,
        payload: &[u8],
        runner: &mut NativeRunner,
        stats: &mut SessionStats,
    ) -> anyhow::Result<()> {
        let progress = StepProgress {
            begin: self.info.meta.begin,
            step: frame.step,
            input_bytes: payload.len(),
        };
        self.progress(progress.started()).await?;
        let started = Instant::now();
        let rows = decode_activations(payload)?;
        let expected_floats = match self.downstream {
            Downstream::Next { .. } => rows.len(),
            Downstream::Origin { .. } => 0,
        };
        let embd = self.info.meta.embd;
        let StepOutput { mut reply, floats } =
            runner.step(&frame, &rows, embd, expected_floats).await?;
        let hidden = encode_activations(&floats);
        let measured = [
            (
                "wall_ms",
                Value::from(started.elapsed().as_secs_f64() * MILLIS_PER_SECOND),
            ),
            (
                "request_bytes",
                Value::from(NATIVE_HEADER_BYTES + payload.len()),
            ),
            ("response_tensor_bytes", Value::from(hidden.len())),
            ("node_id", Value::from(self.info.node_id.clone())),
        ];
        reply
            .extra
            .extend(measured.map(|(key, value)| (key.to_owned(), value)));
        let metric = serde_json::to_value(&reply).context("failed to encode stage metrics")?;
        frame.metrics.push(metric.clone());
        self.progress(progress.ended(hidden.len(), metric)).await?;
        self.pass_on(frame, reply, &hidden, stats).await
    }

    /// Forwards the frame and our rows to the next worker, or sends the terminal's result.
    async fn pass_on(
        &mut self,
        mut frame: ExecuteFrame,
        reply: StepReply,
        hidden: &[u8],
        stats: &mut SessionStats,
    ) -> anyhow::Result<()> {
        match &mut self.downstream {
            Downstream::Next { send, .. } => {
                frame.payload_bytes = u64::try_from(hidden.len()).context("payload too large")?;
                let deadline = deadline_in(self.timeouts.execute);
                stats.activation_out += write_execute_by(deadline, send, &frame, hidden).await?;
            }
            Downstream::Origin { send, .. } => {
                let result = RingMessage::Result(RingResult {
                    job: self.job.clone(),
                    reply,
                    chain_metrics: frame.metrics,
                });
                let deadline = deadline_in(self.timeouts.control_line);
                stats.token_reply_bytes +=
                    write_control_by(deadline, "the origin", send, &result).await?;
            }
        }
        Ok(())
    }

    async fn progress(&self, message: RingMessage) -> anyhow::Result<()> {
        let deadline = deadline_in(self.timeouts.control_line);
        write_upstream(deadline, &self.writer, &message).await?;
        Ok(())
    }
}

/// Reads the next execute frame within [`CONTROL_LINE_TIMEOUT`], or `None` once the upstream
/// finishes its stream.
async fn read_frame(upstream: &mut BufReader<RecvStream>) -> anyhow::Result<Option<ExecuteFrame>> {
    let message = timeout(CONTROL_LINE_TIMEOUT, read_control(upstream))
        .await
        .context("timed out waiting for the next execute frame")??;
    match message {
        None => Ok(None),
        Some(RingMessage::Execute(frame)) => Ok(Some(frame)),
        Some(other) => bail!("expected an execute frame, got {}", other.kind()),
    }
}

/// The step a worker's progress lines report on, identified like ring.rs does.
struct StepProgress {
    /// First layer of the reporting stage.
    begin: u32,
    step: u32,
    input_bytes: usize,
}

impl StepProgress {
    /// The line sent before running the step.
    fn started(&self) -> RingMessage {
        let phase = if self.step == 0 {
            PREFILL_PHASE
        } else {
            DECODE_PHASE
        };
        progress_message([
            ("event", Value::from(STAGE_START_EVENT)),
            ("begin", Value::from(self.begin)),
            ("step", Value::from(self.step)),
            ("phase", Value::from(phase)),
            ("input_bytes", Value::from(self.input_bytes)),
        ])
    }

    /// The line sent after running the step, carrying the stage's metrics.
    fn ended(&self, output_bytes: usize, result: Value) -> RingMessage {
        progress_message([
            ("event", Value::from(STAGE_END_EVENT)),
            ("begin", Value::from(self.begin)),
            ("step", Value::from(self.step)),
            ("input_bytes", Value::from(self.input_bytes)),
            ("output_bytes", Value::from(output_bytes)),
            ("result", result),
        ])
    }
}

fn progress_message<const N: usize>(fields: [(&str, Value); N]) -> RingMessage {
    RingMessage::Progress(
        fields
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const EMBD: u32 = 5120;
    const JOB: &str = "0123456789abcdef0123456789abcdef";

    fn frame(job: &str, step: u32, pos: u32, n: u32, payload_bytes: u64) -> ExecuteFrame {
        ExecuteFrame {
            job: job.to_owned(),
            step,
            pos,
            n,
            payload_bytes,
            metrics: Vec::new(),
            extra: Map::new(),
        }
    }

    fn node_id() -> String {
        iroh::SecretKey::generate().public().to_string()
    }

    fn stage(begin: u32, end: u32) -> RingStageInfo {
        RingStageInfo {
            meta: StageMetadata {
                protocol: 1,
                begin,
                end,
                layers: 64,
                embd: EMBD,
                vocab: 152_064,
                allocated_bytes: 0,
                source_sha256: "ab".repeat(32),
                tokens: Vec::new(),
                model_sha256: None,
                artifact_sha256: None,
                runner_sha256: None,
                extra: Map::new(),
            },
            node_id: node_id(),
        }
    }

    fn decode(text: &str) -> RingMessage {
        serde_json::from_str(text).unwrap()
    }

    /// Serializes `message` and checks it is the same JSON object as `expected`.
    fn assert_round_trips(message: &RingMessage, expected: &Value) {
        assert_eq!(&serde_json::to_value(message).unwrap(), expected);
        let reparsed: RingMessage = serde_json::from_value(expected.clone()).unwrap();
        assert_eq!(&reparsed, message);
    }

    #[test]
    fn rejects_replay_and_wrong_payload() {
        // ring.rs's own test, case for case.
        let ok = frame("j", 0, 0, 1, 20_480);
        assert_eq!(check_execute(&ok, "j", 0, 0, EMBD).unwrap(), 20_480);
        assert!(check_execute(&ok, "j", 1, 0, EMBD).is_err());
        assert!(check_execute(&ok, "j", 0, 0, 4096).is_err());
        assert!(check_execute(&ok, "other", 0, 0, EMBD).is_err());
    }

    #[test]
    fn check_execute_enforces_order_and_bounds() {
        let size = |n: u32| u64::from(n) * u64::from(EMBD) * 4;
        assert!(check_execute(&frame(JOB, 3, 40, 1, size(1)), JOB, 3, 40, EMBD).is_ok());
        let error = check_execute(&frame(JOB, 3, 39, 1, size(1)), JOB, 3, 40, EMBD).unwrap_err();
        assert!(error.to_string().contains("out-of-order"), "{error:#}");
        // Batch size bounds.
        assert!(check_execute(&frame(JOB, 0, 0, 0, 0), JOB, 0, 0, EMBD).is_err());
        assert!(check_execute(&frame(JOB, 0, 0, 128, size(128)), JOB, 0, 0, EMBD).is_ok());
        assert!(check_execute(&frame(JOB, 0, 0, 129, size(129)), JOB, 0, 0, EMBD).is_err());
        // The context bound: pos + n <= 4096.
        assert!(check_execute(&frame(JOB, 1, 4095, 1, size(1)), JOB, 1, 4095, EMBD).is_ok());
        assert!(check_execute(&frame(JOB, 1, 4095, 2, size(2)), JOB, 1, 4095, EMBD).is_err());
        assert!(
            check_execute(&frame(JOB, 1, u32::MAX, 1, size(1)), JOB, 1, u32::MAX, EMBD).is_err()
        );
        // The payload bound: exactly 16 MiB is allowed, one row wider is not.
        let max = frame(JOB, 0, 0, 128, MAX_PAYLOAD_BYTES as u64);
        assert!(check_execute(&max, JOB, 0, 0, MAX_EMBD).is_ok());
        let wide = 128 * u64::from(MAX_EMBD + 1) * 4;
        assert!(check_execute(&frame(JOB, 0, 0, 128, wide), JOB, 0, 0, MAX_EMBD + 1).is_err());
        assert!(check_execute(&frame(JOB, 0, 0, 1, size(1) + 4), JOB, 0, 0, EMBD).is_err());
    }

    #[test]
    fn check_execute_refuses_a_full_metrics_chain() {
        let mut full = frame(JOB, 0, 0, 1, 4 * u64::from(EMBD));
        full.metrics = vec![json!({}); MAX_CHAIN_STAGES - 1];
        assert!(check_execute(&full, JOB, 0, 0, EMBD).is_ok());
        full.metrics.push(json!({}));
        assert!(check_execute(&full, JOB, 0, 0, EMBD).is_err());
    }

    #[test]
    fn new_frames_declare_their_payload() {
        let first = ExecuteFrame::new(JOB, 0, 0, 3, 8);
        assert_eq!(first.payload_bytes, 96);
        assert!(first.metrics.is_empty());
        assert_eq!(check_execute(&first, JOB, 0, 0, 8).unwrap(), 96);
    }

    #[test]
    fn job_ids_are_32_hex_digits() {
        let job = new_job_id();
        assert_eq!(job.len(), 32);
        assert!(
            job.bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        );
        validate_job_id(&job).unwrap();
        assert_ne!(job, new_job_id());
        let long = format!("{JOB}0");
        for bad in [
            "",
            "j",
            &JOB[1..],
            long.as_str(),
            "0123456789abcdef0123456789abcdeg",
        ] {
            assert!(validate_job_id(bad).is_err(), "{bad:?}");
        }
    }

    #[tokio::test]
    async fn rejects_oversized_metadata() {
        // ring.rs's own test.
        let bytes = vec![b'x'; MAX_CONTROL_LINE_BYTES + 1];
        assert!(
            read_control(&mut BufReader::new(bytes.as_slice()))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn control_line_bound_includes_the_newline() {
        let json = format!("{{\"kind\":\"start\",\"job\":\"{JOB}\"}}");
        let mut at_bound = json.clone().into_bytes();
        at_bound.resize(MAX_CONTROL_LINE_BYTES - 1, b' ');
        at_bound.push(b'\n');
        let message = read_control(&mut at_bound.as_slice()).await.unwrap();
        assert_eq!(
            message,
            Some(RingMessage::Start {
                job: JOB.to_owned()
            })
        );

        let mut over = json.into_bytes();
        over.resize(MAX_CONTROL_LINE_BYTES, b' ');
        over.push(b'\n');
        let error = read_control(&mut over.as_slice()).await.unwrap_err();
        assert!(error.to_string().contains("exceeds"), "{error:#}");
    }

    #[tokio::test]
    async fn read_control_handles_end_of_stream_and_garbage() {
        assert_eq!(read_control(&mut &b""[..]).await.unwrap(), None);
        assert!(
            read_control(&mut &b"{\"kind\":\"start\""[..])
                .await
                .is_err()
        );
        assert!(read_control(&mut &b"not json\n"[..]).await.is_err());
        assert!(
            read_control(&mut &b"{\"kind\":\"unknown\"}\n"[..])
                .await
                .is_err()
        );
        assert!(read_control(&mut &b"{\"job\":\"x\"}\n"[..]).await.is_err());
        assert!(next_control(&mut &b""[..]).await.is_err());
    }

    #[tokio::test]
    async fn write_control_refuses_oversized_lines_without_writing() {
        let mut progress = Map::new();
        progress.insert(
            "pad".to_owned(),
            Value::from("x".repeat(MAX_CONTROL_LINE_BYTES)),
        );
        let mut out = Vec::new();
        assert!(
            write_control(&mut out, &RingMessage::Progress(progress))
                .await
                .is_err()
        );
        assert!(out.is_empty());

        let start = RingMessage::Start {
            job: JOB.to_owned(),
        };
        let written = write_control(&mut out, &start).await.unwrap();
        assert_eq!(written, out.len());
        assert_eq!(out.last(), Some(&b'\n'));
        assert_eq!(
            read_control(&mut out.as_slice()).await.unwrap(),
            Some(start)
        );
    }

    #[tokio::test]
    async fn execute_frames_carry_their_payload() {
        let rows = [1.0f32, -2.5, 0.0, 4.0];
        let payload = encode_activations(&rows);
        let frame = ExecuteFrame::new(JOB, 0, 0, 2, 2);
        let mut out = Vec::new();
        assert_eq!(write_execute(&mut out, &frame, &payload).await.unwrap(), 16);
        assert!(
            write_execute(&mut Vec::new(), &frame, &payload[..12])
                .await
                .is_err()
        );

        let mut reader = BufReader::new(out.as_slice());
        let Some(RingMessage::Execute(read)) = read_control(&mut reader).await.unwrap() else {
            panic!("expected an execute frame");
        };
        let size = check_execute(&read, JOB, 0, 0, 2).unwrap();
        let bytes = read_payload(&mut reader, size).await.unwrap();
        assert_eq!(decode_activations(&bytes).unwrap(), rows);
        assert!(
            read_payload(&mut reader, 1).await.is_err(),
            "nothing follows"
        );
        assert!(decode_activations(&bytes[..3]).is_err());
        assert!(
            read_payload(&mut &[][..], MAX_PAYLOAD_BYTES + 1)
                .await
                .is_err()
        );
    }

    #[test]
    fn messages_match_ring_rs_json() {
        let job = JOB;
        assert_round_trips(
            &RingMessage::Start {
                job: job.to_owned(),
            },
            &json!({"kind": "start", "job": job}),
        );
        assert_round_trips(
            &RingMessage::ResultChannel {
                job: job.to_owned(),
            },
            &json!({"kind": "result_channel", "job": job}),
        );
        assert_round_trips(
            &RingMessage::Accepted {
                job: job.to_owned(),
            },
            &json!({"kind": "accepted", "job": job}),
        );
    }

    #[test]
    fn ready_keeps_native_info_fields_and_node_id() {
        let node = node_id();
        let wire = json!({"kind": "ready", "job": JOB, "stages": [{
            "protocol": 1, "begin": 28, "end": 32, "layers": 64, "embd": 5120, "vocab": 152064,
            "allocated_bytes": 42, "max_rss_bytes": 7, "vm": {"resident_bytes": 1},
            "source_sha256": "ab".repeat(32), "tokens": [1, 2], "node_id": node
        }]});
        let message: RingMessage = serde_json::from_value(wire.clone()).unwrap();
        let RingMessage::Ready { stages, .. } = &message else {
            panic!("expected ready, got {message:?}");
        };
        assert_eq!(stages[0].node_id, node);
        assert_eq!((stages[0].meta.begin, stages[0].meta.end), (28, 32));
        assert_eq!(stages[0].meta.extra["vm"]["resident_bytes"], 1);
        assert!(!stages[0].meta.extra.contains_key("node_id"));
        assert_round_trips(&message, &wire);
        // A stage without its worker's identity is not a ring stage.
        let mut anonymous = wire;
        anonymous["stages"][0]
            .as_object_mut()
            .unwrap()
            .remove("node_id");
        assert!(serde_json::from_value::<RingMessage>(anonymous).is_err());
    }

    #[test]
    fn execute_keeps_metrics_and_unknown_fields() {
        let wire = json!({"kind": "execute", "job": JOB, "step": 2, "pos": 7, "n": 1,
            "payload_bytes": 20480, "metrics": [{"step": 2, "node_id": "x"}], "trace": "t"});
        let message = decode(&wire.to_string());
        let RingMessage::Execute(frame) = &message else {
            panic!("expected execute, got {message:?}");
        };
        assert_eq!((frame.step, frame.pos, frame.n), (2, 7, 1));
        assert_eq!(frame.metrics.len(), 1);
        assert_eq!(frame.extra["trace"], "t");
        assert_round_trips(&message, &wire);
        // ring.rs requires the metrics array.
        let mut bare = wire;
        bare.as_object_mut().unwrap().remove("metrics");
        assert!(serde_json::from_value::<RingMessage>(bare).is_err());
    }

    #[test]
    fn progress_bodies_are_kept_whole() {
        let wire = json!({"kind": "progress", "event": "stage_end", "begin": 28, "step": 0,
            "input_bytes": 20480, "output_bytes": 0, "result": {"token": 9707}});
        let message = decode(&wire.to_string());
        let RingMessage::Progress(body) = &message else {
            panic!("expected progress, got {message:?}");
        };
        assert!(!body.contains_key("kind"));
        assert_eq!(body["result"]["token"], 9707);
        assert_round_trips(&message, &wire);
    }

    fn progress(begin: u32, step: u32, input_bytes: usize) -> StepProgress {
        StepProgress {
            begin,
            step,
            input_bytes,
        }
    }

    #[test]
    fn progress_lines_match_ring_rs_fields() {
        let start = serde_json::to_value(progress(28, 0, 20_480).started()).unwrap();
        assert_eq!(
            start,
            json!({"kind": "progress", "event": "stage_start", "begin": 28, "step": 0,
                "phase": "prefill", "input_bytes": 20480})
        );
        let decode = serde_json::to_value(progress(28, 1, 4).started()).unwrap();
        assert_eq!(decode["phase"], "decode");
        let end = serde_json::to_value(progress(28, 1, 4).ended(8, json!({"step": 1}))).unwrap();
        assert_eq!(
            end,
            json!({"kind": "progress", "event": "stage_end", "begin": 28, "step": 1,
                "input_bytes": 4, "output_bytes": 8, "result": {"step": 1}})
        );
    }

    #[test]
    fn result_keeps_native_reply_fields_and_chain_metrics() {
        let wire = json!({"kind": "result", "job": JOB, "step": 3, "n_floats": 0,
            "token": 9707, "eog": false, "piece_hex": "48656c6c6f", "compute_ms": 12.5,
            "observed_peak_buffer_bytes": 1, "max_rss_bytes": 2, "vm": null, "wall_ms": 13.0,
            "request_bytes": 20500, "response_tensor_bytes": 0, "node_id": "n",
            "chain_metrics": [{"step": 3}, {"step": 3}]});
        let message = decode(&wire.to_string());
        let RingMessage::Result(result) = &message else {
            panic!("expected result, got {message:?}");
        };
        assert_eq!(result.reply.token, 9707);
        assert_eq!(result.reply.piece_bytes().unwrap(), b"Hello");
        assert_eq!(result.chain_metrics.len(), 2);
        assert_eq!(result.reply.extra["node_id"], "n");
        assert!(!result.reply.extra.contains_key("job"));
        assert_round_trips(&message, &wire);
    }

    #[tokio::test]
    async fn read_result_checks_job_step_and_floats() {
        let line = |step: u32, n_floats: usize, job: &str| {
            let mut text = json!({"kind": "result", "job": job, "step": step,
                "n_floats": n_floats, "token": 1, "eog": false, "piece_hex": "41"})
            .to_string();
            text.push('\n');
            text
        };
        let ok = read_result(&mut line(4, 0, JOB).as_bytes(), JOB, 4)
            .await
            .unwrap();
        assert_eq!(ok.reply.piece_bytes().unwrap(), b"A");
        assert!(
            read_result(&mut line(5, 0, JOB).as_bytes(), JOB, 4)
                .await
                .is_err()
        );
        assert!(
            read_result(&mut line(4, 8, JOB).as_bytes(), JOB, 4)
                .await
                .is_err()
        );
        assert!(
            read_result(&mut line(4, 0, &new_job_id()).as_bytes(), JOB, 4)
                .await
                .is_err()
        );
        let progress = "{\"kind\":\"progress\"}\n";
        assert!(read_result(&mut progress.as_bytes(), JOB, 4).await.is_err());
    }

    #[tokio::test]
    async fn result_channel_hello_needs_a_valid_job() {
        let hello = format!("{{\"kind\":\"result_channel\",\"job\":\"{JOB}\"}}\n");
        assert_eq!(
            read_result_channel_hello(&mut hello.as_bytes())
                .await
                .unwrap(),
            JOB
        );
        let bad = "{\"kind\":\"result_channel\",\"job\":\"j\"}\n";
        assert!(
            read_result_channel_hello(&mut bad.as_bytes())
                .await
                .is_err()
        );
        let start = format!("{{\"kind\":\"start\",\"job\":\"{JOB}\"}}\n");
        assert!(
            read_result_channel_hello(&mut start.as_bytes())
                .await
                .is_err()
        );
    }

    #[test]
    fn validate_chain_accepts_a_contiguous_chain() {
        validate_chain(&[stage(28, 32)]).unwrap();
        validate_chain(&[stage(0, 28), stage(28, 32), stage(32, 64)]).unwrap();
    }

    #[test]
    fn validate_chain_rejects_gaps_overlaps_and_size() {
        assert!(validate_chain(&[]).is_err());
        let gap = validate_chain(&[stage(0, 28), stage(30, 64)]).unwrap_err();
        assert!(gap.to_string().contains("noncontiguous"), "{gap:#}");
        assert!(validate_chain(&[stage(0, 30), stage(28, 64)]).is_err());
        let too_long: Vec<_> = (0..17).map(|layer| stage(layer, layer + 1)).collect();
        assert!(validate_chain(&too_long).is_err());
        assert!(validate_chain(&too_long[..16]).is_ok());
    }

    #[test]
    fn validate_chain_rejects_model_mismatches() {
        let mismatches: [fn(&mut StageMetadata); 4] = [
            |meta| meta.layers = 65,
            |meta| meta.embd = 4096,
            |meta| meta.vocab = 1,
            |meta| meta.source_sha256 = "cd".repeat(32),
        ];
        for mismatch in mismatches {
            let mut other = stage(32, 64);
            mismatch(&mut other.meta);
            let error = validate_chain(&[stage(0, 32), other]).unwrap_err();
            assert!(format!("{error:#}").contains("mismatch"), "{error:#}");
        }
    }

    #[test]
    fn validate_chain_rejects_invalid_stages() {
        let invalid: [fn(&mut RingStageInfo); 9] = [
            |stage| stage.meta.protocol = 2,
            |stage| stage.meta.end = stage.meta.begin,
            |stage| stage.meta.end = stage.meta.layers + 1,
            |stage| stage.meta.embd = 0,
            |stage| stage.meta.embd = MAX_EMBD + 1,
            |stage| stage.meta.vocab = 0,
            |stage| stage.meta.source_sha256 = String::new(),
            |stage| stage.meta.source_sha256 = "zz".repeat(32),
            |stage| stage.node_id = "not a node id".to_owned(),
        ];
        for (index, corrupt) in invalid.into_iter().enumerate() {
            let mut bad = stage(32, 64);
            corrupt(&mut bad);
            assert!(validate_chain(&[bad]).is_err(), "case {index}");
        }
    }
}
