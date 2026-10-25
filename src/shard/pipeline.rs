//! Drives one generation through every stage of a sharded model in order.
//!
//! A [`StageLink`] is one stage's byte stream plus its startup metadata: either a local
//! `shard_runner` child on stdio ([`StageLink::spawn_local`]) or a remote stage dialled over
//! Iroh ([`StageLink::connect_remote`]). [`Pipeline::new`] checks every link's metadata against
//! what the manifest expects, and [`Pipeline::generate`] runs greedy decoding: prompt tokens
//! (from the final stage's own tokenization) are prefilled in batches of at most
//! [`MAX_BATCH_TOKENS`], each stage's hidden rows feed the next, and every sampled token goes
//! back to the first stage until end-of-generation, `max_tokens` or the context limit.
//!
//! The pipeline handles any number of stages; only the manifest restricts v1 topologies. Each
//! runner serves one sequence whose cache only grows, so links are single-use and a failed
//! generation cannot be repaired in place. Instead the progress that must survive lives in a
//! [`GenerationState`]: [`Pipeline::resume`] runs a fresh pipeline from it by prefilling the
//! prompt plus every token generated so far (from step and position 0), then decoding on. A
//! first attempt is a resume with nothing generated. [`classify`] tells a caller whether an
//! error is worth another attempt on fresh links and which stage or worker failed.
//!
//! For balanced manifests, [`probe_remote`] asks a full-model worker for its free memory before
//! any range is chosen, and [`StageLink::open_probed`] then opens the session for the chosen
//! range on the same connection.
//!
//! In EXPERIMENTAL ring execution ([`Pipeline::with_ring`]) the pipeline is the local stage 0,
//! which tokenizes, followed by a [`RingLink`] that stands for every remote stage: stage 0's
//! hidden rows go to the ring's entry worker and each step's token comes back from the terminal.
//! The same decoding loop, stop rules and UTF-8 assembly apply,
//! but ring execution is never retried: the front end fails the job on any ring error.

use std::collections::BTreeMap;
use std::future::Future;
use std::io;
use std::marker::PhantomData;
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, bail, ensure};
use iroh::endpoint::{Connection, VarInt};
use iroh::{Endpoint, EndpointId};
use tokio::io::{
    AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader,
};
use tokio::process::Command;
use tokio::sync::oneshot;
use tokio::time::error::Elapsed;
use tokio::time::{timeout, timeout_at};
use tracing::{debug, info, warn};

use crate::admission::AdmittedChild as Child;

use crate::protocol::RPC_BRIDGE_ALPN;
use crate::rpc_bridge::path_kind;
use crate::shard::manifest::same_sha256;
use crate::shard::ring_origin::{RingLink, check_origin_stage};
use crate::shard::wire::{
    self, MAX_BATCH_TOKENS, ProbeReply, SessionRequest, StageInput, StageMetadata, StepReply,
    StreamClosed,
};

/// Bound on waiting for a stage's metadata line; loading a large model can take minutes.
pub const METADATA_TIMEOUT: Duration = Duration::from_secs(300);

/// Bound on dialling a worker and reading its probe reply.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(20);

/// Bound on one stage answering one request (a full prefill batch on a slow CPU included).
pub const STEP_TIMEOUT: Duration = Duration::from_secs(120);

/// Protocol version this pipeline speaks.
const PROTOCOL_VERSION: u32 = 1;

/// `RETURN_LOGITS` argument: the final stage returns only the greedy token.
const NO_LOGITS: &str = "0";

/// How long a local runner gets to exit after its stdin closes before it is killed.
const CHILD_EXIT_GRACE: Duration = Duration::from_secs(5);

/// `tracing` target for lines a local runner writes to stderr.
const RUNNER_LOG_TARGET: &str = "shard_runner";

/// Maximum runner stderr buffered for one tracing event; longer lines are split into chunks.
const MAX_RUNNER_LOG_LINE_BYTES: u64 = 8 * 1024;

/// Application close code for a remote stage connection the generation is done with.
const CONNECTION_DONE: VarInt = VarInt::from_u32(0);

/// Close reason sent with [`CONNECTION_DONE`].
const CONNECTION_DONE_REASON: &[u8] = b"generation finished";

/// Everything needed to start one local stage runner.
#[derive(Clone, Copy, Debug)]
pub struct LocalLaunch<'a> {
    /// The `shard_runner` binary.
    pub runner: &'a Path,
    /// Model or shard file passed as `MODEL`.
    pub model: &'a Path,
    /// First layer (inclusive).
    pub begin: u32,
    /// End of the layer range (exclusive).
    pub end: u32,
    /// Context window.
    pub context: u32,
    /// Layers offloaded to the GPU.
    pub gpu_layers: u32,
    /// Runtime buffer budget.
    pub budget_bytes: u64,
    /// Rendered prompt; the final stage tokenizes it, other stages ignore it.
    pub prompt: &'a str,
    /// Extra environment for the runner process.
    pub env: &'a BTreeMap<String, String>,
}

/// What keeps a link's process or stream alive.
enum LinkGuard {
    /// A local runner, killed when dropped.
    Process(Box<Child>),
    /// The Iroh connection carrying a remote stage's stream.
    Stream(Connection),
}

/// One stage's byte stream, its startup metadata, and whatever keeps it alive.
pub struct StageLink {
    reader: Box<dyn AsyncBufRead + Send + Unpin>,
    writer: Box<dyn AsyncWrite + Send + Unpin>,
    metadata: StageMetadata,
    guard: LinkGuard,
    label: String,
    /// The remote worker serving this stage; `None` for a local runner.
    worker: Option<EndpointId>,
}

impl StageLink {
    /// Spawns a local runner (no preface) and reads its metadata within `metadata_timeout`.
    /// The child is killed if this link is dropped; its stderr goes to `tracing`.
    pub async fn spawn_local(
        launch: &LocalLaunch<'_>,
        metadata_timeout: Duration,
    ) -> anyhow::Result<Self> {
        let label = format!("local stage {}..{}", launch.begin, launch.end);
        let startup_started = Instant::now();
        let mut child = Child::spawn(
            Command::new(launch.runner)
                .arg(launch.model)
                .args(runner_args(launch))
                .envs(launch.env)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true),
        )
        .await
        .with_context(|| format!("failed to start {label} ({})", launch.runner.display()))?;
        let stdin = child.stdin.take().context("runner stdin was not piped")?;
        let stdout = child.stdout.take().context("runner stdout was not piped")?;
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(forward_stderr(label.clone(), stderr));
        }
        info!(stage = %label, pid = ?child.id(), "shard runner started");
        let mut reader = BufReader::new(stdout);
        let metadata = read_metadata(&mut reader, &label, metadata_timeout).await?;
        debug!(stage = %label, startup_us = startup_started.elapsed().as_micros(), "local stage metadata ready");
        Ok(Self {
            reader: Box::new(reader),
            writer: Box::new(stdin),
            metadata,
            guard: LinkGuard::Process(Box::new(child)),
            label,
            worker: None,
        })
    }

    /// Dials `worker` on the RPC bridge ALPN, opens one stream, sends the client preface and
    /// reads the forwarded metadata within `metadata_timeout`.
    pub async fn connect_remote(
        endpoint: &Endpoint,
        worker: EndpointId,
        metadata_timeout: Duration,
    ) -> anyhow::Result<Self> {
        let connection = dial(endpoint, worker).await?;
        Self::open_session(connection, worker, None, metadata_timeout).await
    }

    /// Opens a session for layers `[begin, end)` on the connection `probe` was answered on,
    /// and reads the forwarded metadata within `metadata_timeout`. The worker must hold the
    /// full model (or serve exactly that range).
    pub async fn open_probed(
        probe: RemoteProbe,
        begin: u32,
        end: u32,
        metadata_timeout: Duration,
    ) -> anyhow::Result<Self> {
        let request = SessionRequest::Session { begin, end };
        Self::open_session(
            probe.connection,
            probe.worker,
            Some(request),
            metadata_timeout,
        )
        .await
    }

    /// Opens a stream on `connection`, greets the worker with the original preface (`None`) or
    /// a ranged session request, and reads the forwarded metadata.
    async fn open_session(
        connection: Connection,
        worker: EndpointId,
        request: Option<SessionRequest>,
        metadata_timeout: Duration,
    ) -> anyhow::Result<Self> {
        let label = format!("remote stage on {}", worker.fmt_short());
        let (mut send, recv) = connection
            .open_bi()
            .await
            .with_context(|| Unreachable(format!("failed to open a stream to {label}")))?;
        match &request {
            None => wire::write_preface(&mut send).await,
            Some(request) => wire::write_session_request(&mut send, request).await,
        }
        .with_context(|| format!("failed to greet {label}"))?;
        let mut reader = BufReader::new(recv);
        let metadata_started = Instant::now();
        let metadata = read_metadata(&mut reader, &label, metadata_timeout).await?;
        debug!(stage = %label, metadata_us = metadata_started.elapsed().as_micros(), "remote stage metadata ready");
        info!(stage = %label, begin = metadata.begin, end = metadata.end, "remote stage ready");
        Ok(Self {
            reader: Box::new(reader),
            writer: Box::new(send),
            metadata,
            guard: LinkGuard::Stream(connection),
            label,
            worker: Some(worker),
        })
    }

    /// The stage's startup metadata.
    pub fn metadata(&self) -> &StageMetadata {
        &self.metadata
    }

    /// Names this link as stage `stage` of its pipeline, for [`classify`].
    fn fault(&self, stage: usize) -> StageFault {
        StageFault {
            stage,
            worker: self.worker,
        }
    }

    /// Sends one request and reads its reply, all within `step_timeout`.
    async fn exchange(
        &mut self,
        step: u32,
        position: u32,
        input: &StageInput<'_>,
        max_floats: usize,
        step_timeout: Duration,
    ) -> anyhow::Result<wire::StepOutput> {
        let started = Instant::now();
        let deadline = tokio::time::Instant::now() + step_timeout;
        let (input_kind, input_bytes) = match input {
            StageInput::Tokens(tokens) => ("tokens", tokens.len().saturating_mul(4)),
            StageInput::Hidden { rows, .. } => ("hidden", rows.len().saturating_mul(4)),
        };
        timeout_at(
            deadline,
            wire::write_request(&mut self.writer, step, position, input),
        )
        .await
        .with_context(|| {
            format!(
                "{} timed out writing step {step} within {step_timeout:?}",
                self.label
            )
        })?
        .with_context(|| format!("{} failed writing step {step}", self.label))?;
        let write_us = started.elapsed().as_micros();
        let read_started = Instant::now();
        let output = timeout_at(deadline, wire::read_step(&mut self.reader, max_floats))
            .await
            .with_context(|| {
                format!(
                    "{} timed out reading step {step} within {step_timeout:?}",
                    self.label
                )
            })?
            .with_context(|| format!("{} failed reading step {step}", self.label))?;
        debug!(
            stage = %self.label,
            step,
            position,
            input_kind,
            input_bytes,
            output_bytes = output.floats.len().saturating_mul(4),
            write_us,
            read_us = read_started.elapsed().as_micros(),
            total_us = started.elapsed().as_micros(),
            "shard stage exchange"
        );
        ensure!(
            output.reply.step == step,
            "{} answered step {} to request {step}",
            self.label,
            output.reply.step
        );
        Ok(output)
    }

    /// Ends the session: closes the request stream, then lets a local runner exit (killing
    /// it after a grace period, or at once when `kill` is set) or closes the connection.
    async fn close(self, kill: bool) {
        let Self {
            mut writer,
            guard,
            label,
            ..
        } = self;
        // A failed shutdown means the peer is already gone, which is what closing wants.
        let _ = writer.shutdown().await;
        drop(writer);
        match guard {
            LinkGuard::Process(child) => stop_child(*child, &label, kill).await,
            LinkGuard::Stream(connection) => {
                connection.close(CONNECTION_DONE, CONNECTION_DONE_REASON);
            }
        }
    }
}

/// A worker's answer to a probe, and the connection to open its session on.
pub struct RemoteProbe {
    /// The worker that answered.
    pub worker: EndpointId,
    /// What it reported.
    pub reply: ProbeReply,
    connection: Connection,
}

impl RemoteProbe {
    /// Closes the connection of a probe whose worker was not chosen.
    pub fn discard(self) {
        self.connection
            .close(CONNECTION_DONE, CONNECTION_DONE_REASON);
    }
}

/// Dials `worker` and asks for its free memory and model identity, all within
/// `probe_timeout`. Failures classify as retryable: the worker is unreachable, too old to know
/// probes (it resets the stream), or slow.
pub async fn probe_remote(
    endpoint: &Endpoint,
    worker: EndpointId,
    probe_timeout: Duration,
) -> anyhow::Result<RemoteProbe> {
    let label = format!("worker {}", worker.fmt_short());
    let probe = async {
        let connection = dial(endpoint, worker).await?;
        let (mut send, recv) = connection
            .open_bi()
            .await
            .with_context(|| Unreachable(format!("failed to open a stream to {label}")))?;
        wire::write_session_request(&mut send, &SessionRequest::Probe).await?;
        let reply = wire::read_probe_reply(&mut BufReader::new(recv)).await?;
        anyhow::Ok(RemoteProbe {
            worker,
            reply,
            connection,
        })
    };
    timeout(probe_timeout, probe)
        .await
        .with_context(|| format!("{label} did not answer a probe within {probe_timeout:?}"))?
        .with_context(|| format!("failed to probe {label}"))
}

/// Dials `worker` on the RPC bridge ALPN.
async fn dial(endpoint: &Endpoint, worker: EndpointId) -> anyhow::Result<Connection> {
    let label = format!("remote stage on {}", worker.fmt_short());
    let connect_started = Instant::now();
    let connection = endpoint
        .connect(worker, RPC_BRIDGE_ALPN)
        .await
        .with_context(|| Unreachable(format!("failed to connect to {label}")))?;
    debug!(stage = %label, path = path_kind(&connection), connect_us = connect_started.elapsed().as_micros(), "remote stage connected");
    Ok(connection)
}

/// How the stage a link serves is expected to look.
#[derive(Clone, Debug)]
pub struct StageExpectation {
    /// First layer (inclusive).
    pub begin: u32,
    /// End of the layer range (exclusive).
    pub end: u32,
    /// Total layers in the model.
    pub layers: u32,
    /// SHA-256 of the original model the stage must belong to.
    pub model_sha256: String,
    /// How the stage proves it belongs to that model.
    pub identity: StageIdentity,
}

/// How a stage's metadata proves which model it serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageIdentity {
    /// The relay's `model_sha256` field must equal the expected model hash.
    Remote,
    /// The runner's `source_sha256` must equal the expected model hash, unless it is empty and
    /// the stage's file was verified against a configured `artifact_sha256` at startup.
    Local {
        /// Whether a configured `artifact_sha256` matched the stage's file.
        artifact_verified: bool,
    },
}

/// Limits for one generation, counted over every attempt of it.
#[derive(Clone, Copy, Debug)]
pub struct GenerationLimits {
    /// Most completion tokens to produce in total (at least 1).
    pub max_tokens: u32,
    /// Context window every stage was started with.
    pub context: u32,
}

impl GenerationLimits {
    /// Whether decoding must stop once a token sampled at `position` brings the completion
    /// to `completion_tokens`: the budget is spent, or the token has no position to go to.
    fn spent_after_token(self, completion_tokens: u32, position: u32) -> bool {
        completion_tokens >= self.max_tokens || position >= self.context
    }

    /// Whether a resume is out of room before it starts: the budget is spent, or prefilling
    /// `history` positions (prompt plus generated tokens) overflows the context.
    fn spent_before_resume(self, completion_tokens: u32, history: u32) -> bool {
        completion_tokens >= self.max_tokens || history > self.context
    }
}

/// Why a generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// The model produced end-of-generation.
    Stop,
    /// `max_tokens` or the context window ran out.
    Length,
}

impl FinishReason {
    /// The OpenAI `finish_reason` string.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

/// The result of a finished generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generation {
    /// The full completion text.
    pub text: String,
    /// Why it stopped.
    pub finish: FinishReason,
    /// Tokens in the prompt.
    pub prompt_tokens: u32,
    /// Completion tokens produced (end-of-generation not counted).
    pub completion_tokens: u32,
}

/// The prompt does not fit the context window; a client error, unlike every other failure.
#[derive(Debug, thiserror::Error)]
#[error("the prompt has {prompt_tokens} tokens but the context window is {context}")]
pub struct PromptTooLong {
    /// Tokens in the prompt.
    pub prompt_tokens: u32,
    /// The context window.
    pub context: u32,
}

/// Cooperative request cancellation caused by a disconnected HTTP client.
#[derive(Debug, thiserror::Error)]
#[error("generation canceled because the HTTP client disconnected")]
pub(crate) struct GenerationCancelled;

/// Whether a failed generation may succeed if it is resumed on fresh stage links.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureClass {
    /// A stage or its link went away (I/O error, closed stream, timeout, runner exit, failed
    /// dial): fresh links may well work.
    Retryable,
    /// Anything else (a prompt too long, metadata that does not match the manifest, a
    /// sampled token outside the vocabulary, a protocol violation): retrying cannot help.
    Fatal,
}

/// Context attached to an error from one stage of a running pipeline, naming that stage (its
/// index in pipeline order) and, for a remote stage, the worker serving it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("stage {stage} failed")]
pub struct StageFault {
    /// Index of the stage in pipeline (manifest) order.
    pub stage: usize,
    /// The remote worker serving the stage; `None` for a local runner.
    pub worker: Option<EndpointId>,
}

/// What [`classify`] found out about an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Classification {
    /// Whether another attempt may succeed.
    pub class: FailureClass,
    /// The stage that failed, when known.
    pub fault: Option<StageFault>,
}

/// A remote stage could not be dialled or gave no stream. Attached as context, so its
/// message is the error's own.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct Unreachable(String);

/// Classifies an error from opening stage links or running a pipeline. It is retryable when
/// caused by an I/O error, a stream that closed early (a runner that exited), a step or
/// metadata timeout, or a failed dial; anything else, and always [`PromptTooLong`], is fatal.
pub fn classify(error: &anyhow::Error) -> Classification {
    let transient = error.downcast_ref::<Unreachable>().is_some()
        || error.chain().any(|cause| {
            cause.is::<io::Error>() || cause.is::<Elapsed>() || cause.is::<StreamClosed>()
        });
    let class = if transient && error.downcast_ref::<PromptTooLong>().is_none() {
        FailureClass::Retryable
    } else {
        FailureClass::Fatal
    };
    Classification {
        class,
        fault: error.downcast_ref::<StageFault>().copied(),
    }
}

/// The progress of one generation that outlives any single attempt: the prompt's tokens, the
/// tokens generated so far and the text already handed to `on_text` (including a partly
/// received UTF-8 character), so a resumed attempt never repeats text.
pub struct GenerationState<F, Fut> {
    limits: GenerationLimits,
    /// The first attempt's final-stage tokenization; later attempts must match it.
    prompt_tokens: Option<Vec<u32>>,
    generated: Vec<u32>,
    text: TextSink<F, Fut>,
}

impl<F, Fut> GenerationState<F, Fut>
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = ()>,
{
    /// A generation with nothing produced yet; `on_text` receives each new piece of text and
    /// is awaited before decoding goes on (so a slow reader applies backpressure).
    pub fn new(limits: GenerationLimits, on_text: F) -> Self {
        Self {
            limits,
            prompt_tokens: None,
            generated: Vec::new(),
            text: TextSink {
                assembler: Utf8Assembler::default(),
                text: String::new(),
                on_text,
                _future: PhantomData,
            },
        }
    }

    /// Completion tokens produced so far, over every attempt.
    pub fn completion_tokens(&self) -> u32 {
        token_len(&self.generated)
    }

    /// Records the prompt tokens on the first attempt; on later ones, insists the new final
    /// stage tokenized the prompt identically.
    fn adopt_prompt(&mut self, tokens: &[u32], label: &str) -> anyhow::Result<()> {
        match &self.prompt_tokens {
            None => self.prompt_tokens = Some(tokens.to_vec()),
            Some(first) => ensure!(
                first.as_slice() == tokens,
                "{label} tokenized the prompt into {} tokens that differ from the {} of the \
                 first attempt",
                tokens.len(),
                first.len()
            ),
        }
        Ok(())
    }

    /// Every position a resume must prefill: the prompt, then each generated token.
    fn history(&self) -> Vec<u32> {
        let prompt = self.prompt_tokens.as_deref().unwrap_or_default();
        [prompt, &self.generated].concat()
    }

    /// Whether the limits leave no room to resume.
    fn spent(&self) -> bool {
        let history = token_len(&self.history());
        self.limits
            .spent_before_resume(self.completion_tokens(), history)
    }

    async fn push(&mut self, token: u32, piece: &[u8]) {
        self.generated.push(token);
        self.text.push(piece).await;
    }

    /// Flushes any incomplete trailing bytes and describes the finished generation.
    async fn finish(&mut self, finish: FinishReason) -> Generation {
        self.text.flush().await;
        Generation {
            text: self.text.text.clone(),
            finish,
            prompt_tokens: self.prompt_tokens.as_deref().map_or(0, token_len),
            completion_tokens: self.completion_tokens(),
        }
    }
}

/// Validated stage links, ready for one generation.
pub struct Pipeline {
    links: Vec<StageLink>,
    /// In ring execution, the remote stages after `links`; the ring's terminal is final.
    ring: Option<RingLink>,
    embd: u32,
    vocab: u32,
    prompt_tokens: Vec<u32>,
    step_timeout: Duration,
}

impl Pipeline {
    /// Checks each link's metadata against its expectation (same order and count): layer
    /// range, protocol version, model identity, a nonzero hidden size shared by every stage,
    /// and only the last stage being final. The prompt tokens come from the final stage.
    pub fn new(links: Vec<StageLink>, expected: &[StageExpectation]) -> anyhow::Result<Self> {
        ensure!(!links.is_empty(), "a pipeline needs at least one stage");
        ensure!(
            links.len() == expected.len(),
            "{} stage links for {} expected stages",
            links.len(),
            expected.len()
        );
        let embd = links[0].metadata.embd;
        ensure!(embd > 0, "{} reports a zero hidden size", links[0].label);
        let last = links.len() - 1;
        for (index, (link, expectation)) in links.iter().zip(expected).enumerate() {
            check_metadata(link, expectation, embd, index == last)?;
        }
        let final_meta = &links[last].metadata;
        ensure!(
            !final_meta.tokens.is_empty(),
            "{} reported no prompt tokens",
            links[last].label
        );
        Ok(Self {
            vocab: final_meta.vocab,
            prompt_tokens: final_meta.tokens.clone(),
            links,
            ring: None,
            embd,
            step_timeout: STEP_TIMEOUT,
        })
    }

    /// Ring execution: the local stage 0 (checked against `expected`, and required not to be
    /// final) followed by `ring`, whose chain must continue stage 0 ([`check_origin_stage`]).
    /// Stage 0 tokenizes, so the prompt tokens come from its metadata.
    pub fn with_ring(
        local: StageLink,
        ring: RingLink,
        expected: &StageExpectation,
    ) -> anyhow::Result<Self> {
        let meta = &local.metadata;
        ensure!(meta.embd > 0, "{} reports a zero hidden size", local.label);
        check_metadata(&local, expected, meta.embd, false)?;
        check_origin_stage(meta, ring.stages())
            .with_context(|| format!("{} does not continue into the ring", local.label))?;
        ensure!(
            !meta.tokens.is_empty(),
            "{} reported no prompt tokens",
            local.label
        );
        Ok(Self {
            vocab: meta.vocab,
            prompt_tokens: meta.tokens.clone(),
            embd: meta.embd,
            links: vec![local],
            ring: Some(ring),
            step_timeout: STEP_TIMEOUT,
        })
    }

    /// Replaces the per-step timeout (default [`STEP_TIMEOUT`]).
    pub fn with_step_timeout(mut self, step_timeout: Duration) -> Self {
        self.step_timeout = step_timeout;
        self
    }

    /// Tokens in the prompt the final stage tokenized.
    pub fn prompt_tokens(&self) -> u32 {
        token_len(&self.prompt_tokens)
    }

    /// Fails with [`PromptTooLong`] if the prompt alone overflows `context`.
    pub fn ensure_prompt_fits(&self, context: u32) -> Result<(), PromptTooLong> {
        let prompt_tokens = self.prompt_tokens();
        if prompt_tokens > context {
            return Err(PromptTooLong {
                prompt_tokens,
                context,
            });
        }
        Ok(())
    }

    /// Closes every stage and ring link held by this pipeline.
    pub async fn close(self, kill: bool) {
        if let Some(ring) = self.ring {
            ring.close(!kill).await;
        }
        for link in self.links {
            link.close(kill).await;
        }
    }

    /// Runs greedy decoding to completion, calling `on_text` with each new piece of complete
    /// UTF-8 text as it is produced, then closes every link. On any error the links are killed
    /// and the error is returned; nothing is retried.
    pub async fn generate<F: FnMut(&str)>(
        self,
        limits: GenerationLimits,
        mut on_text: F,
    ) -> anyhow::Result<Generation> {
        let on_text = move |text: String| {
            on_text(&text);
            std::future::ready(())
        };
        self.resume(&mut GenerationState::new(limits, on_text))
            .await
    }

    /// Runs generation with async text delivery and cooperative cancellation. The pipeline
    /// closes remote streams and local runners before returning after a client disconnect.
    pub async fn generate_cancellable<F, Fut>(
        self,
        limits: GenerationLimits,
        mut cancel: oneshot::Receiver<()>,
        on_text: F,
    ) -> anyhow::Result<Generation>
    where
        F: FnMut(String) -> Fut,
        Fut: Future<Output = ()>,
    {
        let mut state = GenerationState::new(limits, on_text);
        self.resume_cancellable(&mut state, &mut cancel).await
    }

    /// Continues the generation in `state` on these (fresh) links, then closes them: prefills
    /// the prompt and every token generated so far, and decodes on from the final reply until
    /// end-of-generation or the limits, which count totals over every attempt. A resume with
    /// no room left finishes with [`FinishReason::Length`]. The first resume records the
    /// prompt tokens; a later one fails if the final stage tokenized the prompt differently.
    /// On any error the links are killed, `state` keeps everything produced so far, and the
    /// error is returned for the caller to [`classify`].
    pub async fn resume<F, Fut>(
        self,
        state: &mut GenerationState<F, Fut>,
    ) -> anyhow::Result<Generation>
    where
        F: FnMut(String) -> Fut,
        Fut: Future<Output = ()>,
    {
        self.resume_inner(state, None).await
    }

    /// [`Pipeline::resume`] that instead ends with [`GenerationCancelled`] once `cancel`
    /// resolves (the client disconnected), after closing every link. `cancel` is borrowed so
    /// one receiver can guard every attempt of a generation; once it has resolved it must not
    /// be passed again.
    pub async fn resume_cancellable<F, Fut>(
        self,
        state: &mut GenerationState<F, Fut>,
        cancel: &mut oneshot::Receiver<()>,
    ) -> anyhow::Result<Generation>
    where
        F: FnMut(String) -> Fut,
        Fut: Future<Output = ()>,
    {
        self.resume_inner(state, Some(cancel)).await
    }

    async fn resume_inner<F, Fut>(
        mut self,
        state: &mut GenerationState<F, Fut>,
        cancel: Option<&mut oneshot::Receiver<()>>,
    ) -> anyhow::Result<Generation>
    where
        F: FnMut(String) -> Fut,
        Fut: Future<Output = ()>,
    {
        let result = match cancel {
            Some(cancel) => tokio::select! {
                biased;
                _ = cancel => Err(GenerationCancelled.into()),
                result = self.run(state) => result,
            },
            None => self.run(state).await,
        };
        let kill = result.is_err();
        self.close(kill).await;
        result
    }

    async fn run<F, Fut>(
        &mut self,
        state: &mut GenerationState<F, Fut>,
    ) -> anyhow::Result<Generation>
    where
        F: FnMut(String) -> Fut,
        Fut: Future<Output = ()>,
    {
        let limits = state.limits;
        ensure!(limits.max_tokens >= 1, "max_tokens must be at least 1");
        let final_label = &self.links[self.links.len() - 1].label;
        state.adopt_prompt(&self.prompt_tokens, final_label)?;
        self.ensure_prompt_fits(limits.context)?;
        if state.spent() {
            return Ok(state.finish(FinishReason::Length).await);
        }
        let mut decoder = Decoder::default();
        let mut reply = self.prefill(&mut decoder, &state.history()).await?;
        let finish = loop {
            if reply.eog {
                break FinishReason::Stop;
            }
            let token = self.sampled_token(&reply)?;
            state.push(token, &reply.piece_bytes()?).await;
            if limits.spent_after_token(state.completion_tokens(), decoder.position) {
                break FinishReason::Length;
            }
            reply = self.step(&mut decoder, &[token]).await?;
        };
        Ok(state.finish(finish).await)
    }

    /// Feeds `history` in batches from position 0 and returns the final stage's reply to the
    /// last batch.
    async fn prefill(
        &mut self,
        decoder: &mut Decoder,
        history: &[u32],
    ) -> anyhow::Result<StepReply> {
        let mut last = None;
        for batch in history.chunks(MAX_BATCH_TOKENS as usize) {
            last = Some(self.step(decoder, batch).await?);
        }
        last.context("the prompt has no tokens")
    }

    /// Sends one batch of tokens through every stage (and the ring, if any) and returns the
    /// final stage's reply.
    async fn step(&mut self, decoder: &mut Decoder, tokens: &[u32]) -> anyhow::Result<StepReply> {
        let started = Instant::now();
        let (step, position) = (decoder.step, decoder.position);
        let count = tokens.len();
        let row_values = count * self.embd as usize;
        let ring_stages = self.ring.as_ref().map_or(0, |ring| ring.stages().len());
        let stage_count = self.links.len() + ring_stages;
        // With a ring, the terminal is final and every local link forwards its rows.
        let final_link = self.ring.is_none().then(|| self.links.len() - 1);
        let mut rows = Vec::new();
        for (index, link) in self.links.iter_mut().enumerate() {
            let input = if index == 0 {
                StageInput::Tokens(tokens)
            } else {
                StageInput::Hidden {
                    rows: &rows,
                    embd: self.embd,
                }
            };
            let is_final = final_link == Some(index);
            let max_floats = if is_final { 0 } else { row_values };
            let output = link
                .exchange(step, position, &input, max_floats, self.step_timeout)
                .await
                .with_context(|| link.fault(index))?;
            if is_final {
                decoder.advance(count);
                debug!(
                    step,
                    position,
                    token_count = count,
                    stage_count,
                    total_us = started.elapsed().as_micros(),
                    "shard pipeline step"
                );
                return Ok(output.reply);
            }
            ensure!(
                output.floats.len() == row_values,
                "{} returned {} floats for {count} rows of {}",
                link.label,
                output.floats.len(),
                self.embd
            );
            rows = output.floats;
        }
        let ring = self
            .ring
            .as_mut()
            .context("the pipeline has no final stage")?;
        let reply = ring.exchange(step, position, count, &rows).await?;
        decoder.advance(count);
        debug!(
            step,
            position,
            token_count = count,
            stage_count,
            total_us = started.elapsed().as_micros(),
            "shard pipeline step"
        );
        Ok(reply)
    }

    /// The final stage's greedy token, checked to be inside the vocabulary.
    fn sampled_token(&self, reply: &StepReply) -> anyhow::Result<u32> {
        u32::try_from(reply.token)
            .ok()
            .filter(|&token| token < self.vocab)
            .with_context(|| {
                format!(
                    "step {} sampled token {} outside the vocabulary of {}",
                    reply.step, reply.token, self.vocab
                )
            })
    }
}

/// Step and position counters shared by every stage.
#[derive(Default)]
struct Decoder {
    step: u32,
    position: u32,
}

impl Decoder {
    fn advance(&mut self, count: usize) {
        self.step += 1;
        self.position += u32::try_from(count).unwrap_or(u32::MAX);
    }
}

/// Collects the completion and hands each newly completed piece of text to the caller.
struct TextSink<F, Fut> {
    assembler: Utf8Assembler,
    text: String,
    on_text: F,
    _future: PhantomData<Fut>,
}

impl<F, Fut> TextSink<F, Fut>
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = ()>,
{
    async fn push(&mut self, piece: &[u8]) {
        let fresh = self.assembler.push(piece);
        self.deliver(fresh).await;
    }

    /// Flushes any incomplete trailing bytes.
    async fn flush(&mut self) {
        let rest = std::mem::take(&mut self.assembler).finish();
        self.deliver(rest).await;
    }

    async fn deliver(&mut self, fresh: String) {
        if !fresh.is_empty() {
            (self.on_text)(fresh.clone()).await;
            self.text.push_str(&fresh);
        }
    }
}

/// Joins token pieces into UTF-8 text. A piece may end inside a character, so an incomplete
/// trailing sequence waits for the next piece; invalid bytes become U+FFFD.
#[derive(Debug, Default)]
pub struct Utf8Assembler {
    pending: Vec<u8>,
}

impl Utf8Assembler {
    /// Adds a piece and returns the text it completes (possibly empty).
    pub fn push(&mut self, piece: &[u8]) -> String {
        self.pending.extend_from_slice(piece);
        let mut text = String::new();
        let mut start = 0;
        loop {
            match std::str::from_utf8(&self.pending[start..]) {
                Ok(valid) => {
                    text.push_str(valid);
                    start = self.pending.len();
                    break;
                }
                Err(error) => {
                    let valid_end = start + error.valid_up_to();
                    text.push_str(&String::from_utf8_lossy(&self.pending[start..valid_end]));
                    let Some(invalid_len) = error.error_len() else {
                        start = valid_end;
                        break;
                    };
                    text.push(char::REPLACEMENT_CHARACTER);
                    start = valid_end + invalid_len;
                }
            }
        }
        self.pending.drain(..start);
        text
    }

    /// Flushes whatever is left, replacing an incomplete trailing sequence with U+FFFD.
    pub fn finish(self) -> String {
        String::from_utf8_lossy(&self.pending).into_owned()
    }
}

fn runner_args(launch: &LocalLaunch<'_>) -> [String; 7] {
    [
        launch.begin.to_string(),
        launch.end.to_string(),
        launch.context.to_string(),
        launch.gpu_layers.to_string(),
        launch.budget_bytes.to_string(),
        launch.prompt.to_owned(),
        NO_LOGITS.to_owned(),
    ]
}

async fn read_metadata<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    label: &str,
    metadata_timeout: Duration,
) -> anyhow::Result<StageMetadata> {
    timeout(metadata_timeout, wire::read_metadata(reader))
        .await
        .with_context(|| format!("{label} sent no metadata within {metadata_timeout:?}"))?
        .with_context(|| format!("failed to read metadata from {label}"))
}

fn check_metadata(
    link: &StageLink,
    expected: &StageExpectation,
    embd: u32,
    is_last: bool,
) -> anyhow::Result<()> {
    let meta = &link.metadata;
    let label = &link.label;
    ensure!(
        meta.protocol == PROTOCOL_VERSION,
        "{label} speaks protocol {}, expected {PROTOCOL_VERSION}",
        meta.protocol
    );
    ensure!(
        (meta.begin, meta.end, meta.layers) == (expected.begin, expected.end, expected.layers),
        "{label} serves layers {}..{} of {}, expected {}..{} of {}",
        meta.begin,
        meta.end,
        meta.layers,
        expected.begin,
        expected.end,
        expected.layers
    );
    ensure!(
        meta.embd == embd,
        "{label} has hidden size {}, but the first stage has {embd}",
        meta.embd
    );
    ensure!(
        meta.is_final() == is_last,
        "{label} is{} the final stage but sits {} the pipeline",
        if meta.is_final() { "" } else { " not" },
        if is_last {
            "at the end of"
        } else {
            "in the middle of"
        }
    );
    check_identity(meta, expected).with_context(|| format!("{label} has the wrong model"))
}

fn check_identity(meta: &StageMetadata, expected: &StageExpectation) -> anyhow::Result<()> {
    let want = &expected.model_sha256;
    let source = &meta.source_sha256;
    ensure!(
        source.is_empty() || same_sha256(source, want),
        "runner source_sha256 {source} does not match model_sha256 {want}"
    );
    match expected.identity {
        StageIdentity::Remote => match &meta.model_sha256 {
            Some(got) if same_sha256(got, want) => Ok(()),
            Some(got) => bail!("relay model_sha256 {got} does not match model_sha256 {want}"),
            None => bail!("the relay did not report model_sha256"),
        },
        StageIdentity::Local { artifact_verified } => {
            ensure!(
                !source.is_empty() || artifact_verified,
                "the runner reports no source_sha256 and no artifact_sha256 is configured \
                 for this stage, so its identity cannot be established"
            );
            Ok(())
        }
    }
}

fn token_len(tokens: &[u32]) -> u32 {
    u32::try_from(tokens.len()).unwrap_or(u32::MAX)
}

/// Logs every line a local runner writes to stderr until the pipe closes.
async fn forward_stderr(label: String, stderr: impl AsyncRead + Unpin) {
    let mut lines = BufReader::new(stderr);
    let mut line = Vec::new();
    loop {
        line.clear();
        match (&mut lines)
            .take(MAX_RUNNER_LOG_LINE_BYTES)
            .read_until(b'\n', &mut line)
            .await
        {
            Ok(0) => return,
            Ok(_) => {
                let truncated =
                    line.last() != Some(&b'\n') && line.len() as u64 == MAX_RUNNER_LOG_LINE_BYTES;
                let text = String::from_utf8_lossy(line.trim_ascii_end());
                info!(target: RUNNER_LOG_TARGET, stage = %label, truncated, "{text}");
            }
            Err(error) => {
                warn!(target: RUNNER_LOG_TARGET, stage = %label, %error, "failed to read runner stderr");
                return;
            }
        }
    }
}

/// Waits for a runner whose stdin has closed to exit, killing it after the grace period (or
/// at once when `kill` is set). Explicit kill-then-wait so Windows has released the process
/// before the next generation starts.
async fn stop_child(mut child: Child, label: &str, kill: bool) {
    if !kill && let Ok(status) = timeout(CHILD_EXIT_GRACE, child.wait()).await {
        info!(stage = %label, ?status, "shard runner exited");
        return;
    }
    if let Err(error) = child.start_kill() {
        warn!(stage = %label, %error, "failed to kill shard runner");
    }
    match child.wait().await {
        Ok(status) => info!(stage = %label, ?status, "shard runner stopped"),
        Err(error) => warn!(stage = %label, %error, "failed to reap shard runner"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assemble(pieces: &[&[u8]]) -> (Vec<String>, String) {
        let mut assembler = Utf8Assembler::default();
        let emitted = pieces.iter().map(|piece| assembler.push(piece)).collect();
        (emitted, assembler.finish())
    }

    #[test]
    fn ascii_pieces_pass_straight_through() {
        let (emitted, rest) = assemble(&[b"He", b"llo"]);
        assert_eq!(emitted, ["He", "llo"]);
        assert_eq!(rest, "");
    }

    #[test]
    fn a_character_split_across_pieces_waits_for_its_last_byte() {
        let euro = "€".as_bytes();
        let (emitted, rest) = assemble(&[&[b'a', euro[0]], &euro[1..2], &[euro[2], b'b']]);
        assert_eq!(emitted, ["a", "", "€b"]);
        assert_eq!(rest, "");
    }

    #[test]
    fn invalid_bytes_become_replacement_characters() {
        let (emitted, rest) = assemble(&[b"a\xffb"]);
        assert_eq!(emitted, ["a\u{fffd}b"]);
        assert_eq!(rest, "");
    }

    #[test]
    fn an_incomplete_tail_is_flushed_lossily_at_the_end() {
        let (emitted, rest) = assemble(&[b"ok", &"€".as_bytes()[..2]]);
        assert_eq!(emitted, ["ok", ""]);
        assert_eq!(rest, "\u{fffd}");
    }

    fn metadata(begin: u32, end: u32) -> StageMetadata {
        StageMetadata {
            protocol: PROTOCOL_VERSION,
            begin,
            end,
            layers: 4,
            embd: 8,
            vocab: 300,
            allocated_bytes: 0,
            source_sha256: String::new(),
            tokens: vec![0, 1],
            model_sha256: None,
            artifact_sha256: None,
            runner_sha256: None,
            extra: serde_json::Map::new(),
        }
    }

    fn expectation(identity: StageIdentity) -> StageExpectation {
        StageExpectation {
            begin: 0,
            end: 4,
            layers: 4,
            model_sha256: "ab".repeat(32),
            identity,
        }
    }

    #[test]
    fn local_identity_needs_a_matching_source_or_a_verified_artifact() {
        let unverified = expectation(StageIdentity::Local {
            artifact_verified: false,
        });
        let verified = expectation(StageIdentity::Local {
            artifact_verified: true,
        });
        let mut meta = metadata(0, 4);
        assert!(check_identity(&meta, &unverified).is_err());
        assert!(check_identity(&meta, &verified).is_ok());
        meta.source_sha256 = "AB".repeat(32);
        assert!(check_identity(&meta, &unverified).is_ok());
        meta.source_sha256 = "cd".repeat(32);
        assert!(check_identity(&meta, &verified).is_err());
    }

    #[tokio::test]
    async fn limits_count_totals_across_a_resume() {
        let limits = GenerationLimits {
            max_tokens: 5,
            context: 8,
        };
        let mut emitted = Vec::new();
        let mut state = GenerationState::new(limits, |text: String| {
            emitted.push(text);
            std::future::ready(())
        });
        state.adopt_prompt(&[0, 1, 2], "final").unwrap();
        // A first attempt produced three tokens before failing: prompt plus those still fit.
        for byte in *b"abc" {
            state.push(u32::from(byte), &[byte]).await;
        }
        assert!(!state.spent());
        // One more token fills the context exactly (3 + 4 = 8), so a resume still fits; at
        // max_tokens (5) the budget is spent even though the context has room.
        state.push(u32::from(b'd'), b"d").await;
        assert!(!state.spent());
        state.push(u32::from(b'e'), b"e").await;
        assert!(state.spent());
        let roomy = GenerationLimits {
            max_tokens: 10,
            ..limits
        };
        assert!(roomy.spent_before_resume(5, 9), "9 positions overflow 8");
        assert!(!roomy.spent_before_resume(5, 8));
        assert!(
            roomy.spent_after_token(4, 8),
            "no position for the next token"
        );
        assert!(roomy.spent_after_token(10, 3));
        assert!(!roomy.spent_after_token(9, 7));
        let generation = state.finish(FinishReason::Length).await;
        drop(state);
        assert_eq!(
            (generation.prompt_tokens, generation.completion_tokens),
            (3, 5)
        );
        assert_eq!(generation.text, "abcde");
        assert_eq!(emitted.concat(), "abcde");
    }

    #[tokio::test]
    async fn a_resume_keeps_a_split_character_and_needs_the_same_prompt() {
        let limits = GenerationLimits {
            max_tokens: 9,
            context: 64,
        };
        let mut emitted = Vec::new();
        let mut state = GenerationState::new(limits, |text: String| {
            emitted.push(text);
            std::future::ready(())
        });
        state.adopt_prompt(&[0, 7], "final").unwrap();
        let euro = "€".as_bytes();
        state.push(1, &euro[..2]).await;
        // The next attempt's final stage must tokenize the prompt the same way.
        assert!(state.adopt_prompt(&[0, 8], "final").is_err());
        state.adopt_prompt(&[0, 7], "final").unwrap();
        state.push(2, &euro[2..]).await;
        assert_eq!(state.history(), [0, 7, 1, 2]);
        assert_eq!(state.finish(FinishReason::Stop).await.text, "€");
        drop(state);
        assert_eq!(emitted, ["€"]);
    }

    #[test]
    fn errors_are_classified_by_their_cause() {
        let closed: anyhow::Error = io::Error::from(io::ErrorKind::UnexpectedEof).into();
        let worker = iroh::SecretKey::from_bytes(&[3; 32]).public();
        let fault = StageFault {
            stage: 0,
            worker: Some(worker),
        };
        let step_failure = closed.context("failed at step 2").context(fault);
        let classified = classify(&step_failure);
        assert_eq!(classified.class, FailureClass::Retryable);
        assert_eq!(classified.fault, Some(fault));
        let dial = anyhow::anyhow!("no route").context(Unreachable("failed to connect".into()));
        assert_eq!(format!("{dial}"), "failed to connect");
        assert_eq!(classify(&dial).class, FailureClass::Retryable);
        let mismatch = anyhow::anyhow!("serves layers 0..1").context("stage 0");
        assert_eq!(classify(&mismatch).class, FailureClass::Fatal);
        assert_eq!(classify(&mismatch).fault, None);
        let too_long: anyhow::Error = PromptTooLong {
            prompt_tokens: 9,
            context: 8,
        }
        .into();
        assert_eq!(classify(&too_long).class, FailureClass::Fatal);
    }

    #[test]
    fn remote_identity_needs_the_relay_model_hash() {
        let remote = expectation(StageIdentity::Remote);
        let mut meta = metadata(0, 4);
        assert!(check_identity(&meta, &remote).is_err());
        meta.model_sha256 = Some("cd".repeat(32));
        assert!(check_identity(&meta, &remote).is_err());
        meta.model_sha256 = Some("ab".repeat(32));
        assert!(check_identity(&meta, &remote).is_ok());
    }
}
