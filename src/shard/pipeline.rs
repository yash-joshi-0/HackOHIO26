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
//! runner serves one sequence, so links are single-use: [`Pipeline::generate`] consumes them,
//! and any error closes every link and fails the generation without retrying.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, bail, ensure};
use iroh::endpoint::{Connection, VarInt};
use iroh::{Endpoint, EndpointId};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::time::{timeout, timeout_at};
use tracing::{debug, info, warn};

use crate::admission::AdmittedChild as Child;

use crate::protocol::RPC_BRIDGE_ALPN;
use crate::rpc_bridge::path_kind;
use crate::shard::manifest::same_sha256;
use crate::shard::wire::{self, MAX_BATCH_TOKENS, StageInput, StageMetadata, StepReply};

/// Bound on waiting for a stage's metadata line; loading a large model can take minutes.
pub const METADATA_TIMEOUT: Duration = Duration::from_secs(300);

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
    Process(Child),
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
            guard: LinkGuard::Process(child),
            label,
        })
    }

    /// Dials `worker` on the RPC bridge ALPN, opens one stream, sends the client preface and
    /// reads the forwarded metadata within `metadata_timeout`.
    pub async fn connect_remote(
        endpoint: &Endpoint,
        worker: EndpointId,
        metadata_timeout: Duration,
    ) -> anyhow::Result<Self> {
        let label = format!("remote stage on {}", worker.fmt_short());
        let connect_started = Instant::now();
        let connection = endpoint
            .connect(worker, RPC_BRIDGE_ALPN)
            .await
            .with_context(|| format!("failed to connect to {label}"))?;
        debug!(stage = %label, path = path_kind(&connection), connect_us = connect_started.elapsed().as_micros(), "remote stage connected");
        let (mut send, recv) = connection
            .open_bi()
            .await
            .with_context(|| format!("failed to open a stream to {label}"))?;
        wire::write_preface(&mut send)
            .await
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
        })
    }

    /// The stage's startup metadata.
    pub fn metadata(&self) -> &StageMetadata {
        &self.metadata
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
            LinkGuard::Process(child) => stop_child(child, &label, kill).await,
            LinkGuard::Stream(connection) => {
                connection.close(CONNECTION_DONE, CONNECTION_DONE_REASON);
            }
        }
    }
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

/// Limits for one generation.
#[derive(Clone, Copy, Debug)]
pub struct GenerationLimits {
    /// Most completion tokens to produce (at least 1).
    pub max_tokens: u32,
    /// Context window every stage was started with.
    pub context: u32,
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

/// Validated stage links, ready for one generation.
pub struct Pipeline {
    links: Vec<StageLink>,
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
            embd,
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

    /// Runs greedy decoding to completion, calling `on_text` with each new piece of complete
    /// UTF-8 text as it is produced, then closes every link. On any error the links are killed
    /// and the error is returned; nothing is retried.
    pub async fn generate<F: FnMut(&str)>(
        mut self,
        limits: GenerationLimits,
        on_text: F,
    ) -> anyhow::Result<Generation> {
        let result = self.run(limits, on_text).await;
        let kill = result.is_err();
        for link in self.links {
            link.close(kill).await;
        }
        result
    }

    async fn run<F: FnMut(&str)>(
        &mut self,
        limits: GenerationLimits,
        on_text: F,
    ) -> anyhow::Result<Generation> {
        ensure!(limits.max_tokens >= 1, "max_tokens must be at least 1");
        self.ensure_prompt_fits(limits.context)?;
        let mut decoder = Decoder::default();
        let mut reply = self.prefill(&mut decoder).await?;
        let mut text = TextSink {
            assembler: Utf8Assembler::default(),
            text: String::new(),
            on_text,
        };
        let finish = loop {
            if reply.eog {
                break FinishReason::Stop;
            }
            let token = self.sampled_token(&reply)?;
            text.push(&reply.piece_bytes()?);
            decoder.completion_tokens += 1;
            if decoder.completion_tokens >= limits.max_tokens || decoder.position >= limits.context
            {
                break FinishReason::Length;
            }
            reply = self.step(&mut decoder, &[token]).await?;
        };
        Ok(Generation {
            text: text.finish(),
            finish,
            prompt_tokens: self.prompt_tokens(),
            completion_tokens: decoder.completion_tokens,
        })
    }

    /// Feeds the prompt in batches and returns the final stage's reply to the last batch.
    async fn prefill(&mut self, decoder: &mut Decoder) -> anyhow::Result<StepReply> {
        let prompt = std::mem::take(&mut self.prompt_tokens);
        let mut last = None;
        for batch in prompt.chunks(MAX_BATCH_TOKENS as usize) {
            last = Some(self.step(decoder, batch).await?);
        }
        self.prompt_tokens = prompt;
        last.context("the prompt has no tokens")
    }

    /// Sends one batch of tokens through every stage and returns the final stage's reply.
    async fn step(&mut self, decoder: &mut Decoder, tokens: &[u32]) -> anyhow::Result<StepReply> {
        let started = Instant::now();
        let (step, position) = (decoder.step, decoder.position);
        let count = tokens.len();
        let row_values = count * self.embd as usize;
        let stage_count = self.links.len();
        let last = stage_count - 1;
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
            let max_floats = if index == last { 0 } else { row_values };
            let output = link
                .exchange(step, position, &input, max_floats, self.step_timeout)
                .await?;
            if index == last {
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
        bail!("the pipeline has no final stage")
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
    completion_tokens: u32,
}

impl Decoder {
    fn advance(&mut self, count: usize) {
        self.step += 1;
        self.position += u32::try_from(count).unwrap_or(u32::MAX);
    }
}

/// Collects the completion and hands each newly completed piece of text to the caller.
struct TextSink<F> {
    assembler: Utf8Assembler,
    text: String,
    on_text: F,
}

impl<F: FnMut(&str)> TextSink<F> {
    fn push(&mut self, piece: &[u8]) {
        let fresh = self.assembler.push(piece);
        self.deliver(&fresh);
    }

    /// Flushes any incomplete trailing bytes and returns the whole completion.
    fn finish(mut self) -> String {
        let rest = std::mem::take(&mut self.assembler).finish();
        self.deliver(&rest);
        self.text
    }

    fn deliver(&mut self, fresh: &str) {
        if !fresh.is_empty() {
            (self.on_text)(fresh);
            self.text.push_str(fresh);
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
        match lines.read_until(b'\n', &mut line).await {
            Ok(0) => return,
            Ok(_) => {
                let text = String::from_utf8_lossy(line.trim_ascii_end());
                info!(target: RUNNER_LOG_TARGET, stage = %label, "{text}");
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
