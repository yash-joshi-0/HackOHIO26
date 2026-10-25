//! Test stand-in for david-codex's `shard_runner`: the same argv and stdio protocol over a toy
//! model whose arithmetic makes every broken link in a stage chain detectable.
//!
//! Usage: `fake_shard_runner MODEL BEGIN END CONTEXT GPU_LAYERS BUDGET_BYTES PROMPT RETURN_LOGITS`
//!
//! MODEL is a JSON file such as `examples/fake-shard-model.json`:
//! `{"layers":4,"embd":8,"vocab":300,"reply":"Hello from the shards.","source_sha256":"..."}`.
//! The optional `source_sha256` is reported in the metadata line as an exported shard's model
//! identity (empty when absent, like a plain GGUF).
//!
//! - The prompt tokenizes to BOS (0) followed by each prompt byte plus one.
//! - The first stage (`BEGIN == 0`) maps token `t` to a row of `embd` copies of `t + END`. A later
//!   stage checks that every row is `embd` copies of `t + BEGIN` for a token `t` and exits with
//!   "broken stage chain" otherwise; a non-final stage then outputs rows of `t + END`.
//! - The final stage (`END == layers`) emits the reply bytes in order, one per step, starting
//!   with the step that reaches the end of its own prompt (earlier prefill steps return token 0
//!   with an empty piece), then end-of-generation (`vocab - 1`). Every input past the prompt
//!   must be the reply token for its position, or it exits with "fed back wrong token". So a
//!   resumed generation (prompt plus already generated tokens in one prefill) continues the
//!   reply where it left off, as a real model would. A final stage whose own prompt is not the
//!   one being run (a ring terminal gets a placeholder) starts the reply after that step.
//! - Like the real runner it enforces step/position order, batch sizes, the context bound and the
//!   opcode for its stage, and exits 1 with a message on stderr on any violation.
//!
//! Environment: `FAKE_SHARD_FAIL_AT_STEP=<n>` exits 1 on receiving step `n`. With
//! `FAKE_SHARD_FAIL_ONCE_FILE=<path>` as well, only the first runner to reach that step fails:
//! it creates `<path>`, and a runner that finds the file already there carries on.
//! `FAKE_SHARD_DELAY_AT_STEP_MS=<ms>` delays each step before replying.
//! `FAKE_SHARD_LOG=<path>` appends `<pid> <begin>:<end> step=<s> position=<p> count=<n>` per
//! received request header.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail, ensure};
use llm_cluster::shard::wire::{
    self, MAX_CONTEXT, MAX_EMBD, Opcode, RequestHeader, StageMetadata, StepReply,
};
use serde::Deserialize;
use serde_json::{Map, Value};
use tokio::io::{AsyncRead, AsyncWrite, BufReader};

/// Arguments after the program name, as for the real runner.
const ARG_COUNT: usize = 8;

/// Smallest context window the real runner accepts.
const MIN_CONTEXT: u32 = 32;

/// Protocol version reported in the metadata line.
const PROTOCOL_VERSION: u32 = 1;

/// Beginning-of-sequence token that starts every tokenized prompt.
const BOS_TOKEN: u32 = 0;

/// BOS, one token per byte value, and an end-of-generation token that collides with neither.
const MIN_VOCAB: u32 = 258;

/// Token returned by prefill steps that do not yet reach the end of the prompt.
const PREFILL_TOKEN: u32 = 0;

/// Token reported by a stage that does not sample.
const NO_TOKEN: i64 = -1;

/// Exits 1 on receiving the step it names.
const FAIL_AT_STEP_ENV: &str = "FAKE_SHARD_FAIL_AT_STEP";
const DELAY_AT_STEP_ENV: &str = "FAKE_SHARD_DELAY_AT_STEP_MS";

/// Marker file that limits `FAKE_SHARD_FAIL_AT_STEP` to the first runner that reaches it.
const FAIL_ONCE_FILE_ENV: &str = "FAKE_SHARD_FAIL_ONCE_FILE";

/// Path to append one line per received request header.
const LOG_ENV: &str = "FAKE_SHARD_LOG";

/// The toy model loaded from MODEL.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FakeModel {
    layers: u32,
    embd: u32,
    vocab: u32,
    reply: String,
    /// Model identity reported as the metadata's `source_sha256`; empty when absent.
    #[serde(default)]
    source_sha256: String,
}

/// Parsed command line.
struct Args {
    model: PathBuf,
    begin: u32,
    end: u32,
    context: u32,
    prompt: String,
}

/// One running stage and its session state.
struct Stage {
    model: FakeModel,
    begin: u32,
    end: u32,
    context: u32,
    prompt_tokens: Vec<u32>,
    fail_at_step: Option<u32>,
    delay_at_step: Duration,
    fail_once_file: Option<PathBuf>,
    log: Option<PathBuf>,
    expected_step: u32,
    expected_position: u32,
    /// Position of the first reply token, fixed by the batch that reaches the prompt's end.
    reply_anchor: Option<u32>,
    /// Whether every input so far at a prompt position was this stage's own prompt token.
    own_prompt: bool,
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("fake shard runner: {error:#}");
        // Exit at once: a pending blocking stdin read would otherwise delay runtime shutdown.
        std::process::exit(1);
    }
}

async fn run() -> anyhow::Result<()> {
    let args = parse_args(std::env::args_os().skip(1))?;
    let model = load_model(&args.model)?;
    let mut stage = Stage::new(model, args)?;
    let mut input = BufReader::new(tokio::io::stdin());
    let mut output = tokio::io::stdout();
    wire::write_metadata(&mut output, &stage.metadata()).await?;
    while let Some(header) = wire::read_request_header(&mut input).await? {
        stage.serve_step(header, &mut input, &mut output).await?;
    }
    Ok(())
}

fn parse_args(args: impl Iterator<Item = std::ffi::OsString>) -> anyhow::Result<Args> {
    let args = args
        .map(|arg| {
            arg.into_string()
                .map_err(|_| anyhow::anyhow!("non-UTF-8 argument"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let [
        model,
        begin,
        end,
        context,
        gpu_layers,
        budget,
        prompt,
        return_logits,
    ] = <[String; ARG_COUNT]>::try_from(args).map_err(|_| {
        anyhow::anyhow!(
            "usage: runner MODEL BEGIN END CTX GPU_LAYERS BUDGET_BYTES PROMPT RETURN_LOGITS"
        )
    })?;
    let begin: u32 = parse_number(&begin, "BEGIN")?;
    let end: u32 = parse_number(&end, "END")?;
    let context: u32 = parse_number(&context, "CTX")?;
    let budget: u64 = parse_number(&budget, "BUDGET_BYTES")?;
    parse_number::<i64>(&gpu_layers, "GPU_LAYERS")?;
    let return_logits: i64 = parse_number(&return_logits, "RETURN_LOGITS")?;
    ensure!(
        begin < end && (MIN_CONTEXT..=MAX_CONTEXT).contains(&context) && budget > 0,
        "invalid partition/context/budget"
    );
    ensure!(
        return_logits == 0,
        "RETURN_LOGITS is not supported by the fake runner"
    );
    Ok(Args {
        model: PathBuf::from(model),
        begin,
        end,
        context,
        prompt,
    })
}

fn parse_number<T: std::str::FromStr>(value: &str, name: &str) -> anyhow::Result<T> {
    value
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid {name} {value:?}"))
}

fn load_model(path: &Path) -> anyhow::Result<FakeModel> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read fake model {}", path.display()))?;
    let model: FakeModel = serde_json::from_str(&text)
        .with_context(|| format!("invalid fake model {}", path.display()))?;
    ensure!(
        model.vocab >= MIN_VOCAB,
        "fake model vocab must be at least {MIN_VOCAB}"
    );
    ensure!(
        (1..=MAX_EMBD).contains(&model.embd),
        "fake model embd must be in 1..={MAX_EMBD}"
    );
    Ok(model)
}

/// BOS followed by each prompt byte plus one.
fn tokenize(prompt: &str) -> Vec<u32> {
    std::iter::once(BOS_TOKEN)
        .chain(prompt.bytes().map(|byte| u32::from(byte) + 1))
        .collect()
}

impl Stage {
    fn new(model: FakeModel, args: Args) -> anyhow::Result<Self> {
        ensure!(
            args.end <= model.layers,
            "range end exceeds the model's layers"
        );
        let fail_at_step = match std::env::var(FAIL_AT_STEP_ENV) {
            Ok(value) => Some(parse_number(&value, FAIL_AT_STEP_ENV)?),
            Err(_) => None,
        };
        let delay_at_step = match std::env::var(DELAY_AT_STEP_ENV) {
            Ok(value) => Duration::from_millis(parse_number(&value, DELAY_AT_STEP_ENV)?),
            Err(_) => Duration::ZERO,
        };
        Ok(Self {
            model,
            begin: args.begin,
            end: args.end,
            context: args.context,
            prompt_tokens: tokenize(&args.prompt),
            fail_at_step,
            delay_at_step,
            fail_once_file: std::env::var_os(FAIL_ONCE_FILE_ENV).map(PathBuf::from),
            log: std::env::var_os(LOG_ENV).map(PathBuf::from),
            expected_step: 0,
            expected_position: 0,
            reply_anchor: None,
            own_prompt: true,
        })
    }

    fn is_final(&self) -> bool {
        self.end == self.model.layers
    }

    fn metadata(&self) -> StageMetadata {
        StageMetadata {
            protocol: PROTOCOL_VERSION,
            begin: self.begin,
            end: self.end,
            layers: self.model.layers,
            embd: self.model.embd,
            vocab: self.model.vocab,
            allocated_bytes: 0,
            source_sha256: self.model.source_sha256.clone(),
            tokens: self.prompt_tokens.clone(),
            model_sha256: None,
            artifact_sha256: None,
            runner_sha256: None,
            extra: runner_extras(),
        }
    }

    async fn serve_step<R, W>(
        &mut self,
        header: RequestHeader,
        input: &mut R,
        output: &mut W,
    ) -> anyhow::Result<()>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        self.log_header(&header)?;
        if self.fail_at_step == Some(header.step) && self.first_to_fail()? {
            bail!("{FAIL_AT_STEP_ENV} reached at step {}", header.step);
        }
        self.check_order(&header)?;
        let tokens = self.read_input(&header, input).await?;
        let (reply, floats) = if self.is_final() {
            (self.sample(&header, &tokens)?, Vec::new())
        } else {
            self.forward(&header, &tokens)
        };
        if !self.delay_at_step.is_zero() {
            tokio::time::sleep(self.delay_at_step).await;
        }
        wire::write_step(output, &reply, &floats).await?;
        self.expected_step += 1;
        self.expected_position += header.token_count;
        Ok(())
    }

    /// Whether this runner should fail: always without a fail-once file, otherwise only if it
    /// is the one that creates the file.
    fn first_to_fail(&self) -> anyhow::Result<bool> {
        let Some(path) = &self.fail_once_file else {
            return Ok(true);
        };
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(_) => Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
            Err(error) => {
                Err(error).with_context(|| format!("failed to create {}", path.display()))
            }
        }
    }

    fn log_header(&self, header: &RequestHeader) -> anyhow::Result<()> {
        let Some(path) = &self.log else {
            return Ok(());
        };
        let mut log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        writeln!(
            log,
            "{} {}:{} step={} position={} count={}",
            std::process::id(),
            self.begin,
            self.end,
            header.step,
            header.position,
            header.token_count
        )
        .with_context(|| format!("failed to write {}", path.display()))
    }

    fn check_order(&self, header: &RequestHeader) -> anyhow::Result<()> {
        let expected_opcode = if self.begin == 0 {
            Opcode::Tokens
        } else {
            Opcode::Hidden
        };
        let in_context = header
            .position
            .checked_add(header.token_count)
            .is_some_and(|end| end <= self.context);
        ensure!(
            header.step == self.expected_step
                && header.position == self.expected_position
                && in_context
                && header.opcode == expected_opcode,
            "invalid/out-of-order execution request {header:?}; expected step {} at position {}",
            self.expected_step,
            self.expected_position
        );
        Ok(())
    }

    /// Reads the payload and recovers the token behind each row.
    async fn read_input<R: AsyncRead + Unpin>(
        &self,
        header: &RequestHeader,
        input: &mut R,
    ) -> anyhow::Result<Vec<u32>> {
        if self.begin == 0 {
            let tokens = wire::read_tokens(input, header.token_count).await?;
            ensure!(
                tokens.iter().all(|&token| token < self.model.vocab),
                "invalid token"
            );
            return Ok(tokens);
        }
        let embd = self.model.embd as usize;
        let values = wire::read_hidden(input, header.token_count as usize * embd).await?;
        values
            .chunks_exact(embd)
            .map(|row| self.decode_row(row))
            .collect()
    }

    fn decode_row(&self, row: &[f32]) -> anyhow::Result<u32> {
        let first = row[0];
        let token = f64::from(first) - f64::from(self.begin);
        let valid = first.is_finite()
            && row.iter().all(|&value| value == first)
            && token.fract() == 0.0
            && (0.0..f64::from(self.model.vocab)).contains(&token);
        ensure!(
            valid,
            "broken stage chain: row starting {first} is not a token offset by {}",
            self.begin
        );
        Ok(token as u32)
    }

    /// A non-final stage: rows of `t + end` for each token.
    fn forward(&self, header: &RequestHeader, tokens: &[u32]) -> (StepReply, Vec<f32>) {
        let floats: Vec<f32> = tokens
            .iter()
            .flat_map(|&token| {
                std::iter::repeat_n((token + self.end) as f32, self.model.embd as usize)
            })
            .collect();
        let reply = step_reply(header.step, floats.len(), NO_TOKEN, false, &[]);
        (reply, floats)
    }

    /// The final stage: once the prompt has been seen, the reply byte for the position after
    /// the batch. Inputs past the reply's start must be the reply tokens for their positions.
    fn sample(&mut self, header: &RequestHeader, tokens: &[u32]) -> anyhow::Result<StepReply> {
        let end = header.position + header.token_count;
        let anchor = match self.reply_anchor {
            Some(anchor) => anchor,
            None => {
                self.note_prompt(header.position, tokens);
                let prompt_len = u32::try_from(self.prompt_tokens.len()).unwrap_or(u32::MAX);
                if end < prompt_len {
                    let token = i64::from(PREFILL_TOKEN);
                    return Ok(step_reply(header.step, 0, token, false, &[]));
                }
                // The stage's own prompt is the one being run: the reply starts right after
                // it, even inside this batch (a resume). Otherwise (a ring terminal started
                // with a placeholder prompt) it starts after this batch.
                let anchor = if self.own_prompt { prompt_len } else { end };
                self.reply_anchor = Some(anchor);
                anchor
            }
        };
        for (position, &token) in (header.position..).zip(tokens) {
            if let Some(index) = position.checked_sub(anchor) {
                let (expected, _) = self.reply_token(index as usize);
                ensure!(
                    token == expected,
                    "fed back wrong token {token} at position {position}; emitted {expected}"
                );
            }
        }
        let (token, piece) = self.reply_token((end - anchor) as usize);
        let eog = piece.is_none();
        let piece: Vec<u8> = piece.into_iter().collect();
        Ok(step_reply(header.step, 0, i64::from(token), eog, &piece))
    }

    /// Notes whether the inputs at prompt positions are this stage's own prompt tokens.
    fn note_prompt(&mut self, position: u32, tokens: &[u32]) {
        for (position, &token) in (position as usize..).zip(tokens) {
            if let Some(&own) = self.prompt_tokens.get(position) {
                self.own_prompt &= own == token;
            }
        }
    }

    /// The token emitted at `index` of the reply and its byte, or end-of-generation (no byte).
    fn reply_token(&self, index: usize) -> (u32, Option<u8>) {
        match self.model.reply.as_bytes().get(index) {
            Some(&byte) => (u32::from(byte) + 1, Some(byte)),
            None => (self.model.vocab - 1, None),
        }
    }
}

/// Fields the real runner prints that the protocol does not interpret.
fn runner_extras() -> Map<String, Value> {
    let mut extra = Map::new();
    extra.insert("max_rss_bytes".to_string(), Value::from(0));
    extra.insert("vm".to_string(), Value::Null);
    extra
}

fn step_reply(step: u32, n_floats: usize, token: i64, eog: bool, piece: &[u8]) -> StepReply {
    StepReply {
        step,
        n_floats,
        token,
        eog,
        piece_hex: wire::encode_hex(piece),
        compute_ms: 0.0,
        observed_peak_buffer_bytes: 0,
        extra: runner_extras(),
    }
}
