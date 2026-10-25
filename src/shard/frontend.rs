//! OpenAI-shaped HTTP frontend that serves completions from a sharded model.
//!
//! [`serve_frontend`] listens on a loopback address and answers like llama-server, so the
//! leader's job API ([`crate::api`]) can sit in front of it unchanged:
//!
//! - `GET /health` answers 200.
//! - `POST /v1/chat/completions` renders the messages with the manifest's chat template, opens
//!   fresh stage links (a runner serves one sequence, so every request gets new processes and
//!   streams), and runs one greedy generation. Sampling fields such as `temperature`, `top_p` or
//!   `seed` are accepted and ignored; `max_tokens` / `max_completion_tokens` are honoured up to
//!   the manifest's `max_tokens`.
//!
//! With an EXPERIMENTAL ring manifest each request instead starts the local stage 0 and a fresh
//! shard-ring session ([`RingLink`]); [`FrontendConfig::ring`] must then carry the registry the
//! leader's endpoint delivers result channels to.
//!
//! One generation runs at a time; later requests wait for the lock. In pipeline execution a
//! retryable failure (see [`crate::shard::pipeline::classify`]) resumes the generation on fresh
//! links, up to [`MAX_GENERATION_RETRIES`] times: completed text is never sent twice, a remote
//! worker that failed is replaced by the stage's next standby (or, with none left that has not
//! failed, tried again), and local stages are started again. Ring execution is never retried:
//! any ring failure fails the generation.
//! A fatal failure, or one that exhausts the retries, before any response bytes are sent is
//! a JSON error with a 5xx status (400 for a prompt longer than the context). One mid-stream
//! sends an SSE `error` event and ends without `[DONE]`, which makes the job API fail the job
//! without committing history. A client that disconnects cancels its generation at once, with
//! no retry, which kills the local runners and closes the remote streams.

use std::collections::HashMap;
use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex as StdMutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, bail};
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::future::try_join_all;
use futures_util::{StreamExt, stream};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::mpsc::{Receiver, Sender, channel};
use tokio::sync::{Mutex, oneshot};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tracing::{info, warn};

use iroh::EndpointId;

use crate::api::CHAT_COMPLETIONS_PATH;
use crate::shard::manifest::{ShardManifest, StageLocation, StageSpec};
use crate::shard::manifest::{same_sha256, sha256_file};
use crate::shard::pipeline::{
    FailureClass, FinishReason, Generation, GenerationCancelled, GenerationLimits, GenerationState,
    LocalLaunch, METADATA_TIMEOUT, Pipeline, PromptTooLong, StageExpectation, StageFault,
    StageIdentity, StageLink, classify,
};
use crate::shard::ring_origin::{RingLink, RingPlan, RingSettings};
use crate::shard::template::{ChatMessage, ChatRenderer};

/// Liveness probe, as llama-server serves it.
pub const HEALTH_PATH: &str = "/health";

/// Model name reported in completion objects.
const MODEL_NAME: &str = "meshthing-shard";

/// Largest accepted request body; long chat histories exceed axum's 2 MiB default.
const MAX_REQUEST_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Payload that ends an OpenAI-style event stream.
const SSE_DONE: &str = "[DONE]";

/// Content type of a streamed reply.
const EVENT_STREAM: &str = "text/event-stream";

/// `Cache-Control` for a streamed reply: proxies must pass chunks through as they come.
const NO_CACHE: &str = "no-cache";

/// Bounds queued text chunks when a client reads more slowly than inference produces.
const GENERATION_EVENT_BUFFER: usize = 16;

/// Attempts after the first that one generation may make after retryable failures.
pub const MAX_GENERATION_RETRIES: u32 = 2;

/// Delay before the first retry; it doubles before each later one.
const RETRY_BACKOFF: Duration = Duration::from_millis(250);

/// No retry starts once this long has passed since the generation began.
const RETRY_DEADLINE: Duration = Duration::from_secs(15 * 60);

/// Where the frontend listens, what it serves, and the endpoint remote stages are dialled from.
pub struct FrontendConfig {
    /// Address to serve on; must be loopback, anything else is refused.
    pub listen: SocketAddr,
    /// The validated sharded model.
    pub manifest: ShardManifest,
    /// Iroh endpoint used to dial remote stages.
    pub endpoint: iroh::Endpoint,
    /// Required for a ring manifest (`"execution": "ring"`) and ignored otherwise: the registry
    /// the endpoint's accept loop delivers terminals' result channels to, and the readiness wait.
    pub ring: Option<RingSettings>,
}

/// What a ring manifest's generations need besides the manifest.
struct RingRuntime {
    plan: RingPlan,
    settings: RingSettings,
}

/// Serves `GET /health` and `POST /v1/chat/completions` on `config.listen` until `shutdown`
/// resolves, then finishes in-flight requests and returns.
///
/// Before binding it validates the manifest and hashes every local stage file that has a
/// configured `artifact_sha256` (once, on a blocking thread), failing on any mismatch. It also
/// loads the chat template once ([`ChatRenderer::load`]), refusing a pinned template that is
/// missing, does not match its `sha256` or does not compile.
pub async fn serve_frontend(
    config: FrontendConfig,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    ensure_loopback(config.listen)?;
    config.manifest.validate()?;
    let ring = ring_runtime(&config.manifest, config.ring)?;
    let renderer = ChatRenderer::load(&config.manifest.chat_template)?;
    verify_artifacts(&config.manifest).await?;
    let state = Arc::new(FrontendState {
        expectations: stage_expectations(&config.manifest),
        manifest: config.manifest,
        endpoint: config.endpoint,
        ring,
        renderer,
        generation_lock: Arc::new(Mutex::new(())),
    });
    let router = Router::new()
        .route(HEALTH_PATH, get(health))
        .route(CHAT_COMPLETIONS_PATH, post(chat_completions))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .with_state(state);
    let listener = TcpListener::bind(config.listen)
        .await
        .with_context(|| format!("failed to bind shard frontend on {}", config.listen))?;
    let local = listener
        .local_addr()
        .context("failed to read shard frontend address")?;
    info!(listen = %local, "shard frontend listening");
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await
        .context("shard frontend failed")
}

/// State shared by every request handler.
struct FrontendState {
    manifest: ShardManifest,
    endpoint: iroh::Endpoint,
    expectations: Vec<StageExpectation>,
    /// Set for a ring manifest only.
    ring: Option<RingRuntime>,
    /// The manifest's chat template, verified and compiled at startup.
    renderer: ChatRenderer,
    /// Held for the whole of a generation, so only one runs at a time.
    generation_lock: Arc<Mutex<()>>,
}

/// The ring runtime of a ring manifest, which needs `settings`; `None` for a pipeline manifest.
fn ring_runtime(
    manifest: &ShardManifest,
    settings: Option<RingSettings>,
) -> anyhow::Result<Option<RingRuntime>> {
    if !manifest.is_ring() {
        return Ok(None);
    }
    let settings = settings.context(
        "a ring manifest (\"execution\": \"ring\") needs the leader's ring settings: \
         FrontendConfig::ring must carry the result-channel registry its endpoint feeds",
    )?;
    Ok(Some(RingRuntime {
        plan: RingPlan::from_manifest(manifest)?,
        settings,
    }))
}

fn ensure_loopback(listen: SocketAddr) -> anyhow::Result<()> {
    if !listen.ip().to_canonical().is_loopback() {
        bail!("shard frontend listen address {listen} is not loopback; refusing to expose it");
    }
    Ok(())
}

/// Hashes each local stage file that has an `artifact_sha256` and checks it matches.
async fn verify_artifacts(manifest: &ShardManifest) -> anyhow::Result<()> {
    for stage in &manifest.stages {
        let StageLocation::Local {
            model,
            artifact_sha256: Some(expected),
            ..
        } = &stage.location
        else {
            continue;
        };
        let path = model.clone();
        let actual = tokio::task::spawn_blocking(move || sha256_file(&path))
            .await
            .context("artifact hashing task failed")??;
        if !same_sha256(&actual, expected) {
            bail!(
                "local stage {}..{} file {} has SHA-256 {actual}, but artifact_sha256 is {expected}",
                stage.begin,
                stage.end,
                model.display()
            );
        }
        info!(stage = %format!("{}..{}", stage.begin, stage.end), "verified artifact_sha256");
    }
    Ok(())
}

/// What each stage's metadata must show. Configured artifact hashes count as verified because
/// [`verify_artifacts`] has already checked them.
fn stage_expectations(manifest: &ShardManifest) -> Vec<StageExpectation> {
    manifest
        .stages
        .iter()
        .map(|stage| StageExpectation {
            begin: stage.begin,
            end: stage.end,
            layers: manifest.layers,
            model_sha256: manifest.model_sha256.clone(),
            identity: match &stage.location {
                StageLocation::Remote { .. } => StageIdentity::Remote,
                StageLocation::Local {
                    artifact_sha256, ..
                } => StageIdentity::Local {
                    artifact_verified: artifact_sha256.is_some(),
                },
            },
        })
        .collect()
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

/// A parsed chat request.
#[derive(Debug, PartialEq, Eq)]
struct ChatRequest {
    prompt: String,
    stream: bool,
    max_tokens: u32,
}

#[derive(Deserialize)]
struct RawChatRequest {
    messages: Vec<RawMessage>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct RawMessage {
    role: String,
    content: MessageContent,
}

/// Message content: a string, or an array of text parts.
#[derive(Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentPart {
    Text { text: String },
}

impl MessageContent {
    fn into_text(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::Parts(parts) => parts
                .into_iter()
                .map(|ContentPart::Text { text }| text)
                .collect(),
        }
    }
}

/// Parses a chat request, renders its prompt, and caps `max_tokens` at `max_tokens_cap`.
fn parse_chat_request(
    body: &[u8],
    renderer: &ChatRenderer,
    max_tokens_cap: u32,
) -> Result<ChatRequest, String> {
    let raw: RawChatRequest = serde_json::from_slice(body).map_err(|error| {
        format!(
            "invalid chat request (messages need a role and string or text-part content): {error}"
        )
    })?;
    if raw.messages.is_empty() {
        return Err("messages must not be empty".to_owned());
    }
    let requested = raw.max_completion_tokens.or(raw.max_tokens);
    if requested == Some(0) {
        return Err("max_tokens must be at least 1".to_owned());
    }
    let messages: Vec<ChatMessage> = raw
        .messages
        .into_iter()
        .map(|message| ChatMessage {
            role: message.role,
            content: message.content.into_text(),
        })
        .collect();
    let prompt = renderer
        .render(&messages)
        .map_err(|error| error.to_string())?;
    Ok(ChatRequest {
        prompt,
        stream: raw.stream.unwrap_or(false),
        max_tokens: requested.map_or(max_tokens_cap, |asked| asked.min(max_tokens_cap)),
    })
}

/// Progress of one generation, sent from its task to the request handler.
enum Event {
    /// Every stage is up and validated; response bytes may now be sent.
    Started,
    /// New completion text.
    Text(String),
    /// The generation finished.
    Finished(Generation),
    /// The generation failed.
    Failed(Failure),
}

/// A failed generation and the HTTP status it maps to before any bytes are sent.
struct Failure {
    status: StatusCode,
    message: String,
}

impl Failure {
    fn from_error(error: &anyhow::Error) -> Self {
        let status = if error.downcast_ref::<PromptTooLong>().is_some() {
            StatusCode::BAD_REQUEST
        } else {
            StatusCode::BAD_GATEWAY
        };
        Self {
            status,
            message: format!("{error:#}"),
        }
    }

    fn task_lost() -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "the generation task ended unexpectedly".to_owned(),
        }
    }
}

/// Signals cancellation when the client goes away; the task remains alive for pipeline cleanup.
struct GenerationTask {
    cancel: Option<oneshot::Sender<()>>,
    _join: JoinHandle<()>,
}

impl Drop for GenerationTask {
    fn drop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
    }
}

/// Identifiers shared by every object of one completion.
struct Completion {
    id: String,
    created: u64,
}

impl Completion {
    fn new() -> Self {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs());
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            created,
        }
    }

    fn buffered(&self, generation: &Generation) -> Value {
        json!({
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": generation.text},
                "finish_reason": generation.finish.as_str(),
            }],
            "usage": {
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_tokens": generation.prompt_tokens + generation.completion_tokens,
            },
        })
    }

    fn chunk(&self, delta: Value, finish: Option<FinishReason>) -> Bytes {
        let chunk = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish.map(FinishReason::as_str),
            }],
        });
        sse_data(&chunk.to_string())
    }
}

async fn chat_completions(State(state): State<Arc<FrontendState>>, body: Bytes) -> Response {
    let manifest = &state.manifest;
    let request = match parse_chat_request(&body, &state.renderer, manifest.max_tokens) {
        Ok(request) => request,
        Err(message) => return error_response(StatusCode::BAD_REQUEST, &message),
    };
    info!(
        stream = request.stream,
        max_tokens = request.max_tokens,
        "shard generation requested"
    );
    let (events_tx, mut events) = channel(GENERATION_EVENT_BUFFER);
    let (cancel_tx, cancel_rx) = oneshot::channel();
    let task = GenerationTask {
        cancel: Some(cancel_tx),
        _join: tokio::spawn(run_generation(
            Arc::clone(&state),
            request.prompt,
            request.max_tokens,
            events_tx,
            cancel_rx,
        )),
    };
    match events.recv().await {
        Some(Event::Started) => {}
        Some(Event::Failed(failure)) => return failure_response(&failure),
        Some(Event::Text(_) | Event::Finished(_)) | None => {
            return failure_response(&Failure::task_lost());
        }
    }
    if request.stream {
        stream_response(events, task)
    } else {
        buffered_response(events, task).await
    }
}

/// Runs one generation under the lock and reports its progress on `events`.
async fn run_generation(
    state: Arc<FrontendState>,
    prompt: String,
    max_tokens: u32,
    events: Sender<Event>,
    mut cancel: oneshot::Receiver<()>,
) {
    let queued_at = std::time::Instant::now();
    let _turn = tokio::select! {
        biased;
        _ = &mut cancel => return,
        turn = Arc::clone(&state.generation_lock).lock_owned() => turn,
    };
    info!(
        queue_wait_ms = queued_at.elapsed().as_millis(),
        "shard generation acquired slot"
    );
    let event = match generate(&state, &prompt, max_tokens, &events, cancel).await {
        Ok(generation) => {
            info!(
                finish = generation.finish.as_str(),
                completion_tokens = generation.completion_tokens,
                "shard generation finished"
            );
            Event::Finished(generation)
        }
        Err(error) if error.downcast_ref::<GenerationCancelled>().is_some() => {
            info!("client disconnected; shard generation canceled after cleanup");
            return;
        }
        Err(error) => {
            warn!(error = %format!("{error:#}"), "shard generation failed");
            Event::Failed(Failure::from_error(&error))
        }
    };
    // The handler may be gone (client disconnected); then nobody needs the outcome.
    let _ = events.send(event).await;
}

/// Runs one generation: a single attempt in ring execution, which is never retried, or the
/// retrying pipeline generation otherwise.
async fn generate(
    state: &FrontendState,
    prompt: &str,
    max_tokens: u32,
    events: &Sender<Event>,
    cancel: oneshot::Receiver<()>,
) -> anyhow::Result<Generation> {
    match &state.ring {
        Some(ring) => generate_ring(state, ring, prompt, max_tokens, events, cancel).await,
        None => generate_pipeline(state, prompt, max_tokens, events, cancel).await,
    }
}

/// Ring execution (EXPERIMENTAL): one attempt; any failure fails the generation.
async fn generate_ring(
    state: &FrontendState,
    ring: &RingRuntime,
    prompt: &str,
    max_tokens: u32,
    events: &Sender<Event>,
    mut cancel: oneshot::Receiver<()>,
) -> anyhow::Result<Generation> {
    let opened = tokio::select! {
        biased;
        _ = &mut cancel => return Err(GenerationCancelled.into()),
        pipeline = open_ring_pipeline(state, ring, prompt) => pipeline,
    };
    let pipeline = opened?;
    let context = state.manifest.context;
    // Checked before `Started` so an oversized prompt is still answered with a 400.
    if let Err(error) = pipeline.ensure_prompt_fits(context) {
        pipeline.close(true).await;
        return Err(error.into());
    }
    let _ = events.send(Event::Started).await;
    let limits = GenerationLimits {
        max_tokens,
        context,
    };
    pipeline
        .generate_cancellable(limits, cancel, {
            let events = events.clone();
            move |text| {
                let events = events.clone();
                async move {
                    let _ = events.send(Event::Text(text)).await;
                }
            }
        })
        .await
}

/// Starts the local stage 0 and the ring session at once and validates them together.
async fn open_ring_pipeline(
    state: &FrontendState,
    ring: &RingRuntime,
    prompt: &str,
) -> anyhow::Result<Pipeline> {
    let (stage, expected) = state
        .manifest
        .stages
        .first()
        .zip(state.expectations.first())
        .context("the ring manifest has no local stage")?;
    let (local, link) = tokio::try_join!(
        spawn_local_stage(state, stage, prompt),
        RingLink::open(&state.endpoint, &ring.plan, &ring.settings),
    )?;
    Pipeline::with_ring(local, link, expected)
}

/// Starts a fresh local runner for a local stage.
async fn spawn_local_stage(
    state: &FrontendState,
    stage: &StageSpec,
    prompt: &str,
) -> anyhow::Result<StageLink> {
    let StageLocation::Local {
        model,
        gpu_layers,
        budget_bytes,
        env,
        ..
    } = &stage.location
    else {
        bail!("stage {}..{} is not local", stage.begin, stage.end);
    };
    let launch = LocalLaunch {
        runner: &state.manifest.runner,
        model,
        begin: stage.begin,
        end: stage.end,
        context: state.manifest.context,
        gpu_layers: *gpu_layers,
        budget_bytes: *budget_bytes,
        prompt,
        env,
    };
    StageLink::spawn_local(&launch, METADATA_TIMEOUT).await
}

/// Pipeline execution: resumes the generation on fresh links after a retryable failure (up to
/// [`MAX_GENERATION_RETRIES`] times, backing off between attempts, and starting no attempt
/// after [`RETRY_DEADLINE`]). A remote worker that failed is tried again only after its
/// stage's other candidates; local stages are simply started again. A client disconnect ends
/// it at once.
async fn generate_pipeline(
    state: &FrontendState,
    prompt: &str,
    max_tokens: u32,
    events: &Sender<Event>,
    mut cancel: oneshot::Receiver<()>,
) -> anyhow::Result<Generation> {
    let limits = GenerationLimits {
        max_tokens,
        context: state.manifest.context,
    };
    let mut progress = GenerationState::new(limits, |text: String| {
        let events = events.clone();
        async move {
            // A send fails only once the handler is gone (the client disconnected).
            let _ = events.send(Event::Text(text)).await;
        }
    });
    let failed = StdMutex::new(FailedWorkers::default());
    let attempt = AttemptContext {
        state,
        prompt,
        events,
        failed: &failed,
    };
    let mut budget = RetryBudget::new();
    let mut started = false;
    loop {
        let error = match attempt.run(&mut progress, &mut started, &mut cancel).await {
            Ok(generation) => return Ok(generation),
            Err(error) => error,
        };
        let fault = attempt.retryable_fault(error)?;
        let Some(delay) = budget.next_delay() else {
            let attempts = budget.attempts();
            let location = fault.location();
            return Err(fault.error.context(format!(
                "generation failed after {attempts} attempts{location}"
            )));
        };
        warn!(
            attempt = budget.attempts(),
            stage = ?fault.stage,
            worker = ?fault.worker.map(|worker| worker.fmt_short().to_string()),
            completion_tokens = progress.completion_tokens(),
            error = %format!("{:#}", fault.error),
            "shard generation attempt failed; retrying on fresh links in {delay:?}"
        );
        tokio::select! {
            biased;
            _ = &mut cancel => return Err(GenerationCancelled.into()),
            () = tokio::time::sleep(delay) => {}
        }
    }
}

/// A failed attempt worth retrying, and where it failed when known.
struct RetryableFault {
    error: anyhow::Error,
    stage: Option<usize>,
    worker: Option<EndpointId>,
}

impl RetryableFault {
    /// Where the attempt failed, as a suffix for the final error, or `""` when unknown.
    fn location(&self) -> String {
        match (self.stage, self.worker) {
            (Some(stage), Some(worker)) => {
                format!(
                    " (last failure: stage {stage} on worker {})",
                    worker.fmt_short()
                )
            }
            (Some(stage), None) => format!(" (last failure: stage {stage}, local runner)"),
            (None, _) => String::new(),
        }
    }
}

/// How many more attempts a generation may make, and how long it waits before each.
struct RetryBudget {
    retries: u32,
    backoff: Duration,
    give_up_at: Instant,
}

impl RetryBudget {
    fn new() -> Self {
        Self {
            retries: 0,
            backoff: RETRY_BACKOFF,
            give_up_at: Instant::now() + RETRY_DEADLINE,
        }
    }

    /// Attempts started so far.
    fn attempts(&self) -> u32 {
        self.retries + 1
    }

    /// Spends one retry and returns the delay before it, or `None` once the retries are used
    /// up or the retry would start after the deadline.
    fn next_delay(&mut self) -> Option<Duration> {
        let delay = self.backoff;
        if self.retries >= MAX_GENERATION_RETRIES || Instant::now() + delay > self.give_up_at {
            return None;
        }
        self.retries += 1;
        self.backoff *= 2;
        Some(delay)
    }
}

/// Remote workers that failed during the current generation, each with the sequence number
/// of its latest failure. A failed worker is not banned, only tried after the others.
#[derive(Default)]
struct FailedWorkers {
    latest: HashMap<EndpointId, u64>,
    failures: u64,
}

impl FailedWorkers {
    fn record(&mut self, worker: EndpointId) {
        self.failures += 1;
        self.latest.insert(worker, self.failures);
    }

    /// `candidates` in the order to try them: those that have not failed keep their order
    /// (primary, then standbys), followed by failed ones, least recently failed first.
    fn order(&self, mut candidates: Vec<EndpointId>) -> Vec<EndpointId> {
        candidates.sort_by_key(|candidate| self.latest.get(candidate).copied());
        candidates
    }
}

/// Locks the failure record. It is never held across an await, and a poisoned lock still
/// holds a usable record, so poisoning is ignored.
fn lock(failed: &StdMutex<FailedWorkers>) -> MutexGuard<'_, FailedWorkers> {
    failed.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What one attempt of a generation needs from its caller.
struct AttemptContext<'a> {
    state: &'a FrontendState,
    prompt: &'a str,
    events: &'a Sender<Event>,
    failed: &'a StdMutex<FailedWorkers>,
}

impl AttemptContext<'_> {
    /// Classifies a failed attempt and records the remote worker it blames, so the next
    /// attempt prefers that stage's other candidates. Returns the error itself when it is
    /// fatal, since a retry cannot help.
    fn retryable_fault(&self, error: anyhow::Error) -> anyhow::Result<RetryableFault> {
        let classified = classify(&error);
        if classified.class == FailureClass::Fatal {
            return Err(error);
        }
        let worker = classified.fault.and_then(|fault| fault.worker);
        if let Some(worker) = worker {
            lock(self.failed).record(worker);
        }
        Ok(RetryableFault {
            error,
            stage: classified.fault.map(|fault| fault.stage),
            worker,
        })
    }

    /// Opens every stage link, validates them, and resumes `progress` on them. `Started`
    /// goes out the first time the links pass validation and the prompt fits. Ends with
    /// [`GenerationCancelled`] as soon as `cancel` resolves.
    async fn run<F, Fut>(
        &self,
        progress: &mut GenerationState<F, Fut>,
        started: &mut bool,
        cancel: &mut oneshot::Receiver<()>,
    ) -> anyhow::Result<Generation>
    where
        F: FnMut(String) -> Fut,
        Fut: Future<Output = ()>,
    {
        let opened = tokio::select! {
            biased;
            _ = &mut *cancel => return Err(GenerationCancelled.into()),
            pipeline = self.open_pipeline() => pipeline,
        };
        let pipeline = opened?;
        // Checked before `Started` so an oversized prompt is still answered with a 400.
        if let Err(error) = pipeline.ensure_prompt_fits(self.state.manifest.context) {
            pipeline.close(true).await;
            return Err(error.into());
        }
        if !*started {
            *started = true;
            // A send fails only once the handler is gone (the client disconnected).
            let _ = self.events.send(Event::Started).await;
        }
        pipeline.resume_cancellable(progress, cancel).await
    }

    /// Opens every stage of the pipeline manifest at once and validates them.
    async fn open_pipeline(&self) -> anyhow::Result<Pipeline> {
        let stages = &self.state.manifest.stages;
        let opening = stages.iter().enumerate();
        let links =
            try_join_all(opening.map(|(index, stage)| self.open_link(index, stage))).await?;
        Pipeline::new(links, &self.state.expectations)
    }

    /// Opens the link of stage `index`: a fresh local runner, or a remote candidate.
    async fn open_link(&self, index: usize, stage: &StageSpec) -> anyhow::Result<StageLink> {
        match &stage.location {
            StageLocation::Remote { .. } => self.open_remote(index, stage).await,
            StageLocation::Local { .. } => spawn_local_stage(self.state, stage, self.prompt).await,
        }
    }

    /// Tries every candidate of stage `index` once, in [`FailedWorkers::order`]. A candidate
    /// that fails with a retryable error is recorded and the next one is tried; a fatal error
    /// ends the attempt. Once all have failed, returns the last error, blamed on that worker.
    async fn open_remote(&self, index: usize, stage: &StageSpec) -> anyhow::Result<StageLink> {
        let candidates = lock(self.failed).order(stage.remote_candidates());
        let mut last_error = None;
        for worker in candidates {
            let endpoint = &self.state.endpoint;
            let error = match StageLink::connect_remote(endpoint, worker, METADATA_TIMEOUT).await {
                Ok(link) => return Ok(link),
                Err(error) => error,
            };
            if classify(&error).class == FailureClass::Fatal {
                return Err(error);
            }
            warn!(
                stage = %format!("{}..{}", stage.begin, stage.end),
                worker = %worker.fmt_short(),
                error = %format!("{error:#}"),
                "remote stage worker failed; trying the next candidate"
            );
            lock(self.failed).record(worker);
            last_error = Some((worker, error));
        }
        let Some((worker, error)) = last_error else {
            bail!("stage {}..{} has no remote worker", stage.begin, stage.end);
        };
        let fault = StageFault {
            stage: index,
            worker: Some(worker),
        };
        Err(error.context(fault))
    }
}

/// Waits for the whole generation and answers with one completion object.
async fn buffered_response(mut events: Receiver<Event>, _task: GenerationTask) -> Response {
    let completion = Completion::new();
    loop {
        match events.recv().await {
            Some(Event::Started | Event::Text(_)) => {}
            Some(Event::Finished(generation)) => {
                return Json(completion.buffered(&generation)).into_response();
            }
            Some(Event::Failed(failure)) => return failure_response(&failure),
            None => return failure_response(&Failure::task_lost()),
        }
    }
}

/// State of one streamed reply; dropping it (client disconnect) aborts the generation.
struct SseStream {
    events: Receiver<Event>,
    completion: Completion,
    _task: GenerationTask,
    ended: bool,
}

impl SseStream {
    async fn next_event(mut self) -> Option<(Result<Bytes, Infallible>, Self)> {
        if self.ended {
            return None;
        }
        let bytes = loop {
            match self.events.recv().await {
                Some(Event::Started) => {}
                Some(Event::Text(text)) => {
                    break self.completion.chunk(json!({"content": text}), None);
                }
                Some(Event::Finished(generation)) => {
                    self.ended = true;
                    let last = self.completion.chunk(json!({}), Some(generation.finish));
                    break [last, sse_data(SSE_DONE)].concat().into();
                }
                Some(Event::Failed(failure)) => {
                    self.ended = true;
                    break sse_error(&failure.message);
                }
                None => {
                    self.ended = true;
                    break sse_error(&Failure::task_lost().message);
                }
            }
        };
        Some((Ok(bytes), self))
    }
}

/// Streams the reply: a role chunk, content chunks, a final chunk with `finish_reason`, then
/// `[DONE]`; or an `error` event and no `[DONE]` if the generation fails.
fn stream_response(events: Receiver<Event>, task: GenerationTask) -> Response {
    let completion = Completion::new();
    let role = completion.chunk(json!({"role": "assistant", "content": ""}), None);
    let state = SseStream {
        events,
        completion,
        _task: task,
        ended: false,
    };
    let body = stream::once(async move { Ok::<_, Infallible>(role) })
        .chain(stream::unfold(state, SseStream::next_event));
    let mut response = Response::new(Body::from_stream(body));
    let headers = response.headers_mut();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static(EVENT_STREAM));
    headers.insert(CACHE_CONTROL, HeaderValue::from_static(NO_CACHE));
    response
}

fn sse_data(data: &str) -> Bytes {
    format!("data: {data}\n\n").into()
}

fn sse_error(message: &str) -> Bytes {
    sse_data(&json!({"error": {"message": message, "type": "server_error"}}).to_string())
}

fn failure_response(failure: &Failure) -> Response {
    error_response(failure.status, &failure.message)
}

fn error_response(status: StatusCode, message: &str) -> Response {
    let kind = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "server_error"
    };
    (
        status,
        Json(json!({"error": {"message": message, "type": kind}})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard::manifest::ChatTemplate;

    const CAP: u32 = 64;

    fn worker(seed: u8) -> EndpointId {
        iroh::SecretKey::from_bytes(&[seed; 32]).public()
    }

    #[test]
    fn failed_workers_are_tried_last_least_recently_failed_first() {
        let (primary, first, second) = (worker(1), worker(2), worker(3));
        let candidates = vec![primary, first, second];
        let mut failed = FailedWorkers::default();
        assert_eq!(failed.order(candidates.clone()), candidates);
        failed.record(primary);
        assert_eq!(failed.order(candidates.clone()), [first, second, primary]);
        failed.record(second);
        failed.record(first);
        assert_eq!(failed.order(candidates.clone()), [primary, second, first]);
        failed.record(primary);
        assert_eq!(failed.order(candidates), [second, first, primary]);
    }

    fn parse(body: Value) -> Result<ChatRequest, String> {
        let renderer = ChatRenderer::load(&ChatTemplate::Chatml).unwrap();
        parse_chat_request(body.to_string().as_bytes(), &renderer, CAP)
    }

    #[test]
    fn chatml_renders_each_message_then_the_assistant_turn() {
        let request = parse(json!({"messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi"},
        ]}))
        .unwrap();
        assert_eq!(
            request.prompt,
            "<|im_start|>system\nBe brief.<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(!request.stream);
        assert_eq!(request.max_tokens, CAP);
    }

    #[test]
    fn text_parts_are_joined() {
        let request = parse(
            json!({"stream": true, "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hel"}, {"type": "text", "text": "lo"},
            ]}]}),
        )
        .unwrap();
        assert!(request.prompt.contains("user\nhello<|im_end|>"));
        assert!(request.stream);
    }

    #[test]
    fn other_content_is_rejected() {
        let image = json!({"type": "image_url", "image_url": {"url": "x"}});
        for content in [
            json!(null),
            json!(7),
            json!([image]),
            json!([{"text": "no type"}]),
        ] {
            let body = json!({"messages": [{"role": "user", "content": content}]});
            assert!(parse(body).is_err(), "{content}");
        }
        assert!(parse(json!({"messages": []})).is_err());
        assert!(parse(json!({"prompt": "hi"})).is_err());
        assert!(parse(json!({"messages": [{"content": "hi"}]})).is_err());
    }

    #[test]
    fn max_tokens_is_honoured_and_capped() {
        let messages = json!([{"role": "user", "content": "hi"}]);
        let asked = |fields: Value| {
            let mut body = json!({"messages": messages});
            body.as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            parse(body).map(|request| request.max_tokens)
        };
        assert_eq!(asked(json!({"max_tokens": 5})), Ok(5));
        assert_eq!(asked(json!({"max_tokens": 5000})), Ok(CAP));
        assert_eq!(
            asked(json!({"max_tokens": 9, "max_completion_tokens": 3})),
            Ok(3)
        );
        assert_eq!(
            asked(json!({"max_tokens": null, "temperature": 0.2})),
            Ok(CAP)
        );
        assert!(asked(json!({"max_tokens": 0})).is_err());
    }

    #[test]
    fn non_loopback_listen_addresses_are_refused() {
        assert!(ensure_loopback("127.0.0.1:8080".parse().unwrap()).is_ok());
        assert!(ensure_loopback("[::1]:8080".parse().unwrap()).is_ok());
        assert!(ensure_loopback("[::ffff:127.0.0.1]:8080".parse().unwrap()).is_ok());
        assert!(ensure_loopback("0.0.0.0:8080".parse().unwrap()).is_err());
        assert!(ensure_loopback("192.168.1.5:8080".parse().unwrap()).is_err());
    }

    #[test]
    fn prompt_too_long_is_a_client_error() {
        let error: anyhow::Error = PromptTooLong {
            prompt_tokens: 9,
            context: 8,
        }
        .into();
        assert_eq!(Failure::from_error(&error).status, StatusCode::BAD_REQUEST);
        let other = anyhow::anyhow!("stage died");
        assert_eq!(Failure::from_error(&other).status, StatusCode::BAD_GATEWAY);
    }
}
