//! Local HTTP API that queues chat requests and proxies them to llama-server.
//!
//! [`serve_api`] exposes an OpenAI-compatible `POST /v1/chat/completions` on a loopback
//! address. Every request takes a [`JobPermit`] first, so llama-server only ever sees one
//! generation at a time and the rest wait in FIFO order. Streamed replies (`"stream": true`)
//! are relayed chunk by chunk as they arrive; the assistant text is assembled on the side for
//! the [`HistorySink`]. A streamed reply completes only after a valid `[DONE]` terminator;
//! truncated, malformed or error-bearing SSE fails the job without committing history.
//! The sink sees each running job's lifecycle: `started` when it gets the permit, then either
//! `record` with the completed turn or `ended` with a [`TurnOutcome`].
//! `GET /jobs` lists job states and `GET /peers` lists peers with authenticated control links.

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::{Stream, StreamExt, stream};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tracing::{debug, info, warn};

use crate::history::{CompletedTurn, HistorySink, StartedTurn, TurnOutcome};
use crate::jobs::{JobInfo, JobPermit, JobQueue, MAX_OUTSTANDING_JOBS};
use crate::network::SharedPeerTable;

/// OpenAI-compatible chat endpoint, served here and by llama-server.
pub const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

/// Lists every tracked job as JSON.
pub const JOBS_PATH: &str = "/jobs";

/// Lists peers with an authenticated control connection to this leader.
pub const PEERS_PATH: &str = "/peers";

/// Browser client served by the leader at the root path.
pub const WEB_CLIENT_PATH: &str = "/";

/// Optional top-level request field naming the conversation; never forwarded upstream.
pub const CONVERSATION_ID_FIELD: &str = "conversation_id";

/// Response header carrying the job's request ID, so clients can find it in `GET /jobs`.
pub const REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-request-id");

/// Largest accepted request body; long chat histories exceed axum's 2 MiB default.
const MAX_REQUEST_BODY_BYTES: usize = 16 * 1024 * 1024;

/// How long to wait for llama-server to accept a connection. There is no overall timeout:
/// a distributed generation can legitimately take minutes.
const UPSTREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Prefix of a server-sent-events line that carries a payload.
const SSE_DATA_PREFIX: &str = "data:";

/// Payload that ends an OpenAI-style event stream.
const SSE_DONE: &str = "[DONE]";

/// Largest buffered SSE event line before the upstream stream is rejected as invalid.
const MAX_SSE_LINE_BYTES: usize = 1024 * 1024;

/// Largest non-streaming upstream response retained in memory.
const MAX_BUFFERED_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Largest assistant response retained for conversation history; streaming to the client can continue.
const MAX_HISTORY_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// JSON pointer to the assistant text in a non-streamed chat completion.
const MESSAGE_CONTENT_POINTER: &str = "/choices/0/message/content";

/// JSON pointer to the text fragment in one streamed chat-completion chunk.
const DELTA_CONTENT_POINTER: &str = "/choices/0/delta/content";

/// Where the API listens and where llama-server is.
#[derive(Clone, Debug)]
pub struct ApiConfig {
    /// Address to serve on; must be loopback, anything else is refused.
    pub listen: SocketAddr,
    /// llama-server's HTTP address.
    pub upstream: SocketAddr,
    /// Authenticated peers currently connected to the leader.
    pub peers: SharedPeerTable,
}

/// State shared by every request handler.
struct ApiState {
    queue: Arc<JobQueue>,
    sink: Arc<dyn HistorySink>,
    client: reqwest::Client,
    upstream_url: String,
}

/// Serves chat completions, job status, and connected-peer status on `config.listen` until
/// `shutdown` resolves, then finishes in-flight requests and returns.
///
/// Each chat request acquires a permit from `queue`, has its optional `conversation_id`
/// stripped, and is forwarded to llama-server (streaming or not). The permit is completed on
/// success, failed on an upstream error or non-2xx reply, and canceled if the client
/// disconnects. `sink` is told when each job starts and how it ends (see [`HistorySink`]).
pub async fn serve_api(
    config: ApiConfig,
    queue: Arc<JobQueue>,
    sink: Arc<dyn HistorySink>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    ensure_loopback(config.listen)?;
    let profiles = match std::env::var_os("LAYERCAKE_PROFILE_CONFIG") {
        Some(path) => {
            Some(crate::profiling::ProfileManager::from_file(std::path::Path::new(&path)).await?)
        }
        None => None,
    };
    serve_api_with_profiles(config, queue, sink, profiles, shutdown).await
}

/// Existing API with optional managed profiling; native execution must share admission.
pub async fn serve_api_with_profiles(
    config: ApiConfig,
    queue: Arc<JobQueue>,
    sink: Arc<dyn HistorySink>,
    profiles: Option<crate::profiling::ProfileManager>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    ensure_loopback(config.listen)?;
    let mut router = queued_router(config.upstream, queue, sink)?
        .merge(peer_router(config.peers))
        .route(WEB_CLIENT_PATH, get(web_client));
    if let Some(profiles) = &profiles {
        router = router.merge(profiles.router());
    }
    let listener = TcpListener::bind(config.listen)
        .await
        .with_context(|| format!("failed to bind api listener on {}", config.listen))?;
    let local = listener
        .local_addr()
        .context("failed to read api listener address")?;
    info!(listen = %local, upstream = %config.upstream, "api listening");
    let result = axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await
        .context("api server failed");
    if let Some(profiles) = profiles {
        profiles.shutdown().await?;
    }
    result
}

/// Reusable queued inference routes for the integrated desktop runtime.
pub fn queued_router(
    upstream: SocketAddr,
    queue: Arc<JobQueue>,
    sink: Arc<dyn HistorySink>,
) -> anyhow::Result<Router> {
    let client = reqwest::Client::builder()
        .connect_timeout(UPSTREAM_CONNECT_TIMEOUT)
        .build()
        .context("failed to build upstream http client")?;
    let state = Arc::new(ApiState {
        queue,
        sink,
        client,
        upstream_url: format!("http://{}{CHAT_COMPLETIONS_PATH}", upstream),
    });
    let router = Router::new()
        .route(CHAT_COMPLETIONS_PATH, post(chat_completions))
        .route(JOBS_PATH, get(list_jobs))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .with_state(state);
    Ok(router)
}

async fn web_client() -> Html<&'static str> {
    Html(include_str!("../web/index.html"))
}

/// Client used by `llm-cluster prompt`: sends one user message as a non-streaming chat request
/// to the API at `api` and returns the assistant text.
pub async fn prompt(api: SocketAddr, text: &str) -> anyhow::Result<String> {
    let request = json!({
        "messages": [{"role": "user", "content": text}],
        "stream": false,
    });
    let response = reqwest::Client::new()
        .post(format!("http://{api}{CHAT_COMPLETIONS_PATH}"))
        .json(&request)
        .send()
        .await
        .with_context(|| format!("failed to reach the api at {api}"))?;
    let status = response.status();
    let chunks = response
        .bytes_stream()
        .map(|chunk| chunk.context("failed to read the api reply body"));
    let body = collect_upstream_chunks(Box::pin(chunks), MAX_BUFFERED_RESPONSE_BYTES)
        .await
        .context("failed to read bounded api reply")?;
    if !status.is_success() {
        bail!("api returned {status}: {}", String::from_utf8_lossy(&body));
    }
    let reply: Value = serde_json::from_slice(&body).context("api reply is not json")?;
    message_text(&reply)
        .map(str::to_owned)
        .with_context(|| format!("api reply has no {MESSAGE_CONTENT_POINTER}"))
}

fn ensure_loopback(listen: SocketAddr) -> anyhow::Result<()> {
    if !listen.ip().is_loopback() {
        bail!("api listen address {listen} is not loopback; refusing to expose the api");
    }
    Ok(())
}

async fn list_jobs(State(state): State<Arc<ApiState>>) -> Json<Vec<JobInfo>> {
    Json(state.queue.jobs())
}

#[derive(Serialize)]
pub struct PeerSummary {
    /// Authenticated Iroh endpoint ID in its canonical display form.
    pub node_id: String,
    /// Unix milliseconds when this peer's current control connection was recorded.
    pub connected_at_unix_ms: u64,
    /// Unix milliseconds of the last accepted heartbeat, if one has arrived.
    pub last_heartbeat_unix_ms: Option<u64>,
}

/// Read-only authenticated control peers, independently of inference readiness.
pub fn peer_router(peers: SharedPeerTable) -> Router {
    Router::new()
        .route(PEERS_PATH, get(list_peers))
        .with_state(peers)
}

async fn list_peers(
    State(peers): State<SharedPeerTable>,
) -> Result<Json<Vec<PeerSummary>>, StatusCode> {
    peer_snapshot(&peers).map(Json)
}

/// Shared projection for the API and the embedded dashboard.
pub fn peer_snapshot(table: &SharedPeerTable) -> Result<Vec<PeerSummary>, StatusCode> {
    let peers = table.lock().map_err(|error| {
        warn!(%error, "peer table lock poisoned; refusing to report peer status");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    let mut summaries = peers
        .peers()
        .map(|peer| PeerSummary {
            node_id: peer.node_id.to_string(),
            connected_at_unix_ms: peer.connected_at_unix_ms,
            last_heartbeat_unix_ms: peer.last_heartbeat_unix_ms,
        })
        .collect::<Vec<_>>();
    summaries.sort_by(|a, b| a.node_id.cmp(&b.node_id));
    Ok(summaries)
}

async fn chat_completions(State(state): State<Arc<ApiState>>, body: Bytes) -> Response {
    let (request, conversation_id) = match parse_chat_request(&body) {
        Ok(parsed) => parsed,
        Err(message) => return error_response(StatusCode::BAD_REQUEST, &message),
    };
    let queued_at = Instant::now();
    let Some(permit) = state.queue.try_acquire(conversation_id.clone()).await else {
        debug!(
            limit = MAX_OUTSTANDING_JOBS,
            "inference queue is full; rejecting request"
        );
        return error_response(
            StatusCode::TOO_MANY_REQUESTS,
            "inference queue is full; retry after a running job finishes",
        );
    };
    let queue_wait_ms = queued_at.elapsed().as_millis();
    let is_streaming = request.get("stream").and_then(Value::as_bool) == Some(true);
    info!(
        request_id = permit.request_id(),
        queue_wait_ms, is_streaming, "job running"
    );
    let turn = StartedTurn {
        request_id: permit.request_id().to_owned(),
        conversation_id,
        request,
    };
    let finish = JobFinish::start(permit, turn, Arc::clone(&state.sink));
    let send = state
        .client
        .post(&state.upstream_url)
        .json(&finish.turn.request);
    let upstream = match send.send().await {
        Ok(upstream) => upstream,
        Err(err) => {
            let reason = format!("upstream request failed: {err}");
            return fail_job(finish, StatusCode::BAD_GATEWAY, reason);
        }
    };
    if !upstream.status().is_success() {
        relay_upstream_error(upstream, finish).await
    } else if is_streaming {
        relay_stream(upstream, finish)
    } else {
        relay_buffered(upstream, finish).await
    }
}

/// Parses a chat request body and removes its `conversation_id`, which llama-server would not
/// understand. Returns the request to forward and the conversation ID.
fn parse_chat_request(body: &[u8]) -> Result<(Value, Option<String>), String> {
    let mut request: Value =
        serde_json::from_slice(body).map_err(|err| format!("request body is not json: {err}"))?;
    let Some(fields) = request.as_object_mut() else {
        return Err("request body must be a json object".to_owned());
    };
    let conversation_id = match fields.remove(CONVERSATION_ID_FIELD) {
        None | Some(Value::Null) => None,
        Some(Value::String(id)) => Some(id),
        Some(_) => return Err(format!("{CONVERSATION_ID_FIELD} must be a string")),
    };
    Ok((request, conversation_id))
}

fn message_text(reply: &Value) -> Option<&str> {
    reply
        .pointer(MESSAGE_CONTENT_POINTER)
        .and_then(Value::as_str)
}

fn buffered_response_text(body: &[u8]) -> Result<String, String> {
    let reply: Value = serde_json::from_slice(body)
        .map_err(|error| format!("upstream reply is not valid JSON: {error}"))?;
    message_text(&reply)
        .map(str::to_owned)
        .ok_or_else(|| format!("upstream reply has no string at {MESSAGE_CONTENT_POINTER}"))
}

/// Marks the job failed and answers with an OpenAI-style error body.
fn fail_job(finish: JobFinish, status: StatusCode, reason: String) -> Response {
    let mut response = error_response(status, &reason);
    if let Ok(request_id) = HeaderValue::from_str(finish.request_id()) {
        response.headers_mut().insert(REQUEST_ID_HEADER, request_id);
    }
    finish.fail(reason);
    response
}

fn error_response(status: StatusCode, message: &str) -> Response {
    let kind = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "server_error"
    };
    let body = json!({"error": {"message": message, "type": kind}});
    (status, Json(body)).into_response()
}

/// Builds the client's response from llama-server's status, content type, and body.
fn relay_response(
    status: StatusCode,
    content_type: Option<HeaderValue>,
    request_id: &str,
    body: Body,
) -> Response {
    let mut response = Response::new(body);
    *response.status_mut() = status;
    let headers = response.headers_mut();
    if let Some(content_type) = content_type {
        headers.insert(CONTENT_TYPE, content_type);
    }
    if let Ok(request_id) = HeaderValue::from_str(request_id) {
        headers.insert(REQUEST_ID_HEADER, request_id);
    }
    response
}

fn upstream_content_type(upstream: &reqwest::Response) -> Option<HeaderValue> {
    upstream.headers().get(CONTENT_TYPE).cloned()
}

/// Passes a non-2xx llama-server reply through unchanged and fails the job.
async fn relay_upstream_error(upstream: reqwest::Response, finish: JobFinish) -> Response {
    let status = upstream.status();
    let content_type = upstream_content_type(&upstream);
    // The status alone explains the failure; an unreadable or oversized error body is relayed empty.
    let body = match buffer_upstream(upstream, MAX_BUFFERED_RESPONSE_BYTES).await {
        Ok(body) => body,
        Err(error) => {
            warn!(request_id = finish.request_id(), %error, "could not buffer upstream error body");
            Bytes::new()
        }
    };
    let response = relay_response(status, content_type, finish.request_id(), Body::from(body));
    let reason = format!("upstream returned {status}");
    finish.fail(reason);
    response
}

/// Reads a whole non-streamed reply, relays it, and completes the job.
async fn relay_buffered(upstream: reqwest::Response, finish: JobFinish) -> Response {
    let status = upstream.status();
    let content_type = upstream_content_type(&upstream);
    let body = match buffer_upstream(upstream, MAX_BUFFERED_RESPONSE_BYTES).await {
        Ok(body) => body,
        Err(err) => {
            let reason = format!("failed to read upstream reply: {err}");
            return fail_job(finish, StatusCode::BAD_GATEWAY, reason);
        }
    };
    let response_text = match buffered_response_text(&body) {
        Ok(text) => text,
        Err(reason) => return fail_job(finish, StatusCode::BAD_GATEWAY, reason),
    };
    let response = relay_response(status, content_type, finish.request_id(), Body::from(body));
    finish.complete(response_text);
    response
}

/// Collects a non-streaming upstream reply without allowing its body to grow without bound.
async fn buffer_upstream(upstream: reqwest::Response, limit: usize) -> anyhow::Result<Bytes> {
    if upstream
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        bail!("upstream response exceeds the {limit}-byte buffering limit");
    }
    let chunks = upstream
        .bytes_stream()
        .map(|chunk| chunk.context("failed to read upstream response body"));
    collect_upstream_chunks(Box::pin(chunks), limit).await
}

async fn collect_upstream_chunks(
    mut chunks: impl Stream<Item = anyhow::Result<Bytes>> + Unpin,
    limit: usize,
) -> anyhow::Result<Bytes> {
    let mut body = Vec::new();
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk?;
        if body.len().saturating_add(chunk.len()) > limit {
            bail!("response exceeds the {limit}-byte buffering limit");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(body))
}

/// Relays a streamed reply chunk by chunk; the job finishes when the stream does.
fn relay_stream(upstream: reqwest::Response, finish: JobFinish) -> Response {
    let status = upstream.status();
    let content_type = upstream_content_type(&upstream);
    let request_id = finish.request_id().to_owned();
    let relay = StreamRelay {
        chunks: Box::pin(upstream.bytes_stream()),
        assembler: SseTextAssembler::default(),
        finish: Some(finish),
    };
    let body = Body::from_stream(stream::unfold(relay, StreamRelay::next_chunk));
    relay_response(status, content_type, &request_id, body)
}

/// Owns a running job's permit until its reply is fully delivered, and reports the job's
/// lifecycle to the history sink: `started` on creation, then exactly one of `record`
/// ([`complete`](Self::complete)) or `ended` for failure, cancellation, or a successful response
/// too large to retain ([`fail`](Self::fail), [`complete_without_history`](Self::complete_without_history),
/// or drop).
///
/// The sink is told before the permit is released, so its calls for this job finish before the
/// next queued job can start.
struct JobFinish {
    turn: StartedTurn,
    sink: Arc<dyn HistorySink>,
    /// `Some` until the job's outcome is reported.
    permit: Option<JobPermit>,
}

impl JobFinish {
    /// Takes over a freshly acquired permit and reports the job as started.
    fn start(permit: JobPermit, turn: StartedTurn, sink: Arc<dyn HistorySink>) -> Self {
        sink.started(&turn);
        Self {
            turn,
            sink,
            permit: Some(permit),
        }
    }

    fn request_id(&self) -> &str {
        &self.turn.request_id
    }

    /// Hands the completed turn to the history sink, then completes the job.
    fn complete(mut self, response_text: String) {
        let Some(permit) = self.permit.take() else {
            return;
        };
        info!(request_id = self.request_id(), "job completed");
        self.sink.record(CompletedTurn {
            request_id: std::mem::take(&mut self.turn.request_id),
            conversation_id: self.turn.conversation_id.take(),
            request: self.turn.request.take(),
            response_text,
        });
        permit.complete();
    }

    /// Completes the job but avoids retaining a response larger than the history limit.
    fn complete_without_history(mut self) {
        let Some(permit) = self.permit.take() else {
            return;
        };
        warn!(
            request_id = self.request_id(),
            limit_bytes = MAX_HISTORY_RESPONSE_BYTES,
            "job completed but response was too large to retain in history"
        );
        self.sink.ended(
            self.request_id(),
            TurnOutcome::CompletedWithoutHistory {
                limit_bytes: MAX_HISTORY_RESPONSE_BYTES,
            },
        );
        permit.complete();
    }

    /// Reports the failure to the history sink, then fails the job with the same reason.
    fn fail(mut self, reason: String) {
        let Some(permit) = self.permit.take() else {
            return;
        };
        warn!(request_id = self.request_id(), %reason, "job failed");
        let outcome = TurnOutcome::Failed {
            reason: reason.clone(),
        };
        self.sink.ended(self.request_id(), outcome);
        permit.fail(reason);
    }
}

impl Drop for JobFinish {
    /// Dropped without an outcome: the client went away, so the job is canceled.
    fn drop(&mut self) {
        let Some(permit) = self.permit.take() else {
            return;
        };
        info!(request_id = self.request_id(), "job canceled");
        self.sink.ended(self.request_id(), TurnOutcome::Canceled);
        // Dropping the permit without an outcome marks the job canceled and frees the slot.
        drop(permit);
    }
}

type UpstreamChunks = Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>;

/// State of one streamed reply. Dropping it before the upstream stream ends (the client
/// disconnected) drops the [`JobFinish`], which cancels the job.
struct StreamRelay {
    chunks: UpstreamChunks,
    assembler: SseTextAssembler,
    /// `None` once the job's outcome is recorded; the stream then ends.
    finish: Option<JobFinish>,
}

impl StreamRelay {
    async fn next_chunk(mut self) -> Option<(std::io::Result<Bytes>, Self)> {
        let finish = self.finish.take()?;
        match self.chunks.next().await {
            Some(Ok(chunk)) => {
                self.assembler.push(&chunk);
                self.finish = Some(finish);
                Some((Ok(chunk), self))
            }
            Some(Err(err)) => {
                finish.fail(format!("upstream stream failed: {err}"));
                Some((Err(std::io::Error::other(err)), self))
            }
            None => match std::mem::take(&mut self.assembler).finish() {
                Ok(Some(text)) => {
                    finish.complete(text);
                    None
                }
                Ok(None) => {
                    finish.complete_without_history();
                    None
                }
                Err(reason) => {
                    finish.fail(reason.to_owned());
                    Some((
                        Err(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            reason,
                        )),
                        self,
                    ))
                }
            },
        }
    }
}

/// Rebuilds the assistant text from OpenAI-style SSE chunks, which may split lines (and UTF-8
/// characters) at arbitrary byte boundaries.
#[derive(Debug, Default)]
struct SseTextAssembler {
    /// Bytes after the last complete line.
    partial_line: Vec<u8>,
    text: String,
    done: bool,
    invalid: bool,
    history_limited: bool,
}

impl SseTextAssembler {
    fn push(&mut self, bytes: &[u8]) {
        if self.invalid {
            return;
        }

        let mut offset = 0;
        while let Some(relative_newline) = bytes[offset..].iter().position(|&byte| byte == b'\n') {
            let line_end = offset + relative_newline + 1;
            let line_len = self.partial_line.len().saturating_add(line_end - offset);
            if line_len > MAX_SSE_LINE_BYTES {
                self.invalid = true;
                self.partial_line.clear();
                return;
            }

            let mut line = std::mem::take(&mut self.partial_line);
            line.extend_from_slice(&bytes[offset..line_end]);
            self.absorb_line(&line);
            line.clear();
            self.partial_line = line;
            if self.invalid {
                self.partial_line.clear();
                return;
            }
            offset = line_end;
        }

        let remaining = &bytes[offset..];
        if self.partial_line.len().saturating_add(remaining.len()) > MAX_SSE_LINE_BYTES {
            self.invalid = true;
            self.partial_line.clear();
        } else {
            self.partial_line.extend_from_slice(remaining);
        }
    }

    fn absorb_line(&mut self, line: &[u8]) {
        let Ok(line) = std::str::from_utf8(line) else {
            self.invalid = true;
            return;
        };
        let Some(data) = line.trim_end().strip_prefix(SSE_DATA_PREFIX) else {
            return;
        };
        let data = data.trim_start();
        if data == SSE_DONE {
            if self.done {
                self.invalid = true;
            } else {
                self.done = true;
            }
            return;
        }
        if self.done {
            self.invalid = true;
            return;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            self.invalid = true;
            return;
        };
        if chunk.get("error").is_some() {
            self.invalid = true;
            return;
        }
        if let Some(fragment) = chunk.pointer(DELTA_CONTENT_POINTER).and_then(Value::as_str) {
            if self.text.len().saturating_add(fragment.len()) > MAX_HISTORY_RESPONSE_BYTES {
                self.text = String::new();
                self.history_limited = true;
            } else if !self.history_limited {
                self.text.push_str(fragment);
            }
        }
    }

    /// A complete response must end with [DONE], including without a final newline.
    fn finish(mut self) -> Result<Option<String>, &'static str> {
        let rest = std::mem::take(&mut self.partial_line);
        self.absorb_line(&rest);
        if self.invalid {
            return Err("upstream SSE contained invalid or error data");
        }
        if !self.done {
            return Err("upstream SSE ended before [DONE]");
        }
        Ok((!self.history_limited).then_some(self.text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::JobStatus;
    use std::sync::Mutex;

    /// One history hook call, keyed by the job's request ID.
    #[derive(Debug, PartialEq)]
    enum Hook {
        Started(String),
        Recorded {
            request_id: String,
            response_text: String,
        },
        Ended(String, TurnOutcome),
    }

    #[derive(Default)]
    struct RecordingSink(Mutex<Vec<Hook>>);

    impl RecordingSink {
        fn hooks(&self) -> std::sync::MutexGuard<'_, Vec<Hook>> {
            self.0.lock().unwrap()
        }
    }

    impl HistorySink for RecordingSink {
        fn started(&self, turn: &StartedTurn) {
            self.hooks().push(Hook::Started(turn.request_id.clone()));
        }

        fn record(&self, turn: CompletedTurn) {
            self.hooks().push(Hook::Recorded {
                request_id: turn.request_id,
                response_text: turn.response_text,
            });
        }

        fn ended(&self, request_id: &str, outcome: TurnOutcome) {
            self.hooks()
                .push(Hook::Ended(request_id.to_owned(), outcome));
        }
    }

    /// Acquires a permit and wraps `events` (split into 3-byte chunks) in a stream relay.
    async fn start_relay(
        queue: &JobQueue,
        sink: &Arc<RecordingSink>,
        events: &str,
    ) -> (StreamRelay, String) {
        let permit = queue.acquire(None).await;
        let request_id = permit.request_id().to_owned();
        let chunks = events
            .as_bytes()
            .chunks(3)
            .map(|chunk| Ok::<_, reqwest::Error>(Bytes::copy_from_slice(chunk)))
            .collect::<Vec<_>>();
        let turn = StartedTurn {
            request_id: request_id.clone(),
            conversation_id: None,
            request: json!({}),
        };
        let relay = StreamRelay {
            chunks: Box::pin(stream::iter(chunks)),
            assembler: SseTextAssembler::default(),
            finish: Some(JobFinish::start(permit, turn, sink.clone())),
        };
        (relay, request_id)
    }

    async fn check_stream(events: &str, succeeds: bool) {
        let queue = Arc::new(JobQueue::new());
        let sink = Arc::new(RecordingSink::default());
        let (relay, request_id) = start_relay(&queue, &sink, events).await;
        let mut remaining = Some(relay);
        let mut failed = false;
        while let Some(relay) = remaining.take() {
            if let Some((chunk, next)) = relay.next_chunk().await {
                failed |= chunk.is_err();
                remaining = Some(next);
            }
        }
        assert_eq!(failed, !succeeds);
        assert_eq!(
            queue.jobs()[0].status,
            if succeeds {
                JobStatus::Completed
            } else {
                JobStatus::Failed
            }
        );
        {
            let hooks = sink.hooks();
            assert_eq!(
                hooks.len(),
                2,
                "started plus exactly one outcome: {hooks:?}"
            );
            assert_eq!(hooks[0], Hook::Started(request_id.clone()));
            if succeeds {
                let recorded = Hook::Recorded {
                    request_id,
                    response_text: "Hello".to_owned(),
                };
                assert_eq!(hooks[1], recorded);
            } else {
                assert!(
                    matches!(&hooks[1], Hook::Ended(id, TurnOutcome::Failed { .. }) if *id == request_id),
                    "unexpected outcome: {:?}",
                    hooks[1]
                );
            }
        }
        let next = tokio::time::timeout(Duration::from_secs(1), queue.acquire(None))
            .await
            .expect("stream outcome must release the running slot");
        next.complete();
    }

    #[tokio::test]
    async fn dropping_a_stream_mid_reply_reports_canceled_once() {
        let queue = JobQueue::new();
        let sink = Arc::new(RecordingSink::default());
        let events = format!("{}data: [DONE]\n\n", delta_event("Hello"));
        let (relay, request_id) = start_relay(&queue, &sink, &events).await;
        let (chunk, relay) = relay.next_chunk().await.expect("first chunk");
        chunk.unwrap();
        drop(relay);
        assert_eq!(
            *sink.hooks(),
            [
                Hook::Started(request_id.clone()),
                Hook::Ended(request_id, TurnOutcome::Canceled),
            ]
        );
        assert_eq!(queue.jobs()[0].status, JobStatus::Canceled);
    }

    #[tokio::test]
    async fn completed_response_without_history_reports_terminal_outcome() {
        let queue = JobQueue::new();
        let sink = Arc::new(RecordingSink::default());
        let permit = queue.acquire(None).await;
        let request_id = permit.request_id().to_owned();
        let turn = StartedTurn {
            request_id: request_id.clone(),
            conversation_id: None,
            request: json!({}),
        };
        let finish = JobFinish::start(permit, turn, sink.clone());

        finish.complete_without_history();

        assert_eq!(queue.jobs()[0].status, JobStatus::Completed);
        assert_eq!(
            *sink.hooks(),
            [
                Hook::Started(request_id.clone()),
                Hook::Ended(
                    request_id,
                    TurnOutcome::CompletedWithoutHistory {
                        limit_bytes: MAX_HISTORY_RESPONSE_BYTES,
                    }
                ),
            ]
        );
    }

    #[tokio::test]
    async fn clean_http_eof_without_done_fails_without_committing_history() {
        check_stream(&delta_event("Hello"), false).await;
    }

    #[tokio::test]
    async fn complete_stream_commits_history_once_and_releases_queue() {
        check_stream(&format!("{}data: [DONE]\n\n", delta_event("Hello")), true).await;
    }

    #[tokio::test]
    async fn error_or_malformed_events_do_not_commit_history_even_with_done() {
        for event in [
            "data: {\"error\":{\"message\":\"worker lost\"}}\n\n",
            "data: {broken\n\n",
        ] {
            check_stream(
                &format!("{}{event}data: [DONE]\n\n", delta_event("Hello")),
                false,
            )
            .await;
        }
        for trailing in [
            "data: [DONE]\n\ndata: [DONE]\n\n",
            "data: [DONE]\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n\n",
        ] {
            check_stream(&format!("{}{trailing}", delta_event("Hello")), false).await;
        }
    }

    fn delta_event(fragment: &str) -> String {
        let chunk = json!({"choices": [{"index": 0, "delta": {"content": fragment}}]});
        format!("data: {chunk}\n\n")
    }

    #[test]
    fn conversation_id_is_stripped_and_returned() {
        let body = br#"{"messages": [], "conversation_id": "c-1", "stream": true}"#;
        let (request, conversation_id) = parse_chat_request(body).unwrap();
        assert_eq!(conversation_id.as_deref(), Some("c-1"));
        assert!(request.get(CONVERSATION_ID_FIELD).is_none());
        assert_eq!(request["stream"], json!(true));
    }

    #[test]
    fn missing_or_null_conversation_id_is_none() {
        assert_eq!(parse_chat_request(br#"{"messages": []}"#).unwrap().1, None);
        let body = br#"{"messages": [], "conversation_id": null}"#;
        assert_eq!(parse_chat_request(body).unwrap().1, None);
    }

    #[test]
    fn malformed_requests_are_rejected() {
        assert!(parse_chat_request(b"not json").is_err());
        assert!(parse_chat_request(b"[1, 2]").is_err());
        assert!(parse_chat_request(br#"{"conversation_id": 7}"#).is_err());
    }

    #[test]
    fn buffered_reply_requires_json_assistant_text() {
        assert_eq!(
            buffered_response_text(br#"{"choices":[{"message":{"content":"hello"}}]}"#).as_deref(),
            Ok("hello")
        );
        assert!(buffered_response_text(b"not json").is_err());
        assert!(buffered_response_text(br#"{"choices":[{"message":{}}]}"#).is_err());
    }

    #[test]
    fn assembler_joins_deltas_split_across_arbitrary_chunks() {
        let events = format!(
            "data: {{\"choices\":[{{\"delta\":{{\"role\":\"assistant\"}}}}]}}\n\n{}{}{}data: [DONE]\n\n",
            delta_event("Hel"),
            delta_event("lo, "),
            delta_event("wörld"),
        );
        let mut assembler = SseTextAssembler::default();
        for piece in events.as_bytes().chunks(7) {
            assembler.push(piece);
        }
        assert_eq!(assembler.finish().unwrap(), Some("Hello, wörld".to_owned()));
    }

    #[test]
    fn assembler_handles_crlf_and_a_final_line_without_newline() {
        let mut assembler = SseTextAssembler::default();
        assembler.push(delta_event("a").replace('\n', "\r\n").as_bytes());
        assembler.push(delta_event("b").as_bytes());
        assembler.push(b"data: [DONE]");
        assert_eq!(assembler.finish().unwrap(), Some("ab".to_owned()));
    }

    #[tokio::test]
    async fn upstream_buffer_stops_at_its_limit() {
        let chunks = stream::iter([
            Ok(Bytes::from_static(b"123")),
            Ok(Bytes::from_static(b"456")),
        ]);
        let error = collect_upstream_chunks(chunks, 5)
            .await
            .expect_err("oversized reply must not be buffered");
        assert!(error.to_string().contains("buffering limit"));
    }

    #[test]
    fn oversized_stream_text_is_not_retained_for_history() {
        let mut assembler = SseTextAssembler {
            text: "x".repeat(MAX_HISTORY_RESPONSE_BYTES),
            ..SseTextAssembler::default()
        };
        assembler.absorb_line(br#"data: {"choices":[{"delta":{"content":"y"}}]}"#);
        assembler.done = true;

        assert!(assembler.history_limited);
        assert!(assembler.text.is_empty());
        assert_eq!(assembler.finish().unwrap(), None);
    }

    #[test]
    fn non_loopback_listen_addresses_are_refused() {
        assert!(ensure_loopback("127.0.0.1:8080".parse().unwrap()).is_ok());
        assert!(ensure_loopback("[::1]:8080".parse().unwrap()).is_ok());
        assert!(ensure_loopback("0.0.0.0:8080".parse().unwrap()).is_err());
        assert!(ensure_loopback("192.168.1.5:8080".parse().unwrap()).is_err());
    }
}
