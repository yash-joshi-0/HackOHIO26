//! Local HTTP API that queues chat requests and proxies them to llama-server.
//!
//! [`serve_api`] exposes an OpenAI-compatible `POST /v1/chat/completions` on a loopback
//! address. Every request takes a [`JobPermit`] first, so llama-server only ever sees one
//! generation at a time and the rest wait in FIFO order. Streamed replies (`"stream": true`)
//! are relayed chunk by chunk as they arrive; the assistant text is assembled on the side for
//! the [`HistorySink`]. A streamed reply completes only after a valid `[DONE]` terminator;
//! truncated, malformed or error-bearing SSE fails the job without committing history.
//! `GET /jobs` lists job states.

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, bail};
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::{Stream, StreamExt, stream};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tracing::{info, warn};

use crate::history::{CompletedTurn, HistorySink};
use crate::jobs::{JobInfo, JobPermit, JobQueue};

/// OpenAI-compatible chat endpoint, served here and by llama-server.
pub const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

/// Lists every tracked job as JSON.
pub const JOBS_PATH: &str = "/jobs";

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

/// JSON pointer to the assistant text in a non-streamed chat completion.
const MESSAGE_CONTENT_POINTER: &str = "/choices/0/message/content";

/// JSON pointer to the text fragment in one streamed chat-completion chunk.
const DELTA_CONTENT_POINTER: &str = "/choices/0/delta/content";

/// Where the API listens and where llama-server is.
#[derive(Clone, Copy, Debug)]
pub struct ApiConfig {
    /// Address to serve on; must be loopback, anything else is refused.
    pub listen: SocketAddr,
    /// llama-server's HTTP address.
    pub upstream: SocketAddr,
}

/// State shared by every request handler.
struct ApiState {
    queue: Arc<JobQueue>,
    sink: Arc<dyn HistorySink>,
    client: reqwest::Client,
    upstream_url: String,
}

/// Serves `POST /v1/chat/completions` and `GET /jobs` on `config.listen` until `shutdown`
/// resolves, then finishes in-flight requests and returns.
///
/// Each chat request acquires a permit from `queue`, has its optional `conversation_id`
/// stripped, and is forwarded to llama-server (streaming or not). The permit is completed on
/// success, failed on an upstream error or non-2xx reply, and canceled if the client
/// disconnects. Completed turns go to `sink`.
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

/// Same API with an explicitly constructed profiler, for the single-process application host.
/// The caller must route its native execution through the same admission owner.
pub async fn serve_api_with_profiles(
    config: ApiConfig,
    queue: Arc<JobQueue>,
    sink: Arc<dyn HistorySink>,
    profiles: Option<crate::profiling::ProfileManager>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    ensure_loopback(config.listen)?;
    let client = reqwest::Client::builder()
        .connect_timeout(UPSTREAM_CONNECT_TIMEOUT)
        .build()
        .context("failed to build upstream http client")?;
    let state = Arc::new(ApiState {
        queue,
        sink,
        client,
        upstream_url: format!("http://{}{CHAT_COMPLETIONS_PATH}", config.upstream),
    });
    let mut router = Router::new()
        .route(WEB_CLIENT_PATH, get(web_client))
        .route(CHAT_COMPLETIONS_PATH, post(chat_completions))
        .route(JOBS_PATH, get(list_jobs))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .with_state(state);
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
    let body = response
        .bytes()
        .await
        .context("failed to read the api reply")?;
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

async fn chat_completions(State(state): State<Arc<ApiState>>, body: Bytes) -> Response {
    let (request, conversation_id) = match parse_chat_request(&body) {
        Ok(parsed) => parsed,
        Err(message) => return error_response(StatusCode::BAD_REQUEST, &message),
    };
    let permit = state.queue.acquire(conversation_id.clone()).await;
    let is_streaming = request.get("stream").and_then(Value::as_bool) == Some(true);
    info!(
        request_id = permit.request_id(),
        is_streaming, "job running"
    );
    let turn = PendingTurn {
        request_id: permit.request_id().to_owned(),
        conversation_id,
        request,
    };
    let send = state.client.post(&state.upstream_url).json(&turn.request);
    let upstream = match send.send().await {
        Ok(upstream) => upstream,
        Err(err) => {
            let reason = format!("upstream request failed: {err}");
            return fail_job(permit, StatusCode::BAD_GATEWAY, reason);
        }
    };
    let finish = JobFinish {
        permit,
        turn,
        sink: Arc::clone(&state.sink),
    };
    if !upstream.status().is_success() {
        relay_upstream_error(upstream, finish.permit).await
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
fn fail_job(permit: JobPermit, status: StatusCode, reason: String) -> Response {
    warn!(request_id = permit.request_id(), %reason, "job failed");
    let mut response = error_response(status, &reason);
    if let Ok(request_id) = HeaderValue::from_str(permit.request_id()) {
        response.headers_mut().insert(REQUEST_ID_HEADER, request_id);
    }
    permit.fail(reason);
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
async fn relay_upstream_error(upstream: reqwest::Response, permit: JobPermit) -> Response {
    let status = upstream.status();
    let content_type = upstream_content_type(&upstream);
    // The status alone explains the failure; an unreadable error body is relayed as empty.
    let body = upstream.bytes().await.unwrap_or_default();
    let response = relay_response(status, content_type, permit.request_id(), Body::from(body));
    let reason = format!("upstream returned {status}");
    warn!(request_id = permit.request_id(), %reason, "job failed");
    permit.fail(reason);
    response
}

/// Reads a whole non-streamed reply, relays it, and completes the job.
async fn relay_buffered(upstream: reqwest::Response, finish: JobFinish) -> Response {
    let status = upstream.status();
    let content_type = upstream_content_type(&upstream);
    let body = match upstream.bytes().await {
        Ok(body) => body,
        Err(err) => {
            let reason = format!("failed to read upstream reply: {err}");
            return fail_job(finish.permit, StatusCode::BAD_GATEWAY, reason);
        }
    };
    let response_text = match buffered_response_text(&body) {
        Ok(text) => text,
        Err(reason) => return fail_job(finish.permit, StatusCode::BAD_GATEWAY, reason),
    };
    let request_id = &finish.turn.request_id;
    let response = relay_response(status, content_type, request_id, Body::from(body));
    finish.complete(response_text);
    response
}

/// Relays a streamed reply chunk by chunk; the job finishes when the stream does.
fn relay_stream(upstream: reqwest::Response, finish: JobFinish) -> Response {
    let status = upstream.status();
    let content_type = upstream_content_type(&upstream);
    let request_id = finish.turn.request_id.clone();
    let relay = StreamRelay {
        chunks: Box::pin(upstream.bytes_stream()),
        assembler: SseTextAssembler::default(),
        finish: Some(finish),
    };
    let body = Body::from_stream(stream::unfold(relay, StreamRelay::next_chunk));
    relay_response(status, content_type, &request_id, body)
}

/// The parts of a running request needed to record its outcome.
struct PendingTurn {
    request_id: String,
    conversation_id: Option<String>,
    request: Value,
}

/// Owns a running job's permit until its reply is fully delivered.
struct JobFinish {
    permit: JobPermit,
    turn: PendingTurn,
    sink: Arc<dyn HistorySink>,
}

impl JobFinish {
    /// Completes the job and hands the turn to the history sink.
    fn complete(self, response_text: String) {
        info!(request_id = self.permit.request_id(), "job completed");
        self.permit.complete();
        self.sink.record(CompletedTurn {
            request_id: self.turn.request_id,
            conversation_id: self.turn.conversation_id,
            request: self.turn.request,
            response_text,
        });
    }
}

type UpstreamChunks = Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>;

/// State of one streamed reply. Dropping it before the upstream stream ends (the client
/// disconnected) drops the permit, which cancels the job.
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
                let reason = format!("upstream stream failed: {err}");
                warn!(request_id = finish.permit.request_id(), %reason, "job failed");
                finish.permit.fail(reason);
                Some((Err(std::io::Error::other(err)), self))
            }
            None => match std::mem::take(&mut self.assembler).finish() {
                Ok(text) => {
                    finish.complete(text);
                    None
                }
                Err(reason) => {
                    warn!(request_id = finish.permit.request_id(), %reason, "job failed");
                    finish.permit.fail(reason.to_owned());
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
}

impl SseTextAssembler {
    fn push(&mut self, bytes: &[u8]) {
        self.partial_line.extend_from_slice(bytes);
        while let Some(newline) = self.partial_line.iter().position(|&byte| byte == b'\n') {
            let line: Vec<u8> = self.partial_line.drain(..=newline).collect();
            self.absorb_line(&line);
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
            self.done = true;
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
            self.text.push_str(fragment);
        }
    }

    /// A complete response must end with [DONE], including without a final newline.
    fn finish(mut self) -> Result<String, &'static str> {
        let rest = std::mem::take(&mut self.partial_line);
        self.absorb_line(&rest);
        if self.invalid {
            return Err("upstream SSE contained invalid or error data");
        }
        if !self.done {
            return Err("upstream SSE ended before [DONE]");
        }
        Ok(self.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::JobStatus;
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingSink(Mutex<Vec<CompletedTurn>>);

    impl HistorySink for RecordingSink {
        fn record(&self, turn: CompletedTurn) {
            self.0.lock().unwrap().push(turn);
        }
    }

    async fn check_stream(events: &str, succeeds: bool) {
        let queue = Arc::new(JobQueue::new());
        let sink = Arc::new(RecordingSink::default());
        let permit = queue.acquire(None).await;
        let request_id = permit.request_id().to_owned();
        let chunks = events
            .as_bytes()
            .chunks(3)
            .map(|chunk| Ok::<_, reqwest::Error>(Bytes::copy_from_slice(chunk)))
            .collect::<Vec<_>>();
        let relay = StreamRelay {
            chunks: Box::pin(stream::iter(chunks)),
            assembler: SseTextAssembler::default(),
            finish: Some(JobFinish {
                permit,
                turn: PendingTurn {
                    request_id,
                    conversation_id: None,
                    request: json!({}),
                },
                sink: sink.clone(),
            }),
        };
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
            let turns = sink.0.lock().unwrap();
            assert_eq!(turns.len(), usize::from(succeeds));
            if succeeds {
                assert_eq!(turns[0].response_text, "Hello");
            }
        }
        let next = tokio::time::timeout(Duration::from_secs(1), queue.acquire(None))
            .await
            .expect("stream outcome must release the running slot");
        next.complete();
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
        assert_eq!(assembler.finish().unwrap(), "Hello, wörld");
    }

    #[test]
    fn assembler_handles_crlf_and_a_final_line_without_newline() {
        let mut assembler = SseTextAssembler::default();
        assembler.push(delta_event("a").replace('\n', "\r\n").as_bytes());
        assembler.push(delta_event("b").as_bytes());
        assembler.push(b"data: [DONE]");
        assert_eq!(assembler.finish().unwrap(), "ab");
    }

    #[test]
    fn non_loopback_listen_addresses_are_refused() {
        assert!(ensure_loopback("127.0.0.1:8080".parse().unwrap()).is_ok());
        assert!(ensure_loopback("[::1]:8080".parse().unwrap()).is_ok());
        assert!(ensure_loopback("0.0.0.0:8080".parse().unwrap()).is_err());
        assert!(ensure_loopback("192.168.1.5:8080".parse().unwrap()).is_err());
    }
}
