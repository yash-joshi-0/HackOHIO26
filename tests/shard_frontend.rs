//! The shard frontend and pipeline drive `fake_shard_runner` stages end to end.
//!
//! Local stages are fake-runner children; a remote stage is played by an in-test worker that
//! accepts the RPC bridge ALPN, spawns a fake-runner prefix, adds `model_sha256` to its metadata
//! (as `shard-serve` and `shard_endpoint.py` do) and relays bytes both ways. Endpoints run on
//! loopback with relays disabled and in-memory address lookup.

use std::collections::{BTreeMap, HashSet};
use std::net::{SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use iroh::address_lookup::memory::MemoryLookup;
use iroh::endpoint::{Connection, presets};
use iroh::{Endpoint, EndpointAddr, RelayMode};
use llm_cluster::protocol::RPC_BRIDGE_ALPN;
use llm_cluster::shard::frontend::{FrontendConfig, MAX_GENERATION_RETRIES, serve_frontend};
use llm_cluster::shard::manifest::{ShardManifest, sha256_file};
use llm_cluster::shard::pipeline::{
    FailureClass, FinishReason, GenerationLimits, GenerationState, LocalLaunch, Pipeline,
    StageExpectation, StageIdentity, StageLink, classify,
};
use llm_cluster::shard::wire::{self, MAX_BATCH_TOKENS};
use serde_json::{Value, json};
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tracing_subscriber::EnvFilter;

mod common;
use common::example_bin;

/// `source_sha256` carried by `examples/fake-shard-model.json`.
const FIXTURE_SHA: &str = "5ca1ab1e5ca1ab1e5ca1ab1e5ca1ab1e5ca1ab1e5ca1ab1e5ca1ab1e5ca1ab1e";
/// A model hash no stage serves.
const OTHER_SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";
/// Layers in the fixture model.
const LAYERS: u32 = 4;
/// Context window for most tests.
const CONTEXT: u32 = 512;
/// Upper bound for anything a test waits on, generous for slow CI machines.
const TEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Metadata and step bound for pipelines built directly by the tests.
const IO_TIMEOUT: Duration = Duration::from_secs(20);
/// Bound on what the silent worker buffers while it waits for the leader to leave.
const SILENT_WORKER_READ_LIMIT: usize = 1 << 20;
const POLL_INTERVAL: Duration = Duration::from_millis(20);
const LOOPBACK_ANY_PORT: &str = "127.0.0.1:0";
const SSE_DONE: &str = "[DONE]";
/// Runtime buffer budget passed to every stage.
const BUDGET_BYTES: u64 = 1_048_576;
/// Makes the fake runner exit on receiving the step it names.
const FAIL_AT_STEP_ENV: &str = "FAKE_SHARD_FAIL_AT_STEP";
/// Makes the fake runner log every request header it receives.
const LOG_ENV: &str = "FAKE_SHARD_LOG";
/// Limits `FAKE_SHARD_FAIL_AT_STEP` to the first runner, which creates this marker file.
const FAIL_ONCE_FILE_ENV: &str = "FAKE_SHARD_FAIL_ONCE_FILE";

fn init_tracing() {
    // Another test in this binary may already have installed the subscriber; that is fine.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
}

fn fixture_model() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("fake-shard-model.json")
}

fn fixture_reply() -> String {
    let text = std::fs::read_to_string(fixture_model()).expect("read model fixture");
    let model: Value = serde_json::from_str(&text).expect("parse model fixture");
    model["reply"].as_str().expect("fixture reply").to_owned()
}

/// A fresh scratch path under the system temp directory.
fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("shard-frontend-{}-{name}", uuid::Uuid::new_v4()))
}

/// Writes a fake model without `source_sha256` (like a plain GGUF) and returns its path.
fn write_plain_model(reply: &str) -> PathBuf {
    let path = temp_path("model.json");
    let model = json!({"layers": LAYERS, "embd": 8, "vocab": 300, "reply": reply});
    std::fs::write(&path, model.to_string()).expect("write plain model");
    path
}

fn local_stage(begin: u32, end: u32, model: &Path, env: &[(&str, &str)]) -> Value {
    let env: BTreeMap<_, _> = env.iter().copied().collect();
    json!({"begin": begin, "end": end, "local": {
        "model": model, "budget_bytes": BUDGET_BYTES, "env": env,
    }})
}

fn remote_stage(begin: u32, end: u32, worker: &Endpoint) -> Value {
    json!({"begin": begin, "end": end, "remote": {"worker": worker.id().to_string()}})
}

fn manifest(model_sha256: &str, context: u32, stages: Vec<Value>) -> ShardManifest {
    let value = json!({
        "runner": example_bin("fake_shard_runner"),
        "model_sha256": model_sha256,
        "layers": LAYERS,
        "context": context,
        "max_tokens": 256,
        "chat_template": "chatml",
        "stages": stages,
    });
    let manifest: ShardManifest = serde_json::from_value(value).expect("parse manifest");
    manifest.validate().expect("valid manifest");
    manifest
}

/// Binds an endpoint on loopback only, with relays and public address lookup disabled.
async fn bind_endpoint(alpns: Vec<Vec<u8>>, lookup: Option<MemoryLookup>) -> Endpoint {
    let mut builder = Endpoint::builder(presets::Minimal)
        .relay_mode(RelayMode::Disabled)
        .clear_ip_transports()
        .bind_addr(LOOPBACK_ANY_PORT)
        .expect("valid bind address")
        .alpns(alpns);
    if let Some(lookup) = lookup {
        builder = builder.address_lookup(lookup);
    }
    builder.bind().await.expect("bind endpoint")
}

/// The endpoint's id plus its bound loopback sockets, so the leader dials it directly.
fn direct_addr(endpoint: &Endpoint) -> EndpointAddr {
    endpoint
        .bound_sockets()
        .into_iter()
        .fold(EndpointAddr::new(endpoint.id()), EndpointAddr::with_ip_addr)
}

fn free_loopback_addr() -> SocketAddr {
    TcpListener::bind(LOOPBACK_ANY_PORT)
        .and_then(|listener| listener.local_addr())
        .expect("free loopback port")
}

/// What the in-test worker serves and how it labels the metadata it forwards.
#[derive(Clone, Copy)]
struct WorkerStage {
    begin: u32,
    end: u32,
    /// `model_sha256` added to the forwarded metadata, or `None` to omit it.
    model_sha256: Option<&'static str>,
}

/// Plays `shard-serve`: one fake-runner session per incoming connection.
struct FakeWorker {
    endpoint: Endpoint,
    task: JoinHandle<()>,
    /// Sessions (leader connections) served so far.
    sessions: Arc<AtomicUsize>,
}

impl FakeWorker {
    async fn start(stage: WorkerStage) -> Self {
        Self::start_failing(stage, None).await
    }

    /// Like [`FakeWorker::start`], but every session's runner exits on receiving step
    /// `fail_at_step`, if set.
    async fn start_failing(stage: WorkerStage, fail_at_step: Option<u32>) -> Self {
        let env = fail_at_step.map(|step| (FAIL_AT_STEP_ENV, step.to_string()));
        Self::start_with_env(stage, env.into_iter().collect()).await
    }

    /// Like [`FakeWorker::start_failing`], but only the first session's runner fails: it
    /// creates `marker`, which later runners find.
    async fn start_failing_once(stage: WorkerStage, fail_at_step: u32, marker: &Path) -> Self {
        let env = vec![
            (FAIL_AT_STEP_ENV, fail_at_step.to_string()),
            (FAIL_ONCE_FILE_ENV, marker.to_string_lossy().into_owned()),
        ];
        Self::start_with_env(stage, env).await
    }

    /// Starts a worker whose session runners get `env`.
    async fn start_with_env(stage: WorkerStage, env: Vec<(&'static str, String)>) -> Self {
        let endpoint = bind_endpoint(vec![RPC_BRIDGE_ALPN.to_vec()], None).await;
        let accepting = endpoint.clone();
        let sessions = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&sessions);
        let task = tokio::spawn(async move {
            while let Some(incoming) = accepting.accept().await {
                let Ok(connection) = incoming.await else {
                    continue;
                };
                counter.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(serve_session(connection, stage, env.clone()));
            }
        });
        Self {
            endpoint,
            task,
            sessions,
        }
    }

    fn sessions(&self) -> usize {
        self.sessions.load(Ordering::SeqCst)
    }

    async fn stop(self) {
        self.endpoint.close().await;
        self.task.abort();
    }
}

/// One worker session: preface, runner, metadata with `model_sha256`, then a byte relay.
async fn serve_session(
    connection: Connection,
    stage: WorkerStage,
    env: Vec<(&'static str, String)>,
) {
    let Ok((mut send, mut recv)) = connection.accept_bi().await else {
        return;
    };
    wire::read_preface(&mut recv).await.expect("client preface");
    let mut child = Command::new(example_bin("fake_shard_runner"))
        .envs(env)
        .arg(fixture_model())
        .args([
            stage.begin.to_string(),
            stage.end.to_string(),
            CONTEXT.to_string(),
            "0".to_owned(),
            BUDGET_BYTES.to_string(),
            "placeholder".to_owned(),
            "0".to_owned(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn worker runner");
    let mut stdin = child.stdin.take().expect("runner stdin");
    let mut stdout = BufReader::new(child.stdout.take().expect("runner stdout"));
    let mut meta = wire::read_metadata(&mut stdout)
        .await
        .expect("runner metadata");
    meta.model_sha256 = stage.model_sha256.map(str::to_owned);
    wire::write_metadata(&mut send, &meta)
        .await
        .expect("forward metadata");
    let upstream = tokio::io::copy(&mut recv, &mut stdin);
    let downstream = async {
        tokio::io::copy(&mut stdout, &mut send).await?;
        send.shutdown().await
    };
    // Either side ending ends the session; the runner dies with it.
    tokio::select! {
        _ = upstream => {}
        _ = downstream => {}
    }
    // Best-effort cleanup: the runner may already have exited on its own.
    let _ = child.start_kill();
    let _ = child.wait().await;
}

/// A worker that greets nobody: it takes the preface, never sends metadata, and reports when
/// the leader's stream goes away.
async fn start_silent_worker() -> (Endpoint, oneshot::Receiver<()>, oneshot::Receiver<()>) {
    let endpoint = bind_endpoint(vec![RPC_BRIDGE_ALPN.to_vec()], None).await;
    let accepting = endpoint.clone();
    let (started_tx, started_rx) = oneshot::channel();
    let (closed_tx, closed_rx) = oneshot::channel();
    tokio::spawn(async move {
        let incoming = accepting.accept().await.expect("incoming connection");
        let connection = incoming.await.expect("connection");
        let (_send, mut recv) = connection.accept_bi().await.expect("stream");
        wire::read_preface(&mut recv).await.expect("client preface");
        let _ = started_tx.send(());
        // Returns once the leader closes the stream or drops the connection.
        let _ = recv.read_to_end(SILENT_WORKER_READ_LIMIT).await;
        let _ = closed_tx.send(());
    });
    (endpoint, started_rx, closed_rx)
}

/// A running frontend and the endpoint it dials from.
struct Frontend {
    addr: SocketAddr,
    endpoint: Endpoint,
    shutdown: oneshot::Sender<()>,
    task: JoinHandle<anyhow::Result<()>>,
}

impl Frontend {
    /// Starts the frontend; `workers` are made dialable through in-memory lookup.
    async fn start(manifest: ShardManifest, workers: &[&FakeWorker]) -> Self {
        let endpoints: Vec<&Endpoint> = workers.iter().map(|worker| &worker.endpoint).collect();
        Self::start_with(manifest, &endpoints).await
    }

    async fn start_with(manifest: ShardManifest, workers: &[&Endpoint]) -> Self {
        init_tracing();
        let lookup = MemoryLookup::new();
        for worker in workers {
            lookup.add_endpoint_info(direct_addr(worker));
        }
        let endpoint = bind_endpoint(Vec::new(), Some(lookup)).await;
        let addr = free_loopback_addr();
        let (shutdown, stopped) = oneshot::channel::<()>();
        let config = FrontendConfig {
            listen: addr,
            manifest,
            endpoint: endpoint.clone(),
            ring: None,
        };
        let task = tokio::spawn(serve_frontend(config, async {
            let _ = stopped.await;
        }));
        let health = format!("http://{addr}/health");
        timeout(TEST_TIMEOUT, async {
            loop {
                if let Ok(response) = reqwest::get(&health).await
                    && response.status().is_success()
                {
                    return;
                }
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        })
        .await
        .expect("frontend becomes healthy");
        Self {
            addr,
            endpoint,
            shutdown,
            task,
        }
    }

    async fn stop(self) {
        let _ = self.shutdown.send(());
        timeout(TEST_TIMEOUT, self.task)
            .await
            .expect("frontend stops")
            .expect("frontend task did not panic")
            .expect("frontend exits cleanly");
        self.endpoint.close().await;
    }

    async fn chat(&self, stream: bool, content: &str) -> (u16, String) {
        let request = json!({
            "model": "anything",
            "temperature": 0.7,
            "stream": stream,
            "messages": [{"role": "user", "content": content}],
        });
        let response = timeout(
            TEST_TIMEOUT,
            reqwest::Client::new()
                .post(format!("http://{}/v1/chat/completions", self.addr))
                .json(&request)
                .send(),
        )
        .await
        .expect("chat request timed out")
        .expect("chat request");
        let status = response.status().as_u16();
        let body = timeout(TEST_TIMEOUT, response.text())
            .await
            .expect("chat body timed out")
            .expect("chat body");
        (status, body)
    }

    /// A buffered request that must succeed; returns the parsed completion.
    async fn chat_ok(&self) -> Value {
        let (status, body) = self.chat(false, "hi").await;
        assert_eq!(status, 200, "{body}");
        serde_json::from_str(&body).expect("completion json")
    }

    /// A buffered request that must fail; returns the error message.
    async fn chat_error(&self, stream: bool) -> String {
        let (status, body) = self.chat(stream, "hi").await;
        assert_eq!(status, 502, "{body}");
        let error: Value = serde_json::from_str(&body).expect("error json");
        error["error"]["message"]
            .as_str()
            .expect("error message")
            .to_owned()
    }
}

/// The `data:` payloads of an SSE body, in order.
fn sse_payloads(body: &str) -> Vec<String> {
    body.split("\n\n")
        .filter_map(|event| event.trim().strip_prefix("data:"))
        .map(|data| data.trim().to_owned())
        .collect()
}

/// Checks a successful stream's shape and returns the concatenated content.
fn streamed_text(body: &str) -> String {
    let payloads = sse_payloads(body);
    assert_eq!(
        payloads.last().map(String::as_str),
        Some(SSE_DONE),
        "{body}"
    );
    let chunks: Vec<Value> = payloads[..payloads.len() - 1]
        .iter()
        .map(|data| serde_json::from_str(data).expect("chunk json"))
        .collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let last = chunks.last().expect("final chunk");
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert!(
        chunks[..chunks.len() - 1]
            .iter()
            .all(|chunk| chunk["choices"][0]["finish_reason"].is_null())
    );
    chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect()
}

fn assert_completion(completion: &Value) {
    assert_eq!(
        completion["choices"][0]["message"]["content"],
        fixture_reply()
    );
    assert_eq!(completion["choices"][0]["finish_reason"], "stop");
    assert_eq!(completion["object"], "chat.completion");
}

#[tokio::test]
async fn single_local_stage_serves_buffered_and_streamed_completions() {
    let model = fixture_model();
    let manifest = manifest(
        FIXTURE_SHA,
        CONTEXT,
        vec![local_stage(0, LAYERS, &model, &[])],
    );
    let frontend = Frontend::start(manifest, &[]).await;
    assert_completion(&frontend.chat_ok().await);
    let (status, body) = frontend.chat(true, "hi").await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(streamed_text(&body), fixture_reply());
    frontend.stop().await;
}

#[tokio::test]
async fn remote_prefix_and_local_final_serve_completions() {
    let worker = FakeWorker::start(WorkerStage {
        begin: 0,
        end: 2,
        model_sha256: Some(FIXTURE_SHA),
    })
    .await;
    let model = fixture_model();
    let stages = vec![
        remote_stage(0, 2, &worker.endpoint),
        local_stage(2, LAYERS, &model, &[]),
    ];
    let frontend = Frontend::start(manifest(FIXTURE_SHA, CONTEXT, stages), &[&worker]).await;
    assert_completion(&frontend.chat_ok().await);
    let (status, body) = frontend.chat(true, "hi").await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(streamed_text(&body), fixture_reply());
    frontend.stop().await;
    worker.stop().await;
}

#[tokio::test]
async fn consecutive_requests_use_fresh_runners() {
    let log = temp_path("headers.log");
    let log_text = log.to_string_lossy().into_owned();
    let model = fixture_model();
    let stage = local_stage(0, LAYERS, &model, &[(LOG_ENV, &log_text)]);
    let frontend = Frontend::start(manifest(FIXTURE_SHA, CONTEXT, vec![stage]), &[]).await;
    assert_completion(&frontend.chat_ok().await);
    assert_completion(&frontend.chat_ok().await);
    frontend.stop().await;
    let lines = std::fs::read_to_string(&log).expect("read runner log");
    std::fs::remove_file(&log).expect("remove runner log");
    let first_steps: Vec<&str> = lines
        .lines()
        .filter(|line| line.contains(" step=0 "))
        .collect();
    assert_eq!(first_steps.len(), 2, "{lines}");
    let pids: HashSet<&str> = first_steps
        .iter()
        .filter_map(|line| line.split_whitespace().next())
        .collect();
    assert_eq!(pids.len(), 2, "each request needs its own runner: {lines}");
}

/// Runners that received step 0, one line each, from a `FAKE_SHARD_LOG` file (then removed).
fn runner_starts(log: &Path) -> Vec<String> {
    let lines = std::fs::read_to_string(log).expect("read runner log");
    std::fs::remove_file(log).expect("remove runner log");
    lines
        .lines()
        .filter(|line| line.contains(" step=0 "))
        .map(str::to_owned)
        .collect()
}

#[tokio::test]
async fn runner_failure_mid_generation_fails_the_request_after_the_retries() {
    let log = temp_path("headers.log");
    let log_text = log.to_string_lossy().into_owned();
    let model = fixture_model();
    let env = [(FAIL_AT_STEP_ENV, "2"), (LOG_ENV, log_text.as_str())];
    let stage = local_stage(0, LAYERS, &model, &env);
    let frontend = Frontend::start(manifest(FIXTURE_SHA, CONTEXT, vec![stage]), &[]).await;
    let message = frontend.chat_error(false).await;
    assert!(message.contains("step 2"), "{message}");
    let attempts = MAX_GENERATION_RETRIES as usize + 1;
    assert!(
        message.contains(&format!("after {attempts} attempts")),
        "{message}"
    );
    assert_eq!(runner_starts(&log).len(), attempts);

    let (status, body) = frontend.chat(true, "hi").await;
    assert_eq!(status, 200, "{body}");
    let payloads = sse_payloads(&body);
    assert!(!payloads.iter().any(|data| data == SSE_DONE), "{body}");
    let last: Value = serde_json::from_str(payloads.last().expect("events")).expect("json");
    assert!(last["error"]["message"].is_string(), "{body}");
    assert!(
        body.contains("\"content\":\"H\""),
        "text before the failure streams: {body}"
    );
    // Each attempt adds two tokens before it fails, and none is streamed twice.
    let pieces: String = payloads
        .iter()
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect();
    assert_eq!(pieces, fixture_reply()[..2 * attempts], "{body}");
    assert_eq!(runner_starts(&log).len(), attempts);
    frontend.stop().await;
}

#[tokio::test]
async fn a_stage_failing_once_is_resumed_without_repeating_text() {
    let worker = FakeWorker::start(WorkerStage {
        begin: 0,
        end: 2,
        model_sha256: Some(FIXTURE_SHA),
    })
    .await;
    let model = fixture_model();
    let marker = temp_path("fail-once");
    let marker_text = marker.to_string_lossy().into_owned();
    let env = [(FAIL_AT_STEP_ENV, "3"), (FAIL_ONCE_FILE_ENV, &marker_text)];
    let stages = vec![
        remote_stage(0, 2, &worker.endpoint),
        local_stage(2, LAYERS, &model, &env),
    ];
    let frontend = Frontend::start(manifest(FIXTURE_SHA, CONTEXT, stages), &[&worker]).await;
    let (status, body) = frontend.chat(true, "hi").await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(streamed_text(&body), fixture_reply(), "{body}");
    assert!(marker.exists(), "the first attempt failed");
    // The prefix worker was not at fault, so the retry dialled it again.
    assert_eq!(worker.sessions(), 2);

    std::fs::remove_file(&marker).expect("remove marker");
    let completion = frontend.chat_ok().await;
    assert_completion(&completion);
    let reply_len = fixture_reply().len();
    assert_eq!(completion["usage"]["completion_tokens"], reply_len);
    assert_eq!(worker.sessions(), 4);
    frontend.stop().await;
    worker.stop().await;
    std::fs::remove_file(&marker).expect("remove marker");
}

/// A remote stage served by `primary`, with `standby` workers behind it.
fn remote_stage_with_standby(
    begin: u32,
    end: u32,
    primary: String,
    standby: &[&Endpoint],
) -> Value {
    let standby: Vec<String> = standby
        .iter()
        .map(|worker| worker.id().to_string())
        .collect();
    json!({"begin": begin, "end": end, "remote": {"worker": primary, "standby": standby}})
}

/// The id of a worker nobody runs, which no lookup can find.
fn unreachable_worker_id() -> String {
    iroh::SecretKey::generate().public().to_string()
}

const PREFIX: WorkerStage = WorkerStage {
    begin: 0,
    end: 2,
    model_sha256: Some(FIXTURE_SHA),
};

#[tokio::test]
async fn an_unreachable_primary_hands_its_stage_to_the_standby() {
    let standby = FakeWorker::start(PREFIX).await;
    let model = fixture_model();
    let stages = vec![
        remote_stage_with_standby(0, 2, unreachable_worker_id(), &[&standby.endpoint]),
        local_stage(2, LAYERS, &model, &[]),
    ];
    let frontend = Frontend::start(manifest(FIXTURE_SHA, CONTEXT, stages), &[&standby]).await;
    assert_completion(&frontend.chat_ok().await);
    assert_eq!(standby.sessions(), 1);
    frontend.stop().await;
    standby.stop().await;
}

#[tokio::test]
async fn a_primary_failing_mid_generation_is_replaced_by_its_standby() {
    // The primary would recover, but the standby that has not failed is preferred.
    let marker = temp_path("primary-fails-once");
    let primary = FakeWorker::start_failing_once(PREFIX, 3, &marker).await;
    let standby = FakeWorker::start(PREFIX).await;
    let model = fixture_model();
    let primary_id = primary.endpoint.id().to_string();
    let stages = vec![
        remote_stage_with_standby(0, 2, primary_id, &[&standby.endpoint]),
        local_stage(2, LAYERS, &model, &[]),
    ];
    let workers = [&primary, &standby];
    let frontend = Frontend::start(manifest(FIXTURE_SHA, CONTEXT, stages), &workers).await;
    let (status, body) = frontend.chat(true, "hi").await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(streamed_text(&body), fixture_reply(), "{body}");
    assert!(marker.exists(), "the primary failed");
    assert_eq!((primary.sessions(), standby.sessions()), (1, 1));
    frontend.stop().await;
    primary.stop().await;
    standby.stop().await;
    std::fs::remove_file(&marker).expect("remove marker");
}

#[tokio::test]
async fn a_remote_stage_without_standby_retries_its_worker() {
    let marker = temp_path("worker-fails-once");
    let worker = FakeWorker::start_failing_once(PREFIX, 3, &marker).await;
    let model = fixture_model();
    let stages = vec![
        remote_stage(0, 2, &worker.endpoint),
        local_stage(2, LAYERS, &model, &[]),
    ];
    let frontend = Frontend::start(manifest(FIXTURE_SHA, CONTEXT, stages), &[&worker]).await;
    let (status, body) = frontend.chat(true, "hi").await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(streamed_text(&body), fixture_reply(), "{body}");
    assert!(marker.exists(), "the first attempt failed");
    assert_eq!(worker.sessions(), 2, "the only worker was dialled again");

    std::fs::remove_file(&marker).expect("remove marker");
    let completion = frontend.chat_ok().await;
    assert_completion(&completion);
    assert_eq!(
        completion["usage"]["completion_tokens"],
        fixture_reply().len()
    );
    assert_eq!(worker.sessions(), 4);
    frontend.stop().await;
    worker.stop().await;
    std::fs::remove_file(&marker).expect("remove marker");
}

#[tokio::test]
async fn a_stage_whose_workers_keep_failing_uses_up_the_retries() {
    let primary = FakeWorker::start_failing(PREFIX, Some(3)).await;
    let standby = FakeWorker::start_failing(PREFIX, Some(3)).await;
    let model = fixture_model();
    let primary_id = primary.endpoint.id().to_string();
    let stages = vec![
        remote_stage_with_standby(0, 2, primary_id, &[&standby.endpoint]),
        local_stage(2, LAYERS, &model, &[]),
    ];
    let workers = [&primary, &standby];
    let frontend = Frontend::start(manifest(FIXTURE_SHA, CONTEXT, stages), &workers).await;
    let message = frontend.chat_error(false).await;
    let attempts = MAX_GENERATION_RETRIES as usize + 1;
    assert!(
        message.contains(&format!("generation failed after {attempts} attempts")),
        "{message}"
    );
    // Primary, then the standby, then the primary again as the least recently failed.
    assert_eq!(
        attempts, 3,
        "the session counts below assume three attempts"
    );
    assert_eq!((primary.sessions(), standby.sessions()), (2, 1));
    let last = primary.endpoint.id().fmt_short().to_string();
    assert!(
        message.contains(&format!("stage 0 on worker {last}")),
        "{message}"
    );
    frontend.stop().await;
    primary.stop().await;
    standby.stop().await;
}

#[tokio::test]
async fn a_fatal_error_is_not_retried_or_failed_over() {
    let wrong_model = FakeWorker::start(WorkerStage {
        model_sha256: Some(OTHER_SHA),
        ..PREFIX
    })
    .await;
    let standby = FakeWorker::start(PREFIX).await;
    let model = fixture_model();
    let primary_id = wrong_model.endpoint.id().to_string();
    let stages = vec![
        remote_stage_with_standby(0, 2, primary_id, &[&standby.endpoint]),
        local_stage(2, LAYERS, &model, &[]),
    ];
    let workers = [&wrong_model, &standby];
    let frontend = Frontend::start(manifest(FIXTURE_SHA, CONTEXT, stages), &workers).await;
    let message = frontend.chat_error(false).await;
    assert!(message.contains("relay model_sha256"), "{message}");
    assert_eq!((wrong_model.sessions(), standby.sessions()), (1, 0));
    frontend.stop().await;
    wrong_model.stop().await;
    standby.stop().await;
}

#[tokio::test]
async fn client_disconnect_drops_the_generation() {
    let (worker, started, closed) = start_silent_worker().await;
    let model = fixture_model();
    let stages = vec![
        remote_stage(0, 2, &worker),
        local_stage(2, LAYERS, &model, &[]),
    ];
    let frontend = Frontend::start_with(manifest(FIXTURE_SHA, CONTEXT, stages), &[&worker]).await;
    let url = format!("http://{}/v1/chat/completions", frontend.addr);
    let request = tokio::spawn(async move {
        reqwest::Client::new()
            .post(url)
            .timeout(TEST_TIMEOUT)
            .json(&json!({"stream": true, "messages": [{"role": "user", "content": "hi"}]}))
            .send()
            .await
    });
    // Wait for an actual remote generation before disconnecting. A short wall
    // timeout can expire during the Iroh handshake under parallel test load.
    timeout(TEST_TIMEOUT, started)
        .await
        .expect("the remote generation starts")
        .expect("silent worker accepted the preface");
    assert!(
        !request.is_finished(),
        "silent worker must still block metadata"
    );
    request.abort();
    assert!(request.await.unwrap_err().is_cancelled());
    timeout(TEST_TIMEOUT, closed)
        .await
        .expect("the remote stream closes after the client leaves")
        .expect("silent worker reports");
    frontend.stop().await;
    worker.close().await;
}

#[tokio::test]
async fn local_stage_with_the_wrong_model_hash_is_refused() {
    let model = fixture_model();
    let stage = local_stage(0, LAYERS, &model, &[]);
    let frontend = Frontend::start(manifest(OTHER_SHA, CONTEXT, vec![stage]), &[]).await;
    let message = frontend.chat_error(false).await;
    assert!(message.contains("source_sha256"), "{message}");
    // Stage checks finish before any response bytes, so a stream fails the same way.
    let message = frontend.chat_error(true).await;
    assert!(message.contains("source_sha256"), "{message}");
    frontend.stop().await;
}

#[tokio::test]
async fn remote_stage_with_the_wrong_range_or_hash_is_refused() {
    let cases = [
        (
            WorkerStage {
                begin: 0,
                end: 1,
                model_sha256: Some(FIXTURE_SHA),
            },
            "serves layers 0..1",
        ),
        (
            WorkerStage {
                begin: 0,
                end: 2,
                model_sha256: Some(OTHER_SHA),
            },
            "relay model_sha256",
        ),
        (
            WorkerStage {
                begin: 0,
                end: 2,
                model_sha256: None,
            },
            "did not report model_sha256",
        ),
    ];
    for (served, expected) in cases {
        let worker = FakeWorker::start(served).await;
        let model = fixture_model();
        let stages = vec![
            remote_stage(0, 2, &worker.endpoint),
            local_stage(2, LAYERS, &model, &[]),
        ];
        let frontend = Frontend::start(manifest(FIXTURE_SHA, CONTEXT, stages), &[&worker]).await;
        let message = frontend.chat_error(false).await;
        assert!(message.contains(expected), "{message}");
        frontend.stop().await;
        worker.stop().await;
    }
}

#[tokio::test]
async fn plain_model_identity_comes_from_a_verified_artifact_hash() {
    let model = write_plain_model(&fixture_reply());
    let digest = sha256_file(&model).expect("hash plain model");
    let plain = local_stage(0, LAYERS, &model, &[]);

    let frontend = Frontend::start(manifest(FIXTURE_SHA, CONTEXT, vec![plain.clone()]), &[]).await;
    let message = frontend.chat_error(false).await;
    assert!(
        message.contains("identity cannot be established"),
        "{message}"
    );
    frontend.stop().await;

    let mut verified = plain.clone();
    verified["local"]["artifact_sha256"] = json!(digest);
    let frontend = Frontend::start(manifest(FIXTURE_SHA, CONTEXT, vec![verified]), &[]).await;
    assert_completion(&frontend.chat_ok().await);
    frontend.stop().await;

    let mut wrong = plain;
    wrong["local"]["artifact_sha256"] = json!(OTHER_SHA);
    let config = FrontendConfig {
        listen: free_loopback_addr(),
        manifest: manifest(FIXTURE_SHA, CONTEXT, vec![wrong]),
        endpoint: bind_endpoint(Vec::new(), None).await,
        ring: None,
    };
    let error = serve_frontend(config, std::future::pending())
        .await
        .expect_err("a mismatched artifact must stop startup");
    assert!(
        format!("{error:#}").contains("artifact_sha256"),
        "{error:#}"
    );
    std::fs::remove_file(&model).expect("remove plain model");
}

#[tokio::test]
async fn bad_requests_and_oversized_prompts_are_client_errors() {
    let model = fixture_model();
    let stage = local_stage(0, LAYERS, &model, &[]);
    let frontend = Frontend::start(manifest(FIXTURE_SHA, 32, vec![stage]), &[]).await;
    let response = reqwest::Client::new()
        .post(format!("http://{}/v1/chat/completions", frontend.addr))
        .json(&json!({"messages": [{"role": "user", "content": {"image": "x"}}]}))
        .send()
        .await
        .expect("bad request");
    assert_eq!(response.status().as_u16(), 400);
    let (status, body) = frontend.chat(false, &"long ".repeat(20)).await;
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("context window"), "{body}");
    frontend.stop().await;
}

#[tokio::test]
async fn non_loopback_listen_is_refused() {
    let model = fixture_model();
    let config = FrontendConfig {
        listen: "0.0.0.0:0".parse().expect("address"),
        manifest: manifest(
            FIXTURE_SHA,
            CONTEXT,
            vec![local_stage(0, LAYERS, &model, &[])],
        ),
        endpoint: bind_endpoint(Vec::new(), None).await,
        ring: None,
    };
    let error = serve_frontend(config, std::future::pending())
        .await
        .expect_err("non-loopback listen must be refused");
    assert!(error.to_string().contains("not loopback"), "{error}");
}

/// Spawns local fake-runner links for `ranges` directly, bypassing the manifest's v1 rules.
async fn local_links(
    model: &Path,
    ranges: &[(u32, u32)],
    context: u32,
    prompt: &str,
) -> Vec<StageLink> {
    local_links_with_env(model, ranges, context, prompt, &BTreeMap::new()).await
}

/// [`local_links`] with extra runner environment.
async fn local_links_with_env(
    model: &Path,
    ranges: &[(u32, u32)],
    context: u32,
    prompt: &str,
    env: &BTreeMap<String, String>,
) -> Vec<StageLink> {
    let runner = example_bin("fake_shard_runner");
    let mut links = Vec::new();
    for &(begin, end) in ranges {
        let launch = LocalLaunch {
            runner: &runner,
            model,
            begin,
            end,
            context,
            gpu_layers: 0,
            budget_bytes: BUDGET_BYTES,
            prompt,
            env,
        };
        links.push(
            StageLink::spawn_local(&launch, IO_TIMEOUT)
                .await
                .expect("spawn local link"),
        );
    }
    links
}

fn expectations(ranges: &[(u32, u32)], artifact_verified: bool) -> Vec<StageExpectation> {
    ranges
        .iter()
        .map(|&(begin, end)| StageExpectation {
            begin,
            end,
            layers: LAYERS,
            model_sha256: FIXTURE_SHA.to_owned(),
            identity: StageIdentity::Local { artifact_verified },
        })
        .collect()
}

#[tokio::test]
async fn pipeline_drives_three_stages_with_chunked_prefill() {
    init_tracing();
    let ranges = [(0, 1), (1, 3), (3, LAYERS)];
    // Long enough to need several prefill batches.
    let prompt = "a long prompt that spans batches ".repeat(10);
    let links = local_links(&fixture_model(), &ranges, CONTEXT, &prompt).await;
    let pipeline = Pipeline::new(links, &expectations(&ranges, false))
        .expect("valid pipeline")
        .with_step_timeout(IO_TIMEOUT);
    assert!(pipeline.prompt_tokens() > 2 * MAX_BATCH_TOKENS);
    let mut pieces = Vec::new();
    let limits = GenerationLimits {
        max_tokens: 256,
        context: CONTEXT,
    };
    let generation = timeout(
        TEST_TIMEOUT,
        pipeline.generate(limits, |text| pieces.push(text.to_owned())),
    )
    .await
    .expect("generation timed out")
    .expect("generation succeeds");
    assert_eq!(generation.text, fixture_reply());
    assert_eq!(pieces.concat(), fixture_reply());
    assert_eq!(generation.finish, FinishReason::Stop);
    assert_eq!(generation.completion_tokens as usize, fixture_reply().len());
}

#[tokio::test]
async fn step_timeout_identifies_the_read_phase() {
    let model = fixture_model();
    let runner = example_bin("fake_shard_runner");
    let env = BTreeMap::from([("FAKE_SHARD_DELAY_AT_STEP_MS".into(), "200".into())]);
    let launch = LocalLaunch {
        runner: &runner,
        model: &model,
        begin: 0,
        end: LAYERS,
        context: CONTEXT,
        gpu_layers: 0,
        budget_bytes: BUDGET_BYTES,
        prompt: "hi",
        env: &env,
    };
    let link = StageLink::spawn_local(&launch, IO_TIMEOUT).await.unwrap();
    let pipeline = Pipeline::new(vec![link], &expectations(&[(0, LAYERS)], false))
        .unwrap()
        .with_step_timeout(Duration::from_millis(50));
    let error = pipeline
        .generate(
            GenerationLimits {
                max_tokens: 1,
                context: CONTEXT,
            },
            |_| {},
        )
        .await
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("timed out reading step 0"),
        "{error:#}"
    );
}

#[tokio::test]
async fn pipeline_stops_at_max_tokens_and_at_the_context_limit() {
    init_tracing();
    let whole = [(0, LAYERS)];
    let links = local_links(&fixture_model(), &whole, CONTEXT, "hi").await;
    let pipeline = Pipeline::new(links, &expectations(&whole, false)).expect("valid pipeline");
    let limits = GenerationLimits {
        max_tokens: 5,
        context: CONTEXT,
    };
    let generation = pipeline.generate(limits, |_| {}).await.expect("generation");
    assert_eq!(
        (generation.text.as_str(), generation.finish),
        ("Hello", FinishReason::Length)
    );

    // "hi" is 3 tokens: the last prefill step samples one token and 29 decode steps fill the
    // remaining context, so 30 tokens come out before the window runs out.
    let model = write_plain_model(&"x".repeat(40));
    let context = 32;
    let links = local_links(&model, &whole, context, "hi").await;
    let pipeline = Pipeline::new(links, &expectations(&whole, true)).expect("valid pipeline");
    let limits = GenerationLimits {
        max_tokens: 256,
        context,
    };
    let generation = pipeline.generate(limits, |_| {}).await.expect("generation");
    assert_eq!(generation.finish, FinishReason::Length);
    assert_eq!(generation.completion_tokens, context - 2);
    assert_eq!(generation.text, "x".repeat(30));
    std::fs::remove_file(&model).expect("remove plain model");
}

#[tokio::test]
async fn a_resumed_pipeline_counts_limits_over_every_attempt() {
    init_tracing();
    let whole = [(0, LAYERS)];
    let marker = temp_path("fail-once");
    let env = BTreeMap::from([
        (FAIL_AT_STEP_ENV.to_owned(), "3".to_owned()),
        (
            FAIL_ONCE_FILE_ENV.to_owned(),
            marker.to_string_lossy().into_owned(),
        ),
    ]);
    let model = fixture_model();
    let pipeline_for = |prompt: &'static str| {
        let (model, env) = (&model, &env);
        async move {
            let links = local_links_with_env(model, &whole, CONTEXT, prompt, env).await;
            Pipeline::new(links, &expectations(&whole, false))
                .expect("valid pipeline")
                .with_step_timeout(IO_TIMEOUT)
        }
    };
    let limits = GenerationLimits {
        max_tokens: 5,
        context: CONTEXT,
    };
    let mut pieces = Vec::new();
    let mut state = GenerationState::new(limits, |text: String| {
        pieces.push(text);
        std::future::ready(())
    });

    // Steps 0..=2 sample "Hel", then the runner exits on step 3.
    let error = pipeline_for("hi")
        .await
        .resume(&mut state)
        .await
        .expect_err("step 3 fails");
    let classified = classify(&error);
    assert_eq!(classified.class, FailureClass::Retryable, "{error:#}");
    assert_eq!(classified.fault.map(|fault| fault.stage), Some(0));
    assert_eq!(state.completion_tokens(), 3);

    // A final stage that tokenizes the prompt differently cannot continue it.
    let error = pipeline_for("ho")
        .await
        .resume(&mut state)
        .await
        .expect_err("prompt mismatch");
    assert_eq!(classify(&error).class, FailureClass::Fatal, "{error:#}");
    assert!(
        format!("{error:#}").contains("tokenized the prompt"),
        "{error:#}"
    );

    // max_tokens counts the first attempt's three tokens too.
    let generation = timeout(TEST_TIMEOUT, pipeline_for("hi").await.resume(&mut state))
        .await
        .expect("resume timed out")
        .expect("resume succeeds");
    drop(state);
    assert_eq!(
        (
            generation.text.as_str(),
            generation.finish,
            generation.completion_tokens
        ),
        ("Hello", FinishReason::Length, 5)
    );
    assert_eq!(pieces, ["H", "e", "l", "l", "o"]);
    std::fs::remove_file(&marker).expect("remove marker");
}

#[tokio::test]
async fn pipeline_rejects_links_that_do_not_match_their_expectations() {
    init_tracing();
    let ranges = [(0, 2), (2, LAYERS)];
    let links = local_links(&fixture_model(), &ranges, CONTEXT, "hi").await;
    let wrong = expectations(&[(0, 1), (1, LAYERS)], false);
    let error = Pipeline::new(links, &wrong).err().expect("range mismatch");
    assert!(
        format!("{error:#}").contains("serves layers 0..2"),
        "{error:#}"
    );

    // A final stage in the middle of the pipeline is refused.
    let swapped = [(0, LAYERS), (0, LAYERS)];
    let links = local_links(&fixture_model(), &swapped, CONTEXT, "hi").await;
    let error = Pipeline::new(links, &expectations(&swapped, false))
        .err()
        .expect("final stage first");
    assert!(format!("{error:#}").contains("final stage"), "{error:#}");
}
