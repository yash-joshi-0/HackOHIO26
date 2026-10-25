//! The local API proxies chat requests to a fake llama-server, one job at a time.
//!
//! Each test launches the `fake_llama_server` example (built by `cargo test`) on a free loopback
//! port and serves the API in-process in front of it.

use std::net::{SocketAddr, TcpListener};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::routing::post;
use llm_cluster::api::{self, ApiConfig, REQUEST_ID_HEADER};
use llm_cluster::history::{CompletedTurn, HistorySink, StartedTurn, TurnOutcome};
use llm_cluster::jobs::{JobQueue, JobStatus, MAX_OUTSTANDING_JOBS};
use llm_cluster::node::PeerTable;
use serde_json::{Value, json};
use tokio::process::{Child, Command};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

mod common;
use common::example_bin;

/// Upper bound for anything a test waits on, generous for slow CI machines.
const WAIT_LIMIT: Duration = Duration::from_secs(20);
const POLL_INTERVAL: Duration = Duration::from_millis(20);
/// How long the fake server sleeps before answering in the FIFO test.
const SLOW_REPLY_MS: u64 = 300;
/// A reply delay no test waits out: requests are abandoned while the fake server sleeps.
const STALLED_REPLY_MS: u64 = 60_000;

fn free_loopback_addr() -> SocketAddr {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .expect("free loopback port")
}

async fn wait_for<T, F, Fut>(what: &str, mut probe: F) -> T
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Option<T>>,
{
    let deadline = tokio::time::Instant::now() + WAIT_LIMIT;
    loop {
        if let Some(value) = probe().await {
            return value;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for {what}"
        );
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

async fn is_http_ok(url: &str) -> bool {
    match reqwest::get(url).await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

struct FakeLlama {
    addr: SocketAddr,
    child: Child,
}

impl FakeLlama {
    /// Launches the fake server without waiting for it to listen.
    fn spawn(extra_args: &[&str]) -> Self {
        let addr = free_loopback_addr();
        let child = Command::new(example_bin("fake_llama_server"))
            .args(["--host", "127.0.0.1", "--port", &addr.port().to_string()])
            .args(["--model", "ignored.gguf", "-ngl", "99"])
            .args(extra_args)
            .stdout(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .expect("spawn fake_llama_server");
        Self { addr, child }
    }

    /// Launches the fake server and waits until `/health` answers 200.
    async fn start(extra_args: &[&str]) -> Self {
        let fake = Self::spawn(extra_args);
        let health = fake.url("/health");
        wait_for("fake llama-server", || async {
            is_http_ok(&health).await.then_some(())
        })
        .await;
        fake
    }

    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }
}

/// One history hook call, without the request ID it was for.
#[derive(Clone, Debug, PartialEq)]
enum Hook {
    Started,
    Recorded,
    Ended(TurnOutcome),
}

#[derive(Default)]
struct RecordingSink {
    started: Mutex<Vec<StartedTurn>>,
    turns: Mutex<Vec<CompletedTurn>>,
    /// Every hook call in order, with its request ID.
    calls: Mutex<Vec<(String, Hook)>>,
}

impl RecordingSink {
    fn turns(&self) -> Vec<CompletedTurn> {
        self.turns.lock().unwrap().clone()
    }

    fn started_turns(&self) -> Vec<StartedTurn> {
        self.started.lock().unwrap().clone()
    }

    /// The hook calls made for one job, in order.
    fn calls_for(&self, request_id: &str) -> Vec<Hook> {
        let calls = self.calls.lock().unwrap();
        calls
            .iter()
            .filter(|(id, _)| id == request_id)
            .map(|(_, hook)| hook.clone())
            .collect()
    }

    /// Asserts the job was started and then failed, and returns the failure reason.
    fn failure_reason(&self, request_id: &str) -> String {
        match &self.calls_for(request_id)[..] {
            [Hook::Started, Hook::Ended(TurnOutcome::Failed { reason })] => reason.clone(),
            calls => panic!("expected started then failed for {request_id}, got {calls:?}"),
        }
    }

    fn log(&self, request_id: &str, hook: Hook) {
        self.calls
            .lock()
            .unwrap()
            .push((request_id.to_owned(), hook));
    }
}

impl HistorySink for RecordingSink {
    fn started(&self, turn: &StartedTurn) {
        self.started.lock().unwrap().push(turn.clone());
        self.log(&turn.request_id, Hook::Started);
    }

    fn record(&self, turn: CompletedTurn) {
        self.log(&turn.request_id, Hook::Recorded);
        self.turns.lock().unwrap().push(turn);
    }

    fn ended(&self, request_id: &str, outcome: TurnOutcome) {
        self.log(request_id, Hook::Ended(outcome));
    }
}

struct TestApi {
    addr: SocketAddr,
    sink: Arc<RecordingSink>,
    queue: Arc<JobQueue>,
    client: reqwest::Client,
    shutdown: Option<oneshot::Sender<()>>,
    server: JoinHandle<anyhow::Result<()>>,
}

impl TestApi {
    async fn start(upstream: SocketAddr) -> Self {
        let addr = free_loopback_addr();
        let sink = Arc::new(RecordingSink::default());
        let queue = Arc::new(JobQueue::new());
        let (shutdown, shutdown_rx) = oneshot::channel::<()>();
        let config = ApiConfig {
            listen: addr,
            upstream,
            peers: Arc::new(Mutex::new(PeerTable::new())),
        };
        let server = tokio::spawn(api::serve_api(
            config,
            Arc::clone(&queue),
            Arc::clone(&sink) as Arc<dyn HistorySink>,
            async move {
                // A dropped sender also means "shut down".
                let _ = shutdown_rx.await;
            },
        ));
        let jobs = format!("http://{addr}/jobs");
        wait_for("api", || async { is_http_ok(&jobs).await.then_some(()) }).await;
        Self {
            addr,
            sink,
            queue,
            client: reqwest::Client::new(),
            shutdown: Some(shutdown),
            server,
        }
    }

    fn chat_url(&self) -> String {
        format!("http://{}/v1/chat/completions", self.addr)
    }

    async fn post_chat(&self, body: &Value) -> reqwest::Response {
        self.client
            .post(self.chat_url())
            .json(body)
            .send()
            .await
            .expect("chat request")
    }

    /// Like `post_chat`, but returns a `'static` future that can be spawned and aborted.
    fn post_chat_owned(
        &self,
        body: Value,
    ) -> impl Future<Output = reqwest::Result<reqwest::Response>> + Send + 'static {
        self.client.post(self.chat_url()).json(&body).send()
    }

    async fn jobs(&self) -> Vec<Value> {
        let url = format!("http://{}/jobs", self.addr);
        let jobs: Value = reqwest::get(url).await.unwrap().json().await.unwrap();
        jobs.as_array().expect("jobs array").clone()
    }

    async fn job_status(&self, request_id: &str) -> String {
        let jobs = self.jobs().await;
        let job = jobs
            .iter()
            .find(|job| job["request_id"] == request_id)
            .expect("job listed");
        job["status"].as_str().unwrap().to_owned()
    }

    async fn stop(mut self) {
        self.shutdown.take().unwrap().send(()).unwrap();
        let result = tokio::time::timeout(WAIT_LIMIT, self.server)
            .await
            .expect("api stops after shutdown")
            .unwrap();
        result.expect("api exits cleanly");
    }
}

fn chat_body(text: &str) -> Value {
    json!({"messages": [{"role": "user", "content": text}]})
}

fn request_id_of(response: &reqwest::Response) -> String {
    response.headers()[REQUEST_ID_HEADER]
        .to_str()
        .unwrap()
        .to_owned()
}

#[tokio::test]
async fn truncated_upstream_sse_fails_client_body_and_does_not_commit_history() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router =
        axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(|| async {
                let chunks =
                    futures_util::stream::unfold(false, |sent| async move {
                        if sent {
                            tokio::time::sleep(Duration::from_millis(50)).await;
                            None
                        } else {
                            Some((Ok::<_, std::io::Error>(axum::body::Bytes::from_static(
                    b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n"
                )), true))
                        }
                    });
                (
                    [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                    axum::body::Body::from_stream(chunks),
                )
            }),
        );
    let upstream = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let api = TestApi::start(address).await;
    let mut body = chat_body("test interrupted answer");
    body["stream"] = json!(true);
    let response = api.post_chat(&body).await;
    let request_id = request_id_of(&response);
    assert!(
        response.bytes().await.is_err(),
        "incomplete SSE must not look like a complete HTTP response"
    );
    wait_for("failed job", || async {
        (api.job_status(&request_id).await == "failed").then_some(())
    })
    .await;
    assert!(
        api.sink.turns().is_empty(),
        "partial response must not become completed history"
    );
    api.sink.failure_reason(&request_id);
    api.stop().await;
    upstream.abort();
    let _ = upstream.await;
}

#[tokio::test]
async fn non_streaming_requests_are_proxied_and_strip_conversation_id() {
    let fake = FakeLlama::start(&[]).await;
    let api = TestApi::start(fake.addr).await;

    assert_eq!(
        api::prompt(api.addr, "hello mesh").await.unwrap(),
        "hello mesh"
    );

    let mut body = chat_body("which conversation?");
    body["conversation_id"] = json!("conv-1");
    // The fake server rejects requests that still carry conversation_id, so 200 proves stripping.
    let response = api.post_chat(&body).await;
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let request_id = request_id_of(&response);
    let reply: Value = response.json().await.unwrap();
    assert_eq!(
        reply["choices"][0]["message"]["content"],
        "which conversation?"
    );

    let turns = api.sink.turns();
    assert_eq!(turns.len(), 2);
    assert_eq!(turns[0].response_text, "hello mesh");
    assert_eq!(turns[0].conversation_id, None);
    assert_eq!(turns[1].request_id, request_id);
    assert_eq!(turns[1].conversation_id.as_deref(), Some("conv-1"));
    assert_eq!(turns[1].response_text, "which conversation?");
    assert!(turns[1].request.get("conversation_id").is_none());

    let started = api.sink.started_turns();
    assert_eq!(started.len(), 2);
    for (started, completed) in started.iter().zip(&turns) {
        assert_eq!(started.request_id, completed.request_id);
        assert_eq!(started.conversation_id, completed.conversation_id);
        assert_eq!(started.request, completed.request);
        assert_eq!(
            api.sink.calls_for(&completed.request_id),
            [Hook::Started, Hook::Recorded]
        );
    }

    let jobs = api.jobs().await;
    assert_eq!(jobs.len(), 2);
    assert!(jobs.iter().all(|job| job["status"] == "completed"));
    assert_eq!(jobs[1]["request_id"], request_id.as_str());
    assert_eq!(jobs[1]["conversation_id"], "conv-1");
    api.stop().await;
}

#[tokio::test]
async fn streamed_replies_are_relayed_chunk_by_chunk() {
    let fake = FakeLlama::start(&[]).await;
    let api = TestApi::start(fake.addr).await;
    let text = "one two three four five six seven eight";
    let mut body = chat_body(text);
    body["stream"] = json!(true);
    body["conversation_id"] = json!("conv-stream");

    let mut response = api.post_chat(&body).await;
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let request_id = request_id_of(&response);
    let first = response.chunk().await.unwrap().expect("first chunk");
    // The upstream is still streaming, so the proxy is not buffering the whole reply.
    assert_eq!(api.job_status(&request_id).await, "running");

    let mut received = first.to_vec();
    let mut chunk_count = 1;
    while let Some(chunk) = response.chunk().await.unwrap() {
        received.extend_from_slice(&chunk);
        chunk_count += 1;
    }
    let received = String::from_utf8(received).unwrap();
    assert!(
        chunk_count > 1,
        "expected several chunks, got {chunk_count}"
    );
    assert!(received.matches("data: ").count() > text.split(' ').count());
    assert!(received.trim_end().ends_with("data: [DONE]"));

    let turns = api.sink.turns();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].response_text, text);
    assert_eq!(turns[0].conversation_id.as_deref(), Some("conv-stream"));
    assert_eq!(
        api.sink.calls_for(&request_id),
        [Hook::Started, Hook::Recorded]
    );
    assert_eq!(api.sink.started_turns()[0].request, turns[0].request);
    assert_eq!(api.job_status(&request_id).await, "completed");
    api.stop().await;
}

#[tokio::test]
async fn client_disconnect_mid_stream_cancels_the_job() {
    let fake = FakeLlama::start(&[]).await;
    let api = TestApi::start(fake.addr).await;
    let mut body = chat_body(&"word ".repeat(100));
    body["stream"] = json!(true);

    let mut response = api.post_chat(&body).await;
    let request_id = request_id_of(&response);
    response.chunk().await.unwrap().expect("first chunk");
    drop(response);

    wait_for("job canceled", || async {
        (api.job_status(&request_id).await == "canceled").then_some(())
    })
    .await;
    assert!(api.sink.turns().is_empty());
    assert_eq!(
        api.sink.calls_for(&request_id),
        [Hook::Started, Hook::Ended(TurnOutcome::Canceled)]
    );
    assert_eq!(api::prompt(api.addr, "next").await.unwrap(), "next");
    api.stop().await;
}

/// Request ID of the first job listed with `status`, once one exists.
async fn wait_for_job_with_status(api: &TestApi, status: &str) -> String {
    wait_for(status, || async {
        api.jobs()
            .await
            .iter()
            .find(|job| job["status"] == status)
            .map(|job| job["request_id"].as_str().unwrap().to_owned())
    })
    .await
}

async fn wait_for_status(api: &TestApi, request_id: &str, status: &str) {
    wait_for(status, || async {
        (api.job_status(request_id).await == status).then_some(())
    })
    .await;
}

#[tokio::test]
async fn queued_cancels_make_no_history_calls_and_running_cancels_end_canceled() {
    // Slow enough that both requests are abandoned long before the fake server answers.
    let fake = FakeLlama::start(&["--delay-ms", &STALLED_REPLY_MS.to_string()]).await;
    let api = TestApi::start(fake.addr).await;
    let running = tokio::spawn(api.post_chat_owned(chat_body("running")));
    let running_id = wait_for_job_with_status(&api, "running").await;
    let queued = tokio::spawn(api.post_chat_owned(chat_body("queued")));
    let queued_id = wait_for_job_with_status(&api, "queued").await;

    queued.abort();
    wait_for_status(&api, &queued_id, "canceled").await;
    assert!(api.sink.calls_for(&queued_id).is_empty());

    running.abort();
    wait_for_status(&api, &running_id, "canceled").await;
    assert_eq!(
        api.sink.calls_for(&running_id),
        [Hook::Started, Hook::Ended(TurnOutcome::Canceled)]
    );
    assert!(api.sink.turns().is_empty());
    api.stop().await;
}

#[tokio::test]
async fn concurrent_requests_run_strictly_one_after_another() {
    let fake = FakeLlama::start(&["--delay-ms", &SLOW_REPLY_MS.to_string()]).await;
    let api = TestApi::start(fake.addr).await;
    let first = tokio::spawn(api::prompt(api.addr, "first"));
    let second = tokio::spawn(api::prompt(api.addr, "second"));

    let observed = wait_for("one running and one queued job", || async {
        let statuses: Vec<String> = api
            .jobs()
            .await
            .iter()
            .map(|job| job["status"].as_str().unwrap().to_owned())
            .collect();
        let is_running = statuses.iter().any(|status| status == "running");
        let is_queued = statuses.iter().any(|status| status == "queued");
        (is_running && is_queued).then_some(statuses)
    })
    .await;
    assert_eq!(observed.len(), 2);

    assert_eq!(first.await.unwrap().unwrap(), "first");
    assert_eq!(second.await.unwrap().unwrap(), "second");
    let mut jobs = api.jobs().await;
    assert!(jobs.iter().all(|job| job["status"] == "completed"));
    jobs.sort_by_key(|job| job["started_at_unix_ms"].as_u64().unwrap());
    let earlier_finished = jobs[0]["finished_at_unix_ms"].as_u64().unwrap();
    let later_started = jobs[1]["started_at_unix_ms"].as_u64().unwrap();
    assert!(
        later_started >= earlier_finished,
        "second job started at {later_started} before the first finished at {earlier_finished}"
    );
    api.stop().await;
}

#[tokio::test]
async fn full_inference_queue_returns_429_without_registering_overflow() {
    let fake = FakeLlama::start(&[]).await;
    let api = TestApi::start(fake.addr).await;
    let first = api.queue.acquire(None).await;
    let waiters: Vec<_> = (0..MAX_OUTSTANDING_JOBS - 1)
        .map(|index| {
            let queue = Arc::clone(&api.queue);
            tokio::spawn(async move { queue.acquire(Some(format!("queued-{index}"))).await })
        })
        .collect();
    tokio::time::sleep(POLL_INTERVAL * 3).await;

    let response = api.post_chat(&chat_body("overflow")).await;
    assert_eq!(response.status(), reqwest::StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(api.queue.jobs().len(), MAX_OUTSTANDING_JOBS);

    first.complete();
    for waiter in waiters {
        waiter.await.unwrap().complete();
    }
    api.stop().await;
}

#[tokio::test]
async fn upstream_failures_mark_jobs_failed() {
    let fake = FakeLlama::start(&["--fail-completions"]).await;
    let api = TestApi::start(fake.addr).await;
    let response = api.post_chat(&chat_body("hi")).await;
    assert_eq!(
        response.status(),
        reqwest::StatusCode::INTERNAL_SERVER_ERROR
    );
    let request_id = request_id_of(&response);
    assert!(api::prompt(api.addr, "hi").await.is_err());
    let jobs = api.jobs().await;
    assert_eq!(jobs.len(), 2);
    assert!(jobs.iter().all(|job| job["status"] == "failed"));
    assert_eq!(jobs[0]["request_id"], request_id.as_str());
    assert!(jobs[0]["error"].as_str().unwrap().contains("500"));
    assert!(api.sink.turns().is_empty());
    for job in &jobs {
        let reason = api.sink.failure_reason(job["request_id"].as_str().unwrap());
        assert_eq!(job["error"], reason.as_str());
    }
    api.stop().await;

    let unreachable = TestApi::start(free_loopback_addr()).await;
    let response = unreachable.post_chat(&chat_body("hi")).await;
    assert_eq!(response.status(), reqwest::StatusCode::BAD_GATEWAY);
    let request_id = request_id_of(&response);
    let jobs = unreachable.jobs().await;
    assert_eq!(jobs[0]["status"], "failed");
    assert!(jobs[0]["error"].as_str().unwrap().contains("upstream"));
    assert!(
        unreachable
            .sink
            .failure_reason(&request_id)
            .contains("upstream")
    );
    unreachable.stop().await;
}

async fn upstream_with_success_body(body: &'static str) -> (SocketAddr, JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fake upstream");
    let addr = listener.local_addr().expect("fake upstream address");
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move || async move {
            (
                axum::http::StatusCode::OK,
                [("content-type", "application/json")],
                body,
            )
        }),
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve fake upstream")
    });
    (addr, server)
}

#[tokio::test]
async fn invalid_success_replies_fail_job_and_are_not_saved_to_history() {
    for body in ["not json", r#"{"choices":[{"message":{}}]}"#] {
        let (upstream, server) = upstream_with_success_body(body).await;
        let api = TestApi::start(upstream).await;
        let response = api.post_chat(&chat_body("hello")).await;

        assert_eq!(response.status(), reqwest::StatusCode::BAD_GATEWAY);
        let request_id = request_id_of(&response);
        let jobs = api.jobs().await;
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0]["request_id"], request_id);
        assert_eq!(jobs[0]["status"], "failed");
        assert!(jobs[0]["error"].as_str().is_some());
        assert!(api.sink.turns().is_empty());
        api.sink.failure_reason(&request_id);

        api.stop().await;
        server.abort();
    }
}

#[tokio::test]
async fn malformed_requests_are_rejected_without_a_job() {
    let api = TestApi::start(free_loopback_addr()).await;
    let response = api
        .client
        .post(api.chat_url())
        .body("not json")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    assert!(api.jobs().await.is_empty());

    let response = api
        .post_chat(&json!({"messages": [], "conversation_id": "x".repeat(257)}))
        .await;
    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
    assert!(api.jobs().await.is_empty());
    api.stop().await;
}

#[tokio::test]
async fn non_loopback_listen_address_is_refused() {
    let config = ApiConfig {
        listen: "0.0.0.0:0".parse().unwrap(),
        upstream: free_loopback_addr(),
        peers: Arc::new(Mutex::new(PeerTable::new())),
    };
    let result = tokio::time::timeout(
        WAIT_LIMIT,
        api::serve_api(
            config,
            Arc::new(JobQueue::new()),
            Arc::new(RecordingSink::default()),
            std::future::pending(),
        ),
    )
    .await
    .expect("refused without serving");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("loopback"), "unexpected error: {err}");
}

#[tokio::test]
async fn fake_server_logs_its_argv_and_ignores_llama_server_flags() {
    let log = std::env::temp_dir().join(format!(
        "meshthing-fake-llama-argv-{}.json",
        std::process::id()
    ));
    // A leftover log from an aborted run may or may not exist; either way we start fresh.
    let _ = std::fs::remove_file(&log);
    let log_arg = log.to_str().unwrap();
    let fake = FakeLlama::start(&["--argv-log", log_arg]).await;

    let logged: Value = serde_json::from_slice(&std::fs::read(&log).unwrap()).unwrap();
    assert_eq!(logged["pid"].as_u64(), fake.child.id().map(u64::from));
    let port = fake.addr.port().to_string();
    let expected = json!([
        "--host",
        "127.0.0.1",
        "--port",
        port,
        "--model",
        "ignored.gguf",
        "-ngl",
        "99",
        "--argv-log",
        log_arg,
    ]);
    assert_eq!(logged["args"], expected);
    std::fs::remove_file(&log).unwrap();
}

#[tokio::test]
async fn fake_server_can_fail_health_forever() {
    let fake = FakeLlama::spawn(&["--fail-health"]);
    let health = fake.url("/health");
    let status = wait_for("fake llama-server to answer", || async {
        reqwest::get(&health)
            .await
            .ok()
            .map(|response| response.status())
    })
    .await;
    assert_eq!(status, reqwest::StatusCode::SERVICE_UNAVAILABLE);
}

#[test]
fn job_status_serializes_as_lowercase() {
    let json = serde_json::to_value(JobStatus::Queued).unwrap();
    assert_eq!(json, "queued");
}
