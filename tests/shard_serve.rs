//! End-to-end tests of `shard-serve`: a leader-side endpoint drives the `fake_shard_runner`
//! example through the worker's Iroh relay with the shard wire codec. Everything runs on loopback
//! with relays disabled, so the endpoints talk over a direct path without internet access.

use std::path::PathBuf;
use std::time::Duration;

use iroh::address_lookup::memory::MemoryLookup;
use iroh::endpoint::{Connection, ConnectionError, RecvStream, SendStream, presets};
use iroh::{Endpoint, EndpointAddr, EndpointId, RelayMode};
use llm_cluster::protocol::RPC_BRIDGE_ALPN;
use llm_cluster::rpc_bridge::{PEER_NOT_APPROVED, PEER_NOT_APPROVED_REASON, PeerAllowlist};
use llm_cluster::shard::serve::{
    ShardServeConfig, ShardServeOptions, serve_shard, serve_shard_with,
};
use llm_cluster::shard::wire::{self, SessionRequest, StageInput, StageMetadata};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, BufReader};
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tracing_subscriber::EnvFilter;

mod common;
use common::example_bin;

const TEST_TIMEOUT: Duration = Duration::from_secs(60);
const LOOPBACK_ANY_PORT: &str = "127.0.0.1:0";
/// The served prefix stage: layers `[BEGIN, END)` of the fixture's four.
const BEGIN: u32 = 0;
const END: u32 = 2;
const CONTEXT: u32 = 128;
const BUDGET_BYTES: u64 = 1 << 20;
/// How long a queued session must stay unanswered to show it waits for the running one.
const QUEUE_GRACE: Duration = Duration::from_millis(500);
/// Prefill tokens the leader sends (BOS then "hi" as the fake tokenizer encodes it).
const PREFILL: [u32; 3] = [0, 105, 106];
/// Decode tokens sent one step at a time after the prefill.
const DECODE: [u32; 2] = [73, 102];
/// Layers in the fixture model, which a full-model worker may hand out in any range.
const FIXTURE_LAYERS: u32 = 4;
/// Memory a full-model worker in these tests offers, far below any real machine's free memory.
const OFFER_BYTES: u64 = 12_345;

fn init_tracing() {
    // Another test in this binary may already have installed the subscriber; that is fine.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
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

/// The worker's id plus its bound loopback sockets, so a client dials it directly.
fn direct_addr(endpoint: &Endpoint) -> EndpointAddr {
    endpoint
        .bound_sockets()
        .into_iter()
        .fold(EndpointAddr::new(endpoint.id()), EndpointAddr::with_ip_addr)
}

fn model_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("fake-shard-model.json")
}

fn sha256_hex(path: &PathBuf) -> String {
    let bytes = std::fs::read(path).expect("read file to hash");
    wire::encode_hex(&Sha256::digest(&bytes))
}

fn config(expected_sha256: String) -> ShardServeConfig {
    ShardServeConfig {
        runner: example_bin("fake_shard_runner"),
        model: model_fixture(),
        begin: BEGIN,
        end: END,
        context: CONTEXT,
        gpu_layers: 0,
        budget_bytes: BUDGET_BYTES,
        expected_sha256,
        source_sha256: None,
    }
}

/// Binds the worker's endpoint, before its allowlist is known.
async fn bind_worker() -> Endpoint {
    init_tracing();
    bind_endpoint(vec![RPC_BRIDGE_ALPN.to_vec()], None).await
}

/// A running `serve_shard`.
struct Worker {
    endpoint: Endpoint,
    task: JoinHandle<anyhow::Result<()>>,
}

impl Worker {
    fn serve(endpoint: Endpoint, approved: &[EndpointId]) -> Self {
        let allowed = PeerAllowlist::new(approved.iter().copied()).expect("allowlist");
        let config = config(sha256_hex(&model_fixture()));
        let task = tokio::spawn(serve_shard(endpoint.clone(), config, allowed));
        Self { endpoint, task }
    }

    /// Serves the fixture with balanced-mode `options`.
    fn serve_with(endpoint: Endpoint, approved: &[EndpointId], options: ShardServeOptions) -> Self {
        let allowed = PeerAllowlist::new(approved.iter().copied()).expect("allowlist");
        let config = config(sha256_hex(&model_fixture()));
        let task = tokio::spawn(serve_shard_with(endpoint.clone(), config, options, allowed));
        Self { endpoint, task }
    }

    /// Connects `client` (bound with [`client_for`]) to this worker on the shard ALPN.
    async fn connect(&self, client: &Endpoint) -> Connection {
        let connect = client.connect(self.endpoint.id(), RPC_BRIDGE_ALPN);
        timeout(TEST_TIMEOUT, connect)
            .await
            .expect("connect timed out")
            .expect("connect to worker")
    }

    /// Closing the endpoint must make `serve_shard` return `Ok(())`.
    async fn shut_down(self) {
        self.endpoint.close().await;
        timeout(TEST_TIMEOUT, self.task)
            .await
            .expect("serve_shard stops after endpoint close")
            .expect("serve_shard did not panic")
            .expect("serve_shard returns Ok");
    }
}

/// A leader-side client endpoint that knows how to reach `worker`.
async fn client_for(worker: &Endpoint) -> Endpoint {
    let lookup = MemoryLookup::new();
    lookup.add_endpoint_info(direct_addr(worker));
    bind_endpoint(Vec::new(), Some(lookup)).await
}

/// One shard session seen from the leader.
struct Session {
    send: SendStream,
    recv: BufReader<RecvStream>,
    meta: StageMetadata,
    step: u32,
    position: u32,
}

/// Opens a stream, sends the preface, and returns once the stream is underway (no metadata yet).
async fn open_stream(connection: &Connection) -> (SendStream, BufReader<RecvStream>) {
    let (mut send, recv) = connection.open_bi().await.expect("open shard stream");
    wire::write_preface(&mut send).await.expect("send preface");
    (send, BufReader::new(recv))
}

async fn read_meta(recv: &mut BufReader<RecvStream>) -> StageMetadata {
    timeout(TEST_TIMEOUT, wire::read_metadata(recv))
        .await
        .expect("metadata timed out")
        .expect("read metadata")
}

/// Opens a stream for a [`SessionRequest`] (probe or ranged session).
async fn open_request(
    connection: &Connection,
    request: &SessionRequest,
) -> (SendStream, BufReader<RecvStream>) {
    let (mut send, recv) = connection.open_bi().await.expect("open shard stream");
    wire::write_session_request(&mut send, request)
        .await
        .expect("send session request");
    (send, BufReader::new(recv))
}

/// Opens a session for layers `[begin, end)`.
async fn open_ranged_session(connection: &Connection, begin: u32, end: u32) -> Session {
    let (send, mut recv) = open_request(connection, &SessionRequest::Session { begin, end }).await;
    let meta = read_meta(&mut recv).await;
    Session {
        send,
        recv,
        meta,
        step: 0,
        position: 0,
    }
}

/// Asks for a range the worker must refuse: the stream ends without metadata.
async fn assert_range_refused(connection: &Connection, begin: u32, end: u32) {
    let (_send, mut recv) = open_request(connection, &SessionRequest::Session { begin, end }).await;
    let refused = timeout(TEST_TIMEOUT, wire::read_metadata(&mut recv))
        .await
        .expect("worker ends the refused session");
    assert!(refused.is_err(), "layers {begin}:{end} must be refused");
}

async fn probe(connection: &Connection) -> wire::ProbeReply {
    let (_send, mut recv) = open_request(connection, &SessionRequest::Probe).await;
    let reply = timeout(TEST_TIMEOUT, wire::read_probe_reply(&mut recv))
        .await
        .expect("probe timed out")
        .expect("probe reply");
    let mut rest = Vec::new();
    timeout(TEST_TIMEOUT, recv.read_to_end(&mut rest))
        .await
        .expect("worker finishes the probe stream")
        .expect("clean end of probe stream");
    assert!(
        rest.is_empty(),
        "unexpected bytes after the probe reply {rest:?}"
    );
    reply
}

async fn open_session(connection: &Connection) -> Session {
    let (send, mut recv) = open_stream(connection).await;
    let meta = read_meta(&mut recv).await;
    Session {
        send,
        recv,
        meta,
        step: 0,
        position: 0,
    }
}

impl Session {
    /// Sends one batch of tokens and checks the prefix stage's rows (`t + end` per token).
    async fn run_step(&mut self, tokens: &[u32]) {
        let embd = self.meta.embd as usize;
        let end = self.meta.end;
        let input = StageInput::Tokens(tokens);
        wire::write_request(&mut self.send, self.step, self.position, &input)
            .await
            .expect("send request");
        let output = timeout(
            TEST_TIMEOUT,
            wire::read_step(&mut self.recv, tokens.len() * embd),
        )
        .await
        .expect("step timed out")
        .expect("read step");
        assert_eq!(output.reply.step, self.step);
        assert_eq!(output.reply.token, -1, "a prefix stage does not sample");
        let expected: Vec<f32> = tokens
            .iter()
            .flat_map(|&token| std::iter::repeat_n((token + end) as f32, embd))
            .collect();
        assert_eq!(output.floats, expected);
        self.step += 1;
        self.position += u32::try_from(tokens.len()).expect("small batch");
    }

    /// Prefill plus a couple of decode steps.
    async fn drive(&mut self) {
        self.run_step(&PREFILL).await;
        for token in DECODE {
            self.run_step(&[token]).await;
        }
    }

    /// Finishes sending and waits for the worker to finish its side cleanly.
    async fn close(mut self) {
        self.send.finish().expect("finish shard stream");
        let mut rest = Vec::new();
        timeout(TEST_TIMEOUT, self.recv.read_to_end(&mut rest))
            .await
            .expect("worker finishes the stream")
            .expect("clean end of stream");
        assert!(rest.is_empty(), "unexpected trailing bytes {rest:?}");
    }
}

fn assert_stamped(meta: &StageMetadata, model_sha256: &str) {
    assert_eq!((meta.begin, meta.end), (BEGIN, END));
    assert!(!meta.is_final());
    assert_eq!(meta.model_sha256.as_deref(), Some(model_sha256));
    assert_eq!(meta.artifact_sha256.as_deref(), Some(model_sha256));
    let runner_sha256 = sha256_hex(&example_bin("fake_shard_runner"));
    assert_eq!(meta.runner_sha256.as_deref(), Some(runner_sha256.as_str()));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn approved_leader_runs_sessions_with_a_fresh_runner_each() {
    let endpoint = bind_worker().await;
    let client = client_for(&endpoint).await;
    let worker = Worker::serve(endpoint, &[client.id()]);
    let connection = worker.connect(&client).await;
    let model_sha256 = sha256_hex(&model_fixture());

    let mut first = open_session(&connection).await;
    assert_stamped(&first.meta, &model_sha256);
    first.drive().await;
    first.close().await;

    // A second session on the same connection starts over at step 0, which only a fresh runner
    // accepts (the fake runner enforces step order).
    let mut second = open_session(&connection).await;
    assert_stamped(&second.meta, &model_sha256);
    second.drive().await;
    second.close().await;

    client.close().await;
    worker.shut_down().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn back_to_back_leaders_are_served_one_at_a_time() {
    let endpoint = bind_worker().await;
    let first_client = client_for(&endpoint).await;
    let second_client = client_for(&endpoint).await;
    let worker = Worker::serve(endpoint, &[first_client.id(), second_client.id()]);

    let first_connection = worker.connect(&first_client).await;
    let mut first = open_session(&first_connection).await;
    first.run_step(&PREFILL).await;

    let second_connection = worker.connect(&second_client).await;
    let (second_send, mut second_recv) = open_stream(&second_connection).await;
    let queued = timeout(QUEUE_GRACE, wire::read_metadata(&mut second_recv)).await;
    assert!(
        queued.is_err(),
        "the second session must wait while the first runner is still running"
    );

    first.run_step(&[DECODE[0]]).await;
    first.close().await;
    let mut second = Session {
        meta: read_meta(&mut second_recv).await,
        send: second_send,
        recv: second_recv,
        step: 0,
        position: 0,
    };
    second.drive().await;
    second.close().await;

    first_client.close().await;
    second_client.close().await;
    worker.shut_down().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unapproved_peer_is_refused() {
    let endpoint = bind_worker().await;
    let stranger = client_for(&endpoint).await;
    let someone_else = iroh::SecretKey::generate().public();
    let worker = Worker::serve(endpoint, &[someone_else]);
    // The handshake itself succeeds; authorization happens after it.
    let connection = worker.connect(&stranger).await;
    if let Ok((mut send, _recv)) = connection.open_bi().await {
        // The connection may already be closed, in which case the write fails, as it should.
        let _ = send.write_all(wire::CLIENT_PREFACE).await;
    }
    let closed = timeout(TEST_TIMEOUT, connection.closed())
        .await
        .expect("worker closes the connection");

    let ConnectionError::ApplicationClosed(close) = closed else {
        panic!("expected an application close, got {closed:?}");
    };
    assert_eq!(close.error_code, PEER_NOT_APPROVED);
    assert_eq!(&close.reason[..], PEER_NOT_APPROVED_REASON);

    stranger.close().await;
    worker.shut_down().await;
}

#[tokio::test]
async fn wrong_expected_sha256_fails_at_startup() {
    let endpoint = bind_worker().await;
    let allowed = PeerAllowlist::new([endpoint.id()]).expect("allowlist");
    let result = timeout(
        TEST_TIMEOUT,
        serve_shard(endpoint.clone(), config("0".repeat(64)), allowed),
    )
    .await
    .expect("startup check finishes");
    let error = result.expect_err("a mismatched model must be refused");
    assert!(error.to_string().contains("checksum mismatch"), "{error:#}");
    endpoint.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn garbage_preface_ends_only_that_session() {
    let endpoint = bind_worker().await;
    let client = client_for(&endpoint).await;
    let worker = Worker::serve(endpoint, &[client.id()]);
    let connection = worker.connect(&client).await;

    let (mut send, recv) = connection.open_bi().await.expect("open stream");
    send.write_all(b"HELLO")
        .await
        .expect("send garbage preface");
    let mut recv = BufReader::new(recv);
    let refused = timeout(TEST_TIMEOUT, wire::read_metadata(&mut recv))
        .await
        .expect("worker ends the session");
    assert!(refused.is_err(), "a bad preface must not start a runner");

    let mut next = open_session(&connection).await;
    next.drive().await;
    next.close().await;

    client.close().await;
    worker.shut_down().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_full_model_worker_answers_probes_and_serves_any_requested_range() {
    let endpoint = bind_worker().await;
    let client = client_for(&endpoint).await;
    let options = ShardServeOptions {
        layers: Some(FIXTURE_LAYERS),
        offer_bytes: Some(OFFER_BYTES),
    };
    let worker = Worker::serve_with(endpoint, &[client.id()], options);
    let connection = worker.connect(&client).await;
    let model_sha256 = sha256_hex(&model_fixture());

    let reply = probe(&connection).await;
    assert_eq!(reply.available_bytes, OFFER_BYTES, "capped by the offer");
    assert_eq!(reply.model_sha256, model_sha256);
    assert_eq!(reply.layers, Some(FIXTURE_LAYERS));
    assert_eq!((reply.begin, reply.end), (BEGIN, END));

    // Each session gets its own range from the same full model file.
    for end in [3, 1] {
        let mut session = open_ranged_session(&connection, 0, end).await;
        assert_eq!((session.meta.begin, session.meta.end), (0, end));
        assert_eq!(
            session.meta.model_sha256.as_deref(),
            Some(model_sha256.as_str())
        );
        session.drive().await;
        session.close().await;
    }
    // A probe between sessions still works, and needs no runner.
    assert_eq!(probe(&connection).await.layers, Some(FIXTURE_LAYERS));

    assert_range_refused(&connection, 0, FIXTURE_LAYERS + 1).await;
    assert_range_refused(&connection, 2, 2).await;
    // The original preface still gets the configured range.
    let mut legacy = open_session(&connection).await;
    assert_stamped(&legacy.meta, &model_sha256);
    legacy.drive().await;
    legacy.close().await;

    client.close().await;
    worker.shut_down().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_fixed_range_worker_refuses_other_ranges() {
    let endpoint = bind_worker().await;
    let client = client_for(&endpoint).await;
    let worker = Worker::serve(endpoint, &[client.id()]);
    let connection = worker.connect(&client).await;

    let reply = probe(&connection).await;
    assert_eq!(
        reply.layers, None,
        "a fixed-range worker cannot be balanced"
    );
    assert!(reply.available_bytes > 0, "free memory is measured");
    assert_range_refused(&connection, 0, 3).await;
    // Asking for exactly its range is fine.
    let mut session = open_ranged_session(&connection, BEGIN, END).await;
    assert_stamped(&session.meta, &sha256_hex(&model_fixture()));
    session.drive().await;
    session.close().await;

    client.close().await;
    worker.shut_down().await;
}

#[tokio::test]
async fn a_full_model_worker_refuses_a_configured_range_past_the_model() {
    let endpoint = bind_worker().await;
    let allowed = PeerAllowlist::new([endpoint.id()]).expect("allowlist");
    let mut config = config(sha256_hex(&model_fixture()));
    config.end = FIXTURE_LAYERS + 1;
    let options = ShardServeOptions {
        layers: Some(FIXTURE_LAYERS),
        offer_bytes: None,
    };
    let result = timeout(
        TEST_TIMEOUT,
        serve_shard_with(endpoint.clone(), config, options, allowed),
    )
    .await
    .expect("startup check finishes");
    let error = result.expect_err("a range past the model must be refused");
    assert!(error.to_string().contains("does not fit"), "{error:#}");
    endpoint.close().await;
}
