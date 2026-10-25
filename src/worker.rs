//! Worker process: supervised ggml-rpc-server, RPC bridge, and heartbeats to the leader.
//!
//! A worker binds one Iroh endpoint (one identity per process), keeps llama.cpp's
//! `ggml-rpc-server` running on a loopback port, and exposes it to the leader through
//! [`rpc_bridge::serve_worker`]. When a leader is configured it also holds a control link: it
//! dials the leader, exchanges hellos, and writes [`Heartbeat`] frames on one unidirectional
//! stream, redialling with backoff whenever the link drops. Heartbeat sequence numbers keep
//! counting across reconnects, so every heartbeat this process sends has a larger sequence than
//! the one before it.
//!
//! The worker also watches the leader's own heartbeats on the link (see
//! [`network::watch_heartbeats`]). Once the leader has sent one, a gap longer than
//! [`WorkerConfig::leader_heartbeat_timeout`] makes the worker close the link with
//! [`network::HEARTBEAT_TIMEOUT_CODE`] and redial, instead of waiting for QUIC's idle timeout to
//! notice a crashed or suspended leader. A leader that never heartbeats is never timed out.

use std::convert::Infallible;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, anyhow};
use iroh::endpoint::ConnectionError;
use iroh::{Endpoint, EndpointAddr, EndpointId};
use tracing::{info, warn};

use crate::network::{self, HEARTBEAT_EPOCH, SharedPeerTable};
use crate::node::PeerTable;
use crate::process::{self, ChildSpec, Readiness, RestartPolicy};
use crate::protocol::{CONTROL_ALPN, Heartbeat, Hello, RPC_BRIDGE_ALPN};
use crate::rpc_bridge;

/// How often a worker sends a heartbeat to its leader unless configured otherwise.
pub const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);

/// How long a worker waits for the leader's next heartbeat before dropping the link and
/// redialling, unless configured otherwise: three missed beats at the leader's 2s default
/// (`leader::DEFAULT_HEARTBEAT_INTERVAL`) plus a second of slack.
pub const DEFAULT_LEADER_HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(7);

/// Log label for the supervised RPC server.
const RPC_SERVER_NAME: &str = "ggml-rpc-server";

/// Longest the RPC server may take to accept TCP connections after each (re)start.
const RPC_READY_TIMEOUT: Duration = Duration::from_secs(30);

/// Workers keep no replicated log yet, so their hello always reports an empty one.
const LAST_LOG_INDEX: u64 = 0;

/// Longest one attempt to dial the leader and finish the hello may take.
const LEADER_DIAL_TIMEOUT: Duration = Duration::from_secs(30);

/// First pause before redialling a leader whose link dropped.
const RECONNECT_INITIAL_BACKOFF: Duration = Duration::from_millis(500);

/// Longest pause between redial attempts.
const RECONNECT_MAX_BACKOFF: Duration = Duration::from_secs(10);

/// A hello alone does not reset backoff; require several heartbeat periods of stability.
const BACKOFF_RESET_AFTER_HEARTBEATS: u32 = 5;
const BACKOFF_RESET_MIN_DURATION: Duration = Duration::from_secs(1);

/// Everything `llm-cluster worker` needs to run.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Secret key file holding this node's identity; created if absent.
    pub key_path: PathBuf,
    /// Cluster this worker belongs to; the leader rejects hellos from other clusters.
    pub cluster_id: String,
    /// The `ggml-rpc-server` binary to supervise.
    pub rpc_server: PathBuf,
    /// Loopback port the RPC server listens on, usually [`rpc_bridge::DEFAULT_RPC_PORT`].
    pub rpc_port: u16,
    /// Extra RPC server arguments, appended after `-H 127.0.0.1 -p <rpc_port>`.
    pub rpc_args: Vec<String>,
    /// Leader to keep a control link to; `None` runs the bridge only.
    pub leader: Option<EndpointId>,
    /// Peers besides the leader allowed to use this worker's RPC bridge (`--allow-peer`).
    pub allowed_peers: Vec<EndpointId>,
    /// Time between heartbeats on the control link.
    pub heartbeat_interval: Duration,
    /// How long the worker waits for the leader's next heartbeat before closing the control link
    /// with [`network::HEARTBEAT_TIMEOUT_CODE`] and redialling, usually
    /// [`DEFAULT_LEADER_HEARTBEAT_TIMEOUT`]. It arms only once the leader has sent a heartbeat on
    /// the current link, so a leader that never heartbeats is never timed out. Must exceed the
    /// leader's heartbeat interval and must not be zero; `None` disables it, leaving a lost
    /// leader to QUIC's idle timeout.
    pub leader_heartbeat_timeout: Option<Duration>,
}

/// Runs a worker until the RPC server fails for good or the future is dropped.
///
/// Binds one endpoint accepting [`CONTROL_ALPN`] and [`RPC_BRIDGE_ALPN`], prints `node id: <hex>`,
/// then behaves as [`run_worker_on`], dialling the leader by node ID alone.
pub async fn run_worker(config: WorkerConfig) -> anyhow::Result<()> {
    let endpoint = network::bind_endpoint(&config.key_path, worker_alpns()).await?;
    println!("node id: {}", endpoint.id());
    let leader = config.leader.map(EndpointAddr::from);
    run_worker_on(endpoint, leader, &config).await
}

/// Runs a worker on an endpoint the caller has already bound, dialling `leader` for the control link.
///
/// `config.key_path` and `config.leader` are ignored: the endpoint carries the identity, and
/// `leader` may carry direct addresses (as tests on isolated endpoints need). Supervises the RPC
/// server on `127.0.0.1:<rpc_port>`, bridges [`RPC_BRIDGE_ALPN`] streams to it, and, if `leader`
/// is set, sends heartbeats. Only `leader` and `config.allowed_peers` may use the bridge; with
/// neither, this fails before starting anything. Returns an error once the RPC server's restarts
/// are exhausted. Dropping the future kills the RPC server and closes the endpoint.
pub async fn run_worker_on(
    endpoint: Endpoint,
    leader: Option<EndpointAddr>,
    config: &WorkerConfig,
) -> anyhow::Result<()> {
    let _close_on_drop = CloseOnDrop(endpoint.clone());
    let outcome = serve(&endpoint, leader, config).await;
    endpoint.close().await;
    outcome
}

/// ALPNs a worker endpoint accepts.
fn worker_alpns() -> Vec<Vec<u8>> {
    vec![CONTROL_ALPN.to_vec(), RPC_BRIDGE_ALPN.to_vec()]
}

/// Starts the RPC server, then runs the bridge and control link until one of them ends.
async fn serve(
    endpoint: &Endpoint,
    leader: Option<EndpointAddr>,
    config: &WorkerConfig,
) -> anyhow::Result<()> {
    let leader_id = leader.as_ref().map(|addr| addr.id);
    let allowed = bridge_allowlist(config, leader_id)?;
    let rpc_addr = SocketAddr::from((Ipv4Addr::LOCALHOST, config.rpc_port));
    let mut rpc_server = process::spawn_supervised(rpc_server_spec(config, rpc_addr))
        .await
        .with_context(|| format!("failed to start {RPC_SERVER_NAME}"))?;
    info!(%rpc_addr, "{RPC_SERVER_NAME} ready");

    let outcome = tokio::select! {
        exited = rpc_server.wait() => match exited {
            Ok(()) => Err(anyhow!("{RPC_SERVER_NAME} stopped")),
            Err(error) => Err(error.context(format!("{RPC_SERVER_NAME} exhausted its restarts"))),
        },
        served = rpc_bridge::serve_worker(endpoint.clone(), rpc_addr, allowed) => served,
        () = keep_control_link(endpoint, leader, config) => Ok(()),
    };
    if let Err(error) = rpc_server.shutdown().await {
        warn!(error = %format!("{error:#}"), "failed to stop {RPC_SERVER_NAME} cleanly");
    }
    outcome
}

/// Peers allowed to use the RPC bridge: the configured extras plus the leader this worker serves.
fn bridge_allowlist(
    config: &WorkerConfig,
    leader: Option<EndpointId>,
) -> anyhow::Result<rpc_bridge::PeerAllowlist> {
    let peers = config.allowed_peers.iter().copied().chain(leader);
    rpc_bridge::PeerAllowlist::new(peers)
        .context("worker has no approved peers: pass --leader or --allow-peer")
}

/// Command line and supervision policy for the RPC server; it never listens beyond loopback.
fn rpc_server_spec(config: &WorkerConfig, rpc_addr: SocketAddr) -> ChildSpec {
    let mut args = vec![
        "-H".to_owned(),
        rpc_addr.ip().to_string(),
        "-p".to_owned(),
        rpc_addr.port().to_string(),
    ];
    args.extend(config.rpc_args.iter().cloned());
    ChildSpec {
        name: RPC_SERVER_NAME.to_owned(),
        program: config.rpc_server.clone(),
        args,
        readiness: Readiness::TcpPort(rpc_addr),
        ready_timeout: RPC_READY_TIMEOUT,
        restart: RestartPolicy::default(),
    }
}

/// Keeps the control link to `leader` up until the endpoint closes; never resolves without a leader.
async fn keep_control_link(
    endpoint: &Endpoint,
    leader: Option<EndpointAddr>,
    config: &WorkerConfig,
) {
    let Some(leader) = leader else {
        return std::future::pending().await;
    };
    let mut link = ControlLink {
        endpoint,
        leader,
        local: network::local_hello(endpoint, &config.cluster_id, LAST_LOG_INDEX),
        peers: Arc::new(Mutex::new(PeerTable::new())),
        heartbeats: HeartbeatSequence::new(endpoint.id().to_string()),
        backoff: ReconnectBackoff::default(),
        interval: config.heartbeat_interval,
        leader_heartbeat_timeout: config.leader_heartbeat_timeout,
    };
    link.run().await;
}

/// State of the worker's control link that survives reconnects.
struct ControlLink<'a> {
    endpoint: &'a Endpoint,
    leader: EndpointAddr,
    local: Hello,
    /// Holds the leader while a link is up, with the last heartbeat the leader sent on it.
    peers: SharedPeerTable,
    heartbeats: HeartbeatSequence,
    backoff: ReconnectBackoff,
    interval: Duration,
    leader_heartbeat_timeout: Option<Duration>,
}

impl ControlLink<'_> {
    /// Runs sessions back to back, pausing with backoff between them, until the endpoint closes.
    async fn run(&mut self) {
        let leader = self.leader.id.fmt_short();
        loop {
            let Err(error) = self.session().await;
            if self.endpoint.is_closed() {
                return;
            }
            let delay = self.backoff.next_delay();
            warn!(
                %leader,
                error = %format!("{error:#}"),
                retry_in_ms = delay.as_millis(),
                "control link to leader lost"
            );
            tokio::time::sleep(delay).await;
        }
    }

    /// Dials the leader, then sends heartbeats on one stream while watching the leader's, until
    /// the link fails.
    async fn session(&mut self) -> anyhow::Result<Infallible> {
        let dial = network::dial_peer(self.endpoint, self.leader.clone(), &self.local, &self.peers);
        let session = tokio::time::timeout(LEADER_DIAL_TIMEOUT, dial)
            .await
            .context("timed out dialling the leader")??;
        info!(leader = %session.hello().node_id, "control link to leader established");
        let connected_at = tokio::time::Instant::now();

        let watch = network::watch_heartbeats(&session, &self.peers, self.leader_heartbeat_timeout);
        // Both run for the whole session: the send loop only ends on a failed write, and the
        // watch only when the connection closes, whether the leader closed it or the watch did.
        let heartbeats = &mut self.heartbeats;
        let send = network::write_heartbeats(session.connection(), self.interval, || {
            heartbeats.next_heartbeat()
        });
        let outcome = tokio::select! {
            sent = send => sent,
            reason = watch => Err(link_lost(reason)),
        };
        self.backoff
            .reset_after_session(connected_at.elapsed(), self.interval);
        outcome
    }
}

/// Explains why [`network::watch_heartbeats`] saw the control link close.
///
/// Only the watch closes the connection locally during a session (closing the endpoint ends
/// the worker instead of redialling), so a local close means the leader went quiet.
fn link_lost(reason: ConnectionError) -> anyhow::Error {
    match reason {
        ConnectionError::LocallyClosed => anyhow!("leader stopped heartbeating"),
        other => anyhow!("leader closed the control link: {other}"),
    }
}

/// Produces this node's heartbeats with strictly increasing sequence numbers, starting at 1.
struct HeartbeatSequence {
    node_id: String,
    last: u64,
}

impl HeartbeatSequence {
    fn new(node_id: String) -> Self {
        Self { node_id, last: 0 }
    }

    fn next_heartbeat(&mut self) -> Heartbeat {
        self.last += 1;
        Heartbeat {
            node_id: self.node_id.clone(),
            epoch: HEARTBEAT_EPOCH,
            sequence: self.last,
        }
    }
}

/// Delay between redials of the leader: exponential per consecutive failure, restarting after a
/// control session remains up for several heartbeat periods.
#[derive(Default)]
struct ReconnectBackoff {
    consecutive_failures: u32,
}

impl ReconnectBackoff {
    /// The worker redials for as long as it runs, so only the delays of this policy are used.
    const POLICY: RestartPolicy = RestartPolicy {
        max_restarts: u32::MAX,
        initial_backoff: RECONNECT_INITIAL_BACKOFF,
        max_backoff: RECONNECT_MAX_BACKOFF,
    };

    fn next_delay(&mut self) -> Duration {
        let delay = Self::POLICY.backoff(self.consecutive_failures);
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        delay
    }

    fn reset(&mut self) {
        self.consecutive_failures = 0;
    }

    fn reset_after_session(&mut self, connected_for: Duration, interval: Duration) {
        let stable_for = interval
            .saturating_mul(BACKOFF_RESET_AFTER_HEARTBEATS)
            .max(BACKOFF_RESET_MIN_DURATION);
        if connected_for >= stable_for {
            self.reset();
        }
    }
}

/// Closes the endpoint in the background if the worker future is dropped before it could.
struct CloseOnDrop(Endpoint);

impl Drop for CloseOnDrop {
    fn drop(&mut self) {
        if self.0.is_closed() {
            return;
        }
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            let endpoint = self.0.clone();
            runtime.spawn(async move { endpoint.close().await });
        }
    }
}

#[cfg(test)]
mod tests {
    use iroh::SecretKey;

    use super::*;

    fn config() -> WorkerConfig {
        WorkerConfig {
            key_path: PathBuf::from("worker.key"),
            cluster_id: "test".to_owned(),
            rpc_server: PathBuf::from("ggml-rpc-server"),
            rpc_port: rpc_bridge::DEFAULT_RPC_PORT,
            rpc_args: vec!["--threads".to_owned(), "4".to_owned()],
            leader: None,
            allowed_peers: Vec::new(),
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            leader_heartbeat_timeout: Some(DEFAULT_LEADER_HEARTBEAT_TIMEOUT),
        }
    }

    #[test]
    fn a_locally_closed_link_means_the_leader_went_quiet() {
        let quiet = link_lost(ConnectionError::LocallyClosed);
        let timed_out = link_lost(ConnectionError::TimedOut);

        assert_eq!(quiet.to_string(), "leader stopped heartbeating");
        assert!(
            timed_out
                .to_string()
                .starts_with("leader closed the control link"),
            "{timed_out}"
        );
    }

    #[test]
    fn bridge_allowlist_approves_the_leader_and_extra_peers() {
        let leader = SecretKey::generate().public();
        let extra = SecretKey::generate().public();
        let stranger = SecretKey::generate().public();
        let mut config = config();
        config.allowed_peers = vec![extra];

        let allowed = bridge_allowlist(&config, Some(leader)).expect("allowlist");

        assert!(allowed.contains(&leader));
        assert!(allowed.contains(&extra));
        assert!(!allowed.contains(&stranger));
    }

    #[test]
    fn bridge_allowlist_fails_closed_without_leader_or_peers() {
        let error = bridge_allowlist(&config(), None).expect_err("no approved peers");
        assert!(format!("{error:#}").contains("--allow-peer"), "{error:#}");
    }

    #[test]
    fn rpc_server_listens_on_loopback_with_extra_args_last() {
        let rpc_addr = SocketAddr::from((Ipv4Addr::LOCALHOST, 50_100));
        let spec = rpc_server_spec(&config(), rpc_addr);

        assert_eq!(
            spec.args,
            ["-H", "127.0.0.1", "-p", "50100", "--threads", "4"]
        );
        assert_eq!(spec.program, PathBuf::from("ggml-rpc-server"));
        assert!(matches!(spec.readiness, Readiness::TcpPort(addr) if addr == rpc_addr));
    }

    #[test]
    fn worker_accepts_control_and_bridge_alpns() {
        assert_eq!(
            worker_alpns(),
            [CONTROL_ALPN.to_vec(), RPC_BRIDGE_ALPN.to_vec()]
        );
    }

    #[test]
    fn heartbeats_count_up_from_one_in_epoch_zero() {
        let node_id = SecretKey::generate().public().to_string();
        let mut sequence = HeartbeatSequence::new(node_id.clone());

        let sent: Vec<Heartbeat> = (0..3).map(|_| sequence.next_heartbeat()).collect();

        assert_eq!(
            sent.iter().map(|beat| beat.sequence).collect::<Vec<_>>(),
            [1, 2, 3]
        );
        assert!(
            sent.iter()
                .all(|beat| beat.epoch == 0 && beat.node_id == node_id)
        );
    }

    #[test]
    fn reconnect_backoff_doubles_up_to_the_cap_and_resets() {
        let mut backoff = ReconnectBackoff::default();

        let delays: Vec<Duration> = (0..7).map(|_| backoff.next_delay()).collect();
        assert_eq!(delays[0], RECONNECT_INITIAL_BACKOFF);
        assert_eq!(delays[1], RECONNECT_INITIAL_BACKOFF * 2);
        assert!(delays.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(delays[6], RECONNECT_MAX_BACKOFF);

        backoff.reset();
        assert_eq!(backoff.next_delay(), RECONNECT_INITIAL_BACKOFF);
    }

    #[test]
    fn short_control_sessions_do_not_reset_reconnect_backoff() {
        let mut backoff = ReconnectBackoff::default();
        assert_eq!(backoff.next_delay(), RECONNECT_INITIAL_BACKOFF);
        assert_eq!(backoff.next_delay(), RECONNECT_INITIAL_BACKOFF * 2);

        backoff.reset_after_session(Duration::from_secs(9), Duration::from_secs(2));
        assert_eq!(backoff.next_delay(), RECONNECT_INITIAL_BACKOFF * 4);

        backoff.reset_after_session(Duration::from_secs(10), Duration::from_secs(2));
        assert_eq!(backoff.next_delay(), RECONNECT_INITIAL_BACKOFF);
    }
}
