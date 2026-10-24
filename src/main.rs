//! MeshThing command-line entry point: start a node, dial peers, or run the RPC bridge.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use clap::{Parser, Subcommand};
use iroh::Endpoint;
use llm_cluster::network::{self, SharedPeerTable};
use llm_cluster::node::PeerTable;
use llm_cluster::protocol::{CONTROL_ALPN, RPC_BRIDGE_ALPN};
use llm_cluster::rpc_bridge;
use tracing_subscriber::EnvFilter;

/// Log filter used when `RUST_LOG` is unset.
const DEFAULT_LOG_FILTER: &str = "info";

/// Last log index announced in hellos until the replicated log exists.
const EMPTY_LOG_INDEX: u64 = 0;

/// QUIC application close code for a normal, intentional shutdown.
const CLOSE_CODE_DONE: u32 = 0;

/// QUIC close reason sent alongside [`CLOSE_CODE_DONE`].
const CLOSE_REASON_DONE: &[u8] = b"done";

#[derive(Debug, Parser)]
#[command(
    name = "meshthing",
    about = "Peer-to-peer mesh for splitting one LLM across devices"
)]
struct Cli {
    /// File holding this node's secret key; created on first run.
    #[arg(long, global = true, default_value = "meshthing.key")]
    key: PathBuf,

    /// Cluster ID announced in hellos; peers from other clusters are rejected.
    #[arg(long, global = true, default_value = "meshthing")]
    cluster: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Check caller-collected memory and compute evidence for an advisory plan.
    Verify {
        /// JSON produced by the plan command.
        plan: PathBuf,
        /// Original planning profile JSON, containing current peer profiles.
        profile: PathBuf,
        /// JSON RunEvidence collected over model load and fresh inference.
        evidence: PathBuf,
        #[arg(long, default_value_t = 2)]
        min_peers: usize,
    },
    /// Read measured profiles from JSON and print an advisory plan; never launches inference.
    Plan {
        /// JSON file containing model, peers, links, devices, and model_path.
        profile: PathBuf,
        /// Require at least this many contributors.
        #[arg(long, default_value_t = 2)]
        min_peers: usize,
    },
    /// Print this node's ID and accept control connections from peers.
    Listen,
    /// Connect to a listening node, exchange hellos, and print the peer's hello.
    Connect {
        /// Node ID printed by the peer's `listen`.
        node_id: String,
    },
    /// Worker side of the RPC bridge: forward incoming bridge streams to a local RPC server.
    BridgeServe {
        /// Local RPC server to forward bridge streams to.
        #[arg(long, default_value = "127.0.0.1:50052")]
        target: SocketAddr,
    },
    /// Leader side of the RPC bridge: expose a remote worker's RPC server on a local port.
    BridgeConnect {
        /// Node ID printed by the worker's `bridge-serve`.
        node_id: String,
        /// Local address to accept RPC clients on.
        #[arg(long, default_value = "127.0.0.1:50052")]
        listen: SocketAddr,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logging();
    let cli = Cli::parse();
    match cli.command {
        Command::Verify {
            plan,
            profile,
            evidence,
            min_peers,
        } => {
            let output: llm_cluster::planning::PlanningOutput =
                serde_json::from_str(&std::fs::read_to_string(plan)?)?;
            let request: llm_cluster::planning::PlanningRequest =
                serde_json::from_str(&std::fs::read_to_string(profile)?)?;
            let evidence = serde_json::from_str(&std::fs::read_to_string(evidence)?)?;
            llm_cluster::planning::verify_output(&request, &output, &evidence, min_peers)
                .map_err(|error| anyhow::anyhow!("verification failed: {error:?}"))?;
            println!("{{\"verified\":true}}");
            Ok(())
        }
        Command::Plan { profile, min_peers } => {
            let input = std::fs::read_to_string(profile)?;
            let request = serde_json::from_str(&input)?;
            let output = llm_cluster::planning::build_plan(&request, min_peers)
                .map_err(|error| anyhow::anyhow!("planning failed: {error:?}"))?;
            println!("{}", serde_json::to_string_pretty(&output)?);
            Ok(())
        }
        Command::Listen => listen(&cli.key, &cli.cluster).await,
        Command::Connect { ref node_id } => connect(&cli.key, &cli.cluster, node_id).await,
        Command::BridgeServe { target } => bridge_serve(&cli.key, target).await,
        Command::BridgeConnect {
            ref node_id,
            listen,
        } => bridge_connect(&cli.key, node_id, listen).await,
    }
}

/// Sends logs to stderr so stdout carries only intentional CLI output such as the node ID.
fn init_logging() {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG_FILTER));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

async fn listen(key: &Path, cluster: &str) -> anyhow::Result<()> {
    let endpoint = network::bind_endpoint(key, vec![CONTROL_ALPN.to_vec()]).await?;
    print_node_id(&endpoint);
    let local = network::local_hello(&endpoint, cluster, EMPTY_LOG_INDEX);
    let serve = network::serve_control(endpoint.clone(), local, new_peer_table());
    run_until_ctrl_c(&endpoint, serve).await
}

async fn connect(key: &Path, cluster: &str, node_id: &str) -> anyhow::Result<()> {
    let remote = network::parse_node_id(node_id)?;
    let endpoint = network::bind_endpoint(key, vec![CONTROL_ALPN.to_vec()]).await?;
    let local = network::local_hello(&endpoint, cluster, EMPTY_LOG_INDEX);
    let (connection, hello) =
        network::dial_peer(&endpoint, remote, &local, &new_peer_table()).await?;
    println!("peer hello: {hello:?}");
    connection.close(CLOSE_CODE_DONE.into(), CLOSE_REASON_DONE);
    endpoint.close().await;
    Ok(())
}

async fn bridge_serve(key: &Path, target: SocketAddr) -> anyhow::Result<()> {
    let endpoint = network::bind_endpoint(key, vec![RPC_BRIDGE_ALPN.to_vec()]).await?;
    print_node_id(&endpoint);
    let serve = rpc_bridge::serve_worker(endpoint.clone(), target);
    run_until_ctrl_c(&endpoint, serve).await
}

async fn bridge_connect(key: &Path, node_id: &str, listen: SocketAddr) -> anyhow::Result<()> {
    let worker = network::parse_node_id(node_id)?;
    let endpoint = network::bind_endpoint(key, vec![RPC_BRIDGE_ALPN.to_vec()]).await?;
    let lead = rpc_bridge::run_leader(endpoint.clone(), worker, listen);
    run_until_ctrl_c(&endpoint, lead).await
}

/// Runs `task` until it finishes or Ctrl-C arrives, then closes `endpoint` so peers see a clean shutdown.
async fn run_until_ctrl_c(
    endpoint: &Endpoint,
    task: impl Future<Output = anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let result = tokio::select! {
        result = task => result,
        signal = tokio::signal::ctrl_c() => signal.map_err(anyhow::Error::from),
    };
    endpoint.close().await;
    result
}

/// Prints the node ID in the one format peers and bots copy into `connect` / `bridge-connect`.
fn print_node_id(endpoint: &Endpoint) {
    println!("node id: {}", endpoint.id());
}

fn new_peer_table() -> SharedPeerTable {
    Arc::new(Mutex::new(PeerTable::new()))
}
