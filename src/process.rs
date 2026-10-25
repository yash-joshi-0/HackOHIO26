//! Cross-platform child-process supervision: spawn, readiness, restart, shutdown.
//!
//! [`spawn_supervised`] starts a program from an argv (never through a shell), forwards its
//! stdout and stderr to `tracing` line by line, and waits until it is ready to serve. A
//! background supervisor task then restarts the child with exponential backoff whenever it
//! exits, until the [`RestartPolicy`] budget runs out. Only portable process APIs are used
//! (`kill`, never Unix signals), so this behaves the same on Windows, macOS and Linux.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use anyhow::{Context, anyhow, bail};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::process::Command;
use tokio::sync::{oneshot, watch};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::admission::AdmittedChild as Child;

/// `tracing` target for lines a child writes to stdout or stderr.
const CHILD_LOG_TARGET: &str = "child";

/// Pause between readiness probes while a child starts up.
const READY_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Upper bound on one readiness probe, so a stalled HTTP exchange cannot hang the poll loop.
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// Upper bound on one probe's TCP connect. Windows retries a refused loopback connect for about
/// two seconds instead of failing at once, so without this a probe that starts just before the
/// child binds would notice readiness late.
const CONNECT_TIMEOUT: Duration = Duration::from_millis(200);

/// Longest HTTP status line read during a [`Readiness::HttpOk`] probe.
const MAX_STATUS_LINE_LEN: u64 = 1024;

const DEFAULT_MAX_RESTARTS: u32 = 3;
const DEFAULT_INITIAL_BACKOFF: Duration = Duration::from_millis(500);
const DEFAULT_MAX_BACKOFF: Duration = Duration::from_secs(8);

/// Stored in place of a PID while no child is running (no user process ever has PID 0).
const NO_PID: u32 = 0;

/// How the supervisor decides that a freshly started child is ready to serve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Readiness {
    /// Ready once a TCP connect to this address succeeds.
    TcpPort(SocketAddr),
    /// Ready once `GET http://{addr}{path}` returns HTTP 200 (plain HTTP/1.1 over a TcpStream).
    HttpOk {
        /// Address the child's HTTP server listens on.
        addr: SocketAddr,
        /// Request path, starting with `/` (e.g. `/health`).
        path: String,
    },
}

/// How often, and how patiently, a child that exits is started again.
///
/// The restart budget covers the whole life of a [`Supervised`] child: once `max_restarts`
/// restarts have been used, the next exit is final.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RestartPolicy {
    /// Restarts allowed before giving up; 0 disables restarting.
    pub max_restarts: u32,
    /// Delay before the first restart.
    pub initial_backoff: Duration,
    /// Cap on the delay, which doubles with every restart.
    pub max_backoff: Duration,
}

impl Default for RestartPolicy {
    /// Three restarts, waiting 500 ms, 1 s, then 2 s (never more than 8 s).
    fn default() -> Self {
        Self {
            max_restarts: DEFAULT_MAX_RESTARTS,
            initial_backoff: DEFAULT_INITIAL_BACKOFF,
            max_backoff: DEFAULT_MAX_BACKOFF,
        }
    }
}

impl RestartPolicy {
    /// Delay before retry number `retry` (0-based): `initial_backoff` doubled once per earlier
    /// retry, capped at `max_backoff`. Also usable for any other retry loop, such as reconnects.
    pub fn backoff(&self, retry: u32) -> Duration {
        let factor = 2u32.saturating_pow(retry);
        self.initial_backoff
            .saturating_mul(factor)
            .min(self.max_backoff)
    }
}

/// Everything needed to start, probe and restart one child process.
#[derive(Clone, Debug)]
pub struct ChildSpec {
    /// Label used in logs and errors, e.g. `"ggml-rpc-server"`.
    pub name: String,
    /// Executable to run.
    pub program: PathBuf,
    /// Arguments passed as argv, never through a shell.
    pub args: Vec<String>,
    /// How to tell the child is ready to serve.
    pub readiness: Readiness,
    /// How long each start may take to become ready before it is killed and counted as failed.
    pub ready_timeout: Duration,
    /// What to do when the child exits.
    pub restart: RestartPolicy,
}

/// A running, supervised child process.
///
/// Prefer [`Supervised::shutdown`]; dropping the handle also kills the child, but without
/// waiting for it to exit.
pub struct Supervised {
    pid: Arc<AtomicU32>,
    outcome: watch::Receiver<Option<String>>,
    stop: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<anyhow::Result<()>>>,
}

/// Spawns the child, forwards its stdout/stderr to tracing (target `"child"`, field `name`),
/// and waits for readiness.
///
/// Errors if the program cannot be started, or if the child exits or isn't ready within
/// `ready_timeout` (it is killed first). The child is `kill_on_drop`.
pub async fn spawn_supervised(spec: ChildSpec) -> anyhow::Result<Supervised> {
    let pid = Arc::new(AtomicU32::new(NO_PID));
    let (stop_tx, stop_rx) = oneshot::channel();
    let (outcome_tx, outcome_rx) = watch::channel(None);
    let mut supervisor = Supervisor {
        spec,
        pid: Arc::clone(&pid),
        stop: stop_rx,
        outcome: outcome_tx,
    };
    let Some(child) = supervisor.start_once().await? else {
        bail!("child start was cancelled");
    };
    Ok(Supervised {
        pid,
        outcome: outcome_rx,
        stop: Some(stop_tx),
        task: Some(tokio::spawn(supervisor.run(child))),
    })
}

impl Supervised {
    /// Resolves only when the child has exited and restarts are exhausted, with an error
    /// describing the last failure. Restarts re-run readiness before counting as healthy.
    /// Cancel-safe, so it can be raced against a shutdown signal.
    pub async fn wait(&mut self) -> anyhow::Result<()> {
        match self.outcome.wait_for(Option::is_some).await {
            Ok(outcome) => Err(anyhow!("{}", outcome.as_deref().unwrap_or_default())),
            Err(_) => bail!("process supervisor stopped unexpectedly"),
        }
    }

    /// Stops the child (kill, then wait so no orphan is left) and the supervisor.
    /// Succeeds even if the child already exited for good.
    pub async fn shutdown(mut self) -> anyhow::Result<()> {
        if let Some(stop) = self.stop.take() {
            // Sending fails only once the supervisor has ended, and then no child is left.
            let _ = stop.send(());
        }
        let Some(task) = self.task.take() else {
            return Ok(());
        };
        task.await.context("process supervisor task failed")?
    }

    /// Current child PID, or `None` between a crash and its restart.
    pub fn id(&self) -> Option<u32> {
        match self.pid.load(Ordering::Acquire) {
            NO_PID => None,
            pid => Some(pid),
        }
    }
}

impl Drop for Supervised {
    fn drop(&mut self) {
        // Aborting drops the supervisor's `Child`, and `kill_on_drop` kills the process.
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

/// What a restart attempt ended with.
enum Restart {
    Running(Box<Child>),
    Stopped,
    GaveUp(String),
}

/// State owned by the background task that watches and restarts one child.
struct Supervisor {
    spec: ChildSpec,
    pid: Arc<AtomicU32>,
    stop: oneshot::Receiver<()>,
    outcome: watch::Sender<Option<String>>,
}

impl Supervisor {
    /// Watches `child` until shutdown is requested or the restart budget is spent. Returns the
    /// result of stopping the child; giving up is reported through `outcome` instead.
    async fn run(mut self, mut child: Child) -> anyhow::Result<()> {
        let mut restarts = 0;
        loop {
            let exited = tokio::select! {
                status = child.wait() => Some(describe_exit(status)),
                _ = &mut self.stop => None,
            };
            let Some(exit) = exited else {
                let stopped = stop_child(&mut child).await;
                self.set_pid(None);
                return stopped;
            };
            self.set_pid(None);
            warn!(name = %self.spec.name, %exit, "child exited unexpectedly");
            match self.restart(&mut restarts, exit).await {
                Restart::Running(next) => child = *next,
                Restart::Stopped => return Ok(()),
                Restart::GaveUp(reason) => {
                    self.outcome.send_replace(Some(reason));
                    return Ok(());
                }
            }
        }
    }

    /// Starts the child again after a backoff, retrying until it is ready or the budget runs out.
    async fn restart(&mut self, restarts: &mut u32, mut last_failure: String) -> Restart {
        let policy = self.spec.restart;
        while *restarts < policy.max_restarts {
            let delay = policy.backoff(*restarts);
            *restarts += 1;
            info!(name = %self.spec.name, restart = *restarts, ?delay, "restarting child");
            tokio::select! {
                () = tokio::time::sleep(delay) => {}
                _ = &mut self.stop => return Restart::Stopped,
            }
            match self.start_once().await {
                Ok(Some(child)) => return Restart::Running(Box::new(child)),
                Ok(None) => return Restart::Stopped,
                Err(error) => {
                    warn!(name = %self.spec.name, error = %format!("{error:#}"), "restart failed");
                    last_failure = format!("{error:#}");
                }
            }
        }
        Restart::GaveUp(format!(
            "{} exhausted {} restarts; last failure: {last_failure}",
            self.spec.name, policy.max_restarts
        ))
    }

    /// Spawns the child and waits for readiness. `Ok(None)` means shutdown was requested.
    async fn start_once(&mut self) -> anyhow::Result<Option<Child>> {
        let mut child = tokio::select! {
            child = spawn_child(&self.spec) => child?,
            _ = &mut self.stop => return Ok(None),
        };
        self.set_pid(child.id());
        let ready = tokio::select! {
            ready = wait_ready(&self.spec, &mut child) => Some(ready),
            _ = &mut self.stop => None,
        };
        let failure = match ready {
            Some(Ok(())) => {
                info!(name = %self.spec.name, pid = ?child.id(), "child ready");
                return Ok(Some(child));
            }
            Some(Err(error)) => Some(error),
            None => None,
        };
        if let Err(error) = stop_child(&mut child).await {
            warn!(name = %self.spec.name, error = %format!("{error:#}"), "failed to stop child");
        }
        self.set_pid(None);
        failure.map_or(Ok(None), Err)
    }

    fn set_pid(&self, pid: Option<u32>) {
        self.pid.store(pid.unwrap_or(NO_PID), Ordering::Release);
    }
}

/// Starts the program with piped output forwarded to tracing; the child dies with its handle.
async fn spawn_child(spec: &ChildSpec) -> anyhow::Result<Child> {
    let mut child = Child::spawn(
        Command::new(&spec.program)
            .args(&spec.args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true),
    )
    .await
    .with_context(|| format!("failed to start {} ({})", spec.name, spec.program.display()))?;
    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(forward_lines(spec.name.clone(), "stdout", stdout));
    }
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(forward_lines(spec.name.clone(), "stderr", stderr));
    }
    info!(name = %spec.name, pid = ?child.id(), program = %spec.program.display(), "child started");
    Ok(child)
}

/// Logs every line of a child's output stream until it closes. Invalid UTF-8 is replaced.
async fn forward_lines(name: String, stream: &'static str, pipe: impl AsyncRead + Unpin) {
    let mut reader = BufReader::new(pipe);
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line).await {
            Ok(0) => return,
            Ok(_) => {
                let text = String::from_utf8_lossy(line.trim_ascii_end());
                info!(target: CHILD_LOG_TARGET, name = %name, stream, "{text}");
            }
            Err(error) => {
                warn!(target: CHILD_LOG_TARGET, name = %name, stream, %error, "failed to read child output");
                return;
            }
        }
    }
}

/// Waits until the child passes its readiness probe, failing if it exits or times out first.
async fn wait_ready(spec: &ChildSpec, child: &mut Child) -> anyhow::Result<()> {
    let ready = async {
        tokio::select! {
            status = child.wait() => {
                bail!("{} {} before becoming ready", spec.name, describe_exit(status))
            }
            () = poll_until_ready(&spec.readiness) => Ok(()),
        }
    };
    tokio::time::timeout(spec.ready_timeout, ready)
        .await
        .map_err(|_| {
            anyhow!(
                "{} was not ready within {:?}",
                spec.name,
                spec.ready_timeout
            )
        })?
}

async fn poll_until_ready(readiness: &Readiness) {
    while !probe(readiness).await {
        tokio::time::sleep(READY_POLL_INTERVAL).await;
    }
}

/// Runs one readiness check; any error or stall counts as "not ready yet".
async fn probe(readiness: &Readiness) -> bool {
    let attempt = async {
        match readiness {
            Readiness::TcpPort(addr) => connect(*addr).await.is_ok(),
            Readiness::HttpOk { addr, path } => http_ok(*addr, path).await.unwrap_or(false),
        }
    };
    tokio::time::timeout(PROBE_TIMEOUT, attempt)
        .await
        .unwrap_or(false)
}

async fn connect(addr: SocketAddr) -> std::io::Result<TcpStream> {
    tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(addr))
        .await
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::TimedOut))?
}

/// Sends a minimal HTTP/1.1 GET and reports whether the status code is 200.
async fn http_ok(addr: SocketAddr, path: &str) -> std::io::Result<bool> {
    let mut stream = connect(addr).await?;
    let request = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await?;
    let mut status_line = Vec::new();
    BufReader::new(stream)
        .take(MAX_STATUS_LINE_LEN)
        .read_until(b'\n', &mut status_line)
        .await?;
    Ok(is_ok_status_line(&status_line))
}

fn is_ok_status_line(line: &[u8]) -> bool {
    let line = String::from_utf8_lossy(line);
    let mut parts = line.split_ascii_whitespace();
    matches!(
        (parts.next(), parts.next()),
        (Some(version), Some("200")) if version.starts_with("HTTP/1.")
    )
}

/// Kills the child (if it is still running) and reaps it, so no orphan or zombie is left.
async fn stop_child(child: &mut Child) -> anyhow::Result<()> {
    if let Err(error) = child.start_kill() {
        // Killing fails when the child has already exited; only a live child is a real error.
        if child.try_wait()?.is_none() {
            return Err(error).context("failed to kill child");
        }
    }
    child.wait().await.context("failed to reap child")?;
    Ok(())
}

fn describe_exit(status: std::io::Result<ExitStatus>) -> String {
    match status {
        Ok(status) => format!("exited with {status}"),
        Err(error) => format!("could not be waited on: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    fn policy(initial_ms: u64, max_ms: u64) -> RestartPolicy {
        RestartPolicy {
            max_restarts: 10,
            initial_backoff: Duration::from_millis(initial_ms),
            max_backoff: Duration::from_millis(max_ms),
        }
    }

    #[test]
    fn backoff_doubles_from_initial() {
        let policy = policy(100, 10_000);
        let delays: Vec<u64> = (0..5)
            .map(|retry| policy.backoff(retry).as_millis() as u64)
            .collect();
        assert_eq!(delays, [100, 200, 400, 800, 1600]);
    }

    #[test]
    fn backoff_is_capped_at_max() {
        let policy = policy(500, 8_000);
        assert_eq!(policy.backoff(4), Duration::from_secs(8));
        assert_eq!(policy.backoff(5), Duration::from_secs(8));
        assert_eq!(policy.backoff(u32::MAX), Duration::from_secs(8));
    }

    #[test]
    fn default_policy_matches_agreed_interface() {
        let policy = RestartPolicy::default();
        assert_eq!(policy.max_restarts, 3);
        assert_eq!(policy.backoff(0), Duration::from_millis(500));
        assert_eq!(policy.backoff(10), Duration::from_secs(8));
    }

    #[test]
    fn status_line_must_be_http_200() {
        assert!(is_ok_status_line(b"HTTP/1.1 200 OK\r\n"));
        assert!(is_ok_status_line(b"HTTP/1.0 200\r\n"));
        assert!(!is_ok_status_line(b"HTTP/1.1 503 Service Unavailable\r\n"));
        assert!(!is_ok_status_line(b"HTTP/1.1 2000 Odd\r\n"));
        assert!(!is_ok_status_line(b"SSH-2.0-OpenSSH 200\r\n"));
        assert!(!is_ok_status_line(b""));
    }

    /// Serves one fixed HTTP response to every connection.
    async fn http_server(response: &'static str) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = [0u8; 1024];
                let _ = socket.read(&mut request).await;
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        addr
    }

    #[tokio::test]
    async fn http_probe_accepts_200_only() {
        let healthy = http_server("HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n").await;
        let failing = http_server("HTTP/1.1 503 Service Unavailable\r\n\r\n").await;
        let path = "/health".to_string();
        assert!(
            probe(&Readiness::HttpOk {
                addr: healthy,
                path: path.clone()
            })
            .await
        );
        assert!(
            !probe(&Readiness::HttpOk {
                addr: failing,
                path
            })
            .await
        );
    }

    #[tokio::test]
    async fn tcp_probe_needs_a_listener() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        assert!(probe(&Readiness::TcpPort(addr)).await);
        drop(listener);
        assert!(!probe(&Readiness::TcpPort(addr)).await);
    }
}
