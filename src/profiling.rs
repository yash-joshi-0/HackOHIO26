//! Managed local profiler lifecycle. Only startup-configured candidates may be launched.
//!
//! Uses the same process-local gate as managed native inference. A helper must exit with a
//! matching final report confirming native cleanup before capacity is reusable. Missing or
//! incomplete cleanup evidence fences compute, including after forced helper termination.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use anyhow::{Context, ensure};
use axum::extract::{DefaultBodyLimit, Path as RoutePath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::watch;
use tokio::time::timeout;
use tracing::{info, warn};
use uuid::Uuid;

use crate::admission::{ComputeAdmission, ProfileLease};

const MAX_REPORT_BYTES: u64 = 8 * 1024 * 1024;
const MAX_FINISHED_RUNS: usize = 64;

/// Trusted operator configuration, never accepted from an HTTP request or a mesh peer.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileConfig {
    pub python: PathBuf,
    pub script: PathBuf,
    pub script_sha256: String,
    pub output_root: PathBuf,
    pub candidates: BTreeMap<String, PathBuf>,
    pub max_run_seconds: u64,
    pub cleanup_grace_seconds: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileStatus {
    Running,
    Cancelling,
    Finished,
    Fenced,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProfileInfo {
    pub run_id: String,
    pub candidate: String,
    pub status: ProfileStatus,
    pub outcome: Option<String>,
    pub error: Option<String>,
    pub output_dir: PathBuf,
    pub cleanup_confirmed: bool,
    pub eligible_for_placement: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    #[error("profile service is unavailable or shutting down")]
    Unavailable,
    #[error("compute opt-in is required")]
    OptInRequired,
    #[error("unknown configured candidate")]
    UnknownCandidate,
    #[error("unknown profile run")]
    NotFound,
    #[error("managed native compute is busy")]
    Busy,
    #[error("compute fenced: native cleanup is unconfirmed")]
    Fenced,
    #[error("the diagnostic helper requires a POSIX host")]
    Unsupported,
}

struct Run {
    info: ProfileInfo,
    cancel: watch::Sender<bool>,
}

type Runs = Arc<Mutex<VecDeque<Run>>>;

fn lock(runs: &Runs) -> MutexGuard<'_, VecDeque<Run>> {
    runs.lock().unwrap_or_else(PoisonError::into_inner)
}

struct ManagerInner {
    config: ProfileConfig,
    admission: ComputeAdmission,
    runs: Runs,
    closed: AtomicBool,
}

impl Drop for ManagerInner {
    fn drop(&mut self) {
        for run in lock(&self.runs).iter() {
            run.cancel.send_replace(true);
        }
    }
}

#[derive(Clone)]
pub struct ProfileManager(Arc<ManagerInner>);

impl ProfileManager {
    pub async fn from_file(path: &Path) -> anyhow::Result<Self> {
        let config = serde_json::from_slice(&read_bounded(path, 1024 * 1024).await?)?;
        Self::new(config).await
    }

    pub async fn new(config: ProfileConfig) -> anyhow::Result<Self> {
        Self::with_admission(config, ComputeAdmission::process()).await
    }

    pub async fn with_admission(
        mut config: ProfileConfig,
        admission: ComputeAdmission,
    ) -> anyhow::Result<Self> {
        ensure!(
            config.script.is_absolute() && config.output_root.is_absolute(),
            "script and output root must be absolute paths"
        );
        ensure!(
            (1..=3600).contains(&config.max_run_seconds),
            "invalid profile time limit"
        );
        ensure!(
            (1..=30).contains(&config.cleanup_grace_seconds),
            "invalid cleanup grace"
        );
        ensure!(
            !config.candidates.is_empty() && config.candidates.len() <= 128,
            "configure 1..128 candidates"
        );
        ensure!(
            config.python.is_absolute() && config.python.is_file(),
            "python must name an absolute local executable"
        );
        config.script = tokio::fs::canonicalize(&config.script).await?;
        verify_script(&config).await?;
        for (name, path) in &mut config.candidates {
            ensure!(path.is_absolute(), "candidate paths must be absolute");
            ensure!(
                !name.is_empty()
                    && name.len() <= 64
                    && name
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b"-_".contains(&c)),
                "invalid candidate name"
            );
            *path = tokio::fs::canonicalize(&*path).await?;
            ensure!(path.is_file(), "candidate request must be a regular file");
        }
        tokio::fs::create_dir_all(&config.output_root).await?;
        config.output_root = tokio::fs::canonicalize(&config.output_root).await?;
        Ok(Self(Arc::new(ManagerInner {
            config,
            admission,
            runs: Arc::new(Mutex::new(VecDeque::new())),
            closed: AtomicBool::new(false),
        })))
    }

    pub fn admission(&self) -> &ComputeAdmission {
        &self.0.admission
    }

    pub fn start(
        &self,
        candidate: &str,
        compute_enabled: bool,
    ) -> Result<ProfileInfo, ProfileError> {
        let runtime =
            tokio::runtime::Handle::try_current().map_err(|_| ProfileError::Unavailable)?;
        if self.0.closed.load(Ordering::Acquire) {
            return Err(ProfileError::Unavailable);
        }
        if !cfg!(unix) {
            return Err(ProfileError::Unsupported);
        }
        if !compute_enabled {
            return Err(ProfileError::OptInRequired);
        }
        let request = self
            .0
            .config
            .candidates
            .get(candidate)
            .ok_or(ProfileError::UnknownCandidate)?
            .clone();
        let lease = self.0.admission.try_profile().map_err(|_| {
            if self.0.admission.is_fenced() {
                ProfileError::Fenced
            } else {
                ProfileError::Busy
            }
        })?;
        let id = Uuid::new_v4().to_string();
        let info = ProfileInfo {
            output_dir: self.0.config.output_root.join(&id),
            run_id: id,
            candidate: candidate.to_owned(),
            status: ProfileStatus::Running,
            outcome: None,
            error: None,
            cleanup_confirmed: false,
            eligible_for_placement: false,
        };
        let (cancel, receiver) = watch::channel(false);
        let mut runs = lock(&self.0.runs);
        if self.0.closed.load(Ordering::Acquire) {
            return Err(ProfileError::Unavailable);
        }
        while runs.len() >= MAX_FINISHED_RUNS {
            let Some(index) = runs.iter().position(|r| {
                matches!(
                    r.info.status,
                    ProfileStatus::Finished | ProfileStatus::Fenced
                )
            }) else {
                break;
            };
            runs.remove(index);
        }
        runs.push_back(Run {
            info: info.clone(),
            cancel,
        });
        drop(runs);
        runtime.spawn(run_profile(
            self.0.config.clone(),
            request,
            info.clone(),
            receiver,
            lease,
            Arc::clone(&self.0.runs),
        ));
        Ok(info)
    }

    pub fn status(&self, id: &str) -> Result<ProfileInfo, ProfileError> {
        lock(&self.0.runs)
            .iter()
            .find(|r| r.info.run_id == id)
            .map(|r| r.info.clone())
            .ok_or(ProfileError::NotFound)
    }

    pub fn cancel(&self, id: &str) -> Result<ProfileInfo, ProfileError> {
        let mut runs = lock(&self.0.runs);
        let run = runs
            .iter_mut()
            .find(|r| r.info.run_id == id)
            .ok_or(ProfileError::NotFound)?;
        if run.info.status == ProfileStatus::Running {
            run.info.status = ProfileStatus::Cancelling;
            run.cancel.send_replace(true);
        }
        Ok(run.info.clone())
    }

    /// Request cleanup and wait before the host tears down its async runtime.
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        self.0.closed.store(true, Ordering::Release);
        for run in lock(&self.0.runs).iter_mut() {
            if matches!(
                run.info.status,
                ProfileStatus::Running | ProfileStatus::Cancelling
            ) {
                run.info.status = ProfileStatus::Cancelling;
                run.cancel.send_replace(true);
            }
        }
        let stopped = timeout(
            Duration::from_secs(self.0.config.cleanup_grace_seconds + 5),
            async {
                loop {
                    if lock(&self.0.runs).iter().all(|r| {
                        matches!(
                            r.info.status,
                            ProfileStatus::Finished | ProfileStatus::Fenced
                        )
                    }) {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
            },
        )
        .await;
        if stopped.is_err() {
            self.0.admission.fence();
            anyhow::bail!("profile shutdown timed out; compute fenced");
        }
        Ok(())
    }

    /// Merge only into a trusted loopback control listener. Arbitrary paths/commands are absent.
    pub fn router(&self) -> Router {
        Router::new()
            .route("/profiles", post(start))
            .route("/profiles/{id}", get(status).delete(cancel))
            .layer(DefaultBodyLimit::max(1024))
            .with_state(self.clone())
    }
}

async fn verify_script(config: &ProfileConfig) -> anyhow::Result<()> {
    let bytes = read_bounded(&config.script, 1024 * 1024).await?;
    ensure!(
        hex::encode(Sha256::digest(&bytes)) == config.script_sha256,
        "profiler script checksum mismatch"
    );
    Ok(())
}

async fn read_bounded(path: &Path, max: u64) -> anyhow::Result<Vec<u8>> {
    ensure!(
        tokio::fs::metadata(path).await?.is_file(),
        "expected a regular file"
    );
    let file = tokio::fs::File::open(path).await?;
    ensure!(file.metadata().await?.is_file(), "expected a regular file");
    let mut bytes = Vec::new();
    file.take(max + 1).read_to_end(&mut bytes).await?;
    ensure!(
        bytes.len() as u64 <= max,
        "file exceeds control-plane size limit"
    );
    Ok(bytes)
}

async fn run_profile(
    config: ProfileConfig,
    request: PathBuf,
    info: ProfileInfo,
    mut cancel: watch::Receiver<bool>,
    mut lease: ProfileLease,
    runs: Runs,
) {
    info!(run_id = %info.run_id, candidate = %info.candidate, "profile starting");
    let cancel_file = config.output_root.join(format!(".cancel-{}", info.run_id));
    let result = execute_profile(
        &config,
        &request,
        &info,
        &cancel_file,
        &mut cancel,
        &mut lease,
        &runs,
    )
    .await;
    let (outcome, error, confirmed) = match result {
        Ok(done) => (Some(done.outcome), done.detail, true),
        Err((error, confirmed)) => (
            Some("failed".to_owned()),
            Some(format!("{error:#}")),
            confirmed,
        ),
    };
    if confirmed {
        lease.confirm_cleanup();
    }
    // The fence is established before publishing terminal status or releasing capacity.
    drop(lease);
    let mut records = lock(&runs);
    if let Some(run) = records.iter_mut().find(|r| r.info.run_id == info.run_id) {
        run.info.status = if confirmed {
            ProfileStatus::Finished
        } else {
            ProfileStatus::Fenced
        };
        run.info.outcome = outcome;
        run.info.error = error;
        run.info.cleanup_confirmed = confirmed;
        info!(run_id = %info.run_id, status = ?run.info.status, outcome = ?run.info.outcome, "profile finished");
    }
    // Cancellation markers and all run evidence are retained for audit; a UUID is never reused.
}

struct HelperOutcome {
    outcome: String,
    detail: Option<String>,
}

async fn execute_profile(
    config: &ProfileConfig,
    request: &Path,
    info: &ProfileInfo,
    cancel_file: &Path,
    cancel: &mut watch::Receiver<bool>,
    lease: &mut ProfileLease,
    runs: &Runs,
) -> Result<HelperOutcome, (anyhow::Error, bool)> {
    if *cancel.borrow() {
        return Ok(HelperOutcome {
            outcome: "cancelled".to_owned(),
            detail: None,
        });
    }
    timeout(
        Duration::from_secs(config.max_run_seconds),
        verify_script(config),
    )
    .await
    .context("script validation timed out")
    .and_then(|v| v)
    .map_err(|e| (e, true))?;
    if *cancel.borrow() {
        return Ok(HelperOutcome {
            outcome: "cancelled".to_owned(),
            detail: None,
        });
    }
    lease.execution_started();
    let mut child = Command::new(&config.python)
        .arg(&config.script)
        .arg(request)
        .arg("--output-dir")
        .arg(&info.output_dir)
        .arg("--cancel-file")
        .arg(cancel_file)
        .arg("--run-id")
        .arg(&info.run_id)
        .arg("--confirm-idle")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| (e.into(), true))?;
    let exited = tokio::select! {
        status = child.wait() => status,
        _ = tokio::time::sleep(Duration::from_secs(config.max_run_seconds)) => {
            stop_helper(&mut child, config, cancel_file, runs, &info.run_id).await
        }
        _ = cancel.changed() => {
            stop_helper(&mut child, config, cancel_file, runs, &info.run_id).await
        }
    }
    .map_err(|e| (e.into(), false))?;
    let report: Value = serde_json::from_slice(
        &read_bounded(&info.output_dir.join("report.json"), MAX_REPORT_BYTES)
            .await
            .map_err(|e| (e.context("missing final profiler report"), false))?,
    )
    .map_err(|e| (e.into(), false))?;
    let outcome = report["status"].as_str().unwrap_or("invalid");
    let terminal = matches!(
        outcome,
        "passed"
            | "failed"
            | "cancelled"
            | "timed_out"
            | "limit_exceeded"
            | "insufficient_duration"
    );
    let started = report["counters"]["children_started"].as_u64();
    let reaped = report["counters"]["children_reaped"].as_u64();
    let confirmed = report["schema_version"] == 1
        && report["kind"] == "native_candidate_diagnostic"
        && report["profile_id"] == info.run_id
        && report["cleanup_confirmed"] == true
        && started.is_some()
        && started == reaped
        && report["eligible_for_placement"] == false
        && terminal
        && exited.success() == (outcome == "passed");
    if !confirmed {
        return Err((
            anyhow::anyhow!(
                "helper exit/report did not confirm native cleanup; inspect retained evidence"
            ),
            false,
        ));
    }
    Ok(HelperOutcome {
        outcome: outcome.to_owned(),
        detail: report["error"]
            .as_str()
            .map(|text| text.chars().take(1024).collect()),
    })
}

async fn stop_helper(
    child: &mut tokio::process::Child,
    config: &ProfileConfig,
    cancel_file: &Path,
    runs: &Runs,
    id: &str,
) -> std::io::Result<std::process::ExitStatus> {
    if let Some(run) = lock(runs).iter_mut().find(|r| r.info.run_id == id) {
        run.info.status = ProfileStatus::Cancelling;
    }
    let signalled = timeout(
        Duration::from_secs(2),
        tokio::fs::write(cancel_file, b"cancel\n"),
    )
    .await;
    if matches!(signalled, Ok(Ok(())))
        && let Ok(status) = timeout(
            Duration::from_secs(config.cleanup_grace_seconds),
            child.wait(),
        )
        .await
    {
        return status;
    }
    warn!(
        run_id = id,
        "profiler cancellation unconfirmed; terminating helper and fencing compute"
    );
    let _ = child.start_kill();
    let _ = child.wait().await;
    Err(std::io::Error::other(
        "helper forced to stop; native descendants require inspection",
    ))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StartRequest {
    candidate: String,
    compute_enabled: bool,
}

async fn start(
    State(manager): State<ProfileManager>,
    Json(request): Json<StartRequest>,
) -> Response {
    match manager.start(&request.candidate, request.compute_enabled) {
        Ok(info) => (StatusCode::ACCEPTED, Json(info)).into_response(),
        Err(error) => response_error(error),
    }
}

async fn status(
    State(manager): State<ProfileManager>,
    RoutePath(id): RoutePath<String>,
) -> Response {
    match manager.status(&id) {
        Ok(info) => Json(info).into_response(),
        Err(error) => response_error(error),
    }
}

async fn cancel(
    State(manager): State<ProfileManager>,
    RoutePath(id): RoutePath<String>,
) -> Response {
    match manager.cancel(&id) {
        Ok(info) => Json(info).into_response(),
        Err(error) => response_error(error),
    }
}

fn response_error(error: ProfileError) -> Response {
    let status = match error {
        ProfileError::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        ProfileError::OptInRequired => StatusCode::FORBIDDEN,
        ProfileError::UnknownCandidate | ProfileError::NotFound => StatusCode::NOT_FOUND,
        ProfileError::Busy => StatusCode::CONFLICT,
        ProfileError::Fenced => StatusCode::SERVICE_UNAVAILABLE,
        ProfileError::Unsupported => StatusCode::NOT_IMPLEMENTED,
    };
    (status, Json(json!({"error": error.to_string()}))).into_response()
}
