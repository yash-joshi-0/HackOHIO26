//! Real Rust owner -> Python profiler -> synthetic native process, with no hardware inference.
#![cfg(unix)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::Request;
use llm_cluster::admission::{AdmittedChild, ComputeAdmission};
use llm_cluster::api::{ApiConfig, serve_api_with_profiles};
use llm_cluster::history::LoggingHistorySink;
use llm_cluster::jobs::JobQueue;
use llm_cluster::profiling::{
    ProfileConfig, ProfileError, ProfileInfo, ProfileManager, ProfileStatus,
};
use llm_cluster::shard::pipeline::{LocalLaunch, StageLink};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::process::Command;
use tokio::sync::oneshot;
use tokio::time::timeout;
use tower::ServiceExt;

mod common;

const WAIT: Duration = Duration::from_secs(20);

#[test]
fn runtime_shutdown_confirms_killed_child_without_spuriously_fencing() {
    let gate = ComputeAdmission::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let child = AdmittedChild::spawn_on(
            Command::new(python()).args(["-c", "import time; time.sleep(60)"]),
            &gate,
        )
        .await
        .unwrap();
        drop(child);
        // Return without yielding to the spawned async reaper.
    });
    drop(runtime);
    assert!(!gate.is_fenced());
    assert!(gate.try_profile().is_ok());
}

struct Fixture {
    dir: tempfile::TempDir,
    config: ProfileConfig,
    admission: ComputeAdmission,
}

fn python() -> PathBuf {
    let output = std::process::Command::new("python3")
        .args(["-c", "import sys; print(sys.executable)"])
        .output()
        .unwrap();
    assert!(output.status.success());
    PathBuf::from(String::from_utf8(output.stdout).unwrap().trim())
}

impl Fixture {
    fn new(mode: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let scripts =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("experiments/model-validation");
        let python = python();
        let output = std::process::Command::new(&python).arg("-c")
            .arg("import sys; from pathlib import Path; from test_profile_candidate import CandidateTests; c=CandidateTests(); c.root=Path(sys.argv[1]); c.fixture(sys.argv[2])")
            .arg(dir.path()).arg(mode).current_dir(&scripts).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let script = scripts.join("profile_candidate.py");
        let config = ProfileConfig {
            script_sha256: hash(&script),
            script,
            python,
            output_root: dir.path().join("runs"),
            candidates: BTreeMap::from([("candidate".to_owned(), dir.path().join("request.json"))]),
            max_run_seconds: 15,
            cleanup_grace_seconds: 5,
        };
        Self {
            dir,
            config,
            admission: ComputeAdmission::default(),
        }
    }

    async fn manager(&self) -> ProfileManager {
        ProfileManager::with_admission(self.config.clone(), self.admission.clone())
            .await
            .unwrap()
    }

    fn replace_helper(&mut self, source: &str) {
        self.config.script = self.dir.path().join("helper.py");
        std::fs::write(&self.config.script, source).unwrap();
        self.config.script_sha256 = hash(&self.config.script);
    }
}

fn hash(path: &Path) -> String {
    hex::encode(Sha256::digest(std::fs::read(path).unwrap()))
}

async fn finished(manager: &ProfileManager, id: &str) -> ProfileInfo {
    timeout(WAIT, async {
        loop {
            let info = manager.status(id).unwrap();
            if matches!(info.status, ProfileStatus::Finished | ProfileStatus::Fenced) {
                return info;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("profile failed to terminate")
}

async fn exists(path: &Path) {
    timeout(WAIT, async {
        while !path.exists() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("process marker did not appear");
}

#[tokio::test]
async fn real_helper_finishes_and_returns_capacity_without_granting_eligibility() {
    let fixture = Fixture::new("ok");
    let manager = fixture.manager().await;
    let run = manager.start("candidate", true).unwrap();
    let done = finished(&manager, &run.run_id).await;
    assert_eq!(done.status, ProfileStatus::Finished, "{done:?}");
    assert_eq!(done.outcome.as_deref(), Some("passed"));
    assert!(done.cleanup_confirmed);
    assert!(!done.eligible_for_placement);
    let report: Value =
        serde_json::from_slice(&std::fs::read(done.output_dir.join("report.json")).unwrap())
            .unwrap();
    assert_eq!(report["profile_id"], run.run_id);
    assert_eq!(report["counters"]["children_started"], 2);
    assert_eq!(report["counters"]["children_reaped"], 2);
    assert!(fixture.admission.try_profile().is_ok());
    manager.shutdown().await.unwrap();
    assert!(matches!(
        manager.start("candidate", true),
        Err(ProfileError::Unavailable)
    ));
}

#[tokio::test]
async fn rejected_correctness_does_not_quarantine_cleaned_up_native_work() {
    let fixture = Fixture::new("incorrect");
    let manager = fixture.manager().await;
    let run = manager.start("candidate", true).unwrap();
    let done = finished(&manager, &run.run_id).await;
    assert_eq!(done.status, ProfileStatus::Finished, "{done:?}");
    assert_eq!(done.outcome.as_deref(), Some("failed"));
    assert!(
        done.error
            .as_deref()
            .unwrap()
            .contains("reference output mismatch")
    );
    assert!(fixture.admission.try_profile().is_ok());
}

#[tokio::test]
async fn inference_prevents_profile_admission_and_opt_in_is_required() {
    let fixture = Fixture::new("ok");
    let manager = fixture.manager().await;
    assert!(matches!(
        manager.start("candidate", false),
        Err(ProfileError::OptInRequired)
    ));
    assert!(matches!(
        manager.start("unregistered", true),
        Err(ProfileError::UnknownCandidate)
    ));
    let inference = fixture.admission.inference().await.unwrap();
    assert!(matches!(
        manager.start("candidate", true),
        Err(ProfileError::Busy)
    ));
    assert!(!fixture.dir.path().join("pid").exists());
    drop(inference);
    let run = manager.start("candidate", true).unwrap();
    assert_eq!(
        finished(&manager, &run.run_id).await.status,
        ProfileStatus::Finished
    );
}

#[tokio::test]
async fn cancel_reaps_native_before_waiting_inference_starts() {
    let fixture = Fixture::new("step_hang");
    let manager = fixture.manager().await;
    let run = manager.start("candidate", true).unwrap();
    exists(&fixture.dir.path().join("pid")).await;
    let native_pid = std::fs::read_to_string(fixture.dir.path().join("pid")).unwrap();
    let marker = fixture.dir.path().join("inference-started");
    let marker_arg = marker.clone();
    let admission = fixture.admission.clone();
    let python = fixture.config.python.clone();
    let next = tokio::spawn(async move {
        let mut child = AdmittedChild::spawn_on(
            Command::new(python)
                .arg("-c")
                .arg("import sys; from pathlib import Path; Path(sys.argv[1]).touch()")
                .arg(marker_arg),
            &admission,
        )
        .await
        .unwrap();
        assert!(child.wait().await.unwrap().success());
    });
    tokio::task::yield_now().await;
    assert!(!marker.exists());
    assert!(matches!(
        manager.start("candidate", true),
        Err(ProfileError::Busy)
    ));
    manager.cancel(&run.run_id).unwrap();
    let done = finished(&manager, &run.run_id).await;
    assert_eq!(done.status, ProfileStatus::Finished, "{done:?}");
    assert_eq!(done.outcome.as_deref(), Some("cancelled"));
    timeout(WAIT, next).await.unwrap().unwrap();
    assert!(marker.exists());
    let absent = std::process::Command::new(&fixture.config.python).arg("-c")
        .arg("import os,sys;\ntry: os.kill(int(sys.argv[1]),0)\nexcept ProcessLookupError: sys.exit(0)\nsys.exit(1)")
        .arg(native_pid).status().unwrap();
    assert!(
        absent.success(),
        "cancelled native process survived admission release"
    );
}

#[tokio::test]
async fn dropping_inference_child_keeps_lease_until_reap() {
    let admission = ComputeAdmission::default();
    let mut child = AdmittedChild::spawn_on(
        Command::new(python())
            .args(["-c", "import time; time.sleep(60)"])
            .stdin(Stdio::null()),
        &admission,
    )
    .await
    .unwrap();
    let pid = child.id().unwrap();
    // Cancel a pending wait, then drop its process owner as an HTTP disconnect would.
    assert!(
        timeout(Duration::from_millis(20), child.wait())
            .await
            .is_err()
    );
    drop(child);
    assert!(admission.try_profile().is_err());
    timeout(WAIT, async {
        loop {
            if let Ok(lease) = admission.try_profile() {
                drop(lease);
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let output = std::process::Command::new(python()).arg("-c")
        .arg("import os,sys;\ntry: os.kill(int(sys.argv[1]),0)\nexcept ProcessLookupError: sys.exit(0)\nsys.exit(1)")
        .arg(pid.to_string()).status().unwrap();
    assert!(output.success());
}

#[tokio::test]
async fn shard_stage_uses_the_process_admission_owner() {
    let gate = ComputeAdmission::process();
    let fixture = Fixture::new("ok");
    let manager = ProfileManager::new(fixture.config.clone()).await.unwrap();
    let runner = common::example_bin("fake_shard_runner");
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/fake-shard-model.json");
    let env = BTreeMap::new();
    let launch = LocalLaunch {
        runner: &runner,
        model: &model,
        begin: 0,
        end: 4,
        context: 64,
        gpu_layers: 0,
        budget_bytes: 1_048_576,
        prompt: "hi",
        env: &env,
    };
    let stage = StageLink::spawn_local(&launch, WAIT).await.unwrap();
    assert!(gate.try_profile().is_err());
    assert!(matches!(
        manager.start("candidate", true),
        Err(ProfileError::Busy)
    ));
    drop(stage);
    timeout(WAIT, async {
        loop {
            if gate.try_profile().is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let run = manager.start("candidate", true).unwrap();
    assert_eq!(
        finished(&manager, &run.run_id).await.status,
        ProfileStatus::Finished
    );
}

#[tokio::test]
async fn missing_final_report_fences_all_new_managed_work() {
    let mut fixture = Fixture::new("ok");
    fixture.replace_helper("import sys; sys.exit(7)\n");
    let manager = fixture.manager().await;
    let run = manager.start("candidate", true).unwrap();
    let done = finished(&manager, &run.run_id).await;
    assert_eq!(done.status, ProfileStatus::Fenced);
    assert!(!done.cleanup_confirmed);
    assert!(fixture.admission.inference().await.is_err());
    assert!(matches!(
        manager.start("candidate", true),
        Err(ProfileError::Fenced)
    ));
}

#[tokio::test]
async fn forced_helper_termination_does_not_release_usable_capacity() {
    let mut fixture = Fixture::new("ok");
    fixture.replace_helper("import time; time.sleep(60)\n");
    fixture.config.max_run_seconds = 1;
    fixture.config.cleanup_grace_seconds = 1;
    let manager = fixture.manager().await;
    let run = manager.start("candidate", true).unwrap();
    let done = finished(&manager, &run.run_id).await;
    assert_eq!(done.status, ProfileStatus::Fenced);
    assert!(fixture.admission.is_fenced());
}

#[tokio::test]
async fn stale_helper_report_cannot_confirm_cleanup_for_this_run() {
    let mut fixture = Fixture::new("ok");
    fixture.replace_helper(
        r#"import json,sys
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--output-dir')+1]); p.mkdir()
(p/'report.json').write_text(json.dumps({'schema_version':1,'kind':'native_candidate_diagnostic',
 'profile_id':'old-run','status':'passed','cleanup_confirmed':True,'eligible_for_placement':False,
 'counters':{'children_started':0,'children_reaped':0}}))
"#,
    );
    let manager = fixture.manager().await;
    let run = manager.start("candidate", true).unwrap();
    assert_eq!(
        finished(&manager, &run.run_id).await.status,
        ProfileStatus::Fenced
    );
}

#[tokio::test]
async fn dropping_unconfirmed_execution_lease_fences_even_on_task_abort() {
    let admission = ComputeAdmission::default();
    let owner = admission.clone();
    let (ready, started) = oneshot::channel();
    let task = tokio::spawn(async move {
        let mut lease = owner.try_profile().unwrap();
        lease.execution_started();
        ready.send(()).unwrap();
        std::future::pending::<()>().await;
        drop(lease);
    });
    started.await.unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert!(admission.inference().await.is_err());
}

#[tokio::test]
async fn loopback_api_uses_registered_candidates_and_drains_on_shutdown() {
    let fixture = Fixture::new("step_hang");
    let manager = fixture.manager().await;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let (stop, stopped) = oneshot::channel();
    let service = tokio::spawn(serve_api_with_profiles(
        ApiConfig {
            listen: addr,
            upstream: "127.0.0.1:1".parse().unwrap(),
        },
        Arc::new(JobQueue::new()),
        Arc::new(LoggingHistorySink),
        Some(manager.clone()),
        async {
            let _ = stopped.await;
        },
    ));
    let client = reqwest::Client::new();
    let root = format!("http://{addr}");
    timeout(WAIT, async {
        while client.get(format!("{root}/jobs")).send().await.is_err() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    let rejected = client
        .post(format!("{root}/profiles"))
        .json(&json!({"candidate":"candidate","compute_enabled":true,"command":"unexpected"}))
        .send()
        .await
        .unwrap();
    assert_eq!(rejected.status(), 422);
    let denied = client
        .post(format!("{root}/profiles"))
        .json(&json!({"candidate":"candidate","compute_enabled":false}))
        .send()
        .await
        .unwrap();
    assert_eq!(denied.status(), 403);
    let started = client
        .post(format!("{root}/profiles"))
        .json(&json!({"candidate":"candidate","compute_enabled":true}))
        .send()
        .await
        .unwrap();
    assert_eq!(started.status(), 202);
    let body: Value = started.json().await.unwrap();
    let id = body["run_id"].as_str().unwrap();
    exists(&fixture.dir.path().join("pid")).await;
    assert_eq!(
        client
            .get(format!("{root}/profiles/{id}"))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    stop.send(()).unwrap();
    timeout(WAIT, service).await.unwrap().unwrap().unwrap();
    let done = manager.status(id).unwrap();
    assert_eq!(done.status, ProfileStatus::Finished, "{done:?}");
    assert!(done.cleanup_confirmed);
    assert!(fixture.admission.try_profile().is_ok());
}

#[tokio::test]
async fn control_router_checks_opt_in_schema_and_cancel_without_sockets() {
    let fixture = Fixture::new("step_hang");
    let manager = fixture.manager().await;
    let router = manager.router();
    for (payload, expected) in [
        (
            json!({"candidate":"candidate","compute_enabled":true,"command":"unexpected"}),
            422,
        ),
        (
            json!({"candidate":"candidate","compute_enabled":false}),
            403,
        ),
        (json!({"candidate":"unknown","compute_enabled":true}), 404),
    ] {
        let reply = router
            .clone()
            .oneshot(
                Request::post("/profiles")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(reply.status(), expected);
    }
    let reply = router
        .clone()
        .oneshot(
            Request::post("/profiles")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"candidate":"candidate","compute_enabled":true}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(reply.status(), 202);
    let bytes = axum::body::to_bytes(reply.into_body(), 65536)
        .await
        .unwrap();
    let info: Value = serde_json::from_slice(&bytes).unwrap();
    let id = info["run_id"].as_str().unwrap();
    exists(&fixture.dir.path().join("pid")).await;
    let reply = router
        .clone()
        .oneshot(
            Request::get(format!("/profiles/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(reply.status(), 200);
    let reply = router
        .oneshot(
            Request::delete(format!("/profiles/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(reply.status(), 200);
    manager.shutdown().await.unwrap();
    let done = manager.status(id).unwrap();
    assert_eq!(done.status, ProfileStatus::Finished, "{done:?}");
    assert_eq!(done.outcome.as_deref(), Some("cancelled"));
    assert!(fixture.admission.try_profile().is_ok());
}
