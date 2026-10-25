use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use llm_cluster::ring_deployment::*;
use llm_cluster::shard::manifest::{Execution, ShardManifest, StageLocation};
use serde_json::{Value, json};

#[path = "common/ring_fixture.rs"]
mod fixture;

#[test]
fn measured_ranges_become_real_manifest_and_worker_configs() {
    let (request, registry) = fixture::inputs(1000);
    let bundle = prepare(&request, &registry, 1100).unwrap();
    assert!(!bundle.activation_ready);
    assert!(bundle.experimental && bundle.plan.advisory);
    assert_eq!(bundle.plan.predicted_decode_us, 36);
    assert_eq!(bundle.plan.assignments.len(), 3);
    assert_eq!(bundle.expires_at_unix_ms, 61_000);
    let manifest: ShardManifest = serde_json::from_value(bundle.leader_manifest.clone()).unwrap();
    manifest.validate().unwrap();
    assert_eq!(manifest.execution, Execution::Ring);
    assert_eq!((manifest.stages[0].begin, manifest.stages[0].end), (0, 1));
    assert!(matches!(
        manifest.stages[0].location,
        StageLocation::Local { .. }
    ));
    assert_eq!(
        manifest.remote_workers(),
        [
            registry.devices[1].endpoint_id,
            registry.devices[2].endpoint_id
        ]
    );
    let entry = bundle.workers[0].config().unwrap();
    let terminal = bundle.workers[1].config().unwrap();
    assert_eq!(entry.upstream, bundle.leader.endpoint_id);
    assert_eq!(entry.origin, bundle.leader.endpoint_id);
    assert_eq!(entry.next, Some(bundle.workers[1].stage.endpoint_id));
    assert_eq!(terminal.upstream, bundle.workers[0].stage.endpoint_id);
    assert_eq!(terminal.origin, bundle.leader.endpoint_id);
    assert_eq!(terminal.next, None);
    assert_eq!(entry.source_sha256, Some(request.workload.model_sha256));
    assert_eq!(
        (entry.begin, entry.end, entry.context, entry.budget_bytes),
        (1, 3, 512, 256)
    );
    // The native buffer limit stays distinct from the planner's physical pool charge.
    assert_eq!(bundle.plan.assignments[1].charged_bytes.ram, 512);
    assert!(
        bundle.workers[0]
            .argv
            .iter()
            .any(|arg| arg == "/synthetic/entry/runner with spaces")
    );
}

#[test]
fn artifact_roles_bind_stage_requirements_without_claiming_tensor_verification() {
    let (request, mut registry) = fixture::inputs(1000);
    registry.devices[0].artifacts[0].role = Some(ArtifactRole::First);
    registry.devices[1].artifacts[0].role = Some(ArtifactRole::Middle);
    registry.devices[2].artifacts[0].role = Some(ArtifactRole::Final);
    let bundle = prepare(&request, &registry, 1100).unwrap();
    assert!(!bundle.activation_ready);
    assert_eq!(bundle.leader.artifact.role, Some(ArtifactRole::First));
    assert_eq!(
        bundle.workers[1].stage.artifact.role,
        Some(ArtifactRole::Final)
    );
    validate(&bundle, &request, &registry, 1200).unwrap();
    for (device, wrong_role) in [
        (0, ArtifactRole::Final),
        (1, ArtifactRole::First),
        (2, ArtifactRole::Middle),
        (2, ArtifactRole::Full),
    ] {
        let mut wrong = registry.clone();
        wrong.devices[device].artifacts[0].role = Some(wrong_role);
        let error = prepare(&request, &wrong, 1100).unwrap_err();
        assert!(error.to_string().contains("artifact role"));
    }
}

#[test]
fn omitted_artifact_roles_preserve_legacy_json_and_bundle_hashes() {
    let (request, registry) = fixture::inputs(1000);
    let serialized = serde_json::to_value(&registry).unwrap();
    assert!(
        serialized["devices"][0]["artifacts"][0]
            .get("role")
            .is_none()
    );
    let round_trip = serde_json::from_value(serialized).unwrap();
    let bundle = prepare(&request, &registry, 1100).unwrap();
    assert_eq!(bundle, prepare(&request, &round_trip, 1100).unwrap());
    let mut with_roles = registry.clone();
    with_roles.devices[2].artifacts[0].role = Some(ArtifactRole::Final);
    assert!(validate(&bundle, &request, &with_roles, 1200).is_err());
    assert_ne!(
        bundle.preparation_id,
        prepare(&request, &with_roles, 1100).unwrap().preparation_id
    );
}

#[test]
fn unavailable_peer_replans_to_another_measured_range_without_hardcoded_layers() {
    let (mut request, registry) = fixture::inputs(1000);
    let first = prepare(&request, &registry, 1100).unwrap();
    request.peers[2].healthy = false;
    let replacement = prepare(&request, &registry, 1100).unwrap();
    assert_eq!(first.plan.assignments.len(), 3);
    assert_eq!(replacement.plan.assignments.len(), 2);
    assert_eq!(replacement.leader.artifact.end, 2);
    assert_eq!(replacement.workers[0].stage.artifact.begin, 2);
    assert_eq!(replacement.workers[0].stage.artifact.end, 4);
    assert_eq!(replacement.workers[0].next, None);
    assert_ne!(first.preparation_id, replacement.preparation_id);
    request.peers[1].healthy = false;
    assert!(prepare(&request, &registry, 1100).is_err());
}

#[test]
fn current_time_invalidates_stale_inputs_even_when_snapshot_claims_old_now() {
    let (request, registry) = fixture::inputs(1000);
    assert!(prepare(&request, &registry, 61_001).is_err());
    assert!(prepare(&request, &registry, 999).is_err());
    let bundle = prepare(&request, &registry, 1100).unwrap();
    validate(&bundle, &request, &registry, 61_000).unwrap();
    assert!(validate(&bundle, &request, &registry, 61_001).is_err());
    assert!(validate(&bundle, &request, &registry, 1099).is_err());
}

#[test]
fn expired_capacity_or_selected_link_cannot_be_hidden_by_fresh_range_timing() {
    let (mut request, registry) = fixture::inputs(1000);
    request.peers[0].capacity_observed_at_unix_ms = 1;
    assert!(prepare(&request, &registry, 61_002).is_err());
    let (mut request, registry) = fixture::inputs(1000);
    request.links.retain(|edge| edge.to != "leader");
    assert!(prepare(&request, &registry, 1100).is_err());
}

#[test]
fn rejects_ambiguous_or_mismatched_artifacts_identities_and_workloads() {
    let changes: &[fn(&mut DeploymentRegistry)] = &[
        |r| r.devices[1].artifacts.clear(),
        |r| r.devices[1].artifacts[0].begin = 2,
        |r| r.devices[1].artifacts[0].range_profile_id = "another-profile".into(),
        |r| {
            let duplicate = r.devices[1].artifacts[0].clone();
            r.devices[1].artifacts.push(duplicate)
        },
        |r| r.devices[1].endpoint_id = r.devices[0].endpoint_id,
        |r| r.devices[1].resource_id = r.devices[0].resource_id.clone(),
        |r| r.devices[1].runner_sha256 = "bad".into(),
        |r| r.devices[1].artifacts[0].artifact_sha256 = "bad".into(),
        |r| r.devices[1].artifacts[0].budget_bytes = 0,
        |r| r.devices[1].runner = "relative/runner".into(),
        |r| r.devices[1].key_path = "/key\n--debug".into(),
        |r| r.runtime_id = "another-build".into(),
        |r| r.model_sha256 = "ff".repeat(32),
        |r| r.leader.api = "0.0.0.0:8080".parse().unwrap(),
        |r| r.leader.upstream_port = 8080,
        |r| r.chat_template = json!({"jinja":{"path":"relative.jinja","sha256":"ab".repeat(32)}}),
    ];
    for (index, change) in changes.iter().enumerate() {
        let (request, mut registry) = fixture::inputs(1000);
        change(&mut registry);
        assert!(prepare(&request, &registry, 1100).is_err(), "case {index}");
    }
}

#[test]
fn native_destination_paths_can_belong_to_a_different_os() {
    let (request, mut registry) = fixture::inputs(1000);
    registry.devices[1].runner = "C:\\Program Files\\Layercake\\runner.exe".into();
    registry.devices[1].key_path = "C:\\Layercake\\identity.key".into();
    registry.devices[1].artifacts[0].model = "D:\\Models\\range.gguf".into();
    let bundle = prepare(&request, &registry, 1100).unwrap();
    assert_eq!(bundle.workers[0].stage.runner, registry.devices[1].runner);
}

#[test]
fn modifying_generated_content_or_original_inputs_invalidates_bundle() {
    let (request, registry) = fixture::inputs(1000);
    let bundle = prepare(&request, &registry, 1100).unwrap();
    validate(&bundle, &request, &registry, 1200).unwrap();
    let changes: &[fn(&mut PreparedRing)] = &[
        |b| b.activation_ready = true,
        |b| b.leader_manifest["stages"][0]["end"] = json!(2),
        |b| b.workers[0].argv.push("--debug".into()),
        |b| b.workers[0].next = None,
        |b| b.plan.predicted_decode_us = 1,
        |b| b.leader.runner_sha256 = "ff".repeat(32),
        |b| b.expires_at_unix_ms += 1,
    ];
    for change in changes {
        let mut changed = bundle.clone();
        change(&mut changed);
        assert!(validate(&changed, &request, &registry, 1200).is_err());
    }
    let mut changed_registry = registry.clone();
    changed_registry.devices[1].artifacts[0].gpu_layers = 99;
    assert!(validate(&bundle, &request, &changed_registry, 1200).is_err());
}

#[test]
fn new_bundle_directory_contains_real_manifest_and_is_never_overwritten() {
    let (request, registry) = fixture::inputs(1000);
    let bundle = prepare(&request, &registry, 1100).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("prepared");
    write_bundle(&bundle, &output).unwrap();
    ShardManifest::load(&output.join("leader-manifest.json")).unwrap();
    let saved: PreparedRing =
        serde_json::from_slice(&std::fs::read(output.join("bundle.json")).unwrap()).unwrap();
    assert_eq!(saved, bundle);
    assert!(output.join("worker-2.json").is_file());
    std::fs::write(output.join("operator-note"), b"keep").unwrap();
    assert!(write_bundle(&bundle, &output).is_err());
    assert_eq!(
        std::fs::read(output.join("operator-note")).unwrap(),
        b"keep"
    );
}

#[test]
fn cli_prepares_and_revalidates_bundle_without_binding_or_launching() {
    let now: u64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .try_into()
        .unwrap();
    let (request, registry) = fixture::inputs(now);
    let dir = tempfile::tempdir().unwrap();
    let profile = dir.path().join("profile.json");
    let registry_path = dir.path().join("registry.json");
    let output = dir.path().join("bundle");
    std::fs::write(&profile, serde_json::to_vec(&request).unwrap()).unwrap();
    std::fs::write(&registry_path, serde_json::to_vec(&registry).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_llm-cluster"))
        .arg("prepare-ring")
        .arg(&profile)
        .arg(&registry_path)
        .arg("--output-dir")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let status: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(status["activation_ready"], false);
    let result = Command::new(env!("CARGO_BIN_EXE_llm-cluster"))
        .arg("validate-ring")
        .arg(output.join("bundle.json"))
        .arg(&profile)
        .arg(&registry_path)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let mut stale = request;
    for peer in &mut stale.peers {
        peer.capacity_observed_at_unix_ms = 1;
    }
    std::fs::write(&profile, serde_json::to_vec(&stale).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_llm-cluster"))
        .arg("prepare-ring")
        .arg(&profile)
        .arg(&registry_path)
        .arg("--output-dir")
        .arg(dir.path().join("rejected"))
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!dir.path().join("rejected").exists());
}
