//! Prepare the existing ring runtime's inputs from measured placement.
//!
//! The registry is trusted operator input, not remotely supplied commands. Preparing
//! a bundle neither qualifies diagnostic evidence nor reserves or starts a device.

use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use iroh::EndpointId;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::measured_placement::{self, MeasuredPlacement, MeasuredRequest};
use crate::shard::manifest::{ChatTemplate, ShardManifest};
use crate::shard::ring::RingWorkerConfig;

/// A trusted inventory of already selected native builds and exported artifacts.
/// All paths are destination-host paths, not paths on the machine preparing it.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentRegistry {
    pub schema_version: u32,
    pub model_sha256: String,
    pub runtime_id: String,
    pub chat_template: Value,
    pub max_tokens: u32,
    pub leader: LeaderSettings,
    pub devices: Vec<DeviceBinding>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LeaderSettings {
    pub cluster: String,
    /// Where the generated leader-manifest.json will be staged on the leader.
    pub manifest_path: String,
    pub api: SocketAddr,
    pub upstream_port: u16,
    pub worker_wait_seconds: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceBinding {
    pub peer_id: String,
    /// Physical device/resource owner. Multiple endpoints cannot multiply its budget.
    pub resource_id: String,
    pub endpoint_id: EndpointId,
    pub key_path: String,
    pub runner: String,
    pub runner_sha256: String,
    pub artifacts: Vec<ArtifactBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactBinding {
    pub range_profile_id: String,
    pub begin: usize,
    pub end: usize,
    /// Catalog/export role assertion; destination tensor verification is still required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<ArtifactRole>,
    pub model: String,
    pub artifact_sha256: String,
    pub gpu_layers: u32,
    /// Native GGML buffer limit, NOT total RAM/VRAM or the planner's pool peak.
    pub budget_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactRole {
    First,
    Middle,
    Final,
    Full,
}

impl ArtifactRole {
    fn for_range(begin: usize, end: usize, layers: usize) -> Self {
        match (begin == 0, end == layers) {
            (true, true) => Self::Full,
            (true, false) => Self::First,
            (false, true) => Self::Final,
            (false, false) => Self::Middle,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedStage {
    pub peer_id: String,
    pub resource_id: String,
    pub endpoint_id: EndpointId,
    pub key_path: String,
    pub runner: String,
    pub runner_sha256: String,
    pub artifact: ArtifactBinding,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedWorker {
    pub stage: PreparedStage,
    pub context: u32,
    pub source_sha256: String,
    pub upstream: EndpointId,
    pub origin: EndpointId,
    pub next: Option<EndpointId>,
    /// Arguments to the existing llm-cluster binary, never a shell command.
    pub argv: Vec<String>,
}

impl PreparedWorker {
    /// The actual M-27 worker configuration. Hosts must validate the whole bundle
    /// and local files before starting it, and own admission for its full lifetime.
    pub fn config(&self) -> anyhow::Result<RingWorkerConfig> {
        Ok(RingWorkerConfig {
            runner: self.stage.runner.clone().into(),
            model: self.stage.artifact.model.clone().into(),
            begin: self.stage.artifact.begin.try_into()?,
            end: self.stage.artifact.end.try_into()?,
            context: self.context,
            gpu_layers: self.stage.artifact.gpu_layers,
            budget_bytes: self.stage.artifact.budget_bytes,
            expected_sha256: self.stage.artifact.artifact_sha256.clone(),
            source_sha256: Some(self.source_sha256.clone()),
            upstream: self.upstream,
            origin: self.origin,
            next: self.next,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedRing {
    pub schema_version: u32,
    pub experimental: bool,
    /// Preparing files does not certify evidence, staging or reservations.
    pub activation_ready: bool,
    pub preparation_id: String,
    pub request_sha256: String,
    pub registry_sha256: String,
    pub expires_at_unix_ms: u64,
    pub plan: MeasuredPlacement,
    pub leader: PreparedStage,
    pub leader_manifest: Value,
    pub leader_argv: Vec<String>,
    pub workers: Vec<PreparedWorker>,
    pub required_before_activation: Vec<String>,
}

/// Recompute the constrained placement using the current time, then bind each
/// selected range to exactly one artifact. Does not read remote paths or launch.
pub fn prepare(
    request: &MeasuredRequest,
    registry: &DeploymentRegistry,
    now_unix_ms: u64,
) -> anyhow::Result<PreparedRing> {
    ensure!(now_unix_ms > 0, "current time must be positive");
    ensure!(
        request.observed_at_unix_ms <= now_unix_ms,
        "profile snapshot is from the future"
    );
    let mut current = request.clone();
    current.observed_at_unix_ms = now_unix_ms;
    let plan = measured_placement::plan_ring(&current)
        .map_err(|error| anyhow::anyhow!("ring placement failed: {error:?}"))?;
    bind_plan(&current, registry, plan)
}

fn bind_plan(
    request: &MeasuredRequest,
    registry: &DeploymentRegistry,
    plan: MeasuredPlacement,
) -> anyhow::Result<PreparedRing> {
    validate_registry(request, registry)?;
    let inventory: HashMap<_, _> = registry
        .devices
        .iter()
        .map(|device| (device.peer_id.as_str(), device))
        .collect();
    let mut selected = Vec::new();
    let mut resources = HashSet::new();
    let mut earliest = request.observed_at_unix_ms;
    for assignment in &plan.assignments {
        let device = inventory
            .get(assignment.peer_id.as_str())
            .with_context(|| format!("no device binding for {}", assignment.peer_id))?;
        ensure!(
            resources.insert(device.resource_id.as_str()),
            "selected endpoints share physical resource {}",
            device.resource_id
        );
        let mut matches = device.artifacts.iter().filter(|artifact| {
            artifact.range_profile_id == assignment.range_profile_id
                && artifact.begin == assignment.layers.start
                && artifact.end == assignment.layers.end
        });
        let artifact = matches.next().with_context(|| {
            format!(
                "no exact artifact binding for {} profile {} range {}..{}",
                assignment.peer_id,
                assignment.range_profile_id,
                assignment.layers.start,
                assignment.layers.end
            )
        })?;
        ensure!(matches.next().is_none(), "ambiguous artifact binding");
        selected.push(PreparedStage {
            peer_id: device.peer_id.clone(),
            resource_id: device.resource_id.clone(),
            endpoint_id: device.endpoint_id,
            key_path: device.key_path.clone(),
            runner: device.runner.clone(),
            runner_sha256: device.runner_sha256.clone(),
            artifact: artifact.clone(),
        });
        let peer = request
            .peers
            .iter()
            .find(|peer| peer.id == assignment.peer_id)
            .context("selected peer is missing from measured request")?;
        earliest = earliest.min(peer.capacity_observed_at_unix_ms);
        let range = peer
            .ranges
            .iter()
            .find(|range| {
                range.profile_id == assignment.range_profile_id
                    && range.begin == assignment.layers.start
                    && range.end == assignment.layers.end
                    && range.workload == request.workload
            })
            .context("selected range evidence is missing")?;
        earliest = earliest.min(range.measured_at_unix_ms);
    }
    // Require unique profile references so an ID names one piece of evidence.
    // The planner already checks exact ranges and directed payload keys.
    for id in &plan.link_profile_ids {
        let edge = request
            .links
            .iter()
            .find(|edge| &edge.profile_id == id)
            .context("selected link evidence is missing")?;
        earliest = earliest.min(edge.measured_at_unix_ms);
    }
    let expires_at_unix_ms = earliest
        .checked_add(request.max_profile_age_ms)
        .context("profile expiration overflows")?;
    ensure!(
        expires_at_unix_ms >= request.observed_at_unix_ms,
        "selected observations have expired"
    );
    let (leader, remotes) = selected.split_first().context("empty placement")?;
    ensure!(
        leader.peer_id == request.origin_peer_id && !remotes.is_empty(),
        "ring requires leader-local stage zero and at least one remote"
    );
    let mut stages = vec![json!({
        "begin": leader.artifact.begin,
        "end": leader.artifact.end,
        "local": {
            "model": leader.artifact.model,
            "gpu_layers": leader.artifact.gpu_layers,
            "budget_bytes": leader.artifact.budget_bytes,
            "artifact_sha256": leader.artifact.artifact_sha256,
        }
    })];
    stages.extend(remotes.iter().map(|stage| {
        json!({
            "begin": stage.artifact.begin, "end": stage.artifact.end,
            "remote": {"worker": stage.endpoint_id}
        })
    }));
    let leader_manifest = json!({
        "runner": leader.runner,
        "model_sha256": request.workload.model_sha256,
        "layers": request.workload.layers,
        "context": request.workload.context_tokens,
        "max_tokens": registry.max_tokens,
        "chat_template": registry.chat_template,
        "execution": "ring",
        "stages": stages,
    });
    let manifest: ShardManifest = serde_json::from_value(leader_manifest.clone())?;
    manifest
        .validate()
        .context("generated ring manifest is invalid")?;
    let mut workers = Vec::new();
    for (index, stage) in remotes.iter().enumerate() {
        let upstream = selected[index].endpoint_id;
        let next = remotes.get(index + 1).map(|stage| stage.endpoint_id);
        let mut argv = vec![
            "--key".into(),
            stage.key_path.clone(),
            "shard-ring-worker".into(),
            "--runner".into(),
            stage.runner.clone(),
            "--model".into(),
            stage.artifact.model.clone(),
            "--range".into(),
            format!("{}:{}", stage.artifact.begin, stage.artifact.end),
            "--context".into(),
            request.workload.context_tokens.to_string(),
            "--gpu-layers".into(),
            stage.artifact.gpu_layers.to_string(),
            "--budget-bytes".into(),
            stage.artifact.budget_bytes.to_string(),
            "--expected-sha256".into(),
            stage.artifact.artifact_sha256.clone(),
            "--source-sha256".into(),
            request.workload.model_sha256.clone(),
            "--upstream".into(),
            upstream.to_string(),
            "--origin".into(),
            leader.endpoint_id.to_string(),
        ];
        if let Some(next) = next {
            argv.extend(["--next".into(), next.to_string()]);
        }
        let worker = PreparedWorker {
            stage: stage.clone(),
            context: request.workload.context_tokens,
            source_sha256: request.workload.model_sha256.clone(),
            upstream,
            origin: leader.endpoint_id,
            next,
            argv,
        };
        worker.config()?;
        workers.push(worker);
    }
    let leader_argv = vec![
        "--key".into(),
        leader.key_path.clone(),
        "--cluster".into(),
        registry.leader.cluster.clone(),
        "leader".into(),
        "--backend".into(),
        "shard".into(),
        "--shard-manifest".into(),
        registry.leader.manifest_path.clone(),
        "--api".into(),
        registry.leader.api.to_string(),
        "--llama-port".into(),
        registry.leader.upstream_port.to_string(),
        "--worker-wait".into(),
        format!("{}s", registry.leader.worker_wait_seconds),
    ];
    let request_sha256 = digest(request)?;
    let registry_sha256 = digest(registry)?;
    let preparation_id = digest(&(&request_sha256, &registry_sha256, &plan))?;
    Ok(PreparedRing {
        schema_version: 1,
        experimental: true,
        activation_ready: false,
        preparation_id,
        request_sha256,
        registry_sha256,
        expires_at_unix_ms,
        plan,
        leader: leader.clone(),
        leader_manifest,
        leader_argv,
        workers,
        required_before_activation: [
            "qualified correctness, backend, workload, memory, thermal and link evidence",
            "current compute opt-in, usable capacity and exclusive deployment ownership",
            "verified local identity, native build, dependencies and staged artifact on each peer",
            "all-worker readiness and end-to-end generation validation",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
    })
}

/// Verify all generated content against its original observations/registry, and
/// reject expiration at the caller's current time. Refresh changed inputs by
/// preparing a new bundle rather than editing generated commands.
pub fn validate(
    bundle: &PreparedRing,
    request: &MeasuredRequest,
    registry: &DeploymentRegistry,
    now_unix_ms: u64,
) -> anyhow::Result<()> {
    ensure!(
        now_unix_ms >= bundle.plan.planned_at_unix_ms && now_unix_ms <= bundle.expires_at_unix_ms,
        "bundle is expired or from the future; refresh observations and prepare again"
    );
    let expected = prepare(request, registry, bundle.plan.planned_at_unix_ms)?;
    ensure!(
        &expected == bundle,
        "bundle differs from its measured inputs or registry"
    );
    Ok(())
}

/// Writes a new directory, with bundle.json last as its completion marker.
/// Existing output is never overwritten. An I/O error removes only files this
/// operation created, leaving no completion marker for a partially written bundle.
pub fn write_bundle(bundle: &PreparedRing, output: &Path) -> anyhow::Result<()> {
    fs::create_dir(output)
        .with_context(|| format!("create new bundle directory {}", output.display()))?;
    let mut created: Vec<PathBuf> = Vec::new();
    let result = (|| {
        let mut write = |name: &str, value: &Value| -> anyhow::Result<()> {
            let path = output.join(name);
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            created.push(path);
            serde_json::to_writer_pretty(&mut file, value)?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            Ok(())
        };
        write("leader-manifest.json", &bundle.leader_manifest)?;
        for (index, worker) in bundle.workers.iter().enumerate() {
            write(
                &format!("worker-{}.json", index + 1),
                &serde_json::to_value(worker)?,
            )?;
        }
        write("bundle.json", &serde_json::to_value(bundle)?)?;
        Ok(())
    })();
    if result.is_err() {
        for path in created.iter().rev() {
            let _ = fs::remove_file(path);
        }
        let _ = fs::remove_dir(output);
    }
    result
}

fn validate_registry(
    request: &MeasuredRequest,
    registry: &DeploymentRegistry,
) -> anyhow::Result<()> {
    ensure!(
        registry.schema_version == 1,
        "unsupported deployment registry version"
    );
    ensure!(
        registry.model_sha256 == request.workload.model_sha256
            && registry.runtime_id == request.workload.runtime_id,
        "registry model/runtime differs from measured workload"
    );
    ensure!(
        request.workload.batch_tokens == 1,
        "ring preparation requires single-token decode evidence"
    );
    ensure!(
        registry.max_tokens > 0 && registry.max_tokens <= request.workload.context_tokens,
        "max_tokens must be positive and fit context"
    );
    ensure!(
        registry.leader.api.ip().is_loopback() && registry.leader.api.port() != 0,
        "leader API must use a nonzero loopback port"
    );
    ensure!(
        registry.leader.upstream_port > 0
            && registry.leader.upstream_port != registry.leader.api.port(),
        "leader API and upstream ports must be nonzero and different"
    );
    ensure!(
        (1..=3600).contains(&registry.leader.worker_wait_seconds),
        "worker wait must be 1..=3600 seconds"
    );
    text_field(&registry.leader.cluster, "cluster")?;
    absolute_path(&registry.leader.manifest_path, "leader manifest path")?;
    let template: ChatTemplate =
        serde_json::from_value(registry.chat_template.clone()).context("invalid chat template")?;
    if let ChatTemplate::Jinja(template) = template {
        absolute_path(
            template
                .jinja
                .path
                .to_str()
                .context("template path is not UTF-8")?,
            "template path",
        )?;
        sha256(&template.jinja.sha256, "template hash")?;
    }
    let mut peer_ids = HashSet::new();
    let mut endpoint_ids = HashSet::new();
    for device in &registry.devices {
        text_field(&device.peer_id, "peer ID")?;
        text_field(&device.resource_id, "resource ID")?;
        ensure!(
            peer_ids.insert(&device.peer_id),
            "duplicate device binding for {}",
            device.peer_id
        );
        ensure!(
            endpoint_ids.insert(device.endpoint_id),
            "duplicate endpoint identity in registry"
        );
        ensure!(
            request.peers.iter().any(|peer| peer.id == device.peer_id),
            "registry references unknown peer {}",
            device.peer_id
        );
        absolute_path(&device.key_path, "key path")?;
        absolute_path(&device.runner, "runner path")?;
        sha256(&device.runner_sha256, "runner hash")?;
        let mut profiles = HashSet::new();
        for artifact in &device.artifacts {
            text_field(&artifact.range_profile_id, "range profile ID")?;
            ensure!(
                profiles.insert(&artifact.range_profile_id),
                "duplicate artifact profile binding"
            );
            ensure!(
                artifact.begin < artifact.end && artifact.end <= request.workload.layers,
                "invalid artifact range"
            );
            if let Some(role) = artifact.role {
                ensure!(
                    role == ArtifactRole::for_range(
                        artifact.begin,
                        artifact.end,
                        request.workload.layers
                    ),
                    "artifact role does not match its range and stage position"
                );
            }
            absolute_path(&artifact.model, "artifact path")?;
            sha256(&artifact.artifact_sha256, "artifact hash")?;
            ensure!(
                artifact.budget_bytes > 0,
                "native buffer budget must be positive"
            );
        }
    }
    let mut ids = HashSet::new();
    for peer in &request.peers {
        for range in &peer.ranges {
            ensure!(
                ids.insert(range.profile_id.as_str()),
                "range profile IDs must be unique"
            );
        }
    }
    for edge in &request.links {
        ensure!(
            ids.insert(edge.profile_id.as_str()),
            "link profile IDs must be unique and distinct from ranges"
        );
    }
    Ok(())
}

fn text_field(value: &str, field: &str) -> anyhow::Result<()> {
    ensure!(
        !value.trim().is_empty() && value.len() <= 4096 && !value.chars().any(char::is_control),
        "{field} must be nonempty bounded text without control characters"
    );
    Ok(())
}

fn absolute_path(value: &str, field: &str) -> anyhow::Result<()> {
    text_field(value, field)?;
    let bytes = value.as_bytes();
    // Registry paths belong to their destination, which may use another OS.
    let windows_drive = bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'\\' | b'/');
    ensure!(
        value.starts_with('/') || value.starts_with("\\\\") || windows_drive,
        "{field} must be absolute on its destination host"
    );
    Ok(())
}

fn sha256(value: &str, field: &str) -> anyhow::Result<()> {
    ensure!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{field} must be 64 hex digits"
    );
    Ok(())
}

fn digest(value: &impl Serialize) -> anyhow::Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(value)?)))
}
