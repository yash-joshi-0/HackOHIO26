//! Policy checks over trusted local evidence, followed by the existing range search.
//! This checks completeness and consistency; it does not authenticate measurements.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, ensure};
use iroh::EndpointId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::measured_placement::{
    self, LinkObservation, MeasuredError, MeasuredPlacement, MeasuredRequest, MemoryBytes,
    PeerObservation, RangeObservation, Workload,
};
use crate::ring_deployment::{self, DeploymentRegistry, PreparedRing};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRequest {
    pub schema_version: u32,
    pub workload: Workload,
    pub origin_peer_id: String,
    pub min_peers: usize,
    pub policy: QualificationPolicy,
    pub peers: Vec<PeerAvailability>,
    pub ranges: Vec<RangeEvidence>,
    pub links: Vec<TransferEvidence>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationPolicy {
    pub trial_policy_version: String,
    pub trial_policy_declared_at_unix_ms: u64,
    pub range_max_age_ms: u64,
    pub capacity_max_age_ms: u64,
    pub link_max_age_ms: u64,
    pub min_sessions: usize,
    pub min_prefill_samples: usize,
    pub min_decode_samples: usize,
    pub min_transfer_samples: usize,
    pub min_session_service_us: u64,
    pub max_thermal: ThermalState,
    pub allow_battery: bool,
    pub transport: String,
    pub protocol: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionIdentity {
    pub runner_sha256: String,
    /// Hash of the pinned native dependency manifest.
    pub dependencies_sha256: String,
    /// Hash of the effective native configuration, including backend settings.
    pub configuration_sha256: String,
    pub effective_backend: String,
    pub backend_evidence_sha256: String,
    pub gpu_layers: u32,
    pub native_buffer_budget_bytes: u64,
    pub context_tokens: u32,
    pub sequence_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerSource {
    Ac,
    Battery,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CurrentConditions {
    pub power: PowerSource,
    pub thermal: ThermalState,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PeerAvailability {
    pub id: String,
    pub resource_id: String,
    pub endpoint_id: EndpointId,
    pub authorized: bool,
    pub compute_enabled: bool,
    pub healthy: bool,
    pub observed_at_unix_ms: u64,
    pub budget_bytes: MemoryBytes,
    pub available_bytes: MemoryBytes,
    pub reserved_bytes: MemoryBytes,
    pub execution: Option<ExecutionIdentity>,
    pub conditions: Option<CurrentConditions>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialOutcome {
    Passed,
    Failed,
    TimedOut,
    Cancelled,
    LimitExceeded,
    InsufficientDuration,
    CleanupFailed,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorrectnessEvidence {
    pub passed: bool,
    pub fixture_sha256: String,
    /// Independently retained validation of the reference, not just its byte hash.
    pub reference_validation_sha256: String,
    pub reference_runtime_id: String,
    pub reference_backend: String,
    pub checked_steps: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryScope {
    WholeContribution,
    GgmlBuffers,
    ProcessRss,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryEvidence {
    pub scope: MemoryScope,
    pub evidence_sha256: String,
    /// Explain attribution, disjoint physical pools, load/staging and process/driver coverage.
    pub accounting: String,
    pub load_peak_bytes: MemoryBytes,
    pub inference_peak_bytes: MemoryBytes,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConditionEvidence {
    pub power: PowerSource,
    pub worst_thermal: ThermalState,
    pub throttled: bool,
    pub monitoring_complete: bool,
    pub evidence_sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Prefill,
    Decode,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StepSample {
    pub phase: Phase,
    pub position: u32,
    pub tokens: u32,
    pub compute_us: u64,
    pub service_us: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SessionEvidence {
    pub load_us: u64,
    /// Complete state history in one fresh context, starting at position zero.
    pub steps: Vec<StepSample>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RangeEvidence {
    pub profile_id: String,
    pub trial_policy_sha256: String,
    pub peer_id: String,
    pub resource_id: String,
    pub report_sha256: String,
    pub artifact_sha256: String,
    pub workload: Workload,
    pub begin: usize,
    pub end: usize,
    pub measured_at_unix_ms: u64,
    pub outcome: TrialOutcome,
    pub cleanup_confirmed: bool,
    pub execution: Option<ExecutionIdentity>,
    pub correctness: Option<CorrectnessEvidence>,
    pub memory: Option<MemoryEvidence>,
    pub conditions: Option<ConditionEvidence>,
    pub requested_sessions: usize,
    pub sessions: Vec<SessionEvidence>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkPath {
    Direct,
    Relay,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TransferEvidence {
    pub profile_id: String,
    pub report_sha256: String,
    pub from: String,
    pub to: String,
    pub from_endpoint_id: EndpointId,
    pub to_endpoint_id: EndpointId,
    pub workload: Workload,
    pub payload_bytes: u64,
    pub measured_at_unix_ms: u64,
    pub transport: String,
    pub protocol: String,
    pub path: LinkPath,
    pub outcome: TrialOutcome,
    /// Whole directed payload transfers, excluding native execution time.
    pub transfer_us: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Exclusion {
    pub kind: String,
    pub id: String,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedPlan {
    pub schema_version: u32,
    pub advisory: bool,
    pub hardware_attested: bool,
    pub activation_ready: bool,
    pub input_sha256: String,
    pub evaluated_at_unix_ms: u64,
    pub valid_until_unix_ms: Option<u64>,
    pub objective: String,
    pub exclusions: Vec<Exclusion>,
    pub plan: Option<MeasuredPlacement>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedDeployment {
    pub qualification: QualifiedPlan,
    pub bundle: PreparedRing,
}

fn sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn named(value: &str) -> bool {
    !value.trim().is_empty() && value.len() <= 256 && !value.chars().any(char::is_control)
}

fn identity_valid(identity: &ExecutionIdentity) -> bool {
    sha(&identity.runner_sha256)
        && sha(&identity.dependencies_sha256)
        && sha(&identity.configuration_sha256)
        && named(&identity.effective_backend)
        && sha(&identity.backend_evidence_sha256)
        && identity.native_buffer_budget_bytes > 0
        && (32..=4096).contains(&identity.context_tokens)
        && identity.sequence_count == 1
}

fn fresh(time: u64, age: u64, now: u64) -> bool {
    time > 0 && time <= now && now - time <= age && time.checked_add(age).is_some()
}

fn require(reasons: &mut Vec<String>, condition: bool, reason: &str) {
    if !condition {
        reasons.push(reason.to_owned());
    }
}

fn exclude(exclusions: &mut Vec<Exclusion>, kind: &str, id: &str, reasons: Vec<String>) -> bool {
    if reasons.is_empty() {
        return false;
    }
    exclusions.push(Exclusion {
        kind: kind.into(),
        id: id.into(),
        reasons,
    });
    true
}

fn digest(value: &impl Serialize) -> anyhow::Result<String> {
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(value)?)))
}

/// Pins predeclared trial acceptance thresholds, independently of decision TTLs.
pub fn trial_policy_sha256(policy: &QualificationPolicy) -> anyhow::Result<String> {
    digest(&serde_json::json!({
        "version": policy.trial_policy_version,
        "declared_at_unix_ms": policy.trial_policy_declared_at_unix_ms,
        "min_sessions": policy.min_sessions,
        "min_prefill_samples": policy.min_prefill_samples,
        "min_decode_samples": policy.min_decode_samples,
        "min_session_service_us": policy.min_session_service_us,
        "max_thermal": policy.max_thermal,
        "allow_battery": policy.allow_battery,
    }))
}

fn peak(memory: &MemoryEvidence) -> MemoryBytes {
    MemoryBytes {
        ram: memory
            .load_peak_bytes
            .ram
            .max(memory.inference_peak_bytes.ram),
        vram: memory
            .load_peak_bytes
            .vram
            .max(memory.inference_peak_bytes.vram),
        shared: memory
            .load_peak_bytes
            .shared
            .max(memory.inference_peak_bytes.shared),
    }
}

fn p95(samples: &[u64]) -> u64 {
    let mut ordered = samples.to_vec();
    ordered.sort_unstable();
    ordered[(ordered.len() * 95).div_ceil(100) - 1]
}

struct Timings {
    decode: Vec<u64>,
    prefill_count: usize,
    steps: usize,
    longest_service_us: u64,
}

fn timings(range: &RangeEvidence) -> Option<Timings> {
    if range.sessions.is_empty()
        || range.sessions.len() > 64
        || range.sessions.len() != range.requested_sessions
    {
        return None;
    }
    let mut result = Timings {
        decode: vec![],
        prefill_count: 0,
        steps: 0,
        longest_service_us: 0,
    };
    for session in &range.sessions {
        if session.load_us == 0 || session.steps.is_empty() {
            return None;
        }
        let first = &range.sessions[0].steps;
        if session.steps.len() != first.len()
            || session.steps.iter().zip(first).any(|(step, reference)| {
                (step.phase, step.position, step.tokens)
                    != (reference.phase, reference.position, reference.tokens)
            })
        {
            return None;
        }
        let mut position = 0u32;
        let mut decoding = false;
        let mut had_prefill = false;
        let mut duration = 0u64;
        for sample in &session.steps {
            result.steps += 1;
            if result.steps > 65_536
                || sample.position != position
                || sample.tokens == 0
                || sample.compute_us == 0
                || sample.service_us == 0
                || sample.compute_us > sample.service_us
            {
                return None;
            }
            position = position.checked_add(sample.tokens)?;
            if position > range.workload.context_tokens {
                return None;
            }
            duration = duration.checked_add(sample.service_us)?;
            match sample.phase {
                Phase::Prefill if !decoding => {
                    had_prefill = true;
                    result.prefill_count += 1;
                }
                Phase::Decode if had_prefill && sample.tokens == 1 => {
                    decoding = true;
                    result.decode.push(sample.service_us);
                }
                _ => return None,
            }
        }
        if !decoding || position != range.workload.context_tokens {
            return None;
        }
        result.longest_service_us = result.longest_service_us.max(duration);
    }
    Some(result)
}

struct Compiled {
    request: MeasuredRequest,
    exclusions: Vec<Exclusion>,
    peer_expiry: HashMap<String, u64>,
    range_expiry: HashMap<String, u64>,
    link_expiry: HashMap<String, u64>,
}

fn compile(input: &EvidenceRequest, now: u64) -> anyhow::Result<Compiled> {
    let policy = &input.policy;
    let trial_policy_sha = trial_policy_sha256(policy)?;
    ensure!(
        input.schema_version == 2 && now > 0,
        "unsupported evidence schema or current time"
    );
    ensure!(
        policy.range_max_age_ms > 0
            && named(&policy.trial_policy_version)
            && policy.trial_policy_declared_at_unix_ms > 0
            && policy.trial_policy_declared_at_unix_ms <= now
            && policy.capacity_max_age_ms > 0
            && policy.link_max_age_ms > 0
            && (1..=64).contains(&policy.min_sessions)
            && (1..=65_536).contains(&policy.min_prefill_samples)
            && (1..=65_536).contains(&policy.min_decode_samples)
            && (1..=65_536).contains(&policy.min_transfer_samples)
            && policy.min_session_service_us > 0
            && policy.max_thermal <= ThermalState::Fair
            && policy.transport == "iroh"
            && policy.protocol.as_bytes() == crate::shard::ring::RING_ALPN,
        "invalid qualification policy"
    );
    ensure!(
        (32..=4096).contains(&input.workload.context_tokens) && input.workload.batch_tokens == 1,
        "unsupported ring context or decode batch"
    );
    ensure!(
        input.peers.len() <= 128 && input.ranges.len() <= 4096 && input.links.len() <= 4096,
        "evidence input exceeds record limits"
    );
    let mut ids = HashSet::new();
    let mut resources = HashSet::new();
    let mut endpoints = HashSet::new();
    for peer in &input.peers {
        ensure!(
            named(&peer.id)
                && ids.insert(peer.id.as_str())
                && named(&peer.resource_id)
                && resources.insert(peer.resource_id.as_str())
                && endpoints.insert(peer.endpoint_id),
            "peer, physical resource and endpoint identities must be unique"
        );
    }
    ensure!(
        ids.contains(input.origin_peer_id.as_str()),
        "origin missing from current peer availability"
    );
    let mut profile_ids = HashSet::new();
    for id in input
        .ranges
        .iter()
        .map(|r| &r.profile_id)
        .chain(input.links.iter().map(|l| &l.profile_id))
    {
        ensure!(
            named(id) && profile_ids.insert(id),
            "profile IDs must be nonempty and globally unique"
        );
    }
    let mut result = Compiled {
        request: MeasuredRequest {
            schema_version: 1,
            workload: input.workload.clone(),
            origin_peer_id: input.origin_peer_id.clone(),
            observed_at_unix_ms: now,
            // Each kind is checked separately below. Preserve original timestamps.
            max_profile_age_ms: policy
                .range_max_age_ms
                .max(policy.capacity_max_age_ms)
                .max(policy.link_max_age_ms),
            min_peers: input.min_peers,
            peers: vec![],
            links: vec![],
        },
        exclusions: vec![],
        peer_expiry: HashMap::new(),
        range_expiry: HashMap::new(),
        link_expiry: HashMap::new(),
    };
    let mut available = HashMap::new();
    for peer in &input.peers {
        let mut reasons = vec![];
        require(
            &mut reasons,
            peer.authorized && peer.compute_enabled && peer.healthy,
            "compute_not_authorized_or_healthy",
        );
        require(
            &mut reasons,
            fresh(peer.observed_at_unix_ms, policy.capacity_max_age_ms, now),
            "capacity_stale_or_future",
        );
        require(
            &mut reasons,
            peer.execution.as_ref().is_some_and(identity_valid),
            "effective_execution_identity_missing_or_invalid",
        );
        require(
            &mut reasons,
            peer.conditions.as_ref().is_some_and(|c| {
                c.thermal <= policy.max_thermal
                    && (policy.allow_battery || c.power == PowerSource::Ac)
            }),
            "current_conditions_missing_or_disallowed",
        );
        let eligible = !exclude(&mut result.exclusions, "peer", &peer.id, reasons);
        if eligible {
            available.insert(peer.id.as_str(), peer);
            result.peer_expiry.insert(
                peer.id.clone(),
                peer.observed_at_unix_ms + policy.capacity_max_age_ms,
            );
        }
        result.request.peers.push(PeerObservation {
            id: peer.id.clone(),
            authorized: eligible,
            healthy: eligible,
            capacity_observed_at_unix_ms: peer.observed_at_unix_ms,
            budget_bytes: peer.budget_bytes.clone(),
            available_bytes: peer.available_bytes.clone(),
            reserved_bytes: peer.reserved_bytes.clone(),
            ranges: vec![],
        });
    }
    for range in &input.ranges {
        let mut reasons = vec![];
        let peer = available.get(range.peer_id.as_str());
        require(
            &mut reasons,
            range.trial_policy_sha256 == trial_policy_sha
                && policy.trial_policy_declared_at_unix_ms <= range.measured_at_unix_ms,
            "trial_policy_missing_changed_or_declared_after_measurement",
        );
        require(&mut reasons, peer.is_some(), "peer_unavailable");
        require(
            &mut reasons,
            peer.is_some_and(|p| p.resource_id == range.resource_id),
            "physical_resource_changed",
        );
        require(
            &mut reasons,
            range.workload == input.workload,
            "workload_mismatch",
        );
        require(
            &mut reasons,
            range.begin < range.end && range.end <= input.workload.layers,
            "invalid_range",
        );
        require(
            &mut reasons,
            sha(&range.report_sha256) && sha(&range.artifact_sha256),
            "artifact_or_report_provenance_missing",
        );
        require(
            &mut reasons,
            fresh(range.measured_at_unix_ms, policy.range_max_age_ms, now),
            "range_stale_or_future",
        );
        require(
            &mut reasons,
            range.outcome == TrialOutcome::Passed && range.cleanup_confirmed,
            "trial_not_passed_or_cleanup_unconfirmed",
        );
        require(
            &mut reasons,
            range.execution.as_ref().is_some_and(|e| {
                identity_valid(e) && e.context_tokens >= input.workload.context_tokens
            }) && peer.is_some_and(|p| p.execution == range.execution),
            "execution_identity_missing_or_changed",
        );
        let samples = timings(range);
        require(
            &mut reasons,
            samples.is_some(),
            "invalid_or_incomplete_state_history",
        );
        require(
            &mut reasons,
            samples.as_ref().is_some_and(|s| {
                range.sessions.len() >= policy.min_sessions
                    && s.prefill_count >= policy.min_prefill_samples
                    && s.decode.len() >= policy.min_decode_samples
                    && s.longest_service_us >= policy.min_session_service_us
            }),
            "insufficient_samples_or_sustained_session",
        );
        require(
            &mut reasons,
            range.correctness.as_ref().is_some_and(|c| {
                c.passed
                    && sha(&c.fixture_sha256)
                    && sha(&c.reference_validation_sha256)
                    && named(&c.reference_runtime_id)
                    && named(&c.reference_backend)
                    && samples.as_ref().is_some_and(|s| s.steps == c.checked_steps)
            }),
            "correctness_or_validated_reference_missing",
        );
        require(
            &mut reasons,
            range.memory.as_ref().is_some_and(|m| {
                m.scope == MemoryScope::WholeContribution
                    && sha(&m.evidence_sha256)
                    && named(&m.accounting)
                    && {
                        let peak = peak(m);
                        peak.ram > 0 || peak.vram > 0 || peak.shared > 0
                    }
            }),
            "complete_physical_pool_peaks_missing",
        );
        if let (Some(memory), Some(peer)) = (&range.memory, peer) {
            let peak = peak(memory);
            let fits = [
                (
                    peak.ram,
                    peer.reserved_bytes.ram,
                    peer.budget_bytes.ram,
                    peer.available_bytes.ram,
                ),
                (
                    peak.vram,
                    peer.reserved_bytes.vram,
                    peer.budget_bytes.vram,
                    peer.available_bytes.vram,
                ),
                (
                    peak.shared,
                    peer.reserved_bytes.shared,
                    peer.budget_bytes.shared,
                    peer.available_bytes.shared,
                ),
            ]
            .iter()
            .all(|&(peak, reserved, budget, available)| {
                peak.checked_add(reserved)
                    .is_some_and(|total| total <= budget && total <= available)
            });
            require(&mut reasons, fits, "physical_pool_budget_exceeded");
        }
        require(
            &mut reasons,
            range.conditions.as_ref().is_some_and(|c| {
                c.monitoring_complete
                    && !c.throttled
                    && c.worst_thermal <= policy.max_thermal
                    && sha(&c.evidence_sha256)
                    && (policy.allow_battery || c.power == PowerSource::Ac)
                    && peer
                        .and_then(|p| p.conditions.as_ref())
                        .is_some_and(|current| current.power == c.power)
            }),
            "sampled_conditions_missing_or_incompatible",
        );
        if exclude(&mut result.exclusions, "range", &range.profile_id, reasons) {
            continue;
        }
        let observation = RangeObservation {
            begin: range.begin,
            end: range.end,
            decode_us: p95(&samples.unwrap().decode),
            peak_bytes: peak(range.memory.as_ref().unwrap()),
            profile_id: range.profile_id.clone(),
            measured_at_unix_ms: range.measured_at_unix_ms,
            workload: range.workload.clone(),
        };
        result
            .request
            .peers
            .iter_mut()
            .find(|p| p.id == range.peer_id)
            .unwrap()
            .ranges
            .push(observation);
        result.range_expiry.insert(
            range.profile_id.clone(),
            range.measured_at_unix_ms + policy.range_max_age_ms,
        );
    }
    // Devices with no eligible range cannot contribute to this decision. Keep
    // them in the identity snapshot without consuming the search's peer limit.
    for peer in &mut result.request.peers {
        if peer.authorized && peer.ranges.is_empty() {
            peer.authorized = false;
            peer.healthy = false;
            available.remove(peer.id.as_str());
            exclude(
                &mut result.exclusions,
                "peer",
                &peer.id,
                vec!["no_qualified_ranges".into()],
            );
        }
    }
    for edge in &input.links {
        let mut reasons = vec![];
        require(
            &mut reasons,
            edge.from != edge.to
                && available.contains_key(edge.from.as_str())
                && available.contains_key(edge.to.as_str()),
            "link_endpoint_unavailable",
        );
        require(
            &mut reasons,
            edge.workload == input.workload
                && edge.payload_bytes > 0
                && [
                    input.workload.activation_payload_bytes,
                    input.workload.control_payload_bytes,
                ]
                .contains(&edge.payload_bytes),
            "link_workload_or_payload_mismatch",
        );
        require(
            &mut reasons,
            available
                .get(edge.from.as_str())
                .is_some_and(|p| p.endpoint_id == edge.from_endpoint_id)
                && available
                    .get(edge.to.as_str())
                    .is_some_and(|p| p.endpoint_id == edge.to_endpoint_id),
            "link_endpoint_identity_changed",
        );
        require(
            &mut reasons,
            edge.transport == policy.transport && edge.protocol == policy.protocol,
            "link_transport_or_protocol_mismatch",
        );
        require(
            &mut reasons,
            edge.outcome == TrialOutcome::Passed && sha(&edge.report_sha256),
            "link_outcome_or_provenance_invalid",
        );
        require(
            &mut reasons,
            fresh(edge.measured_at_unix_ms, policy.link_max_age_ms, now),
            "link_stale_or_future",
        );
        require(
            &mut reasons,
            edge.transfer_us.len() >= policy.min_transfer_samples
                && edge.transfer_us.len() <= 65_536
                && edge.transfer_us.iter().all(|&t| t > 0),
            "invalid_transfer_samples",
        );
        if exclude(&mut result.exclusions, "link", &edge.profile_id, reasons) {
            continue;
        }
        result.request.links.push(LinkObservation {
            from: edge.from.clone(),
            to: edge.to.clone(),
            payload_bytes: edge.payload_bytes,
            transfer_us: p95(&edge.transfer_us),
            profile_id: edge.profile_id.clone(),
            measured_at_unix_ms: edge.measured_at_unix_ms,
            path: match edge.path {
                LinkPath::Direct => "direct",
                LinkPath::Relay => "relay",
            }
            .into(),
            workload: edge.workload.clone(),
        });
        result.link_expiry.insert(
            edge.profile_id.clone(),
            edge.measured_at_unix_ms + policy.link_max_age_ms,
        );
    }
    result
        .exclusions
        .sort_by(|a, b| (&a.kind, &a.id).cmp(&(&b.kind, &b.id)));
    Ok(result)
}

fn assess(input: &EvidenceRequest, compiled: &Compiled, now: u64) -> anyhow::Result<QualifiedPlan> {
    let plan = match measured_placement::plan_ring(&compiled.request) {
        Ok(plan) => Some(plan),
        Err(MeasuredError::NoFeasiblePlacement) => None,
        Err(error) => anyhow::bail!("qualified placement rejected: {error:?}"),
    };
    let valid_until_unix_ms = plan.as_ref().map(|plan| {
        plan.assignments
            .iter()
            .flat_map(|a| {
                [
                    compiled.peer_expiry[&a.peer_id],
                    compiled.range_expiry[&a.range_profile_id],
                ]
            })
            .chain(
                plan.link_profile_ids
                    .iter()
                    .map(|id| compiled.link_expiry[id]),
            )
            .min()
            .unwrap()
    });
    Ok(QualifiedPlan {
        schema_version: 2,
        advisory: true,
        hardware_attested: false,
        activation_ready: false,
        input_sha256: digest(input)?,
        evaluated_at_unix_ms: now,
        valid_until_unix_ms,
        objective: "sum_whole_range_service_p95_and_directed_transfer_p95".into(),
        exclusions: compiled.exclusions.clone(),
        plan,
    })
}

pub fn plan_ring(input: &EvidenceRequest, now_unix_ms: u64) -> anyhow::Result<QualifiedPlan> {
    let compiled = compile(input, now_unix_ms)?;
    assess(input, &compiled, now_unix_ms)
}

/// Bind selected range evidence to the actual registered artifact and native settings.
pub fn prepare_ring(
    input: &EvidenceRequest,
    registry: &DeploymentRegistry,
    now_unix_ms: u64,
) -> anyhow::Result<QualifiedDeployment> {
    let compiled = compile(input, now_unix_ms)?;
    let qualification = assess(input, &compiled, now_unix_ms)?;
    let plan = qualification
        .plan
        .as_ref()
        .context("no feasible placement from qualified evidence")?;
    let bundle = ring_deployment::prepare(&compiled.request, registry, now_unix_ms)?;
    for stage in std::iter::once(&bundle.leader).chain(bundle.workers.iter().map(|w| &w.stage)) {
        let evidence = input
            .ranges
            .iter()
            .find(|r| r.profile_id == stage.artifact.range_profile_id)
            .unwrap();
        let peer = input.peers.iter().find(|p| p.id == stage.peer_id).unwrap();
        let execution = evidence.execution.as_ref().unwrap();
        ensure!(
            stage.endpoint_id == peer.endpoint_id && stage.resource_id == peer.resource_id,
            "registry peer/resource association differs from current availability"
        );
        ensure!(
            stage.artifact.artifact_sha256 == evidence.artifact_sha256
                && stage.runner_sha256 == execution.runner_sha256
                && stage.artifact.gpu_layers == execution.gpu_layers
                && stage.artifact.budget_bytes == execution.native_buffer_budget_bytes,
            "selected artifact or native build/configuration differs from measured evidence"
        );
    }
    ensure!(
        &bundle.plan == plan,
        "qualified and prepared placements differ"
    );
    Ok(QualifiedDeployment {
        qualification,
        bundle,
    })
}

/// Refresh the input externally first. Changed availability/evidence invalidates the envelope.
pub fn validate_deployment(
    saved: &QualifiedDeployment,
    input: &EvidenceRequest,
    registry: &DeploymentRegistry,
    now_unix_ms: u64,
) -> anyhow::Result<()> {
    let at = saved.qualification.evaluated_at_unix_ms;
    let expiry = saved
        .qualification
        .valid_until_unix_ms
        .context("deployment has no qualification expiry")?;
    ensure!(
        now_unix_ms >= at && now_unix_ms <= expiry,
        "qualified deployment is stale or from the future"
    );
    let expected = prepare_ring(input, registry, at)?;
    ensure!(
        &expected == saved,
        "qualified deployment or current inputs changed"
    );
    Ok(())
}
