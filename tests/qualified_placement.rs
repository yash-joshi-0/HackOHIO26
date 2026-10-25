use llm_cluster::qualified_placement::*;
use llm_cluster::ring_deployment::DeploymentRegistry;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[path = "common/ring_fixture.rs"]
mod fixture;

fn policy(now: u64) -> QualificationPolicy {
    QualificationPolicy {
        trial_policy_version: "synthetic-policy-v1".into(),
        trial_policy_declared_at_unix_ms: now - 1,
        range_max_age_ms: 60_000,
        capacity_max_age_ms: 1000,
        link_max_age_ms: 500,
        min_sessions: 2,
        min_prefill_samples: 2,
        min_decode_samples: 6,
        min_transfer_samples: 2,
        min_session_service_us: 20,
        max_thermal: ThermalState::Fair,
        allow_battery: false,
        transport: "iroh".into(),
        protocol: "meshthing/shard-ring/1".into(),
    }
}

fn inputs(now: u64) -> (EvidenceRequest, DeploymentRegistry) {
    let (mut measured, mut registry) = fixture::inputs(now);
    measured.workload.context_tokens = 32;
    registry.max_tokens = 2;
    let execution = ExecutionIdentity {
        runner_sha256: "cd".repeat(32),
        dependencies_sha256: "11".repeat(32),
        configuration_sha256: "22".repeat(32),
        effective_backend: "synthetic-cpu".into(),
        backend_evidence_sha256: "99".repeat(32),
        gpu_layers: 0,
        native_buffer_budget_bytes: 256,
        context_tokens: 32,
        sequence_count: 1,
    };
    let peers = measured
        .peers
        .iter()
        .zip(&registry.devices)
        .map(|(peer, device)| PeerAvailability {
            id: peer.id.clone(),
            resource_id: device.resource_id.clone(),
            endpoint_id: device.endpoint_id,
            authorized: true,
            compute_enabled: true,
            healthy: true,
            observed_at_unix_ms: now,
            budget_bytes: peer.budget_bytes.clone(),
            available_bytes: peer.available_bytes.clone(),
            reserved_bytes: peer.reserved_bytes.clone(),
            execution: Some(execution.clone()),
            conditions: Some(CurrentConditions {
                power: PowerSource::Ac,
                thermal: ThermalState::Nominal,
            }),
        })
        .collect();
    let ranges = measured.peers.iter().flat_map(|peer| peer.ranges.iter().map(|range| {
        let steps = (0..4).map(|position| StepSample {
            phase: if position == 0 { Phase::Prefill } else { Phase::Decode },
            position: if position == 0 { 0 } else { position + 28 },
            tokens: if position == 0 { 29 } else { 1 },
            compute_us: range.decode_us, service_us: range.decode_us,
        }).collect();
        RangeEvidence {
            trial_policy_sha256: trial_policy_sha256(&policy(now)).unwrap(),
            profile_id: range.profile_id.clone(), peer_id: peer.id.clone(), report_sha256: "33".repeat(32),
            resource_id: format!("physical-{}", peer.id),
            artifact_sha256: "ef".repeat(32), workload: measured.workload.clone(),
            begin: range.begin, end: range.end, measured_at_unix_ms: now,
            outcome: TrialOutcome::Passed, cleanup_confirmed: true, execution: Some(execution.clone()),
            correctness: Some(CorrectnessEvidence { passed: true, fixture_sha256: "44".repeat(32),
                reference_validation_sha256: "55".repeat(32), reference_runtime_id: "synthetic-reference".into(),
                reference_backend: "synthetic-cpu".into(), checked_steps: 8 }),
            memory: Some(MemoryEvidence { scope: MemoryScope::WholeContribution, evidence_sha256: "66".repeat(32),
                accounting: "synthetic disjoint pools; complete load and process/driver contribution".into(),
                load_peak_bytes: range.peak_bytes.clone(), inference_peak_bytes: range.peak_bytes.clone() }),
            conditions: Some(ConditionEvidence { power: PowerSource::Ac, worst_thermal: ThermalState::Nominal,
                throttled: false, monitoring_complete: true, evidence_sha256: "77".repeat(32) }),
            sessions: vec![SessionEvidence { load_us: 1, steps }; 2],
            requested_sessions: 2,
        }
    })).collect();
    let links = measured
        .links
        .iter()
        .map(|link| TransferEvidence {
            profile_id: link.profile_id.clone(),
            report_sha256: "88".repeat(32),
            from: link.from.clone(),
            to: link.to.clone(),
            from_endpoint_id: registry
                .devices
                .iter()
                .find(|d| d.peer_id == link.from)
                .unwrap()
                .endpoint_id,
            to_endpoint_id: registry
                .devices
                .iter()
                .find(|d| d.peer_id == link.to)
                .unwrap()
                .endpoint_id,
            workload: measured.workload.clone(),
            payload_bytes: link.payload_bytes,
            measured_at_unix_ms: now,
            transport: "iroh".into(),
            protocol: "meshthing/shard-ring/1".into(),
            path: LinkPath::Direct,
            outcome: TrialOutcome::Passed,
            transfer_us: vec![2, 2],
        })
        .collect();
    (
        EvidenceRequest {
            schema_version: 2,
            workload: measured.workload,
            origin_peer_id: measured.origin_peer_id,
            min_peers: 2,
            policy: policy(now),
            peers,
            ranges,
            links,
        },
        registry,
    )
}

#[test]
fn complete_evidence_uses_whole_range_costs_and_real_deployment_configs() {
    let (input, registry) = inputs(1000);
    let deployment = prepare_ring(&input, &registry, 1100).unwrap();
    let report = &deployment.qualification;
    assert!(!report.hardware_attested && !report.activation_ready && report.advisory);
    assert!(report.exclusions.is_empty());
    assert_eq!(report.valid_until_unix_ms, Some(1500));
    assert_eq!(report.plan.as_ref().unwrap().predicted_decode_us, 36);
    assert_eq!(report.plan.as_ref().unwrap().assignments.len(), 3);
    assert!(!deployment.bundle.activation_ready);
    assert_eq!(deployment.bundle.workers[1].config().unwrap().context, 32);
    validate_deployment(&deployment, &input, &registry, 1500).unwrap();
}

#[test]
fn short_lived_capacity_and_links_cannot_borrow_range_evidence_lifetime() {
    let (mut input, registry) = inputs(1000);
    input.policy.capacity_max_age_ms = 200;
    let deployment = prepare_ring(&input, &registry, 1100).unwrap();
    assert_eq!(deployment.qualification.valid_until_unix_ms, Some(1200));
    assert!(deployment.bundle.expires_at_unix_ms > 1200);
    assert!(validate_deployment(&deployment, &input, &registry, 1201).is_err());
    let expired = plan_ring(&input, 1201).unwrap();
    assert!(expired.plan.is_none());
    assert!(
        expired
            .exclusions
            .iter()
            .any(|e| e.kind == "peer" && e.reasons.contains(&"capacity_stale_or_future".into()))
    );
    let (input, registry) = inputs(1000);
    let deployment = prepare_ring(&input, &registry, 1100).unwrap();
    assert!(validate_deployment(&deployment, &input, &registry, 1501).is_err());
    assert!(plan_ring(&input, 1501).unwrap().plan.is_none());
}

#[test]
fn incomplete_diagnostic_fields_are_excluded_instead_of_invented() {
    let mutations: &[fn(&mut RangeEvidence)] = &[
        |r| r.memory = None,
        |r| r.memory.as_mut().unwrap().scope = MemoryScope::GgmlBuffers,
        |r| r.memory.as_mut().unwrap().scope = MemoryScope::ProcessRss,
        |r| r.execution = None,
        |r| r.execution.as_mut().unwrap().effective_backend.clear(),
        |r| {
            r.execution
                .as_mut()
                .unwrap()
                .backend_evidence_sha256
                .clear()
        },
        |r| r.conditions = None,
        |r| r.conditions.as_mut().unwrap().monitoring_complete = false,
        |r| r.conditions.as_mut().unwrap().throttled = true,
        |r| r.correctness = None,
        |r| {
            r.correctness
                .as_mut()
                .unwrap()
                .reference_validation_sha256
                .clear()
        },
        |r| r.correctness.as_mut().unwrap().checked_steps -= 1,
        |r| r.outcome = TrialOutcome::Cancelled,
        |r| r.cleanup_confirmed = false,
        |r| r.requested_sessions = 3,
        |r| r.artifact_sha256.clear(),
    ];
    for (case, mutate) in mutations.iter().enumerate() {
        let (mut input, _) = inputs(1000);
        for range in &mut input.ranges {
            mutate(range);
        }
        let report = plan_ring(&input, 1100).unwrap();
        assert!(report.plan.is_none(), "case {case}");
        assert_eq!(
            report
                .exclusions
                .iter()
                .filter(|e| e.kind == "range")
                .count(),
            input.ranges.len()
        );
    }
}

#[test]
fn state_history_and_sustained_duration_are_checked_per_session() {
    let mutations: &[fn(&mut RangeEvidence)] = &[
        |r| r.sessions[0].steps[1].position += 1,
        |r| r.sessions[0].steps[2].phase = Phase::Prefill,
        |r| {
            r.sessions[0].steps.pop();
        },
        |r| r.sessions[0].steps[1].tokens = 2,
        |r| r.sessions[0].steps[1].service_us = 0,
        |r| r.sessions[0].steps[1].service_us = 1,
        |r| r.sessions[0].steps[1].service_us = u64::MAX,
    ];
    for mutate in mutations {
        let (mut input, _) = inputs(1000);
        mutate(&mut input.ranges[0]);
        let report = plan_ring(&input, 1100).unwrap();
        assert!(report.exclusions.iter().any(|e| {
            e.id == input.ranges[0].profile_id
                && e.reasons
                    .contains(&"invalid_or_incomplete_state_history".into())
        }));
    }
    let (mut input, _) = inputs(1000);
    input.policy.min_session_service_us = 61;
    let trial_policy_sha = trial_policy_sha256(&input.policy).unwrap();
    for range in &mut input.ranges {
        range.trial_policy_sha256 = trial_policy_sha.clone();
        for session in &mut range.sessions {
            for step in &mut session.steps {
                step.service_us = 10;
                step.compute_us = 10;
            }
        }
    }
    // Two 40-us sessions must not be added into one sustained 80-us session.
    assert!(plan_ring(&input, 1100).unwrap().plan.is_none());
}

#[test]
fn departure_replans_using_remaining_measured_ranges() {
    let (mut input, registry) = inputs(1000);
    input.peers[2].healthy = false;
    let deployment = prepare_ring(&input, &registry, 1100).unwrap();
    let assignments = &deployment.qualification.plan.as_ref().unwrap().assignments;
    assert_eq!(assignments.len(), 2);
    assert_eq!(assignments[0].layers, 0..2);
    assert_eq!(assignments[1].layers, 2..4);
    input.peers[1].compute_enabled = false;
    assert!(plan_ring(&input, 1100).unwrap().plan.is_none());
}

#[test]
fn build_conditions_protocol_and_payload_cannot_be_substituted() {
    let (mut input, _) = inputs(1000);
    input.peers[0]
        .execution
        .as_mut()
        .unwrap()
        .configuration_sha256 = "99".repeat(32);
    assert!(plan_ring(&input, 1100).unwrap().plan.is_none());
    let (mut input, _) = inputs(1000);
    input.peers[0].conditions.as_mut().unwrap().power = PowerSource::Battery;
    assert!(plan_ring(&input, 1100).unwrap().plan.is_none());
    let (mut input, _) = inputs(1000);
    for edge in &mut input.links {
        edge.protocol = "other-protocol".into();
    }
    assert!(plan_ring(&input, 1100).unwrap().plan.is_none());
    let (mut input, _) = inputs(1000);
    for edge in &mut input.links {
        edge.payload_bytes = 999;
    }
    assert!(plan_ring(&input, 1100).unwrap().plan.is_none());
}

#[test]
fn selected_registry_binds_artifact_runner_owner_and_native_budget() {
    let changes: &[fn(&mut DeploymentRegistry)] = &[
        |r| r.devices[0].runner_sha256 = "99".repeat(32),
        |r| r.devices[0].artifacts[0].artifact_sha256 = "99".repeat(32),
        |r| r.devices[0].artifacts[0].gpu_layers = 99,
        |r| r.devices[0].artifacts[0].budget_bytes += 1,
        |r| r.devices[0].resource_id = "changed-resource".into(),
        |r| r.devices[0].endpoint_id = iroh::SecretKey::from_bytes(&[9; 32]).public(),
    ];
    for mutate in changes {
        let (input, mut registry) = inputs(1000);
        mutate(&mut registry);
        assert!(prepare_ring(&input, &registry, 1100).is_err());
    }
}

#[test]
fn changed_inputs_or_tampered_envelope_do_not_revalidate() {
    let (input, registry) = inputs(1000);
    let saved = prepare_ring(&input, &registry, 1100).unwrap();
    let mut changed = input.clone();
    changed.peers[0].observed_at_unix_ms += 1;
    assert!(validate_deployment(&saved, &changed, &registry, 1150).is_err());
    let mut tampered = saved.clone();
    tampered.qualification.valid_until_unix_ms = Some(61_000);
    assert!(validate_deployment(&tampered, &input, &registry, 2000).is_err());
    let mut tampered = saved.clone();
    tampered.qualification.hardware_attested = true;
    assert!(validate_deployment(&tampered, &input, &registry, 1150).is_err());
    assert!(validate_deployment(&saved, &input, &registry, 1099).is_err());
}

#[test]
fn identity_aliases_and_duplicate_evidence_fail_closed() {
    let (mut input, _) = inputs(1000);
    input.peers[1].resource_id = input.peers[0].resource_id.clone();
    assert!(plan_ring(&input, 1100).is_err());
    let (mut input, _) = inputs(1000);
    input.links[0].profile_id = input.ranges[0].profile_id.clone();
    assert!(plan_ring(&input, 1100).is_err());
    let (mut input, _) = inputs(1000);
    let mut duplicate = input.ranges[0].clone();
    duplicate.profile_id = "same-range-another-trial".into();
    input.ranges.push(duplicate);
    assert!(plan_ring(&input, 1100).is_err());
}

#[test]
fn p95_uses_observed_tails_without_splitting_range_costs() {
    let (mut input, _) = inputs(1000);
    input.ranges[0].sessions[0].steps[1].service_us = 20;
    let plan = plan_ring(&input, 1100).unwrap().plan.unwrap();
    assert_eq!(plan.assignments[0].decode_us, 20);
    assert_eq!(plan.predicted_decode_us, 46);
    let edge = input
        .links
        .iter_mut()
        .find(|e| e.from == "leader" && e.to == "entry" && e.payload_bytes == 32)
        .unwrap();
    edge.transfer_us = vec![2, 7];
    assert_eq!(
        plan_ring(&input, 1100)
            .unwrap()
            .plan
            .unwrap()
            .predicted_decode_us,
        51
    );
}

#[test]
fn reused_peer_names_do_not_reuse_another_device_or_endpoint_evidence() {
    let (mut input, _) = inputs(1000);
    input.peers[0].resource_id = "replacement-device".into();
    let result = plan_ring(&input, 1100).unwrap();
    assert!(result.plan.is_none());
    assert!(
        result
            .exclusions
            .iter()
            .any(|e| e.reasons.contains(&"physical_resource_changed".into()))
    );
    let (mut input, _) = inputs(1000);
    input.peers[0].endpoint_id = iroh::SecretKey::from_bytes(&[9; 32]).public();
    let result = plan_ring(&input, 1100).unwrap();
    assert!(result.plan.is_none());
    assert!(
        result
            .exclusions
            .iter()
            .any(|e| e.reasons.contains(&"link_endpoint_identity_changed".into()))
    );
}

#[test]
fn future_range_and_transfer_observations_are_not_eligible() {
    let (mut input, _) = inputs(1000);
    for range in &mut input.ranges {
        range.measured_at_unix_ms = 1200;
    }
    assert!(plan_ring(&input, 1100).unwrap().plan.is_none());
    let (mut input, _) = inputs(1000);
    for edge in &mut input.links {
        edge.measured_at_unix_ms = 1200;
    }
    assert!(plan_ring(&input, 1100).unwrap().plan.is_none());
}

#[test]
fn trial_acceptance_policy_cannot_be_changed_after_measuring() {
    let (mut input, _) = inputs(1000);
    input.policy.min_session_service_us = 1;
    let report = plan_ring(&input, 1100).unwrap();
    assert!(report.plan.is_none());
    assert!(report.exclusions.iter().any(|e| {
        e.reasons
            .contains(&"trial_policy_missing_changed_or_declared_after_measurement".into())
    }));
    let (mut input, _) = inputs(1000);
    input.policy.trial_policy_declared_at_unix_ms = 1050;
    let hash = trial_policy_sha256(&input.policy).unwrap();
    for range in &mut input.ranges {
        range.trial_policy_sha256 = hash.clone();
    }
    assert!(plan_ring(&input, 1100).unwrap().plan.is_none());
    let (mut input, _) = inputs(1000);
    let hash = trial_policy_sha256(&input.policy).unwrap();
    input.policy.capacity_max_age_ms += 1;
    assert_eq!(trial_policy_sha256(&input.policy).unwrap(), hash);
    assert!(plan_ring(&input, 1100).unwrap().plan.is_some());
}

#[test]
fn native_diagnostic_schema_cannot_be_used_as_qualified_evidence() {
    let diagnostic = serde_json::json!({
        "schema_version": 1, "kind": "native_candidate_diagnostic",
        "status": "passed", "eligible_for_placement": false,
    });
    assert!(serde_json::from_value::<EvidenceRequest>(diagnostic).is_err());
    let (mut input, _) = inputs(1000);
    input.policy.protocol = "different-wire-protocol".into();
    assert!(plan_ring(&input, 1100).is_err());
}

#[test]
fn joined_devices_without_qualified_ranges_do_not_exhaust_search_capacity() {
    let (mut input, _) = inputs(1000);
    for number in 10u8..17 {
        let mut peer = input.peers[0].clone();
        peer.id = format!("unprofiled-{number}");
        peer.resource_id = format!("physical-unprofiled-{number}");
        peer.endpoint_id = iroh::SecretKey::from_bytes(&[number; 32]).public();
        input.peers.push(peer);
    }
    let result = plan_ring(&input, 1100).unwrap();
    assert_eq!(result.plan.unwrap().predicted_decode_us, 36);
    assert_eq!(
        result
            .exclusions
            .iter()
            .filter(|e| e.reasons.contains(&"no_qualified_ranges".into()))
            .count(),
        7
    );
}

#[test]
fn whole_physical_pool_peak_and_reserves_enforce_capacity() {
    let (mut input, _) = inputs(1000);
    for range in &mut input.ranges {
        if range.peer_id == "leader" {
            range.memory.as_mut().unwrap().load_peak_bytes.ram = 1025;
        }
    }
    assert!(plan_ring(&input, 1100).unwrap().plan.is_none());
    let (mut input, _) = inputs(1000);
    input.peers[0].reserved_bytes.ram = 513;
    assert!(plan_ring(&input, 1100).unwrap().plan.is_none());
}

#[test]
fn cli_checks_prepares_and_revalidates_without_opening_sockets() {
    let now: u64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .try_into()
        .unwrap();
    let (mut input, registry) = inputs(now);
    input.policy.capacity_max_age_ms = 60_000;
    input.policy.link_max_age_ms = 60_000;
    let dir = tempfile::tempdir().unwrap();
    let evidence_path = dir.path().join("evidence.json");
    let registry_path = dir.path().join("registry.json");
    let deployment_path = dir.path().join("deployment.json");
    std::fs::write(&evidence_path, serde_json::to_vec(&input).unwrap()).unwrap();
    std::fs::write(&registry_path, serde_json::to_vec(&registry).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_llm-cluster"))
        .arg("qualified-plan")
        .arg(&evidence_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: QualifiedPlan = serde_json::from_slice(&output.stdout).unwrap();
    assert!(plan.plan.is_some());
    let output = Command::new(env!("CARGO_BIN_EXE_llm-cluster"))
        .arg("prepare-qualified-ring")
        .arg(&evidence_path)
        .arg(&registry_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::write(&deployment_path, &output.stdout).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_llm-cluster"))
        .arg("validate-qualified-ring")
        .arg(&deployment_path)
        .arg(&evidence_path)
        .arg(&registry_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap()["activation_ready"],
        false
    );
}
