use llm_cluster::measured_placement::*;
use llm_cluster::ring_deployment::*;
use serde_json::json;

/// Entirely synthetic observations and identities: no hardware qualification.
pub fn inputs(now: u64) -> (MeasuredRequest, DeploymentRegistry) {
    let workload = Workload {
        model_sha256: "ab".repeat(32),
        runtime_id: "synthetic-runtime".into(),
        context_tokens: 512,
        batch_tokens: 1,
        layers: 4,
        activation_payload_bytes: 32,
        control_payload_bytes: 4,
    };
    let memory = || MemoryBytes {
        ram: 1024,
        vram: 0,
        shared: 0,
    };
    let definitions = [
        ("leader", vec![(0, 1, 10), (0, 2, 90)]),
        ("entry", vec![(1, 3, 10), (2, 4, 90)]),
        ("terminal", vec![(3, 4, 10)]),
    ];
    let peers: Vec<_> = definitions
        .iter()
        .map(|(id, ranges)| PeerObservation {
            id: (*id).into(),
            authorized: true,
            healthy: true,
            capacity_observed_at_unix_ms: now,
            budget_bytes: memory(),
            available_bytes: memory(),
            reserved_bytes: MemoryBytes::default(),
            ranges: ranges
                .iter()
                .map(|&(begin, end, decode_us)| RangeObservation {
                    begin,
                    end,
                    decode_us,
                    peak_bytes: MemoryBytes {
                        ram: 512,
                        ..MemoryBytes::default()
                    },
                    profile_id: format!("{id}-{begin}-{end}"),
                    measured_at_unix_ms: now,
                    workload: workload.clone(),
                })
                .collect(),
        })
        .collect();
    let mut links = Vec::new();
    for from in &peers {
        for to in &peers {
            if from.id == to.id {
                continue;
            }
            for payload_bytes in [4, 32] {
                links.push(LinkObservation {
                    from: from.id.clone(),
                    to: to.id.clone(),
                    payload_bytes,
                    transfer_us: 2,
                    profile_id: format!("{}-{}-{payload_bytes}", from.id, to.id),
                    measured_at_unix_ms: now,
                    path: "synthetic-direct".into(),
                    workload: workload.clone(),
                });
            }
        }
    }
    let devices = peers
        .iter()
        .enumerate()
        .map(|(index, peer)| DeviceBinding {
            peer_id: peer.id.clone(),
            resource_id: format!("physical-{}", peer.id),
            endpoint_id: iroh::SecretKey::from_bytes(&[(index + 1) as u8; 32]).public(),
            key_path: format!("/synthetic/{}/identity.key", peer.id),
            runner: format!("/synthetic/{}/runner with spaces", peer.id),
            runner_sha256: "cd".repeat(32),
            artifacts: peer
                .ranges
                .iter()
                .map(|range| ArtifactBinding {
                    role: None,
                    range_profile_id: range.profile_id.clone(),
                    begin: range.begin,
                    end: range.end,
                    model: format!("/synthetic/{}/{}.gguf", peer.id, range.profile_id),
                    artifact_sha256: "ef".repeat(32),
                    gpu_layers: 0,
                    budget_bytes: 256,
                })
                .collect(),
        })
        .collect();
    let registry = DeploymentRegistry {
        schema_version: 1,
        model_sha256: workload.model_sha256.clone(),
        runtime_id: workload.runtime_id.clone(),
        chat_template: json!("chatml"),
        max_tokens: 64,
        leader: LeaderSettings {
            cluster: "synthetic-ring".into(),
            manifest_path: "/synthetic/leader/leader-manifest.json".into(),
            api: "127.0.0.1:8080".parse().unwrap(),
            upstream_port: 8081,
            worker_wait_seconds: 60,
        },
        devices,
    };
    (
        MeasuredRequest {
            schema_version: 1,
            workload,
            origin_peer_id: "leader".into(),
            observed_at_unix_ms: now,
            max_profile_age_ms: 60_000,
            min_peers: 2,
            peers,
            links,
        },
        registry,
    )
}
