use llm_cluster::placement::*;

fn model() -> ModelProfile {
    ModelProfile {
        model_hash: "sha256:test".into(),
        runtime_id: "runtime:test".into(),
        context_tokens: 10,
        layers: vec![
            LayerMemory {
                weight_bytes: 100,
                kv_bytes_per_token: 2
            };
            4
        ],
    }
}
fn peer(id: &str, budget: u64, cost: u64) -> PeerProfile {
    PeerProfile {
        id: id.into(),
        model_hash: "sha256:test".into(),
        runtime_id: "runtime:test".into(),
        context_tokens: 10,
        authorized: true,
        healthy: true,
        memory_budget_bytes: budget,
        reserved_bytes: 50,
        decode_us_per_layer: vec![cost; 4],
    }
}
fn link(a: &str, b: &str, cost: u64) -> LinkProfile {
    LinkProfile {
        from: a.into(),
        to: b.into(),
        transfer_us: cost,
    }
}

#[test]
fn pools_memory_and_accounts_for_kv_and_reserves() {
    let peers = [peer("a", 290, 10), peer("b", 290, 20)];
    let result = plan(&model(), &peers, &[link("a", "b", 7)], 2).unwrap();
    assert_eq!(
        result.assignments,
        vec![
            Assignment {
                peer_id: "a".into(),
                layers: 0..2,
                estimated_bytes: 290
            },
            Assignment {
                peer_id: "b".into(),
                layers: 2..4,
                estimated_bytes: 290
            },
        ]
    );
    assert_eq!(result.predicted_decode_us, 67);
    assert_eq!(
        plan(&model(), &peers[..1], &[], 1),
        Err(PlanError::NoFeasiblePlacement)
    );
}

#[test]
fn skips_a_slow_middle_device_and_prices_the_actual_transition() {
    let peers = [
        peer("c", 290, 20),
        peer("slow", 1000, 1000),
        peer("a", 290, 10),
    ];
    let links = [
        link("a", "slow", 1),
        link("slow", "c", 1),
        link("a", "c", 7),
    ];
    let result = plan(&model(), &peers, &links, 2).unwrap();
    assert_eq!(
        result
            .assignments
            .iter()
            .map(|a| a.peer_id.as_str())
            .collect::<Vec<_>>(),
        ["a", "c"]
    );
    assert_eq!(result.predicted_decode_us, 67);
}

#[test]
fn does_not_claim_distribution_when_one_peer_is_faster() {
    let peers = [peer("a", 1000, 10), peer("b", 1000, 100)];
    let links = [link("a", "b", 100)];
    assert_eq!(
        plan(&model(), &peers, &links, 1).unwrap().assignments.len(),
        1
    );
    assert_eq!(
        plan(&model(), &peers, &links, 2).unwrap().assignments.len(),
        2
    );
}

#[test]
fn refuses_missing_links_and_insufficient_memory() {
    let peers = [peer("a", 290, 10), peer("b", 289, 20)];
    assert_eq!(
        plan(&model(), &peers, &[link("a", "b", 1)], 2),
        Err(PlanError::NoFeasiblePlacement)
    );
    let peers = [peer("a", 290, 10), peer("b", 290, 20)];
    assert_eq!(
        plan(&model(), &peers, &[], 2),
        Err(PlanError::NoFeasiblePlacement)
    );
    let result = plan(&model(), &peers, &[link("b", "a", 1)], 2).unwrap();
    assert_eq!(
        result
            .assignments
            .iter()
            .map(|assignment| assignment.peer_id.as_str())
            .collect::<Vec<_>>(),
        ["b", "a"]
    );
}

#[test]
fn filters_untrusted_unhealthy_and_incompatible_peers() {
    for field in 0..5 {
        let mut b = peer("b", 1000, 10);
        match field {
            0 => b.authorized = false,
            1 => b.healthy = false,
            2 => b.model_hash = "other".into(),
            3 => b.runtime_id = "other".into(),
            _ => b.context_tokens = 100,
        }
        assert_eq!(
            plan(&model(), &[peer("a", 290, 10), b], &[link("a", "b", 1)], 2),
            Err(PlanError::NoFeasiblePlacement)
        );
    }
}

#[test]
fn detects_bad_profiles_and_arithmetic_overflow() {
    let mut m = model();
    m.layers[0].kv_bytes_per_token = u64::MAX;
    assert_eq!(
        plan(&m, &[peer("a", 1000, 1)], &[], 1),
        Err(PlanError::Overflow)
    );
    assert!(matches!(
        plan(&model(), &[peer("a", 1000, 1), peer("a", 1000, 1)], &[], 1),
        Err(PlanError::InvalidInput(_))
    ));
    let mut p = peer("a", 1000, 1);
    p.decode_us_per_layer.clear();
    assert!(matches!(
        plan(&model(), &[p], &[], 1),
        Err(PlanError::InvalidInput(_))
    ));
}

#[test]
fn rejects_layer_count_above_exact_search_bound() {
    let mut m = model();
    m.layers.resize(
        129,
        LayerMemory {
            weight_bytes: 100,
            kv_bytes_per_token: 2,
        },
    );

    assert!(matches!(
        plan(&m, &[], &[], 1),
        Err(PlanError::InvalidInput(_))
    ));
}

#[test]
fn matches_exhaustive_two_peer_boundary_search() {
    for budget_a in [170, 290, 410, 530] {
        for budget_b in [170, 290, 410, 530] {
            for transfer in [0, 9, 1000] {
                let peers = [peer("a", budget_a, 7), peer("b", budget_b, 13)];
                let expected = (1..4)
                    .filter(|k| 50 + 120 * k <= budget_a && 50 + 120 * (4 - k) <= budget_b)
                    .map(|k| 7 * k + 13 * (4 - k) + transfer)
                    .min();
                let actual = plan(&model(), &peers, &[link("a", "b", transfer)], 2);
                match expected {
                    Some(cost) => assert_eq!(actual.unwrap().predicted_decode_us, cost),
                    None => assert_eq!(actual, Err(PlanError::NoFeasiblePlacement)),
                }
            }
        }
    }
}
