use llm_cluster::measured_placement::*;

fn memory(shared: u64) -> MemoryBytes {
    MemoryBytes {
        ram: 0,
        vram: 0,
        shared,
    }
}

fn workload() -> Workload {
    Workload {
        model_sha256: "abababababababababababababababababababababababababababababababab".into(),
        runtime_id: "runtime:build".into(),
        context_tokens: 512,
        batch_tokens: 1,
        layers: 4,
        activation_payload_bytes: 20_480,
        control_payload_bytes: 4,
    }
}

fn range(begin: usize, end: usize, decode_us: u64, id: &str) -> RangeObservation {
    RangeObservation {
        begin,
        end,
        decode_us,
        peak_bytes: memory(100),
        profile_id: id.into(),
        measured_at_unix_ms: 900,
        workload: workload(),
    }
}

fn peer(id: &str, ranges: Vec<RangeObservation>) -> PeerObservation {
    PeerObservation {
        id: id.into(),
        authorized: true,
        healthy: true,
        capacity_observed_at_unix_ms: 900,
        budget_bytes: memory(150),
        available_bytes: memory(150),
        reserved_bytes: memory(20),
        ranges,
    }
}

fn link(from: &str, to: &str, bytes: u64, time: u64) -> LinkObservation {
    LinkObservation {
        from: from.into(),
        to: to.into(),
        payload_bytes: bytes,
        transfer_us: time,
        profile_id: format!("{from}-{to}-{bytes}"),
        measured_at_unix_ms: 900,
        path: "direct".into(),
        workload: workload(),
    }
}

fn request() -> MeasuredRequest {
    MeasuredRequest {
        schema_version: 1,
        workload: workload(),
        origin_peer_id: "a".into(),
        observed_at_unix_ms: 1000,
        max_profile_age_ms: 200,
        min_peers: 2,
        // Deliberately give the better first-stage peer second in input order.
        peers: vec![
            peer(
                "a",
                vec![range(0, 2, 80, "a-prefix"), range(2, 4, 10, "a-suffix")],
            ),
            peer(
                "b",
                vec![range(0, 2, 10, "b-prefix"), range(2, 4, 80, "b-suffix")],
            ),
        ],
        links: vec![
            link("a", "b", 4, 2),
            link("b", "a", 4, 3),
            link("a", "b", 20_480, 5),
            link("b", "a", 20_480, 7),
        ],
    }
}

#[test]
fn chooses_peer_order_and_exact_ranges_with_ingress_cost() {
    let output = plan(&request()).unwrap();
    assert_eq!(
        output
            .assignments
            .iter()
            .map(|a| a.peer_id.as_str())
            .collect::<Vec<_>>(),
        ["b", "a"]
    );
    assert_eq!(output.assignments[0].layers, 0..2);
    assert_eq!(output.assignments[1].layers, 2..4);
    assert_eq!(output.predicted_decode_us, 29); // ingress 2 + range 10 + link 7 + range 10
    assert_eq!(output.link_profile_ids, ["a-b-4", "b-a-20480"]);
    assert!(output.advisory);
}

#[test]
fn ring_search_finds_supported_alternative_to_cheaper_remote_first_plan() {
    let input = request();
    assert_eq!(plan(&input).unwrap().assignments[0].peer_id, "b");
    let ring = plan_ring(&input).unwrap();
    assert_eq!(ring.assignments[0].peer_id, "a");
    assert_eq!(ring.assignments[0].layers, 0..2);
    assert_eq!(ring.assignments[1].peer_id, "b");
    assert_eq!(ring.predicted_decode_us, 80 + 5 + 80 + 3);
    assert_eq!(ring.link_profile_ids, ["a-b-20480", "b-a-4"]);
    validate_ring_output(&input, &ring).unwrap();
    assert!(validate_ring_output(&input, &plan(&input).unwrap()).is_err());
}

#[test]
fn ring_requires_origin_remote_and_single_token_decode() {
    let mut input = request();
    input.min_peers = 1;
    input.peers[0].ranges.push(range(0, 4, 1, "a-whole"));
    assert_eq!(plan(&input).unwrap().assignments.len(), 1);
    assert_eq!(plan_ring(&input).unwrap().assignments.len(), 2);
    input.peers[1].healthy = false;
    assert_eq!(plan_ring(&input), Err(MeasuredError::NoFeasiblePlacement));
    input.peers[1].healthy = true;
    input.peers[0].authorized = false;
    assert_eq!(plan_ring(&input), Err(MeasuredError::NoFeasiblePlacement));
    input.peers[0].authorized = true;
    input.workload.batch_tokens = 2;
    assert!(matches!(
        plan_ring(&input),
        Err(MeasuredError::InvalidInput(_))
    ));
}

#[test]
fn peers_without_eligible_ranges_do_not_count_against_search_limit() {
    let mut input = request();
    for index in 0..7 {
        input.peers.push(peer(&format!("empty-{index}"), vec![]));
    }
    assert_eq!(input.peers.len(), 9);

    let output = plan(&input).unwrap();
    assert_eq!(
        output
            .assignments
            .iter()
            .map(|assignment| assignment.peer_id.as_str())
            .collect::<Vec<_>>(),
        ["b", "a"]
    );
}

#[test]
fn requires_fresh_matching_range_and_feedback_measurements() {
    let mut input = request();
    input.peers[1].ranges[0].measured_at_unix_ms = 1;
    // The other orientation is still feasible, with its own feedback cost.
    let fallback = plan(&input).unwrap();
    assert_eq!(fallback.assignments[0].peer_id, "a");
    assert_eq!(fallback.predicted_decode_us, 80 + 5 + 80 + 3);

    input
        .links
        .retain(|edge| !(edge.from == "b" && edge.to == "a" && edge.payload_bytes == 4));
    assert_eq!(plan(&input), Err(MeasuredError::NoFeasiblePlacement));

    let mut input = request();
    input.peers[1].ranges[0].workload.runtime_id = "other-build".into();
    assert_eq!(plan(&input).unwrap().assignments[0].peer_id, "a");
}

#[test]
fn charges_shared_memory_once_and_refuses_over_budget_ranges() {
    let mut input = request();
    input.peers[1].ranges[0].peak_bytes = memory(131);
    assert_eq!(plan(&input).unwrap().assignments[0].peer_id, "a");
    input.peers[0].ranges[0].peak_bytes = memory(131);
    assert_eq!(plan(&input), Err(MeasuredError::NoFeasiblePlacement));

    let mut input = request();
    input.peers[1].ranges[0].peak_bytes = MemoryBytes {
        ram: 1,
        vram: 0,
        shared: 100,
    };
    assert_eq!(plan(&input).unwrap().assignments[0].peer_id, "a");
}

#[test]
fn rejects_stale_available_capacity() {
    let mut input = request();
    input.peers[1].capacity_observed_at_unix_ms = 1;
    assert_eq!(plan(&input), Err(MeasuredError::NoFeasiblePlacement));

    let mut input = request();
    input.peers[1].available_bytes = memory(119);
    assert_eq!(plan(&input), Err(MeasuredError::NoFeasiblePlacement));
}

#[test]
fn saved_plan_requires_current_profiles_and_exact_cost() {
    let input = request();
    let mut output = plan(&input).unwrap();
    validate_output(&input, &output).unwrap();
    output.predicted_decode_us += 1;
    assert!(matches!(
        validate_output(&input, &output),
        Err(MeasuredError::InvalidInput(_))
    ));

    let mut later = request();
    later.observed_at_unix_ms = 1200;
    assert_eq!(
        validate_output(&later, &plan(&input).unwrap()),
        Err(MeasuredError::NoFeasiblePlacement)
    );
}

#[test]
fn rejects_ambiguous_or_malformed_profiles() {
    let mut input = request();
    input.workload.model_sha256 = "not-a-sha256".into();
    assert!(matches!(plan(&input), Err(MeasuredError::InvalidInput(_))));

    let mut input = request();
    input.links.push(input.links[0].clone());
    assert!(matches!(plan(&input), Err(MeasuredError::InvalidInput(_))));
    input.links.last_mut().unwrap().measured_at_unix_ms = 1;
    plan(&input).unwrap();

    let mut input = request();
    let duplicate = input.peers[0].ranges[0].clone();
    input.peers[0].ranges.push(duplicate);
    assert!(matches!(plan(&input), Err(MeasuredError::InvalidInput(_))));

    let mut value = serde_json::to_value(request()).unwrap();
    value["workload"]["unknown_budget"] = 999.into();
    assert!(serde_json::from_value::<MeasuredRequest>(value).is_err());
}

#[test]
fn bounds_search_state_growth_and_profile_labels() {
    let mut input = request();
    input.peers[0].ranges[0].profile_id = "x".repeat(257);
    assert!(matches!(plan(&input), Err(MeasuredError::InvalidInput(_))));
    let mut input = request();
    input.peers[0].ranges = vec![input.peers[0].ranges[0].clone(); 4097];
    assert!(matches!(
        plan(&input),
        Err(MeasuredError::InvalidInput(message)) if message.contains("4096 range observations")
    ));

    let mut workload = workload();
    workload.layers = 20;
    let mut peers = Vec::new();
    for peer_index in 0..8 {
        let id = format!("peer-{peer_index}");
        let mut observations = Vec::new();
        for begin in 0..workload.layers {
            for end in begin + 1..=workload.layers {
                let mut observation = range(begin, end, 1, &format!("{id}-{begin}-{end}"));
                observation.workload = workload.clone();
                observations.push(observation);
            }
        }
        peers.push(peer(&id, observations));
    }
    let mut links = Vec::new();
    for from in 0..peers.len() {
        for to in 0..peers.len() {
            if from == to {
                continue;
            }
            for payload in [
                workload.control_payload_bytes,
                workload.activation_payload_bytes,
            ] {
                let mut observation = link(&peers[from].id, &peers[to].id, payload, 1);
                observation.workload = workload.clone();
                links.push(observation);
            }
        }
    }
    let dense = MeasuredRequest {
        schema_version: 1,
        workload,
        origin_peer_id: peers[0].id.clone(),
        observed_at_unix_ms: 1000,
        max_profile_age_ms: 200,
        min_peers: 2,
        peers,
        links,
    };
    assert!(matches!(
        plan(&dense),
        Err(MeasuredError::InvalidInput(message)) if message.contains("16384 states")
    ));
}
