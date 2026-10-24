use llm_cluster::{planning::*, verification::*};

fn fixture() -> (PlanningRequest, PlanningOutput, RunEvidence) {
    let request: PlanningRequest =
        serde_json::from_str(include_str!("../examples/placement-profile.json")).unwrap();
    let output = build_plan(&request, 2).unwrap();
    let evidence = RunEvidence {
        model_hash: request.model.model_hash.clone(),
        runtime_id: request.model.runtime_id.clone(),
        context_tokens: request.model.context_tokens,
        devices: request
            .peers
            .iter()
            .map(|p| DeviceEvidence {
                peer_id: p.id.clone(),
                peak_bytes: p.memory_budget_bytes,
                compute_events: 10,
            })
            .collect(),
    };
    (request, output, evidence)
}

#[test]
fn verifies_saved_plan_against_current_inputs() {
    let (request, output, evidence) = fixture();
    assert_eq!(verify_output(&request, &output, &evidence, 2), Ok(()));
}

#[test]
fn refuses_truncated_model_and_altered_launch_arguments() {
    let (request, mut output, evidence) = fixture();
    output.placement.assignments.last_mut().unwrap().layers.end -= 1;
    assert_eq!(
        verify_output(&request, &output, &evidence, 2),
        Err(VerificationError::InvalidPlan)
    );
    let (request, mut output, evidence) = fixture();
    output.server_args.push("--unexpected".into());
    assert_eq!(
        verify_output(&request, &output, &evidence, 2),
        Err(VerificationError::InvalidPlan)
    );
}

#[test]
fn rejects_changed_model_memory_endpoint_and_final_peer() {
    let (mut request, output, evidence) = fixture();
    request.model.runtime_id = "patched-runtime".into();
    assert_eq!(
        verify_output(&request, &output, &evidence, 2),
        Err(VerificationError::IdentityMismatch)
    );
    let (mut request, output, evidence) = fixture();
    request.model.layers[0].weight_bytes += 1;
    assert_eq!(
        verify_output(&request, &output, &evidence, 2),
        Err(VerificationError::InvalidPlan)
    );
    let (mut request, output, evidence) = fixture();
    request.devices[1].rpc_endpoint = Some("127.0.0.1:50054".parse().unwrap());
    assert_eq!(
        verify_output(&request, &output, &evidence, 2),
        Err(VerificationError::InvalidPlan)
    );
    let (mut request, output, evidence) = fixture();
    request.final_peer_id = Some("worker".into());
    assert_eq!(
        verify_output(&request, &output, &evidence, 2),
        Err(VerificationError::InvalidPlan)
    );
}
