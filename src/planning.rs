//! File-driven advisory planning without network or process side effects.
use crate::placement::{self, LinkProfile, ModelProfile, PeerProfile, Placement, PlanError};
use crate::rpc_plan::{self, RuntimeDevice};
use crate::verification::{self, RunEvidence, VerificationError};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanningRequest {
    pub model: ModelProfile,
    pub peers: Vec<PeerProfile>,
    pub links: Vec<LinkProfile>,
    /// One discovered device for every candidate peer; order is immaterial.
    pub devices: Vec<RuntimeDevice>,
    pub model_path: String,
    /// Require the final block group on this peer (for example, the local device).
    #[serde(default)]
    pub final_peer_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PlanningOutput {
    pub advisory: bool,
    pub placement: Placement,
    /// Pass as individual process arguments, not as a shell command.
    pub server_args: Vec<String>,
}

/// Verify a saved launch plan against current model/profile inputs and evidence.
/// This validates the proposed assignment and supplied observations, not actual
/// runtime layer/output ownership or the authenticity of collected telemetry.
pub fn verify_output(
    request: &PlanningRequest,
    output: &PlanningOutput,
    evidence: &RunEvidence,
    min_peers: usize,
) -> Result<(), VerificationError> {
    use VerificationError::{IdentityMismatch, InvalidPlan};
    let placement = &output.placement;
    if placement.model_hash != request.model.model_hash
        || placement.runtime_id != request.model.runtime_id
        || placement.context_tokens != request.model.context_tokens
    {
        return Err(IdentityMismatch);
    }
    if !output.advisory
        || request.model.layers.is_empty()
        || placement.assignments.last().map(|a| a.layers.end) != Some(request.model.layers.len())
        || request.final_peer_id.as_ref().is_some_and(|id| {
            placement
                .assignments
                .last()
                .is_none_or(|a| &a.peer_id != id)
        })
    {
        return Err(InvalidPlan);
    }
    verification::verify_run(placement, &request.peers, evidence, min_peers)?;
    let mut devices = Vec::new();
    for assignment in &placement.assignments {
        let peer = request
            .peers
            .iter()
            .find(|p| p.id == assignment.peer_id)
            .ok_or(InvalidPlan)?;
        let layers = request
            .model
            .layers
            .get(assignment.layers.clone())
            .ok_or(InvalidPlan)?;
        let mut bytes = peer.reserved_bytes;
        for layer in layers {
            if layer.weight_bytes == 0 {
                return Err(InvalidPlan);
            }
            bytes = layer
                .kv_bytes_per_token
                .checked_mul(placement.context_tokens)
                .and_then(|kv| kv.checked_add(layer.weight_bytes))
                .and_then(|memory| bytes.checked_add(memory))
                .ok_or(InvalidPlan)?;
        }
        if bytes != assignment.estimated_bytes || bytes > peer.memory_budget_bytes {
            return Err(InvalidPlan);
        }
        let mut matches = request
            .devices
            .iter()
            .filter(|d| d.peer_id == assignment.peer_id);
        devices.push(matches.next().ok_or(InvalidPlan)?.clone());
        if matches.next().is_some() {
            return Err(InvalidPlan);
        }
    }
    let argv =
        rpc_plan::server_args(placement, &devices, &request.model_path).map_err(|_| InvalidPlan)?;
    if argv != output.server_args {
        return Err(InvalidPlan);
    }
    Ok(())
}

pub fn build_plan(
    request: &PlanningRequest,
    min_peers: usize,
) -> Result<PlanningOutput, PlanError> {
    let mut ids = HashSet::new();
    for device in &request.devices {
        if !ids.insert(&device.peer_id)
            || !request.peers.iter().any(|peer| peer.id == device.peer_id)
        {
            return Err(PlanError::InvalidInput(
                "device mappings must reference unique candidate peers".into(),
            ));
        }
    }
    if request.devices.len() != request.peers.len() {
        return Err(PlanError::InvalidInput(
            "provide one device mapping for every candidate peer".into(),
        ));
    }
    let placement = placement::plan_with_final_peer(
        &request.model,
        &request.peers,
        &request.links,
        min_peers,
        request.final_peer_id.as_deref(),
    )?;
    let devices = placement
        .assignments
        .iter()
        .map(|assignment| {
            request
                .devices
                .iter()
                .find(|device| device.peer_id == assignment.peer_id)
                .expect("candidate device mappings validated above")
                .clone()
        })
        .collect::<Vec<_>>();
    let server_args = rpc_plan::server_args(&placement, &devices, &request.model_path)?;
    Ok(PlanningOutput {
        advisory: true,
        placement,
        server_args,
    })
}
