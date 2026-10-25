//! Advisory placement from exact range and directed-link observations.
//! This path does not convert range timings into fictitious per-layer costs.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::ops::Range;

const MAX_PEERS: usize = 8;
const MAX_LAYERS: usize = 1024;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub model_sha256: String,
    pub runtime_id: String,
    pub context_tokens: u32,
    pub batch_tokens: u32,
    pub layers: usize,
    /// Bytes sent between adjacent stages for one decode step.
    pub activation_payload_bytes: u64,
    /// Bytes sent from the origin to stage 0 and from the final stage back.
    pub control_payload_bytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryBytes {
    /// Dedicated host memory. Unified-memory allocations belong in `shared`.
    pub ram: u64,
    /// Dedicated device memory. Unified-memory allocations belong in `shared`.
    pub vram: u64,
    /// One shared physical pool, counted once rather than as RAM plus VRAM.
    pub shared: u64,
}

impl MemoryBytes {
    fn add(&self, other: &Self) -> Option<Self> {
        Some(Self {
            ram: self.ram.checked_add(other.ram)?,
            vram: self.vram.checked_add(other.vram)?,
            shared: self.shared.checked_add(other.shared)?,
        })
    }

    fn fits(&self, budget: &Self) -> bool {
        self.ram <= budget.ram && self.vram <= budget.vram && self.shared <= budget.shared
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RangeObservation {
    pub begin: usize,
    pub end: usize,
    pub decode_us: u64,
    /// Peak allocation for this exact range during load and inference.
    pub peak_bytes: MemoryBytes,
    pub profile_id: String,
    pub measured_at_unix_ms: u64,
    pub workload: Workload,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PeerObservation {
    pub id: String,
    pub authorized: bool,
    pub healthy: bool,
    /// Fresh observation of usable capacity after existing reservations.
    pub capacity_observed_at_unix_ms: u64,
    /// Owner-approved maximum contribution by physical pool.
    pub budget_bytes: MemoryBytes,
    /// Currently usable capacity by physical pool, after other workloads.
    pub available_bytes: MemoryBytes,
    /// Other processes, staging and headroom charged to the same pools.
    pub reserved_bytes: MemoryBytes,
    pub ranges: Vec<RangeObservation>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LinkObservation {
    pub from: String,
    pub to: String,
    pub payload_bytes: u64,
    pub transfer_us: u64,
    pub profile_id: String,
    pub measured_at_unix_ms: u64,
    /// For example `direct` or `relay`; measurements are not interchangeable.
    pub path: String,
    pub workload: Workload,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredRequest {
    pub schema_version: u32,
    pub workload: Workload,
    pub origin_peer_id: String,
    pub observed_at_unix_ms: u64,
    pub max_profile_age_ms: u64,
    pub min_peers: usize,
    pub peers: Vec<PeerObservation>,
    pub links: Vec<LinkObservation>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredAssignment {
    pub peer_id: String,
    pub layers: Range<usize>,
    pub range_profile_id: String,
    pub charged_bytes: MemoryBytes,
    pub decode_us: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredPlacement {
    pub schema_version: u32,
    pub advisory: bool,
    pub objective: String,
    pub planned_at_unix_ms: u64,
    pub max_profile_age_ms: u64,
    pub workload: Workload,
    pub origin_peer_id: String,
    pub assignments: Vec<MeasuredAssignment>,
    /// Ordered ingress, interstage and feedback link observation IDs.
    pub link_profile_ids: Vec<String>,
    pub predicted_decode_us: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeasuredError {
    InvalidInput(String),
    NoFeasiblePlacement,
    Overflow,
}

#[derive(Clone)]
struct Candidate {
    cost: u64,
    assignments: Vec<MeasuredAssignment>,
    links: Vec<String>,
}

fn invalid(message: &str) -> MeasuredError {
    MeasuredError::InvalidInput(message.into())
}

fn fresh(time: u64, request: &MeasuredRequest) -> bool {
    time <= request.observed_at_unix_ms
        && request.observed_at_unix_ms - time <= request.max_profile_age_ms
}

fn link<'a>(
    links: &'a HashMap<(String, String, u64), &'a LinkObservation>,
    from: &str,
    to: &str,
    payload: u64,
) -> Option<&'a LinkObservation> {
    links.get(&(from.into(), to.into(), payload)).copied()
}

/// Exact subset search over measured contiguous ranges and directed links.
/// Candidate order is independent of input order; each peer can own one range.
pub fn plan(request: &MeasuredRequest) -> Result<MeasuredPlacement, MeasuredError> {
    plan_with_topology(request, false)
}

/// M-27 ring placement: layer zero stays on the origin, followed by at least one
/// remote stage. Search with the constraint instead of rejecting the unconstrained
/// winner, which could hide a feasible ring alternative.
pub fn plan_ring(request: &MeasuredRequest) -> Result<MeasuredPlacement, MeasuredError> {
    if request.workload.batch_tokens != 1 {
        return Err(invalid(
            "ring planning requires single-token decode observations",
        ));
    }
    plan_with_topology(request, true)
}

fn plan_with_topology(
    request: &MeasuredRequest,
    ring: bool,
) -> Result<MeasuredPlacement, MeasuredError> {
    if request.schema_version != 1
        || !valid_sha256(&request.workload.model_sha256)
        || request.workload.runtime_id.is_empty()
        || request.workload.layers == 0
        || request.workload.layers > MAX_LAYERS
        || request.workload.context_tokens == 0
        || request.workload.batch_tokens == 0
        || request.workload.activation_payload_bytes == 0
        || request.workload.control_payload_bytes == 0
        || request.origin_peer_id.is_empty()
        || request.observed_at_unix_ms == 0
        || request.max_profile_age_ms == 0
        || request.min_peers == 0
    {
        return Err(invalid("invalid measured workload or policy"));
    }
    let mut ids = HashSet::new();
    for peer in &request.peers {
        if peer.id.is_empty() || !ids.insert(peer.id.as_str()) {
            return Err(invalid("peer IDs must be nonempty and unique"));
        }
        for range in &peer.ranges {
            if range.begin >= range.end
                || range.end > request.workload.layers
                || range.decode_us == 0
                || range.profile_id.is_empty()
            {
                return Err(invalid("invalid range observation"));
            }
        }
    }
    if !ids.contains(request.origin_peer_id.as_str()) {
        return Err(invalid("origin must be a candidate peer"));
    }
    for edge in &request.links {
        if edge.from == edge.to
            || !ids.contains(edge.from.as_str())
            || !ids.contains(edge.to.as_str())
            || edge.payload_bytes == 0
            || edge.transfer_us == 0
            || edge.profile_id.is_empty()
            || edge.path.is_empty()
        {
            return Err(invalid("invalid link observation"));
        }
    }
    let mut active = request
        .peers
        .iter()
        .filter(|p| p.authorized && p.healthy && fresh(p.capacity_observed_at_unix_ms, request))
        .collect::<Vec<_>>();
    active.sort_by(|a, b| a.id.cmp(&b.id));
    if active.len() > MAX_PEERS {
        return Err(invalid("exact search supports at most 8 eligible peers"));
    }
    if request.min_peers > active.len() || request.min_peers > request.workload.layers {
        return Err(MeasuredError::NoFeasiblePlacement);
    }
    let mut ranges = Vec::new();
    for peer in &active {
        let mut observed = HashMap::new();
        for range in &peer.ranges {
            if range.workload != request.workload || !fresh(range.measured_at_unix_ms, request) {
                continue;
            }
            let charged = range
                .peak_bytes
                .add(&peer.reserved_bytes)
                .ok_or(MeasuredError::Overflow)?;
            if !charged.fits(&peer.budget_bytes) || !charged.fits(&peer.available_bytes) {
                continue;
            }
            if observed
                .insert((range.begin, range.end), (range, charged))
                .is_some()
            {
                return Err(invalid("duplicate eligible range observation"));
            }
        }
        ranges.push(observed);
    }
    let mut links = HashMap::new();
    for edge in &request.links {
        if edge.workload != request.workload || !fresh(edge.measured_at_unix_ms, request) {
            continue;
        }
        if links
            .insert(
                (edge.from.clone(), edge.to.clone(), edge.payload_bytes),
                edge,
            )
            .is_some()
        {
            return Err(invalid("duplicate eligible link observation"));
        }
    }
    // (used-peer bitmask, final peer, completed layer prefix).
    let mut states: HashMap<(usize, usize, usize), Candidate> = HashMap::new();
    for (index, peer) in active.iter().enumerate() {
        if ring && peer.id != request.origin_peer_id {
            continue;
        }
        for (&(begin, end), (range, charged)) in &ranges[index] {
            if begin != 0 {
                continue;
            }
            let ingress = if peer.id == request.origin_peer_id {
                None
            } else {
                link(
                    &links,
                    &request.origin_peer_id,
                    &peer.id,
                    request.workload.control_payload_bytes,
                )
            };
            if peer.id != request.origin_peer_id && ingress.is_none() {
                continue;
            }
            let cost = range
                .decode_us
                .checked_add(ingress.map_or(0, |edge| edge.transfer_us))
                .ok_or(MeasuredError::Overflow)?;
            let candidate = Candidate {
                cost,
                assignments: vec![MeasuredAssignment {
                    peer_id: peer.id.clone(),
                    layers: begin..end,
                    range_profile_id: range.profile_id.clone(),
                    charged_bytes: charged.clone(),
                    decode_us: range.decode_us,
                }],
                links: ingress.map_or_else(Vec::new, |edge| vec![edge.profile_id.clone()]),
            };
            insert_cheaper(&mut states, (1 << index, index, end), candidate);
        }
    }
    let subset_limit = 1usize << active.len();
    for mask in 1..subset_limit {
        for last in 0..active.len() {
            for end in 1..request.workload.layers {
                let Some(previous) = states.get(&(mask, last, end)).cloned() else {
                    continue;
                };
                for (next, peer) in active.iter().enumerate() {
                    if mask & (1 << next) != 0 {
                        continue;
                    }
                    let Some(edge) = link(
                        &links,
                        &active[last].id,
                        &peer.id,
                        request.workload.activation_payload_bytes,
                    ) else {
                        continue;
                    };
                    for (&(begin, finish), (range, charged)) in &ranges[next] {
                        if begin != end {
                            continue;
                        }
                        let cost = previous
                            .cost
                            .checked_add(edge.transfer_us)
                            .and_then(|x| x.checked_add(range.decode_us))
                            .ok_or(MeasuredError::Overflow)?;
                        let mut candidate = previous.clone();
                        candidate.cost = cost;
                        candidate.links.push(edge.profile_id.clone());
                        candidate.assignments.push(MeasuredAssignment {
                            peer_id: peer.id.clone(),
                            layers: begin..finish,
                            range_profile_id: range.profile_id.clone(),
                            charged_bytes: charged.clone(),
                            decode_us: range.decode_us,
                        });
                        insert_cheaper(&mut states, (mask | (1 << next), next, finish), candidate);
                    }
                }
            }
        }
    }
    let mut best: Option<Candidate> = None;
    for ((mask, last, end), candidate) in states {
        let min_peers = if ring {
            request.min_peers.max(2)
        } else {
            request.min_peers
        };
        if end != request.workload.layers || (mask.count_ones() as usize) < min_peers {
            continue;
        }
        let peer = active[last];
        let feedback = if peer.id == request.origin_peer_id {
            None
        } else {
            link(
                &links,
                &peer.id,
                &request.origin_peer_id,
                request.workload.control_payload_bytes,
            )
        };
        if peer.id != request.origin_peer_id && feedback.is_none() {
            continue;
        }
        let mut candidate = candidate;
        candidate.cost = candidate
            .cost
            .checked_add(feedback.map_or(0, |edge| edge.transfer_us))
            .ok_or(MeasuredError::Overflow)?;
        if let Some(edge) = feedback {
            candidate.links.push(edge.profile_id.clone());
        }
        if best
            .as_ref()
            .is_none_or(|current| better(&candidate, current))
        {
            best = Some(candidate);
        }
    }
    let best = best.ok_or(MeasuredError::NoFeasiblePlacement)?;
    Ok(MeasuredPlacement {
        schema_version: 1,
        advisory: true,
        objective: "sequential_decode_latency".into(),
        planned_at_unix_ms: request.observed_at_unix_ms,
        max_profile_age_ms: request.max_profile_age_ms,
        workload: request.workload.clone(),
        origin_peer_id: request.origin_peer_id.clone(),
        assignments: best.assignments,
        link_profile_ids: best.links,
        predicted_decode_us: best.cost,
    })
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Recompute a saved advisory plan against current profiles before using it.
/// Callers must refresh `observed_at_unix_ms` and capacity first.
pub fn validate_output(
    request: &MeasuredRequest,
    output: &MeasuredPlacement,
) -> Result<(), MeasuredError> {
    if &plan(request)? == output {
        Ok(())
    } else {
        Err(invalid("saved measured plan differs from current profiles"))
    }
}

/// Recompute the saved output under the actual ring backend's topology.
pub fn validate_ring_output(
    request: &MeasuredRequest,
    output: &MeasuredPlacement,
) -> Result<(), MeasuredError> {
    if &plan_ring(request)? == output {
        Ok(())
    } else {
        Err(invalid(
            "saved ring plan differs from current profiles or topology",
        ))
    }
}

fn insert_cheaper(
    states: &mut HashMap<(usize, usize, usize), Candidate>,
    key: (usize, usize, usize),
    candidate: Candidate,
) {
    if states
        .get(&key)
        .is_none_or(|current| better(&candidate, current))
    {
        states.insert(key, candidate);
    }
}

fn better(candidate: &Candidate, current: &Candidate) -> bool {
    candidate.cost < current.cost
        || (candidate.cost == current.cost
            && candidate
                .assignments
                .iter()
                .map(|a| {
                    (
                        &a.peer_id,
                        a.layers.start,
                        a.layers.end,
                        &a.range_profile_id,
                    )
                })
                .cmp(current.assignments.iter().map(|a| {
                    (
                        &a.peer_id,
                        a.layers.start,
                        a.layers.end,
                        &a.range_profile_id,
                    )
                }))
                .is_lt())
}
