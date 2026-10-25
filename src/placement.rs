//! Plan contiguous layers across a measured directed graph of trusted devices.
//! All costs must come from measurements for the same model/runtime/context.
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::ops::Range;

/// Maximum eligible peers in the exact joint order-and-range search.
const MAX_SEARCH_PEERS: usize = 8;
/// Dense search states grow with layer count; keep their allocation bounded.
const MAX_SEARCH_LAYERS: usize = 256;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProfile {
    pub model_hash: String,
    pub runtime_id: String,
    pub context_tokens: u64,
    pub layers: Vec<LayerMemory>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LayerMemory {
    pub weight_bytes: u64,
    pub kv_bytes_per_token: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PeerProfile {
    pub id: String,
    pub model_hash: String,
    pub runtime_id: String,
    pub context_tokens: u64,
    pub authorized: bool,
    pub healthy: bool,
    /// Opt-in application budget, not total physical/free memory.
    pub memory_budget_bytes: u64,
    /// Conservative non-layer allocation: runtime scratch, headroom, and any
    /// embedding/output tensors owned by this device. Reserve even if unused.
    pub reserved_bytes: u64,
    pub decode_us_per_layer: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LinkProfile {
    pub from: String,
    pub to: String,
    /// Measured per-token transfer overhead for this ordered pair. For RPC,
    /// use observed controller-mediated overhead, not a fictitious direct link.
    pub transfer_us: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Assignment {
    pub peer_id: String,
    pub layers: Range<usize>,
    pub estimated_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    pub model_hash: String,
    pub runtime_id: String,
    pub context_tokens: u64,
    pub assignments: Vec<Assignment>,
    pub predicted_decode_us: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    InvalidInput(String),
    Overflow,
    NoFeasiblePlacement,
}

#[derive(Clone)]
struct Candidate {
    cost: u64,
    assignments: Vec<Assignment>,
}

type SearchStates = Vec<Vec<Vec<Option<Candidate>>>>;

struct SearchContext<'a> {
    peers: &'a [PeerProfile],
    active: &'a [usize],
    memory: &'a [u64],
    layer_costs: Vec<Vec<u64>>,
    link_costs: Vec<Vec<Option<u64>>>,
}

struct SearchPosition {
    subset: usize,
    last: usize,
    start: usize,
}

/// Jointly chooses directed peer order and contiguous layer ranges for decode latency.
/// Each transition needs a measured link; memory budgets are hard constraints.
/// Returns estimates only; a backend must verify actual allocation after load.
pub fn plan(
    model: &ModelProfile,
    peers: &[PeerProfile],
    links: &[LinkProfile],
    min_peers: usize,
) -> Result<Placement, PlanError> {
    plan_with_final_peer(model, peers, links, min_peers, None)
}

/// Like `plan`, but optionally require the final block group on a specific peer.
/// This constrains the advisory placement; runtime output tensor ownership still
/// requires post-load verification.
pub fn plan_with_final_peer(
    model: &ModelProfile,
    peers: &[PeerProfile],
    links: &[LinkProfile],
    min_peers: usize,
    final_peer_id: Option<&str>,
) -> Result<Placement, PlanError> {
    let invalid = |s: &str| PlanError::InvalidInput(s.to_owned());
    if let Some(id) = final_peer_id
        && !peers.iter().any(|peer| peer.id == id)
    {
        return Err(invalid("required final peer must be a known candidate"));
    }
    let n = model.layers.len();
    if n == 0
        || model.model_hash.is_empty()
        || model.runtime_id.is_empty()
        || model.context_tokens == 0
        || min_peers == 0
    {
        return Err(invalid(
            "model identity, context, layers and min_peers must be nonempty",
        ));
    }
    if n > MAX_SEARCH_LAYERS {
        return Err(invalid(
            "exact placement search supports at most 256 layers",
        ));
    }
    let mut ids = HashSet::new();
    for p in peers {
        if p.id.is_empty() || !ids.insert(p.id.as_str()) {
            return Err(invalid("peer IDs must be nonempty and unique"));
        }
        if p.decode_us_per_layer.len() != n || p.decode_us_per_layer.contains(&0) {
            return Err(invalid(
                "each peer needs positive measured cost for every layer",
            ));
        }
    }
    let mut link_ids = HashSet::new();
    for l in links {
        if l.from == l.to
            || !ids.contains(l.from.as_str())
            || !ids.contains(l.to.as_str())
            || !link_ids.insert((l.from.as_str(), l.to.as_str()))
        {
            return Err(invalid(
                "links must reference distinct known peers without duplicates",
            ));
        }
    }
    if min_peers > peers.len() || min_peers > n {
        return Err(PlanError::NoFeasiblePlacement);
    }
    let memory = layer_memory_prefix(model)?;
    let active = eligible_peers(model, peers);
    if min_peers > active.len() {
        return Err(PlanError::NoFeasiblePlacement);
    }
    if active.len() > MAX_SEARCH_PEERS {
        return Err(invalid(
            "exact peer-order search supports at most 8 eligible peers",
        ));
    }

    let best = search_topology(
        model,
        peers,
        links,
        &memory,
        &active,
        min_peers,
        final_peer_id,
    )?;
    Ok(Placement {
        model_hash: model.model_hash.clone(),
        runtime_id: model.runtime_id.clone(),
        context_tokens: model.context_tokens,
        assignments: best.assignments.clone(),
        predicted_decode_us: best.cost,
    })
}

fn layer_memory_prefix(model: &ModelProfile) -> Result<Vec<u64>, PlanError> {
    let mut memory = vec![0u64];
    for layer in &model.layers {
        if layer.weight_bytes == 0 {
            return Err(PlanError::InvalidInput(
                "layer weight bytes must be positive".into(),
            ));
        }
        let kv_bytes = layer
            .kv_bytes_per_token
            .checked_mul(model.context_tokens)
            .ok_or(PlanError::Overflow)?;
        let layer_bytes = layer
            .weight_bytes
            .checked_add(kv_bytes)
            .ok_or(PlanError::Overflow)?;
        let next = memory
            .last()
            .copied()
            .and_then(|current| current.checked_add(layer_bytes))
            .ok_or(PlanError::Overflow)?;
        memory.push(next);
    }
    Ok(memory)
}

fn eligible_peers(model: &ModelProfile, peers: &[PeerProfile]) -> Vec<usize> {
    let mut active = peers
        .iter()
        .enumerate()
        .filter(|(_, peer)| {
            peer.authorized
                && peer.healthy
                && peer.model_hash == model.model_hash
                && peer.runtime_id == model.runtime_id
                && peer.context_tokens == model.context_tokens
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    active.sort_by(|left, right| peers[*left].id.cmp(&peers[*right].id));
    active
}

fn search_topology(
    model: &ModelProfile,
    peers: &[PeerProfile],
    links: &[LinkProfile],
    memory: &[u64],
    active: &[usize],
    min_peers: usize,
    final_peer_id: Option<&str>,
) -> Result<Candidate, PlanError> {
    let peer_count = active.len();
    let layer_count = model.layers.len();
    let subset_count = 1usize
        .checked_shl(peer_count as u32)
        .ok_or(PlanError::Overflow)?;
    let context = SearchContext {
        peers,
        active,
        memory,
        layer_costs: decode_cost_prefixes(peers, active, layer_count)?,
        link_costs: directed_link_costs(peers, links, active),
    };
    let mut states = vec![vec![vec![None::<Candidate>; layer_count + 1]; peer_count]; subset_count];

    seed_search_states(&context, &mut states)?;
    extend_search_states(&context, &mut states)?;
    select_best_candidate(&context, &states, min_peers, final_peer_id)
}

fn directed_link_costs(
    peers: &[PeerProfile],
    links: &[LinkProfile],
    active: &[usize],
) -> Vec<Vec<Option<u64>>> {
    let mut costs = vec![vec![None; active.len()]; active.len()];
    for link in links {
        let from = active
            .iter()
            .position(|index| peers[*index].id == link.from);
        let to = active.iter().position(|index| peers[*index].id == link.to);
        if let (Some(from), Some(to)) = (from, to) {
            costs[from][to] = Some(link.transfer_us);
        }
    }
    costs
}

fn decode_cost_prefixes(
    peers: &[PeerProfile],
    active: &[usize],
    layer_count: usize,
) -> Result<Vec<Vec<u64>>, PlanError> {
    let mut costs = vec![vec![0u64; layer_count + 1]; active.len()];
    for (active_index, peer_index) in active.iter().enumerate() {
        for layer in 0..layer_count {
            costs[active_index][layer + 1] = costs[active_index][layer]
                .checked_add(peers[*peer_index].decode_us_per_layer[layer])
                .ok_or(PlanError::Overflow)?;
        }
    }
    Ok(costs)
}

#[allow(clippy::needless_range_loop)] // Indices address matching peer and layer dimensions.
fn seed_search_states(
    context: &SearchContext<'_>,
    states: &mut SearchStates,
) -> Result<(), PlanError> {
    for active_index in 0..context.active.len() {
        let peer = &context.peers[context.active[active_index]];
        for end in 1..context.memory.len() {
            let bytes = context.memory[end]
                .checked_add(peer.reserved_bytes)
                .ok_or(PlanError::Overflow)?;
            if bytes > peer.memory_budget_bytes {
                break;
            }
            states[1 << active_index][active_index][end] = Some(Candidate {
                cost: context.layer_costs[active_index][end],
                assignments: vec![Assignment {
                    peer_id: peer.id.clone(),
                    layers: 0..end,
                    estimated_bytes: bytes,
                }],
            });
        }
    }
    Ok(())
}

fn extend_search_states(
    context: &SearchContext<'_>,
    states: &mut SearchStates,
) -> Result<(), PlanError> {
    let layer_count = context.memory.len() - 1;
    for subset in 1..states.len() {
        if subset.count_ones() as usize >= layer_count {
            continue;
        }
        for last in 0..context.active.len() {
            if subset & (1 << last) == 0 {
                continue;
            }
            for start in 1..layer_count {
                let Some(previous) = states[subset][last][start].clone() else {
                    continue;
                };
                extend_from_state(
                    context,
                    states,
                    SearchPosition {
                        subset,
                        last,
                        start,
                    },
                    previous,
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::needless_range_loop)] // Indices address matching peer and layer dimensions.
fn extend_from_state(
    context: &SearchContext<'_>,
    states: &mut SearchStates,
    position: SearchPosition,
    previous: Candidate,
) -> Result<(), PlanError> {
    for next in 0..context.active.len() {
        if position.subset & (1 << next) != 0 {
            continue;
        }
        let Some(link_cost) = context.link_costs[position.last][next] else {
            continue;
        };
        let peer = &context.peers[context.active[next]];
        let next_subset = position.subset | (1 << next);
        for end in position.start + 1..context.memory.len() {
            let bytes = context.memory[end]
                .checked_sub(context.memory[position.start])
                .and_then(|layer_bytes| layer_bytes.checked_add(peer.reserved_bytes))
                .ok_or(PlanError::Overflow)?;
            if bytes > peer.memory_budget_bytes {
                break;
            }
            let segment_cost =
                context.layer_costs[next][end] - context.layer_costs[next][position.start];
            let cost = previous
                .cost
                .checked_add(link_cost)
                .and_then(|total| total.checked_add(segment_cost))
                .ok_or(PlanError::Overflow)?;
            let slot = &mut states[next_subset][next][end];
            if slot.as_ref().is_none_or(|best| cost < best.cost) {
                let mut assignments = previous.assignments.clone();
                assignments.push(Assignment {
                    peer_id: peer.id.clone(),
                    layers: position.start..end,
                    estimated_bytes: bytes,
                });
                *slot = Some(Candidate { cost, assignments });
            }
        }
    }
    Ok(())
}

#[allow(clippy::needless_range_loop)] // The peer index is also used in the bitmask and context.
fn select_best_candidate(
    context: &SearchContext<'_>,
    states: &[Vec<Vec<Option<Candidate>>>],
    min_peers: usize,
    final_peer_id: Option<&str>,
) -> Result<Candidate, PlanError> {
    let layer_count = states[0][0].len() - 1;
    let mut best: Option<&Candidate> = None;
    for (subset, peer_states) in states.iter().enumerate().skip(1) {
        if (subset.count_ones() as usize) < min_peers {
            continue;
        }
        for last in 0..context.active.len() {
            if subset & (1 << last) == 0
                || final_peer_id.is_some_and(|id| context.peers[context.active[last]].id != id)
            {
                continue;
            }
            if let Some(candidate) = &peer_states[last][layer_count]
                && best.is_none_or(|current| candidate.cost < current.cost)
            {
                best = Some(candidate);
            }
        }
    }
    best.cloned().ok_or(PlanError::NoFeasiblePlacement)
}
