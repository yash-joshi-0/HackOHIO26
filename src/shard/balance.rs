//! Memory-weighted layer balancing: how many contiguous layers each node serves.
//!
//! [`plan_layers`] is the pure algorithm behind balanced shard manifests (M-28). Each node
//! brings a memory budget (what it can spend on this generation: free memory minus a safety
//! reserve) and a fixed per-stage overhead (runner scratch, plus the output head for the final
//! node). What remains is the node's **weight**. Layers are handed out one at a time with smooth
//! weighted round robin (the nginx scheme: every round each eligible node's `current` grows by
//! its weight, the node with the largest `current` takes the layer, and its `current` drops by
//! the round's total weight), so counts end up proportional to the weights while never giving
//! a node more layers than fit in its budget. A node that cannot fit one layer gets none.
//!
//! The counts are then laid out as contiguous ranges in node order, so the caller lists nodes
//! in a stable order with the final node (which tokenizes, samples and owns the output head)
//! last. The final node always gets at least one layer. Whichever node ends up owning layer 0
//! also loads the token embedding (`first_stage_bytes`): every non-final node is charged it up
//! front, since any of them may end up first, and the final node only when it is the sole node
//! able to hold a layer. If every layer would otherwise land on the final node, one layer is
//! moved to the largest non-final node that can hold it, so the final node never needs room
//! for the embedding as well.
//!
//! Each layer is charged the largest per-layer cost, which is exact for the uniform decoder
//! layers of supported architectures and conservative otherwise. The result is deterministic:
//! the same inputs always give the same ranges (ties go to the earlier node).

use std::cmp::Reverse;

/// One node's memory, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeMemory {
    /// What the node may spend on this generation: free memory minus the safety reserve.
    pub budget_bytes: u64,
    /// Bytes the node's stage needs whatever its range: runner scratch and, for the final
    /// node, the output norm and head. The token embedding is charged separately.
    pub fixed_bytes: u64,
}

/// What [`plan_layers`] balances.
#[derive(Clone, Copy, Debug)]
pub struct BalanceInput<'a> {
    /// Nodes in layout order; the last one is the final stage.
    pub nodes: &'a [NodeMemory],
    /// Memory each decoder layer costs (weights plus its KV cache for the context), one entry
    /// per layer. Its length is the model's layer count.
    pub layer_bytes: &'a [u64],
    /// Extra bytes for whichever node owns layer 0 (the token embedding).
    pub first_stage_bytes: u64,
}

/// One node's share of the layers: the half-open range `[begin, end)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerAssignment {
    /// Index of the node in [`BalanceInput::nodes`].
    pub node: usize,
    /// First layer (inclusive).
    pub begin: u32,
    /// End of the range (exclusive).
    pub end: u32,
}

impl LayerAssignment {
    /// Layers in the range.
    pub fn layers(&self) -> u32 {
        self.end - self.begin
    }
}

/// Why no plan exists.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BalanceError {
    /// There are no nodes or no layers to balance.
    #[error("nothing to balance: {nodes} nodes and {layers} layers")]
    Empty {
        /// Nodes given.
        nodes: usize,
        /// Layers given.
        layers: usize,
    },
    /// The final node cannot hold even one layer, but it must own the last one.
    #[error(
        "the final stage cannot fit one layer: {budget_bytes} bytes of budget, {needed_bytes} \
         needed for its overhead plus one {layer_bytes}-byte layer"
    )]
    FinalStageTooSmall {
        /// The final node's budget.
        budget_bytes: u64,
        /// Its overhead plus one layer.
        needed_bytes: u64,
        /// The cost charged per layer.
        layer_bytes: u64,
    },
    /// Together the nodes cannot hold every layer.
    #[error(
        "not enough memory for {layers} layers: the nodes can hold {capacity} of them \
         ({layer_bytes} bytes each, after reserves and per-stage overheads)"
    )]
    InsufficientMemory {
        /// Layers in the model.
        layers: u32,
        /// Layers the nodes could hold between them.
        capacity: u64,
        /// The cost charged per layer.
        layer_bytes: u64,
    },
}

/// Splits the layers between `input.nodes` in proportion to their free memory, never past a
/// node's budget; see the module docs. Returns one assignment per node that gets layers, in
/// node order, covering `0..layers` contiguously, with the final node last.
pub fn plan_layers(input: &BalanceInput<'_>) -> Result<Vec<LayerAssignment>, BalanceError> {
    let layers = layer_count(input)?;
    let unit = input.layer_bytes.iter().copied().max().unwrap_or(0).max(1);
    let final_index = input.nodes.len() - 1;
    let mut caps: Vec<NodeCap> = input
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let first = if index == final_index {
                0
            } else {
                input.first_stage_bytes
            };
            NodeCap::new(node, first, unit)
        })
        .collect();
    let others_can_start = caps[..final_index].iter().any(|cap| cap.max_layers > 0);
    if !others_can_start {
        // The final node will own layer 0 too, so it must also hold the embedding.
        caps[final_index] = NodeCap::new(&input.nodes[final_index], input.first_stage_bytes, unit);
    }
    check_capacity(input.nodes, &caps, layers, unit)?;
    let mut counts = weighted_round_robin(&caps, final_index, layers);
    if counts[final_index] == u64::from(layers) && others_can_start {
        hand_one_layer_to_a_non_final_node(&caps, final_index, &mut counts);
    }
    Ok(contiguous_ranges(&counts))
}

/// A node's weight and how many layers fit in its budget.
#[derive(Clone, Copy, Debug)]
struct NodeCap {
    /// Bytes the node needs before any layer.
    overhead: u64,
    /// Bytes left for layers after the node's overheads.
    weight: u64,
    /// Layers that fit in `weight`.
    max_layers: u64,
}

impl NodeCap {
    fn new(node: &NodeMemory, first_stage_bytes: u64, unit: u64) -> Self {
        let overhead = node.fixed_bytes.saturating_add(first_stage_bytes);
        let weight = node.budget_bytes.saturating_sub(overhead);
        Self {
            overhead,
            weight,
            max_layers: weight / unit,
        }
    }
}

fn layer_count(input: &BalanceInput<'_>) -> Result<u32, BalanceError> {
    let empty = BalanceError::Empty {
        nodes: input.nodes.len(),
        layers: input.layer_bytes.len(),
    };
    if input.nodes.is_empty() || input.layer_bytes.is_empty() {
        return Err(empty);
    }
    u32::try_from(input.layer_bytes.len()).map_err(|_| empty)
}

/// Fails unless the final node fits a layer and all nodes together fit every layer.
fn check_capacity(
    nodes: &[NodeMemory],
    caps: &[NodeCap],
    layers: u32,
    unit: u64,
) -> Result<(), BalanceError> {
    let final_index = caps.len() - 1;
    if caps[final_index].max_layers == 0 {
        return Err(BalanceError::FinalStageTooSmall {
            budget_bytes: nodes[final_index].budget_bytes,
            needed_bytes: caps[final_index].overhead.saturating_add(unit),
            layer_bytes: unit,
        });
    }
    let capacity: u64 = caps
        .iter()
        .map(|cap| cap.max_layers.min(u64::from(layers)))
        .sum();
    if capacity < u64::from(layers) {
        return Err(BalanceError::InsufficientMemory {
            layers,
            capacity,
            layer_bytes: unit,
        });
    }
    Ok(())
}

/// Hands out `layers` one at a time by smooth weighted round robin among nodes with room left.
/// The final node takes the first layer unconditionally, since it must own at least one.
/// The caller has checked that the nodes have room for every layer.
fn weighted_round_robin(caps: &[NodeCap], final_index: usize, layers: u32) -> Vec<u64> {
    let mut counts = vec![0u64; caps.len()];
    let mut current = vec![0i128; caps.len()];
    counts[final_index] = 1;
    for _ in 1..layers {
        let eligible: Vec<usize> = (0..caps.len())
            .filter(|&index| counts[index] < caps[index].max_layers)
            .collect();
        let total: i128 = eligible
            .iter()
            .map(|&index| i128::from(caps[index].weight))
            .sum();
        for &index in &eligible {
            current[index] += i128::from(caps[index].weight);
        }
        // Largest `current` wins; ties go to the earlier node.
        let Some(&picked) = eligible
            .iter()
            .max_by_key(|&&index| (current[index], Reverse(index)))
        else {
            break;
        };
        current[picked] -= total;
        counts[picked] += 1;
    }
    counts
}

/// Moves one layer from the final node to the non-final node with the largest weight that can
/// hold one (the earliest such node on a tie).
fn hand_one_layer_to_a_non_final_node(caps: &[NodeCap], final_index: usize, counts: &mut [u64]) {
    let receiver = (0..final_index)
        .filter(|&index| caps[index].max_layers > 0)
        .max_by_key(|&index| (caps[index].weight, Reverse(index)));
    if let Some(receiver) = receiver {
        counts[receiver] += 1;
        counts[final_index] -= 1;
    }
}

/// Lays per-node counts out as contiguous ranges in node order, skipping nodes with none.
fn contiguous_ranges(counts: &[u64]) -> Vec<LayerAssignment> {
    let mut next = 0u32;
    let mut assignments = Vec::new();
    for (node, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        // Counts sum to the layer count, which fits a u32.
        let count = u32::try_from(count).unwrap_or(u32::MAX);
        assignments.push(LayerAssignment {
            node,
            begin: next,
            end: next + count,
        });
        next += count;
    }
    assignments
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;
    const MIB: u64 = 1 << 20;

    fn node(budget_bytes: u64, fixed_bytes: u64) -> NodeMemory {
        NodeMemory {
            budget_bytes,
            fixed_bytes,
        }
    }

    fn plan(
        nodes: &[NodeMemory],
        layers: usize,
        layer_bytes: u64,
        first: u64,
    ) -> Vec<(usize, u32, u32)> {
        let layer_bytes = vec![layer_bytes; layers];
        let input = BalanceInput {
            nodes,
            layer_bytes: &layer_bytes,
            first_stage_bytes: first,
        };
        plan_layers(&input)
            .expect("a plan")
            .iter()
            .map(|assignment| (assignment.node, assignment.begin, assignment.end))
            .collect()
    }

    fn error(nodes: &[NodeMemory], layers: usize, layer_bytes: u64, first: u64) -> BalanceError {
        let layer_bytes = vec![layer_bytes; layers];
        plan_layers(&BalanceInput {
            nodes,
            layer_bytes: &layer_bytes,
            first_stage_bytes: first,
        })
        .expect_err("no plan")
    }

    /// The worked example in docs/layer-balancing.md: a Raspberry Pi with 2 GB free, a laptop
    /// with 8 GB and a workstation (the leader, final) with 48 GB, over 64 layers of a 32B
    /// model. Reserve 512 MiB and stage overhead 256 MiB everywhere; 400 MB of embedding for
    /// the first stage and 400 MB of output head for the final one.
    fn pi_laptop_workstation() -> (Vec<NodeMemory>, u64, u64) {
        let reserve = 512 * MIB;
        let overhead = 256 * MIB;
        let output = 400_000_000;
        let nodes = vec![
            node(2 * GB - reserve, overhead),
            node(8 * GB - reserve, overhead),
            node(48 * GB - reserve, overhead + output),
        ];
        let layer = 300_000_000 + 4096 * 512; // weights + KV for a 512-token context
        (nodes, layer, 400_000_000)
    }

    #[test]
    fn the_pi_gets_few_layers_and_the_workstation_most() {
        let (nodes, layer, first) = pi_laptop_workstation();
        let ranges = plan(&nodes, 64, layer, first);
        assert_eq!(ranges, [(0, 0, 1), (1, 1, 9), (2, 9, 64)]);
    }

    #[test]
    fn shares_follow_the_weights() {
        // Weights 1 : 3 : 4 over 64 layers: 8, 24, 32.
        let nodes = [node(1000, 0), node(3000, 0), node(4000, 0)];
        let ranges = plan(&nodes, 64, 10, 0);
        assert_eq!(ranges, [(0, 0, 8), (1, 8, 32), (2, 32, 64)]);
    }

    #[test]
    fn memory_caps_are_never_exceeded() {
        let layer = 7;
        for budgets in [[30, 500, 100], [8, 8, 1000], [70, 70, 70], [15, 1000, 22]] {
            let nodes: Vec<NodeMemory> = budgets.iter().map(|&budget| node(budget, 1)).collect();
            let Ok(assignments) = plan_layers(&BalanceInput {
                nodes: &nodes,
                layer_bytes: &[layer; 20],
                first_stage_bytes: 3,
            }) else {
                continue;
            };
            let final_index = nodes.len() - 1;
            for assignment in &assignments {
                let mut used =
                    nodes[assignment.node].fixed_bytes + u64::from(assignment.layers()) * layer;
                if assignment.begin == 0 {
                    used += 3;
                }
                assert!(
                    used <= nodes[assignment.node].budget_bytes,
                    "{budgets:?}: node {} uses {used}",
                    assignment.node
                );
            }
            assert_eq!(assignments.last().map(|last| last.node), Some(final_index));
        }
    }

    #[test]
    fn a_node_that_cannot_fit_one_layer_gets_none() {
        let nodes = [node(99, 0), node(10_000, 0), node(10_000, 0)];
        let ranges = plan(&nodes, 10, 100, 0);
        assert_eq!(ranges, [(1, 0, 5), (2, 5, 10)]);
        // Its overhead alone can push it below one layer.
        let nodes = [node(150, 60), node(1000, 0)];
        assert_eq!(plan(&nodes, 4, 100, 0), [(1, 0, 4)]);
    }

    #[test]
    fn caps_push_the_rest_to_nodes_with_room() {
        // Equal weights would split 2 / 2 anyway; a capped heavy node passes the rest on.
        let nodes = [node(200, 0), node(250, 0)];
        assert_eq!(plan(&nodes, 4, 100, 0), [(0, 0, 2), (1, 2, 4)]);
        let nodes = [node(1_000_000, 0), node(300, 0)];
        assert_eq!(plan(&nodes, 8, 100, 0), [(0, 0, 7), (1, 7, 8)]);
        // A tiny share rounds to nothing when another non-final node already holds layer 0.
        let nodes = [node(300, 0), node(1_000_000, 0), node(10_000, 0)];
        assert_eq!(plan(&nodes, 20, 100, 0), [(1, 0, 19), (2, 19, 20)]);
    }

    #[test]
    fn the_result_is_deterministic() {
        let (nodes, layer, first) = pi_laptop_workstation();
        let first_plan = plan(&nodes, 64, layer, first);
        for _ in 0..10 {
            assert_eq!(plan(&nodes, 64, layer, first), first_plan);
        }
        // Equal weights split evenly, ties going to the earlier node.
        let nodes = [node(1000, 0), node(1000, 0), node(1000, 0)];
        assert_eq!(plan(&nodes, 4, 10, 0), [(0, 0, 1), (1, 1, 2), (2, 2, 4)]);
    }

    #[test]
    fn insufficient_total_memory_is_an_error() {
        let nodes = [node(250, 0), node(250, 0)];
        assert_eq!(
            error(&nodes, 5, 100, 0),
            BalanceError::InsufficientMemory {
                layers: 5,
                capacity: 4,
                layer_bytes: 100
            }
        );
        let message = error(&nodes, 5, 100, 0).to_string();
        assert!(
            message.contains("not enough memory for 5 layers"),
            "{message}"
        );
    }

    #[test]
    fn a_final_node_that_cannot_fit_a_layer_is_an_error() {
        let nodes = [node(10_000, 0), node(150, 100)];
        assert!(matches!(
            error(&nodes, 4, 100, 0),
            BalanceError::FinalStageTooSmall { .. }
        ));
    }

    #[test]
    fn nothing_to_balance_is_an_error() {
        assert!(matches!(error(&[], 4, 1, 0), BalanceError::Empty { .. }));
        assert!(matches!(
            error(&[node(10, 0)], 0, 1, 0),
            BalanceError::Empty { .. }
        ));
    }

    #[test]
    fn a_single_node_takes_everything_including_the_embedding() {
        assert_eq!(plan(&[node(1000, 100)], 8, 100, 0), [(0, 0, 8)]);
        // 8 layers + 100 overhead fit, but not with a 200-byte embedding as well.
        let error = error(&[node(1000, 100)], 8, 100, 200);
        assert!(
            matches!(error, BalanceError::InsufficientMemory { capacity: 7, .. }),
            "{error:?}"
        );
    }

    #[test]
    fn the_final_node_is_always_last_and_never_holds_the_embedding_needlessly() {
        // The final node's weight dwarfs the other, which would get no layer at all; one is
        // handed over so the final node never has to hold the embedding too.
        let nodes = [node(150, 0), node(1_000_000, 0)];
        assert_eq!(plan(&nodes, 16, 100, 40), [(0, 0, 1), (1, 1, 16)]);
        // The final node gets a layer even with the smallest weight.
        let nodes = [node(1_000_000, 0), node(100, 0)];
        assert_eq!(plan(&nodes, 16, 100, 0), [(0, 0, 15), (1, 15, 16)]);
    }

    #[test]
    fn uneven_layers_are_charged_at_the_largest_cost() {
        // The first node could hold layers 0 and 1 (200 bytes) but is charged 150 per layer,
        // so it counts as holding one; the conservative charge never overcommits a node.
        let nodes = [node(299, 0), node(300, 0)];
        let input = BalanceInput {
            nodes: &nodes,
            layer_bytes: &[50, 150, 100, 100],
            first_stage_bytes: 0,
        };
        assert!(matches!(
            plan_layers(&input),
            Err(BalanceError::InsufficientMemory {
                capacity: 3,
                layer_bytes: 150,
                ..
            })
        ));
    }
}
