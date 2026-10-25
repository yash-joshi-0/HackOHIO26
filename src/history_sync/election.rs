use crate::history_sync::history::{Epoch, NodeId, Seq};

/// What a surviving peer reports when the main process is lost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub id: NodeId,
    /// Highest epoch this peer has heard of; the new epoch must be above every one of these.
    pub epoch: Epoch,
    /// Epoch its history was written under (`History::last_epoch`); used for ranking.
    pub history_epoch: Epoch,
    pub len: Seq,
    /// When this peer last connected to the mesh, from `PeerInfo::connected_at_unix_ms`.
    /// Earlier means connected longer.
    pub connected_at_unix_ms: u64,
}

/// Outcome of a failover election.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Elected {
    pub id: NodeId,
    pub epoch: Epoch,
}

/// Picks the replacement main process: the peer with the most up-to-date history
/// (newest history epoch, then most entries), then whichever peer has been
/// connected the longest, then the lowest node ID. Every peer that sees the same
/// candidate set picks the same winner. The new epoch is one above the highest
/// epoch any candidate has heard of. Returns `None` when there are no candidates or the highest
/// observed epoch is exhausted and cannot be incremented safely.
///
/// Ranking uses the history's epoch, not the highest epoch heard (Raft's "last log
/// term" rule): a peer that learned of a newer epoch but never received that
/// leader's entries isn't more up to date than its history. History comes before
/// connection time so the new leader never has fewer entries than a peer that
/// could have won instead; connection time is a wall-clock reading and devices'
/// clocks can differ, so it only breaks ties.
///
/// Prototype only: assumes survivors agree on the candidate set. A real mesh
/// needs a consensus round or a quorum to rule out two simultaneous winners.
pub fn elect(candidates: &[Candidate]) -> Option<Elected> {
    let max_epoch = candidates
        .iter()
        .map(|c| c.epoch.max(c.history_epoch))
        .max()?;
    let epoch = max_epoch.checked_add(1)?;
    let winner = candidates.iter().max_by(|a, b| {
        (a.history_epoch, a.len)
            .cmp(&(b.history_epoch, b.len))
            .then(b.connected_at_unix_ms.cmp(&a.connected_at_unix_ms))
            .then(b.id.cmp(&a.id))
    })?;
    Some(Elected {
        id: winner.id,
        epoch,
    })
}
