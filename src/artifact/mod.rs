//! Whole exported GGUF staging. A verified file is staged, never inference-ready.
//!
//! Catalogs must come from the mesh's chosen seed/pin. Hash verification establishes
//! byte identity; worker admission still owns metadata/runtime compatibility and
//! numerical qualification. Local imports are explicit, never remote file URLs.

mod cache;
pub mod cli;
mod gguf;
mod seed;
pub(crate) mod transport;

pub use cache::{ArtifactCache, Phase, Progress};
pub use seed::{Seed, SeedSpec};
pub use tokio_util::sync::CancellationToken;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPin {
    /// Immutable repository commit, not a moving branch name.
    pub revision: String,
    pub source_sha256: String,
    pub source_bytes: u64,
    pub architecture: String,
    /// Decoder layers, excluding MTP blocks.
    pub layers: u32,
}
impl ModelPin {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (40..=64).contains(&self.revision.len())
                && self.revision.bytes().all(|c| c.is_ascii_hexdigit()),
            "model revision must be an immutable commit hash"
        );
        validate_hash(&self.source_sha256)?;
        ensure!(
            self.source_bytes > 0 && self.layers > 0 && self.layers <= 4096,
            "invalid source size/layer count"
        );
        ensure!(
            ["llama", "qwen2", "deepseek2", "qwen35"].contains(&self.architecture.as_str()),
            "unsupported architecture"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    First,
    Middle,
    Final,
    Full,
}
impl Role {
    pub fn for_range(begin: u32, end: u32, layers: u32) -> Self {
        match (begin == 0, end == layers) {
            (true, true) => Self::Full,
            (true, false) => Self::First,
            (false, true) => Self::Final,
            (false, false) => Self::Middle,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "transport", rename_all = "snake_case", deny_unknown_fields)]
pub enum Location {
    Iroh { peer: crate::mesh::Peer },
    Https { url: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    pub model: ModelPin,
    pub begin: u32,
    pub end: u32,
    pub role: Role,
    pub bytes: u64,
    pub artifact_sha256: String,
    /// Hash of the exporter's sidecar, checked against actual GGUF metadata/tensors
    /// by the seed. This is provenance, not numerical qualification.
    pub export_report_sha256: String,
    pub location: Location,
}
impl Manifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.version == 1, "unsupported artifact manifest version");
        self.model.validate()?;
        validate_hash(&self.artifact_sha256)?;
        validate_hash(&self.export_report_sha256)?;
        ensure!(
            self.begin < self.end && self.end <= self.model.layers,
            "invalid artifact layer range"
        );
        ensure!(
            self.role == Role::for_range(self.begin, self.end, self.model.layers),
            "artifact role does not match range"
        );
        ensure!(
            self.bytes > 0 && self.bytes <= self.model.source_bytes,
            "invalid artifact size"
        );
        match &self.location {
            Location::Https { url } => {
                let url = reqwest::Url::parse(url)?;
                ensure!(
                    url.scheme() == "https"
                        && url.host_str().is_some()
                        && url.username().is_empty()
                        && url.password().is_none()
                        && url.fragment().is_none(),
                    "artifact URL must be HTTPS without credentials or fragment"
                );
            }
            Location::Iroh { peer } => {
                let _: iroh::EndpointId = peer.id.parse()?;
                ensure!(
                    peer.addresses.len() <= 16 && peer.invite.is_empty(),
                    "invalid artifact peer address"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub version: u32,
    pub model: ModelPin,
    pub shards: Vec<Manifest>,
}
impl Catalog {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == 1 && !self.shards.is_empty() && self.shards.len() <= 4096,
            "invalid catalog"
        );
        self.model.validate()?;
        let mut hashes = std::collections::BTreeSet::new();
        let mut ranges = std::collections::BTreeSet::new();
        for shard in &self.shards {
            shard.validate()?;
            ensure!(shard.model == self.model, "catalog mixes model pins");
            ensure!(
                hashes.insert(&shard.artifact_sha256),
                "duplicate catalog artifact"
            );
            ensure!(
                ranges.insert((shard.begin, shard.end)),
                "duplicate catalog layer range"
            );
        }
        Ok(())
    }
    /// Caller still decides whether this candidate fits the device and profiling role.
    pub fn smallest_candidate(&self) -> Result<&Manifest> {
        self.validate()?;
        Ok(self.shards.iter().min_by_key(|s| s.bytes).unwrap())
    }
    pub fn exact_range(&self, begin: u32, end: u32) -> Result<&Manifest> {
        self.validate()?;
        self.shards
            .iter()
            .find(|s| s.begin == begin && s.end == end)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "seed has no exported shard for {begin}..{end}; export it with export_prefix.py"
                )
            })
    }
}

pub(crate) fn validate_hash(hash: &str) -> Result<()> {
    ensure!(
        hash.len() == 64
            && hash
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
        "SHA-256 must be 64 lowercase hex characters"
    );
    Ok(())
}

#[cfg(test)]
mod tests;
