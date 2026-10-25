use super::*;
use anyhow::{Context, bail};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use tokio::fs;
use tokio::io::AsyncReadExt;

const MAX_EXPORT_REPORT_BYTES: usize = 8 * 1024 * 1024;

async fn read_export_report(path: &Path) -> Result<Vec<u8>> {
    let file = fs::File::open(path).await?;
    let metadata = file.metadata().await?;
    ensure!(metadata.is_file(), "export report must be a regular file");
    ensure!(
        metadata.len() <= MAX_EXPORT_REPORT_BYTES as u64,
        "export report too large"
    );
    let mut bytes = Vec::new();
    file.take(MAX_EXPORT_REPORT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .await?;
    ensure!(
        bytes.len() <= MAX_EXPORT_REPORT_BYTES,
        "export report too large"
    );
    Ok(bytes)
}

/// Private operator configuration. Paths never appear in the public catalog.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedSpec {
    pub model: ModelPin,
    pub source: PathBuf,
    /// Explicit existing export paths; each requires its adjacent .gguf.json report.
    pub shards: Vec<PathBuf>,
}

#[derive(Deserialize)]
pub(super) struct ExportReport {
    pub source_sha256: String,
    pub artifact_sha256: String,
    pub begin: u32,
    pub end: u32,
    pub layers: u32,
    pub architecture: String,
    pub source_bytes: u64,
    pub bytes: u64,
    pub tensors: Vec<Tensor>,
}
#[derive(Deserialize)]
pub(super) struct Tensor {
    pub name: String,
    pub bytes: u64,
}

#[derive(Clone)]
pub struct Seed {
    catalog: Catalog,
    files: BTreeMap<String, PathBuf>,
}
impl Seed {
    /// Verify full source and every selected existing export before advertising it.
    /// Does not export, load a model, start inference, or modify the source files.
    pub async fn verify(
        spec: SeedSpec,
        peer: crate::mesh::Peer,
        cancel: &CancellationToken,
    ) -> Result<Self> {
        spec.model.validate()?;
        ensure!(
            !spec.shards.is_empty() && spec.shards.len() <= 4096,
            "select 1..4096 exported shards"
        );
        cache::verify_file(
            &spec.source,
            spec.model.source_bytes,
            &spec.model.source_sha256,
            cancel,
        )
        .await
        .context("pinned source verification")?;
        let mut shards = Vec::new();
        let mut files = BTreeMap::new();
        for path in spec.shards {
            let path = fs::canonicalize(&path).await?;
            let mut report_path = path.as_os_str().to_os_string();
            report_path.push(".json");
            let report_path = PathBuf::from(report_path);
            let bytes = read_export_report(&report_path).await?;
            let report: ExportReport = serde_json::from_slice(&bytes)?;
            ensure!(
                report.source_sha256 == spec.model.source_sha256
                    && report.source_bytes == spec.model.source_bytes
                    && report.layers == spec.model.layers
                    && report.architecture == spec.model.architecture,
                "export {} does not match pinned decoder model",
                path.display()
            );
            use sha2::{Digest, Sha256};
            let manifest = Manifest {
                version: 1,
                model: spec.model.clone(),
                begin: report.begin,
                end: report.end,
                role: Role::for_range(report.begin, report.end, report.layers),
                bytes: report.bytes,
                artifact_sha256: report.artifact_sha256.clone(),
                export_report_sha256: hex::encode(Sha256::digest(&bytes)),
                location: Location::Iroh { peer: peer.clone() },
            };
            manifest.validate()?;
            cache::verify_file(&path, manifest.bytes, &manifest.artifact_sha256, cancel)
                .await
                .with_context(|| format!("verifying {}", path.display()))?;
            let inspect_path = path.clone();
            let inspect_manifest = manifest.clone();
            tokio::task::spawn_blocking(move || {
                gguf::check_export(&inspect_path, &inspect_manifest, &report)
            })
            .await??;
            if cancel.is_cancelled() {
                bail!("artifact operation cancelled");
            }
            files.insert(manifest.artifact_sha256.clone(), path);
            shards.push(manifest);
        }
        let catalog = Catalog {
            version: 1,
            model: spec.model,
            shards,
        };
        catalog.validate()?;
        Ok(Self { catalog, files })
    }
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }
    pub(crate) fn artifact(&self, hash: &str) -> Result<(&Path, u64)> {
        let path = self
            .files
            .get(hash)
            .context("artifact is not advertised by this seed")?;
        let manifest = self
            .catalog
            .shards
            .iter()
            .find(|s| s.artifact_sha256 == hash)
            .unwrap();
        Ok((path, manifest.bytes))
    }
}
