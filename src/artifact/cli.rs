use super::*;
use anyhow::{Context, bail};
use clap::{Args, Subcommand};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::io::AsyncReadExt;

const MAX_ARTIFACT_CONFIG_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Args)]
pub struct Options {
    #[command(subcommand)]
    command: Command,
}
#[derive(Debug, Subcommand)]
enum Command {
    /// Verify existing full source and exports; write public catalog without serving.
    Catalog {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        seed_id: String,
        #[arg(long)]
        output: PathBuf,
    },
    /// Serve verified exports through an already enrolled node's Iroh identity.
    Seed {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        catalog: PathBuf,
    },
    /// Stage a manifest, the smallest catalog candidate, or an exact exported range.
    Fetch {
        #[arg(long, required_unless_present = "catalog", conflicts_with = "catalog")]
        manifest: Option<PathBuf>,
        #[arg(long)]
        catalog: Option<PathBuf>,
        #[arg(long, requires = "end")]
        begin: Option<u32>,
        #[arg(long, requires = "begin")]
        end: Option<u32>,
        #[arg(long)]
        cache: PathBuf,
        #[arg(long)]
        quota_bytes: u64,
        #[arg(long)]
        state_dir: Option<PathBuf>,
        /// Reuse an existing local export instead of fetching it from the seed.
        #[arg(long)]
        import: Option<PathBuf>,
        /// Protect this artifact from eviction until the core explicitly releases it.
        #[arg(long)]
        retain: bool,
    },
}
async fn read_limited(reader: impl tokio::io::AsyncRead + Unpin, limit: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut bytes)
        .await?;
    anyhow::ensure!(
        bytes.len() <= limit,
        "artifact configuration exceeds the {limit}-byte limit"
    );
    Ok(bytes)
}

async fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = tokio::fs::File::open(path).await?;
    anyhow::ensure!(
        file.metadata().await?.len() <= MAX_ARTIFACT_CONFIG_BYTES as u64,
        "artifact configuration exceeds the {MAX_ARTIFACT_CONFIG_BYTES}-byte limit"
    );
    let bytes = read_limited(file, MAX_ARTIFACT_CONFIG_BYTES).await?;
    Ok(serde_json::from_slice(&bytes)?)
}
async fn write_catalog(path: &PathBuf, catalog: &Catalog) -> Result<()> {
    // create_new avoids replacing a catalog still in use by another process.
    use tokio::io::AsyncWriteExt;
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .await?;
    file.write_all(&serde_json::to_vec_pretty(catalog)?).await?;
    file.sync_all().await?;
    Ok(())
}
pub async fn run(options: Options) -> Result<()> {
    let cancel = CancellationToken::new();
    let signal_token = cancel.clone();
    let signal = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            signal_token.cancel();
        }
    });
    let result = run_command(options.command, &cancel).await;
    signal.abort();
    result
}
async fn run_command(command: Command, cancel: &CancellationToken) -> Result<()> {
    match command {
        Command::Catalog {
            spec,
            seed_id,
            output,
        } => {
            let peer = crate::mesh::Peer {
                id: seed_id,
                addresses: vec![],
                invite: String::new(),
                relay: None,
            };
            let seed = Seed::verify(read_json(&spec).await?, peer, cancel).await?;
            write_catalog(&output, seed.catalog()).await?;
            println!(
                "{}",
                serde_json::json!({"catalog": output, "shards": seed.catalog().shards.len(), "status": "verified_seed_catalog", "inference_ready": false})
            );
        }
        Command::Seed {
            spec,
            state_dir,
            catalog,
        } => {
            let node = crate::mesh::Node::open(state_dir).await?;
            let result: Result<()> = async {
                ensure!(!node.snapshot().await?["mesh"].is_null(), "create or join a mesh before serving artifacts");
                let seed = Arc::new(Seed::verify(read_json(&spec).await?, node.address(), cancel).await?);
                write_catalog(&catalog, seed.catalog()).await?;
                node.set_artifact_seed(Some(seed)).await;
                println!("{}", serde_json::json!({"catalog": catalog, "seed": node.id(), "status": "serving", "inference_ready": false}));
                cancel.cancelled().await;
                Ok(())
            }.await;
            node.close().await;
            result?;
        }
        Command::Fetch {
            manifest,
            catalog,
            begin,
            end,
            cache,
            quota_bytes,
            state_dir,
            import,
            retain,
        } => {
            let manifest: Manifest = if let Some(path) = manifest {
                ensure!(
                    begin.is_none() && end.is_none(),
                    "range selection requires --catalog"
                );
                read_json(&path).await?
            } else {
                let catalog: Catalog =
                    read_json(&catalog.context("manifest or catalog required")?).await?;
                match (begin, end) {
                    (None, None) => catalog.smallest_candidate()?.clone(),
                    (Some(begin), Some(end)) => catalog.exact_range(begin, end)?.clone(),
                    _ => bail!("provide both --begin and --end"),
                }
            };
            manifest.validate()?;
            let node = match state_dir {
                Some(state_dir)
                    if import.is_none() && matches!(manifest.location, Location::Iroh { .. }) =>
                {
                    Some(crate::mesh::Node::open(state_dir).await?)
                }
                _ => None,
            };
            let result: Result<()> = async {
                let cache = ArtifactCache::open(cache, manifest.model.clone(), quota_bytes, node.clone()).await?;
                if retain { cache.retain(&manifest.artifact_sha256, true, cancel).await?; }
                let mut progress = cache.progress();
                let reporter = tokio::spawn(async move {
                    while progress.changed().await.is_ok() {
                        if let Some(progress) = progress.borrow_and_update().as_ref() { eprintln!("{}", serde_json::to_string(progress).unwrap()); }
                    }
                });
                let result = if let Some(path) = import { cache.import_artifact(&manifest, &path, cancel).await }
                    else { cache.ensure_artifact(&manifest, cancel).await };
                reporter.abort();
                let path = result?;
                println!("{}", serde_json::json!({"status": "staged", "path": path, "artifact_sha256": manifest.artifact_sha256, "begin": manifest.begin, "end": manifest.end, "inference_ready": false}));
                Ok(())
            }.await;
            if let Some(node) = node {
                node.close().await;
            }
            result?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn artifact_config_reader_enforces_limit_while_reading() {
        let error = read_limited(&b"12345"[..], 4).await.unwrap_err();
        assert!(error.to_string().contains("4-byte limit"));
    }
}
