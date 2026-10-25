use super::*;
use anyhow::{Context, bail};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tokio::{
    fs,
    io::{AsyncReadExt, AsyncWriteExt},
    sync::watch,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Checking,
    Downloading,
    Verifying,
    Staged,
    Cancelled,
    Failed,
}
#[derive(Clone, Debug, Serialize)]
pub struct Progress {
    pub artifact_sha256: String,
    pub phase: Phase,
    pub completed_bytes: u64,
    pub total_bytes: u64,
}

/// A private, content-addressed cache for one pinned model. One transfer at a time
/// across processes, so quota reservations and partial files cannot race.
#[derive(Clone)]
pub struct ArtifactCache {
    root: PathBuf,
    model: ModelPin,
    quota: u64,
    node: Option<crate::mesh::Node>,
    client: reqwest::Client,
    progress: Arc<watch::Sender<Option<Progress>>>,
    /// Keeps the single progress channel scoped to one operation at a time.
    operation: Arc<tokio::sync::Mutex<()>>,
}
impl ArtifactCache {
    pub async fn open(
        root: impl AsRef<Path>,
        model: ModelPin,
        quota_bytes: u64,
        node: Option<crate::mesh::Node>,
    ) -> Result<Self> {
        model.validate()?;
        ensure!(quota_bytes > 0, "cache quota must be positive");
        fs::create_dir_all(root.as_ref()).await?;
        let root = fs::canonicalize(root).await?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).await?;
        }
        Ok(Self {
            root,
            model,
            quota: quota_bytes,
            node,
            client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(30))
                .read_timeout(Duration::from_secs(60))
                .build()?,
            progress: Arc::new(watch::channel(None).0),
            operation: Arc::new(tokio::sync::Mutex::new(())),
        })
    }
    /// Observe the serialized operation stream shared by this cache and its clones.
    pub fn progress(&self) -> watch::Receiver<Option<Progress>> {
        self.progress.subscribe()
    }
    fn emit(&self, manifest: &Manifest, phase: Phase, bytes: u64) {
        self.progress.send_replace(Some(Progress {
            artifact_sha256: manifest.artifact_sha256.clone(),
            phase,
            completed_bytes: bytes,
            total_bytes: manifest.bytes,
        }));
    }
    fn path(&self, hash: &str, extension: &str) -> PathBuf {
        self.root.join(format!("{hash}.{extension}"))
    }
    async fn lock(&self, cancel: &CancellationToken) -> Result<std::fs::File> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.root.join(".lock"))?;
        loop {
            check_cancel(cancel)?;
            match fs2::FileExt::try_lock_exclusive(&file) {
                Ok(()) => return Ok(file),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    tokio::select! { _ = cancel.cancelled() => bail!("artifact operation cancelled"), _ = tokio::time::sleep(Duration::from_millis(50)) => {} }
                }
                Err(e) => return Err(e.into()),
            }
        }
    }
    /// Retain before ensure_artifact and hold until profiling/inference releases the
    /// file. Retained artifacts survive LRU eviction and process restart. The owner
    /// explicitly releases stale retentions; the downloader never guesses liveness.
    pub async fn retain(
        &self,
        hash: &str,
        retained: bool,
        cancel: &CancellationToken,
    ) -> Result<()> {
        validate_hash(hash)?;
        let _lock = self.lock(cancel).await?;
        let path = self.path(hash, "keep");
        if retained {
            fs::write(path, []).await?;
        } else {
            remove_if_exists(&path).await?;
        }
        Ok(())
    }
    /// Download only this exported range. Successful return means STAGED.
    /// Cancellation/disconnect preserves a resumable .part. Every cache hit and
    /// completed transfer is hashed before its path is returned.
    pub async fn ensure_artifact(
        &self,
        manifest: &Manifest,
        cancel: &CancellationToken,
    ) -> Result<PathBuf> {
        self.ensure(manifest, None, cancel).await
    }
    /// Explicit local operator import, verified by the same path as downloads.
    /// A received manifest can never direct the downloader to a local file.
    pub async fn import_artifact(
        &self,
        manifest: &Manifest,
        source: &Path,
        cancel: &CancellationToken,
    ) -> Result<PathBuf> {
        self.ensure(manifest, Some(source), cancel).await
    }
    async fn ensure(
        &self,
        manifest: &Manifest,
        local: Option<&Path>,
        cancel: &CancellationToken,
    ) -> Result<PathBuf> {
        manifest.validate()?;
        ensure!(
            manifest.model == self.model,
            "artifact belongs to a different pinned model"
        );
        let _operation = tokio::select! {
            _ = cancel.cancelled() => bail!("artifact operation cancelled"),
            operation = self.operation.lock() => operation,
        };
        self.emit(manifest, Phase::Checking, 0);
        let result = self.ensure_locked(manifest, local, cancel).await;
        if result.is_err() {
            let done = self
                .progress
                .borrow()
                .as_ref()
                .map_or(0, |p| p.completed_bytes);
            self.emit(
                manifest,
                if cancel.is_cancelled() {
                    Phase::Cancelled
                } else {
                    Phase::Failed
                },
                done,
            );
        }
        result
    }
    async fn ensure_locked(
        &self,
        manifest: &Manifest,
        local: Option<&Path>,
        cancel: &CancellationToken,
    ) -> Result<PathBuf> {
        let _lock = self.lock(cancel).await?;
        let hash = &manifest.artifact_sha256;
        let target = self.path(hash, "gguf");
        let partial = self.path(hash, "part");
        let mut offset = regular_size(&partial).await?.unwrap_or(0);
        if regular_size(&target).await?.is_some() {
            self.emit(manifest, Phase::Verifying, 0);
            match verify_file(&target, manifest.bytes, hash, cancel).await {
                Ok(()) => {
                    remove_if_exists(&partial).await?;
                    self.reserve(hash, manifest.bytes, 0).await?;
                    fs::write(self.path(hash, "used"), []).await?;
                    self.emit(manifest, Phase::Staged, manifest.bytes);
                    return Ok(target);
                }
                Err(e) if cancel.is_cancelled() => return Err(e),
                Err(_) => {
                    fs::remove_file(&target).await?;
                }
            }
        }
        if offset > manifest.bytes {
            fs::remove_file(&partial).await?;
            offset = 0;
        }
        self.reserve(hash, manifest.bytes, manifest.bytes - offset)
            .await?;
        check_cancel(cancel)?;
        if offset < manifest.bytes {
            let mut opened = transport::open(
                manifest,
                offset,
                local,
                self.node.as_ref(),
                &self.client,
                cancel,
            )
            .await?;
            // A server ignoring Range is safe only if we truncate and restart.
            ensure!(
                opened.offset == offset || opened.offset == 0,
                "source returned unexpected offset"
            );
            if opened.offset != offset {
                self.reserve(hash, manifest.bytes, manifest.bytes).await?;
                offset = 0;
            }
            let mut out = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(offset == 0)
                .open(&partial)
                .await?;
            out.seek(std::io::SeekFrom::Start(offset)).await?;
            self.emit(manifest, Phase::Downloading, offset);
            let mut buffer = vec![0u8; 1024 * 1024];
            let transfer: Result<()> = async {
                loop {
                    let count = tokio::select! {
                        _ = cancel.cancelled() => bail!("artifact operation cancelled"),
                        _ = transport::membership_changed(&mut opened.membership) => bail!("mesh membership changed during artifact transfer"),
                        read = tokio::time::timeout(Duration::from_secs(60), opened.reader.read(&mut buffer)) => read.context("artifact transfer stalled")??,
                    };
                    if count == 0 { break; }
                    ensure!(count as u64 <= manifest.bytes - offset, "source exceeded advertised artifact size");
                    out.write_all(&buffer[..count]).await?;
                    offset += count as u64;
                    self.emit(manifest, Phase::Downloading, offset);
                }
                ensure!(offset == manifest.bytes, "artifact transfer ended early at {offset}/{}; partial retained", manifest.bytes);
                Ok(())
            }.await;
            // Flush even after cancellation; a later process resumes actual file length.
            out.sync_all().await?;
            transfer?;
        }
        self.emit(manifest, Phase::Verifying, 0);
        if let Err(error) = verify_file(&partial, manifest.bytes, hash, cancel).await {
            if !cancel.is_cancelled() {
                remove_if_exists(&partial).await?;
            }
            return Err(error);
        }
        check_cancel(cancel)?;
        fs::OpenOptions::new()
            .write(true)
            .open(&partial)
            .await?
            .sync_all()
            .await?;
        fs::rename(&partial, &target).await?;
        #[cfg(unix)]
        std::fs::File::open(&self.root)?.sync_all()?;
        fs::write(self.path(hash, "used"), []).await?;
        self.emit(manifest, Phase::Staged, manifest.bytes);
        Ok(target)
    }
    async fn reserve(&self, current: &str, current_size: u64, additional_disk: u64) -> Result<()> {
        ensure!(current_size <= self.quota, "artifact exceeds cache quota");
        let mut used = current_size;
        let mut entries = Vec::new();
        let mut dir = fs::read_dir(&self.root).await?;
        while let Some(entry) = dir.next_entry().await? {
            let path = entry.path();
            let Some(hash) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if hash == current
                || validate_hash(hash).is_err()
                || !matches!(
                    path.extension().and_then(|s| s.to_str()),
                    Some("gguf" | "part")
                )
            {
                continue;
            }
            let meta = entry.metadata().await?;
            ensure!(
                entry.file_type().await?.is_file(),
                "cache artifacts must be regular files"
            );
            used = used
                .checked_add(meta.len())
                .context("cache size overflow")?;
            if !fs::try_exists(self.path(hash, "keep")).await? {
                let accessed = match fs::metadata(self.path(hash, "used")).await {
                    Ok(used) => used.modified()?,
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => meta.modified()?,
                    Err(e) => return Err(e.into()),
                };
                entries.push((accessed, path, meta.len()));
            }
        }
        entries.sort_by_key(|e| e.0);
        let headroom = 64 * 1024 * 1024;
        for (_, path, size) in entries {
            if used <= self.quota
                && fs2::available_space(&self.root)? >= additional_disk.saturating_add(headroom)
            {
                break;
            }
            fs::remove_file(&path).await?;
            remove_if_exists(&path.with_extension("used")).await?;
            used -= size;
        }
        ensure!(
            used <= self.quota,
            "cache quota exhausted by retained artifacts; release unused ranges"
        );
        ensure!(
            additional_disk == 0
                || fs2::available_space(&self.root)? >= additional_disk.saturating_add(headroom),
            "insufficient free disk space (64 MiB headroom required)"
        );
        Ok(())
    }
}

use tokio::io::AsyncSeekExt;
pub(crate) fn check_cancel(cancel: &CancellationToken) -> Result<()> {
    ensure!(!cancel.is_cancelled(), "artifact operation cancelled");
    Ok(())
}
async fn regular_size(path: &Path) -> Result<Option<u64>> {
    match fs::symlink_metadata(path).await {
        Ok(m) => {
            ensure!(m.is_file(), "cache entry must be a regular file");
            Ok(Some(m.len()))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}
async fn remove_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}
pub(crate) async fn verify_file(
    path: &Path,
    expected_bytes: u64,
    expected_hash: &str,
    cancel: &CancellationToken,
) -> Result<()> {
    let mut file = fs::File::open(path).await?;
    ensure!(
        file.metadata().await?.is_file() && file.metadata().await?.len() == expected_bytes,
        "artifact size mismatch: {}",
        path.display()
    );
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 4 * 1024 * 1024];
    let mut total = 0u64;
    loop {
        check_cancel(cancel)?;
        let n = file.read(&mut buffer).await?;
        if n == 0 {
            break;
        }
        total += n as u64;
        ensure!(total <= expected_bytes, "artifact grew during verification");
        hasher.update(&buffer[..n]);
    }
    check_cancel(cancel)?;
    ensure!(
        total == expected_bytes && hex::encode(hasher.finalize()) == expected_hash,
        "artifact SHA-256 mismatch: {}",
        path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn cancellation_while_waiting_does_not_overwrite_progress() {
        let dir = tempfile::tempdir().unwrap();
        let model = ModelPin {
            revision: "a".repeat(40),
            source_sha256: "b".repeat(64),
            source_bytes: 1,
            architecture: "llama".into(),
            layers: 1,
        };
        let cache = ArtifactCache::open(dir.path(), model.clone(), 1024, None)
            .await
            .unwrap();
        let manifest = Manifest {
            version: 1,
            model,
            begin: 0,
            end: 1,
            role: Role::Full,
            bytes: 1,
            artifact_sha256: "c".repeat(64),
            export_report_sha256: "d".repeat(64),
            location: Location::Https {
                url: "https://seed.invalid/shard.gguf".into(),
            },
        };
        let _guard = cache.operation.lock().await;
        let cancel = CancellationToken::new();
        let operation_cache = cache.clone();
        let operation_cancel = cancel.clone();
        let task = tokio::spawn(async move {
            operation_cache
                .ensure_artifact(&manifest, &operation_cancel)
                .await
        });

        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap();
        assert!(result.unwrap_err().to_string().contains("cancelled"));
        assert!(cache.progress().borrow().is_none());
    }
}
