use super::*;
use sha2::{Digest, Sha256};
use std::{path::Path, sync::Arc};
use tokio::fs;
fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn pin() -> ModelPin {
    ModelPin {
        revision: "a".repeat(40),
        source_sha256: "b".repeat(64),
        source_bytes: 32 * 1024 * 1024,
        architecture: "qwen35".into(),
        layers: 4,
    }
}
fn manifest(bytes: &[u8]) -> Manifest {
    Manifest {
        version: 1,
        model: pin(),
        begin: 1,
        end: 2,
        role: Role::Middle,
        bytes: bytes.len() as u64,
        artifact_sha256: hash(bytes),
        export_report_sha256: "c".repeat(64),
        location: Location::Https {
            url: "https://seed.invalid/shard.gguf".into(),
        },
    }
}
async fn cache(path: &Path, quota: u64) -> ArtifactCache {
    ArtifactCache::open(path, pin(), quota, None).await.unwrap()
}

#[tokio::test]
async fn import_resume_cache_reuse_and_corruption_repair() {
    let dir = tempfile::tempdir().unwrap();
    let bytes = b"an already exported whole shard";
    let m = manifest(bytes);
    let source = dir.path().join("source");
    fs::write(&source, bytes).await.unwrap();
    let root = dir.path().join("cache");
    let c = cache(&root, 1024).await;
    let part = root.join(format!("{}.part", m.artifact_sha256));
    fs::write(&part, &bytes[..7]).await.unwrap();
    // A new cache object (process restart) resumes from the durable partial length.
    drop(c);
    let c = cache(&root, 1024).await;
    let cancel = CancellationToken::new();
    let path = c.import_artifact(&m, &source, &cancel).await.unwrap();
    assert_eq!(fs::read(&path).await.unwrap(), bytes);
    assert!(!part.exists());
    assert_eq!(c.progress().borrow().as_ref().unwrap().phase, Phase::Staged);
    // No source or reachable network is needed for a verified cache hit.
    fs::remove_file(&source).await.unwrap();
    assert_eq!(c.ensure_artifact(&m, &cancel).await.unwrap(), path);
    fs::write(&path, vec![b'x'; bytes.len()]).await.unwrap();
    fs::write(&source, bytes).await.unwrap();
    assert_eq!(c.import_artifact(&m, &source, &cancel).await.unwrap(), path);
    assert_eq!(fs::read(&path).await.unwrap(), bytes);
}

#[tokio::test]
async fn wrong_hash_never_publishes_and_cancel_preserves_full_partial() {
    let dir = tempfile::tempdir().unwrap();
    let bytes = b"correct exported shard";
    let m = manifest(bytes);
    let c = cache(dir.path(), 1024).await;
    let part = dir.path().join(format!("{}.part", m.artifact_sha256));
    let target = part.with_extension("gguf");
    fs::write(&part, vec![0; bytes.len()]).await.unwrap();
    assert!(
        c.ensure_artifact(&m, &CancellationToken::new())
            .await
            .is_err()
    );
    assert!(!target.exists() && !part.exists());
    fs::write(&part, bytes).await.unwrap();
    let cancel = CancellationToken::new();
    cancel.cancel();
    assert!(c.ensure_artifact(&m, &cancel).await.is_err());
    assert!(part.exists() && !target.exists());
    assert_eq!(
        c.ensure_artifact(&m, &CancellationToken::new())
            .await
            .unwrap(),
        std::fs::canonicalize(target).unwrap()
    );
}

#[tokio::test]
async fn quota_evicts_reusable_lru_but_never_retained_or_unowned_files() {
    let dir = tempfile::tempdir().unwrap();
    let c = cache(dir.path(), 8).await;
    let cancel = CancellationToken::new();
    let a = manifest(b"aaaa");
    let b = manifest(b"bbbb");
    let d = manifest(b"dddd");
    let source = dir.path().join("operator-source");
    fs::write(&source, b"aaaa").await.unwrap();
    let ap = c.import_artifact(&a, &source, &cancel).await.unwrap();
    c.retain(&a.artifact_sha256, true, &cancel).await.unwrap();
    fs::write(&source, b"bbbb").await.unwrap();
    let bp = c.import_artifact(&b, &source, &cancel).await.unwrap();
    c.retain(&b.artifact_sha256, true, &cancel).await.unwrap();
    fs::write(&source, b"dddd").await.unwrap();
    assert!(
        c.import_artifact(&d, &source, &cancel)
            .await
            .unwrap_err()
            .to_string()
            .contains("retained")
    );
    assert!(ap.exists() && bp.exists());
    c.retain(&b.artifact_sha256, false, &cancel).await.unwrap();
    let dp = c.import_artifact(&d, &source, &cancel).await.unwrap();
    assert!(ap.exists() && dp.exists() && !bp.exists() && source.exists());
    let oversized = manifest(b"too large for this quota");
    assert!(
        c.import_artifact(&oversized, &source, &cancel)
            .await
            .unwrap_err()
            .to_string()
            .contains("exceeds cache quota")
    );
}

#[test]
fn manifests_pin_model_role_range_and_whole_shard_location() {
    let m = manifest(b"artifact");
    m.validate().unwrap();
    for mutate in [
        |m: &mut Manifest| m.model.revision = "main".into(),
        |m: &mut Manifest| m.artifact_sha256 = "../evil".into(),
        |m: &mut Manifest| m.role = Role::Final,
        |m: &mut Manifest| m.end = 5,
        |m: &mut Manifest| {
            m.location = Location::Https {
                url: "file:///etc/passwd".into(),
            }
        },
    ] {
        let mut bad = m.clone();
        mutate(&mut bad);
        assert!(bad.validate().is_err());
    }
    let mut other = m.clone();
    other.model.source_sha256 = "f".repeat(64);
    assert!(
        Catalog {
            version: 1,
            model: m.model.clone(),
            shards: vec![m.clone(), other]
        }
        .validate()
        .is_err()
    );
    let mut larger = manifest(b"a larger exported shard");
    larger.begin = 2;
    larger.end = 3;
    let catalog = Catalog {
        version: 1,
        model: m.model.clone(),
        shards: vec![larger, m],
    };
    assert_eq!(catalog.smallest_candidate().unwrap().begin, 1);
    assert_eq!(catalog.exact_range(2, 3).unwrap().begin, 2);
    assert!(catalog.exact_range(0, 1).is_err());
}

// A minimal physical GGUF fixture. It tests export metadata and transfer integrity,
// not native inference or the numerical validity of a model.
async fn fixture(dir: &Path, peer: crate::mesh::Peer) -> Seed {
    let source = vec![42; 8 * 1024 * 1024];
    let source_path = dir.join("source.gguf");
    fs::write(&source_path, &source).await.unwrap();
    let mut model = pin();
    model.source_sha256 = hash(&source);
    model.source_bytes = source.len() as u64;
    fn string(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }
    let mut data = Vec::new();
    data.extend(b"GGUF");
    data.extend(3u32.to_le_bytes());
    data.extend(1u64.to_le_bytes());
    data.extend(5u64.to_le_bytes());
    for (key, value) in [
        ("general.architecture", "qwen35"),
        ("meshthing.source_sha256", model.source_sha256.as_str()),
    ] {
        string(&mut data, key);
        data.extend(8u32.to_le_bytes());
        string(&mut data, value);
    }
    for (key, value) in [
        ("qwen35.block_count", 4u32),
        ("meshthing.shard_begin", 1),
        ("meshthing.shard_end", 2),
    ] {
        string(&mut data, key);
        data.extend(4u32.to_le_bytes());
        data.extend(value.to_le_bytes());
    }
    string(&mut data, "blk.1.attn_norm.weight");
    data.extend(1u32.to_le_bytes());
    data.extend((1024u64 * 1024).to_le_bytes());
    data.extend(0u32.to_le_bytes());
    data.extend(0u64.to_le_bytes());
    data.resize(data.len().div_ceil(32) * 32, 0);
    data.extend(vec![21; 4 * 1024 * 1024]);
    let shard = dir.join("middle.gguf");
    fs::write(&shard, &data).await.unwrap();
    let report = serde_json::json!({"source_sha256": model.source_sha256, "artifact_sha256":hash(&data),"begin":1,"end":2,"layers":4,"architecture":"qwen35","source_bytes":model.source_bytes,"bytes":data.len(),"tensors":[{"name":"blk.1.attn_norm.weight","bytes":4*1024*1024}]});
    fs::write(
        shard.with_extension("gguf.json"),
        serde_json::to_vec(&report).unwrap(),
    )
    .await
    .unwrap();
    Seed::verify(
        SeedSpec {
            model,
            source: source_path,
            shards: vec![shard],
        },
        peer,
        &CancellationToken::new(),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn actual_iroh_seed_requires_membership_and_resumes_whole_export() {
    let seed_dir = tempfile::tempdir().unwrap();
    let device_dir = tempfile::tempdir().unwrap();
    let stranger_dir = tempfile::tempdir().unwrap();
    let seed_node = crate::mesh::Node::open_local(seed_dir.path())
        .await
        .unwrap();
    let device = crate::mesh::Node::open_local(device_dir.path())
        .await
        .unwrap();
    let stranger = crate::mesh::Node::open_local(stranger_dir.path())
        .await
        .unwrap();
    let seed = Arc::new(fixture(seed_dir.path(), seed_node.address()).await);
    let manifest = seed.catalog().shards[0].clone();
    seed_node.set_artifact_seed(Some(seed.clone())).await;
    let cache_root = device_dir.path().join("cache");
    let c = ArtifactCache::open(
        &cache_root,
        manifest.model.clone(),
        16 * 1024 * 1024,
        Some(device.clone()),
    )
    .await
    .unwrap();
    assert!(
        c.ensure_artifact(&manifest, &CancellationToken::new())
            .await
            .is_err()
    );
    seed_node.create("Artifact test", "Seed").await.unwrap();
    // Exercise the seed's authorization gate independently of the cache's
    // client-side membership check.
    let outsider = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![transport::ALPN.to_vec()])
        .bind()
        .await
        .unwrap();
    let address = seed_node.address();
    let mut endpoint_addr = iroh::EndpointAddr::new(address.id.parse().unwrap());
    for ip in address.addresses {
        endpoint_addr = endpoint_addr.with_ip_addr(ip);
    }
    let connection = outsider
        .connect(endpoint_addr, transport::ALPN)
        .await
        .unwrap();
    let reason = tokio::time::timeout(std::time::Duration::from_secs(3), connection.closed())
        .await
        .unwrap();
    assert!(format!("{reason:?}").contains("membership required"));
    outsider.close().await;
    seed_node.invite().await.unwrap();
    let mut peer = seed_node.address();
    peer.invite = seed_node.snapshot().await.unwrap()["invite"]["id"]
        .as_str()
        .unwrap()
        .into();
    device.join(peer, "Compute").await.unwrap();
    let request = seed_node.snapshot().await.unwrap()["pending"][0]["id"]
        .as_str()
        .unwrap()
        .to_string();
    seed_node
        .decide(&seed_node.id(), &request, true)
        .await
        .unwrap();
    device.refresh().await.unwrap();
    let source = fs::read(seed_dir.path().join("middle.gguf")).await.unwrap();
    let part = cache_root.join(format!("{}.part", manifest.artifact_sha256));
    fs::write(&part, &source[..33333]).await.unwrap();
    let path = tokio::time::timeout(
        std::time::Duration::from_secs(15),
        c.ensure_artifact(&manifest, &CancellationToken::new()),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(fs::read(&path).await.unwrap(), source);
    assert!(!part.exists());
    // Leaving prevents new fetches even with a previously valid seed locator.
    device.leave().await.unwrap();
    fs::remove_file(&path).await.unwrap();
    assert!(
        c.ensure_artifact(&manifest, &CancellationToken::new())
            .await
            .is_err()
    );
    // A different mesh cannot fetch through a valid cryptographic endpoint identity.
    stranger.create("Other mesh", "Stranger").await.unwrap();
    let stranger_cache = ArtifactCache::open(
        stranger_dir.path().join("cache"),
        manifest.model.clone(),
        16 * 1024 * 1024,
        Some(stranger.clone()),
    )
    .await
    .unwrap();
    assert!(
        stranger_cache
            .ensure_artifact(&manifest, &CancellationToken::new())
            .await
            .is_err()
    );
    seed_node.close().await;
    device.close().await;
    stranger.close().await;
}

#[tokio::test]
async fn seed_rejects_mismatched_sidecar_and_missing_terminal_role() {
    let dir = tempfile::tempdir().unwrap();
    let node = crate::mesh::Node::open_local(dir.path()).await.unwrap();
    let seed = fixture(dir.path(), node.address()).await;
    let manifest = &seed.catalog().shards[0];
    let shard = dir.path().join("middle.gguf");
    let report_path = shard.with_extension("gguf.json");
    let mut report: serde_json::Value =
        serde_json::from_slice(&fs::read(&report_path).await.unwrap()).unwrap();
    report["tensors"][0]["name"] = "blk.1.missing.weight".into();
    let parsed: seed::ExportReport = serde_json::from_value(report.clone()).unwrap();
    assert!(
        gguf::check_export(&shard, manifest, &parsed)
            .unwrap_err()
            .to_string()
            .contains("physical GGUF tensors")
    );
    // Reclassify this physical middle export as terminal without adding output
    // tensors: the seed must reject it even though its block range is complete.
    let mut physical = fs::read(&shard).await.unwrap();
    let key = b"qwen35.block_count";
    let position = physical.windows(key.len()).position(|w| w == key).unwrap() + key.len() + 4;
    physical[position..position + 4].copy_from_slice(&2u32.to_le_bytes());
    let terminal = dir.path().join("bad-final.gguf");
    fs::write(&terminal, physical).await.unwrap();
    let mut final_manifest = manifest.clone();
    final_manifest.model.layers = 2;
    final_manifest.role = Role::Final;
    assert!(
        gguf::check_export(&terminal, &final_manifest, &parsed)
            .unwrap_err()
            .to_string()
            .contains("missing output tensors")
    );
    report["layers"] = 65.into();
    fs::write(&report_path, serde_json::to_vec(&report).unwrap())
        .await
        .unwrap();
    let spec = SeedSpec {
        model: manifest.model.clone(),
        source: dir.path().join("source.gguf"),
        shards: vec![shard.clone()],
    };
    assert!(
        Seed::verify(spec, node.address(), &CancellationToken::new())
            .await
            .is_err()
    );
    fs::write(&report_path, vec![0; 8 * 1024 * 1024 + 1])
        .await
        .unwrap();
    let error = Seed::verify(
        SeedSpec {
            model: manifest.model.clone(),
            source: dir.path().join("source.gguf"),
            shards: vec![shard],
        },
        node.address(),
        &CancellationToken::new(),
    )
    .await
    .err()
    .expect("oversized export report must be rejected")
    .to_string();
    assert!(error.contains("export report too large"), "{error}");
    node.close().await;
}

#[tokio::test]
async fn cancellation_during_transfer_resumes_after_cache_restart() {
    let dir = tempfile::tempdir().unwrap();
    let bytes = vec![73; 24 * 1024 * 1024];
    let m = manifest(&bytes);
    let source = dir.path().join("source");
    fs::write(&source, &bytes).await.unwrap();
    let root = dir.path().join("cache");
    let c = cache(&root, 32 * 1024 * 1024).await;
    let cancel = CancellationToken::new();
    let observer_cancel = cancel.clone();
    let mut progress = c.progress();
    let observer = tokio::spawn(async move {
        while progress.changed().await.is_ok() {
            let p = progress.borrow_and_update().clone().unwrap();
            if p.phase == Phase::Downloading
                && p.completed_bytes > 0
                && p.completed_bytes < p.total_bytes
            {
                observer_cancel.cancel();
                return;
            }
        }
        panic!("no intermediate download progress");
    });
    assert!(c.import_artifact(&m, &source, &cancel).await.is_err());
    observer.await.unwrap();
    let part = root.join(format!("{}.part", m.artifact_sha256));
    let size = fs::metadata(&part).await.unwrap().len();
    assert!(size > 0 && size < m.bytes);
    assert_eq!(
        c.progress().borrow().as_ref().unwrap().phase,
        Phase::Cancelled
    );
    drop(c);
    let restarted = cache(&root, 32 * 1024 * 1024).await;
    let path = restarted
        .import_artifact(&m, &source, &CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(fs::read(path).await.unwrap(), bytes);
}

#[tokio::test]
async fn cross_process_lock_is_cancellable_and_concurrent_cache_users_serialize() {
    let dir = tempfile::tempdir().unwrap();
    let c = cache(dir.path(), 1024).await;
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.path().join(".lock"))
        .unwrap();
    fs2::FileExt::lock_exclusive(&lock).unwrap();
    let m = manifest(b"bytes");
    let cancel = CancellationToken::new();
    let delayed = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        delayed.cancel();
    });
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            c.ensure_artifact(&m, &cancel)
        )
        .await
        .unwrap()
        .is_err()
    );
    drop(lock);
    let source = dir.path().join("source");
    fs::write(&source, b"bytes").await.unwrap();
    let second = cache(dir.path(), 1024).await;
    let cancel = CancellationToken::new();
    let (a, b) = tokio::join!(
        c.import_artifact(&m, &source, &cancel),
        second.import_artifact(&m, &source, &cancel)
    );
    assert_eq!(a.unwrap(), b.unwrap());
}
