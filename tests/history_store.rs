//! Durable history: reload after restart, crash recovery, export, and save-before-ack.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use llm_cluster::history_sync::history::{Entry, History};
use llm_cluster::history_sync::leader::Leader;
use llm_cluster::history_sync::message::SyncMessage;
use llm_cluster::history_sync::node::Node;
use llm_cluster::history_sync::sim::node_id as id;
use llm_cluster::history_sync::store::{
    HistoryStore, JobStatus, ModelManifest, SNAPSHOT_EVERY, StoreError, StoreMeta,
};

const CONVERSATION: &str = "conv-1";

/// A fresh, empty data directory that is removed when dropped.
struct DataDir(PathBuf);

impl DataDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "meshthing-history-store-{}-{name}",
            std::process::id()
        ));
        // A leftover directory from an aborted run may or may not exist; start fresh.
        let _ = fs::remove_dir_all(&path);
        Self(path)
    }

    fn log_path(&self) -> PathBuf {
        self.0.join("history").join(CONVERSATION).join("log.jsonl")
    }
}

impl Drop for DataDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn model(hash: &str) -> ModelManifest {
    ModelManifest {
        model_hash: hash.to_owned(),
        generation: BTreeMap::from([("ctx_size".to_owned(), "8192".to_owned())]),
    }
}

fn meta() -> StoreMeta {
    StoreMeta::new(CONVERSATION, "cluster-a", model("sha256:abc"))
}

fn job(n: usize) -> Vec<Entry> {
    let request_id = format!("job-{n}");
    vec![
        Entry::JobStarted {
            request_id: request_id.clone(),
            prompt: format!("prompt {n}"),
        },
        Entry::OutputCommitted {
            request_id: request_id.clone(),
            offset: 0,
            text: format!("response {n}"),
        },
        Entry::JobCompleted { request_id },
    ]
}

fn append_raw(path: &PathBuf, bytes: &[u8]) {
    let mut file = OpenOptions::new().append(true).open(path).unwrap();
    file.write_all(bytes).unwrap();
}

#[test]
fn history_and_epoch_survive_a_restart() {
    let dir = DataDir::new("restart");
    let mut store = HistoryStore::open(&dir.0, meta()).unwrap();
    store.append(&job(0)).unwrap();
    store.append(&job(1)).unwrap();
    store.set_epoch(4).unwrap();
    let before = store.history().clone();
    drop(store);

    let store = HistoryStore::load(&dir.0, CONVERSATION).unwrap();
    assert_eq!(store.history(), &before);
    assert_eq!(store.epoch(), 4);
    assert_eq!(store.meta().model, model("sha256:abc"));

    // Opening again with the same identity keeps what is stored.
    let reopened = HistoryStore::open(&dir.0, meta()).unwrap();
    assert_eq!(reopened.len(), 6);
}

#[test]
fn torn_final_log_line_is_dropped() {
    let dir = DataDir::new("torn");
    let mut store = HistoryStore::open(&dir.0, meta()).unwrap();
    store.append(&job(0)).unwrap();
    drop(store);
    // A crash in the middle of writing the next line.
    append_raw(
        &dir.log_path(),
        br#"{"generation":0,"index":3,"entry":{"ty"#,
    );

    let mut store = HistoryStore::load(&dir.0, CONVERSATION).unwrap();
    assert_eq!(store.len(), 3);
    store.append(&job(1)).unwrap();
    drop(store);
    assert_eq!(HistoryStore::load(&dir.0, CONVERSATION).unwrap().len(), 6);
}

#[test]
fn corrupt_complete_log_line_is_an_error() {
    let dir = DataDir::new("corrupt");
    let mut store = HistoryStore::open(&dir.0, meta()).unwrap();
    store.append(&job(0)).unwrap();
    drop(store);
    append_raw(&dir.log_path(), b"not json\n");

    assert!(matches!(
        HistoryStore::load(&dir.0, CONVERSATION),
        Err(StoreError::Json { .. })
    ));
}

#[test]
fn log_is_folded_into_the_snapshot() {
    let dir = DataDir::new("compact");
    let mut store = HistoryStore::open(&dir.0, meta()).unwrap();
    let early_log = {
        store.append(&job(0)).unwrap();
        fs::read(dir.log_path()).unwrap()
    };
    let jobs = SNAPSHOT_EVERY as usize; // 3 events each: well past the threshold
    for n in 1..jobs {
        store.append(&job(n)).unwrap();
    }
    let expected = store.history().clone();
    let log_lines = fs::read_to_string(dir.log_path()).unwrap().lines().count();
    assert!(log_lines < SNAPSHOT_EVERY as usize, "log was not compacted");
    drop(store);

    // A crash after the snapshot was written but before the log was cut leaves lines the
    // snapshot already covers; they are skipped, not applied twice.
    append_raw(&dir.log_path(), &early_log);
    let store = HistoryStore::load(&dir.0, CONVERSATION).unwrap();
    assert_eq!(store.history(), &expected);
}

#[test]
fn durable_append_succeeds_when_snapshot_compaction_fails() {
    let dir = DataDir::new("compact-failure");
    let mut store = HistoryStore::open(&dir.0, meta()).unwrap();
    let snapshot_temp = dir
        .0
        .join("history")
        .join(CONVERSATION)
        .join("snapshot.tmp");
    fs::create_dir(&snapshot_temp).unwrap();
    let entries: Vec<_> = (0..SNAPSHOT_EVERY as usize).flat_map(job).collect();

    // The log has been synced even though compaction cannot create its temporary snapshot file.
    store.append(&entries).unwrap();
    assert_eq!(store.history().entries(), entries);
    drop(store);

    let recovered = HistoryStore::load(&dir.0, CONVERSATION).unwrap();
    assert_eq!(recovered.history().entries(), entries);
}

#[test]
fn replace_discards_the_old_history() {
    let dir = DataDir::new("replace");
    let mut store = HistoryStore::open(&dir.0, meta()).unwrap();
    store.append(&job(0)).unwrap();
    store.append(&job(1)).unwrap();
    let old_log = fs::read(dir.log_path()).unwrap();

    let snapshot = History::from_entries(job(7));
    store.replace(&snapshot, 3).unwrap();
    store.append(&job(8)).unwrap();
    let expected = store.history().clone();
    drop(store);

    // Old-generation lines left behind by a crash mid-replace are ignored.
    let log = dir.log_path();
    let mut combined = old_log;
    combined.extend(fs::read(&log).unwrap());
    fs::write(&log, combined).unwrap();

    let store = HistoryStore::load(&dir.0, CONVERSATION).unwrap();
    assert_eq!(store.history(), &expected);
    assert_eq!(store.epoch(), 3);
}

#[test]
fn replace_returns_error_when_snapshot_generation_is_exhausted() {
    let dir = DataDir::new("generation-exhausted");
    drop(HistoryStore::open(&dir.0, meta()).unwrap());

    let snapshot_path = dir
        .0
        .join("history")
        .join(CONVERSATION)
        .join("snapshot.json");
    let mut snapshot: serde_json::Value =
        serde_json::from_slice(&fs::read(&snapshot_path).unwrap()).unwrap();
    snapshot["generation"] = serde_json::json!(u64::MAX);
    fs::write(&snapshot_path, serde_json::to_vec(&snapshot).unwrap()).unwrap();

    let mut store = HistoryStore::load(&dir.0, CONVERSATION).unwrap();
    let replacement = History::from_entries(job(1));
    assert!(matches!(
        store.replace(&replacement, 1),
        Err(StoreError::GenerationExhausted { .. })
    ));
    assert!(
        store.is_empty(),
        "failed replacement must leave memory unchanged"
    );
}

#[test]
fn opening_with_another_model_or_cluster_is_refused() {
    let dir = DataDir::new("mismatch");
    HistoryStore::open(&dir.0, meta()).unwrap();

    let other_model = StoreMeta::new(CONVERSATION, "cluster-a", model("sha256:other"));
    let other_cluster = StoreMeta::new(CONVERSATION, "cluster-b", model("sha256:abc"));
    assert!(matches!(
        HistoryStore::open(&dir.0, other_model),
        Err(StoreError::Mismatch {
            field: "model hash",
            ..
        })
    ));
    assert!(matches!(
        HistoryStore::open(&dir.0, other_cluster),
        Err(StoreError::Mismatch {
            field: "cluster ID",
            ..
        })
    ));

    let mut other_settings = meta();
    other_settings
        .model
        .generation
        .insert("ctx_size".to_owned(), "4096".to_owned());
    assert!(matches!(
        HistoryStore::open(&dir.0, other_settings),
        Err(StoreError::GenerationSettingsMismatch { .. })
    ));
}

#[test]
fn unsafe_or_missing_conversations_are_refused() {
    let dir = DataDir::new("ids");
    for bad in ["", ".", "..", "../escape", "a/b", "a\\b", "spaces here"] {
        let meta = StoreMeta::new(bad, "cluster-a", model("sha256:abc"));
        assert!(
            matches!(
                HistoryStore::open(&dir.0, meta),
                Err(StoreError::InvalidConversationId(_))
            ),
            "{bad:?} was accepted"
        );
    }
    assert!(matches!(
        HistoryStore::load(&dir.0, "never-created"),
        Err(StoreError::NotFound(_))
    ));
}

#[test]
fn export_separates_completed_and_interrupted_jobs() {
    let dir = DataDir::new("export");
    let mut store = HistoryStore::open(&dir.0, meta()).unwrap();
    store.append(&job(0)).unwrap();
    let interrupted = "job-1".to_owned();
    store
        .append(&[
            Entry::JobStarted {
                request_id: interrupted.clone(),
                prompt: "long prompt".to_owned(),
            },
            Entry::OutputCommitted {
                request_id: interrupted.clone(),
                offset: 0,
                text: "Hello, ".to_owned(),
            },
            // A replayed chunk and a chunk after a gap are not joined.
            Entry::OutputCommitted {
                request_id: interrupted.clone(),
                offset: 0,
                text: "Hello, ".to_owned(),
            },
            Entry::OutputCommitted {
                request_id: interrupted.clone(),
                offset: 99,
                text: "lost".to_owned(),
            },
            Entry::OutputCommitted {
                request_id: interrupted.clone(),
                offset: 7,
                text: "mesh".to_owned(),
            },
        ])
        .unwrap();
    store.set_epoch(2).unwrap();

    let export = store.export();
    assert_eq!(export.conversation_id, CONVERSATION);
    assert_eq!(export.cluster_id, "cluster-a");
    assert_eq!(export.model, model("sha256:abc"));
    assert_eq!(export.epoch, 2);
    assert_eq!(export.committed_len, 8);
    assert_eq!(export.entries.len(), 8);
    assert_eq!(export.jobs.len(), 2);
    assert_eq!(export.jobs[0].status, JobStatus::Completed);
    assert_eq!(export.jobs[0].committed_output, "response 0");
    assert_eq!(export.jobs[1].status, JobStatus::Interrupted);
    assert_eq!(export.jobs[1].prompt, "long prompt");
    assert_eq!(export.jobs[1].committed_output, "Hello, mesh");
    assert_eq!(export.jobs[1].committed_output_offset, 11);

    let json = serde_json::to_string(&export).unwrap();
    assert!(json.contains(r#""status":"interrupted""#), "{json}");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&json).unwrap()["format_version"],
        1
    );
}

#[test]
fn replica_acks_only_what_is_on_disk() {
    let dir = DataDir::new("durable-ack");
    let mut store = HistoryStore::open(&dir.0, meta()).unwrap();
    let (leader_id, replica_id) = (id(0), id(1));
    let mut leader = Leader::new(leader_id, 1, Default::default(), [replica_id]);
    let mut replica = Node::from_store(replica_id, &store);

    leader.start_job("job-0", "prompt");
    for (_, msg) in leader.complete_job("job-0", "response") {
        for (_, reply) in replica.handle_durable(leader_id, msg, &mut store).unwrap() {
            assert!(matches!(reply, SyncMessage::Ack { len: 4, .. }));
            // At the moment the Ack exists, the entries are already on disk.
            let on_disk = HistoryStore::load(&dir.0, CONVERSATION).unwrap();
            assert_eq!(on_disk.history(), leader.history());
            assert_eq!(on_disk.epoch(), 1);
        }
    }
}

#[test]
fn restarted_replica_reloads_and_catches_up() {
    let dir = DataDir::new("restart-replica");
    let (leader_id, replica_id) = (id(0), id(1));
    let mut leader = Leader::new(leader_id, 1, Default::default(), [replica_id]);

    let deliver = |leader: &mut Leader, node: &mut Node, store: &mut HistoryStore| {
        let mut queue = leader.sync();
        while let Some((_, msg)) = queue.pop() {
            for (_, reply) in node.handle_durable(leader_id, msg, store).unwrap() {
                queue.extend(leader.handle(replica_id, reply));
            }
        }
    };

    let mut store = HistoryStore::open(&dir.0, meta()).unwrap();
    let mut replica = Node::from_store(replica_id, &store);
    leader.complete_task("job-0", "a", "b");
    deliver(&mut leader, &mut replica, &mut store);
    drop((replica, store));

    // The replica process restarts while the leader keeps going.
    leader.complete_task("job-1", "c", "d");
    let mut store = HistoryStore::load(&dir.0, CONVERSATION).unwrap();
    assert_eq!(store.len(), 4);
    let mut replica = Node::from_store(replica_id, &store);
    deliver(&mut leader, &mut replica, &mut store);

    assert_eq!(replica.history(), leader.history());
    let on_disk = HistoryStore::load(&dir.0, CONVERSATION).unwrap();
    assert_eq!(on_disk.history(), leader.history());
}

#[test]
fn leader_persists_before_sending() {
    let dir = DataDir::new("leader-persist");
    let mut store = HistoryStore::open(&dir.0, meta()).unwrap();
    let mut node = Node::Leader(Leader::new(id(0), 1, Default::default(), [id(1)]));

    let out = node
        .as_leader_mut()
        .unwrap()
        .complete_task("job-0", "a", "b");
    node.persist(&mut store).unwrap();
    assert!(!out.is_empty());

    let on_disk = HistoryStore::load(&dir.0, CONVERSATION).unwrap();
    assert_eq!(on_disk.history(), node.history());
    assert_eq!(on_disk.epoch(), 1);
}

fn meta_path(dir: &DataDir) -> PathBuf {
    dir.0.join("history").join(CONVERSATION).join("meta.json")
}

#[test]
fn epoch_is_saved_with_the_history_not_in_meta() {
    let dir = DataDir::new("epoch-with-history");
    let mut store = HistoryStore::open(&dir.0, meta()).unwrap();
    let meta_before = fs::read(meta_path(&dir)).unwrap();

    store.append(&job(0)).unwrap();
    store.set_epoch(2).unwrap();
    store.replace(&History::from_entries(job(1)), 3).unwrap();
    store.set_epoch(4).unwrap();
    drop(store);

    assert_eq!(
        fs::read(meta_path(&dir)).unwrap(),
        meta_before,
        "meta.json holds identity only and is never rewritten"
    );
    let store = HistoryStore::load(&dir.0, CONVERSATION).unwrap();
    assert_eq!(store.epoch(), 4);
    assert_eq!(store.history(), &History::from_entries(job(1)));
}

#[test]
fn epoch_never_goes_down() {
    let dir = DataDir::new("epoch-monotonic");
    let mut store = HistoryStore::open(&dir.0, meta()).unwrap();
    store.set_epoch(5).unwrap();
    store.set_epoch(3).unwrap();
    assert_eq!(store.epoch(), 5);
    store.replace(&History::from_entries(job(0)), 2).unwrap();
    assert_eq!(store.epoch(), 5);
    drop(store);
    assert_eq!(HistoryStore::load(&dir.0, CONVERSATION).unwrap().epoch(), 5);
}

#[test]
fn torn_write_keeps_the_epoch_it_was_written_under() {
    let dir = DataDir::new("torn-epoch");
    let mut store = HistoryStore::open(&dir.0, meta()).unwrap();
    store.append(&job(0)).unwrap();
    drop(store);
    // A record(epoch 6, entries) cut off after the epoch line, mid-way through the entries.
    append_raw(
        &dir.log_path(),
        br#"{"generation":0,"epoch":6}
{"generation":0,"index":3,"entry":{"type":"JobStarted","request_id":"job-1","prompt":"p"}}
{"generation":0,"index":4,"entry":{"ty"#,
    );

    let store = HistoryStore::load(&dir.0, CONVERSATION).unwrap();
    assert_eq!(store.epoch(), 6);
    assert_eq!(
        store.len(),
        4,
        "complete lines are kept, the torn one dropped"
    );
}

#[test]
fn epoch_covers_every_epoch_advanced_in_the_history() {
    let dir = DataDir::new("epoch-from-history");
    let mut store = HistoryStore::open(&dir.0, meta()).unwrap();
    let advanced = Entry::EpochAdvanced {
        epoch: 7,
        leader: id(0),
    };
    store.append(std::slice::from_ref(&advanced)).unwrap();
    assert_eq!(store.epoch(), 7);
    drop(store);
    assert_eq!(HistoryStore::load(&dir.0, CONVERSATION).unwrap().epoch(), 7);

    // Even if a log only has the entry (no epoch line), loading never reports a lower epoch.
    let dir = DataDir::new("epoch-from-history-raw");
    drop(HistoryStore::open(&dir.0, meta()).unwrap());
    let line = format!(
        "{}\n",
        serde_json::json!({"generation": 0, "index": 0, "entry": advanced})
    );
    append_raw(&dir.log_path(), line.as_bytes());
    assert_eq!(HistoryStore::load(&dir.0, CONVERSATION).unwrap().epoch(), 7);
}

#[test]
fn version_1_stores_are_refused() {
    let dir = DataDir::new("v1");
    drop(HistoryStore::open(&dir.0, meta()).unwrap());
    let v1 = serde_json::json!({
        "format_version": 1,
        "conversation_id": CONVERSATION,
        "cluster_id": "cluster-a",
        "model": {"model_hash": "sha256:abc", "generation": {}},
        "epoch": 2
    });
    fs::write(meta_path(&dir), v1.to_string()).unwrap();
    assert!(matches!(
        HistoryStore::load(&dir.0, CONVERSATION),
        Err(StoreError::UnsupportedVersion { found: 1, .. })
    ));
}
