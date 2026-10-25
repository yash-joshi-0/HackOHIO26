//! Durable, per-conversation history on disk, and the export format used for recovery.
//!
//! Layout under `<data_dir>/history/<conversation_id>/`:
//! - `meta.json`: format version, conversation and cluster IDs, model manifest; written once.
//! - `snapshot.json`: the epoch and the first entries of the history; replaced atomically.
//! - `log.jsonl`: one line per change after the snapshot, either
//!   `{"generation", "index", "entry"}` or `{"generation", "epoch"}`.
//!
//! The epoch lives with the history, never in a separate file, so a crash can't leave new
//! history with an old epoch. Writes reach the disk before [`HistoryStore::record`]
//! returns. A crash at any point leaves a loadable prefix: a torn final log line is
//! dropped, log lines the snapshot already covers are skipped, and lines from before a
//! [`HistoryStore::replace`] carry an older generation and are ignored.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::history_sync::history::{Entry, Epoch, History, Seq};

/// Version of the on-disk layout. Version 1 kept the epoch in `meta.json`.
pub const FORMAT_VERSION: u32 = 2;

/// Version of [`ConversationExport`].
pub const EXPORT_FORMAT_VERSION: u32 = 1;

/// The log is folded into a new snapshot once it holds this many entries.
pub const SNAPSHOT_EVERY: Seq = 100;

/// Data directory used by the CLI when `--data-dir` is not given.
pub const DEFAULT_DATA_DIR: &str = "meshthing-data";

const META_FILE: &str = "meta.json";
const SNAPSHOT_FILE: &str = "snapshot.json";
const LOG_FILE: &str = "log.jsonl";

/// Longest accepted conversation ID; it becomes a directory name.
const MAX_CONVERSATION_ID_LEN: usize = 128;

/// The model a conversation was generated with; a recovered leader needs it to re-prefill.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Hash of the exact model file, e.g. `sha256:…`.
    pub model_hash: String,
    /// Generation settings needed to continue identically (context size, sampler, template, …).
    #[serde(default)]
    pub generation: BTreeMap<String, String>,
}

/// Identity of a stored conversation (`meta.json`); never changes after creation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreMeta {
    pub format_version: u32,
    pub conversation_id: String,
    pub cluster_id: String,
    pub model: ModelManifest,
}

impl StoreMeta {
    pub fn new(conversation_id: &str, cluster_id: &str, model: ModelManifest) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            conversation_id: conversation_id.to_owned(),
            cluster_id: cluster_id.to_owned(),
            model,
        }
    }
}

/// `snapshot.json`. `generation` changes on every [`HistoryStore::replace`].
#[derive(Serialize, Deserialize)]
struct Snapshot {
    generation: u64,
    epoch: Epoch,
    entries: Vec<Entry>,
}

/// Borrowed snapshot used for serialization so compaction doesn't clone the full history.
#[derive(Serialize)]
struct SnapshotRef<'a> {
    generation: u64,
    epoch: Epoch,
    entries: &'a [Entry],
}

/// One line of `log.jsonl`, as read.
#[derive(Deserialize)]
#[serde(untagged)]
enum LogLine {
    Entry {
        generation: u64,
        index: Seq,
        entry: Entry,
    },
    Epoch {
        generation: u64,
        epoch: Epoch,
    },
}

/// An entry line of `log.jsonl`, as written.
#[derive(Serialize)]
struct EntryLineRef<'a> {
    generation: u64,
    index: Seq,
    entry: &'a Entry,
}

/// An epoch line of `log.jsonl`, as written.
#[derive(Serialize)]
struct EpochLine {
    generation: u64,
    epoch: Epoch,
}

/// Errors from reading or writing a stored conversation.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{path}: invalid JSON: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("{path}: {reason}")]
    Corrupt { path: PathBuf, reason: String },
    #[error(
        "invalid conversation ID {0:?}: use 1-{MAX_CONVERSATION_ID_LEN} letters, digits, '-', '_' or '.'"
    )]
    InvalidConversationId(String),
    #[error("{path}: stored {field} {stored:?} does not match {expected:?}")]
    Mismatch {
        path: PathBuf,
        field: &'static str,
        stored: String,
        expected: String,
    },
    #[error("{path}: stored generation settings {stored:?} do not match {expected:?}")]
    GenerationSettingsMismatch {
        path: PathBuf,
        stored: BTreeMap<String, String>,
        expected: BTreeMap<String, String>,
    },
    #[error("no stored conversation at {0}")]
    NotFound(PathBuf),
    #[error("{path}: history generation counter is exhausted")]
    GenerationExhausted { path: PathBuf },
    #[error("{path}: format version {found} is not supported (expected {FORMAT_VERSION})")]
    UnsupportedVersion { path: PathBuf, found: u32 },
}

/// One conversation's history, kept in memory and mirrored to disk.
#[derive(Debug)]
pub struct HistoryStore {
    dir: PathBuf,
    meta: StoreMeta,
    history: History,
    /// Highest leadership epoch this node has accepted for the conversation.
    epoch: Epoch,
    generation: u64,
    /// How many entries of `history` are in `snapshot.json`; the rest are in the log.
    snapshot_len: Seq,
    log: File,
}

impl HistoryStore {
    /// Opens the stored conversation `meta.conversation_id`, creating it from `meta` if absent.
    ///
    /// An existing conversation must belong to the same cluster and model; its stored epoch
    /// and history are kept. A new one starts empty in epoch 0.
    pub fn open(data_dir: &Path, meta: StoreMeta) -> Result<Self, StoreError> {
        let dir = conversation_dir(data_dir, &meta.conversation_id)?;
        if dir.join(META_FILE).exists() {
            let store = Self::load(data_dir, &meta.conversation_id)?;
            let path = dir.join(META_FILE);
            check_same(
                &path,
                "cluster ID",
                &store.meta.cluster_id,
                &meta.cluster_id,
            )?;
            check_same(
                &path,
                "model hash",
                &store.meta.model.model_hash,
                &meta.model.model_hash,
            )?;
            if store.meta.model.generation != meta.model.generation {
                return Err(StoreError::GenerationSettingsMismatch {
                    path,
                    stored: store.meta.model.generation.clone(),
                    expected: meta.model.generation,
                });
            }
            return Ok(store);
        }
        fs::create_dir_all(&dir).map_err(io_at(&dir))?;
        let meta = StoreMeta {
            format_version: FORMAT_VERSION,
            ..meta
        };
        let snapshot = Snapshot {
            generation: 0,
            epoch: 0,
            entries: Vec::new(),
        };
        write_json_atomic(&dir.join(SNAPSHOT_FILE), &snapshot)?;
        let log = open_log(&dir.join(LOG_FILE))?;
        // meta.json last: its presence marks the conversation as fully created.
        write_json_atomic(&dir.join(META_FILE), &meta)?;
        Ok(Self {
            dir,
            meta,
            history: History::new(),
            epoch: 0,
            generation: 0,
            snapshot_len: 0,
            log,
        })
    }

    /// Opens an existing stored conversation without creating anything.
    pub fn load(data_dir: &Path, conversation_id: &str) -> Result<Self, StoreError> {
        let dir = conversation_dir(data_dir, conversation_id)?;
        let meta_path = dir.join(META_FILE);
        if !meta_path.exists() {
            return Err(StoreError::NotFound(dir));
        }
        let meta: StoreMeta = read_json(&meta_path)?;
        if meta.format_version != FORMAT_VERSION {
            return Err(StoreError::UnsupportedVersion {
                path: meta_path,
                found: meta.format_version,
            });
        }
        check_same(
            &meta_path,
            "conversation ID",
            &meta.conversation_id,
            conversation_id,
        )?;

        let snapshot: Snapshot = read_json(&dir.join(SNAPSHOT_FILE))?;
        let snapshot_len = snapshot.entries.len() as Seq;
        let mut history = History::from_entries(snapshot.entries);
        let mut epoch = snapshot.epoch;
        let log_path = dir.join(LOG_FILE);
        for line in read_log(&log_path)? {
            match line {
                LogLine::Epoch {
                    generation,
                    epoch: logged,
                } if generation == snapshot.generation => epoch = epoch.max(logged),
                LogLine::Entry {
                    generation,
                    index,
                    entry,
                } if generation == snapshot.generation && index >= snapshot_len => {
                    if index != history.len() {
                        return Err(StoreError::Corrupt {
                            path: log_path,
                            reason: format!(
                                "expected entry {} but found entry {index}",
                                history.len()
                            ),
                        });
                    }
                    history.extend([entry]);
                }
                _ => {}
            }
        }
        // A history is never newer than the epoch that wrote it.
        epoch = epoch.max(history.last_epoch());
        let log = open_log(&log_path)?;
        Ok(Self {
            dir,
            meta,
            history,
            epoch,
            generation: snapshot.generation,
            snapshot_len,
            log,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn meta(&self) -> &StoreMeta {
        &self.meta
    }

    pub fn history(&self) -> &History {
        &self.history
    }

    pub fn epoch(&self) -> Epoch {
        self.epoch
    }

    pub fn len(&self) -> Seq {
        self.history.len()
    }

    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Appends entries to the end of the history; they are on disk when this returns.
    pub fn append(&mut self, entries: &[Entry]) -> Result<(), StoreError> {
        self.record(self.epoch, entries)
    }

    /// Records a newer leadership epoch. An epoch at or below the stored one is ignored.
    pub fn set_epoch(&mut self, epoch: Epoch) -> Result<(), StoreError> {
        self.record(epoch, &[])
    }

    /// Records a newer epoch (if `epoch` is above the stored one) and appends `entries`,
    /// in one write that is on disk when this returns.
    ///
    /// The epoch line comes first, so a crash part-way through keeps the epoch and a
    /// prefix of the entries, never entries without the epoch they were written under.
    pub fn record(&mut self, epoch: Epoch, entries: &[Entry]) -> Result<(), StoreError> {
        // Entries can't be newer than the epoch stored with them.
        let epoch = entries
            .iter()
            .filter_map(|entry| match entry {
                Entry::EpochAdvanced { epoch, .. } => Some(*epoch),
                _ => None,
            })
            .fold(epoch, Epoch::max);
        let raise_epoch = epoch > self.epoch;
        if !raise_epoch && entries.is_empty() {
            return Ok(());
        }
        let log_path = self.dir.join(LOG_FILE);
        let mut buffer = Vec::new();
        if raise_epoch {
            let line = EpochLine {
                generation: self.generation,
                epoch,
            };
            push_line(&mut buffer, &line, &log_path)?;
        }
        for (offset, entry) in entries.iter().enumerate() {
            let line = EntryLineRef {
                generation: self.generation,
                index: self.history.len() + offset as Seq,
                entry,
            };
            push_line(&mut buffer, &line, &log_path)?;
        }
        // One write per call, so a crash tears at most the last line.
        self.log.write_all(&buffer).map_err(io_at(&log_path))?;
        self.log.sync_data().map_err(io_at(&log_path))?;
        self.epoch = self.epoch.max(epoch);
        self.history.extend(entries.iter().cloned());
        if self.history.len() - self.snapshot_len >= SNAPSHOT_EVERY
            && let Err(error) = self.compact()
        {
            // The log append was already synced above. Returning an error here would tell the
            // caller the append failed even though replay after restart includes these entries.
            warn!(%error, "history append is durable, but snapshot compaction failed");
        }
        Ok(())
    }

    /// Replaces the whole history, e.g. with a snapshot from a new leader, together with
    /// its epoch, in one atomic write. The epoch never goes down.
    pub fn replace(&mut self, history: &History, epoch: Epoch) -> Result<(), StoreError> {
        // Write the new snapshot under a new generation first: until the log is truncated,
        // its old lines carry the previous generation and are ignored on load.
        let generation =
            self.generation
                .checked_add(1)
                .ok_or_else(|| StoreError::GenerationExhausted {
                    path: self.dir.join(SNAPSHOT_FILE),
                })?;
        let epoch = epoch.max(self.epoch);
        self.write_snapshot(generation, epoch, history)?;
        self.truncate_log()?;
        self.generation = generation;
        self.epoch = epoch;
        self.history = history.clone();
        self.snapshot_len = self.history.len();
        Ok(())
    }

    /// Everything this node has stored for the conversation, in the export format.
    pub fn export(&self) -> ConversationExport {
        ConversationExport {
            format_version: EXPORT_FORMAT_VERSION,
            conversation_id: self.meta.conversation_id.clone(),
            cluster_id: self.meta.cluster_id.clone(),
            model: self.meta.model.clone(),
            epoch: self.epoch,
            committed_len: self.history.len(),
            jobs: job_states(&self.history),
            entries: self.history.entries().to_vec(),
        }
    }

    /// Folds the log into the snapshot so loading stays fast and the log stays short.
    fn compact(&mut self) -> Result<(), StoreError> {
        self.write_snapshot(self.generation, self.epoch, &self.history)?;
        self.truncate_log()?;
        self.snapshot_len = self.history.len();
        Ok(())
    }

    fn write_snapshot(
        &self,
        generation: u64,
        epoch: Epoch,
        history: &History,
    ) -> Result<(), StoreError> {
        let snapshot = SnapshotRef {
            generation,
            epoch,
            entries: history.entries(),
        };
        write_json_atomic(&self.dir.join(SNAPSHOT_FILE), &snapshot)
    }

    /// Empties the log. Uses a separate write handle: on Windows an append-only handle
    /// lacks `FILE_WRITE_DATA`, so `set_len` on `self.log` fails with access denied.
    /// Appends through `self.log` always go to the current end, so they land at offset 0 next.
    fn truncate_log(&mut self) -> Result<(), StoreError> {
        let path = self.dir.join(LOG_FILE);
        let file = OpenOptions::new()
            .write(true)
            .open(&path)
            .map_err(io_at(&path))?;
        file.set_len(0).map_err(io_at(&path))?;
        file.sync_all().map_err(io_at(&path))
    }
}

/// A stored conversation in the form recovery tools consume (serialized as JSON).
///
/// "Committed" means on this node's disk; it does not yet imply a quorum of nodes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationExport {
    pub format_version: u32,
    pub conversation_id: String,
    pub cluster_id: String,
    pub model: ModelManifest,
    pub epoch: Epoch,
    /// Number of committed entries: the log boundary a restore resumes from.
    pub committed_len: Seq,
    pub jobs: Vec<JobState>,
    pub entries: Vec<Entry>,
}

/// Whether a job's response was fully committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Completed,
    /// Started but never completed: re-run it, or resume after `committed_output_offset`.
    Interrupted,
}

/// One job reconstructed from the log.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobState {
    pub request_id: String,
    pub prompt: String,
    pub status: JobStatus,
    /// The response text committed so far.
    pub committed_output: String,
    /// Length of `committed_output` in bytes; the offset the next chunk would start at.
    pub committed_output_offset: u64,
}

/// Rebuilds each job's state from the log, in the order the jobs started.
///
/// Output chunks are joined only while they line up end to end; a chunk that repeats
/// earlier bytes, or leaves a gap, is ignored. Events for unknown requests are ignored.
pub fn job_states(history: &History) -> Vec<JobState> {
    let mut jobs: Vec<JobState> = Vec::new();
    for entry in history.entries() {
        match entry {
            Entry::JobStarted { request_id, prompt } => jobs.push(JobState {
                request_id: request_id.clone(),
                prompt: prompt.clone(),
                status: JobStatus::Interrupted,
                committed_output: String::new(),
                committed_output_offset: 0,
            }),
            Entry::OutputCommitted {
                request_id,
                offset,
                text,
            } => {
                if let Some(job) = find_job(&mut jobs, request_id)
                    && *offset == job.committed_output_offset
                {
                    job.committed_output.push_str(text);
                    job.committed_output_offset += text.len() as u64;
                }
            }
            Entry::JobCompleted { request_id } => {
                if let Some(job) = find_job(&mut jobs, request_id) {
                    job.status = JobStatus::Completed;
                }
            }
            Entry::EpochAdvanced { .. } => {}
        }
    }
    jobs
}

/// The most recent job with `request_id`.
fn find_job<'a>(jobs: &'a mut [JobState], request_id: &str) -> Option<&'a mut JobState> {
    jobs.iter_mut()
        .rev()
        .find(|job| job.request_id == request_id)
}

/// `<data_dir>/history/<conversation_id>`, after checking the ID is a safe directory name.
pub fn conversation_dir(data_dir: &Path, conversation_id: &str) -> Result<PathBuf, StoreError> {
    let valid = !conversation_id.is_empty()
        && conversation_id.len() <= MAX_CONVERSATION_ID_LEN
        && conversation_id != "."
        && conversation_id != ".."
        && conversation_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if !valid {
        return Err(StoreError::InvalidConversationId(
            conversation_id.to_owned(),
        ));
    }
    Ok(data_dir.join("history").join(conversation_id))
}

fn check_same(
    path: &Path,
    field: &'static str,
    stored: &str,
    expected: &str,
) -> Result<(), StoreError> {
    if stored == expected {
        return Ok(());
    }
    Err(StoreError::Mismatch {
        path: path.to_owned(),
        field,
        stored: stored.to_owned(),
        expected: expected.to_owned(),
    })
}

fn io_at(path: &Path) -> impl FnOnce(io::Error) -> StoreError + '_ {
    move |source| StoreError::Io {
        path: path.to_owned(),
        source,
    }
}

/// Serializes `line` as one JSON line onto `buffer`.
fn push_line<T: Serialize>(buffer: &mut Vec<u8>, line: &T, path: &Path) -> Result<(), StoreError> {
    serde_json::to_writer(&mut *buffer, line).map_err(|source| StoreError::Json {
        path: path.to_owned(),
        source,
    })?;
    buffer.push(b'\n');
    Ok(())
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, StoreError> {
    let bytes = fs::read(path).map_err(io_at(path))?;
    serde_json::from_slice(&bytes).map_err(|source| StoreError::Json {
        path: path.to_owned(),
        source,
    })
}

/// Writes `value` to a temporary file, flushes it, then renames it over `path`, so readers
/// see either the old file or the complete new one.
fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), StoreError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|source| StoreError::Json {
        path: path.to_owned(),
        source,
    })?;
    let temp = path.with_extension("tmp");
    let mut file = File::create(&temp).map_err(io_at(&temp))?;
    file.write_all(&bytes).map_err(io_at(&temp))?;
    file.sync_all().map_err(io_at(&temp))?;
    drop(file);
    fs::rename(&temp, path).map_err(io_at(path))?;
    sync_parent_dir(path)
}

/// Makes a rename durable. Only Unix can open and sync a directory.
fn sync_parent_dir(path: &Path) -> Result<(), StoreError> {
    #[cfg(unix)]
    if let Some(dir) = path.parent() {
        File::open(dir)
            .and_then(|dir_file| dir_file.sync_all())
            .map_err(io_at(dir))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn open_log(path: &Path) -> Result<File, StoreError> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(io_at(path))
}

/// Reads every complete log line. A final line without its newline was torn by a crash
/// mid-write: it is dropped and cut from the file.
fn read_log(path: &Path) -> Result<Vec<LogLine>, StoreError> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(io_at(path)(error)),
    };
    let complete = bytes
        .iter()
        .rposition(|&byte| byte == b'\n')
        .map_or(0, |last| last + 1);
    if complete < bytes.len() {
        let file = OpenOptions::new()
            .write(true)
            .open(path)
            .map_err(io_at(path))?;
        file.set_len(complete as u64).map_err(io_at(path))?;
        file.sync_all().map_err(io_at(path))?;
    }
    bytes[..complete]
        .split(|&byte| byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| {
            serde_json::from_slice(line).map_err(|source| StoreError::Json {
                path: path.to_owned(),
                source,
            })
        })
        .collect()
}
