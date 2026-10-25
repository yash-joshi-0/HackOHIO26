//! Shard manifest: which stages make up a sharded model and where each one runs.
//!
//! The leader's front end reads a JSON manifest such as:
//!
//! ```json
//! {
//!   "runner": "/opt/meshthing/shard_runner",
//!   "model_sha256": "<64 hex digits: the original model's SHA-256>",
//!   "layers": 64,
//!   "context": 4096,
//!   "max_tokens": 512,
//!   "chat_template": "chatml",
//!   "stages": [
//!     {"begin": 0, "end": 32, "remote": {"worker": "<iroh endpoint id>"}},
//!     {"begin": 32, "end": 64, "local": {"model": "/models/qwen-32-64.gguf", "budget_bytes": 1073741824}}
//!   ]
//! }
//! ```
//!
//! Each stage object carries `begin`, `end` and exactly one location key (`remote` or `local`).
//! A remote stage may list `"standby": ["<iroh endpoint id>", ...]`: workers holding the same
//! artifact that are tried, in order, when the primary fails. Unknown fields are refused
//! everywhere, so a typo cannot silently drop a setting.
//!
//! `chat_template` is either `"chatml"` or a pinned Jinja template object (see [`ChatTemplate`]).
//! [`ShardManifest::load`] resolves a relative template path against the manifest's directory.
//!
//! [`ShardManifest::validate`] enforces the structural rules (contiguous coverage of
//! `0..layers`, bounds on `context` and `max_tokens`, well-formed hashes, no worker listed
//! twice, as primary or standby) and the topology of the manifest's [`Execution`] mode:
//!
//! - `"execution": "pipeline"` (the default, M-25 v1): a single local whole-model stage, or one
//!   remote prefix followed by a local final stage.
//! - `"execution": "ring"` (EXPERIMENTAL, `meshthing/shard-ring/1`): a local stage `0..k` that
//!   tokenizes, then one or more remote ring workers covering `k..layers` in order. The last
//!   remote stage is the terminal, which returns tokens straight to the leader. Ring execution
//!   fails the job on any error, so its stages may not list `standby` workers.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail, ensure};
use iroh::EndpointId;
use serde::Deserialize;
use serde::de::{self, MapAccess, Visitor, value::MapAccessDeserializer};
use sha2::{Digest, Sha256};

use crate::config::read_json_file;
use crate::shard::wire::{self, MAX_CONTEXT};

/// Smallest context window the shard runner accepts.
pub const MIN_CONTEXT: u32 = 32;

/// Completion length used when the manifest does not set `max_tokens`.
pub const DEFAULT_MAX_TOKENS: u32 = 512;

/// Length of a SHA-256 digest written as hex.
const SHA256_HEX_LEN: usize = 64;

/// Bytes read per chunk while hashing a model file.
const HASH_CHUNK_BYTES: usize = 1 << 20;

/// Name of the built-in ChatML template in `chat_template`.
const CHATML: &str = "chatml";

/// Where pipeline-mode topology errors point: the experimental ring execution mode (M-27).
const RING_ALTERNATIVE: &str =
    "set \"execution\": \"ring\" for experimental ring forwarding (M-27)";

/// A sharded model: the runner binary, the model's identity, and its stages in layer order.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardManifest {
    /// `shard_runner` binary used for local stages.
    pub runner: PathBuf,
    /// SHA-256 of the original (unsharded) model, as 64 hex digits. Every stage must prove it
    /// serves a slice of this model.
    pub model_sha256: String,
    /// Total layers in the model.
    pub layers: u32,
    /// Context window passed to every stage, in `MIN_CONTEXT..=MAX_CONTEXT`.
    pub context: u32,
    /// Upper bound on completion tokens; a request may ask for fewer.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Prompt template the front end renders chat messages with. Required: a wrong template
    /// silently degrades output, so it is never assumed.
    pub chat_template: ChatTemplate,
    /// How the stages are driven; `pipeline` when absent.
    #[serde(default)]
    pub execution: Execution,
    /// Stages in layer order, covering `0..layers` without gaps or overlap.
    pub stages: Vec<StageSpec>,
}

/// How the leader drives a manifest's stages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Execution {
    /// M-25 v1: the leader exchanges hidden rows with every stage in turn.
    #[default]
    Pipeline,
    /// EXPERIMENTAL `meshthing/shard-ring/1`: the leader runs stage 0 and sends its rows to the
    /// first ring worker; workers forward activations peer to peer and the terminal returns
    /// tokens straight to the leader. Pending david-codex's production sign-off.
    Ring,
}

/// How chat messages become the prompt string the tokenizing stage (the final stage, or stage 0
/// in a ring) tokenizes. Written as the string `"chatml"` or as a pinned Jinja template object;
/// [`crate::shard::template::ChatRenderer`] renders both. New templates are added as new variants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatTemplate {
    /// `"chatml"`: `<|im_start|>{role}\n{content}<|im_end|>\n` per message, then
    /// `<|im_start|>assistant\n`.
    Chatml,
    /// `{"jinja": {"path": ..., "sha256": ...}, "enable_thinking": false}`: an official Jinja
    /// chat template (such as a GGUF's `tokenizer.chat_template`), pinned by its SHA-256 and
    /// checked once when the front end starts.
    Jinja(PinnedTemplate),
}

/// The object form of `chat_template`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedTemplate {
    /// The template file and its pinned hash.
    pub jinja: JinjaFile,
    /// Passed to the template as `enable_thinking`; `false` when absent.
    #[serde(default)]
    pub enable_thinking: bool,
}

/// A Jinja template file pinned by its SHA-256.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JinjaFile {
    /// The template file. [`ShardManifest::load`] resolves a relative path against the
    /// manifest's directory; a manifest built in memory keeps it relative to the working
    /// directory, as model paths are.
    pub path: PathBuf,
    /// SHA-256 of the file's bytes, as 64 hex digits.
    pub sha256: String,
}

impl<'de> Deserialize<'de> for ChatTemplate {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ChatTemplateVisitor)
    }
}

/// Accepts `"chatml"` or a [`PinnedTemplate`] object, with errors that name both forms.
struct ChatTemplateVisitor;

impl<'de> Visitor<'de> for ChatTemplateVisitor {
    type Value = ChatTemplate;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(
            "\"chatml\" or a pinned template {\"jinja\": {\"path\": ..., \"sha256\": ...}}",
        )
    }

    fn visit_str<E: de::Error>(self, name: &str) -> Result<ChatTemplate, E> {
        if name == CHATML {
            return Ok(ChatTemplate::Chatml);
        }
        Err(E::custom(format!(
            "unknown chat_template {name:?}: expected \"{CHATML}\" or a pinned \
             {{\"jinja\": {{\"path\": ..., \"sha256\": ...}}}} object"
        )))
    }

    fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<ChatTemplate, A::Error> {
        PinnedTemplate::deserialize(MapAccessDeserializer::new(map)).map(ChatTemplate::Jinja)
    }
}

impl ChatTemplate {
    /// Makes a relative Jinja template path relative to `dir` instead.
    fn resolve_relative_to(&mut self, dir: &Path) {
        if let Self::Jinja(pinned) = self
            && pinned.jinja.path.is_relative()
        {
            pinned.jinja.path = dir.join(&pinned.jinja.path);
        }
    }
}

/// One contiguous layer range and where it runs.
#[derive(Clone, Debug)]
pub struct StageSpec {
    /// First layer (inclusive).
    pub begin: u32,
    /// End of the range (exclusive).
    pub end: u32,
    /// Where the stage runs.
    pub location: StageLocation,
}

/// Where a stage runs. Open for more roles, so match it with a wildcard arm outside this crate.
/// In a ring manifest every `Remote` stage is a `shard-ring-worker`, not a `shard-serve` relay.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum StageLocation {
    /// On another peer running `shard-serve` (or `bridge-serve` + `shard_endpoint.py`), dialled
    /// on the RPC bridge ALPN.
    Remote {
        /// The worker's Iroh endpoint ID.
        worker: EndpointId,
        /// Workers that hold the identical stage artifact and take over, in order, when the
        /// primary cannot be reached or fails mid-generation. Empty by default.
        standby: Vec<EndpointId>,
    },
    /// In a `shard_runner` child process on this machine.
    Local {
        /// The model or exported shard file the runner loads.
        model: PathBuf,
        /// Layers the runner offloads to the GPU.
        gpu_layers: u32,
        /// Runtime buffer budget passed to the runner; must be positive.
        budget_bytes: u64,
        /// SHA-256 of `model`, checked once when the front end starts. It establishes the
        /// stage's identity when the runner reports no `source_sha256` (a plain GGUF).
        artifact_sha256: Option<String>,
        /// Extra environment variables for the runner process (for example
        /// `CUDA_VISIBLE_DEVICES`). Empty by default.
        env: BTreeMap<String, String>,
    },
}

/// The JSON form of a stage: range plus exactly one location key. Kept separate because
/// `deny_unknown_fields` does not combine with `#[serde(flatten)]`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStageSpec {
    begin: u32,
    end: u32,
    #[serde(default)]
    remote: Option<RawRemote>,
    #[serde(default)]
    local: Option<RawLocal>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRemote {
    worker: EndpointId,
    #[serde(default)]
    standby: Vec<EndpointId>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLocal {
    model: PathBuf,
    #[serde(default)]
    gpu_layers: u32,
    budget_bytes: u64,
    #[serde(default)]
    artifact_sha256: Option<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

impl<'de> Deserialize<'de> for StageSpec {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawStageSpec::deserialize(deserializer)?;
        let location = match (raw.remote, raw.local) {
            (Some(RawRemote { worker, standby }), None) => {
                StageLocation::Remote { worker, standby }
            }
            (None, Some(local)) => StageLocation::Local {
                model: local.model,
                gpu_layers: local.gpu_layers,
                budget_bytes: local.budget_bytes,
                artifact_sha256: local.artifact_sha256,
                env: local.env,
            },
            _ => {
                return Err(serde::de::Error::custom(format!(
                    "stage {}..{} needs exactly one of \"remote\" or \"local\"",
                    raw.begin, raw.end
                )));
            }
        };
        Ok(Self {
            begin: raw.begin,
            end: raw.end,
            location,
        })
    }
}

fn default_max_tokens() -> u32 {
    DEFAULT_MAX_TOKENS
}

impl StageSpec {
    /// Whether the stage runs in a local runner process.
    pub fn is_local(&self) -> bool {
        matches!(self.location, StageLocation::Local { .. })
    }

    /// The remote worker serving this stage, if any.
    pub fn remote_worker(&self) -> Option<EndpointId> {
        match self.location {
            StageLocation::Remote { worker, .. } => Some(worker),
            _ => None,
        }
    }

    /// Every worker that may serve this remote stage, in the order they are tried: the
    /// primary, then its standbys. Empty for a local stage.
    pub fn remote_candidates(&self) -> Vec<EndpointId> {
        match &self.location {
            StageLocation::Remote { worker, standby } => std::iter::once(*worker)
                .chain(standby.iter().copied())
                .collect(),
            _ => Vec::new(),
        }
    }
}

impl ShardManifest {
    /// Reads and parses the manifest at `path` (errors name the file), then validates it.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let mut manifest: Self = read_json_file(path, "shard manifest")?;
        if let Some(dir) = path.parent() {
            manifest.chat_template.resolve_relative_to(dir);
        }
        manifest
            .validate()
            .with_context(|| format!("invalid shard manifest {}", path.display()))?;
        Ok(manifest)
    }

    /// Checks the structural rules and the topology of the execution mode; see the module docs.
    pub fn validate(&self) -> anyhow::Result<()> {
        self.validate_limits()?;
        self.validate_coverage()?;
        self.validate_stages()?;
        match self.execution {
            Execution::Pipeline => self.validate_v1_topology(),
            Execution::Ring => self.validate_ring_topology(),
        }
    }

    /// Whether the manifest opts into ring execution.
    pub fn is_ring(&self) -> bool {
        self.execution == Execution::Ring
    }

    /// Every remote worker the front end may dial, in stage order: each remote stage's
    /// primary followed by its standbys.
    pub fn remote_workers(&self) -> Vec<EndpointId> {
        self.stages
            .iter()
            .flat_map(StageSpec::remote_candidates)
            .collect()
    }

    fn validate_limits(&self) -> anyhow::Result<()> {
        ensure!(
            (MIN_CONTEXT..=MAX_CONTEXT).contains(&self.context),
            "context {} is outside {MIN_CONTEXT}..={MAX_CONTEXT}",
            self.context
        );
        ensure!(self.max_tokens >= 1, "max_tokens must be at least 1");
        check_sha256(&self.model_sha256, "model_sha256")?;
        if let ChatTemplate::Jinja(pinned) = &self.chat_template {
            check_sha256(&pinned.jinja.sha256, "chat_template.jinja.sha256")?;
        }
        Ok(())
    }

    fn validate_coverage(&self) -> anyhow::Result<()> {
        ensure!(!self.stages.is_empty(), "the manifest has no stages");
        let mut next = 0;
        for stage in &self.stages {
            ensure!(
                stage.begin < stage.end,
                "stage {}..{} is empty or reversed",
                stage.begin,
                stage.end
            );
            ensure!(
                stage.begin == next,
                "stage {}..{} should begin at layer {next}: stages must cover 0..{} contiguously",
                stage.begin,
                stage.end,
                self.layers
            );
            next = stage.end;
        }
        ensure!(
            next == self.layers,
            "stages end at layer {next}, but the model has {} layers",
            self.layers
        );
        Ok(())
    }

    fn validate_stages(&self) -> anyhow::Result<()> {
        let mut workers = HashSet::new();
        for stage in &self.stages {
            match &stage.location {
                StageLocation::Remote { .. } => {
                    for worker in stage.remote_candidates() {
                        ensure!(
                            workers.insert(worker),
                            "worker {} is listed more than once: a worker may not serve more \
                             than one stage, as primary or standby",
                            worker.fmt_short()
                        );
                    }
                }
                StageLocation::Local {
                    budget_bytes,
                    artifact_sha256,
                    ..
                } => {
                    ensure!(
                        *budget_bytes > 0,
                        "local stage {}..{} needs a positive budget_bytes",
                        stage.begin,
                        stage.end
                    );
                    if let Some(hash) = artifact_sha256 {
                        check_sha256(hash, "artifact_sha256")?;
                    }
                }
            }
        }
        Ok(())
    }

    /// v1 accepts a single local stage, or one remote prefix plus a local final stage.
    fn validate_v1_topology(&self) -> anyhow::Result<()> {
        let remote = self.stages.iter().filter(|stage| !stage.is_local()).count();
        let local = self.stages.len() - remote;
        let ends_local = self.stages.last().is_some_and(StageSpec::is_local);
        ensure!(
            ends_local,
            "the final stage must be local in v1 (it tokenizes and samples); \
             for a remote final stage, {RING_ALTERNATIVE}"
        );
        ensure!(
            remote <= 1,
            "v1 allows at most one remote stage, found {remote}; \
             for chained remote stages, {RING_ALTERNATIVE}"
        );
        ensure!(
            local <= 1,
            "v1 runs only the final stage locally, found {local} local stages; \
             for other topologies, {RING_ALTERNATIVE}"
        );
        Ok(())
    }

    /// Ring execution: a local stage 0 that does not reach the last layer, then only remote
    /// stages (at least one) up to `layers`. Coverage and duplicate workers are checked before.
    fn validate_ring_topology(&self) -> anyhow::Result<()> {
        let (first, rest) = self
            .stages
            .split_first()
            .context("the manifest has no stages")?;
        ensure!(
            first.is_local() && first.begin == 0,
            "ring execution needs a local stage 0 starting at layer 0 (it tokenizes and runs \
             the first layers), but stage {}..{} is remote",
            first.begin,
            first.end
        );
        ensure!(
            first.end < self.layers && !rest.is_empty(),
            "ring execution needs at least one remote stage after the local stage {}..{}",
            first.begin,
            first.end
        );
        if let Some(local) = rest.iter().find(|stage| stage.is_local()) {
            bail!(
                "ring execution runs only stage 0 locally, but stage {}..{} is local too; \
                 every later stage must be a remote ring worker",
                local.begin,
                local.end
            );
        }
        if let Some(stage) = rest
            .iter()
            .find(|stage| stage.remote_candidates().len() > 1)
        {
            bail!(
                "ring execution does not support standby workers (stage {}..{} lists some): \
                 a failed ring job fails rather than retrying; use \"execution\": \"pipeline\" \
                 for standby failover",
                stage.begin,
                stage.end
            );
        }
        Ok(())
    }
}

/// Whether two hex digests name the same bytes (hex case is not significant).
pub fn same_sha256(left: &str, right: &str) -> bool {
    left.eq_ignore_ascii_case(right)
}

/// Hashes a file with SHA-256 and returns lowercase hex. Blocking: call it from
/// `spawn_blocking` in async code.
pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; HASH_CHUNK_BYTES];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(wire::encode_hex(&hasher.finalize()))
}

fn check_sha256(value: &str, field: &str) -> anyhow::Result<()> {
    if value.len() != SHA256_HEX_LEN || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{field} must be {SHA256_HEX_LEN} hex digits, got {value:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    const HASH: &str = "5ca1ab1e5ca1ab1e5ca1ab1e5ca1ab1e5ca1ab1e5ca1ab1e5ca1ab1e5ca1ab1e";

    fn worker_id(seed: u8) -> String {
        iroh::SecretKey::from_bytes(&[seed; 32])
            .public()
            .to_string()
    }

    fn local(begin: u32, end: u32) -> Value {
        json!({"begin": begin, "end": end, "local": {"model": "shard.gguf", "budget_bytes": 1024}})
    }

    fn remote(begin: u32, end: u32, seed: u8) -> Value {
        json!({"begin": begin, "end": end, "remote": {"worker": worker_id(seed)}})
    }

    fn manifest_json(stages: Vec<Value>) -> Value {
        json!({
            "runner": "shard_runner",
            "model_sha256": HASH,
            "layers": 4,
            "context": 128,
            "chat_template": "chatml",
            "stages": stages,
        })
    }

    fn parse(value: Value) -> anyhow::Result<ShardManifest> {
        Ok(serde_json::from_value(value)?)
    }

    fn validate(value: Value) -> anyhow::Result<ShardManifest> {
        let manifest = parse(value)?;
        manifest.validate()?;
        Ok(manifest)
    }

    fn error_of(value: Value) -> String {
        format!(
            "{:#}",
            validate(value).expect_err("manifest must be rejected")
        )
    }

    #[test]
    fn remote_prefix_and_local_final_is_valid() {
        let manifest = validate(manifest_json(vec![remote(0, 2, 1), local(2, 4)])).unwrap();
        assert_eq!(manifest.max_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(manifest.chat_template, ChatTemplate::Chatml);
        assert_eq!(manifest.remote_workers().len(), 1);
        assert_eq!(manifest.remote_workers()[0].to_string(), worker_id(1));
        let StageLocation::Local {
            gpu_layers,
            artifact_sha256,
            env,
            ..
        } = &manifest.stages[1].location
        else {
            panic!("final stage should be local");
        };
        assert_eq!((*gpu_layers, artifact_sha256, env.len()), (0, &None, 0));
    }

    #[test]
    fn single_local_stage_is_valid() {
        let manifest = validate(manifest_json(vec![local(0, 4)])).unwrap();
        assert!(manifest.remote_workers().is_empty());
    }

    #[test]
    fn chat_template_is_required_and_only_chatml_is_known() {
        let mut value = manifest_json(vec![local(0, 4)]);
        value.as_object_mut().unwrap().remove("chat_template");
        assert!(parse(value.clone()).is_err());
        value["chat_template"] = json!("llama2");
        let error = parse(value.clone()).unwrap_err().to_string();
        assert!(
            error.contains("llama2") && error.contains("jinja"),
            "{error}"
        );
        value["chat_template"] = json!(7);
        assert!(parse(value).is_err());
    }

    fn pinned_template(template: Value) -> Value {
        let mut value = manifest_json(vec![local(0, 4)]);
        value["chat_template"] = template;
        value
    }

    #[test]
    fn pinned_jinja_template_parses_with_thinking_off_by_default() {
        let jinja = json!({"path": "qwen.jinja", "sha256": HASH});
        let manifest = validate(pinned_template(json!({"jinja": jinja}))).unwrap();
        let expected = ChatTemplate::Jinja(PinnedTemplate {
            jinja: JinjaFile {
                path: PathBuf::from("qwen.jinja"),
                sha256: HASH.to_owned(),
            },
            enable_thinking: false,
        });
        assert_eq!(manifest.chat_template, expected);
        let thinking = pinned_template(json!({"jinja": jinja, "enable_thinking": true}));
        let ChatTemplate::Jinja(pinned) = validate(thinking).unwrap().chat_template else {
            panic!("expected a pinned template");
        };
        assert!(pinned.enable_thinking);
    }

    #[test]
    fn pinned_jinja_template_is_strict() {
        let jinja = json!({"path": "qwen.jinja", "sha256": HASH});
        for bad in [
            json!({}),
            json!({"jinja": {"path": "qwen.jinja"}}),
            json!({"jinja": {"sha256": HASH}}),
            json!({"jinja": jinja, "tools": []}),
            json!({"jinja": {"path": "qwen.jinja", "sha256": HASH, "url": "x"}}),
            json!({"jinja": jinja, "enable_thinking": "no"}),
        ] {
            assert!(parse(pinned_template(bad.clone())).is_err(), "{bad}");
        }
        let short = json!({"jinja": {"path": "qwen.jinja", "sha256": "abc"}});
        assert!(error_of(pinned_template(short)).contains("chat_template.jinja.sha256"));
    }

    #[test]
    fn load_resolves_a_relative_template_path_against_the_manifest() {
        let dir = std::env::temp_dir().join(format!("shard-template-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let absolute = dir.join("elsewhere.jinja");
        for (given, expected) in [
            (PathBuf::from("qwen.jinja"), dir.join("qwen.jinja")),
            (absolute.clone(), absolute),
        ] {
            let path = dir.join("manifest.json");
            let template = json!({"jinja": {"path": given, "sha256": HASH}});
            std::fs::write(&path, pinned_template(template).to_string()).unwrap();
            let ChatTemplate::Jinja(pinned) = ShardManifest::load(&path).unwrap().chat_template
            else {
                panic!("expected a pinned template");
            };
            assert_eq!(pinned.jinja.path, expected);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unknown_fields_are_refused_at_every_level() {
        let mut top = manifest_json(vec![local(0, 4)]);
        top["temperature"] = json!(0.7);
        assert!(parse(top).is_err());
        let mut stage = local(0, 4);
        stage["gpu"] = json!(1);
        assert!(parse(manifest_json(vec![stage])).is_err());
        let mut inner = local(0, 4);
        inner["local"]["budget"] = json!(1);
        assert!(parse(manifest_json(vec![inner])).is_err());
    }

    #[test]
    fn stage_needs_exactly_one_location() {
        let mut both = local(0, 4);
        both["remote"] = json!({"worker": worker_id(1)});
        let error = parse(manifest_json(vec![both])).unwrap_err().to_string();
        assert!(error.contains("exactly one"), "{error}");
        let neither = json!({"begin": 0, "end": 4});
        assert!(parse(manifest_json(vec![neither])).is_err());
    }

    #[test]
    fn stages_must_cover_all_layers_contiguously() {
        assert!(error_of(manifest_json(vec![])).contains("no stages"));
        assert!(error_of(manifest_json(vec![local(1, 4)])).contains("begin at layer 0"));
        assert!(error_of(manifest_json(vec![local(0, 3)])).contains("4 layers"));
        let gap = manifest_json(vec![remote(0, 1, 1), local(2, 4)]);
        assert!(error_of(gap).contains("begin at layer 1"));
        let overlap = manifest_json(vec![remote(0, 3, 1), local(2, 4)]);
        assert!(error_of(overlap).contains("begin at layer 3"));
        let empty = manifest_json(vec![remote(0, 0, 1), local(0, 4)]);
        assert!(error_of(empty).contains("empty or reversed"));
    }

    #[test]
    fn context_must_be_in_range() {
        for (context, valid) in [(31, false), (32, true), (MAX_CONTEXT, true), (4097, false)] {
            let mut value = manifest_json(vec![local(0, 4)]);
            value["context"] = json!(context);
            assert_eq!(validate(value).is_ok(), valid, "context {context}");
        }
    }

    #[test]
    fn max_tokens_must_be_positive() {
        let mut value = manifest_json(vec![local(0, 4)]);
        value["max_tokens"] = json!(0);
        assert!(error_of(value.clone()).contains("max_tokens"));
        value["max_tokens"] = json!(1);
        assert_eq!(validate(value).unwrap().max_tokens, 1);
    }

    #[test]
    fn hashes_must_be_64_hex_digits() {
        for bad in ["", "abc", &HASH[1..], &format!("{}g", &HASH[1..])] {
            let mut value = manifest_json(vec![local(0, 4)]);
            value["model_sha256"] = json!(bad);
            assert!(error_of(value).contains("model_sha256"), "{bad:?}");
        }
        let mut upper = manifest_json(vec![local(0, 4)]);
        upper["model_sha256"] = json!(HASH.to_uppercase());
        assert!(validate(upper).is_ok());
        let mut stage = local(0, 4);
        stage["local"]["artifact_sha256"] = json!("1234");
        assert!(error_of(manifest_json(vec![stage])).contains("artifact_sha256"));
    }

    #[test]
    fn local_budget_must_be_positive() {
        let mut stage = local(0, 4);
        stage["local"]["budget_bytes"] = json!(0);
        assert!(error_of(manifest_json(vec![stage])).contains("budget_bytes"));
    }

    #[test]
    fn duplicate_workers_are_refused() {
        // Duplicates are checked before the v1 topology rules, so this is reported first.
        let value = manifest_json(vec![remote(0, 1, 7), remote(1, 2, 7), local(2, 4)]);
        assert!(error_of(value).contains("more than once"));
    }

    fn with_standby(mut stage: Value, seeds: &[u8]) -> Value {
        let ids: Vec<String> = seeds.iter().map(|&seed| worker_id(seed)).collect();
        stage["remote"]["standby"] = json!(ids);
        stage
    }

    #[test]
    fn standby_workers_follow_the_primary() {
        let stage = with_standby(remote(0, 2, 1), &[2, 3]);
        let manifest = validate(manifest_json(vec![stage, local(2, 4)])).unwrap();
        let expected: Vec<String> = [1, 2, 3].map(worker_id).into();
        let ids = |workers: Vec<EndpointId>| -> Vec<String> {
            workers.iter().map(EndpointId::to_string).collect()
        };
        assert_eq!(ids(manifest.stages[0].remote_candidates()), expected);
        assert_eq!(ids(manifest.remote_workers()), expected);
        assert_eq!(
            manifest.stages[0].remote_worker().map(|id| id.to_string()),
            Some(worker_id(1))
        );
        assert!(manifest.stages[1].remote_candidates().is_empty());
        let plain = validate(manifest_json(vec![remote(0, 2, 1), local(2, 4)])).unwrap();
        assert_eq!(plain.stages[0].remote_candidates().len(), 1);
    }

    #[test]
    fn standby_workers_must_be_distinct_everywhere() {
        let twice = with_standby(remote(0, 2, 1), &[2, 2]);
        assert!(error_of(manifest_json(vec![twice, local(2, 4)])).contains("more than once"));
        let primary_again = with_standby(remote(0, 2, 1), &[1]);
        let error = error_of(manifest_json(vec![primary_again, local(2, 4)]));
        assert!(error.contains("more than once"), "{error}");
        // A standby reused by another stage is refused before the v1 topology rules.
        let first = with_standby(remote(0, 1, 1), &[3]);
        let second = remote(1, 2, 3);
        let error = error_of(manifest_json(vec![first, second, local(2, 4)]));
        assert!(error.contains("more than once"), "{error}");
    }

    #[test]
    fn standby_entries_must_be_endpoint_ids_and_unknown_remote_fields_are_refused() {
        let mut bad_id = remote(0, 2, 1);
        bad_id["remote"]["standby"] = json!(["not-an-id"]);
        assert!(parse(manifest_json(vec![bad_id, local(2, 4)])).is_err());
        let mut not_list = remote(0, 2, 1);
        not_list["remote"]["standby"] = json!(worker_id(2));
        assert!(parse(manifest_json(vec![not_list, local(2, 4)])).is_err());
        let mut typo = remote(0, 2, 1);
        typo["remote"]["standbys"] = json!([worker_id(2)]);
        assert!(parse(manifest_json(vec![typo, local(2, 4)])).is_err());
    }

    #[test]
    fn v1_requires_a_local_final_stage() {
        let error = error_of(manifest_json(vec![local(0, 2), remote(2, 4, 1)]));
        assert!(error.contains("final stage must be local"), "{error}");
        assert!(error.contains("\"execution\": \"ring\""), "{error}");
    }

    #[test]
    fn v1_allows_one_remote_stage() {
        let value = manifest_json(vec![remote(0, 1, 1), remote(1, 2, 2), local(2, 4)]);
        let error = error_of(value);
        assert!(error.contains("at most one remote"), "{error}");
        assert!(error.contains("\"execution\": \"ring\""), "{error}");
    }

    #[test]
    fn v1_runs_only_the_final_stage_locally() {
        let error = error_of(manifest_json(vec![local(0, 2), local(2, 4)]));
        assert!(error.contains("only the final stage locally"), "{error}");
        assert!(error.contains("\"execution\": \"ring\""), "{error}");
    }

    fn ring_json(stages: Vec<Value>) -> Value {
        let mut value = manifest_json(stages);
        value["execution"] = json!("ring");
        value
    }

    #[test]
    fn execution_defaults_to_pipeline_and_rejects_unknown_modes() {
        let manifest = validate(manifest_json(vec![local(0, 4)])).unwrap();
        assert_eq!(manifest.execution, Execution::Pipeline);
        assert!(!manifest.is_ring());
        let mut explicit = manifest_json(vec![remote(0, 2, 1), local(2, 4)]);
        explicit["execution"] = json!("pipeline");
        assert_eq!(validate(explicit).unwrap().execution, Execution::Pipeline);
        let mut unknown = manifest_json(vec![local(0, 4)]);
        unknown["execution"] = json!("p2p_ring");
        assert!(parse(unknown).is_err());
    }

    #[test]
    fn pipeline_rules_are_unchanged_by_the_ring_mode() {
        // Shapes only a ring accepts stay refused without the opt-in.
        let chained = manifest_json(vec![local(0, 1), remote(1, 3, 1), remote(3, 4, 2)]);
        assert!(error_of(chained).contains("final stage must be local"));
        let remote_final = manifest_json(vec![local(0, 2), remote(2, 4, 1)]);
        assert!(validate(remote_final).is_err());
    }

    #[test]
    fn ring_with_local_stage_0_and_remote_workers_is_valid() {
        let manifest = validate(ring_json(vec![
            local(0, 1),
            remote(1, 3, 1),
            remote(3, 4, 2),
        ]))
        .unwrap();
        assert!(manifest.is_ring());
        let workers: Vec<String> = manifest
            .remote_workers()
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(workers, [worker_id(1), worker_id(2)]);
        // A single remote worker is both the entry and the terminal.
        assert!(validate(ring_json(vec![local(0, 2), remote(2, 4, 1)])).is_ok());
    }

    #[test]
    fn ring_stage_0_must_be_local() {
        let error = error_of(ring_json(vec![remote(0, 1, 1), remote(1, 4, 2)]));
        assert!(error.contains("local stage 0"), "{error}");
    }

    #[test]
    fn ring_needs_a_remote_stage() {
        let error = error_of(ring_json(vec![local(0, 4)]));
        assert!(error.contains("at least one remote stage"), "{error}");
    }

    #[test]
    fn ring_runs_only_stage_0_locally() {
        let middle = ring_json(vec![local(0, 1), local(1, 3), remote(3, 4, 1)]);
        assert!(error_of(middle).contains("only stage 0 locally"));
        let last = ring_json(vec![local(0, 1), remote(1, 3, 1), local(3, 4)]);
        assert!(error_of(last).contains("only stage 0 locally"));
    }

    #[test]
    fn ring_workers_must_be_distinct_and_contiguous() {
        let duplicate = ring_json(vec![local(0, 1), remote(1, 3, 7), remote(3, 4, 7)]);
        assert!(error_of(duplicate).contains("more than one stage"));
        let gap = ring_json(vec![local(0, 1), remote(2, 4, 1)]);
        assert!(error_of(gap).contains("begin at layer 1"));
        let short = ring_json(vec![local(0, 1), remote(1, 3, 1)]);
        assert!(error_of(short).contains("4 layers"));
    }

    #[test]
    fn ring_refuses_standby_workers() {
        let stage = with_standby(remote(1, 4, 1), &[2]);
        let error = error_of(ring_json(vec![local(0, 1), stage]));
        assert!(error.contains("does not support standby"), "{error}");
    }

    #[test]
    fn ring_keeps_the_common_rules() {
        let mut bad_hash = ring_json(vec![local(0, 1), remote(1, 4, 1)]);
        bad_hash["model_sha256"] = json!("abc");
        assert!(error_of(bad_hash).contains("model_sha256"));
        let mut context = ring_json(vec![local(0, 1), remote(1, 4, 1)]);
        context["context"] = json!(MIN_CONTEXT - 1);
        assert!(error_of(context).contains("context"));
        let mut template = ring_json(vec![local(0, 1), remote(1, 4, 1)]);
        template.as_object_mut().unwrap().remove("chat_template");
        assert!(parse(template).is_err());
        let mut budget = local(0, 1);
        budget["local"]["budget_bytes"] = json!(0);
        assert!(error_of(ring_json(vec![budget, remote(1, 4, 1)])).contains("budget_bytes"));
    }

    #[test]
    fn load_names_the_file_and_validates() {
        let dir = std::env::temp_dir().join(format!("shard-manifest-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let good = dir.join("good.json");
        std::fs::write(&good, manifest_json(vec![local(0, 4)]).to_string()).unwrap();
        assert_eq!(ShardManifest::load(&good).unwrap().stages.len(), 1);
        let bad = dir.join("bad.json");
        std::fs::write(&bad, manifest_json(vec![local(0, 3)]).to_string()).unwrap();
        let error = format!("{:#}", ShardManifest::load(&bad).unwrap_err());
        assert!(
            error.contains("bad.json") && error.contains("4 layers"),
            "{error}"
        );
        assert!(ShardManifest::load(&dir.join("missing.json")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sha256_file_matches_known_digest() {
        let path = std::env::temp_dir().join(format!("shard-hash-{}", uuid::Uuid::new_v4()));
        std::fs::write(&path, b"abc").unwrap();
        let digest = sha256_file(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert!(same_sha256(&digest, &digest.to_uppercase()));
    }
}
