//! Bounded, lossless import of native candidate diagnostics.
//!
//! Importing a diagnostic preserves its raw report and every sample, but never
//! promotes the producer's `passed` state to qualified placement evidence.

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const IMPORT_SCHEMA_VERSION: u32 = 1;
pub const SOURCE_SCHEMA_VERSION: u64 = 1;
pub const SOURCE_KIND: &str = "native_candidate_diagnostic";
pub const MAX_REPORT_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_SAMPLES_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_SAMPLE_LINE_BYTES: usize = 1024 * 1024;
pub const MAX_SAMPLE_COUNT: usize = 65_536;

/// Retained, lossless source data with explicit evidence gaps. This is not an
/// `EvidenceRequest` and cannot be passed to the qualified placement planner.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeDiagnosticImport {
    pub schema_version: u32,
    pub kind: String,
    pub source_kind: String,
    pub source_schema_version: u64,
    pub profile_id: String,
    pub outcome: String,
    pub eligible_for_placement: bool,
    pub report_sha256: String,
    pub samples_sha256: String,
    /// Full source report, including request, all repetitions, counters and outcome.
    pub report: Value,
    /// Every nonempty JSONL record in original order, including failed checks.
    pub samples: Vec<Value>,
    /// The current diagnostic producer lacks these independent admission inputs.
    pub qualification_exclusions: Vec<String>,
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn required_string<'a>(value: &'a Value, key: &str) -> anyhow::Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .with_context(|| format!("native diagnostic has no valid {key}"))
}

/// Import exact producer v1 bytes. Hashes cover the bytes as supplied, including
/// whitespace and line endings; records are never dropped to salvage a trial.
pub fn import_native_candidate_diagnostic(
    report_bytes: &[u8],
    samples_bytes: &[u8],
) -> anyhow::Result<NativeDiagnosticImport> {
    ensure!(
        !report_bytes.is_empty() && report_bytes.len() <= MAX_REPORT_BYTES,
        "native diagnostic report is empty or exceeds 8 MiB"
    );
    ensure!(
        samples_bytes.len() <= MAX_SAMPLES_BYTES,
        "native diagnostic samples exceed 64 MiB"
    );
    let report: Value = serde_json::from_slice(report_bytes).context("invalid report JSON")?;
    ensure!(
        report.is_object(),
        "native diagnostic report must be an object"
    );
    ensure!(
        report.get("kind").and_then(Value::as_str) == Some(SOURCE_KIND)
            && report.get("schema_version").and_then(Value::as_u64) == Some(SOURCE_SCHEMA_VERSION),
        "unsupported native diagnostic kind or schema version"
    );
    ensure!(
        report
            .get("eligible_for_placement")
            .and_then(Value::as_bool)
            == Some(false),
        "native diagnostic eligibility flag must remain false"
    );
    let profile_id = required_string(&report, "profile_id")?.to_owned();
    let outcome = required_string(&report, "status")?.to_owned();
    ensure!(
        matches!(
            outcome.as_str(),
            "running"
                | "passed"
                | "failed"
                | "cancelled"
                | "timed_out"
                | "limit_exceeded"
                | "insufficient_duration"
                | "cleanup_failed"
        ),
        "unsupported native diagnostic outcome"
    );
    ensure!(
        report.get("sessions").is_some_and(Value::is_array),
        "native diagnostic sessions must be retained as an array"
    );

    let mut samples = Vec::new();
    for (index, line) in samples_bytes.split(|byte| *byte == b'\n').enumerate() {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        ensure!(
            line.len() <= MAX_SAMPLE_LINE_BYTES,
            "native diagnostic sample line {} exceeds 1 MiB",
            index + 1
        );
        ensure!(
            samples.len() < MAX_SAMPLE_COUNT,
            "native diagnostic has more than 65536 samples"
        );
        let sample: Value = serde_json::from_slice(line)
            .with_context(|| format!("invalid native diagnostic sample line {}", index + 1))?;
        ensure!(
            sample.is_object(),
            "sample line {} must be an object",
            index + 1
        );
        samples.push(sample);
    }

    Ok(NativeDiagnosticImport {
        schema_version: IMPORT_SCHEMA_VERSION,
        kind: "native_candidate_diagnostic_import".into(),
        source_kind: SOURCE_KIND.into(),
        source_schema_version: SOURCE_SCHEMA_VERSION,
        profile_id,
        outcome,
        eligible_for_placement: false,
        report_sha256: sha256(report_bytes),
        samples_sha256: sha256(samples_bytes),
        report,
        samples,
        qualification_exclusions: vec![
            "diagnostic_is_not_qualified_evidence".into(),
            "effective_backend_not_independently_attested".into(),
            "whole_physical_pool_load_and_inference_peaks_missing".into(),
            "sensor_backed_power_and_thermal_conditions_missing".into(),
            "predeclared_m5_trial_policy_missing".into(),
            "independent_reference_validation_record_missing".into(),
            "current_capacity_and_reservations_missing".into(),
            "directed_transfer_evidence_missing".into(),
        ],
    })
}
