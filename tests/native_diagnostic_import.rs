use llm_cluster::native_diagnostic::{
    IMPORT_SCHEMA_VERSION, SOURCE_KIND, import_native_candidate_diagnostic,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::process::Command;

fn report(status: &str, eligible: bool) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "schema_version": 1,
        "kind": SOURCE_KIND,
        "profile_id": "profile-123",
        "status": status,
        "eligible_for_placement": eligible,
        "phase": "complete",
        "completed_steps": 2,
        "request": {"range": {"begin": 4, "end": 8}},
        "sessions": [
            {"repetition": 0, "load_ms": 123.0, "execution_seconds": 10.0},
            {"repetition": 1, "load_ms": 130.0, "execution_seconds": 11.0}
        ]
    }))
    .unwrap()
}

fn sample(repetition: u64, passed: bool) -> Value {
    json!({
        "repetition": repetition,
        "step": repetition,
        "position": repetition,
        "n": 1,
        "phase": "decode",
        "compute_ms": 4.5,
        "service_ms": 5.0,
        "correctness": {"passed": passed, "max_abs_error": 0.0},
        "observed_peak_buffer_bytes": 1024,
        "max_rss_bytes": 4096,
        "vm": null
    })
}

#[test]
fn passing_report_remains_diagnostic_and_preserves_every_source_record() {
    let report = report("passed", false);
    let samples = format!("{}\r\n{}\n", sample(0, true), sample(1, false));
    let imported = import_native_candidate_diagnostic(&report, samples.as_bytes()).unwrap();

    assert_eq!(imported.schema_version, IMPORT_SCHEMA_VERSION);
    assert_eq!(imported.source_kind, SOURCE_KIND);
    assert_eq!(imported.profile_id, "profile-123");
    assert_eq!(imported.outcome, "passed");
    assert!(!imported.eligible_for_placement);
    assert_eq!(imported.report_sha256, hex::encode(Sha256::digest(&report)));
    assert_eq!(
        imported.samples_sha256,
        hex::encode(Sha256::digest(samples.as_bytes()))
    );
    assert_eq!(imported.report["sessions"].as_array().unwrap().len(), 2);
    assert_eq!(imported.samples, vec![sample(0, true), sample(1, false)]);
    assert!(
        imported
            .qualification_exclusions
            .contains(&"effective_backend_not_independently_attested".into())
    );
    let serialized = serde_json::to_value(imported).unwrap();
    assert!(serialized.get("qualified_range").is_none());
    assert_eq!(serialized["eligible_for_placement"], false);
}

#[test]
fn unsuccessful_and_incomplete_outcomes_are_preserved_not_dropped() {
    for outcome in ["failed", "cancelled", "timed_out", "insufficient_duration"] {
        let imported = import_native_candidate_diagnostic(&report(outcome, false), b"").unwrap();
        assert_eq!(imported.outcome, outcome);
        assert_eq!(imported.report["status"], outcome);
        assert!(imported.samples.is_empty());
        assert!(!imported.eligible_for_placement);
    }
}

#[test]
fn unsupported_or_promoted_source_reports_are_rejected() {
    let mut wrong_kind: Value = serde_json::from_slice(&report("passed", false)).unwrap();
    wrong_kind["kind"] = json!("other");
    assert!(
        import_native_candidate_diagnostic(&serde_json::to_vec(&wrong_kind).unwrap(), b"").is_err()
    );

    let mut wrong_version: Value = serde_json::from_slice(&report("passed", false)).unwrap();
    wrong_version["schema_version"] = json!(2);
    assert!(
        import_native_candidate_diagnostic(&serde_json::to_vec(&wrong_version).unwrap(), b"")
            .is_err()
    );

    assert!(import_native_candidate_diagnostic(&report("passed", true), b"").is_err());
    assert!(import_native_candidate_diagnostic(&report("unknown", false), b"").is_err());
}

#[test]
fn malformed_or_oversized_sample_stream_is_rejected_without_partial_import() {
    let source = report("running", false);
    assert!(import_native_candidate_diagnostic(&source, b"{broken}\n").is_err());
    assert!(
        import_native_candidate_diagnostic(
            &source,
            &vec![b'x'; llm_cluster::native_diagnostic::MAX_SAMPLE_LINE_BYTES + 1]
        )
        .is_err()
    );
    assert!(
        import_native_candidate_diagnostic(
            &source,
            &vec![b' '; llm_cluster::native_diagnostic::MAX_SAMPLES_BYTES + 1]
        )
        .is_err()
    );
}

#[test]
fn cli_imports_files_and_emits_diagnostic_only_json() {
    let dir = tempfile::tempdir().unwrap();
    let report_path = dir.path().join("report.json");
    let samples_path = dir.path().join("samples.jsonl");
    std::fs::write(&report_path, report("passed", false)).unwrap();
    std::fs::write(&samples_path, format!("{}\n", sample(0, true))).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_llm-cluster"))
        .args(["import-native-diagnostic", "--report"])
        .arg(&report_path)
        .arg("--samples")
        .arg(&samples_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let imported: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(imported["profile_id"], "profile-123");
    assert_eq!(imported["outcome"], "passed");
    assert_eq!(imported["eligible_for_placement"], false);
    assert_eq!(imported["samples"].as_array().unwrap().len(), 1);
    assert!(
        imported["qualification_exclusions"]
            .as_array()
            .unwrap()
            .len()
            >= 8
    );
}
