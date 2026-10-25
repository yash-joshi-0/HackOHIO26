//! Drives the `fake_shard_runner` example by hand with the shard wire codec, as the leader's
//! pipeline will drive real shard runners.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use llm_cluster::shard::wire::{self, StageInput, StageMetadata, StepReply};
use tokio::io::BufReader;
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::time::timeout;

mod common;
use common::example_bin;

/// Context window passed to every stage.
const CONTEXT: u32 = 128;
/// Runtime buffer budget passed to every stage.
const BUDGET_BYTES: &str = "1048576";
/// Prompt used by the chain tests.
const PROMPT: &str = "hi";
/// Upper bound on decode steps before a chain test gives up.
const MAX_DECODE_STEPS: usize = 64;
/// Bound on any single interaction with a stage process.
const IO_TIMEOUT: Duration = Duration::from_secs(10);

/// A spawned stage with its pipes and startup metadata.
struct StageProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    meta: StageMetadata,
}

fn model_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("fake-shard-model.json")
}

fn fixture_reply() -> Vec<u8> {
    let text = std::fs::read_to_string(model_fixture()).expect("read model fixture");
    let model: serde_json::Value = serde_json::from_str(&text).expect("parse model fixture");
    model["reply"]
        .as_str()
        .expect("fixture reply")
        .as_bytes()
        .to_vec()
}

async fn spawn_stage(begin: u32, end: u32, envs: &[(&str, &str)]) -> StageProcess {
    let mut child = Command::new(example_bin("fake_shard_runner"))
        .arg(model_fixture())
        .args([
            begin.to_string(),
            end.to_string(),
            CONTEXT.to_string(),
            "0".to_string(),
            BUDGET_BYTES.to_string(),
            PROMPT.to_string(),
            "0".to_string(),
        ])
        .envs(envs.iter().copied())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn fake_shard_runner");
    let stdin = child.stdin.take().expect("stage stdin");
    let mut stdout = BufReader::new(child.stdout.take().expect("stage stdout"));
    let meta = timeout(IO_TIMEOUT, wire::read_metadata(&mut stdout))
        .await
        .expect("metadata timed out")
        .expect("read metadata");
    assert_eq!((meta.begin, meta.end), (begin, end));
    StageProcess {
        child,
        stdin,
        stdout,
        meta,
    }
}

/// Sends one batch through every stage in order and returns the final stage's reply.
async fn run_step(
    stages: &mut [StageProcess],
    step: u32,
    position: u32,
    tokens: &[u32],
) -> StepReply {
    let mut rows = Vec::new();
    for (index, stage) in stages.iter_mut().enumerate() {
        let embd = stage.meta.embd;
        let input = if index == 0 {
            StageInput::Tokens(tokens)
        } else {
            StageInput::Hidden { rows: &rows, embd }
        };
        wire::write_request(&mut stage.stdin, step, position, &input)
            .await
            .expect("write request");
        let max_floats = if stage.meta.is_final() {
            0
        } else {
            tokens.len() * embd as usize
        };
        let output = timeout(IO_TIMEOUT, wire::read_step(&mut stage.stdout, max_floats))
            .await
            .expect("step timed out")
            .expect("read step");
        assert_eq!(output.reply.step, step);
        if stage.meta.is_final() {
            return output.reply;
        }
        assert_eq!(output.reply.token, -1);
        assert_eq!(output.floats.len(), max_floats);
        rows = output.floats;
    }
    panic!("the chain has no final stage");
}

/// Prefills the prompt in two batches, then decodes until end-of-generation.
async fn run_chain(ranges: &[(u32, u32)]) -> Vec<u8> {
    let mut stages = Vec::new();
    for &(begin, end) in ranges {
        stages.push(spawn_stage(begin, end, &[]).await);
    }
    let tokens = stages.last().expect("final stage").meta.tokens.clone();
    assert_eq!(tokens.len(), PROMPT.len() + 1);
    let split = tokens.len() - 1;

    let early = run_step(&mut stages, 0, 0, &tokens[..split]).await;
    assert_eq!(
        (early.token, early.eog, early.piece_hex.as_str()),
        (0, false, "")
    );
    let mut position = split as u32;
    let mut reply = run_step(&mut stages, 1, position, &tokens[split..]).await;
    position += 1;

    let mut text = Vec::new();
    for step in 2.. {
        assert!(step < MAX_DECODE_STEPS as u32, "no end-of-generation");
        text.extend(reply.piece_bytes().expect("piece bytes"));
        if reply.eog {
            break;
        }
        let token = u32::try_from(reply.token).expect("sampled token");
        reply = run_step(&mut stages, step, position, &[token]).await;
        position += 1;
    }
    for stage in stages {
        finish_ok(stage).await;
    }
    text
}

/// Closes the stage's stdin and expects a clean exit.
async fn finish_ok(stage: StageProcess) {
    drop(stage.stdin);
    let output = timeout(IO_TIMEOUT, stage.child.wait_with_output())
        .await
        .expect("exit timed out")
        .expect("wait for stage");
    assert!(output.status.success(), "stage failed: {output:?}");
}

/// Closes the stage's stdin, expects a failing exit and returns its stderr.
async fn finish_failed(stage: StageProcess) -> String {
    drop(stage.stdin);
    let output = timeout(IO_TIMEOUT, stage.child.wait_with_output())
        .await
        .expect("exit timed out")
        .expect("wait for stage");
    assert!(!output.status.success(), "stage should have failed");
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[tokio::test]
async fn prefix_and_final_stages_reproduce_the_fixture_reply() {
    assert_eq!(run_chain(&[(0, 2), (2, 4)]).await, fixture_reply());
}

#[tokio::test]
async fn middle_stage_chains_between_prefix_and_final() {
    assert_eq!(run_chain(&[(0, 1), (1, 3), (3, 4)]).await, fixture_reply());
}

#[tokio::test]
async fn whole_model_stage_generates_alone() {
    assert_eq!(run_chain(&[(0, 4)]).await, fixture_reply());
}

#[tokio::test]
async fn reports_prompt_tokens_in_metadata() {
    let stage = spawn_stage(0, 2, &[]).await;
    let expected: Vec<u32> = std::iter::once(0)
        .chain(PROMPT.bytes().map(|byte| u32::from(byte) + 1))
        .collect();
    assert_eq!(stage.meta.tokens, expected);
    assert!(!stage.meta.is_final());
    assert_eq!(stage.meta.extra.get("vm"), Some(&serde_json::Value::Null));
    finish_ok(stage).await;
}

#[tokio::test]
async fn hidden_row_with_wrong_offset_breaks_the_chain() {
    let mut stage = spawn_stage(2, 4, &[]).await;
    // The BOS row as a stage ending at layer 1 (not 2) would produce it: 0 + 1.
    let rows = vec![1.0f32; stage.meta.embd as usize];
    let input = StageInput::Hidden {
        rows: &rows,
        embd: stage.meta.embd,
    };
    wire::write_request(&mut stage.stdin, 0, 0, &input)
        .await
        .expect("write request");
    let stderr = finish_failed(stage).await;
    assert!(stderr.contains("broken stage chain"), "{stderr}");
}

#[tokio::test]
async fn out_of_order_step_closes_the_session() {
    let mut stage = spawn_stage(0, 2, &[]).await;
    wire::write_request(&mut stage.stdin, 1, 0, &StageInput::Tokens(&[0]))
        .await
        .expect("write request");
    let stderr = finish_failed(stage).await;
    assert!(stderr.contains("out-of-order"), "{stderr}");
}

#[tokio::test]
async fn wrong_fed_back_token_is_refused() {
    let mut stages = vec![spawn_stage(0, 4, &[]).await];
    let tokens = stages[0].meta.tokens.clone();
    let reply = run_step(&mut stages, 0, 0, &tokens).await;
    let wrong = u32::try_from(reply.token).expect("sampled token") + 1;
    let mut stage = stages.pop().expect("one stage");
    let position = tokens.len() as u32;
    wire::write_request(&mut stage.stdin, 1, position, &StageInput::Tokens(&[wrong]))
        .await
        .expect("write request");
    let stderr = finish_failed(stage).await;
    assert!(stderr.contains("fed back wrong token"), "{stderr}");
}

#[tokio::test]
async fn fail_at_step_env_fails_that_step() {
    let mut stage = spawn_stage(0, 2, &[("FAKE_SHARD_FAIL_AT_STEP", "1")]).await;
    let embd = stage.meta.embd as usize;
    wire::write_request(&mut stage.stdin, 0, 0, &StageInput::Tokens(&[0]))
        .await
        .expect("write request");
    let output = timeout(IO_TIMEOUT, wire::read_step(&mut stage.stdout, embd))
        .await
        .expect("step timed out")
        .expect("step 0 succeeds");
    // BOS (token 0) leaves a stage ending at layer 2 as rows of 0 + 2.
    assert_eq!(output.floats, vec![2.0; embd]);
    wire::write_request(&mut stage.stdin, 1, 1, &StageInput::Tokens(&[0]))
        .await
        .expect("write request");
    let stderr = finish_failed(stage).await;
    assert!(stderr.contains("FAKE_SHARD_FAIL_AT_STEP"), "{stderr}");
}

#[tokio::test]
async fn a_prefill_past_the_prompt_continues_the_reply() {
    let reply = fixture_reply();
    let mut stages = vec![spawn_stage(0, 4, &[]).await];
    let mut history = stages[0].meta.tokens.clone();
    // As a resume would: the prompt plus the first two reply tokens in one prefill batch.
    history.extend(reply[..2].iter().map(|&byte| u32::from(byte) + 1));
    let next = run_step(&mut stages, 0, 0, &history).await;
    assert_eq!(next.piece_bytes().expect("piece bytes"), [reply[2]]);
    finish_ok(stages.pop().expect("one stage")).await;

    // A wrong token past the prompt is refused even inside a prefill batch.
    let mut stage = spawn_stage(0, 4, &[]).await;
    let mut history = stage.meta.tokens.clone();
    history.push(u32::from(reply[0]) + 2);
    wire::write_request(&mut stage.stdin, 0, 0, &StageInput::Tokens(&history))
        .await
        .expect("write request");
    let stderr = finish_failed(stage).await;
    assert!(stderr.contains("fed back wrong token"), "{stderr}");
}

#[tokio::test]
async fn fail_once_file_limits_the_failure_to_the_first_runner() {
    let marker =
        std::env::temp_dir().join(format!("fake-shard-fail-once-{}", uuid::Uuid::new_v4()));
    let marker_text = marker.to_string_lossy().into_owned();
    let envs = [
        ("FAKE_SHARD_FAIL_AT_STEP", "0"),
        ("FAKE_SHARD_FAIL_ONCE_FILE", marker_text.as_str()),
    ];
    let mut first = spawn_stage(0, 4, &envs).await;
    let tokens = first.meta.tokens.clone();
    wire::write_request(&mut first.stdin, 0, 0, &StageInput::Tokens(&tokens))
        .await
        .expect("write request");
    let stderr = finish_failed(first).await;
    assert!(stderr.contains("FAKE_SHARD_FAIL_AT_STEP"), "{stderr}");
    assert!(marker.exists(), "the first runner leaves the marker");

    let mut second = vec![spawn_stage(0, 4, &envs).await];
    let reply = run_step(&mut second, 0, 0, &tokens).await;
    assert_eq!(
        reply.piece_bytes().expect("piece bytes"),
        fixture_reply()[..1]
    );
    finish_ok(second.pop().expect("one stage")).await;
    std::fs::remove_file(&marker).expect("remove marker");
}
