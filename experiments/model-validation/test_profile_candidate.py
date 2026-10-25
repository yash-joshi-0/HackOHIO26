"""Process-level checks; fake tensors/timings are test fixtures, not hardware evidence."""

import hashlib
import json
from collections.abc import Callable
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any
import unittest
from unittest.mock import patch

import profile_candidate as profiler


FAKE_RUNNER = r'''#!/usr/bin/env python3
import json, os, signal, struct, sys, time
from pathlib import Path
root = Path(sys.argv[1]).parent
assert sys.argv[7]  # The real native runner rejects an empty tokenization prompt.
(root / "pid").write_text(str(os.getpid()))
config = json.loads((root / "config.json").read_text())
mode = config["mode"]
if mode == "ignore_term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
(root / "ready").write_text("1")
if mode in ("load_hang", "ignore_term"):
    time.sleep(60)
if mode == "stderr_flood":
    sys.stderr.write("x" * 100000)
    sys.stderr.flush()
if mode == "huge_header":
    sys.stdout.write(" " * 70000 + "{}\n")
    sys.stdout.flush()
    time.sleep(60)
begin, end = int(sys.argv[2]), int(sys.argv[3])
embd, layers, vocab = config["embd"], config["layers"], config["vocab"]
info = dict(protocol=1, begin=begin, end=end, layers=layers, embd=embd, vocab=vocab,
            source_sha256=config["source"], allocated_bytes=10, max_rss_bytes=20, vm=None)
if mode == "wrong_source":
    info["source_sha256"] = "f" * 64
print(json.dumps(info), flush=True)
if mode == "write_hang":
    time.sleep(60)
position = step = 0
while True:
    header = sys.stdin.buffer.read(20)
    if not header:
        sys.exit(7 if mode == "exit_failure" else 0)
    magic, op, actual_step, actual_pos, n = struct.unpack("!IIIII", header)
    assert (magic, op, actual_step, actual_pos) == (0x4d534831, 1 if begin == 0 else 2, step, position)
    size = n * 4 * (embd if begin else 1)
    data = sys.stdin.buffer.read(size)
    assert len(data) == size
    if mode == "step_hang":
        time.sleep(60)
    count = n * embd if end < layers else vocab
    value = float(position + n)
    if mode == "incorrect":
        value += 10
    if mode == "nonfinite":
        value = float("nan")
    result = dict(step=step, n_floats=count, compute_ms=0.01,
                  observed_peak_buffer_bytes=12, max_rss_bytes=22, vm=None)
    if mode == "wrong_step":
        result["step"] += 1
    if mode == "wrong_shape":
        result["n_floats"] += 1
    if mode == "missing_memory":
        result.pop("max_rss_bytes")
    print(json.dumps(result), flush=True)
    payload = struct.pack("<" + "f" * count, *([value] * count))
    if mode == "truncated":
        sys.stdout.buffer.write(payload[:2])
        sys.stdout.flush()
        sys.exit(0)
    sys.stdout.buffer.write(payload)
    sys.stdout.flush()
    if mode == "mutate_artifact":
        with open(sys.argv[1], "ab") as artifact:
            artifact.write(b"changed")
    step += 1
    position += n
'''


def pinned(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@unittest.skipUnless(os.name == "posix", "POSIX native-process profiler")
class CandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def fixture(self, mode: str = "ok", begin: int = 1, end: int = 2, embd: int = 2) -> dict[str, Any]:
        artifact = self.root / "model.gguf"
        artifact.write_bytes(b"test-only-artifact")
        runner = self.root / "runner"
        runner.write_text(FAKE_RUNNER)
        runner.chmod(0o700)
        model = {"source_sha256": "a" * 64, "layers": 3, "embd": embd, "vocab": 5}
        (self.root / "config.json").write_text(json.dumps({
            "mode": mode, "source": model["source_sha256"], **{k: model[k] for k in ("layers", "embd", "vocab")}}))
        steps, position = [], 0
        for index, n in enumerate((2, 1, 1)):
            payload = self.root / f"input-{index}.bin"
            expected = self.root / f"expected-{index}.bin"
            payload.write_bytes(struct.pack("<f", 1) * (n * embd) if begin else struct.pack("!I", 1) * n)
            count = n * embd if end < model["layers"] else model["vocab"]
            position += n
            expected.write_bytes(struct.pack("<f", position) * count)
            steps.append({"phase": "prefill" if index == 0 else "decode", "n": n,
                          "input": pinned(payload), "expected": pinned(expected)})
        fixture = {"schema_version": 1, "model": model, "range": {"begin": begin, "end": end},
                   "context_tokens": 32, "reference": {"runtime_id": "test-reference",
                   "runner_sha256": pinned(runner)["sha256"], "backend": "fake-test-only"},
                   "atol": 0.001, "rtol": 0, "steps": steps}
        fixture_path = self.root / "fixture.json"
        fixture_path.write_text(json.dumps(fixture))
        self.request = {"schema_version": 1, "device_id": "test-process", "model": model,
                        "range": fixture["range"], "context_tokens": 32, "artifact": pinned(artifact),
                        "runner": {**pinned(runner), "runtime_id": "fake-test-only", "gpu_layers": 0, "dependencies": []},
                        "fixture": pinned(fixture_path), "repetitions": 2, "min_session_seconds": 0,
                        "conditions": {"power": None, "thermal": None},
                        "limits": {"total_seconds": 30, "load_seconds": 5, "step_seconds": 5,
                        "max_wire_bytes": 8 << 20, "max_fixture_bytes": 4 << 20, "max_log_bytes": 4096,
                        "max_observed_buffer_bytes": 1000, "max_reported_rss_bytes": 1000}}
        self.request_path = self.root / "request.json"
        self.write_request()
        return self.request

    def write_request(self) -> None:
        self.request_path.write_text(json.dumps(self.request))

    def run_profile(self, cancel: Callable[[], bool] = lambda: False) -> dict[str, Any]:
        self.write_request()
        report = profiler.profile(self.request_path, self.root / "out", cancel)
        self.assertEqual(report, json.loads((self.root / "out/report.json").read_text()))
        if (self.root / "pid").exists():
            pid = int((self.root / "pid").read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        return report

    def test_ordered_state_reset_and_separate_phase_observations(self) -> None:
        self.fixture()
        report = self.run_profile()
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["completed_steps"], 6)
        self.assertTrue(report["cleanup_confirmed"])
        self.assertEqual(report["counters"]["children_started"], 2)
        self.assertEqual(report["counters"]["children_reaped"], 2)
        self.assertEqual(report["summary"]["decode"]["compute"]["samples"], 4)
        self.assertEqual(len(report["sessions"]), 2)
        self.assertFalse(report["eligible_for_placement"])
        rows = [json.loads(line) for line in (self.root / "out/samples.jsonl").read_text().splitlines()]
        self.assertEqual([row["position"] for row in rows], [0, 2, 3, 0, 2, 3])

    def test_terminal_logits_and_token_input(self) -> None:
        self.fixture(begin=0, end=3)
        report = self.run_profile()
        self.assertEqual(report["status"], "passed", report)

    def test_checksum_mismatch_prevents_launch(self) -> None:
        self.fixture()
        self.request["artifact"]["sha256"] = "b" * 64
        report = self.run_profile()
        self.assertEqual(report["status"], "failed")
        self.assertIn("checksum", report["error"])
        self.assertFalse((self.root / "pid").exists())

    def test_cancel_before_loading(self) -> None:
        self.fixture()
        self.assertEqual(self.run_profile(lambda: True)["status"], "cancelled")
        self.assertFalse((self.root / "pid").exists())

    def test_load_timeout(self) -> None:
        self.fixture("load_hang")
        self.request["limits"]["load_seconds"] = 0.15
        started = time.monotonic()
        self.assertEqual(self.run_profile()["status"], "timed_out")
        self.assertLess(time.monotonic() - started, 2)

    def test_total_deadline_is_not_reset_by_steps(self) -> None:
        self.fixture("step_hang")
        self.request["limits"]["total_seconds"] = 0.2
        self.assertEqual(self.run_profile()["status"], "timed_out")

    def test_blocked_write_is_bounded(self) -> None:
        self.fixture("write_hang", embd=32768)
        self.request["limits"]["step_seconds"] = 0.15
        started = time.monotonic()
        self.assertEqual(self.run_profile()["status"], "timed_out")
        self.assertLess(time.monotonic() - started, 2)

    def test_cancel_kills_uncooperative_runner(self) -> None:
        self.fixture("ignore_term")
        cancelled_at = []

        def cancel() -> bool:
            if (self.root / "ready").exists():
                if not cancelled_at:
                    cancelled_at.append(time.monotonic())
                return True
            return False

        report = self.run_profile(cancel)
        self.assertEqual(report["status"], "cancelled")
        self.assertLess(time.monotonic() - cancelled_at[0], 1.5)

    def test_incorrect_output_retains_failed_sample(self) -> None:
        self.fixture("incorrect")
        report = self.run_profile()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["completed_steps"], 0)
        sample = json.loads((self.root / "out/samples.jsonl").read_text())
        self.assertFalse(sample["correctness"]["passed"])

    def test_reported_memory_limit(self) -> None:
        self.fixture()
        self.request["limits"]["max_reported_rss_bytes"] = 21
        report = self.run_profile()
        self.assertEqual(report["status"], "limit_exceeded")
        self.assertEqual(report["completed_steps"], 0)

    def test_native_wire_limit(self) -> None:
        self.fixture()
        self.request["limits"]["max_wire_bytes"] = 32
        self.assertEqual(self.run_profile()["status"], "limit_exceeded")

    def test_stderr_limit(self) -> None:
        self.fixture("stderr_flood")
        self.assertEqual(self.run_profile()["status"], "limit_exceeded")
        self.assertLessEqual((self.root / "out/runner.log").stat().st_size, 4096)

    def test_malformed_runner_responses_fail_closed(self) -> None:
        for mode in ("wrong_source", "wrong_step", "wrong_shape", "huge_header", "nonfinite",
                     "truncated", "missing_memory", "exit_failure", "mutate_artifact"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                self.root = Path(directory)
                self.fixture(mode)
                report = self.run_profile()
                self.assertEqual(report["status"], "failed", report)

    def test_short_sessions_do_not_claim_sustained_evidence(self) -> None:
        self.fixture()
        self.request["min_session_seconds"] = 60
        self.assertEqual(self.run_profile()["status"], "insufficient_duration")

    def test_existing_evidence_is_never_overwritten(self) -> None:
        self.fixture()
        self.run_profile()
        with self.assertRaises(FileExistsError):
            profiler.profile(self.request_path, self.root / "out")

    def test_cleanup_failure_cannot_erase_the_report(self) -> None:
        self.fixture()
        with patch.object(profiler.Native, "close", side_effect=PermissionError("test cleanup failure")):
            report = self.run_profile()
        self.assertEqual(report["status"], "cleanup_failed")
        self.assertFalse(report["cleanup_confirmed"])
        self.assertIn("test cleanup failure", report["cleanup_error"])

    def test_invalid_state_fixture_prevents_launch(self) -> None:
        self.fixture()
        fixture_path = self.root / "fixture.json"
        fixture = json.loads(fixture_path.read_text())
        fixture["steps"][1]["n"] = 2
        fixture_path.write_text(json.dumps(fixture))
        self.request["fixture"] = pinned(fixture_path)
        report = self.run_profile()
        self.assertEqual(report["status"], "failed")
        self.assertFalse((self.root / "pid").exists())

    def test_cli_requires_idle_confirmation(self) -> None:
        self.fixture()
        result = subprocess.run([sys.executable, profiler.__file__, str(self.request_path),
                                 "--output-dir", str(self.root / "out")],
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / "out").exists())

    def test_cli_cancel_file(self) -> None:
        self.fixture()
        cancel_file = self.root / "cancel"
        cancel_file.touch()
        result = subprocess.run([sys.executable, profiler.__file__, str(self.request_path),
                                 "--output-dir", str(self.root / "out"), "--confirm-idle",
                                 "--cancel-file", str(cancel_file)], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "cancelled")
        self.assertFalse((self.root / "pid").exists())

    def test_cli_signal_cancellation_retains_report_and_reaps_child(self) -> None:
        self.fixture("load_hang")
        process = subprocess.Popen([sys.executable, profiler.__file__, str(self.request_path),
                                    "--output-dir", str(self.root / "out"), "--confirm-idle"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 3
            while not (self.root / "pid").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue((self.root / "pid").exists())
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=3)
            self.assertEqual(process.returncode, 1, stderr)
            self.assertEqual(json.loads(stdout)["status"], "cancelled")
            report = json.loads((self.root / "out/report.json").read_text())
            self.assertEqual(report["status"], "cancelled")
            with self.assertRaises(ProcessLookupError):
                os.kill(int((self.root / "pid").read_text()), 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            process.stderr.close()


if __name__ == "__main__":
    unittest.main()
