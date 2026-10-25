#!/usr/bin/env python3
"""Profile one local native MSH1 range against a pinned reference fixture (POSIX)."""

import argparse
import array
from collections.abc import Callable
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import statistics
import struct
import subprocess
import sys
import time
from typing import Any, BinaryIO, NoReturn
import uuid


MAX_JSON = 1 << 20
MAX_HEADER = 65536
MAX_TENSOR = 16 << 20
FileIdentity = tuple[int, int, int, int]
FixtureStep = tuple[dict[str, Any], int, bytes, bytes]


class Stopped(Exception):
    def __init__(self, outcome: str, message: str) -> None:
        super().__init__(message)
        self.outcome = outcome


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def number(value: Any, low: float, high: float) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def fields(value: Any, required: str) -> None:
    require(isinstance(value, dict) and set(value) == set(required.split()),
            "unexpected or missing fields: " + required)


def reject_constant(value: str) -> NoReturn:
    raise ValueError("nonfinite JSON constant: " + value)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), "request must be a regular file")
    with path.open("rb") as source:
        data = source.read(MAX_JSON + 1)
    require(len(data) <= MAX_JSON, "JSON exceeds 1 MiB")
    return json.loads(data, parse_constant=reject_constant)


def save(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class Control:
    def __init__(self, seconds: float, cancel: Callable[[], bool] = lambda: False) -> None:
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.cancel = cancel

    def check(self, deadline: float | None = None) -> None:
        if self.cancel():
            raise Stopped("cancelled", "profiling cancelled")
        if time.monotonic() >= min(self.deadline, deadline or self.deadline):
            raise Stopped("timed_out", "profiling deadline reached")


def identity(path: Path) -> FileIdentity:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def verified_file(spec: dict[str, Any], base: Path, control: Control,
                  pins: dict[Path, FileIdentity], max_bytes: int | None = None) -> tuple[Path, bytes | None]:
    fields(spec, "path sha256")
    require(sha(spec["sha256"]), "invalid file SHA-256")
    path = (base / spec["path"]).resolve()
    require(path.is_file(), "expected a regular file")
    before = identity(path)
    if max_bytes is not None:
        require(before[2] <= max_bytes, "fixture exceeds byte limit")
    digest = hashlib.sha256()
    chunks = [] if max_bytes is not None else None
    total = 0
    with path.open("rb") as source:
        while True:
            control.check()
            chunk = source.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None:
                require(total <= max_bytes, "fixture exceeds byte limit")
                chunks.append(chunk)
            digest.update(chunk)
    require(before == identity(path), "file changed while hashing")
    require(digest.hexdigest() == spec["sha256"], "file checksum mismatch: " + path.name)
    pins[path] = before
    return path, b"".join(chunks) if chunks is not None else None


def floats(data: bytes, control: Control | None = None) -> array.array:
    require(len(data) % 4 == 0, "invalid float payload")
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    for index, value in enumerate(values):
        if control and index % 16384 == 0:
            control.check()
        require(math.isfinite(value), "nonfinite tensor")
    return values


def prepare(request: dict[str, Any], base: Path, control: Control,
            pins: dict[Path, FileIdentity]) -> tuple[Path, Path, dict[str, Any], list[FixtureStep]]:
    fields(request, "schema_version device_id model range context_tokens artifact runner fixture repetitions min_session_seconds limits conditions")
    require(type(request["schema_version"]) is int and request["schema_version"] == 1, "unsupported request version")
    require(isinstance(request["device_id"], str) and request["device_id"], "device_id required")
    model = request["model"]
    fields(model, "source_sha256 layers embd vocab")
    require(sha(model["source_sha256"]), "invalid source hash")
    require(integer(model["layers"], 1, 1024) and integer(model["embd"], 1, 32768)
            and integer(model["vocab"], 1, MAX_TENSOR // 4), "invalid model dimensions")
    fields(request["range"], "begin end")
    begin, end = request["range"]["begin"], request["range"]["end"]
    require(integer(begin, 0, model["layers"] - 1) and integer(end, begin + 1, model["layers"]), "invalid range")
    require(integer(request["context_tokens"], 32, 4096), "invalid context")
    require(integer(request["repetitions"], 1, 64), "invalid repetitions")
    require(number(request["min_session_seconds"], 0, 3600), "invalid minimum session duration")
    fields(request["conditions"], "power thermal")
    for value in request["conditions"].values():
        require(value is None or isinstance(value, str), "conditions must be descriptions or null")
    runner = request["runner"]
    fields(runner, "path sha256 runtime_id gpu_layers dependencies")
    require(isinstance(runner["runtime_id"], str) and runner["runtime_id"], "runtime identity required")
    require(integer(runner["gpu_layers"], 0, 1024), "invalid GPU layer request")
    require(isinstance(runner["dependencies"], list) and len(runner["dependencies"]) <= 64, "invalid runtime dependencies")
    executable, _ = verified_file({k: runner[k] for k in ("path", "sha256")}, base, control, pins)
    require(os.access(executable, os.X_OK), "runner is not executable")
    for dependency in runner["dependencies"]:
        verified_file(dependency, base, control, pins)
    artifact, _ = verified_file(request["artifact"], base, control, pins)
    fixture_path, fixture_bytes = verified_file(request["fixture"], base, control, pins, MAX_JSON)
    fixture = json.loads(fixture_bytes, parse_constant=reject_constant)
    fields(fixture, "schema_version model range context_tokens reference atol rtol steps")
    require(type(fixture["schema_version"]) is int and fixture["schema_version"] == 1 and fixture["model"] == model
            and fixture["range"] == request["range"]
            and fixture["context_tokens"] == request["context_tokens"], "fixture workload mismatch")
    fields(fixture["reference"], "runtime_id runner_sha256 backend")
    require(sha(fixture["reference"]["runner_sha256"]) and
            all(isinstance(fixture["reference"][k], str) and fixture["reference"][k]
                for k in ("runtime_id", "backend")), "reference provenance required")
    require(number(fixture["atol"], 0, 1) and number(fixture["rtol"], 0, 1), "invalid comparison tolerances")
    require(isinstance(fixture["steps"], list) and 2 <= len(fixture["steps"]) <= 4096, "fixture needs prefill and decode steps")
    used = len(fixture_bytes)
    position, decoding, steps = 0, False, []
    for step in fixture["steps"]:
        control.check()
        fields(step, "phase n input expected")
        n = step["n"]
        require(integer(n, 1, 128), "invalid batch size")
        require(step["phase"] in ("prefill", "decode"), "invalid workload phase")
        require(not decoding or step["phase"] == "decode", "prefill after decode")
        require(step["phase"] != "decode" or (position > 0 and n == 1), "decode requires prior state and batch 1")
        decoding |= step["phase"] == "decode"
        require(position + n <= request["context_tokens"], "fixture exceeds context")
        blobs = []
        for name in ("input", "expected"):
            remaining = request["limits"]["max_fixture_bytes"] - used
            _, data = verified_file(step[name], fixture_path.parent, control, pins, min(MAX_TENSOR, remaining))
            used += len(data)
            blobs.append(data)
        payload, expected = blobs
        require(len(payload) == n * 4 * (model["embd"] if begin else 1), "wrong input tensor size")
        count = n * model["embd"] if end < model["layers"] else model["vocab"]
        require(len(expected) == count * 4, "wrong reference tensor size")
        floats(expected, control)
        if begin:
            floats(payload, control)
        else:
            require(all(t[0] < model["vocab"] for t in struct.iter_unpack("!I", payload)), "invalid token input")
        steps.append((step, position, payload, expected))
        position += n
    require(decoding, "fixture needs decode evidence")
    return executable, artifact, fixture, steps


def validate_limits(limits: dict[str, Any]) -> None:
    fields(limits, "total_seconds load_seconds step_seconds max_wire_bytes max_fixture_bytes max_log_bytes max_observed_buffer_bytes max_reported_rss_bytes")
    for key in ("total_seconds", "load_seconds", "step_seconds"):
        require(number(limits[key], 0.01, 3600), "invalid time bound: " + key)
    for key in ("max_wire_bytes", "max_fixture_bytes", "max_log_bytes",
                "max_observed_buffer_bytes", "max_reported_rss_bytes"):
        require(integer(limits[key], 1, 1 << 50), "invalid byte bound: " + key)
    require(limits["max_fixture_bytes"] <= 256 << 20, "fixture memory limit exceeds 256 MiB")


class Native:
    """Nonblocking pipes share absolute deadlines, cancellation and byte accounting."""

    def __init__(self, command: list[str], log: BinaryIO, control: Control,
                 limits: dict[str, Any], counters: dict[str, int]) -> None:
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, bufsize=0, start_new_session=True)
        self.control, self.limits, self.counters, self.log = control, limits, counters, log
        self.buffer = bytearray()
        self.stderr = self.process.stderr
        try:
            for stream in (self.process.stdin, self.process.stdout, self.stderr):
                os.set_blocking(stream.fileno(), False)
        except BaseException:
            self.close()
            raise

    def charge(self, name: str, size: int, bound: str) -> None:
        self.counters[name] += size
        if self.counters[name] > self.limits[bound]:
            raise Stopped("limit_exceeded", bound + " reached")

    def pump(self, deadline: float, writing: bool = False) -> bool:
        self.control.check(deadline)
        reads = [self.process.stdout] + ([self.stderr] if self.stderr else [])
        ready, writable, _ = select.select(reads, [self.process.stdin] if writing else [], [],
                                          min(0.05, max(0, deadline - time.monotonic())))
        if self.stderr in ready:
            data = os.read(self.stderr.fileno(), 65536)
            if data:
                room = max(0, self.limits["max_log_bytes"] - self.counters["log_bytes"])
                self.log.write(data[:room])
                self.charge("log_bytes", len(data), "max_log_bytes")
            else:
                self.stderr = None
        if self.process.stdout in ready:
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError("runner closed its output")
            self.charge("wire_bytes", len(data), "max_wire_bytes")
            self.buffer.extend(data)
            require(len(self.buffer) <= MAX_TENSOR + MAX_HEADER, "oversized runner response")
        return bool(writable)

    def line(self, deadline: float) -> dict[str, Any]:
        while b"\n" not in self.buffer:
            require(len(self.buffer) <= MAX_HEADER, "oversized runner metadata")
            self.pump(deadline)
        index = self.buffer.index(b"\n")
        require(index <= MAX_HEADER, "oversized runner metadata")
        data = bytes(self.buffer[:index])
        del self.buffer[:index + 1]
        return json.loads(data, parse_constant=reject_constant)

    def execute(self, index: int, position: int, n: int, payload: bytes,
                count: int, begin: int) -> tuple[dict[str, Any], bytes, float]:
        deadline = min(self.control.deadline, time.monotonic() + self.limits["step_seconds"])
        request = memoryview(struct.pack("!IIIII", 0x4D534831, 1 if begin == 0 else 2,
                                         index, position, n) + payload)
        started = time.monotonic()
        while request:
            if self.pump(deadline, writing=True):
                remaining = self.limits["max_wire_bytes"] - self.counters["wire_bytes"]
                if remaining <= 0:
                    raise Stopped("limit_exceeded", "max_wire_bytes reached")
                written = os.write(self.process.stdin.fileno(), request[:min(65536, remaining)])
                self.charge("wire_bytes", written, "max_wire_bytes")
                request = request[written:]
        meta = self.line(deadline)
        require(type(meta.get("step")) is int and meta["step"] == index and
                type(meta.get("n_floats")) is int and meta["n_floats"] == count, "wrong response step or shape")
        while len(self.buffer) < count * 4:
            self.pump(deadline)
        data = bytes(self.buffer[:count * 4])
        del self.buffer[:count * 4]
        self.control.check(deadline)
        return meta, data, (time.monotonic() - started) * 1000

    def finish(self) -> None:
        """EOF asks the native runner to release its context and exit successfully."""
        require(not self.buffer, "unexpected trailing runner output")
        self.process.stdin.close()
        deadline = min(self.control.deadline, time.monotonic() + self.limits["step_seconds"])
        streams = [self.process.stdout] + ([self.stderr] if self.stderr else [])
        while streams or self.process.poll() is None:
            self.control.check(deadline)
            ready, _, _ = select.select(streams, [], [], 0.05)
            for stream in ready:
                data = os.read(stream.fileno(), 65536)
                if not data:
                    streams.remove(stream)
                elif stream is self.process.stdout:
                    self.charge("wire_bytes", len(data), "max_wire_bytes")
                    raise ValueError("unexpected trailing runner output")
                else:
                    room = max(0, self.limits["max_log_bytes"] - self.counters["log_bytes"])
                    self.log.write(data[:room])
                    self.charge("log_bytes", len(data), "max_log_bytes")
        require(self.process.returncode == 0, "runner exited unsuccessfully")

    def close(self) -> None:
        # Kill the owned process group even if the group leader exited first.
        # A fresh process/context is the state reset between independent repetitions.
        self.signal_group(signal.SIGTERM)
        try:
            self.process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        self.signal_group(signal.SIGKILL)
        self.process.wait(timeout=2)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()

    def signal_group(self, sig: signal.Signals) -> None:
        try:
            os.killpg(self.process.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Some sandboxed hosts deny signalling a group that has already
            # vanished. Never ignore inability to stop a still-running child.
            if self.process.poll() is None:
                raise


def memory(meta: dict[str, Any], limits: dict[str, Any], initial: bool = False) -> None:
    name = "allocated_bytes" if initial else "observed_peak_buffer_bytes"
    for key, bound in ((name, "max_observed_buffer_bytes"), ("max_rss_bytes", "max_reported_rss_bytes")):
        require(integer(meta.get(key), 0, 1 << 50), "missing/invalid memory observation: " + key)
        if meta[key] > limits[bound]:
            raise Stopped("limit_exceeded", bound + " exceeded by runner observation")


def compare(actual: bytes, expected: bytes, atol: float, rtol: float,
            control: Control) -> dict[str, Any]:
    left, right = floats(actual, control), floats(expected, control)
    require(len(left) == len(right) and left, "invalid comparison shape")
    worst, errors = 0.0, 0
    for index, (observed, reference) in enumerate(zip(left, right)):
        if index % 16384 == 0:
            control.check()
        error = abs(observed - reference)
        worst = max(worst, error)
        errors += error > atol + rtol * abs(reference)
    return {"passed": errors == 0, "max_abs_error": worst, "values_outside_tolerance": errors}


def distribution(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"samples": len(values), "p50_ms": statistics.median(ordered),
            "p95_ms": ordered[math.ceil(len(ordered) * 0.95) - 1], "max_ms": ordered[-1]}


def profile(request_path: Path, output: Path,
            cancel: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "kind": "native_candidate_diagnostic",
              "profile_id": str(uuid.uuid4()), "status": "running", "started_at_unix_ms": time.time_ns() // 1_000_000,
              "phase": "validating", "eligible_for_placement": False, "sessions": [], "completed_steps": 0,
              "sequence_count": 1, "reset": "fresh native process per repetition",
              "limitations": ["No shared inference admission or reservations; operator must confirm idle.",
                              "Buffer/RSS observations omit scopes and are not a hard allocation cap.",
                              "Thermal/power conditions are operator descriptions, not monitored limits.",
                              "Backend selection and unlisted dynamic dependencies are not attested.",
                              "Local native protocol only; no link or mesh deployment measurements."]}
    save(output / "report.json", report)
    control, native = None, None
    counters = {"wire_bytes": 0, "log_bytes": 0}
    timings = {phase: {"compute": [], "service": []} for phase in ("prefill", "decode")}
    pins = {}
    last_progress = 0.0
    try:
        request = read_json(request_path)
        validate_limits(request["limits"])
        control = Control(request["limits"]["total_seconds"], cancel)
        report["request"] = request
        executable, artifact, fixture, steps = prepare(request, request_path.resolve().parent, control, pins)
        report["reference"] = fixture["reference"]
        report["tolerance"] = {k: fixture[k] for k in ("atol", "rtol")}
        begin, end = request["range"]["begin"], request["range"]["end"]
        command = [str(executable), str(artifact), str(begin), str(end), str(request["context_tokens"]),
                   str(request["runner"]["gpu_layers"]), str(request["limits"]["max_observed_buffer_bytes"]), "profile", "1"]
        with (output / "samples.jsonl").open("x") as samples, (output / "runner.log").open("xb") as log:
            for repetition in range(request["repetitions"]):
                control.check()
                report["phase"] = "loading"
                save(output / "report.json", report)
                start = time.monotonic()
                native = Native(command, log, control, request["limits"], counters)
                info = native.line(min(control.deadline, start + request["limits"]["load_seconds"]))
                session = {"repetition": repetition, "load_ms": (time.monotonic() - start) * 1000,
                           "initialization": info, "execution_seconds": 0.0}
                report["sessions"].append(session)
                require(info.get("protocol") == 1 and info.get("begin") == begin and info.get("end") == end,
                        "runner protocol/range mismatch")
                require(all(info.get(k) == request["model"][k] for k in ("layers", "embd", "vocab")), "runner model dimensions mismatch")
                source = info.get("source_sha256")
                require(source == request["model"]["source_sha256"] or
                        (source == "" and request["artifact"]["sha256"] == request["model"]["source_sha256"]), "runner source identity mismatch")
                memory(info, request["limits"], initial=True)
                save(output / "report.json", report)
                for index, (step, position, payload, expected) in enumerate(steps):
                    control.check()
                    report["phase"] = step["phase"]
                    meta, actual, elapsed = native.execute(index, position, step["n"], payload, len(expected) // 4, begin)
                    require(number(meta.get("compute_ms"), 0, request["limits"]["step_seconds"] * 1000), "invalid compute timing")
                    checked = compare(actual, expected, fixture["atol"], fixture["rtol"], control)
                    row = {"repetition": repetition, "step": index, "position": position, "n": step["n"],
                           "phase": step["phase"], "elapsed_ms": (time.monotonic() - control.started) * 1000,
                           "compute_ms": meta["compute_ms"], "service_ms": elapsed,
                           "observed_peak_buffer_bytes": meta.get("observed_peak_buffer_bytes"),
                           "max_rss_bytes": meta.get("max_rss_bytes"), "vm": meta.get("vm"), "correctness": checked}
                    samples.write(json.dumps(row, allow_nan=False) + "\n")
                    samples.flush()
                    require(checked["passed"], "reference output mismatch")
                    memory(meta, request["limits"])
                    session["execution_seconds"] += elapsed / 1000
                    report["completed_steps"] += 1
                    timings[step["phase"]]["compute"].append(meta["compute_ms"])
                    timings[step["phase"]]["service"].append(elapsed)
                    if time.monotonic() - last_progress >= 0.25:
                        save(output / "report.json", report)
                        last_progress = time.monotonic()
                report["phase"] = "finishing"
                save(output / "report.json", report)
                native.finish()
                native.close()
                native = None
                save(output / "report.json", report)
        report["phase"] = "verifying_pins"
        for path, before in pins.items():
            control.check()
            require(identity(path) == before, "pinned file changed during profiling")
        if max(s["execution_seconds"] for s in report["sessions"]) < request["min_session_seconds"]:
            raise Stopped("insufficient_duration", "no single session met the requested execution duration")
        report["status"] = "passed"
        report["phase"] = "complete"
    except Stopped as error:
        report.update(status=error.outcome, error=str(error))
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=type(error).__name__ + ": " + str(error))
    finally:
        if native:
            try:
                native.close()
            except Exception as error:
                report["outcome_before_cleanup"] = report["status"]
                report["status"] = "cleanup_failed"
                report["cleanup_error"] = type(error).__name__ + ": " + str(error)
        report["counters"] = counters
        report["summary"] = {phase: {name: distribution(values) for name, values in rows.items()}
                             for phase, rows in timings.items()}
        report["finished_at_unix_ms"] = time.time_ns() // 1_000_000
        report["elapsed_seconds"] = time.monotonic() - control.started if control else None
        save(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cancel-file", type=Path)
    parser.add_argument("--confirm-idle", action="store_true",
                        help="confirm owner compute opt-in and an idle, coordinated local test window")
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("local subprocess profiler requires POSIX")
    if not args.confirm_idle:
        parser.error("--confirm-idle is required; this diagnostic has no shared inference admission gate")
    cancelled = False

    def stop(_signal: int, _frame: Any) -> None:
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    report = profile(args.request, args.output_dir,
                     lambda: cancelled or (args.cancel_file is not None and args.cancel_file.exists()))
    print(json.dumps({"profile_id": report["profile_id"], "status": report["status"],
                      "report": str(args.output_dir / "report.json")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
