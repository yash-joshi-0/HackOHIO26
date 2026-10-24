"""Compare an Android prefix over Iroh with local CPU suffix/reference runners."""

import argparse
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model-validation"))
from shard_trial import Runner, SocketRunner, compare_logits, file_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": "two-layer GLM correctness over Android Iroh; no capacity/speed claim",
              "fixture_sha256": file_hash(args.fixture), "steps": []}
    runners = []
    try:
        prefix = SocketRunner(args.port)
        runners.append(prefix)
        for begin, name in [(1, "suffix"), (0, "reference")]:
            command = [str(args.runner), str(args.fixture), str(begin), "2", "128", "0",
                       "4294967296", "Hello from Android", "1"]
            runners.append(Runner(command, args.output / f"{name}.log"))
        prefix, suffix, reference = runners
        if prefix.info["begin"] != 0 or prefix.info["end"] != 1:
            raise ValueError("unexpected Android shard range")
        for runner in runners:
            if runner.info["layers"] != 2 or any(runner.info[k] != prefix.info[k] for k in ["embd", "vocab", "tokens"]):
                raise ValueError("incompatible fixture metadata")
        report["initialization"] = [r.info for r in runners]
        tokens = prefix.info["tokens"]
        for _ in range(16):
            payload = struct.pack("!" + "I" * len(tokens), *tokens)
            expected, ref_logits = reference.execute(len(tokens), payload)
            pre, hidden = prefix.execute(len(tokens), payload)
            actual, logits = suffix.execute(len(tokens), hidden)
            comparison = compare_logits(ref_logits, logits)
            comparison["same_greedy_token"] = expected["token"] == actual["token"]
            report["steps"].append({"prefix": pre, "suffix": actual, "comparison": comparison})
            if not comparison["same_greedy_token"] or comparison["max_abs_error"] > 0.02:
                raise RuntimeError("cross-device partition equivalence failed")
            tokens = [actual["token"]]
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        for runner in reversed(runners):
            runner.close()
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
