#!/usr/bin/env python3
"""Finish the intentionally reduced, known-imprecise V4.1 attention baseline.

The original lifecycle runner must already be stopped (SIGSTOP) while its
long-context child finishes. This retains that original planned scope and adds
an explicit early-stop reason instead of marking unrun suites successful.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    runner_command = Path(f"/proc/{args.runner_pid}/cmdline").read_bytes()
    server_command = Path(f"/proc/{args.server_pid}/cmdline").read_bytes()
    if b"run-release-profile.py" not in runner_command or b"TensorSharp.Server.Host.dll" not in server_command:
        raise ValueError("PIDs do not identify this validation runner/server")
    original_path = args.output / "profile.json"
    original = json.loads(original_path.read_text())
    model = original["model_id"]
    original_native = original["binaries"]
    report = {"reason": "Independent numerical fixtures found V4.1 flash-attention source precision loss. Retain this version as a reduced pre-attention-fix baseline; complete release coverage will use the corrected candidate.",
              "started_at_unix": time.time(), "status": "waiting-for-long-context", "commands": [],
              "original_profile": str(original_path), "original_binaries": original_native}
    path = args.output / "reduced-baseline-plan.json"

    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(path)

    save()
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        try:
            long_result = json.loads((args.output / "long-context.json").read_text())
            if long_result.get("run_complete"):
                break
        except (OSError, ValueError):
            pass
        time.sleep(5)
    else:
        raise TimeoutError("Long-context suite did not finish")
    common = [sys.executable, str(args.repo / "benchmarks/engine_comparison/validate_inference.py"),
              "--url", args.url, "--engine", "tensorsharp", "--model", model,
              "--weights-id", "58d8ac86298fdf85a2440defee08b1abcad32e45-Q4_K_M",
              "--profile", "layer7-cpumoe12-f16"]
    commands = [
        common + ["--scenarios", "decode,decode_8k", "--concurrency", "1", "--repeats", "3",
                  "--output", str(args.output / "decode-benchmark-reduced.json")],
        common + ["--scenarios", "tool_round_trip,agentic", "--concurrency", "1,4", "--repeats", "1",
                  "--structured-tool-results", "--output", str(args.output / "structured-tool-results.json")],
    ]
    report["status"] = "running-tail"
    save()
    for index, command in enumerate(commands):
        entry = {"argv": command, "started_at_unix": time.time()}
        report["commands"].append(entry)
        save()
        print("Running reduced suite", index, command, flush=True)
        with (args.output / f"reduced-suite-{index}.log").open("w") as log:
            completed = subprocess.run(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT)
        entry.update(exit_code=completed.returncode, finished_at_unix=time.time())
        save()
    # Both PIDs were verified above and remain owned by this run. The stopped
    # original runner is terminated before the server so it cannot launch its
    # superseded remaining suites against a stopped endpoint.
    if Path(f"/proc/{args.runner_pid}/cmdline").read_bytes() != runner_command:
        raise ValueError("Runner PID changed")
    os.kill(args.runner_pid, signal.SIGKILL)
    if Path(f"/proc/{args.server_pid}/cmdline").read_bytes() != server_command:
        raise ValueError("Server PID changed")
    os.killpg(args.server_pid, signal.SIGTERM)
    for _ in range(30):
        status = Path(f"/proc/{args.server_pid}/stat")
        if not status.exists() or status.read_text().split()[2] == "Z":
            break
        time.sleep(1)
    else:
        os.killpg(args.server_pid, signal.SIGKILL)
    original["status"] = "stopped-after-reduced-baseline"
    original["finished_at_unix"] = time.time()
    original["scope_change"] = {"reason": report["reason"], "plan": str(path),
                                "unrun_original_suite_indices": [2, 3, 4, 5, 6]}
    if len(original["suites"]) > 1:
        original["suites"][1]["exit_code"] = int(any(case["status"] != "ok" for case in long_result["cases"]))
    original_path.write_text(json.dumps(original, indent=2) + "\n")
    report.update(status="completed-reduced-scope", finished_at_unix=time.time())
    save()
    print("Reduced baseline complete; server stopped", flush=True)


if __name__ == "__main__":
    main()
