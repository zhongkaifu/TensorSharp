#!/usr/bin/env python3
"""Run real Windows Flash agent workflows against an already running local server.

The operator starts the server and retains its command, model/native identities,
and logs. This client does not rebuild, download models, or enable execution on
the server. --unconfined acknowledges functional Windows execution explicitly;
none of these results establish filesystem or network isolation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-store", type=Path, required=True)
    parser.add_argument("--unconfined", action="store_true")
    parser.add_argument("--target-shell", choices=("powershell", "cmd"), default="powershell")
    parser.add_argument("--timeout", type=float, default=900, help="Per workflow request deadline")
    parser.add_argument("--suite-timeout", type=float, default=1800, help="Total client deadline, including all phases")
    parser.add_argument("--tool-calls", action="store_true", help="Also test declared client-tool calls, thinking on/off")
    args = parser.parse_args()
    if not args.unconfined:
        parser.error("Windows execution requires explicit --unconfined; this suite cannot validate OS isolation")
    if args.timeout <= 0 or args.suite_timeout <= 0:
        parser.error("Timeouts must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory to preserve previous evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    report = {"started_unix": time.time(), "server": args.url, "execution_mode": "unconfined",
              "quality_only": True, "phases": [], "passed": False,
              "harness_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in (
                  Path(__file__).name, "validate-release-agent-workflows.py", "verify-agent-code-artifacts.py",
                  "validate-qwen38-tool-calls.py")},
              "limitations": ["Server binaries/model and resource logs must be retained separately.",
                              "No OS isolation or comparative performance claim."]}
    base = [sys.executable, str(HERE / "validate-release-agent-workflows.py"), "--url", args.url,
            "--target-shell", args.target_shell, "--sandbox-off", "--concurrency", "1",
            "--timeout", str(args.timeout)]
    phases = [
        ("skills", base + ["--scenarios", "skill_selection,skill_run", "--output", str(output / "skills.json")], 2),
        ("actions", base + ["--scenarios", "skill_script_run,shell_run,code_generation_run,code_edit_run",
                             "--output", str(output / "actions.json")], 4),
        ("independent-code", [sys.executable, str(HERE / "verify-agent-code-artifacts.py"),
                              "--workflow-report", str(output / "actions.json"), "--artifact-store",
                              str(args.artifact_store.resolve()), "--sandbox-off", "--python", sys.executable,
                              "--output", str(output / "independent-code.json")], 1),
    ]
    if args.tool_calls:
        phases.append(("tool-calls", [sys.executable, str(HERE / "validate-qwen38-tool-calls.py"),
                       "--url", args.url, "--thinking", "off,on", "--timeout", str(args.timeout),
                       "--output", str(output / "tool-calls.json")], 32))
    def save():
        (output / "suite.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save()
    deadline = time.monotonic() + args.suite_timeout
    for name, command, requests in phases:
        started = time.monotonic()
        entry = {"name": name, "command": command, "exit_code": None, "passed": False}
        report["phases"].append(entry)
        save()
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Suite deadline exhausted before this phase; not executed")
            with (output / (name + ".log")).open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=min(remaining, requests * (args.timeout + 60) + 60),
                                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            entry["exit_code"] = result.returncode
            entry["passed"] = result.returncode == 0
        except Exception as error:
            entry["error"] = str(error)
        entry["wall_seconds"] = time.monotonic() - started
        save()
        print(json.dumps({"phase": name, "passed": entry["passed"], "wall_seconds": entry["wall_seconds"]}), flush=True)
    report["passed"] = all(phase["passed"] for phase in report["phases"])
    report["finished_unix"] = time.time()
    save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
