#!/usr/bin/env python3
"""Pause/resume only this run's model download processes during benchmarks."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pause", "resume"))
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    processes = []
    if args.action == "pause":
        for path in Path("/proc").iterdir():
            if not path.name.isdecimal():
                continue
            try:
                command = (path / "cmdline").read_bytes().split(b"\0")
                scripts = [part.decode() for part in command if part.endswith((b"provision-release-models.py", b"provision-release-catalog.py"))]
                if not any(script.startswith(args.run_root + "/") for script in scripts):
                    continue
                started = (path / "stat").read_text().split()[21]
                os.kill(int(path.name), signal.SIGSTOP)
                processes.append({"pid": int(path.name), "started": started, "command": [part.decode() for part in command if part]})
            except (OSError, ProcessLookupError):
                continue
        args.state.write_text(json.dumps({"paused_at_unix": time.time(), "processes": processes}, indent=2) + "\n")
    else:
        state = json.loads(args.state.read_text())
        for process in state["processes"]:
            path = Path("/proc") / str(process["pid"])
            try:
                if (path / "stat").read_text().split()[21] == process["started"]:
                    os.kill(process["pid"], signal.SIGCONT)
                    processes.append(process)
            except (OSError, ProcessLookupError):
                pass
        state["resumed_at_unix"] = time.time()
        args.state.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({"action": args.action, "pids": [process["pid"] for process in processes]}))


if __name__ == "__main__":
    main()
