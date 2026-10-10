#!/usr/bin/env python3
"""Run one directly owned validation process with a hard deadline and durable outcome.

This is for a single-process probe (or server), not a shell/process-tree launcher.
Native calls cannot safely be cancelled in-process. A deadline kills only this
child and records incomplete execution; it never turns partial evidence into PASS.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def run(command, output, timeout):
    output.mkdir(parents=True, exist_ok=False)
    report = {"command": command, "started_utc": datetime.now(timezone.utc).isoformat(),
              "timeout_seconds": timeout, "complete": False, "timed_out": False,
              "exit_code": None, "quality_passed": None}

    def publish():
        temporary = output / "execution.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "execution.json")

    publish()
    start = time.monotonic()
    child = None
    try:
        with (output / "process.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            report["pid"] = child.pid
            publish()
            try:
                report["exit_code"] = child.wait(timeout=timeout)
                report["complete"] = True
            except subprocess.TimeoutExpired:
                report["timed_out"] = True
                child.kill()
                report["exit_code"] = child.wait(timeout=15)
    except BaseException as error:
        report["error"] = repr(error)
        if child is not None and child.poll() is None:
            child.kill()
            report["exit_code"] = child.wait(timeout=15)
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - start
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["qualification"] = "Exit zero is completed execution only; consult the probe's numerical/semantic evidence."
        publish()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Supply a nonempty command and positive finite timeout")
    report = run(command, args.output.resolve(), args.timeout)
    print(json.dumps(report, indent=2))
    return 124 if report["timed_out"] else report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
