#!/usr/bin/env python3
"""Stop this run's downloads when its owned DeepSeek server begins Engram warming."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    manifest = json.loads((args.profile_output / "profile.json").read_text())
    process = Path("/proc") / str(manifest["server_pid"])
    initial_stat = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
    report = {"server_pid": manifest["server_pid"], "started_at_unix": time.time(), "status": "waiting"}
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline and process.exists():
        if (process / "stat").read_text().rsplit(")", 1)[1].split()[19] != initial_stat:
            break
        for line in (args.profile_output / "server.log").read_text(errors="replace").splitlines():
            if "[dsv41] warming " in line:
                command = [sys.executable, str(Path(__file__).with_name("control-release-downloads.py")),
                           "pause", "--run-root", str(args.run_root),
                           "--state", str(args.run_root / "logs/download-pause.json")]
                result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                report.update(status="paused" if result.returncode == 0 else "failed", marker=line,
                              command=command, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
                              finished_at_unix=time.time())
                (args.profile_output / "downloads-engram-warm-pause.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report), flush=True)
                return result.returncode
        time.sleep(1)
    report.update(status="not-observed", finished_at_unix=time.time())
    (args.profile_output / "downloads-engram-warm-pause.json").write_text(json.dumps(report, indent=2) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
