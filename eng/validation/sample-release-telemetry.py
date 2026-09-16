#!/usr/bin/env python3
"""Record bounded GPU/resource telemetry for one owned inference process."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    process = Path("/proc") / str(args.pid)
    identity = (process / "cmdline").read_bytes()
    start_ticks = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        while process.exists():
            try:
                current_stat = (process / "stat").read_text()
                if ((process / "cmdline").read_bytes() != identity
                        or current_stat.rsplit(")", 1)[1].split()[19] != start_ticks):
                    break
                row = {"time_unix": time.time(), "pid": args.pid, "process_stat": current_stat}
                for name, path in {
                    "process_status": process / "status",
                    "process_io": process / "io",
                    "cgroup_memory_bytes": Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
                    "cgroup_memory_failcnt": Path("/sys/fs/cgroup/memory/memory.failcnt"),
                    "cgroup_memory_stat": Path("/sys/fs/cgroup/memory/memory.stat"),
                    "cgroup_memory_pressure": Path("/sys/fs/cgroup/memory.pressure"),
                    "cgroup_cpu_stat": Path("/sys/fs/cgroup/cpu/cpu.stat"),
                    "host_memory_pressure": Path("/proc/pressure/memory"),
                    "host_io_pressure": Path("/proc/pressure/io"),
                }.items():
                    if path.exists():
                        row[name] = path.read_text()
                result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,power.draw,clocks.sm", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                row["gpu_csv"] = result.stdout
                row["gpu_query_exit_code"] = result.returncode
                output.write(json.dumps(row) + "\n")
                output.flush()
            except (OSError, subprocess.TimeoutExpired):
                break
            time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
