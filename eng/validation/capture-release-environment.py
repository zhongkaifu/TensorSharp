#!/usr/bin/env python3
"""Capture the VM resource limits and exact binaries used by a release run."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dotnet", required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--server", type=Path)
    parser.add_argument("--storage-root", default="/workspace")
    parser.add_argument("--cuda", default="/usr/local/cuda/bin/nvcc")
    args = parser.parse_args()
    commands = {
        "gpu": ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv"],
        "gpu_topology": ["nvidia-smi", "topo", "-m"],
        "competing_gpu_processes": ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"],
        "cpu": ["lscpu"], "memory": ["free", "-b"], "storage": ["df", "-B1", args.storage_root],
        "dotnet": [args.dotnet, "--info"], "cuda": [args.cuda, "--version"],
        "sandbox": ["bwrap", "--version"],
        "user_namespaces": ["unshare", "-Ur", "true"],
    }
    report = {"captured_at_unix": time.time(), "platform": platform.platform(), "commands": {}, "limits": {}, "binaries": {}}
    for name, command in commands.items():
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
            report["commands"][name] = {"command": command, "exit_code": completed.returncode,
                                         "stdout": completed.stdout, "stderr": completed.stderr}
        except Exception as error:
            report["commands"][name] = {"command": command, "error": str(error)}
    for path in ["/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes",
                 "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
                 "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/cpu.max", "/proc/self/status"]:
        if Path(path).exists():
            report["limits"][path] = Path(path).read_text()
    for binary in (args.native, args.server):
        if binary and binary.exists():
            report["binaries"][str(binary)] = {"size": binary.stat().st_size,
                "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
