#!/usr/bin/env python3
"""Explicit GPU gate/timing entry point; NEVER invoked by build.py or CPU tests.

  python eng/validation/Qwen35Int8MmaBench/run.py --build artifacts/qwen38-int8mma --name strict-1
  python eng/validation/Qwen35Int8MmaBench/run.py --build artifacts/qwen38-int8mma --name abba-1 --bench

Only N8 is supported. --bench always repeats the entire strict full-output bit
plan first; there is no tolerance, skip-quality or force-resource override.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import build


def verify_build(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    permit = json.loads((directory / "gpu-quality-permitted.json").read_text())
    if (manifest.get("complete") is not True or manifest.get("resources", {}).get("offline_resource_qualified") is not True
            or manifest.get("activation_rows") != 8 or permit.get("manifest_sha256") != build.digest(directory / "manifest.json")):
        raise ValueError("missing, stale or rejected offline resource gate")
    if (not isinstance(manifest.get("artifacts"), dict) or set(manifest["artifacts"]) != set(build.ARTIFACTS)
            or permit.get("artifacts") != manifest["artifacts"]):
        raise ValueError("resource permit does not identify the built artifacts")
    if not isinstance(manifest.get("tools"), dict) or set(manifest["tools"]) != set(build.TOOLS):
        raise ValueError("prototype source identities are incomplete")
    for name, expected in manifest["artifacts"].items():
        if Path(name).name != name or build.digest(directory / name) != expected:
            raise ValueError(f"built artifact changed: {name}")
    if build.digest(manifest["source"]) != manifest["source_sha256"]:
        raise ValueError("production oracle source changed; rebuild required")
    for name, expected in manifest["tools"].items():
        if Path(name).name != name or build.digest(build.HERE / name) != expected:
            raise ValueError(f"prototype tool changed; rebuild required: {name}")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--name", required=True, help="new result subdirectory under the ignored build directory")
    ap.add_argument("--acts", type=int, default=8)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=3)
    args = ap.parse_args(argv)
    directory = build.evidence_dir(args.build)
    out = directory / args.name
    if Path(args.name).name != args.name or args.name in (".", ".."):
        ap.error("--name must be a single new subdirectory name")
    out.mkdir(parents=True, exist_ok=False)
    state = dict(schema="qwen35-int8mma-run/1", complete=False, started_utc=datetime.now(timezone.utc).isoformat(),
                 activation_rows=args.acts, hardware_counters="NOT_RUN", gpu_invoked=False)
    save = lambda: (out / "run.json").write_text(json.dumps(state, indent=2) + "\n")
    save()
    if args.acts != 8:
        state["status"] = "NOT_SUPPORTED";save();print(json.dumps(state));return 2
    if args.rounds < 1 or args.iterations < 1:
        state["status"] = "INVALID_CONFIGURATION";save();return 1
    try:
        manifest = verify_build(directory)
        command = [str(directory / "int8mma-bench"), str(directory), str(out), "--bench" if args.bench else "--strict",
                   "--acts", "8", "--rounds", str(args.rounds), "--iterations", str(args.iterations)]
        state.update(command=command, build_manifest_sha256=build.digest(directory / "manifest.json"),
                     build_manifest=manifest, gpu_invoked=True)
        save()
        with (out / "runner.jsonl").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        observed = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
        state.update(exit_code=result.returncode, result=observed,
                     complete=result.returncode == 0 and observed.get("complete") is True,
                     runner_log_sha256=build.digest(out / "runner.jsonl"))
        if not state["complete"]: state["status"] = "REJECTED_NOT_VALIDATED"
        else: state["status"] = "STRICT_OUTPUT_BITS_PASS_TIMED" if args.bench else "STRICT_OUTPUT_BITS_PASS_NOT_TIMED"
        save();print(json.dumps(state, indent=2));return 0 if state["complete"] else 1
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        state.update(complete=False, status="REJECTED_NOT_VALIDATED", error=str(error));save();print(json.dumps(state));return 1


if __name__ == "__main__":
    raise SystemExit(main())
