#!/usr/bin/env python3
"""Explicit future GPU entry point, never invoked by offline build/CPU tests.

N8 only; --bench always completes all 96 cases for every candidate first.
There is no quality, spill, resource or tolerance bypass.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("dp4a_rows_build_for_run", HERE / "build.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def verify_build(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    permit = json.loads((directory / build.PERMIT).read_text())
    if (manifest.get("schema") != "qwen35-dp4a-rows-build/1" or manifest.get("complete") is not True
            or manifest.get("activation_rows") != 8
            or manifest.get("strict_cases") != 96 or manifest.get("timed_shapes") != 12
            or permit.get("schema") != "qwen35-dp4a-rows-permission/1"
            or permit.get("manifest_sha256") != build.digest(directory / "manifest.json")):
        raise ValueError("missing, stale or rejected DP4a rows offline build")
    resources = manifest.get("resources")
    if not isinstance(resources, dict) or set(resources) != set(build.SYMBOLS) or not all(
            r.get("offline_resource_qualified") is True for r in resources.values()):
        raise ValueError("incomplete actual resources; no GPU authorization")
    if (manifest.get("oracle") != dict(symbol=build.ORACLE, rows_per_warp=4, block_threads=128)
            or manifest.get("candidates") != [dict(symbol=build.SYMBOLS[r], rows_per_warp=r,
                block_threads=32 if r == 4 else 128,
                role="launch_only_control" if r == 4 else "rows_only_experiment") for r in build.ROWS]):
        raise ValueError("unexpected candidate/launch identity")
    if (not isinstance(manifest.get("artifacts"), dict) or set(manifest["artifacts"]) != set(build.ARTIFACTS)
            or permit.get("artifacts") != manifest["artifacts"]):
        raise ValueError("permit does not identify the built artifacts")
    for name, expected in manifest["artifacts"].items():
        if Path(name).name != name or build.digest(directory / name) != expected:
            raise ValueError(f"built artifact changed: {name}")
    if manifest.get("tools") != {n:build.digest(HERE / n) for n in build.TOOLS}:
        raise ValueError("DP4a rows tool changed; rebuild required")
    if manifest.get("reused_fixture_tools") != {n:build.digest(build.FROZEN / n) for n in build.FROZEN_FILES}:
        raise ValueError("sealed fixture source changed; rebuild required")
    if build.digest(manifest["source"]) != manifest["source_sha256"]:
        raise ValueError("production template source changed; rebuild required")
    plan = json.loads((directory / "plan.json").read_text())
    if plan != build.SEALED.plan() or (directory / "plan.h").read_text() != build.SEALED.plan_header(plan):
        raise ValueError("plan differs from the sealed 96-case fixture")
    return manifest


def validate_result(observed, bench, rounds, iterations):
    return (observed.get("complete") is True
        and observed.get("status") == "STRICT_OUTPUT_BITS_AND_ARGMAX_PASS"
        and observed.get("activation_rows") == 8
        and observed.get("strict_cases_per_candidate") == 96
        and observed.get("strict_candidate_comparisons") == 384
        and observed.get("candidates") == 4
        and observed.get("timed_shapes_per_candidate") == (12 if bench else 0)
        and observed.get("rounds") == rounds and observed.get("iterations_per_sample") == iterations)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--name", required=True, help="NEW result directory name")
    ap.add_argument("--acts", default=8, type=int)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--rounds", default=5, type=int)
    ap.add_argument("--iterations", default=3, type=int)
    args = ap.parse_args(argv)
    directory = build.evidence_dir(args.build)
    if Path(args.name).name != args.name or args.name in (".", ".."):
        ap.error("--name must be a single new subdirectory name")
    out = directory / args.name
    out.mkdir(parents=True, exist_ok=False)
    state = dict(schema="qwen35-dp4a-rows-run/1", complete=False,
        started_utc=datetime.now(timezone.utc).isoformat(), activation_rows=args.acts,
        gpu_invoked=False, hardware_counters="NOT_RUN")
    save = lambda: build.save(out / "run.json", state)
    save()
    if args.acts != 8:
        state["status"] = "NOT_SUPPORTED"; save(); print(json.dumps(state)); return 2
    if args.rounds < 1 or args.iterations < 1:
        state["status"] = "INVALID_CONFIGURATION"; save(); return 1
    try:
        manifest = verify_build(directory)
        command = [str(directory / build.RUNNER), str(directory), str(out), "--bench" if args.bench else "--strict",
                   "--acts", "8", "--rounds", str(args.rounds), "--iterations", str(args.iterations)]
        state.update(command=command, build_manifest_sha256=build.digest(directory / "manifest.json"),
                     build_manifest=manifest, gpu_invoked=True)
        save()
        with (out / "runner.jsonl").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        observed = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
        complete = result.returncode == 0 and validate_result(observed, args.bench, args.rounds, args.iterations)
        state.update(exit_code=result.returncode, result=observed, complete=complete,
            runner_log_sha256=build.digest(out / "runner.jsonl"),
            status=("STRICT_OUTPUT_BITS_AND_ARGMAX_PASS_TIMED" if args.bench else "STRICT_OUTPUT_BITS_AND_ARGMAX_PASS_NOT_TIMED")
                   if complete else "REJECTED_NOT_VALIDATED")
        save(); print(json.dumps(state, indent=2)); return 0 if complete else 1
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        state.update(complete=False, status="REJECTED_NOT_VALIDATED", error=str(error))
        save(); print(json.dumps(state)); return 1


if __name__ == "__main__":
    raise SystemExit(main())
