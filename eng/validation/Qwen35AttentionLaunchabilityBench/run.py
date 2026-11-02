#!/usr/bin/env python3
"""Explicit GPU entry point, one independently built arm at a time."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("launchability_build_for_run", HERE / "build.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def verify_build(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    permit = json.loads((directory / build.PERMIT).read_text())
    arm = manifest.get("arm")
    if arm not in build.ARMS:
        raise ValueError("unknown independent arm")
    variant, o1 = build.ARMS[arm]
    if (manifest.get("schema") != "qwen35-attention-launchability-build/1"
            or manifest.get("complete") is not True or manifest.get("candidate_variant") != variant
            or manifest.get("baseline_flags") != build.flags(False) or manifest.get("candidate_flags") != build.flags(o1)
            or permit.get("schema") != "qwen35-attention-launchability-offline/1"
            or permit.get("quality") != "NOT_RUN" or permit.get("sealed_gate_override") is not False
            or permit.get("manifest_sha256") != build.digest(directory / "manifest.json")):
        raise ValueError("missing/incompatible independent compiled build; sealed marker cannot authorize this campaign")
    build.frozen_module()
    if manifest.get("frozen_tools") != build.PINS or manifest.get("tools") != {n: build.digest(HERE / n) for n in build.TOOLS}:
        raise ValueError("sealed or owned tools changed")
    source = Path(manifest["source"])
    if build.digest(source) != manifest.get("source_sha256"):
        raise ValueError("production source changed after offline build")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(build.artifact_names(variant)) or permit.get("artifacts") != artifacts:
        raise ValueError("artifact identities incomplete")
    for name, value in artifacts.items():
        if Path(name).name != name or build.digest(directory / name) != value:
            raise ValueError("changed build artifact " + name)
    if (directory / "qualified-variants.txt").read_text() != f"0\n{variant}\n":
        raise ValueError("wrong candidate selection")
    for i in (0, variant):
        if (directory / f"variant-{i}.cu").read_text() != build.generated(source.read_text(), i):
            raise ValueError("kernel differs from pinned unchanged arithmetic")
    if (directory / "bench.cu").read_text() != build.runner_source() or build.digest(directory / "checks.h") != build.PINS["checks.h"]:
        raise ValueError("runner/checker source differs")
    resources = manifest.get("resources")
    if not isinstance(resources, dict) or set(resources) != {"0", str(variant)}:
        raise ValueError("missing actual ptxas facts")
    for r in resources.values():
        if any(type(r.get(k)) is not int or r[k] < 0 for k in ("registers_per_thread", "stack_bytes", "spill_store_bytes", "spill_load_bytes", "static_shared_bytes", "dynamic_shared_bytes", "total_shared_bytes")):
            raise ValueError("invalid actual ptxas facts")
    return manifest


def shapes(long_cases=False):
    result = [(0 if r % 2 else 255, r, 1, 1, 2, s, (r + s) % 4)
              for r in (1, 7, 63, 64, 65, 127, 128, 129, 257) for s in (1, 2, 3, 7, 8)]
    result += [(257, 128 // g + 1, 2, g, 2, 3, pattern)
               for g in (1, 2, 3, 6, 8, 16, 64) for pattern in (0, 1, 3, 4)]
    result += [(8191, r, 1, 6, 4, 7, 3) for r in (11, 21, 22, 23, 43, 65)]
    if long_cases:
        result += [(p, 129, 1, 6, 4, 3, 0) for p in (131072, 250000)]
    return result


def timed_shapes(long_cases=False):
    result = [(0, 4096, 1, 6, 4, 1, 0), (19000, 4096, 1, 6, 4, 1, 0),
              (19000, 2657, 1, 6, 4, 1, 0), (19000, 512, 2, 6, 4, 3, 0)]
    if long_cases:
        result += [(p - 4096, 4096, 1, 6, 4, 1, 0) for p in (131072, 250000)]
    return result


def name(shape):
    return "p%d-r%d-s%d-g%d-k%d-x%d-v%d" % shape


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_observation(result, events, manifest, mode, long_cases):
    variant = manifest["candidate_variant"]
    devices = [e for e in events if e.get("event") == "device"]
    if (len(devices) != 1 or not isinstance(devices[0].get("name"), str) or not devices[0]["name"]
            or any(type(devices[0].get(k)) is not int or devices[0][k] <= 0 for k in ("sms", "registers_per_sm", "shared_per_sm"))):
        raise ValueError("missing/duplicate actual device evidence")
    resources = [e for e in events if e.get("event") == "device_resources"]
    ids = []
    for e in resources:
        i = e.get("variant")
        ids.append(i)
        if type(i) is not int or i not in (0, variant):
            raise ValueError("wrong runtime variant")
        expected = manifest["resources"][str(i)]
        if (any(type(e.get(k)) is not int or e[k] < 0 for k in ("regs", "local_bytes", "dynamic_shared", "static_shared", "potential_active_ctas"))
                or e["potential_active_ctas"] < 1 or e["regs"] != expected["registers_per_thread"]
                or e["dynamic_shared"] != expected["dynamic_shared_bytes"] or e["static_shared"] != expected["static_shared_bytes"]):
            raise ValueError("actual driver launchability/identity failed")
    if len(ids) != 2 or set(ids) != {0, variant}:
        raise ValueError("missing/duplicate actual driver resources")
    if result.get("complete") is not True or result.get("device_resource_rejected_variants") != []:
        raise ValueError("native run incomplete/rejected")
    strict = [e for e in events if e.get("event") == "strict_shape_pass"]
    abba = [e for e in events if e.get("event") == "abba_gpu_ms"]
    if mode == "resources":
        if result.get("validation") != "resources_only_not_quality" or strict or abba:
            raise ValueError("resource-only run mislabeled as model quality")
    else:
        expected = {name(s) for s in shapes(long_cases)}
        actual = [e.get("shape") for e in strict]
        if (len(actual) != len(set(actual)) or set(actual) != expected
                or any(type(e.get("variants")) is not int or e["variants"] != 2 or e.get("full_output_and_partial_bits") is not True for e in strict)):
            raise ValueError("strict full-output/partial plan incomplete or duplicated")
        if (type(result.get("strict_shapes")) is not int or result["strict_shapes"] != len(expected)
                or type(result.get("strict_candidate_comparisons")) is not int or result["strict_candidate_comparisons"] != 3 * len(expected)
                or result.get("long_cases") is not long_cases or result.get("timed") is not (mode == "bench")
                or result.get("active_variants") != [0, variant]):
            raise ValueError("native strict identities/counts failed")
        timed = {name(s): s for s in timed_shapes(long_cases)}
        seen = []
        for e in abba:
            n, r = e.get("shape"), e.get("repetition")
            seen.append((n, r))
            if (n not in timed or type(r) is not int or r not in range(5)
                    or type(e.get("variant")) is not int or e["variant"] != variant
                    or type(e.get("kernels_per_half")) is not int or e["kernels_per_half"] != 3
                    or e.get("total_keys") != timed[n][0] + timed[n][1]
                    or e.get("complete_subtiles_and_combine") is not True or e.get("poison_and_verify_each_half") is not True
                    or not all(positive(e.get(k)) for k in ("a", "b", "c", "d"))):
                raise ValueError("invalid complete-pipeline ABBA half")
        wanted = {(n, r) for n in timed for r in range(5)} if mode == "bench" else set()
        if len(seen) != len(set(seen)) or set(seen) != wanted:
            raise ValueError("timing set missing/duplicate/unexpected")
    if any(e.get("event") not in ("device", "device_resources", "strict_shape_pass", "abba_gpu_ms") for e in events):
        raise ValueError("native rejection/unexpected diagnostic event")
    return dict(driver_launchable=True, quality="NOT_RUN" if mode == "resources" else "PASS",
                strict_shapes=len(strict), timing_events=len(abba), resources=resources)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--name", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--bench", action="store_true")
    modes.add_argument("--resources-only", action="store_true")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args(argv)
    directory = build.evidence_dir(args.build)
    if not args.name or Path(args.name).name != args.name or args.name in (".", "..") or args.timeout <= 0:
        parser.error("fresh simple result name and positive timeout required")
    out = directory / args.name
    out.mkdir(exist_ok=False)
    mode = "resources" if args.resources_only else "bench" if args.bench else "strict"
    state = dict(schema="qwen35-attention-launchability-run/1", complete=False, quality_pass=False,
                 gpu_invoked=False, mode=mode, started_utc=datetime.now(timezone.utc).isoformat(),
                 hardware_counters="NOT_RUN", production_integrated=False, sealed_failures_regraded=False)
    persist = lambda: build.save(out / "controller.json", state)
    persist()
    try:
        manifest = verify_build(directory)
        argv = [str(directory / build.RUNNER), str(directory), str(out), "--" + mode] + (["--long"] if args.long else [])
        state.update(command=argv, arm=manifest["arm"], build_manifest_sha256=build.digest(directory / "manifest.json"), gpu_invoked=True)
        persist()
        build.command(argv, out / "runner.jsonl", args.timeout)
        result = json.loads((out / "run.json").read_text())
        events = [json.loads(s) for s in (out / "runner.jsonl").read_text().splitlines() if s.strip()]
        validated = validate_observation(result, events, manifest, mode, args.long)
        verify_build(directory)
        state.update(complete=True, quality_pass=mode != "resources", validated=validated,
                     native_result=result, status="DRIVER_ONLY_NOT_QUALITY" if mode == "resources" else "FULLBITS_PASS_TIMED" if mode == "bench" else "FULLBITS_PASS_NOT_TIMED")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        state.update(complete=False, quality_pass=False, status="REJECTED_NOT_VALIDATED", error=str(error))
    if (out / "runner.jsonl").exists():
        state["runner_log_sha256"] = build.digest(out / "runner.jsonl")
    state["finished_utc"] = datetime.now(timezone.utc).isoformat()
    persist()
    print(json.dumps(state, indent=2))
    return 0 if state["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
