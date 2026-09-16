#!/usr/bin/env python3
"""Replay the completed Gemma 30+16 agent cases with thinking enabled.

Default is a read-only plan. --execute requires a free server lane and the exact
completed baseline binaries. It stages harnesses only under a new output tree,
never builds or updates the WSL repository, and retains independent source tests.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


EXPECTED_NATIVE = "c0abb8bc954ba62ac616b4225d3cba69bf0ced78b973b822416a8df53221d692"
BASELINE_HASHES = {
    "profile.json": "67219f2df1d7678070732d4a11172cbab314ea26dee5792d3d2394d180a970ca",
    "actual-agent-workflows.json": "7fe0e22399981eeb989412db2366b96e2951544427695bc917d59a499297cdf1",
    "distinct-agent-workflows.json": "b30bf32ccc6b9b49c7b8677b9d836956e2e66af0e6d2c196a7e96e24f1f47e55",
}
SOURCE_MANIFEST_HASH = "445dd6d05d7e6746d8cc575bdaec5066586904daeb0c62dd1f93129d4d86f532"
HARNESS_HASHES = {
    "eng/validation/validate-release-agent-workflows.py": "257b4f7152c6d6b5ef47f32ec8a764719b3324fd270efda6ce461908c322ae01",
    "eng/validation/verify-agent-code-artifacts.py": "8ce2f9277f7ca718ba97c41b8ffc0e6e62290dcd69bafebf828c22f9644cf656",
    "benchmarks/server_parallel_skills.py": "fd55b68ec3bcb799480edb32b795c2780430592f5bfa5c140430a8bfc4759d5b",
    "eng/validation/run-release-profile.py": "afd8d4b6366c156b014960d0ae3fbda287ae161ee7f263ac2d35ec2cfadc6470",
    "eng/validation/sample-release-telemetry.py": "5a6de6a167b8df340a7704dd6c6c19c6746d945feb3943368c5aa4a7030a7e6c",
}


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require_hash(path, expected):
    actual = sha(path)
    if actual != expected:
        raise RuntimeError(f"Pinned digest changed: {path}: {actual} != {expected}")


def managed_hashes(repo):
    return {path.name: sha(path) for path in sorted((repo / "TensorSharp.Server.Host/bin").glob("TensorSharp.*.dll"))}


def normalized_cases(report):
    return [(case["scenario"], case["trial"], {key: value for key, value in case["request"].items()
                                            if key not in ("sessionId", "think")}) for case in report["cases"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/home/zhongkaifu/tensorsharp-no-patch-20260915"))
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/c/Works/TensorSharp"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="Run only after the parent schedules an exclusive GPU lane")
    args = parser.parse_args()
    repo = args.base / "repo"
    baseline = args.base / "gemma-e4b-managed-followups/gemma4-e4b-actual-agent"
    output = args.output or args.base / "gemma-e4b-agent-thinking"
    tools_root = output / "harness"
    build_manifest = args.base / "results/gemma-e4b-managed-followups-build/source-manifest.json"
    native = repo / "TensorSharp.Server.Host/bin/libGgmlOps.so"
    for name, expected in BASELINE_HASHES.items():
        require_hash(baseline / name, expected)
    require_hash(build_manifest, SOURCE_MANIFEST_HASH)
    original = read(baseline / "profile.json")
    if not original.get("finished_at_unix") or original.get("status") not in ("passed", "failed"):
        raise RuntimeError("Baseline server campaign is incomplete")
    for name in ("actual-agent-workflows.json", "distinct-agent-workflows.json"):
        if not read(baseline / name).get("run_complete"):
            raise RuntimeError(f"Baseline suite is incomplete: {name}")
    for relative, expected in HARNESS_HASHES.items():
        require_hash(args.source_root / relative, expected)

    profile = copy.deepcopy(original["profile"])
    profile["id"] = "gemma4-e4b-actual-agent-thinking"
    profile["scope"] = "Same 30+16 cases/default cache and exact managed/native baseline; thinking=true is the request change. Quality only; retain every failed oracle."
    profile["expected_native_sha256"] = EXPECTED_NATIVE
    profile["expected_harness_sha256"] = {str(tools_root / relative): expected for relative, expected in HARNESS_HASHES.items()}
    for suite in profile["suites"]:
        suite[1] = str(tools_root / "eng/validation/validate-release-agent-workflows.py")
        suite.append("--thinking")
    artifact_store = repo / "TensorSharp.Server.Host/bin/code-artifacts"
    plan = {"profile": profile, "baseline_hashes": BASELINE_HASHES, "source_manifest_sha256": SOURCE_MANIFEST_HASH,
            "harness_hashes": HARNESS_HASHES, "expected_managed_assemblies": original["managed_assemblies"],
            "expected_native_sha256": EXPECTED_NATIVE, "output": str(output),
            "qualification": "New --thinking flag changes harness hash; original prompts and validation rules remain unchanged. Compare ordered requests excluding only sessionId/think."}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    if output.exists():
        raise RuntimeError("Refusing to overwrite any earlier replay output")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"TensorSharp.Server.Host.dll" in command:
            raise RuntimeError(f"Another server is active: PID {proc.name}")
    require_hash(native, EXPECTED_NATIVE)
    if managed_hashes(repo) != original["managed_assemblies"]:
        raise RuntimeError("Current managed binaries differ from the exact completed baseline")
    baseline_harness = repo / "eng/validation/validate-release-agent-workflows.py"
    require_hash(baseline_harness, read(baseline / "actual-agent-workflows.json")["harness_sha256"])
    output.mkdir(parents=True, exist_ok=False)
    for relative, expected in HARNESS_HASHES.items():
        target = tools_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source_root / relative, target)
        require_hash(target, expected)
    (output / "baseline").mkdir()
    for name in BASELINE_HASHES:
        shutil.copy2(baseline / name, output / "baseline" / name)
    shutil.copy2(baseline_harness, output / "baseline/validate-release-agent-workflows.py")
    shutil.copy2(build_manifest, output / "baseline/source-manifest.json")
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    profile_path = output / "thinking-profile.json"
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    command = [str(args.base / "tools-venv/bin/python"), str(tools_root / "eng/validation/run-release-profile.py"),
               "--profile", str(profile_path), "--repo", str(repo), "--dotnet", "/usr/bin/dotnet",
               "--native", str(native), "--output", str(output / "thinking"), "--ready-timeout", "600"]
    record = {"command": command, "started_at_unix": time.time(), "runner_sha256": sha(Path(__file__))}
    with (output / "driver.log").open("w") as log:
        result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, server_finished_at_unix=time.time(),
                  managed_binaries_still_match=managed_hashes(repo) == original["managed_assemblies"],
                  independent_verifier_runs=[])
    # The lifecycle's exit code can be nonzero for ordinary quality failures.
    # Always run both available reports through the independent source oracle.
    for stem in ("actual-agent-workflows", "distinct-agent-workflows"):
        workflow_path = output / "thinking" / (stem + ".json")
        if not workflow_path.exists():
            record["independent_verifier_runs"].append({"suite": stem, "status": "report-missing"})
            continue
        verify_command = [str(args.base / "tools-venv/bin/python"), str(tools_root / "eng/validation/verify-agent-code-artifacts.py"),
                          "--workflow-report", str(workflow_path), "--artifact-store", str(artifact_store),
                          "--output", str(output / "thinking" / (stem + "-independent-code.json"))]
        with (output / (stem + "-independent.log")).open("w") as log:
            verified = subprocess.run(verify_command, stdout=log, stderr=subprocess.STDOUT)
        record["independent_verifier_runs"].append({"suite": stem, "command": verify_command, "exit_code": verified.returncode})
    comparisons = {}
    for stem, expected_cases in (("actual-agent-workflows", 30), ("distinct-agent-workflows", 16)):
        if not (output / "thinking" / (stem + ".json")).exists() or not (output / "thinking" / (stem + "-independent-code.json")).exists():
            comparisons[stem] = {"complete": False, "error": "Workflow or independent report is missing"}
            continue
        candidate = read(output / "thinking" / (stem + ".json"))
        original_report = read(baseline / (stem + ".json"))
        independent = read(output / "thinking" / (stem + "-independent-code.json"))
        comparisons[stem] = {"complete": candidate.get("run_complete"), "cases": len(candidate["cases"]),
                             "expected_cases": expected_cases, "ordered_requests_match_except_think_session": normalized_cases(candidate) == normalized_cases(original_report),
                             "all_requests_thinking_true": all(c["request"]["think"] is True for c in candidate["cases"]),
                             "baseline_passed": sum(c["status"] == "ok" for c in original_report["cases"]),
                             "thinking_passed": sum(c["status"] == "ok" for c in candidate["cases"]),
                             "independent_code_complete": independent.get("run_complete"),
                             "independent_code_cases": len(independent["cases"]), "expected_independent_code_cases": 10 if expected_cases == 30 else 8,
                             "independent_code_passed": sum(c["status"] == "ok" for c in independent["cases"])}
    record["comparisons"] = comparisons
    record["finished_at_unix"] = time.time()
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    valid = record["managed_binaries_still_match"] and all(c["complete"] and c["cases"] == c["expected_cases"]
                and c["ordered_requests_match_except_think_session"] and c["all_requests_thinking_true"]
                and c["independent_code_complete"] and c["independent_code_cases"] == c["expected_independent_code_cases"] for c in comparisons.values())
    return int(result.returncode != 0 or not valid or any(r.get("exit_code", 1) for r in record["independent_verifier_runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
