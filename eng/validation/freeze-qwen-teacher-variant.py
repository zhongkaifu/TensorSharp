#!/usr/bin/env python3
"""Copy a prepared teacher fixture without rebuilding its managed binaries.

Preparation only: never starts the fixture or a GPU process. The destination
must be new and inside an explicitly supplied validation cache root. Partial
copies are preserved on failure rather than deleting evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(baseline: Path, native: Path, native_sha256: str, destination: Path,
            cache_root: Path, environment: dict[str, str] | None = None, steps: int | None = None) -> dict:
    baseline, native, destination, cache_root = (p.resolve() for p in (baseline, native, destination, cache_root))
    expected = native_sha256.lower()
    if not re.fullmatch(r"[a-f0-9]{64}", expected) or sha(native) != expected:
        raise ValueError("Candidate native SHA256 does not match")
    if destination == cache_root or not destination.is_relative_to(cache_root):
        raise ValueError("Destination must be below the explicit cache root")
    if destination == baseline or baseline.is_relative_to(destination) or destination.is_relative_to(baseline):
        raise ValueError("Baseline and destination must be disjoint")
    if destination.exists():
        raise FileExistsError("Destination already exists; previous evidence is never overwritten")
    app = baseline / "bin/Release/net10.0"
    if not (app / "InferenceWeb.Tests.dll").is_file() or not (app / "GgmlOps.dll").is_file():
        raise ValueError("Baseline lacks the frozen teacher app or native library")
    plan = json.loads((baseline / "plan.json").read_text(encoding="utf-8"))
    if not isinstance(plan.get("environment"), dict):
        raise ValueError("Baseline has no explicit environment")
    overrides = dict(environment or {})
    plan_changes = {}
    if steps is not None:
        if type(steps) is not int or steps < 8:
            raise ValueError("The teacher fixture requires at least eight steps")
        key = "TS_TEST_QWEN4EXP_BATCH_STEPS"
        if key in overrides and overrides[key] != str(steps):
            raise ValueError("Step count and environment override disagree")
        overrides[key] = str(steps)
        plan_changes["steps"] = {"before": plan.get("steps"), "after": steps}
        plan["steps"] = steps
        plan["scope"] = str(plan.get("scope", "")) + f" Explicit variant override: requested decode steps={steps}; other geometry and fixture remain unchanged."
    if not all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) and isinstance(value, str)
               for key, value in overrides.items()):
        raise ValueError("Invalid environment override")
    before = {str(p.relative_to(app)): sha(p) for p in app.rglob("*") if p.is_file()}
    destination.mkdir(parents=True, exist_ok=False)
    try:
        target = destination / "bin/Release/net10.0"
        shutil.copytree(app, target)
        for name in ("run-once.py", "sources.json", "references.json"):
            if (baseline / name).is_file():
                shutil.copy2(baseline / name, destination / name)
        if not (destination / "run-once.py").is_file():
            raise ValueError("Baseline has no runner")
        shutil.copy2(native, target / "GgmlOps.dll")
        actual = {str(p.relative_to(target)): sha(p) for p in target.rglob("*") if p.is_file()}
        after = {str(p.relative_to(app)): sha(p) for p in app.rglob("*") if p.is_file()}
        if after != before or actual.keys() != before.keys():
            raise ValueError("Baseline changed or copied file set differs")
        for name, old_sha in before.items():
            if actual[name] != (expected if name == "GgmlOps.dll" else old_sha):
                raise ValueError(f"Unexpected copied-file difference: {name}")
        plan["native_sha256"] = expected
        plan["status"] = "prepared_not_executed"
        deltas = {key: {"before": plan["environment"].get(key), "after": value}
                  for key, value in overrides.items() if plan["environment"].get(key) != value}
        plan["environment"].update(overrides)
        (destination / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        manifest = {"status": "prepared_not_executed", "baseline": str(baseline), "destination": str(destination),
                    "native_source": str(native), "native_sha256": expected,
                    "baseline_plan_sha256": sha(baseline / "plan.json"), "plan_sha256": sha(destination / "plan.json"),
                    "launcher_sha256": sha(destination / "run-once.py"), "environment_changes": deltas,
                    "plan_changes": plan_changes,
                    "all_non_native_files_equal": True, "files": actual,
                    "scope": "Frozen managed teacher app; native and explicitly recorded environment changes only. No execution or acceptance claim."}
        (destination / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
    except BaseException as error:
        (destination / "preparation-failure.json").write_text(json.dumps({"error": repr(error), "baseline": str(baseline)}) + "\n", encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--set-environment", action="append", default=[])
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    environment = {}
    for pair in args.set_environment:
        if "=" not in pair:
            parser.error("Environment overrides require NAME=value")
        key, value = pair.split("=", 1)
        if key in environment:
            parser.error(f"Repeated environment key: {key}")
        environment[key] = value
    report = prepare(args.baseline, args.native, args.native_sha256, args.destination, args.cache_root, environment, args.steps)
    print(json.dumps({"status": report["status"], "destination": report["destination"],
                      "files": len(report["files"]), "environment_changes": report["environment_changes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
