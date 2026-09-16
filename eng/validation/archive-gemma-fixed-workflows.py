#!/usr/bin/env python3
"""Archive only the completed Gemma default-cache code-workflow replay."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import time

spec = importlib.util.spec_from_file_location("gemma_archive", Path(__file__).with_name("archive-gemma-cache-attribution.py"))
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsl-base", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    campaign = "gemma-e4b-agent-cache-fixed"
    source = args.wsl_base / campaign
    profile = shared.read_json(source / "cache-fixed/profile.json")
    runs = shared.read_json(source / "run-results.json")
    if profile.get("status") != "passed" or not profile.get("finished_at_unix"):
        raise RuntimeError("Code-workflow replay is not complete and passing")
    if len(runs) != 1 or runs[0].get("label") != "cache-fixed" or runs[0].get("exit_code") != 0:
        raise RuntimeError("Expected one completed cache-fixed runner")
    reports = {name: shared.read_json(source / "cache-fixed" / name)
               for name in ("actual-agent-workflows.json", "independent-code.json")}
    suites = {name: shared.suite_summary(report) for name, report in reports.items()}
    baseline_path = args.destination / "gemma-e4b-agent-cache-attribution/cache-default"
    baseline = shared.read_json(baseline_path / "profile.json")
    baseline_workflows = shared.read_json(baseline_path / "actual-agent-workflows.json")
    manifest = {"archived_at_unix": time.time(),
                "scope": "Completed default-cache code generation/edit replay only; quality evidence with concurrent Windows portable tests on the shared host",
                "excluded": ["Later/full Gemma followups; no other campaign directory inspected"], "files": []}
    def copy_file(origin, archive):
        before = shared.digest(origin)
        target = args.destination / archive
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and shared.digest(target) != before:
            raise RuntimeError(f"Refusing to replace different evidence: {target}")
        shutil.copy2(origin, target)
        after = shared.digest(target)
        if before != after or before != shared.digest(origin):
            raise RuntimeError(f"Changed source or copy: {origin}")
        manifest["files"].append({"source": str(origin), "archive": archive.as_posix(),
                                  "bytes": target.stat().st_size, "source_sha256": before, "archived_sha256": after})

    for relative in [Path(name) for name in ("cache-fixed-profile.json", "run-results.json", "cache-fixed.log")] + [
        Path("cache-fixed") / name for name in ("server.log", "independent-code.json", "telemetry.jsonl", "actual-agent-workflows.json", "suite-0.log", "profile.json")]:
        copy_file(source / relative, Path(campaign) / relative)

    driver = args.repo / "TestResults/ggml-no-patch-2026-09-15/test-wsl-gemma-cache-after.py"
    driver_archive = Path(campaign) / "test-wsl-gemma-cache-after.py"
    copy_file(driver, driver_archive)
    driver_tree = ast.parse((args.destination / driver_archive).read_text(encoding="utf-8-sig"))
    copied_production = next(ast.literal_eval(node.value) for node in driver_tree.body
                             if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "production" for t in node.targets))
    source_probe = shared.read_json(args.destination / "gemma-cache-regressions-after/run.json")
    if set(copied_production) != set(source_probe["production_source_sha256"]):
        raise RuntimeError("Driver production-copy list does not match the archived after-run manifest")

    def ordered_requests(report):
        return [(case["scenario"], case["trial"], {k: v for k, v in case["request"].items() if k != "sessionId"})
                for case in report["cases"]]
    before_env, after_env = baseline["profile"]["env"], profile["profile"]["env"]
    before_managed, after_managed = baseline["managed_assemblies"], profile["managed_assemblies"]
    identity = {
        "baseline": "gemma-e4b-agent-cache-attribution/cache-default",
        "mapped_native_libraries_identical": baseline["loaded_native_libraries"] == profile["loaded_native_libraries"],
        "workflow_harness_sha256": {"before": baseline_workflows["harness_sha256"], "after": reports["actual-agent-workflows.json"]["harness_sha256"]},
        "ordered_requests_identical_except_session_id": ordered_requests(baseline_workflows) == ordered_requests(reports["actual-agent-workflows.json"]),
        "extra_args_identical": baseline["profile"]["extra_args"] == profile["profile"]["extra_args"],
        "environments_identical": before_env == after_env,
        "cache_environment_settings": {name: {"before": before_env.get(name), "after": after_env.get(name)}
                                       for name in ("TS_SCHED_PREFIX_CACHE", "TS_PREFIX_CHECKPOINTS", "TS_RETAINED_FUSED_CACHE")},
        "changed_managed_assembly_hashes": {name: {"before": before_managed.get(name), "after": after_managed.get(name)}
                                           for name in sorted(set(before_managed) | set(after_managed)) if before_managed.get(name) != after_managed.get(name)},
        "native_source_probe_manifest": "gemma-cache-regressions-after/run.json",
        "production_copy_provenance": {"driver": driver_archive.as_posix(), "driver_sha256": shared.digest(args.destination / driver_archive),
                                       "copied_production_files": copied_production,
                                       "matches_source_probe_manifest": True},
        "qualification": "Managed cache-fix rebuild; seven managed binary hashes differ. The archived driver explicitly copies five production files, matching the cache probe source manifest. Rebuild/dependency metadata can change assembly hashes; no independent full-tree before manifest establishes source equivalence for every other file.",
    }
    cache_evidence = {}
    for label, path in (("before", baseline_path / "server.log"), ("after", source / "cache-fixed/server.log")):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        cache_evidence[label] = {}
        for key, needle in (("retained_enabled", "retained fused-cache continuation: on"), ("checkpoint_taken", "Shared-prefix checkpoint prefix:")):
            found = next(((number, line) for number, line in enumerate(lines, 1) if needle in line), None)
            if found is None:
                raise RuntimeError(f"Expected cache activity banner missing: {label}/{key}")
            cache_evidence[label][key] = {"line": found[0], "text": found[1]}
    identity["cache_activity_evidence"] = cache_evidence
    manifest["comparison_identity"] = identity
    entry = {"campaign": campaign, "profile": profile["profile"]["id"], "status": profile["status"],
             "evidence": f"{campaign}/cache-fixed", "loaded_native_libraries": profile["loaded_native_libraries"],
             "managed_assemblies": profile["managed_assemblies"], "suites": suites,
             "comparison_identity": identity,
             "qualification": "Matched code-only c1/c4 quality replay. Other Windows portable tests partly overlapped on the shared host; timings are not qualified. Does not establish full skills/shell/media replay."}
    summary_path = args.destination / "summary.json"
    summary = [item for item in shared.read_json(summary_path) if item["campaign"] != campaign] + [entry]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.destination / "cache-fixed-archive-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(manifest["files"]), "bytes": sum(f["bytes"] for f in manifest["files"]),
                      "suites": suites, "identity": identity}, indent=2))


if __name__ == "__main__":
    main()
