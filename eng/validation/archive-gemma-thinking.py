#!/usr/bin/env python3
"""Archive the completed Gemma thinking comparison without running workloads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.parse


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def ordered(report):
    return [(c["scenario"], c["trial"], {k: v for k, v in c["request"].items()
            if k not in ("sessionId", "think")}) for c in report["cases"]]


def suite(report):
    by_scenario = {}
    for case in report["cases"]:
        counts = by_scenario.setdefault(case["scenario"], {"cases": 0, "passed": 0})
        counts["cases"] += 1
        counts["passed"] += case["status"] == "ok"
    return {"complete": report["run_complete"], "cases": len(report["cases"]),
            "passed": sum(c["status"] == "ok" for c in report["cases"]),
            "by_scenario": by_scenario,
            "failures": [{k: c[k] for k in ("scenario", "trial", "status", "detail", "source_sha256", "artifact_url") if k in c}
                         for c in report["cases"] if c["status"] != "ok"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsl-base", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    campaign = "gemma-e4b-agent-thinking"
    source = args.wsl_base / campaign
    run, plan = read(source / "run.json"), read(source / "plan.json")
    profile = read(source / "thinking/profile.json")
    baseline = read(source / "baseline/profile.json")
    timing = read(source / "timing-qualification.json")
    require(run.get("finished_at_unix") and profile.get("finished_at_unix"), "Campaign is incomplete")
    require(profile["status"] == "failed" and run["exit_code"] == 1, "Expected preserved failed profile")
    require(timing["performance_qualified"] is False, "This archive is quality-only")
    require(run["managed_binaries_still_match"], "Runner did not verify post-run binary identity")
    require(profile["managed_assemblies"] == baseline["managed_assemblies"], "Managed identities differ")
    require(profile["loaded_native_libraries"] == baseline["loaded_native_libraries"], "Native identities differ")
    require(profile["loaded_native_libraries_after_suites"] == profile["loaded_native_libraries"], "Native changed during run")
    require(set(profile["loaded_native_libraries"].values()) == {plan["expected_native_sha256"]}, "Native pin differs")
    for key in ("env", "extra_args"):
        require(profile["profile"][key] == baseline["profile"][key], f"Profile {key} differs")
    for relative, expected in plan["harness_hashes"].items():
        require(digest(source / "harness" / relative) == expected, f"Harness changed: {relative}")
    for name, expected in plan["baseline_hashes"].items():
        require(digest(source / "baseline" / name) == expected, f"Baseline changed: {name}")
    require(digest(source / "baseline/source-manifest.json") == plan["source_manifest_sha256"], "Source manifest pin differs")
    prior = args.destination / "gemma-e4b-managed-followups"
    require(digest(prior / "build/source-manifest.json") == plan["source_manifest_sha256"], "Archived baseline manifest differs")
    source_manifest = read(source / "baseline/source-manifest.json")
    for relative, expected in source_manifest.items():
        require(digest(prior / "sources" / relative) == expected, f"Archived source mismatch: {relative}")

    reports, comparisons = {}, {}
    for stem, expected_count, expected_independent in (("actual-agent-workflows", 30, 10), ("distinct-agent-workflows", 16, 8)):
        report = read(source / "thinking" / (stem + ".json"))
        independent = read(source / "thinking" / (stem + "-independent-code.json"))
        old = read(source / "baseline" / (stem + ".json"))
        old_independent = read(prior / "gemma4-e4b-actual-agent" / (stem + "-independent-code.json"))
        require(report["run_complete"] and independent["run_complete"], f"Suite incomplete: {stem}")
        require(len(report["cases"]) == expected_count and len(independent["cases"]) == expected_independent, f"Case count differs: {stem}")
        require(ordered(report) == ordered(old), f"Ordered requests differ: {stem}")
        require(all(c["request"].get("think") is True for c in report["cases"]), f"Thinking is not true: {stem}")
        require(independent["workflow_report_sha256"] == digest(source / "thinking" / (stem + ".json")), "Oracle report input hash differs")
        require(independent["harness_sha256"] == plan["harness_hashes"]["eng/validation/verify-agent-code-artifacts.py"], "Oracle source pin differs")
        reports[stem + ".json"], reports[stem + "-independent-code.json"] = report, independent
        failures = {(c["scenario"], c["trial"]) for c in independent["cases"] if c["status"] != "ok"}
        comparisons[stem] = {
            "ordered_requests_identical_except_think_and_session_id": True,
            "baseline_workflows": suite(old), "thinking_workflows": suite(report),
            "baseline_independent_code": suite(old_independent), "thinking_independent_code": suite(independent),
            "thinking_workflows_passing_all_applicable_checks": sum(c["status"] == "ok" and (c["scenario"], c["trial"]) not in failures for c in report["cases"]),
        }

    manifest = {"archived_at_unix": time.time(), "scope": "Completed thinking=true comparison only; no workloads executed by archiver",
                "excluded": ["Active WSL native compile and current repo binaries", "Later campaigns", "Generated Python bytecode"], "files": []}
    def copy(origin, relative, expected=None):
        before = digest(origin)
        require(expected is None or before == expected, f"Pinned source changed: {origin}")
        target = args.destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists() or digest(target) == before, f"Refusing to replace different evidence: {target}")
        shutil.copy2(origin, target)
        require(digest(target) == before == digest(origin), f"Source/copy changed: {origin}")
        manifest["files"].append({"source": str(origin), "archive": relative.as_posix(), "bytes": target.stat().st_size,
                                  "source_sha256": before, "archived_sha256": before})

    for origin in sorted(source.rglob("*")):
        if origin.is_file() and "__pycache__" not in origin.parts:
            copy(origin, Path(campaign) / origin.relative_to(source))
    copy(args.repo / "eng/validation/run-wsl-gemma-thinking.py", Path(campaign) / "run-wsl-gemma-thinking.py", run["runner_sha256"])
    artifact_store = args.wsl_base / "repo/TensorSharp.Server.Host/bin/code-artifacts"
    references = {}
    for name, report in reports.items():
        for case in report["cases"]:
            items = [item for event in case.get("events", []) for item in (event.get("files") or [])]
            if case.get("artifact_url"):
                items.append({"url": case["artifact_url"], "source_sha256": case["source_sha256"]})
            for item in items:
                url = item.get("url", "")
                decoded = urllib.parse.unquote(urllib.parse.urlsplit(url).path)
                prefix = "/api/code/artifacts/"
                require(decoded.startswith(prefix), f"Unexpected artifact URL: {url}")
                relative = Path(decoded[len(prefix):])
                require(not relative.is_absolute() and ".." not in relative.parts, "Artifact escapes store")
                references.setdefault(relative, []).append({"report": name, "scenario": case["scenario"], "trial": case["trial"], **item})
    artifact_entries = []
    for relative, refs in sorted(references.items()):
        expected = {r["source_sha256"] for r in refs if "source_sha256" in r}
        require(len(expected) <= 1, f"Conflicting artifact hashes: {relative}")
        archived = Path(campaign) / "artifacts" / relative
        copy(artifact_store / relative, archived, next(iter(expected), None))
        artifact_entries.append({"archive": archived.as_posix(), "sha256": digest(artifact_store / relative), "references": refs})
    manifest["artifacts"] = artifact_entries
    manifest["comparison_identity"] = {"managed_assemblies_identical": True, "mapped_native_libraries_identical": True,
        "extra_args_identical": True, "environment_identical": True,
        "baseline_source_snapshot": "gemma-e4b-managed-followups/sources", "source_files_hash_verified": len(source_manifest),
        "source_manifest_sha256": plan["source_manifest_sha256"], "harness_hashes": plan["harness_hashes"],
        "source_qualification": "Uses the completed baseline source snapshot and matching run-recorded binary hashes; does not inspect or snapshot the actively rebuilding WSL repository."}
    manifest["comparisons"] = comparisons
    manifest["timing_qualification"] = timing
    manifest["file_count"] = len(manifest["files"])
    manifest["total_bytes"] = sum(f["bytes"] for f in manifest["files"])
    (args.destination / "thinking-archive-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    entry = {"campaign": campaign, "profile": profile["profile"]["id"], "status": "failed", "evidence": campaign + "/thinking",
             "loaded_native_libraries": profile["loaded_native_libraries"], "managed_assemblies": profile["managed_assemblies"],
             "suites": {name: suite(report) for name, report in reports.items()}, "comparisons": comparisons,
             "comparison_identity": manifest["comparison_identity"], "performance_qualified": False,
             "qualification": "Ordinary 29/30 workflows and 9/10 independent code; distinct 16/16 workflow events but only 7/8 independent code. Distinct code_edit_run c4-i3 prints instead of returning, so only 15/16 distinct cases pass all applicable checks. Windows CPU builds overlap; no performance claim."}
    summary_path = args.destination / "summary.json"
    summary = read(summary_path)
    require(not any(i["campaign"] == campaign for i in summary), "Thinking summary already exists; review before replacing")
    summary.append(entry)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": manifest["file_count"], "bytes": manifest["total_bytes"], "artifacts": len(artifact_entries),
                      "source_files_verified": len(source_manifest), "suites": entry["suites"]}, indent=2))


if __name__ == "__main__":
    main()
