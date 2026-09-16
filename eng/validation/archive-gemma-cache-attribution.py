#!/usr/bin/env python3
"""Archive completed Gemma attribution, cache probes and Windows units.

Local filesystem copies only. The active gemma-e4b-agent-cache-fixed directory
is never inspected. Earlier archives and manifests are preserved.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def trx_summary(path):
    root = ET.parse(path).getroot()
    ns = {"t": "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"}
    times = root.find("t:Times", ns)
    counters = root.find("t:ResultSummary/t:Counters", ns)
    if times is None or not times.get("finish") or counters is None:
        raise RuntimeError(f"No completed test result: {path}")
    results = root.findall("t:Results/t:UnitTestResult", ns)
    return {
        "times": dict(times.attrib), "counters": dict(counters.attrib),
        "observed_outcomes": dict(Counter(item.get("outcome") for item in results)),
        "skipped_tests": [item.get("testName") for item in results if item.get("outcome") == "NotExecuted"],
        "test_stdout": [{"test": item.get("testName"), "stdout": item.findtext("t:Output/t:StdOut", namespaces=ns)}
                        for item in results if item.find("t:Output/t:StdOut", ns) is not None],
        "failures": [{"test": item.get("testName"),
                      "message": item.findtext("t:Output/t:ErrorInfo/t:Message", namespaces=ns),
                      "stdout": item.findtext("t:Output/t:StdOut", namespaces=ns)}
                     for item in results if item.get("outcome") == "Failed"],
    }


def suite_summary(report):
    if not report.get("run_complete"):
        raise RuntimeError("Refusing to archive an incomplete workflow report")
    counts = {}
    for case in report["cases"]:
        row = counts.setdefault(case["scenario"], Counter())
        row["cases"] += 1
        row["passed"] += case.get("status") == "ok"
    return {
        "complete": True, "cases": len(report["cases"]),
        "passed": sum(case.get("status") == "ok" for case in report["cases"]),
        "by_scenario": counts,
        "failures": [{"scenario": case["scenario"], "trial": case.get("trial"),
                      "detail": case.get("detail", case.get("error"))}
                     for case in report["cases"] if case.get("status") != "ok"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsl-base", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest = {"archived_at_unix": time.time(),
                "scope": "Completed cache attribution, before/after cache probes and Windows unit followups",
                "excluded": ["gemma-e4b-agent-cache-fixed (active run; not inspected)",
                             "Native binary bytes; original process manifests retain mapped SHA256"],
                "files": []}

    def copy_file(source, relative):
        before = digest(source)
        target = args.destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != before:
            raise RuntimeError(f"Refusing to replace different archived evidence: {target}")
        shutil.copy2(source, target)
        after_source, after_copy = digest(source), digest(target)
        if len({before, after_source, after_copy}) != 1:
            raise RuntimeError(f"Source changed while copying: {source}")
        manifest["files"].append({"source": str(source), "archive": relative.as_posix(),
                                  "bytes": target.stat().st_size, "source_sha256": before,
                                  "archived_sha256": after_copy})

    campaign = "gemma-e4b-agent-cache-attribution"
    campaign_source = args.wsl_base / campaign
    profiles, reports, entries = {}, {}, []
    for label in ("checkpoints-disabled", "retained-disabled", "cache-default"):
        folder = campaign_source / label
        profile = read_json(folder / "profile.json")
        if profile.get("status") not in ("passed", "failed") or not profile.get("finished_at_unix"):
            raise RuntimeError(f"Unfinished profile: {label}")
        profiles[label] = profile
        reports[label] = {name: read_json(folder / name)
                          for name in ("actual-agent-workflows.json", "independent-code.json")}
        entries.append({"campaign": campaign, "profile": profile["profile"]["id"],
                        "status": profile["status"], "evidence": f"{campaign}/{label}",
                        "loaded_native_libraries": profile["loaded_native_libraries"],
                        "managed_assemblies": profile["managed_assemblies"],
                        "suites": {name: suite_summary(report) for name, report in reports[label].items()}})
    for source in sorted(campaign_source.rglob("*")):
        if source.is_file():
            copy_file(source, Path(campaign) / source.relative_to(campaign_source))

    baseline = profiles["cache-default"]
    def requests(report):
        return {(case["scenario"], case["trial"]):
                {key: value for key, value in case["request"].items() if key != "sessionId"}
                for case in report["cases"]}
    manifest["attribution_identity"] = {
        "managed_assemblies_identical": all(p["managed_assemblies"] == baseline["managed_assemblies"] for p in profiles.values()),
        "mapped_natives_identical": all(p["loaded_native_libraries"] == baseline["loaded_native_libraries"] for p in profiles.values()),
        "extra_args_identical": all(p["profile"]["extra_args"] == baseline["profile"]["extra_args"] for p in profiles.values()),
        "workflow_harness_sha256": {label: reports[label]["actual-agent-workflows.json"]["harness_sha256"] for label in profiles},
        "requests_identical_except_session_id": all(requests(reports[label]["actual-agent-workflows.json"]) == requests(reports["cache-default"]["actual-agent-workflows.json"]) for label in profiles),
        "environment_differences_from_default": {
            label: {key: {"profile": p["profile"]["env"].get(key), "default": baseline["profile"]["env"].get(key)}
                    for key in sorted(set(p["profile"]["env"]) | set(baseline["profile"]["env"]))
                    if p["profile"]["env"].get(key) != baseline["profile"]["env"].get(key)}
            for label, p in profiles.items()},
    }

    cache_runs = {}
    for campaign, trx_name in (("gemma-cache-regressions-before", "gemma-cache-before.trx"),
                               ("gemma-cache-residency-before-retry", "gemma-residency-before.trx"),
                               ("gemma-cache-regressions-after", "gemma-cache-after.trx")):
        folder = args.wsl_base / "results" / campaign
        run = read_json(folder / "run.json")
        if not run.get("finished_at_unix") or run.get("exit_code") not in (0, 1):
            raise RuntimeError(f"Unfinished test run: {campaign}")
        cache_runs[campaign] = run
        result = trx_summary(folder / trx_name)
        for name in ("run.json", "tests.log", "build.log", trx_name):
            copy_file(folder / name, Path(campaign) / name)
        expected_sources = run["test_source_sha256"]
        if isinstance(expected_sources, str):
            expected_sources = {"InferenceWeb.Tests/Gemma4CacheResidencyTests.cs": expected_sources}
        source_records = []
        for relative, expected in {**expected_sources, **run.get("production_source_sha256", {})}.items():
            origins = [args.repo / relative, args.wsl_base / "repo" / relative]
            origin = next((p for p in origins if p.is_file() and digest(p) == expected), None)
            record = {"source": relative, "recorded_sha256": expected, "matching_snapshot_available": origin is not None}
            if origin is not None:
                target = Path(campaign) / (Path("production-source") / relative if relative in run.get("production_source_sha256", {}) else Path(relative).name)
                copy_file(origin, target)
                record["archive"] = target.as_posix()
            source_records.append(record)
        entries.append({"campaign": campaign, "status": "failed" if int(result["counters"]["failed"]) else "passed",
                        "evidence": campaign, "run": run, "trx": result,
                        "qualification": "Completed real E4B CUDA cache probe; before/after status is explicit in the campaign name. Does not qualify the full HTTP agent workflow.",
                        "source_snapshots": source_records})

    after = cache_runs["gemma-cache-regressions-after"]
    before = cache_runs["gemma-cache-regressions-before"]
    residency = cache_runs["gemma-cache-residency-before-retry"]
    manifest["cache_probe_identity"] = {
        "all_mapped_native_hashes": sorted({value for run in cache_runs.values() for value in run["mapped_libraries"].values()}),
        "rewind_clone_test_sources_identical": all(after["test_source_sha256"].get(name) == value for name, value in before["test_source_sha256"].items()),
        "residency_test_source_identical": after["test_source_sha256"]["InferenceWeb.Tests/Gemma4CacheResidencyTests.cs"] == residency["test_source_sha256"],
    }

    campaign = "gemma-cache-residency-before"
    source = args.wsl_base / "results" / campaign / "build.log"
    build_text = source.read_text(encoding="utf-8-sig")
    if "error CS0104" not in build_text or "Build FAILED" not in build_text:
        raise RuntimeError("Expected completed failed residency build log")
    copy_file(source, Path(campaign) / "build.log")
    entries.append({"campaign": campaign, "status": "build-failed-tests-not-run", "evidence": campaign,
                    "qualification": "CS0104: Half ambiguous between TensorSharp.Half and System.Half; corrected before the separate failing test retry."})

    test_root = args.repo / "TestResults/ggml-no-patch-2026-09-15"
    for campaign, trx_path, log_name in (
        ("channel-tool-consumer-unit-followup", "gemma-managed-followups/gemma-channel-tool-consumers.trx", "gemma-managed-followups-test.log"),
        ("cache-rewind-scheduler-unit-followup", "gemma-cache-rewind/gemma-rewind-green.trx", "gemma-cache-rewind-test.log")):
        result = trx_summary(test_root / trx_path)
        for source in (test_root / trx_path, test_root / log_name):
            copy_file(source, Path(campaign) / source.name)
        entries.append({"campaign": campaign, "status": "unit-tests-passed-real-model-replay-pending",
                        "evidence": campaign, "trx": result,
                        "qualification": "Windows managed unit evidence only; real-model tests skipped. Does not resolve the recorded WSL CUDA cache failures."})

    summary_path = args.destination / "summary.json"
    summary = read_json(summary_path)
    added_campaigns = {entry["campaign"] for entry in entries}
    summary = [entry for entry in summary if entry["campaign"] not in added_campaigns] + entries
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.destination / "cache-attribution-archive-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(manifest["files"]), "bytes": sum(f["bytes"] for f in manifest["files"]),
                      "identity": manifest["attribution_identity"], "summary_entries_added": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
