#!/usr/bin/env python3
"""Archive only the completed Gemma cache controls and direct clone test.

Uses ordinary local/WSL filesystem paths. It never starts inference, runs a
build, accesses SSH, or reads the still-running cache-attribution campaign.
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


def trx_summary(path):
    root = ET.parse(path).getroot()
    ns = {"t": "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"}
    counters = root.find("t:ResultSummary/t:Counters", ns)
    results = root.findall("t:Results/t:UnitTestResult", ns)
    return {"counters": dict(counters.attrib), "observed_outcomes": dict(Counter(item.attrib.get("outcome") for item in results)),
            "skipped_tests": [item.attrib["testName"] for item in results if item.attrib.get("outcome") == "NotExecuted"],
            "stdout": [item.text for item in root.findall(".//t:StdOut", ns) if item.text]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsl-base", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = args.wsl_base / "gemma-e4b-agent-cache-controls"
    clone = args.wsl_base / "results/gemma-prefix-clone-tests"
    profiles = {}
    for label in ("cache-disabled", "cache-default"):
        document = json.loads((source / label / "profile.json").read_text())
        if document.get("status") not in ("passed", "failed") or not document.get("finished_at_unix"):
            raise RuntimeError(f"Will not archive incomplete profile {label}")
        for name in ("actual-agent-workflows.json", "independent-code.json"):
            if not json.loads((source / label / name).read_text()).get("run_complete"):
                raise RuntimeError(f"Will not archive incomplete report {label}/{name}")
        profiles[label] = document
    clone_run = json.loads((clone / "run.json").read_text())
    if clone_run.get("exit_code") != 0 or not clone_run.get("finished_at_unix"):
        raise RuntimeError("Direct clone qualification has not completed successfully")
    manifest = {"archived_at_unix": time.time(), "scope": "Completed cache controls, direct clone qualification, local unit followup only",
                "excluded": ["gemma-e4b-agent-cache-attribution (running campaign)",
                             "Native binary bytes; the original run manifest retains the actual mapped native SHA256"], "files": []}

    def copy_file(origin, relative):
        before = digest(origin)
        target = args.destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != before:
            raise RuntimeError(f"Refusing to overwrite different archived evidence: {target}")
        shutil.copy2(origin, target)
        after_source, after_copy = digest(origin), digest(target)
        if len({before, after_source, after_copy}) != 1:
            raise RuntimeError(f"Source changed while archiving or copy differs: {origin}")
        manifest["files"].append({"source": str(origin), "archive": str(relative).replace("\\", "/"),
                                  "bytes": target.stat().st_size, "source_sha256": before, "archived_sha256": after_copy})

    for path in sorted(source.rglob("*")):
        if path.is_file():
            copy_file(path, Path("gemma-e4b-agent-cache-controls") / path.relative_to(source))
    for name in ("run.json", "tests.log", "build.log", "gemma-prefix-clone.trx"):
        copy_file(clone / name, Path("gemma-prefix-clone-tests") / name)
    clone_source = args.repo / "InferenceWeb.Tests/Gemma4PrefixCloneExactnessTests.cs"
    if digest(clone_source) != clone_run["test_source_sha256"]:
        raise RuntimeError("Current clone test source differs from the source digest of the completed run")
    copy_file(clone_source, Path("gemma-prefix-clone-tests/Gemma4PrefixCloneExactnessTests.cs"))
    test_root = args.repo / "TestResults/ggml-no-patch-2026-09-15"
    for name in ("gemma-prompt-channel-streaming.log", "gemma-prompt-channel-streaming-green.log"):
        copy_file(test_root / name, Path("channel-parser-unit-followup") / name)
    for name in ("gemma-prompt-channel-streaming.trx", "gemma-prompt-channel-streaming-green.trx"):
        copy_file(test_root / "gemma-prompt-fix" / name, Path("channel-parser-unit-followup") / name)

    entries = []
    for label, profile in profiles.items():
        entry = {"campaign": "gemma-e4b-agent-cache-controls", "profile": profile["profile"]["id"],
                 "status": profile["status"], "evidence": "gemma-e4b-agent-cache-controls/" + label,
                 "loaded_native_libraries": profile["loaded_native_libraries"], "managed_assemblies": profile["managed_assemblies"],
                 "suites": {}}
        for name in ("actual-agent-workflows.json", "independent-code.json"):
            report = json.loads((source / label / name).read_text())
            cases = report["cases"]
            by_scenario = {}
            for case in cases:
                counts = by_scenario.setdefault(case["scenario"], Counter())
                counts["cases"] += 1
                counts["passed"] += case.get("status") == "ok"
            entry["suites"][name] = {"complete": report["run_complete"], "cases": len(cases),
                "passed": sum(case.get("status") == "ok" for case in cases), "by_scenario": by_scenario,
                "failures": [{"scenario": case["scenario"], "trial": case.get("trial"),
                              "detail": case.get("detail", case.get("error"))}
                             for case in cases if case.get("status") != "ok"]}
        entries.append(entry)
    entries.append({"campaign": "gemma-prefix-clone-tests", "profile": "Gemma4PrefixCloneExactnessTests",
                    "status": "passed", "evidence": "gemma-prefix-clone-tests", "qualification": "Real E4B checkpoint and CUDA native; identical continuation shapes, not a full HTTP tool workflow.",
                    "run": clone_run, "trx": trx_summary(clone / "gemma-prefix-clone.trx")})
    unit_path = test_root / "gemma-prompt-fix/gemma-prompt-channel-streaming-green.trx"
    entries.append({"campaign": "channel-parser-unit-followup", "profile": "gemma-prompt-channel-streaming-green",
                    "status": "unit-tests-passed-real-model-replay-pending", "evidence": "channel-parser-unit-followup",
                    "qualification": "Local Windows unit/parser followup only. Six real-model cases skipped; no new real-model replay is established by these results.",
                    "trx": trx_summary(unit_path)})
    summary_path = args.destination / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    names = {item["campaign"] for item in entries}
    summary = [item for item in summary if item["campaign"] not in names] + entries
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    identity = {"managed_assemblies_identical": profiles["cache-disabled"]["managed_assemblies"] == profiles["cache-default"]["managed_assemblies"],
                "mapped_natives_identical": profiles["cache-disabled"]["loaded_native_libraries"] == profiles["cache-default"]["loaded_native_libraries"]}
    workflow_reports = {label: json.loads((source / label / "actual-agent-workflows.json").read_text()) for label in profiles}
    identity["workflow_harness_sha256"] = {label: report["harness_sha256"] for label, report in workflow_reports.items()}
    def requests_by_case(report):
        return {(case["scenario"], case["trial"]): {key: value for key, value in case["request"].items() if key != "sessionId"}
                for case in report["cases"]}
    identity["requests_identical_except_session_id"] = requests_by_case(workflow_reports["cache-disabled"]) == requests_by_case(workflow_reports["cache-default"])
    left, right = profiles["cache-disabled"]["profile"], profiles["cache-default"]["profile"]
    identity["environment_differences"] = {key: {"cache_disabled": left["env"].get(key), "cache_default": right["env"].get(key)}
        for key in sorted(set(left["env"]) | set(right["env"])) if left["env"].get(key) != right["env"].get(key)}
    identity["extra_args_identical"] = left["extra_args"] == right["extra_args"]
    manifest["control_identity"] = identity
    (args.destination / "archive-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"archived_files": len(manifest["files"]), "bytes": sum(item["bytes"] for item in manifest["files"]),
                      "identity": identity, "summary_entries_added": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
