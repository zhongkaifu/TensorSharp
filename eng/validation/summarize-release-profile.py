#!/usr/bin/env python3
"""Summarize recorded profile evidence without changing case outcomes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time


def case_status(case):
    if "status" in case:
        return case["status"]
    if type(case.get("passed")) is bool:
        return "passed" if case["passed"] else "failed"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--brief", action="store_true", help="Print counts and first failures; retain complete summary in --output")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    profile = json.loads((args.directory / "profile.json").read_text())
    report = {"profile": profile["profile"]["id"], "status": profile["status"],
              "server_pid": profile.get("server_pid"), "error": profile.get("error"),
              "elapsed_seconds": profile.get("finished_at_unix", time.time()) - profile["started_at_unix"],
              "load_seconds": profile.get("load_seconds"), "loaded_native_libraries": profile.get("loaded_native_libraries"),
              "completed_suite_count": sum("exit_code" in item for item in profile["suites"]),
              "launched_suite_count": len(profile["suites"]), "suites": {}}
    qualification_path = args.directory / "execution-qualification.json"
    qualification = json.loads(qualification_path.read_text()) if qualification_path.exists() else {}
    report["execution_qualification"] = qualification
    for path in sorted(args.directory.glob("*.json")):
        try:
            document = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        cases = document.get("cases")
        if not isinstance(cases, list):
            continue
        entry = {"run_complete": document.get("run_complete"), "recorded_cases": len(cases),
                 "passed": sum(case_status(item) in ("ok", "pass", "passed") for item in cases),
                 "failed": [{"tag": item.get("tag", item.get("id", item.get("kind"))), "scenario": item.get("scenario"),
                             "status": case_status(item), "detail": item.get("detail", item.get("error", item.get("exception")))}
                            for item in cases if case_status(item) not in ("ok", "pass", "passed")],
                 "measured_medians": {}}
        entry["qualification"] = qualification.get("reports", {}).get(path.name, "No additional qualification annotation")
        metrics = defaultdict(lambda: defaultdict(list))
        for item in cases:
            group = f"{item.get('scenario', 'unknown')}-c{item.get('concurrency', 1)}"
            for turn in item.get("turns", [{"metrics": item.get("metrics", {})}]):
                for name, value in turn.get("metrics", {}).items():
                    if name in ("ttft_ms", "prefill_tps", "decode_tps", "total_wall_ms", "prompt_tokens", "completion_tokens") and isinstance(value, (int, float)):
                        metrics[group][name].append(value)
        for group, fields in metrics.items():
            entry["measured_medians"][group] = {key: {"median": statistics.median(values), "n": len(values)}
                                               for key, values in fields.items()}
        report["suites"][path.name] = entry
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.quiet:
        return
    if args.brief:
        suites = report["suites"]
        report = {key: value for key, value in report.items() if key != "suites"}
        report["suites"] = {name: {"run_complete": item["run_complete"], "recorded_cases": item["recorded_cases"],
                                  "passed": item["passed"], "failure_count": len(item["failed"]),
                                  "first_failure": next(iter(item["failed"]), None), "qualification": item["qualification"]}
                            for name, item in suites.items()}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
