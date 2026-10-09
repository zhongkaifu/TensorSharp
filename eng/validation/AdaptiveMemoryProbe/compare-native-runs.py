#!/usr/bin/env python3
"""Compare resident probe runs differing only in their loaded native library.

Use separate binary directories and alternate control/candidate/candidate/control
processes on an otherwise idle device. --executions verifies bounded-run exit
records; the legacy --control/--candidate interface accepts reports only.
This comparator deliberately requires complete raw-logit equality, not just top-1.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics

spec = importlib.util.spec_from_file_location("native_execution_evidence", Path(__file__).with_name("compare-parallel-runs.py"))
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def compare(control, candidate):
    failures, reports, native, values = [], {}, {}, {}
    identities, prompts, geometry, logits = set(), set(), set(), set()
    for arm, paths in (("control", control), ("candidate", candidate)):
        reports[arm] = [str(path) for path in paths]
        if len(paths) < 2:
            failures.append(f"{arm}: require at least two fresh processes")
        hashes, rows = set(), []
        for path in paths:
            report = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            if report["Mode"] != "resident" or not report["Executed"] or report.get("Error"):
                failures.append(f"{arm}: failed run or non-resident mode")
            if not report.get("ModelSha256") or not report.get("Native"):
                failures.append(f"{arm}: missing checkpoint/native identity")
            identities.add((report["Model"], report.get("ModelSha256"), report["ModelBytes"],
                            report["Context"], report["ModelsAssemblySha256"]))
            hashes.add(tuple(item["Sha256"] for item in report.get("Native", [])))
            prompts.add(tuple(json.loads((Path(path).parent / "prompt.json").read_text(encoding="utf-8-sig"))))
            if not any(row["Warmup"] for row in report["Records"]):
                failures.append(f"{arm}: missing excluded warmup")
            measured = [row for row in report["Records"] if not row["Warmup"]]
            if not measured:
                failures.append(f"{arm}: no measured requests")
            rows.extend(measured)
        if len(hashes) != 1 or not next(iter(hashes), ()):
            failures.append(f"{arm}: inconsistent native libraries")
        native[arm] = [list(item) for item in sorted(hashes)]
        for row in rows:
            geometry.add((row["PromptTokens"], row["DecodeCalls"]))
            logits.add(row["LogitsSha256"])
            if not row["LogitsSha256"]:
                failures.append(f"{arm}: missing complete-logit hash")
        values[arm] = {}
        for metric in ("PrefillTokensPerSecond", "DecodeTokensPerSecond"):
            data = [row[metric] for row in rows]
            if not data or any(not math.isfinite(value) or value <= 0 for value in data):
                failures.append(f"{arm}: invalid {metric} measurements")
                continue
            values[arm][metric] = {"Median": statistics.median(data), "Minimum": min(data),
                                   "Maximum": max(data), "Samples": len(data)}
    if len(identities) != 1 or len(prompts) != 1 or len(geometry) != 1:
        failures.append("Checkpoint, managed assembly, context or exact input geometry differs")
    if len(logits) != 1:
        failures.append("Complete raw-logit histories differ")
    if native["control"] == native["candidate"]:
        failures.append("Both arms loaded the same native library")
    ratios = {metric: values["candidate"][metric]["Median"] / values["control"][metric]["Median"]
              for metric in values["control"] if metric in values["candidate"]}
    return {"ComparableAndBitwiseEqual": not failures, "Failures": failures,
            "NativeIdentities": native, "Measurements": values,
            "CandidateToControlRatio": ratios, "Reports": reports,
            "Scope": "Identical resident checkpoint/managed code/inputs; only native identity differs. "
                     "Warmup excluded; file hashing warms page cache. Full-logit equality is a regression "
                     "check, not independent language quality or a statistical performance guarantee. "
                     "Resident mode has no shared budget ledger; these reports do not prove zero native owners."}


def compare_executions(paths):
    """Stricter current-probe qualification; legacy report-only comparison remains available."""
    try:
        require = evidence.require
        require(len(paths) == 4 and len({str(p.resolve()) for p in paths}) == 4, "Require four distinct executions")
        runs = [evidence.load(path, "serial") for path in paths]
        runs.sort(key=lambda r: evidence.utc(r["execution"]["started_utc"]))
        require(len({r["execution"]["pid"] for r in runs}) == 4, "Reused process evidence")
        require(all(evidence.utc(a["execution"]["finished_utc"]) <= evidence.utc(b["execution"]["started_utc"])
                    for a, b in zip(runs, runs[1:])), "Executions overlap")
        hashes = [r["report"]["Native"][0]["Sha256"] for r in runs]
        require(hashes[0] == hashes[3] and hashes[1] == hashes[2] and hashes[0] != hashes[1], "Require native A/B/B/A")
        directories = [r["directory"] for r in runs]
        require(directories[0] == directories[3] and directories[1] == directories[2] and directories[0] != directories[1], "Use separate frozen binary directories")
        baseline = runs[0]["report"]
        keys = ("ModelSha256", "ModelBytes", "ModelsAssemblySha256", "ProbeAssemblySha256", "ManagedAssembliesSha256",
                "ModelGeometry", "Prompt", "Mode", "Generation", "Teacher", "Steps", "Repeats", "Environment")
        options = lambda r: {k: v for k, v in r["RequestedOptions"].items() if k != "--output"}
        histories = lambda r: [(v["Generated"], v["Consumed"], v["LogitsSha256"]) for v in r["Records"]]
        require(baseline["Mode"] == "resident", "This native comparison requires resident execution")
        require(len({v["LogitsSha256"] for v in baseline["Records"]}) == 1, "Identical requests are nondeterministic")
        for run in runs:
            report = run["report"]
            require(all(report.get(k) == baseline.get(k) for k in keys), "Managed identity, settings or conditioning differs")
            require(options(report) == options(baseline), "Requested settings differ")
            require(histories(report) == histories(baseline), "Complete logits or histories differ")
        result = compare([runs[i]["report_path"] for i in (0, 3)], [runs[i]["report_path"] for i in (1, 2)])
        result["Executions"] = [{k: v for k, v in r.items() if k not in ("report", "execution")} for r in runs]
        result["ExecutionQualification"] = "Completed non-overlapping A/B/B/A processes, identical complete histories/logits, frozen identities, capture disabled and successful native shutdown. Idle hardware and thermal controls remain external evidence."
        return result
    except Exception as error:
        return {"ComparableAndBitwiseEqual": False, "Failures": [str(error)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", nargs="+")
    parser.add_argument("--candidate", nargs="+")
    parser.add_argument("--executions", nargs=4, type=Path, help="Strict current-probe A/B/B/A execution records")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.executions:
        if args.control or args.candidate: parser.error("Use executions or report groups, not both")
        result = compare_executions(args.executions)
    else:
        if not args.control or not args.candidate: parser.error("Supply both report groups or four executions")
        result = compare(args.control, args.candidate)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ComparableAndBitwiseEqual"] else 1)
