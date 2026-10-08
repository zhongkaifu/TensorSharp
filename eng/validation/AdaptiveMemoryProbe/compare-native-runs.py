#!/usr/bin/env python3
"""Compare resident probe runs differing only in their loaded native library.

Use separate binary directories and alternate control/candidate/candidate/control
processes on an otherwise idle device. Keep process exit evidence separately.
This comparator deliberately requires complete raw-logit equality, not just top-1.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.control, args.candidate)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ComparableAndBitwiseEqual"] else 1)
