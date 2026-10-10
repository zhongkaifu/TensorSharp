#!/usr/bin/env python3
"""Compare fresh-process resident/adaptive runs. Generated reports stay ignored."""
import argparse
import json
import math
import pathlib
import statistics


def compare(paths):
    reports = [json.loads(pathlib.Path(p).read_text(encoding="utf-8-sig")) for p in paths]
    if not reports or {r["Mode"] for r in reports} != {"resident", "adaptive"}:
        raise ValueError("Provide resident and adaptive reports")
    failures = []
    identities = {(r["Model"], r.get("ModelSha256"), r["ModelBytes"], r["Context"], r["ModelsAssemblySha256"],
                   tuple(n["Sha256"] for n in r["Native"])) for r in reports}
    if len(identities) != 1:
        failures.append("Reports use different model content, paths, contexts, assemblies or native libraries")
    if any(not r.get("ModelSha256") or not r["Native"] for r in reports):
        failures.append("Every run must capture checkpoint and loaded native identities")
    if any(not r["Executed"] or r.get("Error") for r in reports):
        failures.append("At least one run failed execution or lifecycle checks")
    prompt_paths = [pathlib.Path(p).parent / "prompt.json" for p in paths]
    if len({p.read_bytes() for p in prompt_paths}) != 1:
        failures.append("Prompt token IDs differ")
    rows = [(r["Mode"], x) for r in reports for x in r["Records"] if not x["Warmup"]]
    if len({(x["PromptTokens"], x["DecodeCalls"]) for _, x in rows}) != 1:
        failures.append("Measured prompt or decode lengths differ")
    for metric in ("PrefillTokensPerSecond", "DecodeTokensPerSecond"):
        if any(not math.isfinite(x[metric]) or x[metric] <= 0 for _, x in rows):
            raise ValueError(f"Invalid {metric} measurement")
    if any(not r.get("AfterDispose") or any(p["Reserved"] or p["Committed"] for p in r["AfterDispose"])
           for r in reports if r["Mode"] == "adaptive"):
        failures.append("Adaptive disposal retained charged owners")
    if len({x["LogitsSha256"] for _, x in rows}) != 1:
        failures.append("Complete raw-logit histories differ; perform numerical analysis before declaring parity")
    values = {}
    for mode in ("resident", "adaptive"):
        selected = [x for m, x in rows if m == mode]
        if not selected:
            failures.append(f"No measured samples for {mode}")
            continue
        values[mode] = {metric: {
            "Median": statistics.median(x[metric] for x in selected),
            "Minimum": min(x[metric] for x in selected),
            "Maximum": max(x[metric] for x in selected),
            "Samples": len(selected),
        } for metric in ("PrefillTokensPerSecond", "DecodeTokensPerSecond")}
    ratios = {metric: values["adaptive"][metric]["Median"] / values["resident"][metric]["Median"]
              for metric in ("PrefillTokensPerSecond", "DecodeTokensPerSecond")} if len(values) == 2 else {}
    return {"CorrectnessAndLifecyclePassed": not failures, "Failures": failures,
            "Measurements": values, "AdaptiveToResidentRatio": ratios,
            "Reports": [str(p) for p in paths],
            "Scope": "Alternating fresh-process runs with warmup excluded. Raw-logit equality establishes regression parity, not independent language quality. Report ranges and sample count; no statistical performance guarantee is inferred."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.reports)
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["CorrectnessAndLifecyclePassed"] else 1)
