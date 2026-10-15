#!/usr/bin/env python3
"""Validate fixed-tile or 32/auto/auto/32 processes with unchanged model logits."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics

spec = importlib.util.spec_from_file_location("parallel_evidence", Path(__file__).with_name("compare-parallel-runs.py"))
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)
require = evidence.require
FLAG = "TS_GGML_Q8_PREFILL_TILE"


def compare(paths, automatic=False, vector_arm="parallel"):
    result = {"ComparableAndBitwiseEqual": False, "Errors": [], "Measurements": {}, "Executions": [],
        "Qualification": "Balanced fresh processes, unchanged complete logits and successful exits. Not an independent arithmetic/semantic oracle or proof of optimal throughput. Clock, thermal and concurrent-work controls remain external evidence."}
    try:
        count = 4 if automatic else 6
        require(len(paths) == count and len({str(p.resolve()) for p in paths}) == count, f"Require {count} distinct execution records")
        require(vector_arm in ("serial", "parallel"), "Unknown vector arithmetic")
        # Vector arithmetic is fixed and small-batch arithmetic off in every arm.
        # Reuse the strict exit, lifetime, capture, geometry and denominator checks.
        runs = [evidence.load(path, vector_arm) for path in paths]
        runs.sort(key=lambda run: evidence.utc(run["execution"]["started_utc"]))
        require(len({r["execution"]["pid"] for r in runs}) == count, "Process IDs must be distinct")
        require(all(evidence.utc(a["execution"]["finished_utc"]) <= evidence.utc(b["execution"]["started_utc"])
                    for a, b in zip(runs, runs[1:])), "Processes overlap")
        order = ["32", None, None, "32"] if automatic else ["32", "64", "128", "128", "64", "32"]
        require([r["report"]["Environment"].get(FLAG) for r in runs] == order, "Unexpected tile order")
        baseline = runs[0]["report"]
        keys = ("ModelSha256", "ModelBytes", "ModelsAssemblySha256", "ProbeAssemblySha256", "ManagedAssembliesSha256",
                "ModelGeometry", "Prompt", "Mode", "Generation", "Teacher", "Steps", "Repeats")
        options = lambda report: {k: v for k, v in report["RequestedOptions"].items() if k not in ("--output", "--teacher")}
        env = lambda report: {k: v for k, v in report["Environment"].items() if k != FLAG}
        history = lambda report: [(r["Generated"], r["Consumed"], r["LogitsSha256"]) for r in report["Records"]]
        require(len({r["LogitsSha256"] for r in baseline["Records"]}) == 1, "Same requests are not deterministic")
        for run in runs:
            report = run["report"]
            require(all(report.get(k) == baseline.get(k) for k in keys), "Identity/geometry/conditioning mismatch")
            require(report["Native"][0]["Sha256"] == baseline["Native"][0]["Sha256"], "Native differs")
            require(env(report) == env(baseline) and options(report) == options(baseline), "Other settings differ")
            require(history(report) == history(baseline), "Complete logits or histories changed")
            tile = report["Environment"][FLAG]
            log = Path(run["execution_path"]).with_name("process.log").read_text(encoding="utf-8-sig", errors="replace")
            marker = "[q8-f32] Experimental K-ordered prefill column tile selected: "
            auto_marker = "[q8-f32] Automatic K-ordered prefill tiling: "
            require(log.count(marker) == (0 if tile in (None, "32") else 1), "Actual tile selection missing or duplicated")
            require(log.count(auto_marker) == (1 if tile is None else 0), "Automatic policy missing or active in fixed arm")
            if tile not in (None, "32"): require(marker + tile + "." in log, "Wrong native tile selected")
            result["Executions"].append({k: v for k, v in run.items() if k not in ("report", "execution")})
        tiles = ("32", "auto") if automatic else ("32", "64", "128")
        for tile in tiles:
            rows = [row for run in runs if (run["report"]["Environment"][FLAG] or "auto") == tile for row in run["report"]["Records"] if not row["Warmup"]]
            result["Measurements"][tile] = {key: {"Median": statistics.median(row[key] for row in rows),
                "Minimum": min(row[key] for row in rows), "Maximum": max(row[key] for row in rows), "Samples": len(rows)}
                for key in ("PrefillTokensPerSecond", "DecodeTokensPerSecond")}
        result["RatiosTo32"] = {tile: {key: result["Measurements"][tile][key]["Median"] / result["Measurements"]["32"][key]["Median"]
            for key in result["Measurements"][tile]} for tile in tiles[1:]}
        result["ComparableAndBitwiseEqual"] = True
    except Exception as error:
        result["Errors"].append(str(error))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executions", nargs="+", type=Path, required=True)
    parser.add_argument("--automatic", action="store_true", help="Require unset default vs explicit 32 in ABBA order")
    parser.add_argument("--vector-mode", choices=("serial", "parallel"), default="parallel")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.executions, args.automatic, args.vector_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("ComparableAndBitwiseEqual", "Errors", "Measurements")}, indent=2))
    raise SystemExit(0 if result["ComparableAndBitwiseEqual"] else 1)
