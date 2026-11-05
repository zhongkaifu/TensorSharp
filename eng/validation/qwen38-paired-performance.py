#!/usr/bin/env python3
"""Quality-gated paired descriptive ratios; never infer significance from three repeats.

  python3 eng/validation/qwen38-paired-performance.py --baseline BEFORE.jsonl \\
      --candidate AFTER.jsonl --output docs/validation/RUN/paired-performance.json

Only same-request pairs from scenarios passing EVERY expected repeat on both
sides are used. Repeats are resampled as paired clusters. At <=5 repeats the
bootstrap distribution is enumerated exactly; otherwise 2000 draws are used.
Sequential runs are not randomized experimental pairs: these conditional 95%
intervals do not capture thermal/load/order drift or establish significance.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import statistics


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("qwen38_report_metrics", ROOT / "eng" / "qwen38-engine-report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_last(path):
    raw = Path(path).read_bytes()
    header, cases = None, []
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("kind") == "run":
            header, cases = value, []
        elif value.get("kind") == "case" and header:
            cases.append(value)
    if not header:
        raise ValueError(f"no run header: {path}")
    return header, cases, hashlib.sha256(raw).hexdigest()


def complete_and_passed(cases, expected):
    return (isinstance(expected, int) and expected > 0 and len(cases) == expected
            and {case["repeat"] for case in cases} == set(range(expected))
            and all(report.quality_passed(case) for case in cases))


def paired_ratio(groups):
    groups = [(repeat, pairs) for repeat, pairs in groups if pairs]
    if not groups:
        return {"baseline_median": None, "candidate_median": None, "candidate_over_baseline": None,
                "change_percent": None, "ratio_ci95": None, "paired_observations": 0, "repeats_with_valid_pairs": []}
    before = [a for _, pairs in groups for a, b in pairs]
    after = [b for _, pairs in groups for a, b in pairs]
    a, b = statistics.median(before), statistics.median(after)
    ratio, interval = b / a, None
    n = len(groups)
    if n >= 2:
        selections = itertools.product(range(n), repeat=n) if n <= 5 else \
            (random.Random(1234 + i).choices(range(n), k=n) for i in range(2000))
        boot = []
        for selection in selections:
            pairs = [pair for index in selection for pair in groups[index][1]]
            boot.append(statistics.median(b for a, b in pairs) / statistics.median(a for a, b in pairs))
        interval = [report.percentile(boot, .025), report.percentile(boot, .975)]
    return {"baseline_median": a, "candidate_median": b, "candidate_over_baseline": ratio,
            "change_percent": (ratio - 1) * 100, "ratio_ci95": interval,
            "paired_observations": len(before), "repeats_with_valid_pairs": [r for r, _ in groups],
            "interpretation": "observed_only_less_than_5_percent_not_a_lead" if abs(ratio - 1) < .05 else
                "observed_only_requires_more_repeats_and_drift_control"}


def matched_requests(a, b, request_index=None):
    pairs = list(enumerate(zip(a["requests"], b["requests"])))
    return [(old, new) for i, (old, new) in pairs if (request_index is None or i == request_index)
            and old.get("request_sha256") and old["request_sha256"] == new.get("request_sha256")]


def workflow_seconds(case):
    m, scenario = case.get("metrics", {}), case["scenario"]
    if scenario.startswith("parallel_"):
        return m.get("wall_s")
    if finite(m.get("end_to_end_s")):
        return m["end_to_end_s"]
    if scenario in ("multi_turn", "agentic_loop") and all(finite(r.get("total_s")) for r in case["requests"]):
        return sum(r["total_s"] for r in case["requests"])
    return None


def scenario_metrics(a_by_repeat, b_by_repeat, min_tokens, request_index=None):
    names = ["ttft_s", "request_total_s", "decode_tok_s", "prefill_tok_s", "end_to_end_s", "aggregate_tok_s", "screenshot_ttft_s"]
    metrics = {}
    for name in names:
        groups = []
        for repeat in sorted(a_by_repeat):
            a, b = a_by_repeat[repeat], b_by_repeat[repeat]
            paired_requests = matched_requests(a, b, request_index)
            pairs = []
            if name in ("end_to_end_s", "aggregate_tok_s", "screenshot_ttft_s"):
                if request_index is not None or len(paired_requests) != len(a["requests"]) or len(a["requests"]) != len(b["requests"]):
                    continue
                left, right = (workflow_seconds(a), workflow_seconds(b)) if name == "end_to_end_s" else \
                    (a.get("metrics", {}).get(name), b.get("metrics", {}).get(name))
                if finite(left) and finite(right) and left > 0 and right > 0:
                    pairs = [(left, right)]
            else:
                key = "total_s" if name == "request_total_s" else name
                for old, new in paired_requests:
                    if name == "decode_tok_s" and not all((r.get("completion_tokens") or 0) >= min_tokens
                            and r.get("completion_tokens_source") in ("usage", "tokenizer", "server_timings") for r in (old, new)):
                        continue
                    left, right = old.get(key), new.get(key)
                    if finite(left) and finite(right) and left > 0 and right > 0:
                        pairs.append((left, right))
            groups.append((repeat, pairs))
        metrics[name] = paired_ratio(groups)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-decode-tokens", type=int, default=32)
    args = parser.parse_args()
    h1, c1, sha1 = read_last(args.baseline)
    h2, c2, sha2 = read_last(args.candidate)
    same_harness = h1.get("harness") == h2.get("harness") and bool(h1.get("harness_sha256")) and h1["harness_sha256"] == h2.get("harness_sha256")
    scenarios, rows = sorted({case["scenario"] for case in c1 + c2}), {}
    for scenario in scenarios:
        a, b = [case for case in c1 if case["scenario"] == scenario], [case for case in c2 if case["scenario"] == scenario]
        passed = same_harness and complete_and_passed(a, h1.get("repeats")) and complete_and_passed(b, h2.get("repeats"))
        row = {"both_quality_gates_passed": passed, "baseline_strict_pass_repeats": sum(report.quality_passed(c) for c in a),
               "candidate_strict_pass_repeats": sum(report.quality_passed(c) for c in b), "metrics": {}}
        if passed:
            aa, bb = {c["repeat"]: c for c in a}, {c["repeat"]: c for c in b}
            row["metrics"] = scenario_metrics(aa, bb, args.min_decode_tokens)
            if scenario == "long_generation":
                row["per_prompt"] = {name: scenario_metrics(aa, bb, args.min_decode_tokens, i)
                                     for i, name in enumerate(("code", "prose", "copy"))}
        else:
            row["withheld_reason"] = "failed/incomplete original quality gate or nonidentical harness; no timing comparison permitted"
        rows[scenario] = row
    evidence = {"descriptive_comparison_only": True, "same_actual_harness_sha256": same_harness,
        "baseline": {"path": str(args.baseline), "sha256": sha1, "run_id": h1.get("run_id"), "label": h1.get("label")},
        "candidate": {"path": str(args.candidate), "sha256": sha2, "run_id": h2.get("run_id"), "label": h2.get("label")},
        "ratio_direction": "candidate / baseline; lower latency is shorter, higher throughput is larger; no winner or significance inference",
        "interval_method": "95% percentile bootstrap, resampling whole paired repeat clusters; exact n^n enumeration when n<=5, otherwise 2000 draws",
        "limitations": ["Three sequential repeats do not establish experimental significance or capture thermal/load/order drift.",
                        "Intervals are conditional descriptive resampling of observed repeats, not proof of a stable improvement.",
                        "Ratios under 5% must not be called a lead. Ratios over 5% still require replication and drift controls.",
                        "Only same-request pairs with valid measurements on both sides enter each metric; denominators can differ from unpaired report tables.",
                        "Different response lengths/prose can affect operational E2E latency; narrow factual checks are not comprehensive prose quality.",
                        "TTFT and its derived prefill rate are client semantic-output observations, not GPU kernel timings."],
        "scenarios": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    for name, row in rows.items():
        output = {"scenario": name, "both_quality_gates_passed": row["both_quality_gates_passed"]}
        if row["metrics"]:
            output["changes_percent"] = {k: round(v["change_percent"], 2) for k, v in row["metrics"].items() if v["change_percent"] is not None}
        print(json.dumps(output))


if __name__ == "__main__":
    main()
