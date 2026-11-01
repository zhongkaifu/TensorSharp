#!/usr/bin/env python3
"""Summarise eng/qwen38-engine-bench.py runs: one row per engine, mode and scenario.

  python3 eng/qwen38-engine-report.py artifacts/qwen38-engines/*.jsonl \\
      --csv artifacts/qwen38-engines/summary.csv --md artifacts/qwen38-engines/summary.md

Every number is a median over the requests of all repeats of a scenario in the
LAST run of each (engine, mode, label) found in the inputs. TTFT is over every
request; decode tok/s only over requests that produced at least --min-decode-tokens
completion tokens (a three-token answer has no meaningful rate). multi_turn and
agentic_loop also report the first turn and the later turns separately, since the
later turns reuse the cached prefix. Pass rate counts scenario repeats whose
checks all passed. A scenario is eligible for performance comparison only when
all expected repeats are present and pass every quality check. Raw timings of
failed responses remain in the input JSONL, not the comparison tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import math
import random
import sys
from pathlib import Path


def med(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
    return statistics.median(values) if values else None


def percentile(values, q):
    values = sorted(v for v in values if isinstance(v, (int, float))
                    and not isinstance(v, bool) and math.isfinite(v))
    if not values:
        return None
    at = (len(values) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def repeat_median_ci(groups):
    """Cluster bootstrap the displayed pooled median by resampling whole repeats."""
    groups = [[v for v in group if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
              for group in groups]
    groups = [group for group in groups if group]
    if len(groups) < 2:
        return None, None
    rng = random.Random(1234)
    samples = [statistics.median([v for group in rng.choices(groups, k=len(groups)) for v in group])
               for _ in range(2000)]
    return percentile(samples, .025), percentile(samples, .975)


def quality_passed(case):
    return (case.get("pass") is True and case.get("verdict") == "PASS"
            and bool(case.get("checks")) and all(c.get("ok") is True for c in case["checks"])
            and bool(case.get("requests")) and not case.get("errors")
            and not any(r.get("error") for r in case.get("requests", []))
            and all(r.get("finish_reason") in ("stop", "length", "tool_calls", "function_call")
                    for r in case.get("requests", [])))


def load(paths):
    """{(engine, mode, label): (header, [cases])} keeping the last run per key."""
    runs = {}
    for path in paths:
        current = None
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") == "run":
                current = (rec["engine"], rec.get("mode", "plain"), rec.get("label", ""))
                runs[current] = (rec, [])
            elif rec.get("kind") == "case" and current:
                runs[current][1].append(rec)
    return runs


def summarise(cases, min_decode_tokens, expected_repeats=None):
    by = {}
    for c in cases:
        by.setdefault(c["scenario"], []).append(c)
    rows = {}
    for scenario, group in by.items():
        repeats = [c.get("repeat") for c in group]
        complete = (len(set(repeats)) == len(repeats) and (expected_repeats is None
                    or set(repeats) == set(range(expected_repeats))))
        eligible = complete and all(quality_passed(c) for c in group)
        passed_group = group if eligible else []
        reqs = [r for c in passed_group for r in c.get("requests", [])]
        decode_reqs = [r for r in reqs if r.get("completion_tokens_source") in ("usage", "tokenizer", "server_timings")
                       and (r.get("completion_tokens") or 0) >= min_decode_tokens]
        checks = [c for case in group for c in case.get("checks", [])]
        row = {
            "repeats": len(group),
            "expected_repeats": expected_repeats,
            "pass": sum(1 for c in group if quality_passed(c)),
            "quality_gate": "PASS" if eligible else ("INCOMPLETE" if not complete else "FAIL"),
            "checks_passed": sum(c.get("ok") is True for c in checks),
            "checks_total": len(checks),
            "performance_requests": len(reqs),
            "ttft_s": med([r.get("ttft_s") for r in reqs]),
            "ttft_p90_s": percentile([r.get("ttft_s") for r in reqs], .9),
            "decode_tok_s": med([r.get("decode_tok_s") for r in decode_reqs]),
            "prefill_tok_s": med([r.get("prefill_tok_s") for r in reqs]),
            "total_s": med([r.get("total_s") for r in reqs]),
            "prompt_tokens": med([r.get("prompt_tokens") for r in reqs]),
            "completion_tokens": med([r.get("completion_tokens") for r in reqs]),
            "errors": sum(len(c.get("errors") or []) for c in group),
        }
        for metric in ("ttft_s", "decode_tok_s", "total_s"):
            repeat_values = []
            for c in passed_group:
                selected = c.get("requests", [])
                if metric == "decode_tok_s":
                    selected = [r for r in selected if r in decode_reqs]
                repeat_values.append([r.get(metric) for r in selected])
            low, high = repeat_median_ci(repeat_values)
            row[metric + "_ci95_low"], row[metric + "_ci95_high"] = low, high
        spec = [r.get("spec") for r in reqs if r.get("spec")]
        drafted = sum(s.get("drafted") or 0 for s in spec)
        accepted = sum(s.get("accepted") or 0 for s in spec)
        steps = sum(s.get("steps") or 0 for s in spec)
        stepped = sum(r.get("completion_tokens") or 0 for r in reqs if (r.get("spec") or {}).get("steps"))
        if drafted:
            row["acceptance_rate"] = accepted / drafted
        if steps:
            row["tokens_per_step"] = stepped / steps
        if scenario in ("multi_turn", "agentic_loop"):
            turns = [c["metrics"].get("turn_ttft_s") or [] for c in passed_group]
            row["first_turn_ttft_s"] = med([t[0] for t in turns if t])
            row["later_turn_ttft_s"] = med([x for t in turns for x in t[1:]])
            row["end_to_end_s"] = med([c["metrics"].get("end_to_end_s") for c in passed_group])
        if scenario.startswith("parallel_"):
            # Event counts are not token counts, including in aggregate rates.
            valid_parallel = [c for c in passed_group if all(r.get("completion_tokens_source")
                in ("usage", "tokenizer", "server_timings") for r in c.get("requests", []))]
            row["aggregate_tok_s"] = med([c["metrics"].get("aggregate_tok_s") for c in valid_parallel])
            row["end_to_end_s"] = med([c["metrics"].get("wall_s") for c in passed_group])
        if scenario == "image":
            row["screenshot_ttft_s"] = med([c["metrics"].get("screenshot_ttft_s") for c in passed_group])
        rows[scenario] = row
    return rows


METRICS = ["ttft_s", "decode_tok_s", "prefill_tok_s", "total_s", "first_turn_ttft_s", "later_turn_ttft_s", "screenshot_ttft_s",
           "aggregate_tok_s", "acceptance_rate", "tokens_per_step", "prompt_tokens", "completion_tokens",
           "end_to_end_s", "ttft_p90_s", "ttft_s_ci95_low", "ttft_s_ci95_high",
           "decode_tok_s_ci95_low", "decode_tok_s_ci95_high", "total_s_ci95_low", "total_s_ci95_high"]


def fmt(v, metric):
    if v is None:
        return "-"
    if metric.endswith("tok_s") or metric.startswith("decode_tok_s_ci95"):
        return f"{v:.1f}"
    if metric.endswith("_s") or "_s_ci95_" in metric:
        return f"{v:.3f}"
    if metric in ("acceptance_rate", "tokens_per_step"):
        return f"{v:.2f}"
    return f"{v:.0f}" if metric.endswith("tokens") else f"{v:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path, help="long-format rows: engine, mode, label, scenario, metric, value")
    ap.add_argument("--md", type=Path, help="markdown tables (default: stdout)")
    ap.add_argument("--min-decode-tokens", type=int, default=32)
    args = ap.parse_args()
    runs = load(args.inputs)
    if not runs:
        sys.exit("no runs found")
    table = {key: summarise(cases, args.min_decode_tokens, header.get("repeats"))
             for key, (header, cases) in runs.items()}
    for key, (header, _) in runs.items():
        for scenario in header.get("scenarios", []):
            table[key].setdefault(scenario, {"repeats": 0, "expected_repeats": header.get("repeats"),
                "pass": 0, "quality_gate": "INCOMPLETE", "checks_passed": 0, "checks_total": 0})
    scenarios = []
    for rows in table.values():
        scenarios += [s for s in rows if s not in scenarios]

    out = ["Performance values are withheld unless every expected repeat passes all functional and configured "
           "baseline quality checks. Failed, incomplete and unavailable scenarios are not passing validation. "
           "SSE events are never counted as tokens. Prefill rates use client TTFT, including non-kernel overhead; "
           "decode rates are client estimates and may include speculative streaming bursts. "
           "Different checkpoints, quantization, KV formats, drafters and active lanes are separate stack configurations. "
           "No universal engine ranking follows from these tables; differences under 5% need more repetitions. "
           "The functional checks and optional small public GSM8K subset do not establish comprehensive model quality. "
           "Long-context retrieval/order tests cover specific regressions, not full reasoning quality or RoPE equivalence. "
           "Bootstrap intervals resample whole repeats and remain weak evidence with very few repetitions.\n"]
    for metric in ("ttft_s", "total_s", "decode_tok_s", "prefill_tok_s"):
        out.append(f"\n## {metric} (median)\n")
        keys = sorted(table)
        out.append("| scenario | " + " | ".join(f"{e} {m}{(' ' + l) if l else ''}" for e, m, l in keys) + " |")
        out.append("|---|" + "---:|" * len(keys))
        for s in scenarios:
            out.append(f"| {s} | " + " | ".join(fmt(table[k].get(s, {}).get(metric), metric) for k in keys) + " |")
    out.append("\n## pass rate (scenario repeats)\n")
    keys = sorted(table)
    out.append("| scenario | " + " | ".join(f"{e} {m}{(' ' + l) if l else ''}" for e, m, l in keys) + " |")
    out.append("|---|" + "---:|" * len(keys))
    for s in scenarios:
        cells = []
        for k in keys:
            row = table[k].get(s)
            cells.append(f"{row['pass']}/{row['expected_repeats'] or row['repeats']} ({row['quality_gate']}; "
                         f"checks {row['checks_passed']}/{row['checks_total']})" if row else "unavailable")
        out.append(f"| {s} | " + " | ".join(cells) + " |")
    out.append("\n## extra\n")
    for k in keys:
        for s, row in table[k].items():
            extra = {m: fmt(row.get(m), m) for m in METRICS[4:10] + METRICS[12:] if row.get(m) is not None}
            if extra:
                out.append(f"- {k[0]} {k[1]} {k[2]} {s}: " + ", ".join(f"{m}={v}" for m, v in extra.items()))
    out.append("\n## provenance\n")
    for key, (header, _) in sorted(runs.items()):
        manifest = header.get("manifest")
        out.append(f"- {' '.join(key)}: run `{header.get('run_id', 'unknown')}`, "
                   f"harness `{header.get('harness', 'unknown')}`, nonce seed `{header.get('nonce_seed', 'unknown')}`, "
                   f"tokenizer SHA256 `{header.get('tokenizer_sha256') or 'unrecorded'}`.")
        out.append("  Manifest: " + (json.dumps(manifest, ensure_ascii=False, sort_keys=True)
                                    if manifest else "unrecorded; engine/model configuration equivalence is unverified."))
        for field in ("model_artifacts", "model_configs", "manifest_file", "quality_fixture", "sampling", "long_order_quality"):
            value = header.get(field)
            if value is not None and value != []:
                out.append(f"  {field}: " + json.dumps(value, ensure_ascii=False, sort_keys=True))
        if not header.get("model_artifacts") and not manifest:
            out.append("  Raw model weight SHA256 and source revisions were not recorded; model identity is unverified.")
    text = "\n".join(out) + "\n"
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["engine", "mode", "label", "scenario", "metric", "value", "pass", "repeats",
                        "quality_gate", "checks_passed", "checks_total", "expected_repeats"])
            for (engine, mode, label), rows in sorted(table.items()):
                for s, row in rows.items():
                    status = [row["pass"], row["repeats"], row["quality_gate"], row["checks_passed"],
                              row["checks_total"], row["expected_repeats"]]
                    # Preserve failed/unavailable scenarios in machine-readable
                    # output even though their performance values are withheld.
                    w.writerow([engine, mode, label, s, "quality_gate", row["quality_gate"]] + status)
                    for metric in METRICS:
                        if row.get(metric) is not None:
                            w.writerow([engine, mode, label, s, metric, round(row[metric], 4)] + status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
