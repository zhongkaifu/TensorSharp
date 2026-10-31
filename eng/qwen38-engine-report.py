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
checks all passed.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def med(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.median(values) if values else None


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


def summarise(cases, min_decode_tokens):
    by = {}
    for c in cases:
        by.setdefault(c["scenario"], []).append(c)
    rows = {}
    for scenario, group in by.items():
        reqs = [r for c in group for r in c.get("requests", []) if not r.get("error")]
        row = {
            "repeats": len(group),
            "pass": sum(1 for c in group if c.get("pass")),
            "ttft_s": med([r.get("ttft_s") for r in reqs]),
            "decode_tok_s": med([r.get("decode_tok_s") for r in reqs
                                 if (r.get("completion_tokens") or 0) >= min_decode_tokens]),
            "prefill_tok_s": med([r.get("prefill_tok_s") for r in reqs]),
            "total_s": med([r.get("total_s") for r in reqs]),
            "prompt_tokens": med([r.get("prompt_tokens") for r in reqs]),
            "completion_tokens": med([r.get("completion_tokens") for r in reqs]),
            "errors": sum(len(c.get("errors") or []) for c in group),
        }
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
            turns = [c["metrics"].get("turn_ttft_s") or [] for c in group]
            row["first_turn_ttft_s"] = med([t[0] for t in turns if t])
            row["later_turn_ttft_s"] = med([x for t in turns for x in t[1:]])
        if scenario.startswith("parallel_"):
            row["aggregate_tok_s"] = med([c["metrics"].get("aggregate_tok_s") for c in group])
        if scenario == "image":
            row["screenshot_ttft_s"] = med([c["metrics"].get("screenshot_ttft_s") for c in group])
        rows[scenario] = row
    return rows


METRICS = ["ttft_s", "decode_tok_s", "prefill_tok_s", "total_s", "first_turn_ttft_s", "later_turn_ttft_s", "screenshot_ttft_s",
           "aggregate_tok_s", "acceptance_rate", "tokens_per_step", "prompt_tokens", "completion_tokens"]


def fmt(v, metric):
    if v is None:
        return "-"
    if metric.endswith("tok_s"):
        return f"{v:.1f}"
    if metric.endswith("_s"):
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
    table = {key: summarise(cases, args.min_decode_tokens) for key, (_, cases) in runs.items()}
    scenarios = []
    for rows in table.values():
        scenarios += [s for s in rows if s not in scenarios]

    out = []
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
            cells.append(f"{row['pass']}/{row['repeats']}" if row else "-")
        out.append(f"| {s} | " + " | ".join(cells) + " |")
    out.append("\n## extra\n")
    for k in keys:
        for s, row in table[k].items():
            extra = {m: fmt(row.get(m), m) for m in METRICS[4:10] if row.get(m) is not None}
            if extra:
                out.append(f"- {k[0]} {k[1]} {k[2]} {s}: " + ", ".join(f"{m}={v}" for m, v in extra.items()))
    text = "\n".join(out) + "\n"
    if args.md:
        args.md.write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["engine", "mode", "label", "scenario", "metric", "value", "pass", "repeats"])
            for (engine, mode, label), rows in sorted(table.items()):
                for s, row in rows.items():
                    for metric in METRICS:
                        if row.get(metric) is not None:
                            w.writerow([engine, mode, label, s, metric, round(row[metric], 4), row["pass"], row["repeats"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
