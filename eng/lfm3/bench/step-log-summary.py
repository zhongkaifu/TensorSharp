#!/usr/bin/env python3
"""Summarize the continuous-batching executor's per-step lines (TS_SCHED_STEP_LOG=1) from TensorSharp server logs.

Each `[step] t=HH:MM:SS.fff gap=G ms=M PATH id:D1 id:P256 ...` line is one step: G is the idle time since the previous
step returned (the scheduler, streaming and whatever else the server did between steps), M the serving path's time for
the step (forward, sampling, commit), then each sequence's prefill (P) or decode (D) tokens. A pure decode step of n
rows is the steady state of n streams; its cadence G + M is the TPOT those streams see.

usage: step-log-summary.py [--min-steps K] [--json OUT] LOG...
Per log: pure decode steps grouped by their row count (count, median / p10 / p90 of gap, ms and gap + ms), and the
steps that carried prompt rows (count, median ms, prompt tokens). Groups with fewer than --min-steps steps (default 20)
are folded into one 'other' line.
"""
import argparse
import json
import re
import statistics
import sys
from collections import defaultdict

STEP = re.compile(r"\[step\] t=(\S+) gap=([0-9.]+) ms=([0-9.]+) (\S+)(.*)$")
ROW = re.compile(r"\S+:([PD])(\d+)")


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def summarize(path, min_steps):
    decode = defaultdict(list)    # rows -> [(gap, ms)]
    paths = defaultdict(int)
    prompt_steps = []             # (gap, ms, prompt tokens, decode rows)
    with open(path, errors="replace") as f:
        for line in f:
            m = STEP.search(line)
            if not m:
                continue
            gap, ms, kind = float(m.group(2)), float(m.group(3)), m.group(4)
            rows = ROW.findall(m.group(5))
            if not rows:
                continue
            paths[kind] += 1
            p_tokens = sum(int(t) for k, t in rows if k == "P")
            d_rows = sum(1 for k, _ in rows if k == "D")
            if p_tokens == 0:
                decode[d_rows].append((gap, ms))
            else:
                prompt_steps.append((gap, ms, p_tokens, d_rows))
    out = {"log": path, "paths": dict(paths), "decode": [], "prompt_steps": None}
    other = []
    for n in sorted(decode):
        v = decode[n]
        if len(v) < min_steps:
            other.extend(v)
            continue
        gaps, mss = [g for g, _ in v], [m for _, m in v]
        tot = [g + m for g, m in v]
        out["decode"].append({
            "rows": n, "steps": len(v),
            "gap_p50": statistics.median(gaps), "gap_p10": pct(gaps, 0.1), "gap_p90": pct(gaps, 0.9),
            "ms_p50": statistics.median(mss), "ms_p10": pct(mss, 0.1), "ms_p90": pct(mss, 0.9),
            "cadence_p50": statistics.median(tot), "cadence_p90": pct(tot, 0.9),
        })
    if other:
        out["decode_other_steps"] = len(other)
    if prompt_steps:
        out["prompt_steps"] = {
            "steps": len(prompt_steps),
            "ms_p50": statistics.median(p[1] for p in prompt_steps),
            "gap_p50": statistics.median(p[0] for p in prompt_steps),
            "prompt_tokens": sum(p[2] for p in prompt_steps),
            "decode_rows_riding": sum(p[3] for p in prompt_steps),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--min-steps", type=int, default=20)
    ap.add_argument("--json")
    a = ap.parse_args()
    res = [summarize(p, a.min_steps) for p in a.logs]
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    for r in res:
        print(f"\n{r['log']}: steps by path {r['paths']}")
        print("| decode rows | steps | gap p50 [p10-p90] ms | path ms p50 [p10-p90] | gap + ms p50 (p90) |")
        print("|---|---|---|---|---|")
        for d in r["decode"]:
            print(f"| {d['rows']} | {d['steps']} | {d['gap_p50']:.2f} [{d['gap_p10']:.2f}-{d['gap_p90']:.2f}] | "
                  f"{d['ms_p50']:.2f} [{d['ms_p10']:.2f}-{d['ms_p90']:.2f}] | {d['cadence_p50']:.2f} ({d['cadence_p90']:.2f}) |")
        if r.get("decode_other_steps"):
            print(f"| other | {r['decode_other_steps']} | (row counts with fewer than {a.min_steps} steps) | | |")
        p = r["prompt_steps"]
        if p:
            print(f"steps with prompt rows: {p['steps']} (ms p50 {p['ms_p50']:.1f}, gap p50 {p['gap_p50']:.2f}; "
                  f"{p['prompt_tokens']} prompt tokens, {p['decode_rows_riding']} decode rows riding along)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
