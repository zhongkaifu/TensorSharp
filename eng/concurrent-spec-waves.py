#!/usr/bin/env python3
"""Where a concurrency wave's wall time goes (issue #391): each wave of a concurrent-spec-bench.py run (its records carry
the server log's byte range) cut into its batched speculative steps by how many requests each step carried, from the
server's TS_BATCHED_SPEC_LOG=1 lines ([batched-spec] n=.. draft=.. verify=.. commit=.. drafter=.. ms).

  python3 eng/concurrent-spec-waves.py bench.jsonl [--concurrency 4]

Per arm and concurrency (medians over its waves): the wall, the requests' prompt times (TTFT: the longest), the steps at each
request count (n=4, 3, 2, 1: how many and their summed ms), every step's sum, the rest (the wall less the longest prompt
and the steps: prompt rows outside TTFT, plain steps, host time between steps), the tokens and accepted tokens a step."""
import argparse
import collections
import json
import re
import statistics

STEP = re.compile(rb"\[batched-spec\] n=(\d+) window=(\d+) kept=\[([\d,]*)\] draft=([\d.]+) verify=([\d.]+) commit=([\d.]+) drafter=([\d.]+) ms")


def steps_of(path, lo, hi):
    with open(path, "rb") as f:
        f.seek(lo)
        text = f.read(max(0, hi - lo))
    out = []
    for m in STEP.finditer(text):
        n = int(m.group(1))
        kept = [int(k) for k in m.group(3).split(b",") if k]
        out.append((n, sum(float(m.group(i)) for i in range(4, 8)), sum(kept)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench")
    ap.add_argument("--concurrency", type=int, default=0, help="only waves of this many requests")
    args = ap.parse_args()
    log_of, rows = {}, collections.defaultdict(list)
    for line in open(args.bench):
        rec = json.loads(line)
        if rec.get("kind") == "run":
            log_of[rec["label"]] = rec.get("server_log")
            continue
        if rec.get("kind") != "wave" or "log_range" not in rec or not log_of.get(rec["label"]):
            continue
        if args.concurrency and rec["concurrency"] != args.concurrency:
            continue
        steps = steps_of(log_of[rec["label"]], *rec["log_range"])
        by_n = collections.defaultdict(lambda: [0, 0.0])
        for n, ms, _ in steps:
            by_n[n][0] += 1
            by_n[n][1] += ms
        wall = rec["wave_wall_s"] * 1000
        ttft = max(t or 0 for t in rec["ttft_ms"])
        stepped = sum(v[1] for v in by_n.values())
        kept = sum(k for _, _, k in steps)
        rows[(rec["label"], rec["concurrency"])].append({
            "wall": wall, "ttft": ttft, "steps": stepped, "rest": wall - ttft - stepped, "tokens": rec["tokens"],
            "by_n": {n: tuple(v) for n, v in by_n.items()}, "acc": kept / len(steps) if steps else 0,
            "evals": rec["eval_counts"], "ends": [round(e, 2) for e in rec.get("end_s", [])]})
    med = statistics.median
    print("| arm | C | waves | wall ms | TTFT ms | steps n=4 / 3 / 2 / 1 (count, ms) | steps ms | rest ms | tokens | kept a step |")
    print("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for (label, c), ws in sorted(rows.items()):
        parts = []
        for n in range(c, 0, -1):
            cnt = med([w["by_n"].get(n, (0, 0))[0] for w in ws])
            ms = med([w["by_n"].get(n, (0, 0))[1] for w in ws])
            parts.append(f"{cnt:.0f}, {ms:.0f}")
        print(f"| {label} | {c} | {len(ws)} | {med([w['wall'] for w in ws]):.0f} | {med([w['ttft'] for w in ws]):.0f} | "
              f"{' / '.join(parts)} | {med([w['steps'] for w in ws]):.0f} | {med([w['rest'] for w in ws]):.0f} | "
              f"{med([w['tokens'] for w in ws]):.0f} | {med([w['acc'] for w in ws]):.2f} |")
    for (label, c), ws in sorted(rows.items()):
        for w in ws:
            print(f"  {label} C={c}: evals {w['evals']} ends {w['ends']} s")


if __name__ == "__main__":
    main()
