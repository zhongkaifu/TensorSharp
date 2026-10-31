#!/usr/bin/env python3
"""Summarize one TensorSharp.Cli benchmark suite (--bench-workload / --bench-suite) without judging quality.

A suite runs many benchmark rows in one process, with the model loaded once. Each row prints the --benchmark
lines ("inference benchmark starting", "benchmark run i/N: ...", "benchmark summary", "benchmark sampled
tokens"), framed by "bench suite workload k/K: label=..." and "... done" lines. A cold row first drops the
model's host-mapped tables from the page cache on every node and prints one "bench evict: label=... node=k ..."
line per node; each worker also logs "host-table eviction: node=k ..." in its own process log. With
--bench-json the CLI also writes a JSON summary with exact milliseconds and UTC timestamps for every run.

This tool reads the driver log (and the JSON file when given), computes each row's rates from the actual token
counts and milliseconds (the JSON's exact values when present, else the log's), and medians over the runs after
--discard-runs (a row with no more runs than that keeps all of its runs, and says so). A row fails when its run
count is wrong, a cold row's eviction is not verified on every node, or the suite did not complete.

    python summarize-bench-suite.py node0/process.log --json node0/bench-suite.json \\
        --worker-log 1=node1/process.log --discard-runs 1 --output suite-summary.json
"""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys

SCHEMA = "tensorsharp.bench-suite-summary/1"
FIELD = re.compile(r"(\w+)=(\"[^\"]*\"|\S+)")
WORKLOAD = re.compile(r"bench suite workload (\d+)/(\d+): (label=[^\r\n]*)")
DONE = re.compile(r"bench suite workload (\d+)/(\d+) done: ([^\r\n]*)")
START = re.compile(r"inference benchmark starting: prefillTokens=(\d+) decodeTokens=(\d+) runs=(\d+) chunked=(\S+) decodeMode=(\S+)")
RUN = re.compile(r"benchmark run (\d+)/(\d+): ([^\r\n]*)")
EVICT = re.compile(r"bench evict: label=(\S+) beforeRun=(\d+) (node=[^\r\n]*)")
SAMPLED = re.compile(r"benchmark sampled tokens \((untimed correctness|run1)\): prefillTopToken=(-?\d+) decode=([0-9,]*)")
SUITE_START = re.compile(r"bench suite starting: workloads=(\d+)")
SUITE_DONE = re.compile(r"bench suite complete: workloads=(\d+) wallMs=([0-9.]+)")
WORKER_EVICT = re.compile(r"host-table eviction: (node=[^\r\n]*)")


def value(text):
    text = text.strip('"').rstrip(",")
    if text in ("true", "false"):
        return text == "true"
    if text == "na":
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return None
    return int(number) if re.fullmatch(r"-?\d+", text) else number


def fields(text):
    return {key: value(raw) for key, raw in FIELD.findall(text)}


def parse_log(text):
    """Rows in log order: each with its header fields, the benchmark start, runs, evictions and sampled tokens."""
    rows, current = [], None
    suite = {"started": None, "complete": None}
    for line in text.splitlines():
        if m := SUITE_START.search(line):
            suite["started"] = int(m[1])
        elif m := SUITE_DONE.search(line):
            suite["complete"] = {"workloads": int(m[1]), "wall_ms": float(m[2])}
        elif m := DONE.search(line):
            if current is not None and current["index"] == int(m[1]):
                current["done"] = fields(m[3])
        elif m := WORKLOAD.search(line):
            current = {"index": int(m[1]), "count": int(m[2]), "header": fields(m[3]), "start": None, "runs": [],
                       "evictions": [], "sampled": None, "done": None}
            rows.append(current)
        elif current is None:
            continue
        elif m := START.search(line):
            current["start"] = {"prefill_tokens": int(m[1]), "decode_tokens": int(m[2]), "runs": int(m[3]),
                                "chunked": m[4] == "True" or m[4] == "true", "mode": m[5]}
        elif m := RUN.search(line):
            row = {"run": int(m[1]), "runs": int(m[2])}
            row.update(fields(m[3]))
            current["runs"].append(row)
        elif m := EVICT.search(line):
            entry = {"label": m[1], "before_run": int(m[2])}
            entry.update(fields(m[3]))
            current["evictions"].append(entry)
        elif m := SAMPLED.search(line):
            current["sampled"] = {"source": m[1], "prefill_top_token": int(m[2]),
                                  "decode_token_ids": [int(x) for x in m[3].split(",") if x]}
    return suite, rows


def worker_evictions(text):
    return [fields(m[1]) for m in WORKER_EVICT.finditer(text)]


def rate(tokens, ms):
    return tokens * 1000.0 / ms if tokens and ms and ms > 0 else None


def median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def summarize(log_text, suite_json=None, worker_logs=None, discard=1):
    if discard < 0:
        raise ValueError("discard must be nonnegative")
    suite, rows = parse_log(log_text)
    failures, workloads = [], []
    json_rows = {}
    if suite_json is not None:
        for w in suite_json.get("workloads", []):
            json_rows[w.get("label")] = w
        if suite_json.get("status") != "completed":
            failures.append(f"suite JSON status is {suite_json.get('status')!r}, not 'completed'")
    if suite["started"] is None:
        failures.append("no 'bench suite starting' line")
    if suite["complete"] is None:
        failures.append("no 'bench suite complete' line: the suite stopped early")
    expected = suite["started"]
    if expected is not None and len(rows) != expected:
        failures.append(f"{len(rows)} of {expected} workloads started")
    node_counts = set()
    for row in rows:
        label = row["header"].get("label")
        problems = []
        start = row["start"]
        if start is None:
            problems.append("no 'inference benchmark starting' line")
            start = {"prefill_tokens": None, "decode_tokens": None, "runs": None, "mode": None, "chunked": None}
        if row["done"] is None:
            problems.append("no 'done' line")
        runs = row["runs"]
        if start["runs"] is not None and [r["run"] for r in runs] != list(range(1, start["runs"] + 1)):
            problems.append(f"runs {[r['run'] for r in runs]} instead of 1..{start['runs']}")
        exact = json_rows.get(label)
        samples = (exact or {}).get("samples") or []
        if exact is not None and len(samples) != len(runs):
            problems.append(f"the JSON has {len(samples)} runs, the log {len(runs)}")
        measured_rows = []
        for i, r in enumerate(runs):
            s = samples[i] if i < len(samples) else {}
            prefill_ms = s.get("prefill_ms", r.get("prefillMs"))
            decode_ms = s.get("decode_ms", r.get("decodeMs"))
            measured_rows.append({
                "run": r["run"],
                "prefill_ms": prefill_ms,
                "decode_ms": decode_ms,
                "prefill_tps": rate(start["prefill_tokens"], prefill_ms),
                "decode_tps": rate(start["decode_tokens"], decode_ms),
                "logged_prefill_tps": r.get("prefillTps"),
                "logged_decode_tps": r.get("decodeTps"),
                "prefill_start_utc": s.get("prefill_start_utc"),
                "decode_start_utc": s.get("decode_start_utc"),
                "decode_end_utc": s.get("decode_end_utc"),
            })
        kept = measured_rows[discard:] if len(measured_rows) > discard else measured_rows
        evict = row["header"].get("evict", "none")
        cold_verified = None
        if evict != "none":
            cold = []
            expected_evictions = 1 if evict == "row" else (start["runs"] or 0)
            befores = sorted(set(e["before_run"] for e in row["evictions"]))
            wanted = [1] if evict == "row" else list(range(1, expected_evictions + 1))
            if befores != wanted:
                cold.append(f"evict={evict} but evictions before runs {befores} (expected {wanted})")
            per_eviction = {}
            for e in row["evictions"]:
                per_eviction.setdefault(e["before_run"], []).append(e)
            for before, nodes in sorted(per_eviction.items()):
                node_counts.add(len(nodes))
                for n in nodes:
                    if n.get("verified") is not True:
                        cold.append(f"eviction before run {before}: node {n.get('node')} verified="
                                    f"{n.get('verified')} (supported={n.get('supported')}, "
                                    f"residentAfterMiB={n.get('residentAfterMiB')})")
            done_verified = (row["done"] or {}).get("coldVerified")
            if done_verified is not True and not cold:
                cold.append(f"the row reports coldVerified={done_verified}")
            cold_verified = not cold
            problems += cold
        if start["decode_tokens"] == 0:
            decode_median = None
        else:
            decode_median = median(r["decode_tps"] for r in kept)
        workloads.append({
            "index": row["index"],
            "label": label,
            "header": row["header"],
            "prefill_tokens": start["prefill_tokens"],
            "decode_tokens": start["decode_tokens"],
            "runs": start["runs"],
            "mode": start["mode"],
            "evict": evict,
            "cold_verified": cold_verified,
            "samples": measured_rows,
            "measured_runs": [r["run"] for r in kept],
            "discard_applied": len(measured_rows) > discard,
            "median_prefill_tps": median(r["prefill_tps"] for r in kept),
            "median_decode_tps": decode_median,
            "run1_prefill_tps": measured_rows[0]["prefill_tps"] if measured_rows else None,
            "run1_decode_tps": measured_rows[0]["decode_tps"] if measured_rows else None,
            "evictions": row["evictions"],
            "sampled_tokens": row["sampled"],
            "wall_ms": (row["done"] or {}).get("wallMs"),
            "status": "failed" if problems else "measured",
            "failures": problems,
        })
        failures += [f"{label}: {p}" for p in problems]
    if len(node_counts) > 1:
        failures.append(f"evictions reported different node counts: {sorted(node_counts)}")
    workers = {}
    for node, text in (worker_logs or {}).items():
        lines = worker_evictions(text)
        workers[str(node)] = lines
        driver_side = sum(1 for w in workloads for e in w["evictions"] if str(e.get("node")) == str(node))
        if driver_side and len(lines) != driver_side:
            failures.append(f"node {node} logged {len(lines)} evictions; the driver gathered {driver_side} from it")
    return {
        "schema": SCHEMA,
        "status": "failed" if failures else "measured",
        "failures": failures,
        "discard_runs": discard,
        "suite": suite,
        "workloads": workloads,
        "worker_evictions": workers,
        "limitations": [
            "Rates come from the CLI's own stopwatches (the JSON's exact milliseconds when given); they measure the "
            "emitted decode mode, not answer quality.",
            "A cold row is cold only for the host-mapped tables the model declares, and only for the kernel page "
            "cache: a FUSE daemon's own cache, and pages other processes map, are untouched (as with an "
            "out-of-process posix_fadvise).",
            "Rows in one process share allocator and JIT state; the first row of a process can be slower.",
        ],
    }


def table(report):
    lines = [f"status: {report['status']}"]
    for w in report["workloads"]:
        p, d = w["median_prefill_tps"], w["median_decode_tps"]
        cold = "" if w["evict"] == "none" else f" cold={w['evict']} verified={w['cold_verified']}"
        lines.append(f"  {w['label']}: prefill {p:.1f} tok/s" if p is not None else f"  {w['label']}: prefill n/a")
        lines[-1] += (f", decode {d:.2f} tok/s" if d is not None else ", decode n/a")
        lines[-1] += f" (runs {w['measured_runs']}){cold} [{w['status']}]"
    for f in report["failures"]:
        lines.append("  FAIL " + f)
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", type=Path, help="The driver's process log")
    p.add_argument("--json", type=Path, help="The driver's --bench-json file")
    p.add_argument("--worker-log", action="append", default=[], metavar="NODE=PATH",
                   help="A worker's process log, to cross-check its own eviction lines (repeatable)")
    p.add_argument("--discard-runs", type=int, default=1)
    p.add_argument("--output", type=Path, help="Write the summary JSON here")
    a = p.parse_args(argv)
    workers = {}
    for item in a.worker_log:
        node, sep, path = item.partition("=")
        if not sep:
            p.error(f"--worker-log must be NODE=PATH, got {item!r}")
        workers[node] = Path(path).read_text(encoding="utf-8", errors="replace")
    suite_json = json.loads(a.json.read_text(encoding="utf-8")) if a.json else None
    report = summarize(a.log.read_text(encoding="utf-8", errors="replace"), suite_json, workers, a.discard_runs)
    if a.output:
        a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(table(report))
    return 0 if report["status"] == "measured" else 1


if __name__ == "__main__":
    sys.exit(main())
