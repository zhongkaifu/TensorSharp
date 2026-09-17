#!/usr/bin/env python3
"""Summarize abba.sh runs: per-process medians, a base-vs-base noise floor, a verdict.

    abba_summary.py OUT_DIR --baseline b --candidate a --control c [--max-regression-percent 2]

Each process contributes ONE value per request and metric: the median of its measured
passes (<name>-<n>.json, or <name>-<n>.json.measureK.json with --measure-passes), since
passes in one process share runtime and device state. Arms are compared by the median
over processes. The control arm is the baseline build run again under another name; its
delta against the baseline is the noise floor of this machine and schedule. A candidate
row fails only when it regresses by more than the threshold PLUS that row's noise floor.

Token identity and workload shapes are compare.py's job; run it too.
"""

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys

METRICS = (("PrefillTps", "prefill tok/s", True), ("DecodeTps", "decode tok/s", True), ("TtftMs", "TTFT ms", False))


def load_plan(directory, names):
    """Reject partial/failed executions before calculating any performance verdict."""
    directory = Path(directory)
    lines = (directory / "runs.txt").read_text().splitlines()
    if not lines or lines[-1] != "DONE":
        raise ValueError("runner did not complete successfully (runs.txt must end with DONE)")
    starts = re.findall(r"^(.+)-(\d+) start ", "\n".join(lines), re.MULTILINE)
    ends = re.findall(r"^(.+)-(\d+) rc=(\d+) end ", "\n".join(lines), re.MULTILINE)
    failed = [f"{name}-{run}: exit {rc}" for name, run, rc in ends if int(rc)]
    if failed:
        raise ValueError("failed benchmark process: " + ", ".join(failed))
    plan = json.loads((directory / "run-plan.json").read_text())
    rounds, passes = plan["rounds"], plan["measure_passes"]
    if not isinstance(rounds, int) or not isinstance(passes, int) or rounds < 1 or passes < 1:
        raise ValueError("invalid planned rounds/pass count")
    if not set(names) <= set(plan["arms"]):
        raise ValueError("requested arm is absent from the run plan")
    expected = {(name, str(i)) for name in plan["arms"] for i in range(1, rounds + 1)}
    if len(starts) != len(expected) or set(starts) != expected or len(ends) != len(expected) or {(n, i) for n, i, _ in ends} != expected:
        raise ValueError("missing or duplicate process records in runs.txt")
    return rounds, passes


def load_arm(directory, name, rounds, expected_passes):
    """{process number: [rows of each measured pass]} for one arm."""
    pattern = re.compile(rf"^{re.escape(name)}-(\d+)\.json(?:\.measure(\d+)\.json)?$")
    processes = {}
    shape = None
    for path in sorted(Path(directory).iterdir()):
        match = pattern.match(path.name)
        if match:
            run, measured = int(match.group(1)), match.group(2)
            if (expected_passes > 1) != (measured is not None):
                raise ValueError(f"{path.name}: unexpected single/multi-pass result")
            rows = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"{path.name}: empty or malformed result")
            keys = [(row["Scenario"], row["Label"]) for row in rows]
            if len(set(keys)) != len(keys) or (shape is not None and set(keys) != shape):
                raise ValueError(f"{path.name}: duplicate or missing workload rows")
            shape = set(keys)
            for row in rows:
                for field, _, _ in METRICS:
                    value = float(row[field])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"{path.name}: invalid {field}")
            processes.setdefault(run, {})[int(measured or 1)] = rows
    if set(processes) != set(range(1, rounds + 1)):
        raise ValueError(f"{name}: missing or unexpected process result")
    for run, passes in processes.items():
        if set(passes) != set(range(1, expected_passes + 1)):
            raise ValueError(f"{name}-{run}: incomplete measured passes")
        if expected_passes > 1:
            series = json.loads((Path(directory) / f"{name}-{run}.json.series.json").read_text())
            if [s["Pass"] for s in series] != list(range(1, expected_passes + 1)) or any(s["Failures"] != 0 for s in series):
                raise ValueError(f"{name}-{run}: incomplete or failed pass series")
    return {run: list(passes.values()) for run, passes in processes.items()}


def arm_medians(processes):
    """{(scenario, label): {metric: median over processes of the per-process median}}."""
    per_key = {}
    for passes in processes.values():
        values = {}
        for rows in passes:
            for row in rows:
                key = (row["Scenario"], row["Label"])
                for field, _, _ in METRICS:
                    values.setdefault(key, {}).setdefault(field, []).append(float(row.get(field, 0) or 0))
        for key, fields in values.items():
            for field, vals in fields.items():
                per_key.setdefault(key, {}).setdefault(field, []).append(statistics.median(vals))
    return {key: {field: statistics.median(vals) for field, vals in fields.items()} for key, fields in per_key.items()}


def regression_percent(base, value, higher_is_better):
    """Positive when the value is worse than the base."""
    if base == 0:
        return 0.0
    change = (value - base) / base * 100.0
    return (-change if higher_is_better else change) + 0.0   # + 0.0: never print -0.00%


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--control", help="the baseline build run again under another arm name")
    parser.add_argument("--max-regression-percent", type=float, default=2.0)
    args = parser.parse_args(argv)

    arms = {}
    try:
        names = list(filter(None, (args.baseline, args.candidate, args.control)))
        rounds, passes = load_plan(args.directory, names)
        for name in names:
            processes = load_arm(args.directory, name, rounds, passes)
            arms[name] = (len(processes), arm_medians(processes))
        if any(set(rows) != set(arms[args.baseline][1]) for _, rows in arms.values()):
            raise ValueError("workload rows differ between benchmark arms")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"error: incomplete or invalid benchmark: {error}", file=sys.stderr)
        return 2

    base = arms[args.baseline][1]
    cand = arms[args.candidate][1]
    ctrl = arms[args.control][1] if args.control else None
    print("processes: " + ", ".join(f"{name}={n}" for name, (n, _) in arms.items())
          + "; values are medians over processes of each process's median pass; positive = worse")
    print("scenario / request | metric | baseline | candidate (regression) | control (noise floor) | verdict")
    failures = []
    for key in base:
        if key not in cand:
            failures.append(f"{key[0]} / {key[1]}: missing from the candidate")
            continue
        for field, label, higher in METRICS:
            b = base[key][field]
            if b == 0 and cand[key][field] == 0:
                continue
            regression = regression_percent(b, cand[key][field], higher)
            noise = abs(regression_percent(b, ctrl[key][field], higher)) if ctrl and key in ctrl else 0.0
            allowed = args.max_regression_percent + noise
            verdict = "ok" if regression <= allowed else "REGRESSED"
            if verdict != "ok":
                failures.append(f"{key[0]} / {key[1]}: {label} regressed {regression:.2f}% "
                                f"(allowed {args.max_regression_percent:.1f}% + noise {noise:.2f}%)")
            control = f"{ctrl[key][field]:.2f} ({noise:.2f}%)" if ctrl and key in ctrl else "n/a"
            print(f"{key[0]} / {key[1]} | {label} | {b:.2f} | {cand[key][field]:.2f} ({regression:+.2f}%) | "
                  f"{control} | {verdict}")
    for failure in failures:
        print("FAIL: " + failure, file=sys.stderr)
    if not failures:
        print(f"PASS: no row regressed more than {args.max_regression_percent:.1f}% beyond its noise floor.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
