#!/usr/bin/env python3
"""Compare AgentTurnBench JSON output and performance without extra packages."""

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys


SHAPE_FIELDS = ("Prompt", "Reused", "Fresh", "OutTokens")
METRICS = (("PrefillTps", "prefill tok/s"), ("DecodeTps", "decode tok/s"), ("TtftMs", "TTFT ms"))


def load_rows(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty JSON array of benchmark rows")
    indexed = {}
    for index, row in enumerate(rows):
        where = f"{path}: row {index + 1}"
        if not isinstance(row, dict):
            raise ValueError(f"{where}: expected an object")
        for field in ("Scenario", "Label", "Finish", "Note"):
            if not isinstance(row.get(field), str) or (field != "Note" and not row[field]):
                raise ValueError(f"{where}: missing or invalid {field}")
        key = (row["Scenario"], row["Label"])
        where = f"{path}: {key[0]} / {key[1]}"
        if key in indexed:
            raise ValueError(f"{where}: duplicate scenario/label")
        for field in SHAPE_FIELDS:
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError(f"{where}: missing or invalid {field}")
        if row["Fresh"] != row["Prompt"] - row["Reused"]:
            raise ValueError(f"{where}: Fresh must equal Prompt minus Reused")
        tokens = row.get("Tokens")
        if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError(f"{where}: missing or invalid Tokens; rerun the benchmark with token export")
        if len(tokens) != row["OutTokens"]:
            raise ValueError(f"{where}: Tokens length does not match OutTokens")
        # Concurrent aggregates may additionally retain each request's length.
        if "TokenCounts" in row:
            counts = row["TokenCounts"]
            if (not isinstance(counts, list)
                    or any(type(n) is not int or n < 0 for n in counts)
                    or (counts and sum(counts) != len(tokens))):
                raise ValueError(f"{where}: invalid TokenCounts")
        if "ArrivalOrderFixed" in row and type(row["ArrivalOrderFixed"]) is not bool:
            raise ValueError(f"{where}: invalid ArrivalOrderFixed")
        for field, _ in METRICS:
            value = row.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{where}: missing or invalid {field}")
        if "TokenTimesMs" in row:
            times = row["TokenTimesMs"]
            if (not isinstance(times, list) or len(times) != len(tokens)
                    or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)
                    or any(a > b for a, b in zip(times, times[1:]))):
                raise ValueError(f"{where}: invalid TokenTimesMs; expected one finite, ordered delivery time per token")
            total = row.get("TotalMs")
            if (type(total) not in (int, float) or not math.isfinite(total) or total <= 0
                    or (times and total < times[-1])):
                raise ValueError(f"{where}: invalid TotalMs; must be positive and at least the final token delivery time")
        if (any(reason.strip().lower() in {"error", "failed", "failure", "cancelled", "canceled"}
                for reason in row["Finish"].split("/"))
                or re.search(r"(?:^|\|\s*)(?:ERROR\b|FAIL:)", row["Note"], re.IGNORECASE)):
            raise ValueError(f"{where}: benchmark reported failure: {row['Finish']} {row['Note']}")
        indexed[key] = row
    return indexed


def unordered_concurrent(before, after):
    """Several requests whose arrival order the benchmark did not fix.

    Submissions race the engine thread, so two runs can admit a round in different
    batches (one request alone on the solo fused path, then the rest). Different
    batches take different kernels, which is only floating-point identical up to
    near-ties; greedy decoding then turns a near-tie into a different continuation.
    `--conc-gate` fixes the arrival order and makes the batches, and therefore the
    tokens, reproducible.
    """
    return (len(before.get("TokenCounts") or []) > 1
            and not (before.get("ArrivalOrderFixed") is True and after.get("ArrivalOrderFixed") is True))


def require_same_output(reference, other, path, strict_concurrent=False, notes=None):
    missing, extra = reference.keys() - other.keys(), other.keys() - reference.keys()
    if missing or extra:
        raise ValueError(f"{path}: row keys differ; missing={sorted(missing)}, extra={sorted(extra)}")
    for key, before in reference.items():
        after = other[key]
        where = f"{path}: {key[0]} / {key[1]}"
        for field in (*SHAPE_FIELDS, "Finish"):
            if before.get(field) != after.get(field):
                raise ValueError(f"{where}: {field} changed: {before.get(field)!r} -> {after.get(field)!r}")
        if before.get("TokenCounts", []) != after.get("TokenCounts", []):
            raise ValueError(f"{where}: concurrent request token counts changed")
        if before["Tokens"] != after["Tokens"]:
            first = next(i for i, (a, b) in enumerate(zip(before["Tokens"], after["Tokens"])) if a != b)
            message = f"{where}: output token {first} changed: {before['Tokens'][first]} -> {after['Tokens'][first]}"
            if strict_concurrent or not unordered_concurrent(before, after):
                raise ValueError(message)
            if notes is not None:
                notes.append(message)


def delivery_dominance(baselines, candidates, key):
    """A longer interval after an earlier first token can lower derived decode rate.

    Exempt that rate only when complete observations prove every median token
    arrival and median request completion is at least as early. Concurrent rows
    have no single request timeline and remain subject to the ordinary rate gate.
    """
    rows = [run[key] for run in (*baselines, *candidates)]
    if any("TokenTimesMs" not in row or row.get("TokenCounts") for row in rows):
        return None
    before_total = statistics.median(run[key]["TotalMs"] for run in baselines)
    after_total = statistics.median(run[key]["TotalMs"] for run in candidates)
    if after_total > before_total:
        return None
    for token in range(rows[0]["OutTokens"]):
        before = statistics.median(run[key]["TokenTimesMs"][token] for run in baselines)
        after = statistics.median(run[key]["TokenTimesMs"][token] for run in candidates)
        if after > before:
            return None
    return before_total, after_total


def compare(baselines, candidates, max_regression, token_notes=()):
    regressions = []
    exceptions = []
    print("Positive percentages mean higher throughput or longer TTFT. Repeated runs use medians.")
    print("scenario / request | prefill tok/s | decode tok/s | TTFT ms")
    for key, row in baselines[0].items():
        cells = []
        for field, label in METRICS:
            before = statistics.median(run[key][field] for run in baselines)
            after = statistics.median(run[key][field] for run in candidates)
            # No first token means no TTFT/prefill observation. Decode needs
            # another token after it; concurrent aggregates subtract one per request.
            first_tokens = len(row.get("TokenCounts") or [row["OutTokens"]])
            unavailable = (row["OutTokens"] == 0
                           or (field == "DecodeTps" and row["OutTokens"] <= first_tokens)
                           or (field == "PrefillTps" and (row["Fresh"] == 0
                               or (row.get("TokenCounts") and before == after == 0))))
            if unavailable:
                cells.append("n/a")
                continue
            if before <= 0 or after <= 0:
                raise ValueError(f"{key[0]} / {key[1]}: {label} must be positive for this workload")
            delta = 100 * (after / before - 1)
            cells.append(f"{before:.2f} -> {after:.2f} ({delta:+.2f}%)")
            regression = delta if field == "TtftMs" else -delta
            if regression > max_regression + 1e-9:
                dominance = delivery_dominance(baselines, candidates, key) if field == "DecodeTps" else None
                if dominance is not None:
                    exceptions.append(
                        f"{key[0]} / {key[1]}: decode tok/s decreased {regression:.2f}%, but all "
                        f"{row['OutTokens']} median token delivery times are no later and median TotalMs "
                        f"is {dominance[0]:.2f} -> {dominance[1]:.2f}; no end-to-end latency regression")
                else:
                    regressions.append(f"{key[0]} / {key[1]}: {label} regressed {regression:.2f}%")
        print(f"{key[0]} / {key[1]} | " + " | ".join(cells))
    for exception in exceptions:
        print(f"DECODE-RATE EXCEPTION: {exception}")
    for note in token_notes:
        print(f"TOKENS INFORMATIONAL: {note} (concurrent arrival order not fixed; rerun both with --conc-gate, "
              "or pass --require-concurrent-identity to fail on it)")
    for failure in regressions:
        print(f"FAIL: {failure}", file=sys.stderr)
    if regressions:
        return 1
    tokens = (f"identical tokens and workload shapes in {len(baselines[0])} rows" if not token_notes else
              f"identical workload shapes in {len(baselines[0])} rows, identical tokens except "
              f"{len(token_notes)} unordered concurrent comparison(s) reported above")
    if exceptions:
        print(f"PASS: {tokens}; performance checks passed "
              f"with {len(exceptions)} decode-rate exception(s) supported by complete delivery timelines.")
    else:
        print(f"PASS: {tokens}; no regression above {max_regression:g}%.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="JSON produced before the change")
    parser.add_argument("candidate", help="JSON produced after the change")
    parser.add_argument("--max-regression-percent", type=float, default=5,
                        help="maximum throughput loss or TTFT increase (default: 5)")
    parser.add_argument("--baseline-repeat", action="append", default=[], metavar="JSON",
                        help="additional baseline run; repeat this option for more runs")
    parser.add_argument("--candidate-repeat", action="append", default=[], metavar="JSON",
                        help="additional candidate run; repeat this option for more runs")
    parser.add_argument("--require-concurrent-identity", action="store_true",
                        help="fail on token differences in concurrent rows whose arrival order was not fixed "
                             "(rows benchmarked without --conc-gate); by default they are reported only")
    args = parser.parse_args()
    if not math.isfinite(args.max_regression_percent) or args.max_regression_percent < 0:
        parser.error("--max-regression-percent must be finite and nonnegative")
    try:
        baseline_paths = [args.baseline, *args.baseline_repeat]
        candidate_paths = [args.candidate, *args.candidate_repeat]
        baselines = [load_rows(path) for path in baseline_paths]
        candidates = [load_rows(path) for path in candidate_paths]
        token_notes = []
        for path, run in zip(baseline_paths[1:] + candidate_paths, baselines[1:] + candidates):
            require_same_output(baselines[0], run, path, args.require_concurrent_identity, token_notes)
        return compare(baselines, candidates, args.max_regression_percent, token_notes)
    except (OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
