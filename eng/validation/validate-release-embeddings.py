#!/usr/bin/env python3
"""Reuse embedding API checks with an explicit same-/cross-backend comparison contract."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/EmbeddingBench"))
import embedding_bench


def compare_reference(current, reference, scope):
    minimum_cosine = 0.999 if scope == "cross-backend" else 0.99999
    maximum_component_error = 0.005 if scope == "cross-backend" else None
    result = {"scope": scope, "status": "failed", "minimum_cosine": minimum_cosine,
              "maximum_component_error": maximum_component_error, "failures": [],
              "contract_source": "benchmarks/EmbeddingBench/README.md and docs/validation/embeddings-2026-09/README.md"
              if scope == "cross-backend" else "release helper strict same-backend comparison"}
    failures = result["failures"]
    for field in ("dimensions", "usage", "rankings"):
        equal = current.get(field) == reference.get(field)
        result[field + "_equal"] = equal
        if not equal:
            failures.append(f"Candidate/reference {field} differ")
    if not current.get("rankings") or not all(current["rankings"].values()) or not all(reference.get("rankings", {}).values()):
        failures.append("Candidate/reference retrieval checks did not all pass")
    vectors = current["vectors"]
    reference_vectors = reference["vectors"]
    if not vectors or len(vectors) != len(reference_vectors):
        failures.append("Empty or different reference vector count")
        return result
    cosines, errors = [], []
    for index, (left, right) in enumerate(zip(vectors, reference_vectors)):
        if (len(left) != current["dimensions"] or len(right) != reference["dimensions"]
                or len(left) != len(right) or not left
                or any(not math.isfinite(value) for value in left + right)):
            failures.append(f"Vector {index} has invalid dimensions or nonfinite values")
            continue
        if sum(value * value for value in left) == 0 or sum(value * value for value in right) == 0:
            failures.append(f"Vector {index} has zero norm")
            continue
        cosine = embedding_bench.cosine(left, right)
        error = max(abs(a - b) for a, b in zip(left, right))
        cosines.append(cosine)
        errors.append(error)
        if not math.isfinite(cosine) or cosine < minimum_cosine:
            failures.append(f"Vector {index} cosine {cosine} is below {minimum_cosine}")
        if maximum_component_error is not None and error > maximum_component_error:
            failures.append(f"Vector {index} component error {error} exceeds {maximum_component_error}")
    result.update(vector_cosines=cosines, vector_max_component_errors=errors,
                  min_cosine=min(cosines, default=None), max_component_error=max(errors, default=None))
    result["status"] = "failed" if failures else "passed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--comparison-scope", choices=("same-backend", "cross-backend"), default="same-backend",
                        help="Cross-backend uses documented cosine>=0.999 and max component error<=0.005; same-backend retains cosine>=0.99999")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--prewarm-seconds", type=float, default=10)
    parser.add_argument("--minimum-measure-seconds", type=float, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.prewarm_seconds < 0 or args.minimum_measure_seconds < 0:
        parser.error("repeats must be positive and warmup/measurement durations nonnegative")
    embedding_bench.KEEP_ALIVE = True
    report = {"model": args.model, "url": args.url, "started_at_unix": time.time(), "status": "failed",
              "comparison_scope": args.comparison_scope if args.reference else None}
    try:
        report["correctness"] = embedding_bench.correctness(args.url, args.model, True)
        report["correctness_status"] = "passed"
        started = time.monotonic()
        calls = 0
        while time.monotonic() - started < args.prewarm_seconds:
            embedding_bench.embedding(args.url, args.model, embedding_bench.scenarios()["single_short"])
            calls += 1
        report["prewarm"] = {"seconds": time.monotonic() - started, "calls": calls, "keep_alive": True}
        report["benchmarks"] = embedding_bench.benchmark(args.url, args.model, args.warmup, args.repeats,
                                                        args.minimum_measure_seconds)
        if args.reference:
            reference = json.loads(args.reference.read_text())
            if reference.get("status") != "passed":
                raise ValueError("Reference embedding run did not pass its correctness gate")
            if reference.get("model") != args.model:
                raise ValueError("Reference embedding model differs")
            report["comparison"] = compare_reference(report["correctness"], reference["correctness"], args.comparison_scope)
            report["comparison"].update(reference=str(args.reference), latency_ratios={})
            for name, metrics in report["benchmarks"].items():
                report["comparison"]["latency_ratios"][name] = metrics["median_ms"] / reference["benchmarks"][name]["median_ms"]
            if report["comparison"]["failures"]:
                raise ValueError("; ".join(report["comparison"]["failures"]))
        report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
    report["finished_at_unix"] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report.get("error", ""), flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
