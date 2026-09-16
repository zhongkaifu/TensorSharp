#!/usr/bin/env python3
"""Prepare the exact failed candidate's ordered allocation-history reproduction.

Does not start a server. It requires the original harness digests and confirms
the deterministic reconstructed input against the recorded failed-case hash.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    harness_dir = Path(__file__).resolve().parents[2] / "benchmarks/engine_comparison"
    original = json.loads((args.failed_output / "profile.json").read_text())
    long_report = json.loads((args.failed_output / "long-context.json").read_text())
    for name, expected in long_report["harness_sha256"].items():
        actual = hashlib.sha256((harness_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Original harness digest no longer matches: {name}")
    sys.path.insert(0, str(harness_dir))
    import validate_inference as validation
    import engines
    failed = next(case for case in long_report["cases"] if case["tag"] == "long_32k-c1-r0-i0")
    spec = validation.case_spec(failed["scenario"], failed["tag"])
    initial = {**spec, "sampling": validation.SAMPLING,
               "thinking": long_report["thinking"], "stream": long_report["stream"]}
    actual_hash = validation.digest(initial)
    if actual_hash != failed["input_sha256"]:
        raise RuntimeError("Reconstructed failed input differs from its original recorded digest")
    request = {"messages": spec["messages"], "tools": spec["tools"],
               "response_format": spec["response_format"],
               "extra_body": {**validation.SAMPLING, **engines.thinking_body("tensorsharp", long_report["thinking"])},
               "max_tokens": spec["max_tokens"], "stream": long_report["stream"]}
    profile = copy.deepcopy(original["profile"])
    profile["id"] += "-launch-blocking-reproduction"
    profile["env"]["CUDA_LAUNCH_BLOCKING"] = "1"
    profile["expected_harness_sha256"] = {
        "benchmarks/engine_comparison/" + name: expected
        for name, expected in long_report["harness_sha256"].items()
    }
    profile["suites"] = profile["suites"][:4]
    long_suite = profile["suites"][-1]
    index = long_suite.index("--scenarios") + 1
    if long_suite[index] != "long_8k,long_32k,long_64k":
        raise RuntimeError("Original long suite is not in the expected exact order")
    long_suite[index] = "long_8k,long_32k"
    for suite in profile["suites"]:
        suite[suite.index("--profile") + 1] = profile["id"]
    profile["reproduction"] = {
        "original_profile": original["profile"]["id"],
        "original_native_sha256": original["profile"]["expected_native_sha256"],
        "failed_input_sha256": actual_hash,
        "purpose": "Attribute asynchronous CUDA failure with synchronous launch reporting; not a performance benchmark.",
        "changes": ["CUDA_LAUNCH_BLOCKING=1", "Stop after original 32K case", "Distinct evidence profile ID"],
        "preserved": "All prior suite ordering, inputs, warmups, concurrency, sampling, native binary, model, placement and cache settings. Concurrent completion order is inherently nondeterministic."
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, document in (
        ("launch-blocking-profile.json", profile),
        ("failed-request-reconstructed.json", {
            "provenance": "Reconstructed from the digest-verified unchanged harness, not captured after the failed HTTP call.",
            "recorded_case": failed, "verified_input_sha256": actual_hash,
            "harness_sha256": long_report["harness_sha256"], "initial_input": initial,
            "run_openai_chat_arguments": request,
        }),
    ):
        path = args.output / name
        if path.exists():
            raise FileExistsError(f"Will not overwrite reproduction evidence: {path}")
        path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
