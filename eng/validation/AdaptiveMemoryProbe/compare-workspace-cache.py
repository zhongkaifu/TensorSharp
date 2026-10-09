#!/usr/bin/env python3
"""Qualify idle CUDA workspace ABBA reuse with unchanged transfers and complete logits."""
import argparse
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("cache_evidence", Path(__file__).with_name("compare-host-cache.py"))
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def compare(paths):
    return evidence.compare(paths, workspace=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executions", nargs=4, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.executions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ComparableAndBitwiseEqual"] else 1)
