#!/usr/bin/env python3
"""Compare complete matched-history TS vocabulary captures, including prefill.

The strict gate is relative L2 <= 1e-6 and equal argmax for every captured row.
This is route agreement, not proof that either route implements model semantics.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location("qwen38_teacher", Path(__file__).with_name("qwen38-llama-teacher.py"))
teacher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teacher)
evidence_spec = importlib.util.spec_from_file_location("qwen38_evidence", Path(__file__).with_name("qwen38_capture_evidence.py"))
evidence = importlib.util.module_from_spec(evidence_spec)
evidence_spec.loader.exec_module(evidence)


def load_index(path, iteration):
    index = json.loads(path.read_text(encoding="utf-8-sig"))
    if index.get("format") != "f32le":
        raise ValueError("Expected f32le captures")
    rows = [row for row in index["rows"] if row["iteration"] == iteration and not row["warmup"]]
    if not rows or rows[0]["stage"] != "prefill":
        raise ValueError("Missing initial prefill prediction")
    return Path(index["data_path"]), rows


def compare(left_path, right_path, iteration):
    left_data, left_rows = load_index(left_path, iteration)
    right_data, right_rows = load_index(right_path, iteration)
    if len(left_rows) != len(right_rows):
        raise ValueError("Capture row counts differ")
    comparisons = []
    for position, (left, right) in enumerate(zip(left_rows, right_rows)):
        if left["stage"] != right["stage"] or left["input_tokens"] != right["input_tokens"]:
            raise ValueError(f"Input history or stage differs at row {position}")
        a, b = teacher.read_row(left_data, left), teacher.read_row(right_data, right)
        if len(a) != len(b):
            raise ValueError(f"Vocabulary dimensions differ at row {position}")
        squared_error = math.fsum((x - y) ** 2 for x, y in zip(a, b))
        squared_norm = math.fsum(x * x for x in a)
        other_norm = math.fsum(y * y for y in b)
        relative = math.sqrt(squared_error) / max(math.sqrt(squared_norm), 1e-150)
        left_argmax = max(range(len(a)), key=a.__getitem__)
        right_argmax = max(range(len(b)), key=b.__getitem__)
        cosine = math.fsum(x * y for x, y in zip(a, b)) / math.sqrt(max(squared_norm * other_norm, 1e-300))
        comparisons.append({"row": position, "stage": left["stage"], "input_tokens": left["input_tokens"],
            "elements": len(a), "relative_l2": relative, "cosine": cosine,
            "max_abs_error": max(abs(x - y) for x, y in zip(a, b)), "argmax": [left_argmax, right_argmax],
            "sha256": [left["sha256"], right["sha256"]],
            "passed": relative <= 1e-6 and left_argmax == right_argmax})
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"run_complete": False, "passed": False, "rows": [],
              "gate": "Every row: relative L2 <= 1e-6 and equal argmax; left norm is the denominator.",
              "qualification": "Identical-history route agreement, not an independent quality oracle."}
    errors = []
    try:
        result["index_sha256"] = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.left, args.right)]
        result["rows"] = compare(args.left, args.right, args.iteration)
    except Exception as error:
        errors.append("Capture comparison: " + str(error))
    identities = []
    for label, index_path, report_path in (("left", args.left, args.left_report), ("right", args.right, args.right_report)):
        try:
            identities.append(evidence.validate(index_path, report_path))
        except Exception as error:
            errors.append(label + " completion/identity: " + str(error))
    result["execution_identities"] = identities
    if len(identities) == 2:
        for key in ("checkpoint", "geometry", "decode_mode", "prompt_tokens"):
            if identities[0][key] != identities[1][key]:
                errors.append("Model identity/conditioning differs: " + key)
    result["errors"] = errors
    result["run_complete"] = not errors
    result["passed"] = not errors and bool(result["rows"]) and all(row["passed"] for row in result["rows"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
