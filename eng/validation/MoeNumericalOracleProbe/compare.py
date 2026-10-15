#!/usr/bin/env python3
"""Compare identical-input MoE diagnostic outputs without treating either route as ground truth."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def metrics(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("Empty or different output lengths")
    if not all(math.isfinite(x) for x in left) or not all(math.isfinite(x) for x in right):
        raise ValueError("Nonfinite output")
    error = sum((a - b) ** 2 for a, b in zip(left, right))
    norm_left = sum(a * a for a in left)
    norm_right = sum(b * b for b in right)
    dot = sum(a * b for a, b in zip(left, right))
    return {"elements": len(left), "relative_l2_using_left_norm": math.sqrt(error / norm_left) if norm_left else None,
            "max_absolute": max(abs(a - b) for a, b in zip(left, right)),
            "cosine": dot / math.sqrt(norm_left * norm_right) if norm_left and norm_right else (1 if error == 0 else None),
            "left_l2": math.sqrt(norm_left), "right_l2": math.sqrt(norm_right),
            "error_l2": math.sqrt(error), "absolute_rms_error": math.sqrt(error / len(left)),
            "different_float_values": sum(a != b for a, b in zip(left, right))}


def load_output(directory, row, hidden):
    raw = (directory / f"n{row['Tokens']}.actual.f32").read_bytes()
    if len(raw) != row["Tokens"] * hidden * 4 or hashlib.sha256(raw).hexdigest() != row["FullOutputSha256"]:
        raise ValueError("Output length/hash differs from its report")
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def compare(left_dir, right_dir):
    left, right = read(left_dir / "report.json"), read(right_dir / "report.json")
    if left["Status"] != "diagnostic-completed" or right["Status"] != "diagnostic-completed":
        raise ValueError("Both executions must have completed")
    if left["Layer"] != right["Layer"]:
        raise ValueError("Different source layers")
    for key in ("Hidden", "FeedForward", "OriginalExperts", "LoadedExperts"):
        if left["Geometry"][key] != right["Geometry"][key]:
            raise ValueError(f"Different source geometry: {key}")
    for a, b in zip(left["Weights"], right["Weights"], strict=True):
        for key in ("Name", "Type", "K", "M", "SelectedExperts", "SelectedRawSha256"):
            if a[key] != b[key]:
                raise ValueError(f"Different weight data: {key}")
    a_rows = {row["Tokens"]: row for row in left["Cases"]}
    b_rows = {row["Tokens"]: row for row in right["Cases"]}
    common = sorted(a_rows.keys() & b_rows.keys())
    if not common:
        raise ValueError("No common token-count case")
    rows = []
    for n in common:
        a, b = a_rows[n], b_rows[n]
        for key in ("InputSha256", "OriginalExpertIds", "RoutingWeights"):
            if a[key] != b[key]:
                raise ValueError(f"N{n}: different input/routing {key}")
        row = {"tokens": n, "full_output_pairwise": metrics(
            load_output(left_dir, a, left["Geometry"]["Hidden"]),
            load_output(right_dir, b, right["Geometry"]["Hidden"])),
            "left_sampled_fp64_oracle": a["Metrics"], "right_sampled_fp64_oracle": b["Metrics"]}
        rows.append(row)
    return {"status": "comparison-completed", "numerical_pass_claimed": False,
            "left": {"path": str(left_dir), "route": left["Route"], "native": left["Native"], "geometry": left["Geometry"]},
            "right": {"path": str(right_dir), "route": right["Route"], "native": right["Native"], "geometry": right["Geometry"]},
            "cases": rows, "unmatched_left": sorted(a_rows.keys() - b_rows.keys()),
            "unmatched_right": sorted(b_rows.keys() - a_rows.keys()),
            "limitation": "Pairwise full outputs are compared with identical weight bytes/input/routes. Neither execution is "
                          "treated as exact arithmetic; FP64 oracle metrics cover only the reported output columns. "
                          "Different native expert counts are deliberate dispatch controls and remain visible."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2)
        target.write("\n")
    for row in result["cases"]:
        print(f"N={row['tokens']} pairwise {row['full_output_pairwise']}")


if __name__ == "__main__":
    main()
