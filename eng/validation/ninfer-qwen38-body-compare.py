#!/usr/bin/env python3
"""Compare independently produced, teacher-forced full-body logits captures.

This reports numerical agreement for fixed token inputs, not model/task quality or
performance. It never treats a missing oracle, skipped position or corrupt vector
as a pass. Oracle producers may use the same small interchange schema as the TS
probe, but must identify their independent engine/build in run.json.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys

SCHEMA = "ninfer-qwen38-body-logits/1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    require(isinstance(value, dict), f"Expected object: {path}")
    return value


def read_run(path: Path) -> tuple[dict, dict]:
    run = read_json(path / "run.json")
    require(run.get("schema") == SCHEMA and run.get("complete") is True, "Incomplete/unknown capture run")
    require(isinstance(run.get("engine"), str) and bool(run["engine"]), "Producer engine is absent")
    require(type(run.get("vocabSize")) is int and type(run.get("validVocabSize")) is int
            and 0 < run["validVocabSize"] <= run["vocabSize"], "Invalid vocabulary dimensions")
    inputs = (path / "inputs.json").read_bytes()
    require(sha(inputs) == run.get("inputsSha256"), "Input manifest SHA mismatch")
    manifest = read_json(path / "inputs.json")
    require(manifest.get("schema") == "ninfer-qwen38-body-inputs/1", "Unknown token manifest")
    cases = {}
    for case in manifest["cases"]:
        require(isinstance(case["id"], str) and case["id"] not in cases, "Duplicate/invalid input case")
        require(type(case["chunkSize"]) is int and case["chunkSize"] > 0, "Invalid chunk size")
        require(bool(case["promptTokens"]) and bool(case["continuationTokens"]), "Empty token sequence")
        require(all(type(token) is int and 0 <= token < run["validVocabSize"]
                    for token in case["promptTokens"] + case["continuationTokens"]), "Invalid token ID")
        cases[case["id"]] = case
    require(bool(cases), "Empty case manifest")
    expected = set()
    for case_id, case in cases.items():
        n = len(case["promptTokens"])
        for pos in range(case["chunkSize"], n, case["chunkSize"]):
            expected.add((case_id, "prefill", pos))
        expected.add((case_id, "prefill", n))
        for offset in range(len(case["continuationTokens"])):
            expected.add((case_id, "decode", n + offset + 1))
    captures = {}
    for cap in run["captures"]:
        if cap["pass"] != 0 or cap["kind"] == "clone":
            continue
        key = (cap["caseId"], cap["kind"], cap["position"])
        require(key in expected and key not in captures, "Unexpected/duplicate captured token position")
        captures[key] = cap
    require(set(captures) == expected, "Missing prefill/decode positions")
    return run, captures


def vector(path: Path, cap: dict, vocab: int) -> array.array:
    name = cap["file"]
    require(isinstance(name, str) and Path(name).name == name and name.endswith(".f32"), "Unsafe vector filename")
    file = path / name
    require(not file.is_symlink(), "Vector aliases refused")
    raw = file.read_bytes()
    require(cap["elements"] == vocab and len(raw) == vocab * 4 and sha(raw) == cap["sha256"], "Vector size/SHA mismatch")
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def metrics(reference: array.array, candidate: array.array, valid: int, top_k: int) -> dict:
    require(len(reference) == len(candidate) and 0 < valid <= len(reference), "Vector dimensions differ")
    a, b = reference[:valid], candidate[:valid]
    require(all(math.isfinite(x) for x in a) and all(math.isfinite(x) for x in b), "Nonfinite valid-vocabulary logits")
    arg_a = max(range(valid), key=a.__getitem__)
    arg_b = max(range(valid), key=b.__getitem__)
    head_arg_a = max(range(len(reference)), key=reference.__getitem__)
    head_arg_b = max(range(len(candidate)), key=candidate.__getitem__)
    squared_error = math.fsum((float(x) - y) ** 2 for x, y in zip(a, b))
    norm_a = math.fsum(float(x) ** 2 for x in a)
    norm_b = math.fsum(float(x) ** 2 for x in b)
    dot = math.fsum(float(x) * y for x, y in zip(a, b))
    # A constant logit offset leaves the distribution unchanged; report its size
    # and centered errors as diagnostics while retaining the uncentered bound.
    mean_delta = math.fsum(float(y) - x for x, y in zip(a, b)) / valid
    centered_error = math.fsum((float(y) - x - mean_delta) ** 2 for x, y in zip(a, b))
    top_a = heapq.nlargest(min(top_k, valid), range(valid), key=a.__getitem__)
    top_b = heapq.nlargest(min(top_k, valid), range(valid), key=b.__getitem__)
    require(norm_a > 0 and norm_b > 0, "Zero-norm logit vector")
    return {
        "validElements": valid,
        "referenceArgmax": arg_a,
        "candidateArgmax": arg_b,
        "referenceHeadArgmax": head_arg_a,
        "candidateHeadArgmax": head_arg_b,
        "referencePaddedRowWins": head_arg_a >= valid,
        "candidatePaddedRowWins": head_arg_b >= valid,
        "referenceNonfinitePadding": sum(not math.isfinite(x) for x in reference[valid:]),
        "candidateNonfinitePadding": sum(not math.isfinite(x) for x in candidate[valid:]),
        "argmaxEqual": arg_a == arg_b,
        "validFloatBitsEqual": a.tobytes() == b.tobytes(),
        "fullIncludingPaddingBitsEqual": reference.tobytes() == candidate.tobytes(),
        "maxAbsoluteError": max(abs(float(x) - y) for x, y in zip(a, b)),
        "relativeL2": math.sqrt(squared_error / norm_a),
        "cosine": max(-1.0, min(1.0, dot / math.sqrt(norm_a * norm_b))),
        "meanLogitOffset": mean_delta,
        "centeredRmsError": math.sqrt(centered_error / valid),
        "referenceTopK": top_a,
        "candidateTopK": top_b,
        "topKOverlap": len(set(top_a) & set(top_b)),
    }


def compare(reference_path: Path, candidate_path: Path, *, require_argmax: bool = False,
            max_relative_l2: float | None = None, min_cosine: float | None = None,
            top_k: int = 10) -> dict:
    reference, left = read_run(reference_path)
    candidate, right = read_run(candidate_path)
    require(reference["engine"] != candidate["engine"], "Independent engine identity required; same-engine A/A is not an oracle")
    for key in ("inputsSha256", "artifactDirectorySha256", "frontendResourceHashes", "vocabSize", "validVocabSize"):
        require(key in reference and reference[key] == candidate.get(key), f"Producer identity differs: {key}")
    require(set(left) == set(right), "Captured positions differ")
    rows = []
    passed = True
    for key in sorted(left):
        m = metrics(vector(reference_path, left[key], reference["vocabSize"]),
                    vector(candidate_path, right[key], candidate["vocabSize"]),
                    reference["validVocabSize"], top_k)
        require(left[key]["argmax"] == m["referenceArgmax"] and right[key]["argmax"] == m["candidateArgmax"],
                "Captured argmax metadata differs from saved valid-vocabulary logits")
        gate = ((not require_argmax or m["argmaxEqual"])
                and (max_relative_l2 is None or m["relativeL2"] <= max_relative_l2)
                and (min_cosine is None or m["cosine"] >= min_cosine))
        passed &= gate
        rows.append({"caseId": key[0], "kind": key[1], "position": key[2], "numericGatePassed": gate, **m})
    configured = require_argmax or max_relative_l2 is not None or min_cosine is not None
    return {
        "schema": "ninfer-qwen38-body-comparison/1", "complete": True,
        "referenceEngine": reference["engine"], "candidateEngine": candidate["engine"],
        "inputsSha256": reference["inputsSha256"], "artifactDirectorySha256": reference["artifactDirectorySha256"],
        "gate": {"requireArgmax": require_argmax, "maxRelativeL2": max_relative_l2, "minCosine": min_cosine,
                 "configured": configured, "passed": bool(passed) if configured else None},
        "positions": len(rows), "argmaxMatches": sum(r["argmaxEqual"] for r in rows), "rows": rows,
        "modelQuality": "NOT_RUN", "performance": "NOT_QUALIFIED",
        "limitations": ["Only supplied teacher-forced token positions are compared.",
                        "No tolerance or oracle result implies universal model equivalence.",
                        "Padding rows are preserved but excluded from semantic argmax/error metrics.",
                        "Engine/build provenance and whole-artifact identity need external verification."],
    }


def fresh_output(path: Path) -> None:
    require(path.is_absolute() and path == path.resolve(), "Canonical absolute output required")
    parts = path.parts
    require("artifacts" in parts or any(parts[i:i + 2] == ("docs", "validation") for i in range(len(parts) - 1)),
            "Output must be under ignored artifacts/ or docs/validation/")
    require(not path.exists() and not path.is_symlink(), "Fresh output required")
    require(all(not p.is_symlink() and not p.is_file() for p in path.parents), "Aliased/non-directory output parent refused")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-argmax", action="store_true")
    parser.add_argument("--max-relative-l2", type=float)
    parser.add_argument("--min-cosine", type=float)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        require(args.top_k > 0, "Positive top-k required")
        require(args.max_relative_l2 is None or math.isfinite(args.max_relative_l2) and args.max_relative_l2 >= 0, "Invalid relative-L2 threshold")
        require(args.min_cosine is None or math.isfinite(args.min_cosine) and -1 <= args.min_cosine <= 1, "Invalid cosine threshold")
        fresh_output(args.output)
        result = compare(args.reference, args.candidate, require_argmax=args.require_argmax,
                         max_relative_l2=args.max_relative_l2, min_cosine=args.min_cosine, top_k=args.top_k)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "limitations")}))
        return 0 if result["gate"]["passed"] is not False else 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Comparison refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
