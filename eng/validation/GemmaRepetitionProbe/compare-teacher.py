"""Compare matched-history top-logit evidence, respecting model suppression.

This is a diagnostic, not a full-logit oracle or a quality pass/fail gate.
llama's pre-sampling logprobs may include suppressed tokens; compare permitted
rankings and pairwise logit gaps, which do not depend on the softmax denominator.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prompt-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve old evidence: output already exists")
    teacher = json.loads((args.teacher / "steps.json").read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    prompt = json.loads(args.prompt_record.read_text(encoding="utf-8"))
    suppressed = {item["Id"] for item in prompt["SuppressedTokens"]}
    if len(teacher) != len(reference["completion_probabilities"]):
        raise ValueError("Teacher/reference row counts differ")
    rows = []
    for index, (actual, expected) in enumerate(zip(teacher, reference["completion_probabilities"])):
        if actual["Token"] != expected["id"] or not actual["TeacherForced"]:
            raise ValueError(f"Row {index} is not replaying the reference's history")
        allowed_actual = [t for t in actual["Top"] if t["Token"] not in suppressed]
        allowed_expected = [t for t in expected["top_logprobs"] if t["id"] not in suppressed]
        if not allowed_actual or not allowed_expected:
            raise ValueError(f"Top-list truncation conceals the allowed argmax at {index}")
        actual_map = {t["Token"]: t for t in allowed_actual}
        expected_map = {t["id"]: t for t in allowed_expected}
        anchor = expected["id"]
        if anchor not in actual_map or anchor not in expected_map:
            raise ValueError(f"Top-list truncation conceals the reference token at {index}")
        pairs = []
        for token in actual_map.keys() & expected_map.keys():
            gap_actual = actual_map[token]["Logit"] - actual_map[anchor]["Logit"]
            gap_expected = expected_map[token]["logprob"] - expected_map[anchor]["logprob"]
            pairs.append({"Token": token, "Text": actual_map[token]["Text"],
                          "ActualGap": gap_actual, "ReferenceGap": gap_expected,
                          "AbsoluteGapError": abs(gap_actual - gap_expected)})
        rows.append({"Step": index, "RawActualArgmax": actual["GreedyToken"],
                     "ProductionActualArgmax": allowed_actual[0]["Token"],
                     "ProductionReferenceArgmax": allowed_expected[0]["id"],
                     "ReferenceSample": expected["id"],
                     "Agreement": allowed_actual[0]["Token"] == expected["id"],
                     "CommonPermittedTopTokens": len(pairs),
                     "MaxPairwiseGapError": max(p["AbsoluteGapError"] for p in pairs),
                     "PairwiseGaps": pairs})
    mismatches = [row for row in rows if not row["Agreement"]]
    result = {"Qualification": "Matched-history top-list diagnostic only; no full-logit or language-quality pass is implied.",
              "Rows": len(rows), "SuppressedTokens": sorted(suppressed),
              "ProductionArgmaxMismatches": len(mismatches),
              "MismatchSteps": [r["Step"] for r in mismatches],
              "FirstMismatch": mismatches[0] if mismatches else None,
              "Comparisons": rows}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("Comparisons", "FirstMismatch")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
