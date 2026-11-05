#!/usr/bin/env python3
"""Read-only semantic diagnostics; NEVER regrade the frozen benchmark or its performance gate.

  python3 eng/validation/qwen38-quality-diagnostics.py 'docs/validation/RUN/raw/*.jsonl' \\
      --output docs/validation/RUN/quality-diagnostics.json

Separates a single outer Markdown JSON fence from wrong values, audits generated
math ground truth and request hashes, and reports per-question GSM8K accuracy.
The original verdicts remain authoritative. No relaxed diagnostic is a PASS,
performance eligibility, comprehensive quality claim or engine ranking.
"""
import argparse
from decimal import Decimal
import glob
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("frozen_qwen38_bench", ROOT / "eng" / "qwen38-engine-bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

# Independent arithmetic derivations, after reading every selected public question.
GSM8K_AUDIT = {
    "gsm8k-test-88": ("88000 / (10 + 1)", Decimal(88000) / 11),
    "gsm8k-test-103": ("42 / 3 * 10", Decimal(42) / 3 * 10),
    "gsm8k-test-122": ("8 + 2 * (8 - 2) + 4 * (8 - 2 + 1)", Decimal(8) + 2 * (8 - 2) + 4 * (8 - 2 + 1)),
    "gsm8k-test-186": ("(160 - 10 * 5 - 20) / (3 * 6)", (Decimal(160) - 10 * 5 - 20) / (3 * 6)),
    "gsm8k-test-237": ("(20 + 10) * 3", Decimal(20 + 10) * 3),
    "gsm8k-test-256": ("100 * 0.01 + 40 * 0.05 + 20 * 0.10 + 40", Decimal(100) * Decimal(".01") + 40 * Decimal(".05") + 20 * Decimal(".10") + 40),
    "gsm8k-test-290": ("(22 + (22 - 5)) * 10", Decimal(22 + (22 - 5)) * 10),
    "gsm8k-test-448": ("17 * 20 + (48 - 17) * 25 - 576", Decimal(17) * 20 + (48 - 17) * 25 - 576),
    "gsm8k-test-562": ("32 + 5 * 10", Decimal(32) + 5 * 10),
    "gsm8k-test-674": ("(600 - 20 * 12) / 30", (Decimal(600) - 20 * 12) / 30),
    "gsm8k-test-807": ("2 * 72 / (12 * 3)", Decimal(2 * 72) / (12 * 3)),
    "gsm8k-test-914": ("100 * 30 / 1000", Decimal(100 * 30) / 1000),
}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def verified_text(request):
    text = request.get("content") or ""
    return text, bool(request.get("content_sha256")) and sha(text.encode()) == request["content_sha256"]


def inference_completed(request):
    return not request.get("error") and request.get("finish_reason") in ("stop", "length", "tool_calls", "function_call")


def diagnostic_json(text):
    """Only remove one balanced outer fence. Never search arbitrary prose for a JSON substring."""
    fenced = re.fullmatch(r"\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*", text, re.S | re.I)
    try:
        return json.loads(fenced.group(1) if fenced else text), "outer_json_fence" if fenced else "raw_json"
    except ValueError:
        return None, "unparseable"


def request_context(header):
    client = object.__new__(bench.Client)
    sampling = header.get("sampling", {})
    client.args = SimpleNamespace(temperature=sampling.get("temperature", 0), top_p=sampling.get("top_p", 1),
                                  sampling_seed=sampling.get("seed", 1234))
    client.engine, client.model, client.extra = header["engine"], header.get("model"), header.get("extra_body") or {}
    return bench.Ctx(SimpleNamespace(nonce_seed=header["nonce_seed"]), client, {})


def same_request(ctx, header, scenario, repeat, index, prompt, request):
    body = ctx.client.body(ctx.messages(scenario, repeat, prompt, index), header["answer_tokens"])
    identity = json.dumps(bench.request_identity(body), sort_keys=True, ensure_ascii=False).encode()
    return bool(request.get("request_sha256")) and sha(identity) == request["request_sha256"]


def needle_case(header, case):
    notes, scenario, repeat = case["notes"], case["scenario"], case["repeat"]
    request = case["requests"][0]
    text, complete = verified_text(request)
    actual, format_kind = diagnostic_json(text)
    rng = random.Random(f"{header['nonce_seed']}:{scenario}:{repeat}")
    keys = ("alpha", "beta", "gamma", "delta", "epsilon") if header.get("long_order_quality") else ("alpha", "beta", "gamma")
    codes = {key: f"{rng.choice(bench.CODE_WORDS_A)}-{rng.choice(bench.CODE_WORDS_B)}-{rng.randint(1000, 9999)}" for key in keys}
    order = list(keys)
    if header.get("long_order_quality"):
        rng.shuffle(order)
    ground_truth_valid = codes == notes["code_words"] and order == notes["physical_order"]
    expected = dict(codes)
    if header.get("long_order_quality"):
        at = order.index("gamma")
        expected.update(order=order, before_gamma=order[at - 1] if at else None,
                        after_gamma=order[at + 1] if at + 1 < len(order) else None)
    fields = {key: isinstance(actual, dict) and key in actual and actual[key] == value for key, value in expected.items()}
    semantic_correct = complete and ground_truth_valid and all(fields.values()) and inference_completed(request)
    other_checks = [check for check in case["checks"] if not check["name"].startswith(("needle_", "position_"))]
    other_pass = bool(other_checks) and all(check["ok"] for check in other_checks)
    if request.get("error"):
        category = "request_error_or_capacity_rejection_not_semantic_quality_evidence"
    elif not complete:
        category = "stored_content_incomplete_or_hash_mismatch"
    elif not ground_truth_valid:
        category = "ground_truth_does_not_match_seed"
    elif not case["pass"] and semantic_correct and format_kind == "outer_json_fence" and other_pass:
        category = "format_only_failure"
    elif semantic_correct and case["pass"]:
        category = "strict_and_semantic_correct"
    elif semantic_correct:
        category = "semantic_correct_but_other_gate_failed"
    else:
        category = "semantic_error_or_unparseable"
    return {"repeat": repeat, "original_pass": case["pass"], "original_verdict": case["verdict"],
            "content_sha256": request.get("content_sha256"), "stored_full_content_verified": complete,
            "inference_completed": inference_completed(request),
            "ground_truth_matches_seed": ground_truth_valid, "diagnostic_format": format_kind,
            "diagnostic_semantic_correct": semantic_correct, "category": category,
            "field_correct": fields, "expected": expected, "observed": actual}


def math_case(header, case):
    repeat, ctx = case["repeat"], request_context(header)
    rng = random.Random(f"{header['nonce_seed']}:quality_tasks:{repeat}")
    questions = []
    for i, request in enumerate(case["requests"]):
        numbers = [rng.randint(-999, 999) for _ in range(9)]
        a, b, c = [rng.randint(10, 99) for _ in range(3)]
        # Decimal independently checks integer multiplication and summation.
        expected = {"result": int(Decimal(a) * Decimal(b) - Decimal(c)), "sorted": sorted(numbers),
                    "sum": int(sum((Decimal(n) for n in numbers), Decimal(0))), "unique_count": len(set(numbers))}
        prompt = (f"Calculate {a} * {b} - {c}. Also sort the list {numbers} in ascending order, "
                  "sum its numbers and count its distinct values. Return exactly one JSON object with "
                  "keys result (integer), sorted (array of integers), sum (integer), unique_count (integer). "
                  "Do not use markdown or include other text.")
        text, complete = verified_text(request)
        actual, kind = diagnostic_json(text)
        values = {key: isinstance(actual, dict) and key in actual and actual[key] == value for key, value in expected.items()}
        types = {key: isinstance(actual, dict) and (isinstance(actual.get(key), list)
                 and all(type(n) is int for n in actual[key]) if key == "sorted" else type(actual.get(key)) is int)
                 for key in expected}
        recorded_expected = case["notes"]["expected"][i]
        original = next((check["ok"] for check in case["checks"] if check["name"] == f"task{i}"), None)
        questions.append({"index": i, "original_question_pass": original, "content_sha256": request.get("content_sha256"),
            "stored_full_content_verified": complete, "request_hash_matches_reconstructed_prompt": same_request(ctx, header, "quality_tasks", repeat, i, prompt, request),
            "ground_truth_matches_independent_decimal_reference": expected == recorded_expected,
            "operands": {"a": a, "b": b, "c": c, "numbers": numbers}, "diagnostic_format": kind,
            "field_value_correct": values, "field_type_correct": types,
            "diagnostic_whole_answer_correct": complete and actual == expected and all(types.values()) and inference_completed(request),
            "expected": expected, "observed": actual})
    return {"repeat": repeat, "original_pass": case["pass"], "questions": questions}


def gsm_case(header, case, fixture):
    ctx, questions = request_context(header), []
    examples = {example["id"]: example for example in fixture["examples"]}
    for i, (request, identity) in enumerate(zip(case["requests"], case["notes"]["ids"])):
        example = examples[identity]
        prompt = (example["question"] + "\n\nSolve the problem. Output only the final numeric answer, "
                  "without units, markdown or explanation.")
        text, complete = verified_text(request)
        expected = Decimal(example["answer"])
        strict = bench.numeric_answer(text)
        final_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
        relaxed = bench.numeric_answer(final_line)
        expression, independent = GSM8K_AUDIT[identity]
        semantic = complete and relaxed == expected and inference_completed(request)
        original = next((check["ok"] for check in case["checks"] if check["name"] == identity), None)
        questions.append({"id": identity, "original_question_pass": original,
            "recomputed_strict_matches_original": (strict == expected and not request.get("error")) == original,
            "content_sha256": request.get("content_sha256"), "stored_full_content_verified": complete,
            "request_hash_matches_reconstructed_prompt": same_request(ctx, header, "quality_gsm8k", case["repeat"], i, prompt, request),
            "fixture_answer": example["answer"], "independent_reference_expression": expression,
            "ground_truth_matches_independent_reference": expected == independent,
            "diagnostic_final_standalone_numeric_line": final_line if relaxed is not None else None,
            "diagnostic_final_line_semantically_correct": semantic,
            "format_only_failure": strict != expected and semantic, "returned_text": text})
    return {"repeat": case["repeat"], "original_pass": case["pass"], "questions": questions}


def analyse_run(header, cases, fixture):
    out = {"engine": header["engine"], "mode": header.get("mode"), "label": header.get("label"),
           "run_id": header.get("run_id"), "harness": header.get("harness"),
           "harness_sha256": header.get("harness_sha256"), "scenarios": {}}
    for case in cases:
        scenario = case.get("scenario", "")
        if scenario.startswith("long_prompt_"):
            detail = needle_case(header, case)
        elif scenario == "quality_tasks":
            detail = math_case(header, case)
        elif scenario == "quality_gsm8k":
            detail = gsm_case(header, case, fixture)
        else:
            continue
        out["scenarios"].setdefault(scenario, {"cases": []})["cases"].append(detail)
    for scenario, result in out["scenarios"].items():
        observed = result["cases"]
        result.update(repeats_observed=len(observed), strict_pass_repeats=sum(case["original_pass"] for case in observed))
        if scenario.startswith("long_prompt_"):
            result.update(diagnostic_semantic_correct_repeats=sum(case["diagnostic_semantic_correct"] for case in observed),
                          diagnostic_format_only_failure_repeats=sum(case["category"] == "format_only_failure" for case in observed))
        else:
            questions = [q for case in observed for q in case["questions"]]
            result.update(questions_observed=len(questions), strict_correct_questions=sum(q["original_question_pass"] is True for q in questions),
                          reconstructed_request_hash_matches=sum(q["request_hash_matches_reconstructed_prompt"] for q in questions),
                          complete_stored_response_hash_matches=sum(q["stored_full_content_verified"] for q in questions))
            if scenario == "quality_tasks":
                result.update(ground_truth_reference_matches=sum(q["ground_truth_matches_independent_decimal_reference"] for q in questions),
                              diagnostic_whole_answer_correct=sum(q["diagnostic_whole_answer_correct"] for q in questions),
                              field_value_correct={key: sum(q["field_value_correct"][key] for q in questions) for key in ("result", "sorted", "sum", "unique_count")},
                              field_type_correct={key: sum(q["field_type_correct"][key] for q in questions) for key in ("result", "sorted", "sum", "unique_count")})
            else:
                result.update(ground_truth_reference_matches=sum(q["ground_truth_matches_independent_reference"] for q in questions),
                              diagnostic_final_line_correct_questions=sum(q["diagnostic_final_line_semantically_correct"] for q in questions),
                              diagnostic_format_only_failure_questions=sum(q["format_only_failure"] for q in questions))
    return out


def compare_runs(baseline, candidate):
    """Paired request identity and narrow functional checks, never blanket quality equivalence."""
    before_header, before_cases = baseline
    after_header, after_cases = candidate
    before = {(case["scenario"], case["repeat"]): case for case in before_cases}
    after = {(case["scenario"], case["repeat"]): case for case in after_cases}
    if len(before) != len(before_cases) or len(after) != len(after_cases):
        return {"pairing_error": "duplicate scenario/repeat ids; exact comparison refused"}
    by_scenario, differences = {}, []
    for key in sorted(set(before) | set(after)):
        scenario, repeat = key
        row = by_scenario.setdefault(scenario, {"paired_cases": 0, "baseline_strict_pass_cases": 0,
            "candidate_strict_pass_cases": 0, "paired_requests": 0, "same_request_hashes": 0,
            "complete_response_pairs": 0, "exact_responses_for_same_complete_requests": 0,
            "functional_check_pairs": 0, "functional_check_regressions": 0, "functional_check_improvements": 0,
            "unpaired_cases": [], "unpaired_requests": [], "changed_checks": []})
        if key not in before or key not in after:
            row["unpaired_cases"].append({"repeat": repeat, "missing_from": "baseline" if key not in before else "candidate"})
            continue
        a, b = before[key], after[key]
        row["paired_cases"] += 1
        row["baseline_strict_pass_cases"] += a["pass"] is True
        row["candidate_strict_pass_cases"] += b["pass"] is True
        old_checks = {check["name"]: check["ok"] for check in a["checks"] if check["name"] != "stream_completed" and not check["name"].startswith("baseline_")}
        new_checks = {check["name"]: check["ok"] for check in b["checks"] if check["name"] != "stream_completed" and not check["name"].startswith("baseline_")}
        for name in old_checks.keys() | new_checks.keys():
            if name not in old_checks or name not in new_checks:
                row["changed_checks"].append({"repeat": repeat, "check": name, "baseline": old_checks.get(name), "candidate": new_checks.get(name)})
                continue
            row["functional_check_pairs"] += 1
            row["functional_check_regressions"] += old_checks[name] is True and new_checks[name] is not True
            row["functional_check_improvements"] += old_checks[name] is not True and new_checks[name] is True
            if old_checks[name] != new_checks[name]:
                row["changed_checks"].append({"repeat": repeat, "check": name, "baseline": old_checks[name], "candidate": new_checks[name]})
        old_requests, new_requests = a.get("requests", []), b.get("requests", [])
        if len(old_requests) != len(new_requests):
            row["unpaired_requests"].append({"repeat": repeat, "baseline_count": len(old_requests), "candidate_count": len(new_requests)})
        for i, (old, new) in enumerate(zip(old_requests, new_requests)):
            same_input = bool(old.get("request_sha256")) and old.get("request_sha256") == new.get("request_sha256")
            complete = inference_completed(old) and inference_completed(new)
            same_output = bool(old.get("response_sha256")) and old.get("response_sha256") == new.get("response_sha256")
            row["paired_requests"] += 1
            row["same_request_hashes"] += same_input
            row["complete_response_pairs"] += complete
            row["exact_responses_for_same_complete_requests"] += same_input and complete and same_output
            if not (same_input and complete and same_output):
                difference = {"scenario": scenario, "repeat": repeat, "request_index": i,
                    "same_request_hash": same_input, "both_inferences_completed": complete,
                    "baseline_response_sha256": old.get("response_sha256"), "candidate_response_sha256": new.get("response_sha256"),
                    "baseline_finish_reason": old.get("finish_reason"), "candidate_finish_reason": new.get("finish_reason"),
                    "baseline_content_excerpt": (old.get("content") or "")[:500],
                    "candidate_content_excerpt": (new.get("content") or "")[:500]}
                if scenario.startswith("parallel_"):
                    expected = (12 + 7 * i + repeat) * (31 + 3 * i)
                    products = [re.search(r"(?m)^PRODUCT:\s*(-?\d+)\b", request.get("content") or "") for request in (old, new)]
                    difference["narrow_product_check"] = {"expected": expected,
                        "baseline_correct": bool(products[0]) and int(products[0].group(1)) == expected,
                        "candidate_correct": bool(products[1]) and int(products[1].group(1)) == expected,
                        "limitation": "Only the requested product is checked; differing explanatory prose has not been comprehensively graded."}
                if scenario == "agentic_loop" and i == len(old_requests) - 1 == len(new_requests) - 1:
                    totals = [re.search(r"(?m)^TOTAL:\s*\$?(\d+(?:\.\d+)?)\s*$", request.get("content") or "") for request in (old, new)]
                    difference["narrow_final_total_check"] = {"expected": "195", "independent_expression": "(17 + 26 + 9) * 3.75",
                        "baseline_correct": bool(totals[0]) and Decimal(totals[0].group(1)) == Decimal(195),
                        "candidate_correct": bool(totals[1]) and Decimal(totals[1].group(1)) == Decimal(195),
                        "limitation": "Correct final total and original tool workflow checks do not make differing natural-language responses exact."}
                differences.append(difference)
    totals = {name: sum(row[name] for row in by_scenario.values()) for name in (
        "paired_cases", "baseline_strict_pass_cases", "candidate_strict_pass_cases", "paired_requests", "same_request_hashes",
        "complete_response_pairs", "exact_responses_for_same_complete_requests", "functional_check_pairs",
        "functional_check_regressions", "functional_check_improvements")}
    return {"baseline": {key: before_header.get(key) for key in ("engine", "mode", "label", "run_id", "harness", "harness_sha256", "model")},
            "candidate": {key: after_header.get(key) for key in ("engine", "mode", "label", "run_id", "harness", "harness_sha256", "model")},
            "totals": totals, "scenarios": by_scenario, "differences": differences,
            "limitations": ["Exact response hashes require identical request hashes and completed inference on both sides.",
                            "No changed expected-answer checks is narrow evidence only; different prose is not exact or comprehensively quality-scored.",
                            "Existing plain-versus-DFlash differences do not establish regressions introduced by a later optimization."]}


def read_invocations(path):
    raw = Path(path).read_bytes()
    header, cases, invocations = None, [], []
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("kind") == "run":
            if header:
                invocations.append((header, cases))
            header, cases = rec, []
        elif rec.get("kind") == "case" and header:
            cases.append(rec)
    if header:
        invocations.append((header, cases))
    return raw, invocations


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-baseline", type=Path, help="optional same-engine raw JSONL; compare against its last invocation")
    args = parser.parse_args()
    fixture = json.loads(bench.GSM8K_FIXTURE.read_text())
    outputs, sources, invocations = [], [], []
    for filename in dict.fromkeys(path for pattern in args.inputs for path in sorted(glob.glob(pattern))):
        path = Path(filename)
        raw, records = read_invocations(path)
        sources.append({"path": str(path), "sha256": sha(raw)})
        invocations.extend(records)
        for header, cases in records:
            outputs.append(analyse_run(header, cases, fixture))
    evidence = {"diagnostic_only": True, "original_verdicts_and_performance_eligibility_unchanged": True,
        "policy": "A balanced outer JSON fence and a final standalone GSM8K numeric line are inspected only to classify failures. Never use these diagnostics as relaxed PASS or for performance ranking.",
        "sources": sources, "diagnostic_tool_sha256": sha(Path(__file__).read_bytes()),
        "fixture_sha256": sha(bench.GSM8K_FIXTURE.read_bytes()), "fixture_source_revision": fixture["upstream_revision"],
        "limitations": ["Small generated tasks and public GSM8K subset are not comprehensive quality evaluation.",
                        "Three repeats of the same eight GSM8K examples are 24 responses, not 24 independent test problems.",
                        "Correct final line does not by itself validate every step of a verbose explanation.",
                        "If stored content hash fails, the full answer is unavailable and no semantic correctness is granted.",
                        "Different quantized checkpoints and prompt settings do not isolate inference engine quality."], "runs": outputs}
    if args.compare_baseline:
        baseline_raw, baseline_runs = read_invocations(args.compare_baseline)
        if not baseline_runs:
            parser.error("--compare-baseline has no run header")
        baseline = baseline_runs[-1]
        evidence["comparison_baseline_source"] = {"path": str(args.compare_baseline), "sha256": sha(baseline_raw)}
        evidence["paired_comparisons"] = [compare_runs(baseline, record) for record in invocations
            if record[0]["engine"] == baseline[0]["engine"] and record[0].get("run_id") != baseline[0].get("run_id")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    compact = [{**{k: run[k] for k in ("engine", "mode", "label")}, "scenarios": {
        name: {key: value for key, value in detail.items() if key != "cases"} for name, detail in run["scenarios"].items()}}
        for run in outputs]
    print(json.dumps({"output": str(args.output), "summary": compact}, ensure_ascii=False))


if __name__ == "__main__":
    main()
