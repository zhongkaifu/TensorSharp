#!/usr/bin/env python3
"""Strict, fixed-length throughput comparison against an independently started llama server.

Server ownership, binary/model identity and startup configuration are recorded by
the caller. This client never starts a model or infers those facts from a URL.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request


_spec = importlib.util.spec_from_file_location(
    "qwen_reference", Path(__file__).resolve().parents[1] / "qwen38-llama-reference.py")
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)
SOURCE_REVISION = "4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe"
CONFIGURATION = {"context": 2048, "batch": 2048, "ubatch": 2048,
                 "k_cache": "f16", "v_cache": "f16", "all_gpu": True,
                 "parallel": 1, "speculative": "none"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"Invalid positive finite {name}")
    return value


def tokens_valid(tokens, count):
    return (isinstance(tokens, list) and len(tokens) == count
            and all(type(token) is int and token >= 0 for token in tokens))


def make_request(tokens, prompt_count=643, generated_count=64):
    require(tokens_valid(tokens, prompt_count), "Exact prompt length or token IDs are invalid")
    require(generated_count > 1, "At least two generated tokens are required for decode timing")
    body = _reference.request_body({"synthetic": False, "prompt_tokens": tokens}, generated_count)
    body.update(ignore_eos=True, **{"speculative.type": "none"})
    return body


def validate_identity(identity, ts_report):
    require(identity.get("source_revision") == SOURCE_REVISION,
            "This timing contract was verified against llama.cpp 4ebdf2c74 only")
    for key in ("binary_sha256", "model_sha256"):
        digest = identity.get(key)
        require(isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest.lower()), f"Missing/invalid {key}")
    require(identity["model_sha256"].lower() == ts_report["ModelSha256"].lower(),
            "Server and TensorSharp checkpoint identities differ")
    require(isinstance(identity.get("command"), list) and identity["command"]
            and all(isinstance(arg, str) for arg in identity["command"]), "Actual server command is required")
    require(isinstance(identity.get("startup_evidence"), str) and identity["startup_evidence"].strip(),
            "Retain startup/device-offload evidence separately and name it in startup_evidence")
    require(identity.get("configuration") == CONFIGURATION,
            "Server configuration must be ctx/batch/ubatch 2048, F16 KV, all GPU, one slot, no speculation")


def validate_ts_report(report, prompt_count=643, generated_count=64):
    require(report.get("Executed") is True and report.get("Error") is None, "TensorSharp execution failed")
    require(report.get("Context") == CONFIGURATION["context"], "TensorSharp context must be 2048")
    require(isinstance(report.get("ModelSha256"), str) and report["ModelSha256"], "Missing checkpoint identity")
    rows = report.get("Records")
    require(isinstance(rows, list) and len(rows) == 4, "Require one warmup and three TensorSharp measurements")
    for index, row in enumerate(rows):
        require(row.get("Warmup") is (index == 0), "TensorSharp warmup order/count differs")
        require(row.get("PromptTokens") == prompt_count and row.get("DecodeCalls") == generated_count - 1,
                "TensorSharp prompt/decode work differs")
        require(tokens_valid(row.get("Generated"), generated_count), "Missing/short TensorSharp token history")
        positive(row.get("PrefillMilliseconds"), "TensorSharp prefill milliseconds")
        positive(row.get("DecodeMilliseconds"), "TensorSharp decode milliseconds")
    return rows


def validate_response(result, expected_tokens, prompt_count=643, generated_count=64):
    require(tokens_valid(expected_tokens, generated_count), "Invalid TensorSharp comparison token history")
    require(isinstance(result, dict), "Completion must be an object")
    require(result.get("truncated") is False, "Completion was truncated or truncation evidence is missing")
    require(result.get("stop") is True and result.get("stop_type") == "limit", "Expected fixed-length termination")
    require(tokens_valid(result.get("tokens"), generated_count), "Missing/short/invalid generated token history")
    require(result.get("tokens_predicted") == generated_count and result.get("tokens_evaluated") == prompt_count,
            "Completion token counters differ from requested work")
    timing = result.get("timings", {})
    for field, expected in (("cache_n", 0), ("prompt_n", prompt_count), ("predicted_n", generated_count)):
        require(type(timing.get(field)) is int and timing[field] == expected, f"Invalid timings.{field}")
    settings = result.get("generation_settings", {})
    for field, expected in (("temperature", 0), ("top_k", 0), ("top_p", 1), ("min_p", 0),
                            ("repeat_penalty", 1), ("repeat_last_n", 0), ("presence_penalty", 0),
                            ("frequency_penalty", 0), ("dry_multiplier", 0), ("ignore_eos", True),
                            ("n_predict", generated_count)):
        require(settings.get(field) == expected, f"Unexpected or missing generation_settings.{field}")
    # 4ebdf may retain duplicate NONE enum entries when both startup and request
    # disable speculation. Reject every active type, but not that serialization.
    speculative = settings.get("speculative.types")
    require(isinstance(speculative, str) and all(kind == "none" for kind in speculative.split(",")),
            "Unexpected or missing generation_settings.speculative.types")
    prompt_ms = positive(timing.get("prompt_ms"), "prompt_ms")
    decode_ms = positive(timing.get("predicted_ms"), "predicted_ms")
    prompt_tps, decode_tps = prompt_count * 1000 / prompt_ms, (generated_count - 1) * 1000 / decode_ms
    # 4ebdf server-common.h excludes the first token from timed decode steps.
    for field, expected in (("prompt_per_second", prompt_tps), ("predicted_per_second", decode_tps)):
        actual = positive(timing.get(field), field)
        require(math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6),
                f"{field} disagrees with the verified timing denominator")
    mismatches = [i for i, (left, right) in enumerate(zip(result["tokens"], expected_tokens)) if left != right]
    return {"prefill_ms": prompt_ms, "decode_ms": decode_ms, "prompt_tokens": prompt_count,
            "decode_calls": generated_count - 1, "prefill_tokens_per_second": prompt_tps,
            "decode_tokens_per_second": decode_tps, "raw_token_histories_match": not mismatches,
            "mismatched_token_positions": mismatches, "first_mismatch": mismatches[0] if mismatches else None}


def summarize(rows, ts_rows):
    def stats(values):
        return {"median": statistics.median(values), "minimum": min(values), "maximum": max(values), "samples": len(values)}
    result = {}
    for metric, ts_field, numerator in (("prefill", "PrefillMilliseconds", "PromptTokens"),
                                         ("decode", "DecodeMilliseconds", "DecodeCalls")):
        llama = stats([row[metric + "_tokens_per_second"] for row in rows[1:]])
        ts = stats([row[numerator] * 1000 / row[ts_field] for row in ts_rows[1:]])
        result[metric] = {"llama_tokens_per_second": llama, "tensorsharp_tokens_per_second": ts,
                          "tensorsharp_to_llama_ratio": ts["median"] / llama["median"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True, help="AdaptiveMemoryProbe's exact prompt.json array")
    parser.add_argument("--tensorsharp-report", type=Path, required=True)
    parser.add_argument("--server-identity", type=Path, help="Required for execution; separately recorded actual startup identity")
    parser.add_argument("--server", default="http://127.0.0.1:5099")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180, help="Per-request socket timeout; use run-bounded-probe.py for a hard deadline")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    positive(args.timeout, "timeout")
    prompt_bytes, ts_bytes = args.prompt.read_bytes(), args.tensorsharp_report.read_bytes()
    prompt, ts = json.loads(prompt_bytes), json.loads(ts_bytes)
    require(json.loads((args.tensorsharp_report.parent / "prompt.json").read_bytes()) == prompt,
            "Prompt IDs differ from the TensorSharp report's adjacent prompt.json")
    body = make_request(prompt)
    ts_rows = validate_ts_report(ts)
    identity_bytes = args.server_identity.read_bytes() if args.server_identity else None
    if identity_bytes is not None:
        validate_identity(json.loads(identity_bytes), ts)
    elif not args.prepare_only:
        parser.error("--server-identity is required for actual execution")
    args.output.mkdir(parents=True, exist_ok=False)

    def write(name, data):
        destination = args.output / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(destination)

    write("request.json", body)
    (args.output / "prompt.json").write_bytes(prompt_bytes)
    (args.output / "tensorsharp-report.json").write_bytes(ts_bytes)
    if identity_bytes is not None:
        (args.output / "server-identity.json").write_bytes(identity_bytes)
    report = {"complete": False, "prepare_only": args.prepare_only, "comparison_passed": False,
              "quality_passed": None, "server": args.server,
              "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
              "tensorsharp_report_sha256": hashlib.sha256(ts_bytes).hexdigest(),
              "server_identity_sha256": hashlib.sha256(identity_bytes).hexdigest() if identity_bytes else None,
              "rows": [], "errors": [],
              "scope": ["Forced 64-token greedy completion with EOS ignored is throughput evidence, not answer-quality validation.",
                        "Each request evaluates 643 uncached prompt tokens and 63 timed decode calls, as in the TensorSharp report.",
                        "Raw token histories are compared. No independent full-logit or numerical-math equivalence is established.",
                        "One excluded warmup and three measurements in one server process; caller must ensure an idle device and retain startup evidence.",
                        "Server identity is caller-supplied evidence, not independently inferred from HTTP responses. Cross-engine timings include different engine overheads.",
                        "Ratios describe these samples, not statistical significance or a general performance guarantee."]}
    write("report.json", report)
    if args.prepare_only:
        return 0
    started = time.monotonic()
    try:
        for index, ts_row in enumerate(ts_rows):
            request = urllib.request.Request(args.server.rstrip("/") + "/completion", json.dumps(body).encode("utf-8"),
                                             {"Content-Type": "application/json"})
            request_start = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    raw = response.read()
            except urllib.error.HTTPError as error:
                (args.output / f"response-{index}.http-error.txt").write_bytes(error.read())
                raise
            # Retain the unmodified response before validation, including failed evidence.
            (args.output / f"response-{index}.json").write_bytes(raw)
            row = validate_response(json.loads(raw), ts_row["Generated"])
            row.update(run=index - 1, warmup=index == 0, wall_seconds=time.monotonic() - request_start)
            report["rows"].append(row)
            write("report.json", report)
        report.update(complete=True, measurements=summarize(report["rows"], ts_rows),
                      raw_token_histories_match=all(row["raw_token_histories_match"] for row in report["rows"]))
        report["comparison_passed"] = report["raw_token_histories_match"]
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["wall_seconds"] = time.monotonic() - started
    write("report.json", report)
    return 0 if report["comparison_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
