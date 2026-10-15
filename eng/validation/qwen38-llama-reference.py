#!/usr/bin/env python3
"""Replay a real Flash Next probe's exact prompt IDs on an independently started llama.cpp server.

The server must already use the same verified GGUF. No model or server is started
here. --prepare-only writes the request without using a device or network. Length
termination can provide a throughput sample but never passes complete-answer
quality; compare the labelled timing denominators, not just tokens/second.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def request_body(record, maximum):
    tokens = record.get("prompt_tokens")
    if not isinstance(tokens, list) or not tokens or any(type(value) is not int or value < 0 for value in tokens):
        raise ValueError("Missing or invalid TensorSharp prompt token IDs")
    if record.get("synthetic") is not False:
        raise ValueError("The reference requires a real checkpoint report")
    if maximum <= 0:
        raise ValueError("Maximum generated tokens must be positive")
    return {"prompt": tokens, "n_predict": maximum, "temperature": 0.0, "top_k": 0,
            "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0, "repeat_last_n": 0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0,
            "seed": 42, "cache_prompt": False, "return_tokens": True, "stream": False}


def classify(result, maximum, expected_text=None):
    tokens = result.get("tokens")
    valid = (isinstance(tokens, list) and 0 < len(tokens) <= maximum
             and all(type(value) is int and value >= 0 for value in tokens)
             and len(tokens) == result.get("tokens_predicted")
             and isinstance(result.get("content"), str))
    complete = valid and result.get("stop_type") == "eos" and not result.get("truncated", False)
    return {"token_evidence_valid": valid, "answer_complete": complete,
            "quality_passed": complete and result["content"].strip() == expected_text if expected_text is not None else None,
            "quality_scope": "Exact expected answer and EOS" if expected_text is not None else "Requires human semantic review",
            "throughput_scope": "llama.cpp reported prompt/generation timings; no token-denominator normalization applied"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-report", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:5099")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--expected-text")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    record = json.loads(args.prompt_report.read_text(encoding="utf-8-sig"))
    body = request_body(record, args.max_new)
    args.output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def post(endpoint, value):
        request = urllib.request.Request(args.server.rstrip("/") + endpoint,
            json.dumps(value).encode("utf-8"), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.load(response)

    metadata = {"run_complete": False, "prepare_only": args.prepare_only,
                "prompt_report_sha256": hashlib.sha256(args.prompt_report.read_bytes()).hexdigest(),
                "requested_model_path": record.get("model_path"), "request_prompt_tokens": len(body["prompt"]),
                "limitations": ["Caller must retain actual server command, checkpoint identity, binary hashes and clean source revision.",
                                "The exact token prompt removes template differences, but greedy outputs can diverge between engines.",
                                "No selected-expert cache or shared allocation quota is assumed for llama.cpp."]}
    write("request.json", body)
    write("execution.json", metadata)
    if args.prepare_only:
        return 0
    started = time.monotonic()
    try:
        # Check the renderer's text against the independent tokenizer too. The
        # measured request always uses IDs, so this diagnostic cannot rewrite it.
        if isinstance(record.get("rendered_prompt"), str):
            tokenized = post("/tokenize", {"content": record["rendered_prompt"], "add_special": True, "parse_special": True})
            write("tokenizer-control.json", {"response": tokenized, "same_tokens": tokenized.get("tokens") == body["prompt"]})
        result = post("/completion", body)
        write("completion.json", result)
        (args.output / "output.txt").write_text(result.get("content", ""), encoding="utf-8")
        metadata.update(classify(result, args.max_new, args.expected_text))
        metadata["timings"] = result.get("timings")
        metadata["stop_type"] = result.get("stop_type")
        metadata["captured"] = metadata["token_evidence_valid"]
    except Exception as error:
        metadata.update(captured=False, error=str(error))
    metadata.update(run_complete=True, wall_seconds=time.monotonic() - started)
    write("execution.json", metadata)
    return int(not metadata["captured"] or metadata.get("quality_passed") is False)


if __name__ == "__main__":
    raise SystemExit(main())
