#!/usr/bin/env python3
"""Compare complete TS vocabulary captures with unchanged llama-server predictions.

Every request supplies the exact recorded input history. llama's HTTP interface
returns pre-sampling log probabilities, not raw logits. Comparison therefore
subtracts each engine's value at the same reference token; it does not pretend
the softmax normalization constant is observable. Retain server identity/logs.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import sys
from urllib.request import Request, urlopen


def read_row(path, row):
    count, offset = row["elements"], row["byte_offset"]
    if type(count) is not int or count <= 0 or type(offset) is not int or offset < 0 or offset % 4:
        raise ValueError("Invalid vocabulary capture dimensions")
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read(count * 4)
    if len(data) != count * 4 or hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("Incomplete or changed vocabulary capture")
    values = array.array("f", data)
    if sys.byteorder != "little":
        values.byteswap()
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Nonfinite vocabulary capture")
    return values


def compare(logits, completion):
    predictions = completion.get("completion_probabilities", [])
    if len(predictions) != 1:
        raise ValueError("Expected exactly one prediction row")
    probabilities = predictions[0].get("top_logprobs", [])
    if len(probabilities) != len(logits):
        raise ValueError("Server did not return the complete vocabulary")
    values = [None] * len(logits)
    for item in probabilities:
        token, value = item.get("id"), item.get("logprob")
        if type(token) is not int or not 0 <= token < len(logits) or values[token] is not None:
            raise ValueError("Duplicate or invalid vocabulary token ID")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Nonfinite reference probability")
        values[token] = value
    # llama.cpp represents log(0) by the lowest float, not a recoverable logit.
    clipped = sum(value < -1e20 for value in values)
    if clipped:
        raise ValueError(f"{clipped} softmax values underflowed/clipped; raw-logit comparison is unavailable")
    reference = max(range(len(values)), key=values.__getitem__)
    ours = max(range(len(logits)), key=logits.__getitem__)
    left = [value - logits[reference] for value in logits]
    right = [value - values[reference] for value in values]
    difference = math.fsum((a - b) ** 2 for a, b in zip(left, right))
    denominator = math.fsum(value * value for value in right)
    return {"elements": len(logits), "reference_token": reference, "argmax": [ours, reference],
            "relative_l2_after_common_token_offset": math.sqrt(difference / max(denominator, 1e-300)),
            "max_abs_pairwise_logit_error": max(abs(a - b) for a, b in zip(left, right)),
            "limitation": "HTTP log probabilities include float32 softmax/log rounding; raw logits are not exposed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:5099")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    index = json.loads(args.logits_index.read_text(encoding="utf-8-sig"))
    if index.get("format") != "f32le":
        raise ValueError("Expected f32le diagnostic captures")
    source = Path(index["data_path"])
    rows = [row for row in index["rows"] if row["iteration"] == args.iteration and not row["warmup"]]
    if not rows:
        raise ValueError("No completed rows for the requested iteration")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"run_complete": False, "prepare_only": args.prepare_only,
              "index_sha256": hashlib.sha256(args.logits_index.read_bytes()).hexdigest(), "rows": [],
              "qualification": "Matched-history diagnostic; no engine is assumed to be ground truth.",
              "limitations": ["Record actual server checkpoint/backend/binary identities separately.",
                              "This interface exposes normalized log probabilities, not raw logits."]}
    def save():
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        for number, row in enumerate(rows):
            logits = read_row(source, row)
            tokens = row["input_tokens"]
            if not tokens or any(type(token) is not int or not 0 <= token < len(logits) for token in tokens):
                raise ValueError("Invalid recorded input history")
            body = {"prompt": tokens, "n_predict": 1, "temperature": 0.0, "top_k": 0, "top_p": 1.0,
                    "min_p": 0.0, "repeat_penalty": 1.0, "repeat_last_n": 0, "presence_penalty": 0.0,
                    "frequency_penalty": 0.0, "dry_multiplier": 0.0, "seed": 42, "cache_prompt": True,
                    "return_tokens": True, "n_probs": len(logits), "post_sampling_probs": False,
                    "samplers": ["temperature"], "stream": False}
            (args.output / f"request-{number:03}.json").write_text(json.dumps(body), encoding="utf-8")
            entry = {"row": number, "stage": row["stage"], "input_tokens": tokens, "source_sha256": row["sha256"]}
            report["rows"].append(entry)
            if not args.prepare_only:
                request = Request(args.server.rstrip("/") + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
                with urlopen(request, timeout=args.timeout) as response:
                    completion = json.load(response)
                (args.output / f"completion-{number:03}.json").write_text(json.dumps(completion, ensure_ascii=False), encoding="utf-8")
                entry["comparison"] = compare(logits, completion)
                entry["timings"] = completion.get("timings")
            save()
        report["run_complete"] = True
    except Exception as error:
        report["error"] = str(error)
        save()
        raise
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
