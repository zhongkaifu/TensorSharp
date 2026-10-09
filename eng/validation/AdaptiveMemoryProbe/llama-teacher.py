#!/usr/bin/env python3
"""Compare AdaptiveMemoryProbe captures with a caller-owned llama.cpp server.

All reports must use the same checkpoint and complete token histories. Different
TensorSharp binaries are allowed so a loader change can be judged against an
independent engine, rather than against its own previous outputs. Full-vocabulary
HTTP log probabilities are compared after a common-token offset, not as raw logits.
"""
import argparse
from contextlib import ExitStack
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.request import Request, urlopen


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


HERE = Path(__file__).resolve().parent
capture = module("adaptive_capture", HERE / "compare-captures.py")
reference = module("llama_capture", HERE.parent / "qwen38-llama-teacher.py")
require = capture.require


def validate_inputs(paths, identity):
    require(bool(paths), "Supply at least one complete capture")
    loaded = [capture.load(path) for path in paths]
    first, first_rows, _ = loaded[0]
    require(identity.get("model_sha256", "").lower() == first["ModelSha256"].lower(),
            "Server/checkpoint identity differs")
    require(capture.is_hash(identity.get("binary_sha256")) and identity.get("source_revision")
            and identity.get("command") and identity.get("startup_evidence"), "Incomplete server identity")
    for report, rows, data in loaded:
        for key in ("ModelSha256", "ModelBytes", "ModelGeometry", "Steps", "Repeats"):
            require(report[key] == first[key], "Checkpoint or geometry differs: " + key)
        require([row["InputTokens"] for row in rows] == [row["InputTokens"] for row in first_rows],
                "Teacher histories differ")
        with data.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(4 << 20), b""):
                digest.update(block)
            require(digest.hexdigest() == report["LogitCapture"]["Sha256"].lower(),
                    "Full capture hash differs")
    return loaded


def request_body(row):
    return {"prompt": row["InputTokens"], "n_predict": 1, "temperature": 0.0,
            "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
            "repeat_last_n": 0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "dry_multiplier": 0.0, "seed": 17, "cache_prompt": True,
            "return_tokens": True, "ignore_eos": True, "speculative.type": "none",
            "n_probs": row["Elements"], "post_sampling_probs": False,
            "samplers": ["temperature"], "stream": False}


def compare(logits, completion):
    require(completion.get("truncated") is False and completion.get("stop_type") == "limit"
            and completion.get("tokens_predicted") == 1 and len(completion.get("tokens", [])) == 1,
            "Incomplete reference prediction")
    settings = completion.get("generation_settings", {})
    for key, expected in (("temperature", 0), ("repeat_penalty", 1), ("presence_penalty", 0),
                          ("frequency_penalty", 0), ("dry_multiplier", 0), ("ignore_eos", True)):
        require(settings.get(key) == expected, "Unexpected reference sampling: " + key)
    result = reference.compare(logits, completion)
    require(completion["tokens"] == [result["argmax"][1]], "Reference token is not raw argmax")
    result["numerical_gate_passed"] = (result["relative_l2_after_common_token_offset"] <= .001
                                       and result["argmax"][0] == result["argmax"][1])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:5099")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"complete": False, "numerical_gate_passed": False, "semantic_quality_passed": None,
              "gate": {"relative_l2_after_common_token_offset": .001, "equal_argmax": True},
              "scope": "Matched-history full vocabulary only; not semantic quality or throughput. "
                       "HTTP softmax/log rounding remains; clipped values fail validation. "
                       "Caller must bind the server URL to the recorded binary/model and retain startup logs.",
              "identity_file": str(args.identity), "identity_sha256": hashlib.sha256(args.identity.read_bytes()).hexdigest(),
              "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for path in args.report], "rows": []}

    def save():
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    try:
        identity = json.loads(args.identity.read_text(encoding="utf-8-sig"))
        loaded = validate_inputs(args.report, identity)
        with ExitStack() as stack:
            streams = [stack.enter_context(data.open("rb")) for _, _, data in loaded]
            for number, row in enumerate(loaded[0][1]):
                body = request_body(row)
                (args.output / f"request-{number:03}.json").write_text(json.dumps(body), encoding="utf-8")
                request = Request(args.server.rstrip("/") + "/completion", json.dumps(body).encode(),
                                  {"Content-Type": "application/json"})
                with urlopen(request, timeout=args.timeout) as response:
                    completion = json.load(response)
                with gzip.open(args.output / f"completion-{number:03}.json.gz", "wt", encoding="utf-8") as output:
                    json.dump(completion, output, ensure_ascii=False)
                entry = {"run": row["Run"], "step": row["Step"], "comparisons": []}
                report["rows"].append(entry)
                for index, (_, rows, _) in enumerate(loaded):
                    logits, _ = capture.read_row(streams[index], rows[number])
                    entry["comparisons"].append(compare(logits, completion))
                save()
        report["complete"] = True
        report["numerical_gate_passed"] = all(item["numerical_gate_passed"] for row in report["rows"]
                                               for item in row["comparisons"])
    except Exception as error:
        report["error"] = str(error)
    save()
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0 if report["numerical_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
