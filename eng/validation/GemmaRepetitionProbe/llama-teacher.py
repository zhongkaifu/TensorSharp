#!/usr/bin/env python3
"""Capture llama.cpp predictions for an explicit, identical teacher history.

The caller starts/stops a pinned server and records backend/model identity.
Each request supplies the full intended prefix; cache reuse cannot silently
substitute a previously sampled token when it differs from the teacher.
"""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-record", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8087")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    load = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
    prompt = load(args.prompt_record)
    teacher = load(args.teacher)
    if isinstance(teacher, dict):
        teacher = teacher.get("GeneratedTokens", teacher.get("tokens"))
    if not teacher or any(not isinstance(token, int) or token < 0 for token in teacher):
        raise ValueError("Teacher must provide nonnegative integer token IDs")
    results = []
    for step in range(min(args.steps, len(teacher))):
        prefix = prompt["PromptTokens"] + teacher[:step]
        body = {"prompt": prefix, "n_predict": 1, "temperature": 0.0,
                "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
                "repeat_last_n": 0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                "dry_multiplier": 0.0, "seed": 17, "cache_prompt": True,
                "return_tokens": True, "n_probs": 20, "post_sampling_probs": False,
                "samplers": ["temperature"], "stream": False}
        request = Request(args.server.rstrip("/") + "/completion",
                          json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
        with urlopen(request, timeout=7200) as response:
            completion = json.load(response)
        if len(completion.get("completion_probabilities", [])) != 1:
            raise ValueError(f"Expected one prediction at teacher step {step}")
        results.append({"Step": step, "InputTokens": prefix, "TeacherToken": teacher[step],
                        "Completion": completion})
        (args.output / "rows.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "qualification.json").write_text(json.dumps({
        "qualification": "Diagnostic matched-history predictions, not a language quality pass.",
        "model_sha256_requested": prompt["ModelSha256"], "teacher": str(args.teacher),
        "rows": len(results), "limitation": "Caller must record actual server model hash, backend and library identity."}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
