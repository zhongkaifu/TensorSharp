#!/usr/bin/env python3
"""Check trained audio perception with a recorded speech fixture over real HTTP.

Expected recording: The quick brown fox jumps over the lazy dog. Release
verification code is four eight two one. The prompt never supplies that answer.
The fixture is retained once by path/hash instead of repeating base64 in reports.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/engine_comparison"))
import engines
from validate_inference import SAMPLING, assistant_content


def run_case(args, attachment, scenario, tag):
    prompt = "Transcribe the recording accurately. Return only the transcript." if scenario == "transcribe" else "What verification code is spoken in the recording? Return only its four digits, without spaces."
    request = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": f"[validation {tag}]\n{prompt}"}, attachment]}],
        "extra_body": {**SAMPLING, **engines.thinking_body("tensorsharp", False)},
        "max_tokens": 256, "stream": True}
    result = {"scenario": scenario, "tag": tag, "status": "fail", "prompt": prompt,
              "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()}
    try:
        metrics = engines.run_openai_chat(args.url, args.model, timeout_s=args.timeout, **request)
        result["metrics"] = metrics
        output = assistant_content(metrics).strip()
        if metrics["finish_reason"] != "stop":
            raise ValueError("Response did not finish normally")
        if scenario == "code":
            if output != "4821":
                raise ValueError("Spoken code was not returned as exactly four digits")
        else:
            normalized = " ".join(re.findall(r"\w+", output.lower()))
            if "the quick brown fox jumps over the lazy dog" not in normalized:
                raise ValueError("Transcript omits or changes the recorded sentence")
            if not any(code in normalized for code in ("four eight two one", "4821", "4 8 2 1")):
                raise ValueError("Transcript omits or changes the spoken verification code")
        result["status"] = "ok"
    except Exception as error:
        result["detail"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", default="1,4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    degrees = list(map(int, args.concurrency.split(",")))
    if args.repeats < 1 or not degrees or any(degree < 1 for degree in degrees):
        parser.error("repeats and concurrency must be positive")
    data = args.audio.read_bytes()
    attachment = {"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode(), "format": "wav"}}
    with wave.open(str(args.audio), "rb") as source:
        metadata = {"channels": source.getnchannels(), "sample_rate": source.getframerate(),
                    "samples": source.getnframes(), "sample_width_bytes": source.getsampwidth()}
    report = {"model": args.model, "url": args.url, "started_at_unix": time.time(),
              "audio": {"path": str(args.audio), "sha256": hashlib.sha256(data).hexdigest(), **metadata},
              "run_complete": False, "warmup": [], "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for scenario in ("transcribe", "code"):
        report["warmup"].append(run_case(args, attachment, scenario, f"audio-{scenario}-warmup"))
        for degree in degrees:
            for repeat in range(args.repeats):
                with ThreadPoolExecutor(max_workers=degree) as pool:
                    cases = list(pool.map(lambda index: run_case(args, attachment, scenario,
                        f"audio-{scenario}-c{degree}-r{repeat}-i{index}"), range(degree)))
                for case in cases:
                    case.update(concurrency=degree, repeat=repeat)
                report["cases"].extend(cases)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(scenario, "concurrency", degree, "repeat", repeat,
                      [case["status"] for case in cases], flush=True)
    report["run_complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return int(any(case["status"] != "ok" for case in report["cases"] + report["warmup"]))


if __name__ == "__main__":
    raise SystemExit(main())
