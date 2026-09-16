#!/usr/bin/env python3
"""Check a trained translation checkpoint over the real OpenAI chat API.

The Hy-MT2 publisher describes translation, terminology and structured-data
tasks: https://huggingface.co/tencent/Hy-MT2-1.8B . These authored fixtures
exercise those contracts; they are not a broad translation benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/engine_comparison"))
import engines
from validate_inference import SAMPLING, assistant_content


def specification(scenario, tag):
    if scenario == "zh_en":
        return (f"Translate this Chinese passage into English. Return the translation only.\n"
                f"<source>红色盒子里有三本书。核验编号是 {tag}。</source>",
                lambda output: all(re.search(pattern, output, re.I) for pattern in
                    (r"red\s+box", r"three\s+books|3\s+books", re.escape(tag))))
    if scenario == "en_zh":
        return (f"把下面英文译成中文，只输出译文：\n<source>The blue box contains five apples. Verification code: {tag}.</source>",
                lambda output: "蓝" in output and "苹果" in output and any(number in output for number in ("五", "5")) and tag in output)
    if scenario == "fr_en":
        return (f"Translate this French passage into English, returning only the translation.\n"
                f"<source>Le train part à huit heures. Code de vérification : {tag}.</source>",
                lambda output: "train" in output.lower() and bool(re.search(r"\beight\b|\b8\b", output, re.I)) and tag in output)
    if scenario == "structured_json":
        prompt = ("Translate only the user-visible JSON string values into Chinese. Keep keys, numbers, booleans and the code unchanged. "
                  "Return valid JSON only.\n" + json.dumps({"greeting": "Hello", "color": "red", "count": 3, "ready": True, "code": tag}))
        def check(output):
            data = json.loads(output)
            return (set(data) == {"greeting", "color", "count", "ready", "code"}
                    and data["greeting"] in ("你好", "您好") and data["color"] in ("红", "红色")
                    and type(data["count"]) is int and data["count"] == 3
                    and type(data["ready"]) is bool and data["ready"] is True and data["code"] == tag)
        return prompt, check
    if scenario == "delimiters":
        return (f"Translate each English word into Chinese. Preserve exactly both || delimiters and the code. Return only the translation.\nred || blue || {tag}",
                lambda output: output.count("||") == 2 and "红" in output.split("||")[0]
                    and "蓝" in output.split("||")[1] and output.split("||")[2].strip() == tag)
    if scenario == "long_translation":
        source = "\n".join(f"第 {index} 个红色盒子里有三本书。" for index in range(1, 49)) + f"\n核验编号是 {tag}。"
        return ("Translate the complete Chinese passage into English. Preserve every numbered statement. Return only the translation.\n" + source,
                lambda output: tag in output and len(re.findall(r"red\s+box", output, re.I)) == 48
                    and len(re.findall(r"three\s+books|3\s+books", output, re.I)) == 48)
    raise ValueError(scenario)


def run_case(args, scenario, tag):
    prompt, check = specification(scenario, tag)
    request = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 4096,
               "stream": True, "extra_body": {**SAMPLING, **engines.thinking_body("tensorsharp", False)}}
    result = {"scenario": scenario, "tag": tag, "status": "fail", "request": request,
              "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()}
    try:
        metrics = engines.run_openai_chat(args.url, args.model, timeout_s=args.timeout, **request)
        result["metrics"] = metrics
        if metrics.get("finish_reason") != "stop" or not metrics.get("usage_present"):
            raise ValueError("Missing token usage or completed final answer")
        if not check(assistant_content(metrics).strip()):
            raise ValueError("Translation fails independent meaning/structure checks")
        result["status"] = "ok"
    except Exception as error:
        result["error"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", default="1,4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    report = {"model": args.model, "url": args.url, "started_at_unix": time.time(), "run_complete": False,
              "scope": "Authored translation, delimiter, JSON and 48-sentence completeness fixtures; not broad linguistic evaluation.",
              "publisher_task_reference": "https://huggingface.co/tencent/Hy-MT2-1.8B", "warmup": [], "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for scenario in ("zh_en", "en_zh", "fr_en", "structured_json", "delimiters", "long_translation"):
        report["warmup"].append(run_case(args, scenario, "ts-warm-" + scenario))
        for degree in map(int, args.concurrency.split(",")):
            for repeat in range(args.repeats):
                with ThreadPoolExecutor(max_workers=degree) as pool:
                    cases = list(pool.map(lambda index: run_case(args, scenario, f"ts-{scenario}-{degree}-{repeat}-{index}"), range(degree)))
                for case in cases:
                    case.update(concurrency=degree, repeat=repeat)
                report["cases"].extend(cases)
                args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
                print(scenario, degree, repeat, [item["status"] for item in cases], flush=True)
    report["run_complete"] = True
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return int(any(item["status"] != "ok" for item in report["warmup"] + report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
