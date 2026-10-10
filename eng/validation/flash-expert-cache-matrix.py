#!/usr/bin/env python3
"""Measure explicit selected-expert cache caps in separate real-model processes.

Use only on idle hardware after builds finish. The probe retains matched prompt
IDs for qwen38-llama-reference.py. No language-quality pass is inferred from
finite logits or a length-limited answer; inspect the retained text separately.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=HERE / "Qwen4ExpExpertCacheProbe/bin/Release/net10.0/Qwen4ExpExpertCacheProbe.dll")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--caps-mib", default="2048,6144,8192")
    parser.add_argument("--rounds", type=int, default=2, help="Alternate cap order each round; OS file cache is not reset")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--prompt", default="Explain how a computer works to a curious beginner. Describe the CPU, memory, storage, input, and output in five connected paragraphs with a concrete example.")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    caps = [int(value) for value in args.caps_mib.split(",")]
    if not caps or min(caps) <= 0 or args.rounds < 1 or args.tokens < 1 or args.threads < 1 or args.timeout <= 0:
        parser.error("Caps, rounds, tokens, threads and timeout must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory")
    for path in (args.model, args.probe):
        if not path.is_file():
            parser.error("Missing file: " + str(path))
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    spec = importlib.util.spec_from_file_location("image_bench", HERE / "qwen-image21-bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    report = {"started_unix": time.time(), "prepare_only": args.prepare_only, "runs": [],
              "language_quality_passed": None, "limitations": [
                  "Process working-set and sampled board VRAM do not constitute a hard OS page-cache budget.",
                  "OS file cache is uncontrolled; these are not cold-storage measurements.",
                  "Caller must retain the independent checkpoint integrity report and ensure idle hardware.",
                  "Length-limited generation is an incomplete answer; language quality requires separate review."]}
    for round_index in range(args.rounds):
        for cap in caps if round_index % 2 == 0 else list(reversed(caps)):
            destination = output / f"round-{round_index + 1}-cap-{cap}"
            destination.mkdir()
            command = ["dotnet", str(args.probe.resolve()), "--model", str(args.model.resolve()),
                       "--placement", "host", "--backend", "ggml_cuda", "--generation", "greedy",
                       "--prompt", args.prompt, "--decode-tokens", str(args.tokens), "--max-context", str(max(512, args.tokens + 256)),
                       "--iterations", "1", "--warmup", "0", "--require-cache", "1", "--output", str(destination / "model.json")]
            env = dict(os.environ)
            overrides = dict(CUDA_VISIBLE_DEVICES="0", TS_CPU_MOE="1", TS_HOST_MOE_EXPERT_CACHE_MB=str(cap),
                       TS_HOST_MOE_EXPERT_CACHE_LAYERS="48", TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS="1",
                       TS_HOST_MOE_PIN="0", TS_HOST_MOE_EXPERT_CACHE_PREFETCH="0", TS_GGML_CPU_THREADS=str(args.threads),
                       TS_CPU_MOE_THREADS=str(args.threads),
                       OMP_NUM_THREADS=str(args.threads))
            env.update(overrides)
            for key in ("TENSORSHARP_TP_DEGREE", "TS_GGML_MEMORY_BUDGET", "TS_GGML_MEMORY_BUDGET_MB"):
                env.pop(key, None)
            row = {"cap_mib": cap, "round": round_index + 1, "command": command, "executed": False,
                   "environment": {key: env[key] for key in list(overrides) + ["TS_HOST_MOE_DEVICE_MIN_BATCH",
                       "TS_HOST_MOE_EXPERT_CACHE_BRIDGE", "TS_HOST_MOE_EXPERT_CACHE_OUTPUT_BRIDGE", "NVIDIA_TF32_OVERRIDE"] if key in env}}
            report["runs"].append(row)
            if not args.prepare_only:
                row.update(bench.run_process(command, destination / "process.log", args.timeout, env, True, 1.0))
                row["executed"] = True
                model_path = destination / "model.json"
                if model_path.exists():
                    model = json.loads(model_path.read_text(encoding="utf-8-sig"))
                    row.update({key: model.get(key) for key in ("model_load_ms", "native_sha256", "cache_stats",
                               "cache_stats_before_timed_work", "prompt_tokens", "generated_tokens", "selected_text", "runs")})
                    row["execution_passed"] = row["exit_code"] == 0 and not row["timed_out"] and model.get("passed") is True
                else:
                    row["execution_passed"] = False
            (output / "matrix.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"cap_mib": cap, "round": round_index + 1, "executed": row["executed"], "execution_passed": row.get("execution_passed")}), flush=True)
    if not args.prepare_only and all(row.get("execution_passed") for row in report["runs"]):
        first = report["runs"][0]
        report["identical_prompt_tokens"] = all(row["prompt_tokens"] == first["prompt_tokens"] for row in report["runs"])
        report["identical_generated_tokens"] = all(row["generated_tokens"] == first["generated_tokens"] for row in report["runs"])
    (output / "matrix.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if args.prepare_only or all(row.get("execution_passed") for row in report["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
