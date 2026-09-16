#!/usr/bin/env python3
"""Derive HTTP text/media plans from the independently verified model inventory.

This only writes plans. Default multi-GPU GLM and DeepSeek placement remains
whole-layer splitting; explicitly requested tensor parallelism is a separate
campaign. Requested settings must be reconciled with native startup banners.
"""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", default="/workspace/tensorsharp-no-patch-20260915/media-fixtures")
    parser.add_argument("--models", help="Optional comma-separated exact catalog IDs")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--gpu-overrides", default="", help="Reviewed exact model=GPU-count overrides, comma separated; retained in the plan")
    args = parser.parse_args()
    if len(args.native_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.native_sha256):
        parser.error("Expected lowercase SHA256 native digest")
    requested = set(args.models.split(",")) if args.models else None
    gpu_ids = args.gpus.split(",")
    try:
        gpu_overrides = {model: int(count) for model, count in
                         (item.split("=", 1) for item in args.gpu_overrides.split(",") if item)}
    except ValueError:
        parser.error("--gpu-overrides must contain model=positive-integer entries")
    if any(count < 1 or count > len(gpu_ids) for count in gpu_overrides.values()):
        parser.error("GPU overrides must fit the selected visible GPU count")
    inventory = json.loads(args.inventory.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"source_inventory": str(args.inventory), "native_sha256": args.native_sha256,
               "status": "plans-only", "models": []}
    for model in inventory["models"]:
        if requested and model["id"] not in requested:
            continue
        if not set(model.get("modalities", [])) & {"text", "text_diffusion"}:
            continue
        entry = {"id": model["id"]}
        summary["models"].append(entry)
        if model["status"] != "downloaded-and-verified":
            entry.update(status="not-ready", reason="Checkpoint download/digest verification incomplete")
            continue
        primary = [item for item in model["files"] if item["path"].endswith(".gguf") and
                   not any(word in Path(item["path"]).name.lower() for word in
                           ("mmproj", "dflash", "dspark", "mtp-gemma", "assistant", "drafter"))]
        companions = [item for item in model["files"] if "mmproj" in Path(item["path"]).name.lower()]
        if not primary:
            entry.update(status="not-ready", reason="No primary GGUF in inventory")
            continue
        weight_bytes = sum(item["bytes"] for item in primary)
        # A capacity plan, not a measured allocation. Leave room for context,
        # projection and concurrent request state; the loader remains authoritative.
        degree = max(1, math.ceil(weight_bytes / (32 * 1024 ** 3)))
        if model["id"] in {"mistral3", "muse-glimmer", "nemotron-h47b", "qwen36-27b", "qwen38-27b-nvfp4"}:
            degree = max(degree, 2)
        if model["id"] in gpu_overrides:
            entry["capacity_override"] = {"conservative_gpu_count": degree, "reviewed_gpu_count": gpu_overrides[model["id"]],
                                          "qualification": "Explicit plan override; actual allocation remains an execution gate."}
            degree = gpu_overrides[model["id"]]
        if degree > len(gpu_ids):
            entry.update(status="capacity-review-required", primary_bytes=weight_bytes,
                         reason=f"The conservative 32 GiB/GPU planning budget selects {degree} GPUs; only {len(gpu_ids)} selected. Review actual cache/graph capacity before overriding or adding CPU offload; this is not proof of hardware infeasibility.")
            continue
        profile_id = model["id"] + (f"-http-gpu{degree}-diffusion" if model["id"] == "diffusiongemma" else f"-http-gpu{degree}-f16")
        revision = primary[0]["revision"]
        weights_id = revision + ":" + primary[0]["download_verified_sha256"]
        profile = {"id": profile_id, "model": primary[0]["path"], "backend": "ggml_cuda", "port": 5120,
                   "expected_native_sha256": args.native_sha256, "telemetry": True,
                   "before_suites": [["{python}", "{repo}/eng/validation/control-release-downloads.py", "pause",
                                      "--run-root", "/workspace/tensorsharp-no-patch-20260915",
                                      "--state", "/workspace/tensorsharp-no-patch-20260915/logs/download-pause.json"]],
                   "before_suites_quiet_seconds": 20,
                   "artifact_inventory": {"weights": primary, "companions": companions},
                   "env": {"CUDA_VISIBLE_DEVICES": ",".join(gpu_ids[:degree]), "MAX_CONTEXT": "65536",
                           "KV_CACHE_DTYPE": "f16", "TS_SCHED_MAX_RUNNING_SEQS": "4",
                           "TS_SCHED_MAX_BATCHED_TOKENS": "4096", "TS_SCHED_PREFILL_CHUNK": "512"},
                   "extra_args": ["--no-skills"], "suites": []}
        if degree > 1 and not model["id"].startswith(("glm", "deepseek")):
            profile["extra_args"] += ["--tp", str(degree)]
        if companions:
            profile["extra_args"] += ["--mmproj", companions[0]["path"]]
        common = ["{python}", "{repo}/benchmarks/engine_comparison/validate_inference.py",
                  "--url", "{url}", "--engine", "tensorsharp", "--model", "{model_id}",
                  "--weights-id", weights_id, "--profile", profile_id, "--timeout", "1800"]
        for scenarios, degree_list, repeats, filename, extras in (
            ("short,short_zh,json,json_schema,json_unicode,multi_turn,tool_round_trip,agentic", "1,4", 1, "quality", []),
            ("tool_round_trip,agentic", "1,4", 1, "structured-tools", ["--structured-tool-results"]),
            ("long_8k,long_32k,long_64k", "1", 1, "long-context", []),
            ("decode,decode_8k", "1,4", 3, "decode", []),
        ):
            if model["id"] == "mistral3":
                if filename == "structured-tools":
                    continue
                scenarios = ",".join(name for name in scenarios.split(",") if name not in ("tool_round_trip", "agentic"))
            profile["suites"].append(common + ["--scenarios", scenarios, "--concurrency", degree_list,
                "--repeats", str(repeats), *extras, "--output", "{output}/" + filename + ".json"])
        if model["id"] not in {"mistral3", "hunyuan-dense"}:
            tool_scenarios = "required,named,none_history,single_call,two_calls,required_thinking,named_thinking,invalid_required,invalid_named"
            profile["suites"].append(["{python}", "{repo}/benchmarks/engine_comparison/validate_deepseek41_tools.py",
                "--url", "{url}", "--model", "{model_id}", "--weights-id", weights_id, "--profile", profile_id,
                "--scenarios", tool_scenarios, "--concurrency", "1,4", "--output", "{output}/tool-policies.json"])
        if companions and "image" in model.get("modalities", []):
            scenarios = "image_ocr,multi_image,image_follow_up,image_long_context"
            if "video" in model["modalities"]:
                scenarios += ",video_order,video_timestamp"
            profile["suites"].append(["{python}", "{repo}/benchmarks/engine_comparison/validate_deepseek41_media.py",
                "--url", "{url}", "--model", "{model_id}", "--weights-id", weights_id,
                "--companion-sha256", companions[0]["download_verified_sha256"], "--profile", profile_id,
                "--fixtures", args.fixtures, "--scenarios", scenarios, "--concurrency", "1,4",
                "--output", "{output}/media.json"])
        profile["scope_note"] = "Plan only. HTTP strict semantics and media perception supplement native A/B tokens. Actual confined skills/shell execution, optional drafters, explicit alternate KV/TP/CPU placements and audio are separate lanes."
        if model["id"] == "mistral3":
            profile["scope_note"] += " Mistral3's current forced TensorSharp renderer declares no tool declaration/result support; tool/agent generation cases are excluded as unsupported, not passed."
        if model["id"] == "hunyuan-dense":
            profile["suites"] = [["{python}", "{repo}/eng/validation/validate-release-translation.py",
                                  "--url", "{url}", "--model", "{model_id}", "--concurrency", "1,4",
                                  "--repeats", "3", "--output", "{output}/translation.json"]]
            profile["scope_note"] = "Hy-MT2 specialized translation checkpoint: Chinese/English/French, JSON preservation, delimiters and 48-sentence completeness. These are authored fixtures, not general arithmetic/tools or broad linguistic evaluation. Single-device only."
        if model["id"] == "diffusiongemma":
            profile["env"].update(DIFFUSION_STEPS="48", DIFFUSION_MAX_BATCH="2")
            profile["suites"] = [["{python}", "{repo}/eng/validation/validate-release-diffusion.py",
                                  "--url", "{url}", "--model", "{model_id}", "--repeats", "3",
                                  "--server-log", "{output}/server.log", "--output", "{output}/diffusion.json"]]
            profile["scope_note"] = "Block diffusion only: OpenAI final text, Web UI replacement previews, concurrent/late requests, disconnect recovery and unsupported-tool rejection. Whole-generation latency is not autoregressive decode throughput. HTTP sampler seed is server-generated; exact token parity is not a valid seeded HTTP assumption. Preview frames do not expose block index; exact admission boundaries require separate scheduler tests."
        path = args.output / (profile_id + ".json")
        path.write_text(json.dumps(profile, indent=2) + "\n")
        entry.update(status="planned", profile=str(path), primary_bytes=weight_bytes, selected_gpus=degree)
    (args.output / "plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    for item in summary["models"]:
        print(item["id"], item["status"], item.get("reason", ""))


if __name__ == "__main__":
    main()
