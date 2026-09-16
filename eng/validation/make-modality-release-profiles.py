#!/usr/bin/env python3
"""Prepare release modality profiles from independently verified model inventory.

This writes plans only. Missing or unverified companions stay pending in the
manifest; only fully provisioned profiles are emitted for lifecycle execution.
The baseline must run from an isolated app directory with its own native library.
"""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
from native_release_evidence import require_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", default="/workspace/tensorsharp-no-patch-20260915")
    parser.add_argument("--port", type=int, default=5140)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    args = parser.parse_args()
    try:
        pins = {"baseline": require_sha256(args.baseline_sha256), "candidate": require_sha256(args.candidate_sha256)}
    except ValueError as error:
        parser.error(str(error))
    inventory = json.loads(args.inventory.read_text())
    models = {item["id"]: item for item in inventory["models"]}
    base = PurePosixPath(args.base)
    plan_root = base / "modality-plans"
    if args.output.exists():
        parser.error("Refusing to overwrite an existing profile directory")
    args.output.mkdir(parents=True)
    manifest = {"inventory": str(args.inventory), "status": "prepared-not-executed", "profiles": [],
                "inventory_sha256": hashlib.sha256(args.inventory.read_bytes()).hexdigest(), "native_sha256": pins,
                "pending": [], "reference_policy": "Same weights, settings and managed binaries; isolated old/new native app copies. Managed CPU profiles require no native mappings."}

    def emit(name, kind, model_spec, companions=(), backend="ggml_cuda", extra=(), env=None, plan=None):
        assets, missing = [], []

        def resolve(spec):
            model_id, pattern = spec
            entry = models.get(model_id, {})
            matches = [item for item in entry.get("files", [])
                       if fnmatch.fnmatch(PurePosixPath(item["path"]).name, pattern)]
            if len(matches) != 1:
                missing.append({"model": model_id, "pattern": pattern, "reason": f"Expected one matching artifact, found {len(matches)}"})
                return ""
            item = matches[0]
            if entry.get("status") != "downloaded-and-verified":
                missing.append({"model": model_id, "pattern": pattern, "reason": entry.get("status", "not in inventory")})
            assets.append(item)
            return item["path"]

        model_path = resolve(model_spec)
        flags = ["--host", "127.0.0.1", "--no-webui", "--no-prefix-cache", *extra]
        environment = {"CUDA_VISIBLE_DEVICES": "0", **(env or {})}
        for flag, spec in companions:
            value = resolve(spec)
            if flag.startswith("env:"):
                environment[flag[4:]] = str(PurePosixPath(value).parent)
            elif flag:
                flags += [flag, value]
        if missing:
            manifest["pending"].append({"id": name, "kind": kind, "requirements": missing})
            return
        for revision in ("baseline", "candidate"):
            profile_id = f"{name}-{revision}"
            command = ["{python}", f"{{repo}}/eng/validation/validate-release-{kind}.py", "--url", "{url}"]
            if kind == "generation":
                command += ["--plan", str(plan_root / (plan + ".json")), "--output", "{output}/generation"]
                reference = base / "modalities/baseline" / f"{name}-baseline" / "generation/results.json"
            else:
                command += ["--model", "{model_id}", "--output", f"{{output}}/{kind}.json"]
                reference = base / "modalities/baseline" / f"{name}-baseline" / f"{kind}.json"
                if kind == "audio":
                    command += ["--audio", str(base / "media-fixtures/spoken-fox.wav")]
            if revision == "candidate" and kind != "audio":
                command += ["--reference", str(reference)]
            profile = {"id": profile_id, "model": model_path, "backend": backend, "port": args.port,
                       "expected_native_sha256": pins[revision],
                       "extra_args": flags, "env": environment, "suites": [command],
                       "native_policy": "absent" if backend == "cpu" else "exact",
                       "model_assets": assets, "validation_kind": kind, "revision_role": revision}
            path = args.output / (profile_id + ".json")
            path.write_text(json.dumps(profile, indent=2) + "\n")
            manifest["profiles"].append({"id": profile_id, "kind": kind, "path": str(path),
                                         "output": str(base / "modalities" / revision / profile_id)})

    for model in ("minilm", "snowflake"):
        for backend in ("cpu", "ggml_cpu", "ggml_cuda"):
            emit(f"{model}-{backend}", "embeddings", (model, "*.gguf"), backend=backend,
                 extra=("--embeddings", "--embedding-threads", "8"))

    for model in ("gemma4-e4b", "gemma4-12b", "gemma4-26b-qat", "gemma4-26b", "gemma4-31b"):
        pattern = {"gemma4-e4b": "gemma-4-E4B-it-Q8_0.gguf", "gemma4-12b": "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
                   "gemma4-26b-qat": "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", "gemma4-26b": "gemma-4-26B-A4B-it-Q4_0.gguf",
                   "gemma4-31b": "gemma-4-31B-it-Q4_0.gguf"}[model]
        emit(model + "-audio", "audio", (model, pattern), companions=(("--mmproj", (model, "mmproj*.gguf")),),
             extra=("--no-spec", "--max-tokens", "512"))

    qwen = (("--qwen-image-vae", ("qwen-image-vae", "Qwen_Image-VAE.safetensors")),
            ("--qwen-image-vl", ("qwen-image-vl", "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")),
            ("--qwen-image-mmproj", ("qwen-image-vl", "mmproj-BF16.gguf")))
    emit("qwen-image-base", "generation", ("qwen-image", "*.gguf"), qwen, plan="qwen-image")
    emit("qwen-image-lightning", "generation", ("qwen-image", "*.gguf"),
         qwen + (("--qwen-image-lora", ("qwen-image-lightning", "*.safetensors")),), plan="qwen-image")
    for variant in ("fl2va", "ref2va"):
        emit("h3-" + variant, "generation", ("minimax-h3", f"minimax_h3_{variant}_pruned-Q4_K.gguf"),
             (("--video-text-encoder", ("minimax-h3", "qwen3vl_32b_minimax_h3-Q4_K_M.gguf")),
              ("--video-vae", ("minimax-h3", "minimax_h3_video_vae_fp16.safetensors")),
              ("--audio-vae", ("minimax-h3", "minimax_h3_audio_vae_fp32.safetensors")),
              ("env:TS_VIDEO_TOKENIZER", ("minimax-h3-tokenizer", "vocab.json")),
              ("", ("minimax-h3-tokenizer", "merges.txt"))), plan="h3-" + variant)
    for model in ("wan-turbo", "wan-base", "wan-a14b", "wan21-13b", "wan21-14b", "wan22-t2v-a14b", "wan22-i2v-a14b-base"):
        ti2v = model in ("wan-turbo", "wan-base")
        dual = model in ("wan-a14b", "wan22-t2v-a14b", "wan22-i2v-a14b-base")
        high = "*high_noise*gguf" if model == "wan-a14b" else "*HighNoise*gguf"
        low = "*low_noise*gguf" if model == "wan-a14b" else "*LowNoise*gguf"
        companions = [("--video-text-encoder", ("umt5-encoder", "*.gguf")),
                      ("--video-vae", ("wan-vae" if ti2v else "wan-a14b-vae", "*.safetensors"))]
        if dual:
            companions.append(("--video-dit2", (model, low)))
        plan = "wan-ti2v" if ti2v else "wan-i2v" if model in ("wan-a14b", "wan22-i2v-a14b-base") else "wan21-t2v" if model.startswith("wan21") else "wan-t2v"
        emit(model, "generation", (model, high if dual else "*.gguf"), companions, plan=plan)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(manifest['profiles'])} profiles; {len(manifest['pending'])} workloads need verified artifacts.")


if __name__ == "__main__":
    main()
