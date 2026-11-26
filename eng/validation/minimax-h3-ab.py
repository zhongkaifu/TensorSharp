#!/usr/bin/env python3
"""Alternate frozen H3 binaries and require exact decoded video/audio preservation.

Measures complete CLI processes, including loading and codecs. It does not claim
cold-cache latency, warm-service latency, or independent perceptual quality.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import shutil
import statistics
import subprocess
import time


def generation_helpers():
    spec = importlib.util.spec_from_file_location(
        "h3_generation", Path(__file__).with_name("minimax-h3-generation.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cli", type=Path, required=True)
    parser.add_argument("--candidate-cli", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", required=True,
                        choices=["ggml_metal", "mlx", "ggml_cuda", "cuda", "ggml_cpu", "cpu"])
    parser.add_argument("--checkpoint", choices=["fl2va", "ref2va"], default="fl2va")
    parser.add_argument("--bf16-pedantic", action="store_true",
                        help="Force pedantic F32 BF16 projections in both direct CUDA binaries, including older baselines")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--prompt", default="A red fox walks through a quiet forest, soft natural light, camera held steady. Forest birds chirp.")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--end-image", type=Path)
    parser.add_argument("--ref-image", type=Path, action="append", default=[])
    parser.add_argument("--ref-audio", type=Path, action="append", default=[])
    parser.add_argument("--lora", type=Path,
                        help="A minimax-h3-lora plug-in config for both binaries; its recipe sets the schedule, so --steps "
                             "is the count it fixes and no steps, shift or sampler are passed")
    parser.add_argument("--baseline-env", action="append", default=[], metavar="KEY=VALUE",
                        help="An environment variable for the baseline binary alone (TS_H3_PHASE=1 makes a pre-runtime "
                             "baseline print its step times; the runtime refuses that switch)")
    parser.add_argument("--allow-changed-media", action="store_true",
                        help="Record pairs whose decoded media differ instead of failing (a deliberate numerical change)")
    parser.add_argument("--ggml-source", type=Path,
                        default=Path(__file__).resolve().parents[2] / "ExternalProjects/ggml")
    parser.add_argument("--expected-ggml-revision", required=True,
                        help="Full revision used to build both frozen binaries")
    args = parser.parse_args()
    if args.bf16_pedantic and args.backend != "cuda":
        parser.error("--bf16-pedantic applies only to direct CUDA")
    if min(args.width, args.height, args.frames, args.fps, args.steps, args.rounds) <= 0:
        parser.error("Dimensions, schedule and rounds must be positive")
    if args.fps != 24 or args.width % 32 or args.height % 32 or args.frames % 17 != 5:
        parser.error("Require fps24, 32-pixel spatial grid and frame count 5+17*n")
    if args.output.exists():
        parser.error("Use a fresh output directory")
    for executable in ("dotnet", "ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            parser.error(f"Required executable is not on PATH: {executable}")
    helpers = generation_helpers()
    ggml = helpers.git_state(args.ggml_source)
    if not ggml["available"] or ggml["source_changed"]:
        parser.error("An unchanged upstream ggml checkout is required")
    if ggml["revision"] != args.expected_ggml_revision:
        parser.error("ggml does not match the revision used to build the frozen binaries")
    components = {name: args.models / filename for name, filename in helpers.COMPONENTS.items()}
    components["dit"] = args.models / f"minimax_h3_{args.checkpoint}_pruned-Q4_K.gguf"
    conditioning = [("--image", [args.image] if args.image else []),
                    ("--end-image", [args.end_image] if args.end_image else []),
                    ("--ref-image", args.ref_image), ("--ref-audio", args.ref_audio)]
    if (args.ref_image or args.ref_audio) and (args.image or args.end_image or args.checkpoint != "ref2va"):
        parser.error("Reference inputs require Ref2VA without keyframes")
    if (args.image or args.end_image) and args.checkpoint != "fl2va":
        parser.error("Keyframes require FL2VA")
    baseline_env = {}
    for item in args.baseline_env:
        key, sep, value = item.partition("=")
        if not sep or not key:
            parser.error(f"--baseline-env takes KEY=VALUE, not {item!r}")
        baseline_env[key] = value
    paths = [args.baseline_cli, args.candidate_cli, *components.values(), *([args.lora] if args.lora else []),
             *[args.models / filename for filename in ("vocab.json", "merges.txt", "tokenizer_config.json")],
             *[path for _, items in conditioning for path in items]]
    for path in paths:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    env = os.environ.copy()
    # Refuse silent diagnostic truncation and arithmetic-policy changes.
    inherited = {key: value for key, value in env.items()
                 if key.startswith(("TS_H3_", "TS_GGML_NODE_PROFILE", "SD_TRACE_")) and value}
    if inherited:
        parser.error(f"Clear inherited H3/profiling overrides before comparison: {sorted(inherited)}")
    env.update(TS_VIDEO_VAE=str(components["video_vae"].resolve()),
               TS_VIDEO_TEXT_ENCODER=str(components["text"].resolve()),
               TS_VIDEO_AUDIO_VAE=str(components["audio_vae"].resolve()),
               TS_VIDEO_TOKENIZER=str(args.models.resolve()),
               TS_CPU_THREADS="8", DOTNET_PROCESSOR_COUNT="8", OMP_NUM_THREADS="8", GGML_N_THREADS="8")
    if args.bf16_pedantic:
        env["TS_H3_CUDA_BF16_PEDANTIC"] = "1"
    args.output.mkdir(parents=True)
    report = {"backend": args.backend, "checkpoint": args.checkpoint,
              "precision_policy": "BF16 projections forced to pedantic F32 in both binaries" if args.bf16_pedantic else "binary defaults",
              "geometry": [args.width, args.height, args.frames], "fps": args.fps,
              "prompt": args.prompt, "seed": args.seed, "steps": args.steps,
              "cfg": 1, "flow_shift": None if args.lora else 12,
              "sigmas_f32": None if args.lora else helpers.sigmas(args.steps, 12),
              "lora": str(args.lora.resolve()) if args.lora else None, "baseline_env": baseline_env,
              "hardware": helpers.hardware(), "ggml": ggml,
              "binary_files": helpers.binary_artifacts(args.baseline_cli, args.baseline_cli)
                              + helpers.binary_artifacts(args.candidate_cli, args.candidate_cli),
              "components": {name: {"path": str(path.resolve()), "bytes": path.stat().st_size}
                             for name, path in components.items()},
              "tokenizer_sha256": {name: helpers.sha(args.models / name)
                                   for name in ("vocab.json", "merges.txt", "tokenizer_config.json")},
              "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
              "conditioning": [{"flag": flag, "sha256": helpers.sha(path), "path": str(path.resolve())}
                               for flag, items in conditioning for path in items],
              "environment": {key: value for key, value in env.items()
                              if key.startswith(("TS_", "GGML_", "OMP_", "DOTNET_"))},
              "timing_scope": "Alternating frozen CLI processes; includes model loading and codecs",
              "limitations": ["OS caches are uncontrolled; not warm-model service latency",
                              "Exact before/after preservation is not independent perceptual validation"],
              "executions": [], "pairs": [], "passed": False}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for repeat in range(args.rounds):
        records = {}
        versions = [("baseline", args.baseline_cli), ("candidate", args.candidate_cli)]
        for version, cli in versions if repeat % 2 == 0 else reversed(versions):
            output = (args.output / f"{version}-{repeat}.avi").resolve()
            command = ["dotnet", str(cli.resolve()), "--model", str(components["dit"].resolve()),
                       "--backend", args.backend, "--video-vae", str(components["video_vae"].resolve()),
                       "--video-text-encoder", str(components["text"].resolve()),
                       "--audio-vae", str(components["audio_vae"].resolve()), "--prompt", args.prompt,
                       "--width", str(args.width), "--height", str(args.height), "--video-frames", str(args.frames),
                       "--fps", str(args.fps), "--cfg", "1", "--diffusion-seed", str(args.seed)]
            # A plug-in's recipe fixes the steps, shifts and sampler (an explicit one is refused).
            command += ["--lora", str(args.lora.resolve())] if args.lora else \
                ["--diffusion-steps", str(args.steps), "--flow-shift", "12", "--sampler", "euler"]
            command += [value for flag, items in conditioning for path in items for value in (flag, str(path.resolve()))]
            command += ["--output", str(output)]
            record = {"version": version, "round": repeat, "argv": command, "passed": False}
            started = time.perf_counter()
            with (args.output / f"{version}-{repeat}.log").open("w") as log:
                run_env = {**env, **baseline_env} if version == "baseline" else env
                process = subprocess.Popen(command, env=run_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    record["exit_code"] = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    record.update(exit_code=124, timeout=True)
            record["wall_seconds"] = time.perf_counter() - started
            log_text = (args.output / f"{version}-{repeat}.log").read_text(errors="replace")
            # The diffusion runtime's step lines; a pre-runtime baseline prints its own with TS_H3_PHASE=1.
            record["denoise_step_seconds"] = [float(value) for value in re.findall(
                r"(?:\[diffusion\] minimax-h3 step \d+/\d+ sigma \S+ -> \S+|\[h3\] \.\. step \d+/\d+) ([0-9.]+)s", log_text)]
            if record["exit_code"] == 0:
                try:
                    record["media"], _, _ = helpers.validate_media(output, args, "tensorsharp")
                    # A pre-runtime baseline prints its steps only under TS_H3_PHASE=1 (--baseline-env), which a
                    # wall-clock comparison may leave off; the candidate always prints them.
                    if (version == "candidate" or record["denoise_step_seconds"]) and \
                            len(record["denoise_step_seconds"]) != args.steps:
                        raise RuntimeError("Denoising step count does not match the requested schedule")
                    record["passed"] = True
                except (RuntimeError, subprocess.TimeoutExpired, ValueError, OSError) as error:
                    record["error"] = str(error)
            report["executions"].append(record)
            records[version] = record
            save()
            print(json.dumps({key: record[key] for key in ("version", "round", "exit_code", "wall_seconds", "passed")}), flush=True)
            if not record["passed"]:
                return 1
        before, after = records["baseline"], records["candidate"]
        pair = {"round": repeat,
                "speedup": before["wall_seconds"] / after["wall_seconds"],
                "rgb_exact": before["media"]["decoded_rgb_sha256"] == after["media"]["decoded_rgb_sha256"],
                "audio_exact": before["media"]["decoded_audio_sha256"] == after["media"]["decoded_audio_sha256"]}
        report["pairs"].append(pair)
        save()
        if not (pair["rgb_exact"] and pair["audio_exact"]):
            if not args.allow_changed_media:
                print("FAIL: before/after decoded media changed", flush=True)
                return 1
            print("NOTE: before/after decoded media changed (--allow-changed-media)", flush=True)
    medians = {version: statistics.median(record["wall_seconds"] for record in report["executions"]
                                         if record["version"] == version) for version in ("baseline", "candidate")}
    report.update(passed=True, median_seconds=medians,
                  median_speedup=medians["baseline"] / medians["candidate"])
    report["ggml_after"] = helpers.git_state(args.ggml_source)
    if report["ggml_after"] != ggml:
        report.update(passed=False, error="ggml source changed during comparison")
    binary_after = helpers.binary_artifacts(args.baseline_cli, args.baseline_cli) + helpers.binary_artifacts(args.candidate_cli, args.candidate_cli)
    if binary_after != report["binary_files"]:
        report.update(passed=False, error="Frozen binaries changed during comparison")
    save()
    print(json.dumps({"passed": report["passed"], "median_seconds": medians, "median_speedup": report["median_speedup"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
