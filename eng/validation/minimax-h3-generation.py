#!/usr/bin/env python3
"""Run paired H3 CLI generation with explicit matched noise/sampling inputs.

Whole-process times include model loading and output encoding. Repeated CLI
processes do not establish warm-model latency. Media comparison includes each
CLI's output codec and is diagnostic, rather than a substitute for float oracles.
"""
import argparse
import array
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import signal
import struct
import subprocess
import time
from fractions import Fraction

COMPONENTS = {
    "dit": "minimax_h3_fl2va_pruned-Q4_K.gguf",
    "text": "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
}


def git_state(path):
    state={"path":str(path.resolve()),"available":False,"revision":None,"source_changed":None}
    revision=subprocess.run(["git","-C",str(path),"rev-parse","HEAD"],capture_output=True,text=True)
    if revision.returncode:
        return state
    status=subprocess.run(["git","-C",str(path),"status","--porcelain","--untracked-files=normal"],capture_output=True,text=True)
    if status.returncode:
        return state
    state.update(available=True,revision=revision.stdout.strip(),source_changed=bool(status.stdout.strip()),status=status.stdout)
    return state


def hardware():
    result={"platform":platform.platform(),"machine":platform.machine(),"processor":platform.processor(),"logical_cpus":os.cpu_count()}
    if hasattr(os,"sched_getaffinity"):
        result["affinity_cpus"]=len(os.sched_getaffinity(0))
    for name,path in [("cpu_quota","/sys/fs/cgroup/cpu.max"),("memory_limit","/sys/fs/cgroup/memory.max")]:
        if Path(path).is_file(): result[name]=Path(path).read_text().strip()
    if "cpu_quota" not in result:
        quota,period=[Path("/sys/fs/cgroup/cpu")/name for name in ("cpu.cfs_quota_us","cpu.cfs_period_us")]
        if quota.is_file() and period.is_file():
            result["cpu_quota"]=quota.read_text().strip()+" "+period.read_text().strip()
            result["cgroup_cpu_version"]=1
    if "memory_limit" not in result:
        limit=Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if limit.is_file(): result["memory_limit"]=limit.read_text().strip()
    if Path("/proc/meminfo").is_file():
        result["memory_total"]=next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:"))
    if Path("/proc/cpuinfo").is_file():
        result["cpu_model"]=next((line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),None)
    if platform.system()=="Darwin":
        for name in ("hw.memsize","hw.physicalcpu","hw.logicalcpu","machdep.cpu.brand_string"):
            query=subprocess.run(["sysctl","-n",name],capture_output=True,text=True)
            if query.returncode==0: result[name]=query.stdout.strip()
        displays=subprocess.run(["system_profiler","SPDisplaysDataType","-json"],capture_output=True,text=True,timeout=30)
        if displays.returncode==0:
            result["gpu"]=[{key:value for key,value in item.items() if key in ("_name","sppci_model","sppci_cores","spdisplays_vram","spdisplays_metal")}
                           for item in json.loads(displays.stdout).get("SPDisplaysDataType",[])]
    else:
        try:
            gpu=subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,memory.total,pci.bus_id","--format=csv,noheader"],capture_output=True,text=True,timeout=10)
            if gpu.returncode==0: result["gpu"]=gpu.stdout.strip().splitlines()
        except FileNotFoundError:
            pass
    runtime=subprocess.run(["dotnet","--info"],capture_output=True,text=True,timeout=30)
    result["dotnet_info"]=runtime.stdout if runtime.returncode==0 else None
    return result


def binary_artifacts(cli,baseline):
    files={baseline.resolve(),cli.resolve()}
    parent=cli.parent
    files.update(parent.glob("*.dll"))
    files.update(parent.glob("*.dylib"))
    files.update(parent.glob("*.so"))
    files.update(parent.glob("*.metallib"))
    files.update(parent.glob("cuda_kernels/*.ptx"))
    return [{"path":str(path.resolve()),"sha256":sha(path)} for path in sorted(files) if path.is_file()]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def sigmas(steps, shift):
    result = []
    for index in range(steps):
        timestep = 999.0 if steps == 1 else 999.0 - index * (999.0 / (steps - 1))
        t = (timestep + 1.0) / 1000.0
        result.append(f32(shift * t / (1.0 + (shift - 1.0) * t)))
    return result + [0.0]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path):
    process = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
                              "-of", "json", str(path)], capture_output=True, timeout=60)
    if process.returncode:
        raise RuntimeError(process.stderr.decode(errors="replace"))
    return json.loads(process.stdout)


def decode(path, media):
    command = ["ffmpeg", "-v", "error", "-i", str(path)]
    command += (["-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"] if media == "video"
                else ["-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"])
    process = subprocess.run(command, capture_output=True, timeout=120)
    if process.returncode:
        raise RuntimeError(process.stderr.decode(errors="replace"))
    return process.stdout


def validate_media(path, args, engine):
    details = probe(path)
    video = next((stream for stream in details["streams"] if stream["codec_type"] == "video"), None)
    if not video:
        raise RuntimeError("No video stream")
    count = int(video.get("nb_read_frames", video.get("nb_frames", -1)))
    if (int(video["width"]), int(video["height"]), count) != (args.width, args.height, args.frames):
        raise RuntimeError(f"Unexpected video geometry: {video}")
    if Fraction(video.get("avg_frame_rate", "0/1")) != args.fps:
        raise RuntimeError(f"Unexpected video frame rate: {video.get('avg_frame_rate')}")
    audio_path = path.with_suffix(".wav")
    if not audio_path.is_file():
        audio_path = path
    audio_details = probe(audio_path)
    audio = next((stream for stream in audio_details["streams"] if stream["codec_type"] == "audio"), None)
    if not audio or (int(audio["sample_rate"]), int(audio["channels"])) != (32000, 2):
        raise RuntimeError(f"Expected 32 kHz stereo audio: {audio}")
    audio_raw = decode(audio_path, "audio")
    waveform = array.array("f")
    waveform.frombytes(audio_raw)
    if not waveform or not all(math.isfinite(value) for value in waveform):
        raise RuntimeError("Empty or nonfinite audio")
    duration = len(waveform) / (2 * 32000)
    # Clean SD H3 emits the full 40Hz latent grid. TensorSharp trims excess
    # samples to video duration, but neither implementation pads a short grid.
    latent_samples = math.floor(args.frames * 40 / args.fps + .5) * 800
    expected_samples = latent_samples if engine == "sdcpp" else min(latent_samples, round(args.frames / args.fps * 32000))
    if len(waveform) != 2 * expected_samples:
        raise RuntimeError(f"Audio length mismatch: {len(waveform)//2} samples, expected {expected_samples}")
    video_raw = decode(path, "video")
    if len(video_raw) != args.width * args.height * args.frames * 3:
        raise RuntimeError("Decoded video length mismatch")
    return {"video": video, "audio": audio, "audio_path": str(audio_path), "audio_seconds": duration,
            "audio_peak": max(abs(value) for value in waveform),
            "audio_rms": math.sqrt(sum(value * value for value in waveform) / len(waveform)),
            "decoded_rgb_sha256": hashlib.sha256(video_raw).hexdigest(),
            "decoded_audio_sha256": hashlib.sha256(audio_raw).hexdigest(),
            "files": [{"path": str(item), "bytes": item.stat().st_size, "sha256": sha(item)}
                      for item in dict.fromkeys([path, audio_path])]}, video_raw, waveform


def comparison(reference, actual):
    rv, ra = reference
    tv, ta = actual
    import numpy as np
    result = {"scope": "decoded CLI output; includes codec differences"}
    if len(rv) == len(tv):
        a, b = np.frombuffer(rv, dtype=np.uint8).astype(np.float64), np.frombuffer(tv, dtype=np.uint8).astype(np.float64)
        mse = float(np.mean((a - b) ** 2))
        result["rgb_psnr_db"] = None if mse == 0 else 10 * math.log10(255 ** 2 / mse)
        result["rgb_max_abs"] = float(np.max(np.abs(a - b)))
        result["rgb_rmse"] = math.sqrt(mse)
    common = min(len(ra), len(ta))
    if common:
        a, b = np.asarray(ra[:common], dtype=np.float64), np.asarray(ta[:common], dtype=np.float64)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        result["audio_cosine"] = float(np.dot(a, b)) / max(denominator, 1e-30)
        result["audio_relative_l2"] = float(np.linalg.norm(a - b)) / max(float(np.linalg.norm(a)), 1e-30)
        result["audio_max_abs"] = float(np.max(np.abs(a - b)))
    if len(ra) != len(ta):
        result["audio_sample_counts"] = [len(ra), len(ta)]
        result["audio_comparison_scope"] = "shared prefix; TensorSharp trims excess 40Hz-grid samples to video duration"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cuda", "ggml_cuda", "cpu", "ggml_cpu", "mlx", "ggml_metal"])
    parser.add_argument("--engine", choices=["both", "tensorsharp", "sdcpp"], default="both")
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tensorsharp-cli", type=Path, default=Path("TensorSharp.Cli/bin/TensorSharp.Cli.dll"))
    parser.add_argument("--sdcpp-cli", required=True, type=Path)
    parser.add_argument("--sdcpp-source", type=Path,help="Clean source checkout; default inferred from isolated src/build layout")
    parser.add_argument("--ggml-source", type=Path,default=Path(__file__).resolve().parents[2]/"ExternalProjects/ggml")
    parser.add_argument("--sdcpp-backend")
    parser.add_argument("--case", choices=["smoke", "quality"], default="smoke")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--shift", type=float, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--verify-existing", action="store_true", help="Recheck media from report.json without repeating inference; preserve original evidence")
    parser.add_argument("--prompt", default="A red fox walks through a quiet forest, soft natural light, camera held steady. Forest birds chirp.")
    parser.add_argument("--checkpoint", choices=["fl2va", "ref2va"], default="fl2va")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--end-image", type=Path)
    parser.add_argument("--ref-image", action="append", type=Path, default=[])
    parser.add_argument("--ref-audio", action="append", type=Path, default=[])
    args = parser.parse_args()
    defaults = (256, 160, 5, 1) if args.case == "smoke" else (640, 384, 22, 8)
    args.width, args.height, args.frames, args.steps = [value or default for value, default in
                                                    zip((args.width, args.height, args.frames, args.steps), defaults)]
    if args.fps != 24 or args.width % 32 or args.height % 32 or args.frames % 17 != 5 or args.steps < 1 or args.rounds < 1:
        raise SystemExit("Require fps24, 32-pixel spatial grid, frame count 5+17*n and positive steps/rounds")
    components = {name: (args.models / file).resolve() for name, file in COMPONENTS.items()}
    components["dit"] = (args.models / f"minimax_h3_{args.checkpoint}_pruned-Q4_K.gguf").resolve()
    references = [*args.ref_image, *args.ref_audio]
    keyframes = [path for path in (args.image, args.end_image) if path]
    if references and (keyframes or args.checkpoint != "ref2va"):
        raise SystemExit("References require Ref2VA without keyframes")
    if keyframes and args.checkpoint != "fl2va":
        raise SystemExit("Keyframes require FL2VA")
    for path in [*references, *keyframes]:
        if not path.is_file():
            raise SystemExit(f"Missing conditioning file: {path}")
    for path in [*components.values(), *[args.models / file for file in ("vocab.json", "merges.txt", "tokenizer_config.json")]]:
        if not path.is_file():
            raise SystemExit(f"Missing model/tokenizer: {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    schedule = sigmas(args.steps, args.shift)
    env = os.environ.copy()
    env.update(TS_VIDEO_VAE=str(components["video_vae"]), TS_VIDEO_TEXT_ENCODER=str(components["text"]),
               TS_VIDEO_AUDIO_VAE=str(components["audio_vae"]), TS_VIDEO_TOKENIZER=str(args.models.resolve()),
               TS_CPU_THREADS="8", DOTNET_PROCESSOR_COUNT="8", OMP_NUM_THREADS="8", GGML_N_THREADS="8")
    for name in ("TS_H3_VIS_STOP", "TS_H3_ENC3D_STOP", "TS_GGML_NODE_PROFILE"):
        env.pop(name,None)
    baseline_backend = args.sdcpp_backend or {"cpu": "cpu", "ggml_cpu": "cpu", "mlx": "metal", "ggml_metal": "metal"}.get(args.backend, "cuda0")
    sdcpp_source=args.sdcpp_source or args.sdcpp_cli.resolve().parents[2]/"src"
    report = {"case": args.case, "backend": args.backend, "prompt": args.prompt, "seed": args.seed,
              "geometry": [args.width, args.height, args.frames], "fps": args.fps, "steps": args.steps,
              "cfg": 1, "flow_shift": args.shift, "rng": "SD CUDA Philox / independently verified TensorSharp MiniMaxH3Noise",
              "sigmas_f32": schedule, "timing_scope": "whole CLI process including model loading and output codec",
              "limitations": ["New process per repetition; cold file cache and warm-model performance not established",
                              "Output comparison includes codec differences; component float oracles establish numerical bounds"],
              "quality_parity_established":False,
              "checkpoint": args.checkpoint,
              "conditioning": [{"role": role, "path": str(path.resolve()), "sha256": sha(path)}
                               for role, paths in [("image", [args.image] if args.image else []),
                                                   ("end_image", [args.end_image] if args.end_image else []),
                                                   ("ref_image", args.ref_image), ("ref_audio", args.ref_audio)]
                               for path in paths],
              "source_revisions":{"ggml":git_state(args.ggml_source),
                                  "stable_diffusion_cpp":git_state(sdcpp_source),
                                  "stable_diffusion_cpp_ggml":git_state(sdcpp_source/"ggml")},
              "hardware":hardware(),
              "components": {name: {"path": str(path), "bytes": path.stat().st_size} for name, path in components.items()},
              "environment": {name: value for name,value in env.items() if name.startswith(("TS_H3_","TS_CPU_","TS_CUDA_QMM_","GGML_N_","OMP_NUM_","DOTNET_PROCESSOR_"))},
              "binary_files": binary_artifacts(args.tensorsharp_cli,args.sdcpp_cli),
              "executions": [], "comparisons": []}
    if args.ref_audio:
        report["limitations"].append(
            "Reference audio preprocessing differs by engine: TensorSharp caps/pads to output duration and floors 800-sample hops; "
            "SD preserves the complete waveform and ceil-pads. Identical file hashes do not establish identical processed conditioning; "
            "use explicit benchmark canonicalization and inspect effective lengths before interpreting cross-engine quality metrics.")
    if any(env.get(name) for name in ("TS_H3_SD_TRACE_DIR","SD_TRACE_DIR","TS_H3_LINEAR_TRACE_FILE")):
        report["timing_scope"] = "synchronized diagnostic CLI process; excluded from formal performance timings"
        report["limitations"].append("Boundary downloads and eval callbacks change diagnostic synchronization")
    engines = ["sdcpp", "tensorsharp"] if args.engine == "both" else [args.engine]
    if args.verify_existing:
        original_path=args.output / "report.json"
        original=json.loads(original_path.read_text())
        if original["geometry"] != report["geometry"] or original["fps"] != args.fps:
            raise SystemExit("Requested geometry does not match the recorded generation")
        verified={"original_report": str(original_path.resolve()), "original_report_sha256": sha(original_path),
                  "verification_tool_sha256": sha(Path(__file__)), "rechecked_at": time.time(), "executions": []}
        for record in original["executions"]:
            result={"engine": record["engine"], "round": record["round"], "generation_exit_code": record["exit_code"], "passed": False}
            if record["exit_code"] == 0:
                try:
                    result["media"], _, _=validate_media(Path(record["argv"][-1]),args,record["engine"])
                    result["passed"]=True
                except (RuntimeError, subprocess.TimeoutExpired, ValueError, OSError) as error:
                    result["media_error"]=str(error)
            verified["executions"].append(result)
        verified["passed"]=bool(verified["executions"]) and all(item["passed"] for item in verified["executions"])
        (args.output / "media-recheck.json").write_text(json.dumps(verified,indent=2)+"\n")
        print(json.dumps(verified),flush=True)
        return 0 if verified["passed"] else 1
    for name,state in report["source_revisions"].items():
        if not state["available"] or state["source_changed"]:
            raise SystemExit(f"Cannot establish an unchanged dependency checkout for {name}: {state}")
    success = True
    for round_index in range(args.rounds):
        media = {}
        for engine in (engines if round_index % 2 == 0 else list(reversed(engines))):
            output = (args.output / f"{engine}-{round_index}.avi").resolve()
            if output.exists() or output.with_suffix(".wav").exists():
                raise SystemExit(f"Refusing stale output path: {output}")
            if engine == "sdcpp":
                command = [str(args.sdcpp_cli.resolve()), "-M", "vid_gen", "--diffusion-model", str(components["dit"]),
                           "--vae", str(components["video_vae"]), "--audio-vae", str(components["audio_vae"]), "--llm", str(components["text"]),
                           "-p", args.prompt, "-W", str(args.width), "-H", str(args.height), "--video-frames", str(args.frames),
                           "--steps", str(args.steps), "--fps", str(args.fps), "--cfg-scale", "1", "--flow-shift", str(args.shift),
                           "--sampling-method", "euler", "--sigmas", ",".join(format(sigma, ".9g") for sigma in schedule),
                           "--rng", "cuda", "-s", str(args.seed), "--backend", baseline_backend, "-t", "8", "--mmap",
                           "--offload-to-cpu", "--diffusion-fa", "-o", str(output)]
                conditioning_flags = [("--init-img", [args.image] if args.image else []),
                                      ("--end-img", [args.end_image] if args.end_image else []),
                                      ("--ref-image", args.ref_image), ("--ref-audio", args.ref_audio)]
            else:
                command = ["dotnet", str(args.tensorsharp_cli.resolve()), "--model", str(components["dit"]), "--backend", args.backend,
                           "--video-vae", str(components["video_vae"]), "--video-text-encoder", str(components["text"]),
                           "--audio-vae", str(components["audio_vae"]), "--prompt", args.prompt, "--width", str(args.width),
                           "--height", str(args.height), "--video-frames", str(args.frames), "--diffusion-steps", str(args.steps),
                           "--fps", str(args.fps), "--cfg", "1", "--flow-shift", str(args.shift), "--sampler", "euler",
                           "--diffusion-seed", str(args.seed), "--output", str(output)]
                conditioning_flags = [("--image", [args.image] if args.image else []),
                                      ("--end-image", [args.end_image] if args.end_image else []),
                                      ("--ref-image", args.ref_image), ("--ref-audio", args.ref_audio)]
            # Keep the output last for saved-report media verification.
            command[-2:-2] = [argument for flag, paths in conditioning_flags
                              for path in paths for argument in (flag, str(path.resolve()))]
            record = {"engine": engine, "round": round_index, "argv": command, "started_at": time.time(), "timeout": False}
            started = time.perf_counter()
            with (args.output / f"{engine}-{round_index}.log").open("w") as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
            record["passed"] = record["exit_code"] == 0 and output.is_file()
            if record["passed"]:
                try:
                    record["media"], rgb, audio = validate_media(output, args, engine)
                    media[engine] = rgb, audio
                except (RuntimeError, subprocess.TimeoutExpired, ValueError, OSError) as error:
                    record.update(passed=False, media_error=str(error))
            success &= record["passed"]
            report["executions"].append(record)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: record[key] for key in ("engine", "round", "exit_code", "wall_seconds", "passed")}), flush=True)
        if len(media) == 2:
            try:
                report["comparisons"].append({"round": round_index, **comparison(media["sdcpp"], media["tensorsharp"])})
            except ImportError:
                report["limitations"].append("NumPy unavailable: decoded output metrics omitted")
    report["passed"] = bool(success)
    report["structural_media_passed"] = bool(success)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
