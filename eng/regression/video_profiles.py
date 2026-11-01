"""Bounded CLI video profiles with independent decode and approved numerical gold.

This covers Wan, MiniMax-H3 and LTX generation, audio, and image conditioning.
It does not infer perceptual quality or identity similarity from a playable clip.
All subprocesses, including codecs, use the harness's owned-process containment.
"""
from __future__ import annotations

import array
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import shutil
import sys

from .processes import run_process

SCENARIOS = frozenset({"video_generation", "video_audio_generation", "video_reference"})
_PARAMETERS = {"prompt", "width", "height", "frames", "fps", "steps", "cfg", "seed", "sampler",
               "mode", "audio", "max_output_bytes"}
_EXPECTED = {"max_rgb_rmse", "min_pixel_std", "audio_samples_per_channel", "max_audio_rmse", "min_audio_rms"}
_FIXTURES = {"reference_rgb", "reference_pcm", "first_image", "end_image", "reference_image"}
_COMPANIONS = {"video-vae", "video-text-encoder", "video-dit2", "audio-vae", "video-upscaler"}
_MAX_BYTES = 512 * 1024 ** 2
_IMAGE_BYTES = 32 * 1024 ** 2


def _kind(case):
    return case.get("adapter") or case.get("kind") or case.get("id")


def _number(value, name, low, high, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be finite numeric data")
    if integer and type(value) is not int:
        raise ValueError(name + " must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def validate_profile(profile):
    """Pure typed validation, safe during plan (no processes or model loading)."""
    kind = profile.get("scenario")
    if kind not in SCENARIOS:
        raise ValueError("unknown video scenario")
    if not isinstance(profile.get("models", {}), dict) or set(profile.get("models", {})) - {"primary"}:
        raise ValueError("video profiles consume primary preset companions only; extra model roles are not executed")
    p, e, f = (profile.get(name, {}) for name in ("parameters", "expected", "fixtures"))
    for name, value, keys in (("parameters", p, _PARAMETERS), ("expected", e, _EXPECTED), ("fixtures", f, _FIXTURES)):
        if not isinstance(value, dict) or set(value) - keys:
            raise ValueError("unknown or malformed video " + name)
    required = {"prompt", "width", "height", "frames", "fps", "steps", "cfg", "seed", "sampler"}
    if required - p.keys():
        raise ValueError("missing video parameters: " + ", ".join(sorted(required - p.keys())))
    if not isinstance(p["prompt"], str) or not p["prompt"].strip() or len(p["prompt"]) > 4096 or "\0" in p["prompt"]:
        raise ValueError("prompt must contain 1..4096 non-NUL characters")
    for name, low, high in (("width", 64, 1024), ("height", 64, 1024), ("frames", 2, 121),
                            ("fps", 1, 60), ("steps", 1, 50), ("seed", 0, 2147483647)):
        _number(p[name], name, low, high, True)
    _number(p["cfg"], "cfg", 1, 20)
    _number(p.get("max_output_bytes", 128 * 1024 ** 2), "max_output_bytes", 1024, _MAX_BYTES, True)
    if p["frames"] / p["fps"] > 10 or p["width"] * p["height"] * p["frames"] * 3 > _MAX_BYTES:
        raise ValueError("video exceeds 10 seconds or 512 MiB decoded RGB budget")
    if p["sampler"] not in {"euler", "euler_ancestral", "unipc"}:
        raise ValueError("unsupported sampler")
    mode = p.get("mode", "t2v")
    if mode not in {"t2v", "i2v", "fl2v", "ref"}:
        raise ValueError("unsupported conditioning mode")
    if "audio" in p and type(p["audio"]) is not bool:
        raise ValueError("audio must be boolean")
    audio = p.get("audio", kind == "video_audio_generation")
    if (kind == "video_audio_generation" and not audio) or (kind == "video_generation" and audio):
        raise ValueError("audio selection contradicts scenario")
    if "reference_rgb" not in f or "max_rgb_rmse" not in e:
        raise ValueError("approved reference_rgb fixture and max_rgb_rmse are required")
    _number(e["max_rgb_rmse"], "max_rgb_rmse", 0, 16)
    _number(e.get("min_pixel_std", 1), "min_pixel_std", .000001, 127.5)
    if audio:
        if not {"audio_samples_per_channel", "max_audio_rmse"} <= e.keys() or "reference_pcm" not in f:
            raise ValueError("audio requires approved reference_pcm, audio_samples_per_channel and max_audio_rmse")
        _number(e["audio_samples_per_channel"], "audio_samples_per_channel", 1, 480000, True)
        _number(e["max_audio_rmse"], "max_audio_rmse", 0, .1)
        _number(e.get("min_audio_rms", .000001), "min_audio_rms", .000000001, 1)
    elif "reference_pcm" in f or any(k.startswith("audio_") or k in {"max_audio_rmse", "min_audio_rms"} for k in e):
        raise ValueError("video-only profile must not claim audio assertions")
    images = set(f) & {"first_image", "end_image", "reference_image"}
    if kind == "video_reference":
        valid = ((mode == "i2v" and images == {"first_image"})
                 or (mode == "fl2v" and "end_image" in images and "reference_image" not in images)
                 or (mode == "ref" and images == {"reference_image"}))
        if not valid:
            raise ValueError("video_reference requires image fixtures matching i2v/fl2v/ref mode")
    elif images:
        # I2V-only Wan is also a video_generation preset, so a first frame is allowed.
        if mode != "i2v" or images != {"first_image"}:
            raise ValueError("generation conditioning requires exactly first_image with mode i2v")
    elif mode != "t2v":
        raise ValueError("conditioning mode needs its image fixture")
    for name, fixture in f.items():
        if not isinstance(fixture, dict) or not isinstance(fixture.get("path"), str) or not fixture["path"]:
            raise ValueError("fixture requires a path: " + name)
        digest = fixture.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError("fixture requires exact SHA256: " + name)
    return {"parameters": p, "expected": e, "fixtures": f, "audio": audio, "mode": mode}


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _family(context):
    return context.get("model_family") or context.get("preset", {}).get("family")


def required_options(profile):
    """Required companion options; optional Ingredients is never activated here."""
    config = validate_profile(profile)
    required = {"video-vae", "video-text-encoder"}
    if config["audio"] or profile.get("preset", "").startswith("ltx-2.5"):
        required.add("audio-vae")
    if profile.get("preset", "").startswith("ltx-2.5"):
        required.add("video-upscaler")
    return required


def _family_validation(config, context):
    p, family = config["parameters"], _family(context)
    if family not in {"wan", "minimax-h3", "ltx-2.5"}:
        raise ValueError("video profile requires a source-verified Wan, MiniMax-H3 or LTX 2.5 preset")
    backend = context.get("backend")
    if family == "ltx-2.5":
        if backend not in {"cpu", "cuda", "mlx"}:
            raise ValueError("LTX GGML aliases execute owned kernels; select exact cpu/cuda/mlx for this profile")
        if p["steps"] != 8 or p["cfg"] != 1 or p["sampler"] not in {"euler", "euler_ancestral"}:
            raise ValueError("LTX distilled profile requires steps=8, cfg=1, euler/euler_ancestral")
        if (p["frames"] - 1) % 8 or p["width"] % 64 or p["height"] % 64:
            raise ValueError("LTX profile requires 8n+1 frames and 64-pixel multiples")
        if config["mode"] == "ref":
            raise ValueError("LTX Ingredients identity mode is not qualified here; use i2v/fl2v temporal references (gated adapter is never acquired implicitly)")
    elif family == "minimax-h3":
        if backend not in {"cpu", "cuda", "mlx", "ggml_cpu", "ggml_cuda", "ggml_metal"}:
            raise ValueError("MiniMax-H3 video backend is outside source-qualified scope")
        if (p["frames"] - 5) % 17 or p["fps"] != 24 or p["cfg"] != 1 or p["sampler"] != "euler":
            raise ValueError("MiniMax-H3 requires 17n+5 frames, 24 fps, cfg=1, euler")
        if p["width"] % 16 or p["height"] % 16:
            raise ValueError("MiniMax-H3 dimensions must be multiples of 16")
    else:
        if backend not in {"cpu", "cuda", "ggml_cpu", "ggml_cuda", "ggml_metal", "ggml_vulkan"}:
            raise ValueError("Wan video backend is outside source-qualified scope")
        if config["audio"] or config["mode"] not in {"t2v", "i2v"}:
            raise ValueError("Wan profile supports video only, text or first-frame conditioning")
        preset_id = context.get("preset_id") or context.get("scenario_profile", {}).get("preset", "")
        multiple = 32 if "ti2v" in preset_id else 16
        if "i2v-a14b" in preset_id and config["mode"] != "i2v":
            raise ValueError("Wan I2V-A14B requires an approved first_image fixture and i2v mode")
        if (p["frames"] - 1) % 4 or p["width"] % multiple or p["height"] % multiple or p["sampler"] not in {"euler", "unipc"}:
            raise ValueError(f"Wan requires 4n+1 frames, {multiple}-pixel multiples, euler/unipc")


def availability(case, context):
    profile = context.get("scenario_profile")
    if not profile:
        return False, "provide --scenario-profiles with bounded video parameters and approved RGB24/PCM gold fixtures"
    try:
        config = validate_profile(profile)
        if profile["scenario"] != _kind(case):
            raise ValueError("profile scenario does not match selected case")
        _family_validation(config, context)
        command = context.get("cli_command")
        if not isinstance(command, (tuple, list)) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError("provide a built local CLI argv list")
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise ValueError("install local ffmpeg and ffprobe on PATH; no codec download occurs")
        for name, fixture in config["fixtures"].items():
            path = Path(fixture["path"])
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError("approved fixture missing or empty: " + name)
            limit = _IMAGE_BYTES if name.endswith("image") else _MAX_BYTES
            if path.stat().st_size > limit:
                raise ValueError("fixture exceeds bounded byte budget: " + name)
            if name.endswith("image") and path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                raise ValueError("conditioning fixtures must be PNG/JPEG")
        p, f = config["parameters"], config["fixtures"]
        if Path(f["reference_rgb"]["path"]).stat().st_size != p["width"] * p["height"] * p["frames"] * 3:
            raise ValueError("reference_rgb must be complete frame-major RGB24 data matching dimensions")
        if config["audio"] and Path(f["reference_pcm"]["path"]).stat().st_size != config["expected"]["audio_samples_per_channel"] * 8:
            raise ValueError("reference_pcm must be complete interleaved float32 stereo")
        if not context.get("preparing_model"):
            if not Path(context.get("model_path", "")).is_file():
                raise ValueError("verified local video model is unavailable")
            options = context.get("model_options", {})
            required = {"video-vae", "video-text-encoder"}
            if config["audio"] or _family(context) == "ltx-2.5":
                required.add("audio-vae")
            if _family(context) == "ltx-2.5":
                required.add("video-upscaler")
            for name in required:
                if not options.get(name) or not Path(options[name]).is_file():
                    raise ValueError("verified video companion missing: " + name)
        return True, "bounded CLI video and approved numerical gold prerequisites present"
    except (ValueError, TypeError, OSError, KeyError) as error:
        return False, str(error)


def _execute(command, context, result, stage, backend=None):
    from .scenarios import _require, backend_evidence
    item = run_process(command, cwd=context["repo"], artifact_dir=context["artifact_dir"], stage=stage,
                       timeout_seconds=context.get("timeout_seconds", 600), env=context.get("env"))
    result["processes"].append({k: v for k, v in item.items() if k not in {"stdout", "stderr"}})
    _require(result, stage + ": no timeout", not item["timed_out"], False, item["timed_out"])
    _require(result, stage + ": process exit", item["exit_code"] == 0, 0, item["exit_code"])
    _require(result, stage + ": complete diagnostic output", not item.get("output_truncated", False))
    _require(result, stage + ": owned processes stopped", item.get("teardown", {}).get("verified") is True,
             "verified empty owned container", item.get("teardown"))
    if backend:
        observed = backend_evidence(item["stdout"] + "\n" + item["stderr"], backend)
        result["evidence"]["backend"] = observed
        _require(result, "video actual backend", observed["verified"], backend, observed)
    return item


def _probe(path, context, result, stage):
    from .scenarios import _json_loads
    output = _execute([shutil.which("ffprobe"), "-v", "error", "-count_frames", "-show_streams", "-show_format",
                       "-of", "json", str(path)], context, result, stage)
    return _json_loads(output["stdout"])


def _compare_rgb(actual, reference):
    count, total, square, error, maximum, first = 0, 0, 0, 0, 0, None
    with actual.open("rb") as a, reference.open("rb") as b:
        while True:
            av, bv = a.read(1024 * 1024), b.read(1024 * 1024)
            if len(av) != len(bv):
                raise ValueError("RGB reference and actual lengths differ")
            if not av:
                break
            if first is None:
                first = next(({"byte_index": count + i, "expected": y, "actual": x}
                              for i, (x, y) in enumerate(zip(av, bv)) if x != y), None)
            maximum = max(maximum, max(abs(x - y) for x, y in zip(av, bv)))
            count += len(av)
            total += sum(av)
            square += sum(x * x for x in av)
            error += sum((x - y) ** 2 for x, y in zip(av, bv))
    return {"samples": count, "rmse": math.sqrt(error / count), "std": math.sqrt(max(0, square / count - (total / count) ** 2)),
            "maximum_absolute_difference": maximum, "first_difference": first}


def _compare_audio(actual, reference):
    count, square, error, clipped = 0, 0., 0., 0
    with actual.open("rb") as a, reference.open("rb") as b:
        while True:
            av, bv = a.read(1024 * 1024), b.read(1024 * 1024)
            if len(av) != len(bv) or len(av) % 4:
                raise ValueError("PCM reference and actual float32 layouts differ")
            if not av:
                break
            left, right = array.array("f"), array.array("f")
            left.frombytes(av)
            right.frombytes(bv)
            if sys.byteorder != "little":
                left.byteswap()
                right.byteswap()
            if not all(math.isfinite(x) for x in left) or not all(math.isfinite(x) for x in right):
                raise ValueError("nonfinite PCM output or reference")
            count += len(left)
            square += sum(x * x for x in left)
            error += sum((x - y) ** 2 for x, y in zip(left, right))
            clipped += sum(abs(x) >= .9999 for x in left)
    if not count:
        raise ValueError("empty PCM")
    return {"samples": count, "rms": math.sqrt(square / count), "rmse": math.sqrt(error / count), "clipped_fraction": clipped / count}


def run(case, context, result):
    """Mutate the standard scenario result; outer harness catches failures."""
    from .scenarios import ScenarioUntested, _require
    ready, reason = availability(case, context)
    if not ready:
        raise ScenarioUntested(reason)
    profile = context["scenario_profile"]
    config = validate_profile(profile)
    p, e, fixtures = config["parameters"], config["expected"], config["fixtures"]
    root = Path(context["artifact_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    media, rgb, pcm = root / "generated.mp4", root / "decoded.rgb24", root / "decoded.pcm.f32le"
    for path in (media, media.with_suffix(".wav"), rgb, pcm):
        _require(result, "fresh output: " + path.name, not path.exists())
    hashes = {}
    for name, fixture in fixtures.items():
        hashes[name] = _sha(fixture["path"])
        _require(result, "approved fixture hash: " + name, hashes[name] == fixture["sha256"].lower(), fixture["sha256"], hashes[name])
    required_disk = 2 * p["width"] * p["height"] * p["frames"] * 3 + p.get("max_output_bytes", 128 * 1024 ** 2)
    required_disk += int(context.get("disk_headroom_gib", 5) * 1024 ** 3)
    if shutil.disk_usage(root).free < required_disk:
        raise ScenarioUntested("insufficient artifact disk for video decode plus configured headroom")
    result["inputs"] = {"profile_id": profile.get("id"), "parameters": p, "expected": e, "fixture_sha256": hashes,
                        "scope": "CLI numerical media regression; no perceptual/semantic identity qualification"}
    result["evidence"]["codec_tools"] = {name: {"path": shutil.which(name),
        "sha256": _sha(shutil.which(name)) if Path(shutil.which(name)).is_file() else None} for name in ("ffmpeg", "ffprobe")}
    command = list(context["cli_command"]) + ["--model", str(context["model_path"]), "--backend", context["backend"],
        "--no-skills", "--log-level", "info", "--log-file", "off", "--prompt", p["prompt"], "--output", str(media),
        "--width", str(p["width"]), "--height", str(p["height"]), "--video-frames", str(p["frames"]), "--fps", str(p["fps"]),
        "--diffusion-steps", str(p["steps"]), "--cfg", str(p["cfg"]), "--seed", str(p["seed"]), "--sampler", p["sampler"],
        "--video-mode", config["mode"]]
    for name in sorted(_COMPANIONS):
        if context.get("model_options", {}).get(name):
            command += ["--" + name, str(context["model_options"][name])]
    if not config["audio"]:
        command.append("--no-audio")
    for name, flag in (("first_image", "--image"), ("end_image", "--end-image"), ("reference_image", "--ref-image")):
        if name in fixtures:
            command += [flag, fixtures[name]["path"]]
    process = _execute(command, context, result, "video-generate", context["backend"])
    _require(result, "generated video exists within byte budget", media.is_file() and 0 < media.stat().st_size <= p.get("max_output_bytes", 128 * 1024 ** 2))
    metadata = _probe(media, context, result, "video-probe")
    streams = metadata.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    _require(result, "exactly one video stream", len(video) == 1, 1, len(video))
    v = video[0]
    actual = {"width": v.get("width"), "height": v.get("height"), "frames": int(v.get("nb_read_frames", -1)),
              "fps": str(Fraction(v.get("avg_frame_rate", "0/1")))}
    expected = {"width": p["width"], "height": p["height"], "frames": p["frames"], "fps": str(Fraction(p["fps"]))}
    _require(result, "decoded video geometry and time grid", actual == expected, expected, actual)
    rgb_bytes = p["width"] * p["height"] * p["frames"] * 3
    _execute([shutil.which("ffmpeg"), "-v", "error", "-xerror", "-nostdin", "-n", "-i", str(media), "-map", "0:v:0",
              "-fps_mode", "passthrough", "-frames:v", str(p["frames"] + 1), "-fs", str(rgb_bytes + 1),
              "-f", "rawvideo", "-pix_fmt", "rgb24", str(rgb)], context, result, "video-decode")
    _require(result, "complete bounded RGB decode", rgb.is_file() and rgb.stat().st_size == rgb_bytes, rgb_bytes,
             rgb.stat().st_size if rgb.exists() else None)
    metrics = _compare_rgb(rgb, Path(fixtures["reference_rgb"]["path"]))
    # Preserve comparison diagnostics even when a gate fails below.
    result["evidence"]["video"] = {"ffprobe": metadata, "decoded": metrics, "media_sha256": _sha(media), "rgb_sha256": _sha(rgb)}
    _require(result, "nonblank decoded video", metrics["std"] >= e.get("min_pixel_std", 1), e.get("min_pixel_std", 1), metrics["std"])
    _require(result, "approved full-video RGB reference", metrics["rmse"] <= e["max_rgb_rmse"], e["max_rgb_rmse"], metrics["rmse"])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not config["audio"]:
        _require(result, "video-only route has no soundtrack", not audio_streams and not media.with_suffix(".wav").exists())
    else:
        # The CLI deliberately retains original PCM even where the platform writer cannot mux it.
        audio_path = media.with_suffix(".wav")
        _require(result, "generated original PCM sidecar exists", audio_path.is_file() and 0 < audio_path.stat().st_size <= _MAX_BYTES)
        audio_meta = _probe(audio_path, context, result, "audio-probe")
        tracks = [s for s in audio_meta.get("streams", []) if s.get("codec_type") == "audio"]
        rate = 48000 if _family(context) == "ltx-2.5" else 32000
        _require(result, "family stereo PCM sample rate", len(tracks) == 1 and int(tracks[0].get("sample_rate", 0)) == rate
                 and tracks[0].get("channels") == 2, {"sample_rate": rate, "channels": 2}, tracks)
        samples, duration = e["audio_samples_per_channel"], p["frames"] / p["fps"]
        _require(result, "audio duration matches video grid", abs(samples / rate - duration) <= 1 / p["fps"], duration, samples / rate)
        _execute([shutil.which("ffmpeg"), "-v", "error", "-xerror", "-nostdin", "-n", "-i", str(audio_path),
                  "-map", "0:a:0", "-fs", str(samples * 8 + 1), "-f", "f32le", "-acodec", "pcm_f32le", str(pcm)],
                 context, result, "audio-decode")
        _require(result, "complete bounded stereo PCM decode", pcm.is_file() and pcm.stat().st_size == samples * 8, samples * 8,
                 pcm.stat().st_size if pcm.exists() else None)
        audio_metrics = _compare_audio(pcm, Path(fixtures["reference_pcm"]["path"]))
        result["evidence"]["audio"] = {"ffprobe": audio_meta, "decoded": audio_metrics, "sha256": _sha(pcm),
                                         "muxed_audio_streams": len(audio_streams), "scope": "original CLI WAV sidecar; mux quality is not claimed"}
        _require(result, "finite nonsilent unclipped PCM", audio_metrics["rms"] >= e.get("min_audio_rms", .000001)
                 and audio_metrics["clipped_fraction"] < .99, {"minimum_rms": e.get("min_audio_rms", .000001), "clipped_fraction_below": .99}, audio_metrics)
        _require(result, "approved full-audio PCM reference", audio_metrics["rmse"] <= e["max_audio_rmse"], e["max_audio_rmse"], audio_metrics["rmse"])
    if _kind(case) == "video_reference":
        output = process["stdout"] + "\n" + process["stderr"]
        for name, marker in (("first_image", "image"), ("end_image", "end"), ("reference_image", "ref-img")):
            if name in fixtures:
                _require(result, "conditioning input receipt: " + name, fixtures[name]["path"] in output and marker in output)
        result["evidence"]["conditioning"] = {"mode": config["mode"], "fixtures": hashes,
            "scope": "exact approved conditioned-output numeric oracle; no face/subject identity metric or unsupported-mode coverage"}
