"""Bounded media profiles with supplied reference oracles, never auto-approved gold.

The image oracle is an exact decoded RGBA8 hash for one model/backend/seed recipe;
it is not a perceptual quality score. Embedding oracles identify their provenance
and exact media/text/model/projector. All HTTP is to this invocation's owned host.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import shutil
import socket
import struct
import sys
import time
import urllib.error
import wave

from .image_generation import PNG_LIMIT, UnsupportedPngFormat, local_artifact_route, png_metadata
from .processes import ManagedProcess, run_process

SCENARIOS = frozenset({"embedding_media", "image_generation", "image_editing"})
MEDIA_LIMIT = 2 * 1024 * 1024
_SHA = re.compile(r"[a-fA-F0-9]{64}\Z")


def _number(value, minimum, maximum, *, integer=False):
    return (type(value) is int if integer else type(value) in {int, float}) and math.isfinite(value) and minimum <= value <= maximum


def validate_profile(profile):
    """Validate adapter-specific fields; the root loader resolves fixture paths."""
    kind = profile.get("scenario")
    if kind not in SCENARIOS:
        raise ValueError("unsupported image/media profile scenario")
    if set(profile.get("models", {})) - {"primary"}:
        raise ValueError("image/media profile uses only primary preset config companions; extra model roles are not consumed")
    params, expected, fixtures = (profile.get(key, {}) for key in ("parameters", "expected", "fixtures"))
    if not all(isinstance(value, dict) for value in (params, expected, fixtures)):
        raise ValueError("parameters, expected and fixtures must be objects")
    if kind == "embedding_media":
        if set(params) != {"modality", "text", "dimensions"} or params["modality"] not in {"image", "audio", "video"}:
            raise ValueError("embedding parameters require modality=image|audio|video, text and dimensions")
        if not isinstance(params["text"], str) or len(params["text"]) > 1024 or not _number(params["dimensions"], 1, 8192, integer=True):
            raise ValueError("embedding text/dimensions exceed bounded contract")
        if set(fixtures) != {"media", "oracle"} or set(expected) - {"max_relative_l2", "min_cosine"}:
            raise ValueError("embedding requires media+oracle fixtures and only numerical oracle gates")
        if not _number(expected.get("max_relative_l2", .001), 0, .001) or not _number(expected.get("min_cosine", .999999), .999999, 1):
            raise ValueError("embedding oracle gates may tighten but not weaken .001/.999999")
    else:
        required = {"prompt", "width", "height", "steps", "cfg", "seed"}
        if set(params) - (required | {"negative_prompt"}) or not required <= set(params):
            raise ValueError("image parameters require prompt,width,height,steps,cfg,seed; optional negative_prompt")
        if not isinstance(params["prompt"], str) or not params["prompt"].strip() or len(params["prompt"]) > 1024:
            raise ValueError("image prompt must be nonempty and <=1024 characters")
        if not isinstance(params.get("negative_prompt", " "), str) or len(params.get("negative_prompt", " ")) > 1024:
            raise ValueError("negative_prompt must be text <=1024 characters")
        if any(not _number(params[key], 32, 512, integer=True) or params[key] % 32 for key in ("width", "height")):
            raise ValueError("image dimensions must be multiples of32 in32..512")
        if not _number(params["steps"], 1, 50, integer=True) or not _number(params["cfg"], 0, 20) or not _number(params["seed"], 0, 2**31-1, integer=True):
            raise ValueError("image steps/cfg/seed exceed bounded contract")
        expected_keys = {"pixel_sha256", "min_changed_mean_abs_diff"} if kind == "image_editing" else {"pixel_sha256"}
        if set(expected) != expected_keys or not isinstance(expected["pixel_sha256"], str) or not _SHA.fullmatch(expected["pixel_sha256"]):
            raise ValueError("image expected.pixel_sha256 must be a supplied exact RGBA8 oracle")
        if kind == "image_editing" and (not _number(expected["min_changed_mean_abs_diff"], 0, 255) or expected["min_changed_mean_abs_diff"] == 0):
            raise ValueError("editing requires approved positive min_changed_mean_abs_diff <=255")
        if set(fixtures) != ({"input", "mask"} if kind == "image_editing" else set()):
            raise ValueError("editing requires input+mask fixtures; generation has no input fixture")
    for name, fixture in fixtures.items():
        if not isinstance(fixture, dict) or not isinstance(fixture.get("path"), str) or not isinstance(fixture.get("sha256"), str) or not _SHA.fullmatch(fixture["sha256"]):
            raise ValueError("fixture requires a local path and SHA256: " + name)


def required_options(profile):
    if profile["scenario"] == "embedding_media":
        return {"model", "mmproj"}
    result = {"model", "qwen-image-vae", "qwen-image-vl"}
    if profile["scenario"] == "image_editing":
        result.add("qwen-image-mmproj")
    return result


def availability(case, context):
    profile = context.get("scenario_profile")
    if not profile:
        return False, "provide --scenario-profiles with a typed image/media profile and approved oracle"
    try:
        validate_profile(profile)
    except (ValueError, TypeError) as error:
        return False, str(error)
    if profile["scenario"] != (case.get("adapter") or case.get("kind") or case["id"]):
        return False, "scenario profile does not match selected case"
    if not context.get("server_command") or isinstance(context["server_command"], str):
        return False, "provide --server with a built local TensorSharp.Server.Host binary"
    options = context.get("model_options", {})
    keys = ("mmproj",) if profile["scenario"] == "embedding_media" else ("qwen-image-vae", "qwen-image-vl")
    if profile["scenario"] == "image_editing":
        keys += ("qwen-image-mmproj",)
    for key in keys:
        if not options.get(key) or (not context.get("preparing_model") and not Path(options[key]).is_file()):
            return False, "verified config companion required: " + key
    if profile["scenario"] != "embedding_media":
        variant = options.get("qwen-image-variant", "base")
        if variant not in {"base", "turbo"}:
            return False, "image profile supports source-defined base/turbo model variants"
        if variant == "turbo" and (profile["parameters"]["steps"] != 8 or profile["parameters"]["cfg"] != 1):
            return False, "Qwen Image Turbo requires its source-defined8-step CFG1 schedule"
    for name, fixture in profile["fixtures"].items():
        if not Path(fixture["path"]).is_file():
            return False, "required approved fixture is absent: " + name
    if profile["scenario"] == "embedding_media" and profile["parameters"]["modality"] == "video" and (not shutil.which("ffprobe") or not shutil.which("ffmpeg")):
        return False, "bounded video embeddings require ffprobe and ffmpeg on PATH"
    if profile["scenario"] == "image_editing":
        if not all(importlib.util.find_spec(name) for name in ("numpy", "PIL")):
            return False, "masked editing fidelity executor requires numpy and Pillow in the runner Python"
        if not (Path(context["repo"]) / "eng/validation/qwen-image21-edit-fidelity.py").is_file():
            return False, "repository masked-edit fidelity executor is absent"
    return True, "typed profile prerequisites present; actual model/backend/oracle still require execution"


def _fixture_bytes(profile, name, limit=MEDIA_LIMIT):
    fixture = profile["fixtures"][name]
    path = Path(fixture["path"])
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if not data or len(data) > limit:
        raise ValueError("fixture empty or exceeds byte budget: " + name)
    if hashlib.sha256(data).hexdigest() != fixture["sha256"].lower():
        raise ValueError("fixture SHA256 changed: " + name)
    return data


def _image_input(data, width=None, height=None):
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("bounded image fixtures must be PNG")
    w, h = struct.unpack_from(">II", data, 16)
    if not 1 <= w <= 512 or not 1 <= h <= 512:
        raise ValueError("input PNG exceeds512px geometry budget")
    return png_metadata(data, width or w, height or h)


def _media_part(profile, context, result):
    """Validate complete bounded media before loading a model."""
    data = _fixture_bytes(profile, "media")
    modality = profile["parameters"]["modality"]
    if modality == "image":
        receipt = _image_input(data)
        part = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}
    elif modality == "audio":
        with wave.open(io.BytesIO(data)) as audio:
            frames, rate, channels, width = audio.getnframes(), audio.getframerate(), audio.getnchannels(), audio.getsampwidth()
            if not 0 < frames <= rate * 10 or not 8000 <= rate <= 48000 or channels not in {1, 2} or width not in {1, 2, 3, 4} or len(audio.readframes(frames)) != frames * channels * width:
                raise ValueError("WAV must be complete PCM <=10seconds,8..48kHz,1..2channels")
            receipt = {"frames": frames, "sample_rate": rate, "channels": channels, "seconds": frames / rate}
        part = {"type": "input_audio", "input_audio": {"format": "wav", "data": base64.b64encode(data).decode()}}
    else:
        path = Path(profile["fixtures"]["media"]["path"])
        if path.suffix.lower() not in {".mp4", ".webm"}:
            raise ValueError("video fixture must be MP4 or WebM")
        command = [shutil.which("ffprobe") or "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-count_frames", "-show_entries", "stream=codec_type,width,height,nb_read_frames,duration:format=duration", "-of", "json", str(path)]
        item = run_process(command, cwd=context["repo"], artifact_dir=context["artifact_dir"], stage="media-probe", timeout_seconds=min(30, context.get("timeout_seconds", 600)), env=context.get("env"))
        result["processes"].append({k: v for k, v in item.items() if k not in {"stdout", "stderr"}})
        if item.get("timed_out") or item.get("exit_code") != 0 or item.get("output_truncated") or item.get("teardown", {}).get("verified") is not True:
            raise ValueError("bounded video probe failed, timed out, or cleanup is unverified")
        receipt = json.loads(item["stdout"])
        videos = [stream for stream in receipt.get("streams", []) if stream.get("codec_type") == "video"]
        if len(videos) != 1 or not _number(videos[0].get("width"), 1, 512, integer=True) or not _number(videos[0].get("height"), 1, 512, integer=True) or not 1 <= int(videos[0].get("nb_read_frames", "0")) <= 32 or not 0 < float(receipt.get("format", {}).get("duration", 0)) <= 10:
            raise ValueError("video must have one <=512px stream,1..32frames,and <=10seconds")
        mime = "video/mp4" if path.suffix.lower() == ".mp4" else "video/webm"
        part = {"type": "input_video", "input_video": {"url": "data:" + mime + ";base64," + base64.b64encode(data).decode()}}
    result["evidence"]["media_fixture"] = {"modality": modality, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "validation": receipt}
    return part


def _negative(http, result, route, payload, *, typed=False):
    from .scenarios import _require, _json_loads
    try:
        http.json(route, payload)
    except urllib.error.HTTPError as error:
        records = result["evidence"].get("http_errors", [])
        body = _json_loads(records[-1]["body"]) if records else {}
        detail = body.get("error")
        valid = isinstance(detail, dict) and detail.get("type") == "invalid_request_error" and detail.get("param") == "input" and bool(detail.get("message")) if typed else isinstance(detail, str) and bool(detail)
        _require(result, "invalid media request rejected with diagnostic", error.code == 400 and valid, {"status": 400, "typed": typed}, {"status": error.code, "body": body})
    else:
        _require(result, "invalid media request rejected", False, 400, "accepted")


def _vector(value, dimensions):
    if not isinstance(value, list) or len(value) != dimensions or not all(_number(x, -1e20, 1e20) for x in value):
        raise ValueError("embedding vector shape or finite values invalid")
    norm = math.sqrt(sum(x*x for x in value))
    if not .98 <= norm <= 1.02:
        raise ValueError("embedding vector is not unit normalized")
    return value


def _metrics(actual, reference):
    aa, bb = sum(x*x for x in actual), sum(x*x for x in reference)
    return {"relative_l2": math.sqrt(sum((x-y)**2 for x, y in zip(actual, reference)) / bb),
            "cosine": sum(x*y for x, y in zip(actual, reference)) / math.sqrt(aa*bb)}


def _redact_media_requests(result):
    """Persist hashes/lengths, not the supplied fixture's base64 bytes."""
    def visit(value):
        if isinstance(value, dict):
            for key, child in list(value.items()):
                if key in {"url", "data"} and isinstance(child, str) and (child.startswith("data:") or key == "data"):
                    value[key] = {"encoded_bytes": len(child), "encoded_sha256": hashlib.sha256(child.encode()).hexdigest(), "content": "omitted; use SHA-pinned local fixture"}
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(result.get("inputs", {}).get("requests", []))


def _embeddings(http, identity, profile, context, result, part):
    from .scenarios import _require, _json_loads
    params, expected = profile["parameters"], profile["expected"]
    oracle = _json_loads(_fixture_bytes(profile, "oracle", 1024 * 1024))
    for key, value in {"schema_version": 1, "model": identity["loaded"], "projector": identity.get("loadedMmProj"),
                       "modality": params["modality"], "media_sha256": profile["fixtures"]["media"]["sha256"].lower(),
                       "text": params["text"], "dimensions": params["dimensions"]}.items():
        _require(result, "approved embedding oracle " + key, oracle.get(key) == value, value, oracle.get(key))
    _require(result, "embedding oracle provenance supplied", isinstance(oracle.get("provenance"), str) and bool(oracle["provenance"].strip()))
    reference = _vector(oracle.get("vector"), params["dimensions"])
    listing = http.json("/v1/models").get("data", [])
    _require(result, "one embedding model advertised", len(listing) == 1 and isinstance(listing[0].get("id"), str))
    model = listing[0]["id"]
    _require(result, "embedding advertised alias", model in {identity["loaded"], Path(identity["loaded"]).stem}, identity["loaded"], model)
    content = ([{"type": "text", "text": params["text"]}] if params["text"] else []) + [part]
    payload = {"model": model, "input": [{"content": content}, {"content": content}], "dimensions": params["dimensions"], "encoding_format": "float"}
    try:
        response = http.json("/v1/embeddings", payload)
    finally:
        _redact_media_requests(result)
    rows, usage = response.get("data", []), response.get("usage", {})
    _require(result, "media embedding response envelope", response.get("object") == "list" and response.get("model") == model and len(rows) == 2)
    _require(result, "media embedding positive token accounting", type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] > 0 and usage.get("total_tokens") == usage["prompt_tokens"])
    vectors = []
    for index, row in enumerate(rows):
        _require(result, "media embedding ordered row", row.get("index") == index and row.get("object") == "embedding", index, row)
        vector = _vector(row.get("embedding"), params["dimensions"])
        vectors.append(vector)
        metrics = _metrics(vector, reference)
        result["evidence"].setdefault("embedding_reference_comparisons", []).append(metrics)
        _require(result, "media embedding independent numerical oracle", metrics["relative_l2"] <= expected.get("max_relative_l2", .001) and metrics["cosine"] >= expected.get("min_cosine", .999999), expected, metrics)
    duplicate = _metrics(vectors[0], vectors[1])
    _require(result, "duplicate media embeddings repeat", duplicate["relative_l2"] <= .001 and duplicate["cosine"] >= .999999, ".001/.999999", duplicate)
    for bad in ({"content": []}, {"content": [{"type": "unsupported-regression-type"}]}, {"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!"}}]}):
        _negative(http, result, "/v1/embeddings", dict(payload, input=bad), typed=True)
    result["evidence"]["oracle_scope"] = "Fixed media/text numerical reference and duplicate replay; no universal semantic or cross-backend quality claim"


def _upload_image(http, result, data):
    """Use owned HTTP opener because _Http's raw-body helper caps at64KiB."""
    import urllib.request
    from .scenarios import _require, _json_loads
    boundary = "TensorSharpRegressionImageFixture"
    body = ("--" + boundary + '\r\nContent-Disposition: form-data; name="file"; filename="reference.png"\r\nContent-Type: image/png\r\n\r\n').encode() + data + ("\r\n--" + boundary + "--\r\n").encode()
    request = urllib.request.Request(http.base + "/api/upload", body, headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    result["inputs"].setdefault("uploads", []).append({"route": "/api/upload", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "content_type": "image/png"})
    try:
        with http.opener.open(request, timeout=http.remaining()) as response:
            raw = response.read(65537)
            _require(result, "bounded image fixture upload", response.status == 200 and len(raw) <= 65536)
    except urllib.error.HTTPError as error:
        result["evidence"].setdefault("http_errors", []).append({"route": "/api/upload", "status": error.code, "body": error.read(65536).decode("utf-8", "replace")})
        error.close()
        raise
    receipt = _json_loads(raw)
    result["evidence"].setdefault("image_uploads", []).append(receipt)
    path = receipt.get("file")
    _require(result, "uploaded PNG acknowledged", receipt.get("ok") is True and receipt.get("mediaType") == "image", {"ok": True, "mediaType": "image"}, receipt)
    _require(result, "upload returns confined reference name", isinstance(path, str) and path and Path(path).name == path and "/" not in path and "\\" not in path and path not in {".", ".."}, "bare owned upload filename", path)
    return path


def _download_image(http, value, profile, context, result, *, filename="profile-output.png", expected_hash=None):
    from .scenarios import _require
    params = profile["parameters"]
    _require(result, "generated image metadata", value.get("ok") is True and value.get("width") == params["width"] and value.get("height") == params["height"])
    _require(result, "finite image elapsed duration", _number(value.get("elapsedSeconds"), 0, 1e8))
    route = local_artifact_route(value.get("url"), http.base)
    with http.open(route) as response:
        _require(result, "generated image HTTP content type", response.status == 200 and response.headers.get("Content-Type", "").split(";")[0].strip().lower() == "image/png")
        data = response.read(PNG_LIMIT + 1)
    target = Path(context["artifact_dir"]) / filename
    target.write_bytes(data[:PNG_LIMIT])
    receipt = {"artifact": str(target), "route": route, "capture_truncated": len(data) > PNG_LIMIT}
    result["evidence"].setdefault("generated_images", []).append(receipt)
    info = png_metadata(data, params["width"], params["height"])
    receipt.update(info)
    expected_hash = expected_hash or profile["expected"]["pixel_sha256"]
    _require(result, "approved decoded RGBA8 image oracle", info["pixel_sha256"] == expected_hash.lower(), expected_hash, info["pixel_sha256"])
    return target


def _mask_data(profile):
    from PIL import Image
    data = _fixture_bytes(profile, "mask")
    params = profile["parameters"]
    _image_input(data, params["width"], params["height"])
    with Image.open(io.BytesIO(data)) as image:
        rgb = list(image.convert("RGB").getdata())
    if not all(r == g == b and r in {0, 255} for r, g, b in rgb) or not any(r == 0 for r, _, _ in rgb) or not any(r == 255 for r, _, _ in rgb):
        raise ValueError("mask fixture must contain both protected black and editable white pixels only")
    return data


def _solid_mask(width, height):
    from PIL import Image
    output = io.BytesIO()
    Image.new("L", (width, height), 0).save(output, format="PNG")
    return output.getvalue()


def _fidelity(profile, context, result, output):
    from .scenarios import _require
    target = Path(context["artifact_dir"]) / "image-fidelity.json"
    target.unlink(missing_ok=True)  # Only this adapter-owned report; stale success cannot qualify a new run.
    command = [sys.executable, str(Path(context["repo"]) / "eng/validation/qwen-image21-edit-fidelity.py"),
               "--source", profile["fixtures"]["input"]["path"], "--changed", profile["fixtures"]["mask"]["path"],
               "--selection", profile["fixtures"]["mask"]["path"], "--margin", "0", "--json", str(target), str(output)]
    item = run_process(command, cwd=context["repo"], artifact_dir=context["artifact_dir"], stage="image-fidelity",
                       timeout_seconds=min(30, context.get("timeout_seconds", 600)), env=context.get("env"))
    result["processes"].append({k: v for k, v in item.items() if k not in {"stdout", "stderr"}})
    _require(result, "image fidelity process completed and contained", item.get("exit_code") == 0 and not item.get("timed_out") and not item.get("output_truncated") and item.get("teardown", {}).get("verified") is True)
    result["evidence"]["fidelity_tool"] = {"path": command[1], "sha256": hashlib.sha256(Path(command[1]).read_bytes()).hexdigest()}
    # This existing tool deliberately represents identical-region PSNR as Infinity;
    # only its finite absolute-difference/count metrics are pass/fail oracles here.
    with target.open("rb") as stream:
        raw = stream.read(65537)
    _require(result, "bounded image fidelity report", len(raw) <= 65536)
    value = json.loads(raw)
    rows = value.get("edits", [])
    _require(result, "image fidelity reports exact output", len(rows) == 1 and Path(rows[0].get("png", "")).resolve() == output.resolve())
    row = rows[0]
    evidence = {key: row.get(key) for key in ("sha256", "protected_pixels_changed", "changed_mean_abs_diff", "unchanged_fraction")}
    result["evidence"]["image_fidelity"] = evidence
    _require(result, "protected pixels unchanged", type(row.get("protected_pixels_changed")) is int and row["protected_pixels_changed"] == 0, 0, row.get("protected_pixels_changed"))
    _require(result, "selected region meets approved change bound", _number(row.get("changed_mean_abs_diff"), profile["expected"]["min_changed_mean_abs_diff"], 255), profile["expected"]["min_changed_mean_abs_diff"], row.get("changed_mean_abs_diff"))


def _images(http, identity, profile, context, result, input_data):
    from .scenarios import _require
    params = profile["parameters"]
    payload = {key: params[key] for key in ("prompt", "width", "height", "steps", "cfg", "seed")}
    payload["negativePrompt"] = params.get("negative_prompt", " ")
    payload["targetArea"] = params["width"] * params["height"]
    editing = profile["scenario"] == "image_editing"
    route = "/api/image-edit" if editing else "/api/image-generate"
    if editing:
        payload["imagePaths"] = [_upload_image(http, result, input_data)]
        payload["maskPath"] = _upload_image(http, result, _mask_data(profile))
        payload["maskMode"], payload["maskFeather"], payload["maskCrop"] = "grayscale", 0, False
    result["inputs"]["image_profile"] = payload
    _negative(http, result, route, dict(payload, width=params["width"] + 1))
    _negative(http, result, route, dict(payload, steps=-1))
    if editing:
        _negative(http, result, route, dict(payload, imagePaths=[]))
        _negative(http, result, route, dict(payload, maskMode="invalid-regression-mode"))
        bad_mask = _upload_image(http, result, _solid_mask(1, 1))
        _negative(http, result, route, dict(payload, maskPath=bad_mask))
        no_op_mask = _upload_image(http, result, _solid_mask(params["width"], params["height"]))
        _download_image(http, http.json(route, dict(payload, maskPath=no_op_mask)), profile, context, result,
                        filename="profile-mask-noop.png", expected_hash=png_metadata(input_data, params["width"], params["height"])["pixel_sha256"])
    else:
        _negative(http, result, route, dict(payload, prompt=""))
    output = _download_image(http, http.json(route, payload), profile, context, result)
    if editing:
        _fidelity(profile, context, result, output)
    _require(result, "Qwen image architecture", identity.get("architecture") == "qwen_image", "qwen_image", identity.get("architecture"))
    result["evidence"]["oracle_scope"] = "Exact supplied RGBA8 reference for this bounded recipe; masked editing additionally checks protected RGB and approved selected-region change, plus RGBA no-op preservation. Semantic instruction quality, browser UI, streaming and cancellation are not asserted"


def run(case, context, result):
    """Run only a local owned host; teardown is verified even after oracle failure."""
    from .scenarios import (_Http, _base_args, _require, _attest_offload, backend_evidence,
                            ScenarioFailure, ScenarioUntested)
    profile = context["scenario_profile"]
    validate_profile(profile)
    kind = profile["scenario"]
    input_data, part = None, None
    try:
        if kind == "embedding_media":
            part = _media_part(profile, context, result)
        elif kind == "image_editing":
            input_data = _fixture_bytes(profile, "input")
            result["evidence"]["input_image"] = _image_input(input_data, profile["parameters"]["width"], profile["parameters"]["height"])
            _mask_data(profile)
    except UnsupportedPngFormat as error:
        raise ScenarioUntested(str(error)) from error
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    args = _base_args(context, "cli_image" if kind == "embedding_media" else kind)
    command = list(context["server_command"]) + args + ["--host", "127.0.0.1", "--port", str(port), "--no-webui", "--no-multi-agent"]
    if kind == "embedding_media":
        command.append("--embeddings")
    folder = Path(context["artifact_dir"]) / "media-temp"
    folder.mkdir(parents=True, exist_ok=False)
    env = dict(context.get("env") or {})
    env.update(TEMP=str(folder), TMP=str(folder), TMPDIR=str(folder), VIDEO_MAX_FRAMES="32", VIDEO_SAMPLE_FPS="2")
    process = ManagedProcess(command, cwd=context["repo"], artifact_dir=context["artifact_dir"], stage="profile-server", env=env)
    http = _Http(port, time.monotonic() + context.get("timeout_seconds", 600), result)
    try:
        while True:
            if process.process.poll() is not None:
                raise ScenarioFailure("profile server exited before readiness")
            http.remaining()
            stdout, _ = process.stdout.read()
            stderr, _ = process.stderr.read()
            if "Now listening on: " + http.base in stdout + stderr:
                break
            time.sleep(.1)
        identity = http.json("/api/models")
        result["evidence"]["profile_server_identity"] = identity
        _require(result, "profile actual backend", identity.get("loadedBackend") == context["backend"], context["backend"], identity.get("loadedBackend"))
        _require(result, "profile actual model", identity.get("loaded") == Path(context["model_path"]).name, Path(context["model_path"]).name, identity.get("loaded"))
        _attest_offload(context, result, "profile-startup", stdout + "\n" + stderr)
        if kind == "embedding_media":
            _require(result, "EmbeddingGemma2 media model and projector", identity.get("architecture") == "gemma-embedding2" and identity.get("loadedMmProj") == Path(context["model_options"]["mmproj"]).name,
                     "gemma-embedding2 plus selected projector", identity)
            if profile["parameters"]["modality"] == "image":
                _require(result, "embedding image tower ready", identity.get("visionReady") is True, True, identity.get("visionReady"))
            _embeddings(http, identity, profile, context, result, part)
        else:
            _require(result, "Qwen image architecture before generation", identity.get("architecture") == "qwen_image", "qwen_image", identity.get("architecture"))
            _images(http, identity, profile, context, result, input_data)
    except UnsupportedPngFormat as error:
        raise ScenarioUntested(str(error)) from error
    finally:
        process.stop()
        item = process.snapshot()
        result["processes"].append({k: v for k, v in item.items() if k not in {"stdout", "stderr"}})
        result["evidence"]["profile_server_shutdown"] = item.get("teardown")
        _require(result, "owned profile server stopped", item.get("teardown", {}).get("verified") is True, True, item.get("teardown"))
        proof = backend_evidence(item.get("stdout", "") + "\n" + item.get("stderr", ""), context["backend"])
        # /api/models is the primary identity receipt. Any explicit fallback invalidates it.
        _require(result, "profile server has no backend fallback", proof.get("fallback") is None, None, proof)
        if kind == "embedding_media":
            leftovers = list(folder.glob("tensorsharp-embedding-*"))
            _require(result, "embedding temporary media cleaned", not leftovers, [], [str(p) for p in leftovers])
