#!/usr/bin/env python3
"""Run declared image/video HTTP workloads and validate downloaded media.

Case JSON supplies uploads {alias:path} and cases [{id,endpoint,request,
expected}]. Exact string values @alias in requests resolve to uploaded file
names. This records structural/numerical checks; semantic visual quality is
explicitly pending separate inspection, never inferred from a valid MP4/PNG.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import subprocess
import statistics
import time
import urllib.parse

import numpy as np
from PIL import Image, ImageOps
import requests


def digest(data):
    return hashlib.sha256(data).hexdigest()


def media_pixels(path):
    if path.suffix == ".png":
        with Image.open(path) as source:
            if source.format != "PNG":
                raise ValueError("Image-edit artifact is not a PNG")
            image = np.asarray(source.convert("RGB"))
        return {"width": image.shape[1], "height": image.shape[0], "frames": 1}, image
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,codec_name", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=120)
    stream = json.loads(probe.stdout)["streams"][0]
    metadata = {key: int(stream[key]) for key in ("width", "height")}
    metadata["frames"] = int(stream["nb_read_frames"])
    metadata["codec"] = stream["codec_name"]
    numerator, denominator = stream["r_frame_rate"].split("/")
    metadata["fps"] = float(numerator) / float(denominator)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True, timeout=180).stdout
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(metadata["frames"], metadata["height"], metadata["width"], 3)
    return metadata, pixels


def audio_samples(path):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=sample_rate,channels,codec_name", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=60)
    streams = json.loads(probe.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError("Generated audio does not contain one audio stream")
    stream = streams[0]
    metadata = {"sample_rate": int(stream["sample_rate"]), "channels": int(stream["channels"]),
                "codec": stream["codec_name"]}
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-"],
                         capture_output=True, check=True, timeout=60).stdout
    samples = np.frombuffer(pcm, dtype="<f4")
    if not samples.size or samples.size % metadata["channels"]:
        raise ValueError("Generated audio has no complete sample frames")
    if not np.isfinite(samples).all() or np.std(samples) < 1e-6:
        raise ValueError("Generated audio is non-finite or silent")
    metadata["samples_per_channel"] = samples.size // metadata["channels"]
    metadata["duration_seconds"] = metadata["samples_per_channel"] / metadata["sample_rate"]
    metadata.update(samples=int(samples.size), rms=float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))),
                    clipped_fraction=float(np.mean(np.abs(samples) >= .9999)), sha256=digest(pcm))
    if metadata["clipped_fraction"] >= .99:
        raise ValueError("Generated audio is almost entirely clipped")
    return metadata, samples


def download(args, url, path):
    target = urllib.parse.urljoin(args.url + "/", url)
    if urllib.parse.urlsplit(target).netloc != urllib.parse.urlsplit(args.url).netloc:
        raise ValueError("Generated artifact is not hosted by the tested server")
    response = requests.get(target, timeout=(10, args.timeout))
    response.raise_for_status()
    path.write_bytes(response.content)
    return {"path": str(path), "url": url, "bytes": len(response.content), "sha256": digest(response.content)}


def generation_response(args, case, result, directory):
    """Retain progress and final envelopes for both production HTTP transports."""
    streaming = case["endpoint"].endswith("/stream")
    started = time.perf_counter()
    with requests.post(args.url + case["endpoint"], json=case["request"], stream=streaming,
                       timeout=(10, args.timeout)) as response:
        result["http_status"] = response.status_code
        response.raise_for_status()
        if not streaming:
            body = response.json()
        else:
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise ValueError("Streaming generation did not return event-stream content")
            events, body, payload = [], None, []

            def consume():
                nonlocal body
                if not payload:
                    return
                frame = json.loads("\n".join(payload))
                if body is not None:
                    raise ValueError("Generation sent an event after its final frame")
                event = {"received_seconds": time.perf_counter() - started, **frame}
                if frame.get("image"):
                    prefix = "data:image/png;base64,"
                    if not frame["image"].startswith(prefix):
                        raise ValueError("Generation preview is not an inline PNG")
                    encoded = base64.b64decode(frame["image"][len(prefix):], validate=True)
                    with Image.open(io.BytesIO(encoded)) as preview:
                        if preview.format != "PNG":
                            raise ValueError("Generation preview has incorrect format")
                        preview.verify()
                    preview_path = directory / f"preview-{len(events):04d}.png"
                    preview_path.write_bytes(encoded)
                    event["image"] = {"path": str(preview_path), "sha256": digest(encoded)}
                if frame.get("done") is True:
                    body = frame
                elif not (frame.get("imageEdit") or frame.get("videoGen")):
                    raise ValueError("Unknown generation progress event")
                events.append(event)
                payload.clear()

            result["events"] = events
            for line in response.iter_lines(decode_unicode=True):
                if line == "":
                    consume()
                elif line.startswith("data:"):
                    payload.append(line[5:].lstrip())
            consume()
            if body is None:
                raise ValueError("Generation stream ended without a final event")
            if len(events) < 2:
                raise ValueError("Generation stream returned no progress before completion")
            result["first_event_seconds"] = events[0]["received_seconds"]
            body = {"ok": not body.get("error"), **body}
        result["wall_seconds"] = time.perf_counter() - started
        result["response"] = body
        if case["endpoint"] == "/v1/videos/generations":
            if len(body.get("data", [])) != 1:
                raise ValueError("OpenAI-shaped video response did not contain one artifact")
            artifact = body["data"][0]
            body = {**body, **artifact, "ok": True, "audioUrl": body.get("audio_url")}
            if case["request"].get("response_format") == "b64_json":
                decoded = base64.b64decode(artifact["b64_json"], validate=True)
                result["inline_artifact"] = {"bytes": len(decoded), "sha256": digest(decoded)}
                # Keep the exact bytes once on disk, without duplicating base64 in JSON.
                (directory / "inline-generated.mp4").write_bytes(decoded)
                result["response"]["data"][0]["b64_json"] = result["inline_artifact"]
        return body


def run_case(args, case, tag, reference):
    result = {"id": case["id"], "tag": tag, "status": "failed", "request": case["request"],
              "endpoint": case["endpoint"], "semantic_quality": "pending-separate-visual-review"}
    directory = args.output / tag
    directory.mkdir(parents=True, exist_ok=True)
    try:
        body = generation_response(args, case, result, directory)
        if not body.get("ok") or not body.get("url"):
            raise ValueError("Generation response did not identify a successful artifact")
        suffix = ".png" if "image-edit" in case["endpoint"] else ".mp4"
        output = directory / ("generated" + suffix)
        result["artifact"] = download(args, body["url"], output)
        if result.get("inline_artifact", result["artifact"])["sha256"] != result["artifact"]["sha256"]:
            raise ValueError("Inline video bytes differ from the downloadable artifact")
        metadata, pixels = media_pixels(output)
        result["decoded_media"] = {**metadata, "sha256": digest(pixels.tobytes()),
                                   "mean": float(pixels.mean()), "std": float(pixels.std())}
        for key, expected in case.get("expected", {}).items():
            if key == "audio" or key.startswith("audio_"):
                continue
            if key not in metadata:
                raise ValueError(f"Cannot verify required media field {key}")
            if metadata[key] != expected:
                raise ValueError(f"Decoded {key} {metadata[key]} differs from requested {expected}")
        for key in ("width", "height", "frames", "fps"):
            if key in body and key in metadata and body[key] != metadata[key]:
                raise ValueError(f"Response {key} differs from the decoded artifact")
        if pixels.std() < 1 or pixels.max() - pixels.min() < 8:
            raise ValueError("Generated media is essentially uniform/blank")
        frames = pixels if pixels.ndim == 4 else pixels[None]
        selected = np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int)
        thumbnails = [ImageOps.pad(Image.fromarray(frames[index]), (320, 192)) for index in selected]
        sheet = Image.new("RGB", (320 * len(thumbnails), 192))
        for index, thumbnail in enumerate(thumbnails):
            sheet.paste(thumbnail, (320 * index, 0))
        sheet.save(directory / "contact-sheet.png")
        result["contact_sheet"] = str(directory / "contact-sheet.png")
        if case.get("expected", {}).get("audio") is False and body.get("audioUrl"):
            raise ValueError("Video-only generation returned an unexpected audio sidecar")
        if case.get("expected", {}).get("audio"):
            if not body.get("audioUrl"):
                raise ValueError("Requested joint audio generation produced no audio URL")
            audio_path = directory / "generated.wav"
            result["audio_artifact"] = download(args, body["audioUrl"], audio_path)
            result["audio_decoded"], samples = audio_samples(audio_path)
            for key, expected in case["expected"].items():
                if key.startswith("audio_") and result["audio_decoded"].get(key[6:]) != expected:
                    raise ValueError(f"Generated audio {key[6:]} differs from expected {expected}")
        if reference:
            old = next(item for item in reference["cases"] if item["tag"] == tag)
            if old["status"] != "passed-structural" or old["wall_seconds"] <= 0:
                raise ValueError("Reference case did not pass or has invalid latency")
            old_path = Path(old["artifact"]["path"])
            old_metadata, old_pixels = media_pixels(old_path)
            if old_metadata != metadata or old_pixels.shape != pixels.shape:
                raise ValueError("Baseline/candidate decoded media shapes differ")
            mse = float(np.mean((pixels.astype(np.float32) - old_pixels.astype(np.float32)) ** 2))
            psnr = float(10 * np.log10(255 ** 2 / mse)) if mse else None
            result["comparison"] = {"reference_tag": old["tag"], "decoded_exact": mse == 0,
                                    "psnr_db": psnr, "latency_ratio": result["wall_seconds"] / old["wall_seconds"]}
            if psnr is not None and psnr < args.minimum_psnr:
                raise ValueError(f"Candidate output differs from baseline: PSNR {psnr:.3f} dB")
            if case.get("expected", {}).get("audio"):
                old_audio, old_samples = audio_samples(Path(old["audio_artifact"]["path"]))
                for key in ("sample_rate", "channels", "samples_per_channel"):
                    if old_audio[key] != result["audio_decoded"][key]:
                        raise ValueError(f"Baseline/candidate audio {key} differs")
                error_power = float(np.mean((samples.astype(np.float64) - old_samples) ** 2))
                reference_power = float(np.mean(old_samples.astype(np.float64) ** 2))
                snr = float(10 * np.log10(reference_power / error_power)) if error_power else None
                result["comparison"].update(audio_decoded_exact=error_power == 0, audio_snr_db=snr)
                if snr is not None and snr < args.minimum_audio_snr:
                    raise ValueError(f"Candidate audio differs from baseline: SNR {snr:.3f} dB")
        result["status"] = "passed-structural"
    except Exception as error:
        result["error"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--minimum-psnr", type=float, default=40)
    parser.add_argument("--minimum-audio-snr", type=float, default=40)
    parser.add_argument("--concurrency", default="1,2")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--cases", help="Comma-separated case IDs for an explicitly scoped run")
    parser.add_argument("--no-warmup", action="store_true", help="Single-render preflight; separate from warmed benchmark evidence")
    args = parser.parse_args()
    degrees = list(map(int, args.concurrency.split(",")))
    if args.repeats < 1 or not degrees or any(degree < 1 for degree in degrees):
        parser.error("repeats and concurrency must be positive")
    plan = json.loads(args.plan.read_text())
    if args.cases:
        selected = args.cases.split(",")
        unknown = set(selected) - {case["id"] for case in plan["cases"]}
        if unknown:
            parser.error("Unknown case IDs: " + ", ".join(sorted(unknown)))
        plan["cases"] = [case for case in plan["cases"] if case["id"] in selected]
    if not plan["cases"]:
        parser.error("Plan contains no cases")
    reference = json.loads(args.reference.read_text()) if args.reference else None
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"plan": plan, "url": args.url, "started_at_unix": time.time(), "uploads": {},
              "run_complete": False, "cases": [], "warmup": [],
              "measurement_scope": "preflight-without-warmup" if args.no_warmup else "warmed-repeated",
              "comparison_bounds": {"minimum_psnr_db": args.minimum_psnr, "minimum_audio_snr_db": args.minimum_audio_snr}}
    for alias, filename in plan.get("uploads", {}).items():
        path = Path(filename)
        with path.open("rb") as source:
            response = requests.post(args.url + "/api/upload", files={"file": (path.name, source)}, timeout=(10, 120))
        response.raise_for_status()
        payload = response.json()
        if not payload.get("file"):
            raise ValueError("Upload did not return a server file name: " + str(payload))
        report["uploads"][alias] = {"path": str(path), "sha256": digest(path.read_bytes()), "response": payload}

    def substitute(value):
        if isinstance(value, str) and value.startswith("@"):
            if value[1:] not in report["uploads"]:
                raise ValueError(f"Unknown upload alias {value}")
            return report["uploads"][value[1:]]["response"]["file"]
        if isinstance(value, dict):
            return {key: substitute(item) for key, item in value.items()}
        if isinstance(value, list):
            return [substitute(item) for item in value]
        return value

    def schedule(case):
        case_degrees = case.get("concurrency", degrees)
        case_repeats = case.get("repeats", args.repeats)
        if case_repeats < 1 or not case_degrees or any(degree < 1 for degree in case_degrees):
            raise ValueError("Case repeats and concurrency must be positive")
        return case_degrees, case_repeats

    planned_tags = []
    for case in plan["cases"]:
        case_degrees, case_repeats = schedule(case)
        if not args.no_warmup and case.get("warmup", True):
            planned_tags.append(case["id"] + "-warmup")
        planned_tags += [f"{case['id']}-c{degree}-r{repeat}-i{index}" for degree in case_degrees
                         for repeat in range(case_repeats) for index in range(degree)]
    failed = False
    for original in plan["cases"]:
        case = substitute(original)
        case_degrees, case_repeats = schedule(case)
        if not args.no_warmup and case.get("warmup", True):
            report["warmup"].append(run_case(args, case, case["id"] + "-warmup", None))
            failed = report["warmup"][-1]["status"] != "passed-structural"
            if failed:
                break
        for degree in case_degrees:
            for repeat in range(case_repeats):
                with ThreadPoolExecutor(max_workers=degree) as pool:
                    cases = list(pool.map(lambda index: run_case(args, case,
                        f"{case['id']}-c{degree}-r{repeat}-i{index}", reference), range(degree)))
                for item in cases:
                    item.update(concurrency=degree, repeat=repeat)
                report["cases"].extend(cases)
                (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
                print(case["id"], degree, repeat, [item["status"] for item in cases], flush=True)
                failed = any(item["status"] != "passed-structural" for item in cases)
                if failed:
                    break
            if failed:
                break
        if failed:
            break
    completed_tags = {item["tag"] for item in report["cases"] + report["warmup"]}
    report["not_run"] = [tag for tag in planned_tags if tag not in completed_tags]
    report["run_complete"] = not report["not_run"]
    if failed:
        report["stop_reason"] = "A request failed validation; remaining requests require investigation before retrying."
    report["latency"] = {}
    for case in plan["cases"]:
        for degree in case.get("concurrency", degrees):
            samples = [item["wall_seconds"] for item in report["cases"]
                       if item["id"] == case["id"] and item["concurrency"] == degree
                       and item["status"] == "passed-structural"]
            if samples:
                report["latency"][f"{case['id']}-c{degree}"] = {
                    "successful_samples": len(samples), "samples_seconds": samples,
                    "median_seconds": statistics.median(samples), "min_seconds": min(samples), "max_seconds": max(samples)}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(any(item["status"] != "passed-structural" for item in report["cases"] + report["warmup"]))


if __name__ == "__main__":
    raise SystemExit(main())
