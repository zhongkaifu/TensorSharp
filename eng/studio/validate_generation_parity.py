#!/usr/bin/env python3
"""Compare real image-to-video generation through Studio and the chat video API.

Uses the loaded model's advertised quality defaults, the identical uploaded
image, prompt and exact Int64 seed. This runs inference twice and can take an
hour. Requires ffmpeg/ffprobe and a running local host. Evidence must be stored
in ignored artifacts/ or docs/validation/. A successful comparison establishes
route parity for this sample, not broad perceptual quality or a speed benchmark.
"""
import argparse
import hashlib
import json
import mimetypes
import pathlib
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from fractions import Fraction


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def command(*args):
    return subprocess.run([str(a) for a in args], check=True, capture_output=True).stdout


class Host:
    def __init__(self, url, timeout, cookie=None):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Origin": self.url, "X-TensorSharp-Studio": "1"}
        if cookie:
            self.headers["Cookie"] = cookie

    def open(self, route, body=None, content_type="application/json"):
        require(route.startswith("/") and not route.startswith("//"), "Expected a local host route")
        request = urllib.request.Request(self.url + route, data=body,
                                         headers={**self.headers, "Content-Type": content_type})
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"HTTP {error.code} {route}: {error.read()[:2000].decode(errors='replace')}") from error

    def request(self, route, body=None, content_type="application/json"):
        with self.open(route, body, content_type) as response:
            return response.read()

    def api(self, action, payload=None):
        return json.loads(self.request("/api/studio", json.dumps({"action": action, "payload": payload or {}}).encode()))

    def download(self, route, path):
        with self.open(route) as response, path.open("wb") as destination:
            while block := response.read(1024 * 1024):
                destination.write(block)

    def chat_generation(self, payload, output):
        started = time.monotonic()
        with self.open("/api/video-generate/stream", json.dumps(payload).encode()) as response, output.open("w") as log:
            last_progress = None
            for raw in response:
                require(time.monotonic() - started < self.timeout, "Chat generation exceeded timeout")
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                log.write(json.dumps(event, ensure_ascii=False) + "\n")
                log.flush()
                require(not event.get("error"), "Chat generation failed: " + str(event.get("error")))
                if event.get("done"):
                    return event
                progress = (event.get("phase"), event.get("step"), event.get("detail"))
                if progress != last_progress:
                    print("chat", *progress, flush=True)
                    last_progress = progress
        raise RuntimeError("Chat generation stream ended without a completed video")

    def wait_studio(self, project_id, job_id, output):
        started = time.monotonic()
        last_progress = None
        with output.open("a") as log:
            while time.monotonic() - started < self.timeout:
                jobs = self.api("jobs/list", {"projectId": project_id})
                job = next(job for job in jobs if job["id"] == job_id)
                progress = (job["state"], job["stage"], job.get("progress"), job.get("detail"))
                if progress != last_progress:
                    log.write(json.dumps(job, ensure_ascii=False) + "\n")
                    log.flush()
                    print("studio", *progress, flush=True)
                    last_progress = progress
                if job["state"] in ("completed", "failed", "cancelled", "interrupted"):
                    require(job["state"] == "completed", "Studio generation did not complete: " + json.dumps(job))
                    return job
                time.sleep(2)
        raise TimeoutError("Studio generation exceeded timeout; inspect/cancel job " + job_id)


def media_fingerprint(path, expected, audio, output):
    probe = json.loads(command("ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", path))
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    measured = {"width": video["width"], "height": video["height"], "frames": int(video["nb_read_frames"]),
                "fps": float(Fraction(video["avg_frame_rate"]))}
    require(measured == expected, f"Unexpected output geometry for {path}: {measured} != {expected}")
    hashes_path = output / (path.stem + "-rgb24.framehash")
    command("ffmpeg", "-v", "error", "-xerror", "-y", "-i", path, "-map", "0:v:0", "-an",
            "-pix_fmt", "rgb24", "-c:v", "rawvideo", "-f", "framehash", "-hash", "sha256", hashes_path)
    frame_hashes = [line.split(",")[-1].strip() for line in hashes_path.read_text().splitlines() if line and not line.startswith("#")]
    require(len(frame_hashes) == expected["frames"], "Decoded frame count mismatch")
    audio_streams = [stream for stream in probe["streams"] if stream["codec_type"] == "audio"]
    require(bool(audio_streams) == audio, f"Unexpected audio presence in {path}")
    audio_hash = None
    if audio:
        # Decode both AAC tracks to PCM without any rate/channel conversion.
        raw = command("ffmpeg", "-v", "error", "-xerror", "-i", path, "-map", "0:a:0", "-vn",
                      "-c:a", "pcm_f32le", "-f", "f32le", "-")
        audio_hash = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                      "sampleRate": audio_streams[0]["sample_rate"], "channels": audio_streams[0]["channels"]}
    return {"probe": probe, "videoFrameHashes": frame_hashes, "audioPcm": audio_hash}


def source_sample(path):
    document = json.loads(path.read_text())
    project = document.get("project", document)
    assets = [asset for asset in project["assets"] if asset.get("generation")]
    require(bool(assets), "Source project has no generated assets")
    generation = max(assets, key=lambda asset: asset["generation"]["createdAt"])["generation"]
    return generation["request"]["prompt"], int(generation["effectiveParameters"]["seed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5001")
    parser.add_argument("--connection", type=pathlib.Path, help="Optional TensorAgent connection.json; credentials are never reported")
    parser.add_argument("--image", required=True, type=pathlib.Path)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=pathlib.Path)
    prompts.add_argument("--source-project", type=pathlib.Path, help="Reuse latest generated asset's prompt and effective seed, not its old geometry")
    parser.add_argument("--seed", type=int, help="Default 42, or latest source project's effective seed")
    parser.add_argument("--frames", type=int, help="Explicit legal model frame count; otherwise use advertised default")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200, help="Maximum seconds per generation")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--resume", action="store_true", help="Reuse completed stages from this output directory after an interrupted validation")
    options = parser.parse_args()
    repository = pathlib.Path(__file__).resolve().parents[2]
    output = options.output.resolve()
    require(any(output.is_relative_to(repository / directory) for directory in ("artifacts", "docs/validation")),
            "Evidence must be under ignored artifacts/ or docs/validation/")
    output.mkdir(parents=True, exist_ok=True)
    require(options.resume or not (output / "requests.json").exists(), "Use a new output directory or --resume")
    # Fail before expensive work if either decoder tool is unavailable.
    command("ffmpeg", "-version")
    command("ffprobe", "-version")
    prompt, seed = source_sample(options.source_project) if options.source_project else (
        options.prompt_file.read_text().strip() if options.prompt_file else options.prompt, 42)
    if options.seed is not None:
        seed = options.seed
    require(bool(prompt.strip()) and 0 <= seed <= 9223372036854775807, "A prompt and exact non-negative Int64 seed are required")
    connection = json.loads(options.connection.read_text()) if options.connection else {}
    host = Host(connection.get("baseUrl", options.url), options.timeout, connection.get("cookie"))
    capabilities = host.api("capabilities")
    models = [model for model in capabilities["models"] if model["available"]]
    require(len(models) == 1, "Expected one available loaded model")
    model = models[0]
    for key in ("defaultWidth", "defaultHeight", "defaultFps", "defaultFrameCount"):
        require(model.get(key, 0) > 0, f"Host must advertise {key}; rebuild with the Studio quality defaults fix")
    count = options.frames if options.frames is not None else model["defaultFrameCount"]
    require(count <= model["maxFrames"] and count >= model["temporalOffset"] and
            (count - model["temporalOffset"]) % model["temporalMultiple"] == 0, "Frame count is not on the model's temporal grid")
    audio = model["audioOutput"] and not options.no_audio
    expected = {"width": model["defaultWidth"], "height": model["defaultHeight"], "frames": count, "fps": model["defaultFps"]}
    image_bytes = options.image.read_bytes()
    identity = {"host": host.url, "model": model, "prompt": prompt, "seed": seed, "expected": expected,
                "audio": audio, "inputSha256": hashlib.sha256(image_bytes).hexdigest()}
    manifest_path = output / "requests.json"
    if options.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        require(manifest["identity"] == identity, "Resume inputs/defaults differ from saved validation")
    else:
        project = host.api("projects/create", {"name": "Studio / chat quality parity", "width": expected["width"],
                           "height": expected["height"], "fpsNumerator": expected["fps"], "fpsDenominator": 1})
        project_id = project["id"]
        imported = json.loads(host.request(f"/api/studio/projects/{project_id}/import?fileName=" + urllib.parse.quote(options.image.name),
                                          image_bytes, "application/octet-stream"))
        asset = next(asset for asset in imported["assets"] if asset["sha256"] == identity["inputSha256"])
        boundary = "StudioParity" + uuid.uuid4().hex
        file_name = "conditioning" + options.image.suffix.lower()
        mime = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        multipart = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{file_name}"\r\nContent-Type: {mime}\r\n\r\n'.encode()
                     + image_bytes + f"\r\n--{boundary}--\r\n".encode())
        uploaded = json.loads(host.request("/api/upload", multipart, "multipart/form-data; boundary=" + boundary))
        require(hashlib.sha256(host.request(uploaded["url"])).hexdigest() == identity["inputSha256"], "Chat upload changed image bytes")
        studio = {"projectId": project_id, "requestId": uuid.uuid4().hex, "modelId": model["modelId"], "prompt": prompt,
                  "width": expected["width"], "height": expected["height"], "frameCount": count, "fps": expected["fps"],
                  "steps": model["defaultSteps"], "cfgScale": model["defaultCfgScale"], "seed": seed,
                  "conditioningAssetId": asset["id"], "generateAudio": audio}
        chat = {"prompt": prompt, **expected, "steps": model["defaultSteps"], "cfg": model["defaultCfgScale"],
                "seed": seed, "imagePath": uploaded["file"], "generateAudio": audio, "videoMode": "i2v"}
        manifest = {"identity": identity, "studio": studio, "chat": chat}
        save(manifest_path, manifest)
    report = {"status": "running", "identity": identity, "scope": "Native image-to-video route parity for one fixed sample; not broad perceptual quality or performance equivalence.",
              "sourceRevision": command("git", "-C", repository, "rev-parse", "HEAD").decode().strip(),
              "ggmlRevision": command("git", "-C", repository / "ExternalProjects/ggml", "rev-parse", "HEAD").decode().strip(),
              "ggmlStatus": command("git", "-C", repository / "ExternalProjects/ggml", "status", "--porcelain").decode().strip()}
    save(output / "report.json", report)
    try:
        job_path = output / "studio-job.json"
        job = json.loads(job_path.read_text()) if options.resume and job_path.exists() else host.api("jobs/generate", manifest["studio"])
        save(job_path, job)
        job = host.wait_studio(manifest["studio"]["projectId"], job["id"], output / "studio-progress.jsonl")
        save(job_path, job)
        project = host.api("projects/get", {"projectId": job["projectId"]})
        asset = next(asset for asset in project["assets"] if asset["id"] == job["outputAssetId"])
        save(output / "studio-asset.json", asset)
        provenance = asset["generation"]
        for key, value in {**expected, "seed": seed, "steps": model["defaultSteps"], "cfgScale": model["defaultCfgScale"]}.items():
            require(Fraction(provenance["effectiveParameters"][key]) == value, "Studio provenance mismatch: " + key)
        require(provenance["request"]["seed"] == seed and provenance["request"]["prompt"] == prompt and
                provenance["request"]["conditioningAssetId"] == manifest["studio"]["conditioningAssetId"], "Studio request provenance mismatch")
        host.download(f'/api/studio/projects/{job["projectId"]}/assets/{asset["id"]}', output / "studio.mp4")
        chat_path = output / "chat-result.json"
        result = json.loads(chat_path.read_text()) if options.resume and chat_path.exists() else host.chat_generation(manifest["chat"], output / "chat-progress.jsonl")
        save(chat_path, result)
        for key, value in {**expected, "seed": seed}.items():
            require(result[key] == value, "Chat result mismatch: " + key)
        host.download(result["url"], output / "chat.mp4")
        studio_media = media_fingerprint(output / "studio.mp4", expected, audio, output)
        chat_media = media_fingerprint(output / "chat.mp4", expected, audio, output)
        mismatches = [index for index, (left, right) in enumerate(zip(studio_media["videoFrameHashes"], chat_media["videoFrameHashes"])) if left != right]
        report.update(studio=studio_media, chat=chat_media, studioProvenance=provenance,
                      differingVideoFrames=mismatches, audioPcmEqual=studio_media["audioPcm"] == chat_media["audioPcm"])
        require(not mismatches, f"Decoded video differs on {len(mismatches)} frames")
        require(report["audioPcmEqual"], "Decoded audio differs")
        report["status"] = "passed"
        print(f"PARITY PASSED: {count} decoded RGB frames identical; audio={'identical' if audio else 'disabled'}; {output}", flush=True)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        save(output / "report.json", report)


if __name__ == "__main__":
    main()
