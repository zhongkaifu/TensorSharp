#!/usr/bin/env python3
"""Exercise a running, real Studio host with deterministic media; no inference doubles.

Requires Python 3 stdlib, operator-installed ffmpeg/ffprobe and a running host.
All generated evidence belongs below ignored artifacts/ or docs/validation/.
See docs/video-studio.md for commands. Never writes a session token to reports.
"""
import argparse
import json
import math
import pathlib
import struct
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


def run(*args, binary=False):
    result = subprocess.run([str(x) for x in args], check=True, capture_output=True)
    return result.stdout if binary else result.stdout.decode()


class Host:
    def __init__(self, url, cookie=None):
        self.url = url.rstrip("/")
        self.headers = {"X-TensorSharp-Studio": "1"}
        if cookie:
            self.headers["Cookie"] = cookie

    def request(self, route, body=None, content_type="application/json", headers=None):
        request = urllib.request.Request(self.url + route, data=body,
                                        headers={**self.headers, **(headers or {})})
        if body is not None:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as error:
            return error.code, error.read(), dict(error.headers)

    def api(self, action, payload=None, status=200):
        code, body, _ = self.request("/api/studio", json.dumps({"action": action, "payload": payload or {}}).encode())
        assert code == status, (action, code, body[:1800].decode(errors="replace"))
        return json.loads(body)

    def upload(self, project_id, file):
        route = "/api/studio/projects/" + project_id + "/import?fileName=" + urllib.parse.quote(file.name)
        code, body, _ = self.request(route, file.read_bytes(), "application/octet-stream")
        assert code == 200, ("import", code, body[:1800])
        return json.loads(body)

    def poll(self, project_id, job_id, timeout=1200):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            jobs = self.api("jobs/list", {"projectId": project_id})
            if isinstance(jobs, dict):
                jobs = jobs["jobs"]
            job = next(x for x in jobs if x["id"] == job_id)
            stage = (job["state"], job["stage"])
            if stage != last:
                print("job", job_id, *stage, flush=True)
                last = stage
            if job["state"] in ("completed", "failed", "cancelled", "interrupted"):
                assert job["state"] == "completed", job
                return job
            time.sleep(1)
        raise TimeoutError("Studio job did not reach a terminal state: " + job_id)


def uid():
    return uuid.uuid4().hex


def fixture(output):
    output.mkdir(parents=True, exist_ok=True)
    for color, tone in [("red", 440), ("blue", 880)]:
        target = output / (color + ".mp4")
        run("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=320x180:r=24:d=2",
            "-f", "lavfi", "-i", f"sine=frequency={tone}:sample_rate=48000:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", target)


def verify_media(path, expected_frames=None, expected_width=None, expected_height=None):
    probe = json.loads(run("ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", path))
    run("ffmpeg", "-v", "error", "-xerror", "-i", path, "-f", "null", "-")
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert video["codec_name"] == "h264", video
    if expected_frames is not None:
        assert abs(int(video["nb_read_frames"]) - expected_frames) <= 1, video
    if expected_width is not None:
        assert video["width"] == expected_width and video["height"] == expected_height, video
    return probe


def sample_rgb(path, seconds):
    raw = run("ffmpeg", "-v", "error", "-ss", seconds, "-i", path,
              "-frames:v", 1, "-vf", "crop=32:32:0:0,scale=1:1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-", binary=True)
    assert len(raw) == 3, len(raw)
    return list(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8187")
    parser.add_argument("--connection", type=pathlib.Path, help="TensorAgentHost connection.json (token is never reported)")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--generate", action="store_true", help="Run real inference using the already loaded video model")
    parser.add_argument("--reopen", help="Project ID from a prior run, after restarting its host")
    options = parser.parse_args()
    output = options.output.resolve()
    repository = pathlib.Path(__file__).resolve().parents[2]
    assert any(output.is_relative_to(repository / folder) for folder in ("artifacts", "docs/validation")), "Evidence must be in ignored artifacts/ or docs/validation/"
    output.mkdir(parents=True, exist_ok=True)
    connection = json.loads(options.connection.read_text()) if options.connection else {}
    host = Host(connection.get("baseUrl", options.url), connection.get("cookie"))
    if options.reopen:
        project = host.api("projects/get", {"projectId": options.reopen})
        assert project["clips"] and project["assets"], project
        job = host.api("jobs/render", {"projectId": project["id"], "expectedRevision": project["revision"], "requestId": uid()})
        completed = host.poll(project["id"], job["id"])
        code, media, _ = host.request(f'/api/studio/projects/{project["id"]}/jobs/{job["id"]}/output')
        assert code == 200, code
        path = output / "reopened.mp4"
        path.write_bytes(media)
        probe = verify_media(path, project["durationFrames"], project["width"], project["height"])
        (output / "reopened.json").write_text(json.dumps({"project": project, "job": completed, "probe": probe}, indent=2))
        print("REOPEN PASSED", project["id"], flush=True)
        return

    fixture(output)
    capabilities = host.api("capabilities")
    project = host.api("projects/create", {"name": "Studio deterministic integration", "width": 320, "height": 180,
                                            "fpsNumerator": 24, "fpsDenominator": 1})
    project_id = project["id"]

    def current():
        return host.api("projects/get", {"projectId": project_id})

    def commands(ops, command_id=None):
        return host.api("commands/apply", {"projectId": project_id, "expectedRevision": current()["revision"],
                                            "commandId": command_id or uid(), "commands": ops})

    # Exercise the existing chat upload surface, then its Studio action.
    boundary = "StudioFixture" + uid()
    uploaded_bytes = (("--" + boundary + '\r\nContent-Disposition: form-data; name="file"; filename="red.mp4"\r\nContent-Type: video/mp4\r\n\r\n').encode()
                      + (output / "red.mp4").read_bytes() + ("\r\n--" + boundary + "--\r\n").encode())
    upload_status, uploaded_body, _ = host.request("/api/upload", uploaded_bytes, "multipart/form-data; boundary=" + boundary)
    assert upload_status == 200, (upload_status, uploaded_body[:300])
    uploaded = json.loads(uploaded_body)
    red = host.api("assets/import-chat", {"projectId": project_id, "url": uploaded["url"]})
    blue = host.upload(project_id, output / "blue.mp4")
    red = next(a for a in red["assets"] if a["name"].endswith(".mp4")) if "assets" in red else red.get("asset", red)
    blue = next(a for a in blue["assets"] if a["name"] == "blue.mp4") if "assets" in blue else blue.get("asset", blue)
    project = current()
    video_track = next(t["id"] for t in project["tracks"] if t["kind"] == "video")
    audio_track = next(t["id"] for t in project["tracks"] if t["kind"] == "audio")
    clip_id = uid()
    commands([
        {"type": "insertClip", "clipId": clip_id, "assetId": red["id"], "trackId": video_track, "startFrame": 0, "durationFrames": 48},
        {"type": "insertClip", "assetId": blue["id"], "trackId": video_track, "startFrame": 48, "durationFrames": 48},
        {"type": "insertClip", "assetId": red["id"], "trackId": audio_track, "startFrame": 24, "durationFrames": 48, "volume": 0.25},
        {"type": "addCaption", "text": "TENSORSHARP STUDIO", "startFrame": 6, "durationFrames": 24},
    ])
    before = current()
    proposal = host.api("proposals/create", {"projectId": project_id, "expectedRevision": before["revision"],
                                             "summary": "Trim the opening clip and reduce its volume", "commands": [
        {"type": "trimClip", "clipId": clip_id, "sourceStartFrame": 0, "durationFrames": 24},
        {"type": "updateClip", "clipId": clip_id, "volume": 0.5}]})
    commands([{"type": "updateTrack", "trackId": audio_track, "volume": 0.5}])
    stale_code, stale_body, _ = host.request("/api/studio", json.dumps({"action": "proposals/apply", "payload": {
        "projectId": project_id, "proposalId": proposal["id"], "commandId": uid()}}).encode())
    assert stale_code in (400, 409), (stale_code, stale_body)
    host.api("commands/undo", {"projectId": project_id, "expectedRevision": current()["revision"], "commandId": uid()})
    latest = current()
    proposal = host.api("proposals/create", {"projectId": project_id, "expectedRevision": latest["revision"],
                                             "summary": "Split opening shot as one edit", "commands": [
        {"type": "splitClip", "clipId": clip_id, "atFrame": 24}]})
    host.api("proposals/apply", {"projectId": project_id, "proposalId": proposal["id"], "commandId": uid()})
    assert len(current()["clips"]) == 4
    host.api("commands/undo", {"projectId": project_id, "expectedRevision": current()["revision"], "commandId": uid()})
    assert len(current()["clips"]) == 3
    host.api("commands/redo", {"projectId": project_id, "expectedRevision": current()["revision"], "commandId": uid()})
    assert len(current()["clips"]) == 4

    generation = None
    if options.generate:
        models = capabilities["models"]
        model = next(m for m in models if m["available"] and m["textToVideo"])
        request = {"projectId": project_id, "requestId": uid(), "modelId": model["modelId"],
                   "prompt": "A red toy car drives slowly along a quiet road in warm afternoon light.",
                   "width": 512, "height": 512, "frameCount": model["defaultFrameCount"], "fps": model.get("fixedFps") or 24,
                   "steps": model["defaultSteps"], "cfgScale": model["defaultCfgScale"], "seed": 42,
                   "generateAudio": model.get("audioOutput", False), "shotId": "shot-three"}
        job = host.api("jobs/generate", request)
        generation = host.poll(project_id, job["id"])
        generated = next(a for a in current()["assets"] if a["id"] == generation["outputAssetId"])
        assert generated["generation"]["request"]["prompt"] == request["prompt"]
        assert generated["generation"]["effectiveParameters"]["seed"] == "42"
        commands([{"type": "insertClip", "assetId": generated["id"], "trackId": video_track, "startFrame": 96,
                   "durationFrames": int(round(generated["durationSeconds"] * 24)), "shotId": "shot-three"}])

    project = current()
    job = host.api("jobs/render", {"projectId": project_id, "expectedRevision": project["revision"], "requestId": uid()})
    # Mutating after admission must not change the captured render.
    commands([{"type": "settings", "name": "Edited while rendering"}])
    completed = host.poll(project_id, job["id"])
    assert completed["projectRevision"] == project["revision"], completed
    code, media, media_headers = host.request(f'/api/studio/projects/{project_id}/jobs/{job["id"]}/output')
    assert code == 200, (code, media[:300])
    exported = output / "studio-sample.mp4"
    exported.write_bytes(media)
    probe = verify_media(exported, project["durationFrames"], 320, 180)
    red_pixel, blue_pixel = sample_rgb(exported, 0.4), sample_rgb(exported, 2.4)
    assert red_pixel[0] > red_pixel[1] + 100 and red_pixel[0] > red_pixel[2] + 100, red_pixel
    assert blue_pixel[2] > blue_pixel[0] + 100 and blue_pixel[2] > blue_pixel[1] + 100, blue_pixel
    audio = run("ffmpeg", "-v", "error", "-i", exported, "-vn", "-ar", "48000", "-ac", "1", "-f", "f32le", "-", binary=True)
    samples = struct.unpack("<" + "f" * (len(audio) // 4), audio)
    rms = math.sqrt(sum(x * x for x in samples) / len(samples))
    assert rms > 0.01, rms
    code, ranged, range_headers = host.request(f'/api/studio/projects/{project_id}/jobs/{job["id"]}/output', headers={"Range": "bytes=0-99"})
    assert code == 206 and len(ranged) == 100, (code, len(ranged))

    # A second project must not resolve an asset owned by the first.
    other = host.api("projects/create", {"name": "Isolation check"})
    denied, _, _ = host.request(f'/api/studio/projects/{other["id"]}/assets/{red["id"]}')
    assert denied in (400, 404), denied
    thumbnail_status, thumbnail, _ = host.request(f'/api/studio/thumbnail/{project_id}/{red["id"]}')
    assert thumbnail_status == 200 and thumbnail[:2] == b'\xff\xd8' and len(thumbnail) < 1024 * 1024
    waveform_status, waveform_json, _ = host.request(f'/api/studio/waveform/{project_id}/{red["id"]}')
    waveform = json.loads(waveform_json)
    assert waveform_status == 200 and len(waveform["peaks"]) == 512 and max(waveform["peaks"]) > 0.01
    package_status, package_bytes, _ = host.request(f'/api/studio/projects/{project_id}/package')
    assert package_status == 200
    (output / "sample.tsstudio").write_bytes(package_bytes)
    imported_status, imported_bytes, _ = host.request('/api/studio/packages/import', package_bytes, 'application/zip')
    imported = json.loads(imported_bytes)
    assert imported_status == 200 and imported["id"] != project_id and len(imported["clips"]) == len(current()["clips"]), imported
    assert [a["sha256"] for a in imported["assets"]] == [a["sha256"] for a in current()["assets"]]
    for headers in ({"Origin": "https://untrusted.example"}, {"X-TensorSharp-Studio": ""}):
        rejected, _, _ = host.request('/api/studio', json.dumps({"action": "projects/create", "payload": {"name": "must reject"}}).encode(), headers=headers)
        assert rejected == 403, rejected
    loopback_gate = None
    if not connection:
        rejected, _, _ = host.request('/api/studio', json.dumps({'action': 'projects/list', 'payload': {}}).encode(), headers={'Host': 'untrusted.example'})
        assert rejected == 403, ('Unexpected Host must not bypass the server loopback boundary', rejected)
        loopback_gate = True
    unsupported, unsupported_bytes, _ = host.request('/api/studio', json.dumps({"action": "jobs/generate", "payload": {"projectId": project_id, "retake": True}}).encode())
    assert unsupported == 400 and b"retake" in unsupported_bytes, (unsupported, unsupported_bytes)
    report = {"passed": True, "hostUrl": host.url, "projectId": project_id, "project": current(), "render": completed,
              "generation": generation, "capabilities": capabilities, "probe": probe, "colorSamples": [red_pixel, blue_pixel],
              "audioRms": rms, "staleProposalStatus": stale_code, "crossProjectStatus": denied,
              "portableProjectId": imported["id"], "thumbnailBytes": len(thumbnail), "waveformBuckets": len(waveform["peaks"]),
              "csrfRejected": True, "serverHostBoundaryRejected": loopback_gate, "unsupportedOptionRejected": True,
              "scope": "Real host HTTP, deterministic fixture export/full decode; generation only when --generate; no native app UI/device claim."}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print("PASSED", project_id, exported, flush=True)


if __name__ == "__main__":
    main()
