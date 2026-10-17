#!/usr/bin/env python3
"""Run every generation mode the loaded video model advertises through a real Studio host.

For each advertised mode (t2v, i2v, fl2v, keyframes, ref) this submits one Studio
generation with synthetic still images, waits for the real native job, and checks
the recorded effective mode. Pinned frames are checked relatively: an i2v/fl2v
output's first frame must be closer to the first image than to the last image, and
an fl2v output's last frame the reverse. "fl2v-last" (requested with --modes, run
when fl2v is advertised) is fl2v with only its last frame: the first frame is left
empty, and only the output's last frame is checked. Keyframe and reference distances are
recorded, not asserted (neither is a hard pixel pin). Every mode the model reports
as unavailable must be refused with its advertised reason before a job exists.

Requires Python 3 stdlib, ffmpeg/ffprobe and a host with one video model loaded.
Evidence belongs below ignored artifacts/ or docs/validation/. Example:

  python3 eng/studio/validate_modes.py --url http://127.0.0.1:8187 \
      --width 512 --height 512 --frames 17 --output artifacts/studio-modes/ltx
"""
import argparse
import json
import pathlib

from validate import Host, run, uid, verify_media

PROMPTS = {
    "t2v": "A red toy car drives slowly along a quiet country road in warm afternoon light.",
    "i2v": "The picture slowly comes alive as the camera gently pushes in.",
    "fl2v": "The first scene smoothly transforms into the last scene.",
    "fl2v-last": "The camera drifts through a softly lit room and settles on the final picture.",
    "keyframes": "A colorful test pattern drifts and morphs across the frame.",
    "ref": "The test card from Picture 1 is printed on a large flag waving over a green hill.",
}


def still(output, name, source, width, height):
    path = output / (name + ".png")
    run("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"{source}=s={width}x{height}", "-frames:v", 1, path)
    return path


def thumbnail(path, frame=None):
    """An 8x8 RGB thumbnail of an image, or of one decoded video frame."""
    picked = "scale=8:8:flags=area" if frame is None else f"select=eq(n\\,{frame}),scale=8:8:flags=area"
    raw = run("ffmpeg", "-v", "error", "-i", path, "-vf", picked, "-frames:v", 1, "-pix_fmt", "rgb24", "-f", "rawvideo", "-", binary=True)
    assert len(raw) == 8 * 8 * 3, len(raw)
    return raw


def distance(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8187")
    parser.add_argument("--connection", type=pathlib.Path, help="TensorAgentHost connection.json (token is never reported)")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--modes", help="Comma-separated subset of the advertised modes (default: all)")
    parser.add_argument("--width", type=int, help="Output width (default: the model's Studio default)")
    parser.add_argument("--height", type=int, help="Output height (default: the model's Studio default)")
    parser.add_argument("--frames", type=int, help="Output frames (default: the model's; ref is raised to its minimum)")
    parser.add_argument("--steps", type=int, help="Sampling steps (default: the model's)")
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args()
    output = options.output.resolve()
    repository = pathlib.Path(__file__).resolve().parents[2]
    assert any(output.is_relative_to(repository / folder) for folder in ("artifacts", "docs/validation")), "Evidence must be in ignored artifacts/ or docs/validation/"
    output.mkdir(parents=True, exist_ok=True)
    connection = json.loads(options.connection.read_text()) if options.connection else {}
    host = Host(connection.get("baseUrl", options.url), connection.get("cookie"))

    capabilities = host.api("capabilities")
    model = next(m for m in capabilities["models"] if m["available"])
    modes, unavailable = model["modes"], model.get("unavailableModes") or {}
    print("model", model["modelId"], "family", model["family"], "modes", modes, "unavailable", sorted(unavailable), flush=True)
    selected = options.modes.split(",") if options.modes else modes
    # A last frame alone is fl2v with an empty first frame, so it runs wherever fl2v does.
    assert set(selected) <= set(modes) | ({"fl2v-last"} if "fl2v" in modes else set()), ("Requested modes the model does not advertise", selected, modes)

    width = options.width or model["defaultWidth"]
    height = options.height or model["defaultHeight"]
    multiple, offset = model["temporalMultiple"], model.get("temporalOffset", 0)
    frames = options.frames or model["defaultFrameCount"]
    assert (frames - offset) % multiple == 0, ("frames must be on the model grid", frames, multiple, offset)
    images = {"first": still(output, "first", "testsrc2", width, height), "last": still(output, "last", "smptebars", width, height),
              "guide": still(output, "guide", "rgbtestsrc", width, height), "reference": still(output, "reference", "testsrc", width, height)}
    project = host.api("projects/create", {"name": "Studio mode validation", "width": width, "height": height, "fpsNumerator": 24, "fpsDenominator": 1})
    project_id = project["id"]
    asset = {}
    for name, path in images.items():
        imported = host.upload(project_id, path)
        asset[name] = next(a for a in imported["assets"] if a["name"] == path.name)["id"]

    def sent(mode):
        return "fl2v" if mode == "fl2v-last" else mode

    def request(mode):
        count = frames
        if mode == "ref" and model.get("referenceMinFrames"):
            minimum = model["referenceMinFrames"]
            count = max(count, offset + -(-(minimum - offset) // multiple) * multiple)
        body = {"projectId": project_id, "requestId": uid(), "modelId": model["modelId"], "mode": sent(mode), "prompt": PROMPTS[mode],
                "width": width, "height": height, "frameCount": count, "fps": model.get("fixedFps") or model["defaultFps"],
                "steps": options.steps or model["defaultSteps"], "cfgScale": model["defaultCfgScale"], "seed": options.seed,
                "generateAudio": False, "shotId": "mode-" + mode, "temporalConditions": [], "referenceAssetIds": []}
        if mode in ("i2v", "fl2v"):
            body["conditioningAssetId"] = asset["first"]
        if mode in ("fl2v", "fl2v-last"):
            body["endConditioningAssetId"] = asset["last"]
        if mode == "keyframes":
            body["temporalConditions"] = [{"assetId": asset["guide"], "frameIndex": count // 2, "strength": 1}]
        if mode == "ref":
            body["referenceAssetIds"] = [asset["reference"]]
        return body

    report = {"model": {k: model.get(k) for k in ("modelId", "family", "modes", "unavailableModes", "referenceMinFrames", "maxReferences")},
              "size": [width, height], "frames": frames, "refused": {}, "runs": {}}

    # A mode the checkpoint cannot run is refused, with the reason the UI shows, before any job exists.
    for mode, reason in unavailable.items():
        code, body, _ = host.request("/api/studio", json.dumps({"action": "jobs/generate", "payload": request(mode)}).encode())
        error = json.loads(body)
        assert code == 400 and error["code"] == "unsupported_mode" and error["error"] == reason, (mode, code, error)
        report["refused"][mode] = error
        print("refused", mode, "->", reason, flush=True)
    jobs = host.api("jobs/list", {"projectId": project_id})
    assert not (jobs["jobs"] if isinstance(jobs, dict) else jobs), "A refused mode must not create a job"

    thumbs = {name: thumbnail(path) for name, path in images.items()}
    for mode in selected:
        body = request(mode)
        job = host.poll(project_id, host.api("jobs/generate", body)["id"], timeout=7200)
        generated = next(a for a in host.api("projects/get", {"projectId": project_id})["assets"] if a["id"] == job["outputAssetId"])
        effective = generated["generation"]["effectiveParameters"]
        assert effective["mode"] == sent(mode), (mode, effective)
        assert generated["generation"]["request"]["mode"] == sent(mode), generated["generation"]["request"]
        assert generated["generation"]["request"].get("conditioningAssetId") == body.get("conditioningAssetId"), "no first frame may be added"
        code, media, _ = host.request(f"/api/studio/projects/{project_id}/assets/{generated['id']}")
        assert code == 200, code
        video = output / f"{mode}.mp4"
        video.write_bytes(media)
        verify_media(video, body["frameCount"], width, height)
        first, last = thumbnail(video, 0), thumbnail(video, body["frameCount"] - 1)
        checks = {"firstFrameToFirstImage": distance(first, thumbs["first"]), "firstFrameToLastImage": distance(first, thumbs["last"]),
                  "lastFrameToLastImage": distance(last, thumbs["last"]), "lastFrameToFirstImage": distance(last, thumbs["first"])}
        if mode == "keyframes":
            middle = thumbnail(video, body["frameCount"] // 2)
            checks |= {"guideFrameToGuideImage": distance(middle, thumbs["guide"]), "guideFrameToLastImage": distance(middle, thumbs["last"])}
        if mode in ("i2v", "fl2v"):
            assert checks["firstFrameToFirstImage"] < checks["firstFrameToLastImage"], (mode, "first frame is not pinned", checks)
        if mode in ("fl2v", "fl2v-last"):
            assert checks["lastFrameToLastImage"] < checks["lastFrameToFirstImage"], (mode, "last frame is not pinned", checks)
        report["runs"][mode] = {"request": body, "effective": effective, "distances": checks, "output": video.name}
        print("PASS", mode, json.dumps(checks), flush=True)

    (output / "modes.json").write_text(json.dumps(report, indent=2))
    print("MODES PASSED", model["modelId"], ",".join(selected), flush=True)


if __name__ == "__main__":
    main()
