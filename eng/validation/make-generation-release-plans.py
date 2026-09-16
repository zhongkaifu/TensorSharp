#!/usr/bin/env python3
"""Create named generation API workloads from the release media fixtures."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    uploads = {"first": (args.fixtures / "card-0.png").as_posix(), "last": (args.fixtures / "card-1.png").as_posix(),
               "clip": (args.fixtures / "ordered-cards.mp4").as_posix(), "speech": (args.fixtures / "spoken-fox.wav").as_posix()}
    video = {"width": 640, "height": 384, "frames": 22, "fps": 24, "steps": 20, "cfg": 1.0, "seed": 42}

    def case(name, prompt, overrides, audio=True):
        request = {**video, "prompt": prompt, **overrides}
        expected = {"width": request["width"], "height": request["height"], "frames": request["frames"],
                    "fps": request["fps"], "audio": audio, "codec": "h264"}
        if audio:
            expected.update(audio_sample_rate=32000, audio_channels=2,
                            audio_samples_per_channel=round(request["frames"] / request["fps"] * 32000))
        return {"id": name, "endpoint": "/api/video-generate", "request": request, "expected": expected}

    plans = {
        "h3-fl2va": {"uploads": {key: uploads[key] for key in ("first", "last")}, "cases": [
            case("text-to-video", "A red fox trots slowly through falling snow, cinematic natural lighting.", {"videoMode": "t2v"}),
            case("image-to-video", "A slow gentle camera pan across the printed verification card. Preserve its text and colors.",
                 {"videoMode": "i2v", "imagePath": "@first"}),
            case("first-last-frame", "A smooth transition between two printed verification cards on a white background.",
                 {"videoMode": "fl2v", "imagePath": "@first", "endImage": "@last"}),
        ]},
        "h3-ref2va": {"uploads": uploads, "cases": [
            case("reference-image", "The verification card in <Picture 1> stands on a desk as the camera gently moves closer.",
                 {"videoMode": "ref", "referenceImages": ["@first"]}),
            case("reference-video", "A sequence of printed cards following the colors and appearance in <Video 1>.",
                 {"videoMode": "ref", "referenceVideos": ["@clip"]}),
            case("reference-audio", "A calm presenter reads a verification code in a studio, matching the voice in <Audio 1>.",
                 {"videoMode": "ref", "referenceAudios": ["@speech"]}),
        ]},
        "qwen-image": {"uploads": {key: uploads[key] for key in ("first", "last")}, "cases": [
            {"id": "single-image-edit", "endpoint": "/api/image-edit", "request": {
                "imagePath": "@first", "prompt": "Change the red rectangle to bright blue. Keep the four-digit code and white background unchanged.",
                "targetArea": 307200, "steps": 0, "cfg": 0, "seed": 42},
             "expected": {"width": 640, "height": 480}},
            {"id": "multi-image-edit", "endpoint": "/api/image-edit", "request": {
                "imagePaths": ["@first", "@last"], "prompt": "Combine the two verification cards side by side. Preserve both codes and rectangle colors on a white background.",
                "targetArea": 307200, "steps": 0, "cfg": 0, "seed": 42},
             "expected": {"width": 640, "height": 480}},
        ]},
    }
    for name, modes in {"wan-t2v": ("t2v",), "wan-ti2v": ("t2v", "i2v"), "wan-i2v": ("i2v",)}.items():
        cases = []
        for mode in modes:
            # Stay at the documented ~0.3 MP quality floor for Wan; 640x384
            # is useful for a kernel smoke but below that generation target.
            overrides = {"videoMode": mode, "width": 640, "height": 480, "frames": 33, "steps": 0, "cfg": 0}
            if mode == "i2v":
                overrides["imagePath"] = "@first"
            cases.append(case(mode, "A gentle camera movement across the colorful verification card." if mode == "i2v"
                else "A red fox trots slowly through falling snow, cinematic natural lighting.", overrides, audio=False))
        plans[name] = {"uploads": {"first": uploads["first"]} if "i2v" in modes else {}, "cases": cases}
    plans["wan21-t2v"] = json.loads(json.dumps(plans["wan-t2v"]))
    for item in plans["wan21-t2v"]["cases"]:
        item["request"]["fps"] = item["expected"]["fps"] = 16
    for plan in plans.values():
        first = plan["cases"][0]
        streaming = json.loads(json.dumps(first))
        streaming.update(id=first["id"] + "-stream", endpoint=first["endpoint"] + "/stream",
                         warmup=False, concurrency=[1], repeats=1)
        plan["cases"].append(streaming)
        if first["endpoint"] == "/api/video-generate":
            envelope = json.loads(json.dumps(first))
            envelope.update(id=first["id"] + "-openai-base64", endpoint="/v1/videos/generations",
                            warmup=False, concurrency=[1], repeats=1)
            envelope["request"]["response_format"] = "b64_json"
            plan["cases"].append(envelope)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, plan in plans.items():
        plan["quality_note"] = "Structural output checks and repeated latency measurements require separate contact-sheet/audio review before making a visual or audio quality claim. Steps=0 selects the checkpoint's documented base or distilled recipe."
        path = args.output / (name + ".json")
        path.write_text(json.dumps(plan, indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
