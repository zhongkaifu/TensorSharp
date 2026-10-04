#!/usr/bin/env python3
"""Compare isolated and parallel Qwen3.8 image requests on an existing server.

Reuse the shared media harness's OCR/color cards, attachment ordering and image
history checks. Keep each rendered request identical between C=1 and parallel
waves, synchronize parallel starts, and retain exact per-turn comparisons. The
synthetic checks establish narrow image-content correctness, not broad visual
quality. End-to-end throughput includes image processing, prefill and queueing.
Start the server with the exact model and its matching --mmproj first.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time


HARNESS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "engine_comparison"
SCENARIOS = {"image_ocr", "image_long_context", "multi_image", "image_follow_up", "video_order", "video_timestamp"}


def load_media_harness():
    sys.path.insert(0, str(HARNESS_DIR))
    spec = importlib.util.spec_from_file_location("shared_media_validation", HARNESS_DIR / "validate_deepseek41_media.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_images(directory):
    """Create the shared harness's two cards without requiring a video tool."""
    from PIL import Image, ImageDraw, ImageFont
    directory.mkdir(parents=True, exist_ok=True)
    font_path = next((Path(path) for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf") if Path(path).exists()), None)
    font = ImageFont.truetype(str(font_path), 90) if font_path else ImageFont.load_default(size=90)
    cards = (("4821", "red"), ("9364", "blue"))
    files = {}
    for index, (code, color) in enumerate(cards):
        image = Image.new("RGB", (640, 480), "white")
        draw = ImageDraw.Draw(image)
        draw.text((320, 125), code, font=font, fill="black", anchor="mm")
        draw.rectangle((100, 245, 540, 405), fill=color)
        path = directory / f"card-{index}.png"
        image.save(path)
        files[path.name] = sha256(path)
    (directory / "manifest.json").write_text(json.dumps({"sha256": files, "cards": cards}, indent=2) + "\n")


def compare_sequential_baseline(cases, baseline):
    """A failed/incomplete/mismatched control never establishes isolation."""
    failures = []
    if baseline.get("run_complete") is not True or not baseline.get("cases"):
        failures.append("baseline must be a complete, nonempty run")
    planned_controls = baseline.get("execution_plan", {}).get("expected_cases")
    if planned_controls is not None and planned_controls != len(baseline.get("cases", [])):
        failures.append("baseline case count must match its execution plan")

    def complete_turns(case):
        scenario = case.get("scenario")
        required = 2 if scenario == "image_follow_up" else 1
        return scenario in SCENARIOS and len(case.get("turns", [])) == required

    controls = {}
    for case in baseline.get("cases", []):
        key = (case.get("scenario"), case.get("tag"))
        if case.get("concurrency") != 1 or key in controls or case.get("status") != "ok":
            failures.append("baseline must contain unique, passing C=1 requests")
        if not complete_turns(case):
            failures.append("baseline must contain every required scenario turn")
        controls[key] = case
    seen_candidates = set()
    candidate_groups = {}
    for case in cases:
        degree = case.get("concurrency")
        key = (case.get("scenario"), case.get("tag"))
        identity = (*key, degree)
        if identity in seen_candidates:
            failures.append("candidate requests must be unique within each concurrency")
        seen_candidates.add(identity)
        if not isinstance(degree, int) or isinstance(degree, bool) or degree < 1:
            failures.append("candidate concurrency must be a positive integer")
        candidate_groups.setdefault(degree, set()).add(key)
        if not complete_turns(case):
            failures.append("candidate must contain every required scenario turn")
    for keys in candidate_groups.values():
        if keys != set(controls):
            failures.append("each candidate concurrency must cover the same image requests as the baseline")
    comparisons = []
    for case in cases:
        key = (case.get("scenario"), case.get("tag"))
        control = controls.get(key)
        passed = bool(control and case.get("status") == "ok" and complete_turns(case) and complete_turns(control)
                      and len(control.get("turns", [])) == len(case["turns"]))
        if passed:
            for actual, expected in zip(case["turns"], control["turns"]):
                left, right = actual["metrics"], expected["metrics"]
                passed = passed and (actual.get("request_sha256") == expected.get("request_sha256")
                    and actual.get("expected") == expected.get("expected")
                    and left.get("assistant_message", {}).get("content") == right.get("assistant_message", {}).get("content")
                    and left.get("finish_reason") == right.get("finish_reason") == "stop")
        comparisons.append({"scenario": case.get("scenario"), "tag": case.get("tag"),
                            "concurrency": case.get("concurrency"), "passed": bool(passed)})
    if not all(item["passed"] for item in comparisons) or not comparisons:
        failures.append("one or more exact sequential/parallel comparisons failed")
    return {"passed": not failures, "comparisons": comparisons, "failures": failures}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:5001")
    parser.add_argument("--model")
    parser.add_argument("--weights-id")
    parser.add_argument("--companion-sha256")
    parser.add_argument("--profile", default="ggml_metal")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--concurrency", default="1,2")
    parser.add_argument("--requests-per-wave", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--scenarios", default="image_ocr,multi_image,image_follow_up")
    parser.add_argument("--blocking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    args = parser.parse_args()
    if args.prepare:
        if any(name.startswith("video_") for name in args.scenarios.split(",")):
            load_media_harness().prepare(args.fixtures)
        else:
            prepare_images(args.fixtures)
        return 0
    if not all((args.model, args.weights_id, args.companion_sha256, args.output)):
        parser.error("validation requires model, weights-id, companion-sha256 and output")
    degrees = [int(value) for value in args.concurrency.split(",")]
    scenarios = args.scenarios.split(",")
    if (not degrees or any(value < 1 for value in degrees) or len(degrees) != len(set(degrees))
            or args.requests_per_wave < max(degrees) or args.repeats < 1 or args.max_tokens < 1):
        parser.error("use unique positive concurrency, positive repeats/max-tokens, and requests-per-wave >= maximum concurrency")
    if not scenarios or len(scenarios) != len(set(scenarios)) or set(scenarios) - SCENARIOS:
        parser.error("use unique known media scenarios")
    media = load_media_harness()
    fixture_manifest = json.loads((args.fixtures / "manifest.json").read_text())
    for name, expected in fixture_manifest["sha256"].items():
        if sha256(args.fixtures / name) != expected:
            parser.error("fixture SHA256 mismatch: " + name)
    report = {"weights_id": args.weights_id, "companion_sha256": args.companion_sha256,
              "model": args.model, "profile": args.profile, "stream": not args.blocking,
              "sampling": media.SAMPLING, "fixtures": fixture_manifest,
              "harness_sha256": {Path(path).name: sha256(Path(path)) for path in
                                 (__file__, media.__file__, media.engines.__file__,
                                  HARNESS_DIR / "validate_inference.py")},
              "execution_plan": {"scenarios": scenarios, "concurrency": degrees,
                                 "requests_per_wave": args.requests_per_wave, "repeats": args.repeats,
                                 "expected_cases": len(scenarios) * len(degrees) * args.requests_per_wave * args.repeats},
              "run_complete": False, "cases": [], "waves": [],
              "scope": "Synthetic OCR/color, image order and retained image history; end-to-end request throughput includes media processing/prefill/queueing. No broad visual-quality or audio qualification."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for scenario in scenarios:
        for degree in degrees:
            for repetition in range(args.repeats):
                gate = threading.Barrier(degree)
                def run(index):
                    if index < degree:
                        gate.wait(timeout=60)
                    # This leading marker distinguishes repetitions but keeps
                    # the prompt identical between sequential/parallel arms.
                    return media.run_case(args, scenario, f"{scenario}-r{repetition}-i{index}", index)
                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=degree) as pool:
                    cases = list(pool.map(run, range(args.requests_per_wave)))
                elapsed = time.monotonic() - started
                for case in cases:
                    case["concurrency"] = degree
                report["cases"].extend(cases)
                passed = sum(case["status"] == "ok" for case in cases)
                tokens = sum(turn["metrics"].get("completion_tokens", 0)
                             for case in cases for turn in case["turns"])
                report["waves"].append({"scenario": scenario, "concurrency": degree, "repeat": repetition,
                                        "requests": len(cases), "passed": passed, "wall_seconds": elapsed,
                                        "completion_tokens": tokens, "end_to_end_tokens_per_second": tokens / elapsed})
                args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
                print(f"{scenario} C={degree} repeat={repetition}: {passed}/{len(cases)} passed", flush=True)
    report["run_complete"] = True
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        report["sequential_comparison"] = compare_sequential_baseline(report["cases"], baseline)
        if (baseline.get("weights_id"), baseline.get("companion_sha256"), baseline.get("fixtures")) != (
                report["weights_id"], report["companion_sha256"], report["fixtures"]):
            report["sequential_comparison"]["passed"] = False
            report["sequential_comparison"]["failures"].append("model/projector/fixture identity differs")
    elif 1 in degrees and any(degree > 1 for degree in degrees):
        report["sequential_comparison"] = compare_sequential_baseline(report["cases"],
            {"run_complete": True, "cases": [case for case in report["cases"] if case["concurrency"] == 1]})
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return int(any(case["status"] != "ok" for case in report["cases"])
               or report.get("sequential_comparison", {}).get("passed") is False)


if __name__ == "__main__":
    raise SystemExit(main())
