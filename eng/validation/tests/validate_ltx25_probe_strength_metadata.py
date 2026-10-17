#!/usr/bin/env python3
"""Check probe strength planning without loading a model or initializing a device.

The real probe resolves request/production policy and records metadata, then a
deliberately invalid GGUF stops the run at model construction. Outputs are kept
in the caller's generated evidence directory; this is not an inference test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=Path(__file__).resolve().parents[1] /
                        "Ltx25LatentProbe/bin/Debug/net10.0/Ltx25LatentProbe.dll")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.probe.is_file():
        parser.error("Build the probe first or supply --probe.")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Refusing to replace a nonempty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    model = output / "invalid-model.gguf"
    model.write_bytes(b"INVALID-GGUF-FOR-METADATA-ONLY\n")
    image = output / "source-image.bin"
    image.write_bytes(b"source bytes are hashed but never decoded\n")
    source_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    cases = [
        ("omitted", [], .7, 1., False),
        ("explicit-one", ["--image-strength", "1"], 1., 1., True),
        ("explicit-partial", ["--image-strength", ".85"], .85, .85, True),
        ("override-explicit-one", ["--image-strength", "1", "--first-stage-image-strength", ".7"], .7, 1., True),
        ("override-explicit-partial", ["--image-strength", ".85", "--first-stage-image-strength", ".7"], .7, .85, True),
        ("override-shorthand", ["--first-stage-image-strength", ".9"], .9, 1., False),
        ("end-image-preserves-one", ["--end-image", str(image)], 1., 1., False),
    ]
    results = []
    for name, flags, first, refinement, explicit in cases:
        directory = output / name
        command = ["dotnet", str(args.probe.resolve()), "--operation", "generate", "--backend", "cpu", "--decoders", "none",
                   "--model", str(model), "--prompt", "metadata fixture", "--image", str(image), "--seed", "3206401772581117598",
                   "--width", "1024", "--height", "1024", "--frames", "145", "--output", str(directory), *flags]
        process = subprocess.run(command, text=True, capture_output=True)
        (output / (name + ".log")).write_text(process.stdout + process.stderr)
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        stage = report.get("first_image_stage_strengths", {})
        references = report.get("reference_inputs", [])
        passed = (process.returncode != 0 and report.get("completed") is False and
                  "model_factory_ms" not in report and "forward_capture" not in report and
                  abs(stage.get("first_pass", -1) - first) < 1e-6 and
                  abs(stage.get("refinement", -1) - refinement) < 1e-6 and
                  stage.get("image_strength_explicit") is explicit and
                  references and references[0].get("path") == str(image) and references[0].get("sha256") == source_hash)
        results.append(dict(case=name, passed=bool(passed), returncode=process.returncode,
                            expected=dict(first_pass=first, refinement=refinement, image_strength_explicit=explicit),
                            actual=stage))
    summary = dict(scope="CPU metadata/CLI planning only. Invalid GGUF prevents model/device initialization; no inference coverage.",
                   passed=all(case["passed"] for case in results), cases=results)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
