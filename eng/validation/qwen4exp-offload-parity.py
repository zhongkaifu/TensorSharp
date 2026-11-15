#!/usr/bin/env python3
"""Compare complete Qwen4Exp logits in fresh host-graph/optimized/pinned processes.

Build InferenceWeb.Tests first. Example (from the repository root):
  python eng/validation/qwen4exp-offload-parity.py \
    --output docs/validation/qwen4exp-offload-parity

CUDA is required. Missing devices, skipped tests, missing captures, nonfinite values,
different argmaxes, or relative L2 above 1e-6 fail validation. No NumPy is needed.
Evidence includes the test-output native DLL candidate hash and unchanged ggml
revision/status. It does not inspect the test process's loaded module path.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


SCENARIOS = (
    "prefill40-device", "prefill40-host4", "prefill40-host8",
    "prefill160-device", "prefill160-host8",
)


def relative_error(actual, expected):
    if len(actual) != len(expected) or not actual:
        raise ValueError("Missing or different logit dimensions")
    if not all(math.isfinite(v) for v in actual + expected):
        raise ValueError("Nonfinite logits")
    numerator = sum((a - b) ** 2 for a, b in zip(actual, expected))
    denominator = sum(b * b for b in expected)
    return math.sqrt(numerator / max(denominator, 1e-30))


def compare_rows(actual, expected, bound):
    if len(actual) != len(expected) or len(actual) != 7:
        raise ValueError("Expected prefill and six decode rows")
    metrics = []
    for index, (a, b) in enumerate(zip(actual, expected)):
        error = relative_error(a, b)
        argmax_a = max(range(len(a)), key=a.__getitem__)
        argmax_b = max(range(len(b)), key=b.__getitem__)
        metrics.append({"row": index, "relative_l2": error,
                        "max_absolute": max(abs(x - y) for x, y in zip(a, b)),
                        "argmax": argmax_a, "reference_argmax": argmax_b})
        if error > bound or argmax_a != argmax_b:
            raise ValueError(f"Row {index}: relative L2 {error:.9g} exceeds {bound}, "
                             f"or argmax differs ({argmax_a}, {argmax_b})")
    return metrics


def read_capture(directory, scenario):
    return json.loads((directory / f"GgmlCuda-{scenario}.json").read_text(encoding="utf-8"))


def run_case(root, output, configuration, name, decode, pin_mb, reference):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    captures = directory / "logits"
    env = os.environ.copy()
    env.update(TS_TEST_GGML_BACKEND="cuda", TS_HOST_MOE_DECODE=str(decode), TS_HOST_MOE_PIN="1",
               TS_HOST_MOE_PIN_MAX_MB=str(pin_mb), TS_TEST_OFFLOAD_CAPTURE_DIR=str(captures))
    env.pop("TS_TEST_OFFLOAD_REFERENCE_DIR", None)
    if reference is not None:
        env["TS_TEST_OFFLOAD_REFERENCE_DIR"] = str(reference)
    command = ["dotnet", "test", "InferenceWeb.Tests/InferenceWeb.Tests.csproj",
               "-c", configuration, "--no-build", "--no-restore", "--filter",
               "FullyQualifiedName~Qwen4ExpExpertOffloadTests.OffloadedExperts_AnswerAsTheAccelerator_Cuda"
               "|FullyQualifiedName~Qwen4ExpExpertOffloadTests.StreamedPrefill_AnswersAsTheAccelerator_Cuda",
               "--logger", "trx;LogFileName=tests.trx", "--results-directory", str(directory)]
    started = time.monotonic()
    with (directory / "tests.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    results = ET.parse(directory / "tests.trx").findall(".//{*}UnitTestResult")
    outcomes = [entry.attrib["outcome"] for entry in results]
    evidence = {"exit_code": result.returncode, "seconds": time.monotonic() - started,
                "outcomes": outcomes, "decode_kernel": decode, "pin_budget_mb": pin_mb}
    if result.returncode or outcomes != ["Passed", "Passed"]:
        raise ValueError(f"{name}: expected two executed passes, got {outcomes}; "
                         f"exit={result.returncode}; see {directory / 'tests.log'}")
    evidence["cpu_vs_cuda"] = {}
    evidence["graph_parity"] = {}
    for scenario in SCENARIOS:
        rows = read_capture(captures, scenario)
        if reference is not None:
            evidence["graph_parity"][scenario] = compare_rows(
                rows, read_capture(reference, scenario), 1e-6)
        if "host" in scenario:
            device_scenario = scenario.rsplit("-", 1)[0] + "-device"
            evidence["cpu_vs_cuda"][scenario] = compare_rows(
                rows, read_capture(captures, device_scenario),
                2.5e-2 if scenario.startswith("prefill40-") else 2e-2)
    return evidence, captures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configuration", default="Release", choices=("Release", "Debug"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    native_name = "GgmlOps.dll" if os.name == "nt" else "libGgmlOps.so"
    native = root / "InferenceWeb.Tests/bin" / args.configuration / "net10.0" / native_name
    report = {"native_candidate_path": str(native),
              "native_candidate_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
              "ggml_revision": subprocess.check_output(
                  ["git", "-C", str(root / "ExternalProjects/ggml"), "rev-parse", "HEAD"],
                  text=True).strip(),
              "ggml_status": subprocess.check_output(
                  ["git", "-C", str(root / "ExternalProjects/ggml"), "status", "--porcelain"],
                  text=True).strip(),
              "strict_relative_l2_bound": 1e-6,
              "cpu_vs_cuda_relative_l2_bounds": {"prefill40": 2.5e-2, "prefill160": 2e-2},
              "cases": {}, "passed": False}
    try:
        if report["ggml_status"]:
            raise ValueError("Validation requires an unchanged upstream ggml checkout")
        baseline, reference = run_case(root, output, args.configuration, "host-graph", 0, 0, None)
        report["cases"]["host-graph"] = baseline
        for name, budget in (("optimized", 0), ("optimized-pinned256", 256)):
            print(f"Comparing {name} with the host graph", flush=True)
            case, _ = run_case(root, output, args.configuration, name, 1, budget, reference)
            report["cases"][name] = case
        report["passed"] = True
    except (OSError, ValueError, ET.ParseError, subprocess.CalledProcessError) as error:
        report["error"] = str(error)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(output / "report.json"),
                      "error": report.get("error")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
