#!/usr/bin/env python3
"""Freeze/build bindings, run resident operator cases, and compare complete outputs.

No upstream compilation, model loading, GPU execution at prepare, or implicit
acceptance of skipped devices. Invoke from a configured compiler environment.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

SCHEMA = "tensorsharp.quant-projection-bench-controller.v1"
OBJECTS = ("ggml_ops_matmul_precision", "ggml_ops_dsv4_fused",
           "ggml_ops_attention_precision", "ggml_ops_q8_precision")
DRIVER_FILES = ("CMakeLists.txt", "quant_projection_bench.cpp", "bench.py", "README.md")
ENV_PREFIXES = ("TS_", "GGML_", "TENSORSHARP_", "CUDA_", "NCCL_", "OMP_", "GOMP_", "NVIDIA_")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Expected regular non-link input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_fresh(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def git_state(root):
    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    return {"revision": run("rev-parse", "HEAD"), "status": run("status", "--porcelain")}


def expected_inputs(native, upstream):
    """Only receipt-owned objects, their frozen sources, and pinned libraries."""
    native, upstream = Path(native).resolve(), Path(upstream).resolve()
    summary, plan = read(native / "build-summary.json"), read(native / "incremental-plan.json")
    require(summary.get("passed") is True and summary.get("no_upstream_compilation") is True,
            "Native build was not successfully qualified without upstream compilation")
    require(summary["before"] == summary["after"] == plan["before"],
            "Native plan/build input bookends disagree")
    require(summary["before"]["upstream"]["status"] == "", "Native build used modified upstream sources")
    expected = {}

    def add(path, sha256):
        key = str(Path(path).resolve())
        require(key not in expected or expected[key] == sha256, "Conflicting input digest")
        expected[key] = sha256

    for relative, value in summary["source_snapshot"].items():
        path = native / "source" / relative
        require(path.resolve().is_relative_to((native / "source").resolve()), "Frozen source escaped its root")
        add(path, value)
    for relative, value in summary["before"]["upstream"]["files"].items():
        path = upstream / relative
        require(path.resolve().is_relative_to(upstream), "Upstream file escaped its root")
        add(path, value)
    libraries = Path(plan["Libraries"]).resolve()
    for relative, value in summary["before"]["libraries"].items():
        path = libraries / relative
        require(path.resolve().is_relative_to(libraries), "Library escaped its root")
        add(path, value)
    objects = []
    for name in OBJECTS:
        matches = [(key, value) for key, value in plan["Modes"]["production"]["FinalObjectPaths"].items()
                   if Path(key.replace("\\", "/")).name in (name + ".cu.obj", name + ".cu.o")]
        require(len(matches) == 1, "Expected exactly one final owned object for " + name)
        key, value = matches[0]
        origin = summary["production_object_provenance"][key]
        path = Path(value).resolve()
        require(path == Path(origin["Path"]).resolve(), "Final object path disagrees with its build provenance")
        add(path, origin["Sha256"])
        objects.append(str(path))
    for filename in ("build-summary.json", "incremental-plan.json"):
        add(native / filename, pin(native / filename)["sha256"])
    return summary, plan, expected, objects


def verify(inputs, upstream, revision):
    actual = {path: pin(path) for path in sorted(inputs)}
    for path, expected in inputs.items():
        require(actual[path]["sha256"] == expected, "Input bytes changed: " + path)
    state = git_state(upstream)
    require(state == {"revision": revision, "status": ""}, "Upstream revision or cleanliness changed")
    return actual


def controller_identity(prepared, prepared_path):
    """Bind the Python actually executing, not just a nearby frozen copy."""
    actual = Path(__file__).resolve()
    frozen = Path(prepared_path).resolve().parent / "source/bench.py"
    expected = prepared["driver"]["bench.py"]
    actual_pin, frozen_pin = pin(actual), pin(frozen)
    require(actual_pin == expected, "Executing controller differs from the prepared frozen driver")
    require(frozen_pin == expected, "Prepared frozen controller changed")
    return {"path": str(actual), "pin": actual_pin, "frozen_path": str(frozen), "frozen_pin": frozen_pin}


def controller_observation():
    actual = Path(__file__).resolve()
    return {"path": str(actual), "pin": pin(actual)}


def prepare(args):
    output = Path(args.output).resolve()
    source = Path(__file__).resolve().parent
    require(not output.exists(), "Preparation output must be fresh")
    require(not output.is_relative_to(source), "Preparation output must be outside the benchmark source directory")
    summary, plan, inputs, objects = expected_inputs(args.native_build, args.ggml_root)
    revision = summary["before"]["upstream"]["revision"]
    before = verify(inputs, args.ggml_root, revision)
    driver = {name: pin(source / name) for name in DRIVER_FILES}
    output.mkdir(parents=True)
    snapshot = output / "source"
    snapshot.mkdir()
    for name in DRIVER_FILES:
        shutil.copyfile(source / name, snapshot / name)
        require(pin(snapshot / name) == driver[name] == pin(source / name), "Benchmark source changed while freezing")
        inputs[str((snapshot / name).resolve())] = driver[name]["sha256"]
    native = Path(args.native_build).resolve()
    configure = [args.cmake, "-S", str(snapshot), "-B", str(output / "build"), "-G", args.generator,
                 "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DTSG_PROJECTION_ENABLE_CUDA=ON",
                 "-DTSG_GGML_ROOT=" + str(Path(args.ggml_root).resolve()),
                 "-DTSG_GGML_LIBRARIES=" + str(Path(plan["Libraries"]).resolve()),
                 "-DTSG_PROJECTION_NATIVE_ROOT=" + str(native / "source/TensorSharp.GGML.Native"),
                 "-DTSG_PROJECTION_OWNED_CUDA_OBJECTS=" + ";".join(objects)]
    report = {"schema": SCHEMA, "prepared_utc": stamp(), "native_build": str(native),
              "upstream": str(Path(args.ggml_root).resolve()), "upstream_revision": revision,
              "inputs": inputs, "before": before, "driver": driver,
              "configure_argv": configure,
              "build_argv": [args.cmake, "--build", str(output / "build"), "--target", "QuantProjectionBench", "--parallel", "2"],
              "execution_performed": False, "passing_device_or_model_claim": False}
    report["after"] = verify(inputs, args.ggml_root, revision)
    require(all(report["after"][key] == value for key, value in before.items()), "Prepare input bookends changed")
    write_fresh(output / "prepared.json", report)
    print(json.dumps({"prepared": str(output / "prepared.json"), "configure_argv": configure,
                      "build_argv": report["build_argv"], "execution_performed": False}, indent=2))


def build_driver(args):
    prepared_path = Path(args.prepared).resolve()
    prepared = read(prepared_path)
    require(prepared.get("schema") == SCHEMA, "Unknown preparation schema")
    entry_controller = controller_identity(prepared, prepared_path)
    root = prepared_path.parent
    require(not (root / "build").exists() and not (root / "build-receipt.json").exists(), "Build directory/receipt must be fresh")
    before = verify(prepared["inputs"], prepared["upstream"], prepared["upstream_revision"])
    actual_controller = controller_identity(prepared, prepared_path)
    require(actual_controller == entry_controller, "Executing controller changed before build")
    report = {"schema": SCHEMA, "started_utc": stamp(), "prepared_path": str(prepared_path),
              "prepared_pin": pin(prepared_path), "before": before, "steps": [], "passed": False,
              "actual_controller_entry": entry_controller, "actual_controller_before": actual_controller,
              "gpu_or_model_execution": False, "no_upstream_compilation": True}
    try:
        environment = {k: v for k, v in os.environ.items() if not k.startswith(ENV_PREFIXES)}
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        for name, argv in (("configure", prepared["configure_argv"]), ("build", prepared["build_argv"])):
            with (root / (name + ".log")).open("x", encoding="utf-8") as log:
                child = subprocess.run(argv, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            report["steps"].append({"name": name, "argv": argv, "exit_code": child.returncode})
            require(child.returncode == 0, name + " failed")
        commands_path = root / "build/compile_commands.json"
        commands = read(commands_path)
        expected = {(root / "source/quant_projection_bench.cpp").resolve(),
                    (Path(prepared["native_build"]) / "source/TensorSharp.GGML.Native/ggml_ops_q8_precision.cpp").resolve(),
                    (Path(prepared["native_build"]) / "source/TensorSharp.GGML.Native/ggml_ops_matmul_precision.cpp").resolve()}
        actual = [Path(command["file"]).resolve() for command in commands]
        require(len(actual) == 3 and set(actual) == expected, "Build compiled unexpected or upstream translation units")
        executable = root / "build" / ("QuantProjectionBench.exe" if os.name == "nt" else "QuantProjectionBench")
        report.update(executable=str(executable), executable_pin=pin(executable),
                      compile_commands_pin=pin(commands_path), compiled_sources=sorted(map(str, actual)), passed=True)
    except subprocess.TimeoutExpired as error:
        # subprocess.run kills and waits for its direct child on timeout. It
        # does not promise containment of compiler descendants on all platforms.
        report.update(error=str(error), timeout_direct_child_reaped=True,
                      compiler_descendants_reaped=None,
                      timeout_cleanup_limitation="Compiler descendants are not contained or verified reaped; stop them before reusing resources")
    except Exception as error:
        report["error"] = str(error)
    finally:
        try:
            report["after"] = verify(prepared["inputs"], prepared["upstream"], prepared["upstream_revision"])
            report["actual_controller_after_observed"] = controller_observation()
            report["actual_controller_after"] = controller_identity(prepared, prepared_path)
            require(report["after"] == before and report["prepared_pin"] == pin(prepared_path)
                    and report["actual_controller_after"] == actual_controller, "Build input/controller bookends changed")
        except Exception as error:
            report.update(passed=False, input_bookend_error=str(error))
        report["finished_utc"] = stamp()
        write_fresh(root / "build-receipt.json", report)
    print(json.dumps({"passed": report["passed"], "receipt": str(root / "build-receipt.json"), "error": report.get("error")}, indent=2))
    return 0 if report["passed"] else 1


def validate_measurement(value, args):
    require(value.get("schema") == "tensorsharp.quant-projection-bench.v1" and value.get("execution_passed") is True,
            "Executable did not report completed execution")
    for key in ("k", "rows", "columns", "backend", "oracle", "device", "warmup", "repeats"):
        expected = getattr(args, key)
        require(type(value[key]) is type(expected) and value[key] == expected, "Executable geometry/mode/timing mismatch: " + key)
    require(type(value["iterations_per_repeat"]) is int and value["iterations_per_repeat"] == args.iterations,
            "Executable timing mismatch: iterations_per_repeat")
    require(value["allocation_upload_timed"] is False and value["source_bytes_unchanged"] is True,
            "Allocation/upload or source retention contract failed")
    require(value["gpu_event_ms"] is None and value["ordinary_is_numerical_oracle"] is False,
            "Unsupported GPU-event or ordinary numerical-oracle claim")
    complete_values = args.rows * args.columns
    expected = (complete_values if args.oracle == "full" else
                min(args.rows, args.sample_rows) * min(args.columns, args.sample_columns) if args.oracle == "sample" else 0)
    require(type(value["oracle_values"]) is int and value["oracle_values"] == expected, "Incomplete or excessive oracle value count")
    # The frozen native v1 driver conservatively uses this field to identify
    # explicit full mode, even when a small sampled geometry covers all outputs.
    # Retain its raw declaration and separately report completeness derived from
    # the verified exact row/column sample count.
    require(value["oracle_complete_outputs"] is (args.oracle == "full"), "Native oracle-mode completeness declaration mismatch")
    if args.oracle == "none":
        require(value["owned_oracle_passed"] is None, "Absent oracle was counted as passed")
    else:
        require(value["oracle_values"] > 0 and value["owned_oracle_passed"] is True,
                "Owned stored-weight FP64 oracle failed or did not run")
    require(set(value["variants"]) == {"owned", "ordinary_performance_reference"}, "Missing or unexpected timing variants")
    for variant in value["variants"].values():
        samples = variant["wall_samples_ms"]
        require(isinstance(samples, list) and len(samples) == args.repeats
                and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples),
                "Missing/nonfinite/nonpositive timing samples")
    return {"expected_oracle_values": expected,
            "oracle_verified_complete_outputs": args.oracle != "none" and expected == complete_values,
            "native_completeness_field_means_explicit_full_mode": True}


def run_case(args):
    prepared = read(args.prepared)
    require(prepared.get("schema") == SCHEMA, "Unknown preparation schema")
    entry_controller = controller_identity(prepared, args.prepared)
    build_path = Path(args.prepared).resolve().parent / "build-receipt.json"
    build = read(build_path)
    require(build.get("passed") is True and build["prepared_pin"] == pin(args.prepared), "Missing/mismatched qualified benchmark build")
    output, executable = Path(args.output).resolve(), Path(args.executable).resolve()
    require(not output.exists(), "Execution output must be fresh")
    require(executable == Path(build["executable"]).resolve() and pin(executable) == build["executable_pin"],
            "Executable does not match the receipt-bound benchmark build")
    before = verify(prepared["inputs"], prepared["upstream"], prepared["upstream_revision"])
    actual_controller = controller_identity(prepared, args.prepared)
    require(actual_controller == entry_controller, "Executing controller changed before execution")
    executable_pin = pin(executable)
    output.mkdir(parents=True)
    argv = [str(executable)]
    for name in ("k", "rows", "columns", "backend", "device", "threads", "oracle", "warmup", "repeats", "iterations",
                 "sample_rows", "sample_columns", "oracle_max_operations", "max_payload_bytes"):
        argv.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    argv.extend(["--output", str(output / "case")])
    environment = {k: v for k, v in os.environ.items() if not k.startswith(ENV_PREFIXES)}
    environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices if args.backend == "cuda" else "-1"
    environment["NVIDIA_TF32_OVERRIDE"] = "0"
    report = {"schema": SCHEMA, "started_utc": stamp(), "prepared_path": str(Path(args.prepared).resolve()),
              "prepared_pin": pin(args.prepared), "executable": str(executable), "executable_pin": executable_pin,
              "build_receipt_pin": pin(build_path),
              "actual_controller_entry": entry_controller, "actual_controller_before": actual_controller,
              "before": before, "argv": argv, "environment": {k: v for k, v in environment.items() if k.startswith(ENV_PREFIXES)},
              "passed": False, "device_unavailable": False, "model_quality_performance_hardcap_claim": False}
    try:
        with (output / "stdout.log").open("x", encoding="utf-8") as stdout, (output / "stderr.log").open("x", encoding="utf-8") as stderr:
            child = subprocess.run(argv, env=environment, stdout=stdout, stderr=stderr, timeout=args.timeout)
        report["exit_code"] = child.returncode
        report["child_reaped"] = True
        report["device_unavailable"] = child.returncode == 77
        require(child.returncode == 0, f"Benchmark failed/unavailable, exit={child.returncode}; unavailable is not a pass")
        measurement = read(output / "case/measurement.json")
        coverage = validate_measurement(measurement, args)
        binaries = {}
        for name, expected in (("owned.f32le.bin", args.rows * args.columns * 4),
                               ("ordinary.f32le.bin", args.rows * args.columns * 4),
                               ("weights.q5_k.bin", args.k // 256 * 176 * args.rows),
                               ("input.f32le.bin", args.k * args.columns * 4)):
            value = pin(output / "case" / name)
            require(value["bytes"] == expected, "Incomplete binary: " + name)
            binaries[name] = value
        report.update(measurement=measurement, verified_oracle_coverage=coverage, binaries=binaries, passed=True)
    except subprocess.TimeoutExpired as error:
        report.update(error=str(error), timeout_direct_child_reaped=True,
                      descendant_cleanup_verified=False,
                      timeout_cleanup_limitation="Only the direct benchmark child is verified reaped by subprocess.run; descendants are not contained")
    except Exception as error:
        report["error"] = str(error)
    finally:
        try:
            report["after"] = verify(prepared["inputs"], prepared["upstream"], prepared["upstream_revision"])
            report["actual_controller_after_observed"] = controller_observation()
            report["actual_controller_after"] = controller_identity(prepared, args.prepared)
            require(report["before"] == report["after"] and executable_pin == pin(executable)
                    and report["prepared_pin"] == pin(args.prepared)
                    and report["build_receipt_pin"] == pin(build_path)
                    and report["actual_controller_after"] == actual_controller, "Execution input/controller bookends changed")
        except Exception as error:
            report.update(passed=False, input_bookend_error=str(error))
        report["finished_utc"] = stamp()
        write_fresh(output / "receipt.json", report)
    print(json.dumps({"passed": report["passed"], "device_unavailable": report["device_unavailable"],
                      "receipt": str(output / "receipt.json"), "error": report.get("error")}, indent=2))
    return 0 if report["passed"] else 1


def compare_cases(left, right):
    left, right = Path(left).resolve(), Path(right).resolve()
    a, b = read(left / "receipt.json"), read(right / "receipt.json")
    require(a.get("passed") is True and b.get("passed") is True, "Both execution receipts must have passed")
    for root, receipt in ((left, a), (right, b)):
        require(receipt.get("schema") == SCHEMA
                and receipt.get("measurement", {}).get("schema") == "tensorsharp.quant-projection-bench.v1",
                "Unknown comparison receipt/measurement schema")
        require(set(receipt["binaries"]) == {"owned.f32le.bin", "ordinary.f32le.bin", "weights.q5_k.bin", "input.f32le.bin"},
                "Missing or unexpected comparison binary digests")
        for name, expected in receipt["binaries"].items():
            require(pin(root / "case" / name) == expected, "Retained binary changed: " + name)
    for key in ("k", "rows", "columns", "backend", "device"):
        require(a["measurement"][key] == b["measurement"][key], "Compared geometry/device mismatch: " + key)
    for name in ("weights.q5_k.bin", "input.f32le.bin"):
        require(a["binaries"][name] == b["binaries"][name], "Compared sources differ: " + name)
    changed = total = 0
    maximum = squared = reference_squared = 0.0
    with (left / "case/owned.f32le.bin").open("rb") as aa, (right / "case/owned.f32le.bin").open("rb") as bb:
        while True:
            x, y = aa.read(1024 * 1024), bb.read(1024 * 1024)
            require(len(x) == len(y) and len(x) % 4 == 0, "Output size mismatch")
            if not x:
                break
            for i, (u, v) in enumerate(zip(struct.iter_unpack("<f", x), struct.iter_unpack("<f", y))):
                uf, vf = u[0], v[0]
                require(math.isfinite(uf) and math.isfinite(vf), "Nonfinite compared output")
                delta = uf - vf
                maximum = max(maximum, abs(delta)); squared += delta * delta; reference_squared += uf * uf
                changed += x[i * 4:i * 4 + 4] != y[i * 4:i * 4 + 4]
                total += 1
    require(total == a["measurement"]["rows"] * a["measurement"]["columns"], "Incomplete output comparison")
    return {"schema": SCHEMA, "left": str(left), "right": str(right), "complete_values": total,
            "changed_bits_values": changed, "bit_identical": changed == 0,
            "max_abs": maximum, "rmse": math.sqrt(squared / total),
            "relative_l2": math.sqrt(squared / reference_squared) if reference_squared else math.sqrt(squared),
            "model_quality_performance_hardcap_claim": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Freeze the benchmark and bind exact qualified build inputs; does not build/run")
    p.add_argument("--native-build", required=True); p.add_argument("--ggml-root", required=True)
    p.add_argument("--output", required=True); p.add_argument("--cmake", default="cmake"); p.add_argument("--generator", default="Ninja")
    b = sub.add_parser("build", help="Compile only the three owned driver/adapter units against frozen objects; no GPU execution")
    b.add_argument("--prepared", required=True); b.add_argument("--timeout", type=int, default=600)
    r = sub.add_parser("run", help="Verify bookends and execute one explicitly selected device case")
    r.add_argument("--prepared", required=True); r.add_argument("--executable", required=True); r.add_argument("--output", required=True)
    for name in ("k", "rows", "columns"):
        r.add_argument("--" + name, type=int, required=True)
    for name, value in (("device", 0), ("threads", 4), ("warmup", 5), ("repeats", 10), ("iterations", 10),
                        ("sample-rows", 16), ("sample-columns", 16), ("oracle-max-operations", 300000000),
                        ("max-payload-bytes", 1073741824), ("timeout", 600)):
        r.add_argument("--" + name, type=int, default=value)
    r.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    r.add_argument("--oracle", choices=("full", "sample", "none"), default="sample")
    r.add_argument("--cuda-visible-devices", default="0")
    c = sub.add_parser("compare", help="Compare every retained owned F32 output after validating input digests")
    c.add_argument("--left", required=True); c.add_argument("--right", required=True); c.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args); return 0
        if args.command == "build":
            return build_driver(args)
        if args.command == "run":
            return run_case(args)
        report = compare_cases(args.left, args.right)
        write_fresh(args.output, report); print(json.dumps(report, indent=2))
        return 0 if report["bit_identical"] else 2
    except Exception as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
