#!/usr/bin/env python3
"""Independent natural-launchability campaign. Offline build never initializes CUDA."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = HERE.with_name("Qwen35AttentionBench")
SOURCE = ROOT / "TensorSharp.Backends.Cuda/native/kernels/tensorsharp_q35_kernels.cu"
PINS = {
    "build.py": "5253bc9221e013cddd343f8b3334e2a48ceb25a54bcf21d246f8dc0873789d94",
    "bench.cu": "f2127801d08f0487ddea261478838d4715e13035d44ac02c5974d2b153578fba",
    "attention64.cuh": "e88c6e4195f3d39c7f4766f2d2110996b0cc6220b62cf8cdac0247e103e4caa4",
    "checks.h": "5c1e5e1fce09ebbfaa4e0bc6164b5baa1803b8a3b4005e4925b48dec78179c2a",
}
KERNEL_PINS = {
    0: "ce7d38cc3b41372dc796a8a5a6f6db44a1db5db0e0f54c04e6b672b21964334b",
    2: "1fefa1244d751423572d65f336b5132350125f5b3a8b4dc7aee430ba75540678",
    3: "39ac2004cae8c3c179462615f8cf6fa6932dc8194f23c6ea30a34e430266b8a9",
    4: "0294628e194a5386cc186dbb52c1448a343d7b69efae5591cc8159b6a6c2df79",
}
ARMS = {
    "v2-original": (2, False), "v3-original": (3, False), "v4-original": (4, False),
    "v3-ptxas-o1": (3, True), "v4-ptxas-o1": (4, True),
}
BASE_FLAGS = ["-O3", "--extra-device-vectorization", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"]
ENV_NAMES = ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "CUDA_HOME", "CUDA_PATH")
OLD_POLICY = "permitted=v<2 || (regs<=128 && shared+stat<=51200 && local==0 && active>=2);"
NEW_POLICY = "permitted=active>=1;"
RUNNER = "attention-launchability-bench"
PERMIT = "attention-launchability-offline.json"
TOOLS = ("build.py", "run.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def evidence_dir(path):
    path = path.resolve()
    if not any(path.is_relative_to(ROOT / p) for p in ("artifacts", "docs/validation")):
        raise ValueError("output must be inside ignored artifacts/ or docs/validation/")
    return path


def frozen_module():
    if {n: digest(FROZEN / n) for n in PINS} != PINS:
        raise ValueError("sealed generator/runner/attention64/checker changed; review required")
    spec = importlib.util.spec_from_file_location("sealed_attention_for_launchability", FROZEN / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generated(source, variant):
    text = frozen_module().generate(source, variant)
    if hashlib.sha256(text.encode()).hexdigest() != KERNEL_PINS[variant]:
        raise ValueError("generated attention arithmetic differs from the pinned sealed experiment")
    return text


def runner_source():
    frozen_module()
    original = (FROZEN / "bench.cu").read_text()
    if original.count(OLD_POLICY) != 1 or NEW_POLICY in original:
        raise ValueError("runner resource policy anchor changed")
    result = original.replace(OLD_POLICY, NEW_POLICY, 1)
    if result.replace(NEW_POLICY, OLD_POLICY, 1) != original:
        raise ValueError("private runner changed more than the resource policy")
    return result


def flags(o1):
    return [*BASE_FLAGS, "-Xptxas=-v,-warn-spills" + (",-O1" if o1 else "")]


def compile_command(nvcc, arch, directory, variant, arm):
    candidate, o1 = ARMS[arm]
    if variant not in (0, candidate):
        raise ValueError("only baseline0 and this arm's candidate may compile")
    return [nvcc, "-std=c++17", "-cubin", "-arch=" + arch, *flags(o1 if variant else False),
            str(directory / f"variant-{variant}.cu"), "-o", str(directory / f"variant-{variant}.cubin")]


def parse_resources(text, variant):
    if len(re.findall(r"Compiling entry function ['\"]ts_bench_attn_prefill['\"]", text)) != 1:
        raise ValueError("missing/duplicate actual attention ptxas resource record")
    original = frozen_module().parse_resources(text, variant)
    keys = ("registers_per_thread", "stack_bytes", "spill_store_bytes", "spill_load_bytes",
            "static_shared_bytes", "dynamic_shared_bytes", "total_shared_bytes")
    return {**{k: original[k] for k in keys},
            "sealed_policy_would_admit": original["gpu_gate_permitted"],
            "sealed_policy_status": original["status"],
            "status": "OFFLINE_RESOURCE_FACTS_ONLY_DRIVER_NOT_RUN"}


def artifact_names(variant):
    return ("variant-0.cu", f"variant-{variant}.cu", "variant-0.cubin", f"variant-{variant}.cubin",
            "bench.cu", "checks.h", "qualified-variants.txt", RUNNER)


def command(argv, log, timeout):
    """Timeout kills only the newly created, not-yet-reaped child session."""
    with log.open("wb") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    text = log.read_text(errors="replace")
    if code:
        raise subprocess.CalledProcessError(code, argv, output=text)
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", choices=("sm_120a", "sm_120f"), default="sm_120a")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    environment = {n: os.environ.get(n) for n in ENV_NAMES}
    if any((environment[n] or "").strip() for n in ENV_NAMES[:2]):
        parser.error("unset implicit NVCC_PREPEND_FLAGS/NVCC_APPEND_FLAGS")
    try:
        out = evidence_dir(args.out)
        if out.exists() or args.timeout <= 0:
            raise ValueError("fresh output and positive timeout required")
        variant, o1 = ARMS[args.arm]
        text = args.source.read_text()
        sources = {i: generated(text, i) for i in (0, variant)}
        runner = runner_source()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    out.mkdir(parents=True, exist_ok=False)
    for i, source in sources.items():
        (out / f"variant-{i}.cu").write_text(source)
    (out / "bench.cu").write_text(runner)
    (out / "checks.h").write_bytes((FROZEN / "checks.h").read_bytes())
    manifest = dict(schema="qwen35-attention-launchability-build/1", complete=False,
                    generation_complete=True, arm=args.arm, candidate_variant=variant, arch=args.arch,
                    source=str(args.source.resolve()), source_sha256=digest(args.source), frozen_tools=PINS,
                    tools={n: digest(HERE / n) for n in TOOLS}, compiler_environment=environment,
                    baseline_flags=flags(False), candidate_flags=flags(o1),
                    runner_delta="resource predicate only; reverse-delta byte assertion passed",
                    resource_policy="actual driver activeCTAs>=1; record all registers/shared/local/stack/spills; no imposed register/2CTA/no-spill goal",
                    old_resource_failures_regraded=False, gpu_quality="NOT_RUN", gpu_timing="NOT_RUN",
                    driver_resources="NOT_RUN", production_integrated=False, commands=[])
    save(out / "manifest.json", manifest)
    if args.generate_only:
        manifest["status"] = "SOURCE_ONLY_NOT_COMPILED_NOT_GPU_VALIDATED"
        save(out / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2))
        return 0

    def invoke(name, argv):
        entry = dict(name=name, argv=argv, exit_code=None)
        manifest["commands"].append(entry)
        save(out / "manifest.json", manifest)
        try:
            value = command(argv, out / (name + ".log"), args.timeout)
            entry["exit_code"] = 0
            return value
        except subprocess.CalledProcessError as error:
            entry["exit_code"] = error.returncode
            raise
        finally:
            if (out / (name + ".log")).exists():
                entry["log_sha256"] = digest(out / (name + ".log"))
            save(out / "manifest.json", manifest)

    try:
        manifest["nvcc_version"] = invoke("nvcc-version", [args.nvcc, "--version"])
        executable = shutil.which(args.nvcc)
        manifest["nvcc_executable"] = dict(requested=args.nvcc, resolved=executable,
                                          sha256=digest(executable) if executable else None)
        manifest["resources"] = {}
        for i in (0, variant):
            log = invoke(f"variant-{i}", compile_command(args.nvcc, args.arch, out, i, args.arm))
            if not (out / f"variant-{i}.cubin").is_file():
                raise ValueError("compiler success without cubin")
            manifest["resources"][str(i)] = parse_resources(log, i)
        invoke("runner-build", [args.nvcc, "-std=c++17", "-O3", str(out / "bench.cu"),
                                "-lcuda", "-o", str(out / RUNNER)])
        if not (out / RUNNER).is_file():
            raise ValueError("compiler success without runner")
        if digest(args.source) != manifest["source_sha256"] or {n: digest(HERE / n) for n in TOOLS} != manifest["tools"]:
            raise ValueError("production source or owned tools changed during build")
        frozen_module()
        (out / "qualified-variants.txt").write_text(f"0\n{variant}\n")
        manifest.update(complete=True, status="OFFLINE_COMPILED_DRIVER_AND_QUALITY_NOT_RUN",
                        artifacts={n: digest(out / n) for n in artifact_names(variant)})
        save(out / "manifest.json", manifest)
        save(out / PERMIT, dict(schema="qwen35-attention-launchability-offline/1",
                               manifest_sha256=digest(out / "manifest.json"), artifacts=manifest["artifacts"],
                               quality="NOT_RUN", sealed_gate_override=False))
        print(json.dumps(manifest, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        manifest.update(complete=False, status="OFFLINE_FAILED_NOT_VALIDATED", error=str(error))
        save(out / "manifest.json", manifest)
        print(json.dumps(manifest, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
