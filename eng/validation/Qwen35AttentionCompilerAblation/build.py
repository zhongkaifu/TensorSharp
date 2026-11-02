#!/usr/bin/env python3
"""Explicit standalone compiler experiments; never changes frozen variants0..4.

Each arm compiles an unchanged baseline0 plus one variant3/4 candidate. New
output directories are mandatory. Source generation never authorizes GPU work.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = HERE.with_name("Qwen35AttentionBench")
SPEC = importlib.util.spec_from_file_location("frozen_q35_attention_for_ablation", FROZEN / "build.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
EXPERIMENTS = ("no_vectorization", "ptxas_o1", "min_blocks2")
BASE_FLAGS = ("-O3", "--extra-device-vectorization", "--ftz=false", "--prec-div=true", "--prec-sqrt=true")
LAUNCH = "__launch_bounds__(256) ts_bench_attn_prefill("
HINT = "__launch_bounds__(256, 2) ts_bench_attn_prefill("
COMPILER_ENV_NAMES = ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "CUDA_HOME", "CUDA_PATH")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def changed_source(source: str, variant: int, experiment: str) -> str:
    if variant not in (3, 4) or experiment not in EXPERIMENTS:
        raise ValueError("only explicit variant3/4 compiler experiments are supported")
    candidate = BASE.generate(source, variant)
    if experiment == "min_blocks2":
        return BASE.once(candidate, LAUNCH, HINT)
    return candidate


def flags(experiment: str | None) -> tuple[list[str], str]:
    if experiment is not None and experiment not in EXPERIMENTS:
        raise ValueError("unknown compiler experiment")
    options = list(BASE_FLAGS)
    if experiment == "no_vectorization":
        options.remove("--extra-device-vectorization")
    ptxas = "-Xptxas=-v,-warn-spills" + (",-O1" if experiment == "ptxas_o1" else "")
    return options, ptxas


def compile_command(nvcc: str, arch: str, directory: Path, variant: int, experiment: str) -> list[str]:
    # The control's source, front-end flags and ptxas flags are always unchanged.
    options, ptxas = flags(None if variant == 0 else experiment)
    return [nvcc, "-std=c++17", "-cubin", f"-arch={arch}", *options, ptxas,
            str(directory / f"variant-{variant}.cu"), "-o", str(directory / f"variant-{variant}.cubin")]


def build_arm(directory: Path, manifest: dict, nvcc: str, arch: str) -> bool:
    """Publish a marker only after actual successful compilation and resource gates."""
    failed = False
    for entry in manifest["variants"]:
        variant = entry["variant"]
        command = compile_command(nvcc, arch, directory, variant, manifest["experiment"])
        entry["command"] = command
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            log = result.stdout + result.stderr
            entry["exit_code"] = result.returncode
            if result.returncode:
                entry["status"] = "compile_failed_not_validated"
                failed = True
            else:
                binary = directory / f"variant-{variant}.cubin"
                if not binary.is_file():
                    raise ValueError("compiler returned success without the requested cubin")
                entry.update(BASE.parse_resources(log, variant))
                entry["cubin_sha256"] = sha(binary.read_bytes())
        except OSError as error:
            log = str(error) + "\n"
            entry.update(exit_code=None, status="compiler_unavailable_not_run", error=str(error))
            failed = True
        except ValueError as error:
            entry.update(status="resources_unverified_no_gpu_run", error=str(error))
            failed = True
        (directory / f"variant-{variant}.build.log").write_text(log)
        save(directory / "manifest.json", manifest)

    if not failed:
        command = [nvcc, "-std=c++17", "-O3", str(directory / "bench.cu"), "-lcuda", "-o", str(directory / "attention-bench")]
        manifest["runner_command"] = command
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            log = result.stdout + result.stderr
            manifest["runner_exit_code"] = result.returncode
            binary = directory / "attention-bench"
            if result.returncode or not binary.is_file():
                manifest["runner_error"] = "runner compilation failed or output binary missing"
                failed = True
            else:
                manifest["runner_sha256"] = sha(binary.read_bytes())
        except OSError as error:
            log = str(error) + "\n"
            manifest.update(runner_exit_code=None, runner_error=str(error))
            failed = True
        (directory / "runner.build.log").write_text(log)

    candidate = manifest["variants"][1]
    permitted = not failed and candidate.get("two_cta_resource_gate") is True
    manifest.update(complete=not failed, validation="offline_compile_resources_only",
                    candidate_offline_qualified=permitted,
                    gpu_authorization="offline_resources_only_driver_and_full_bits_still_required" if permitted else "NOT_AUTHORIZED")
    # Native handles are not involved here. More importantly, no integer-only
    # synthetic resource description or generation-only result can publish this.
    # A resource-rejected arm receives no marker, even for its baseline alone.
    save(directory / "manifest.json", manifest)
    if permitted:
        temporary = directory / "qualified-variants.tmp"
        temporary.write_text(f"0\n{candidate['variant']}\n")
        temporary.replace(directory / "qualified-variants.txt")
    return not failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="NEW ignored directory, never overwritten/resumed")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", default="sm_120a")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--variants", default="3,4")
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    args = parser.parse_args(argv)
    build_environment = {name: os.environ.get(name) for name in COMPILER_ENV_NAMES}
    if any((build_environment[name] or "").strip() for name in COMPILER_ENV_NAMES[:2]):
        parser.error("NVCC_PREPEND_FLAGS/NVCC_APPEND_FLAGS must be unset; hidden precision/resource overrides are rejected")
    try:
        variants = [int(v) for v in args.variants.split(",")]
    except ValueError:
        parser.error("--variants must be comma-separated3/4")
    experiments = args.experiments.split(",")
    if not variants or len(set(variants)) != len(variants) or any(v not in (3, 4) for v in variants):
        parser.error("--variants must uniquely select only3/4; control0 is always added unchanged")
    if not experiments or len(set(experiments)) != len(experiments) or any(e not in EXPERIMENTS for e in experiments):
        parser.error("select unique explicit experiments: " + ",".join(EXPERIMENTS))
    out = args.out.resolve()
    if not any(out.is_relative_to(ROOT / p) for p in ("artifacts", "docs/validation")):
        parser.error("--out must be inside ignored artifacts/ or docs/validation/")
    ignored = subprocess.run(["git", "check-ignore", str(out / "manifest.json")], cwd=ROOT, capture_output=True)
    if ignored.returncode:
        parser.error("--out is not ignored")
    if out.exists():
        parser.error("--out already exists; refusing to overwrite earlier resource failures or authorization")
    # Complete all source transformations before claiming ownership of an output
    # directory, so a changed upstream anchor cannot leave a qualified old tree.
    source = BASE.SOURCE.read_text()
    control = BASE.generate(source, 0)
    generated = {(v, e): changed_source(source, v, e) for v in variants for e in experiments}
    bench, checks = (FROZEN / "bench.cu").read_bytes(), (FROZEN / "checks.h").read_bytes()
    provenance = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in
                  (BASE.SOURCE, FROZEN / "build.py", FROZEN / "attention64.cuh", FROZEN / "bench.cu", FROZEN / "checks.h", HERE / "build.py")}
    out.mkdir(parents=True, exist_ok=False)
    top = dict(schema="q35-attention-compiler-ablation/1", complete=False,
               validation="source_generation_only" if args.generate_only else "offline_compile_resources_only",
               arch=args.arch, source_sha256=provenance, gpu_runs=0, hardware_counters="NOT_RUN",
               driver_resources="NOT_RUN", model_quality="NOT_RUN", production_integration=False,
               baseline0_source_sha256=sha(control.encode()), baseline0_flags=list(BASE_FLAGS), arms=[],
               build_environment=build_environment,
               compiler=dict(requested=args.nvcc, resolved=shutil.which(args.nvcc), version_status="NOT_RUN"))
    save(out / "manifest.json", top)
    if not args.generate_only:
        command = [args.nvcc, "--version"]
        compiler = top["compiler"]
        compiler["version_command"] = command
        try:
            version = subprocess.run(command, capture_output=True, text=True)
            log = version.stdout + version.stderr
            compiler.update(version_exit_code=version.returncode,
                            version_status="RECORDED" if version.returncode == 0 else "UNAVAILABLE_NOT_VALIDATED")
        except OSError as error:
            log = str(error) + "\n"
            compiler.update(version_exit_code=None, version_status="UNAVAILABLE_NOT_VALIDATED", error=str(error))
        (out / "compiler.version.log").write_text(log)
        compiler["version_log_sha256"] = sha(log.encode())
        if compiler["version_status"] != "RECORDED":
            top["error"] = "compiler version unavailable; no native build or GPU authorization"
            save(out / "manifest.json", top)
            print(json.dumps(top, indent=2))
            return 1
        save(out / "manifest.json", top)
    successful = True
    for variant in variants:
        for experiment in experiments:
            directory = out / f"variant{variant}-{experiment}"
            directory.mkdir()
            (directory / "variant-0.cu").write_text(control)
            (directory / f"variant-{variant}.cu").write_text(generated[variant, experiment])
            (directory / "bench.cu").write_bytes(bench)
            (directory / "checks.h").write_bytes(checks)
            options, ptxas = flags(experiment)
            manifest = dict(schema=top["schema"], experiment=experiment, candidate_variant=variant,
                complete=args.generate_only, validation=top["validation"], arch=args.arch,
                source_sha256=provenance, gpu_runs=0, driver_resources="NOT_RUN", model_quality="NOT_RUN",
                build_environment=build_environment, compiler=top["compiler"],
                hardware_counters="NOT_RUN", candidate_offline_qualified=False, gpu_authorization="NOT_AUTHORIZED",
                compiler_experiment=dict(candidate_flags=options, candidate_ptxas=ptxas,
                    baseline_flags=list(BASE_FLAGS), baseline_ptxas=flags(None)[1],
                    launch_bounds_min_blocks=2 if experiment == "min_blocks2" else None,
                    hint_is_resource_experiment_not_a_pass=experiment == "min_blocks2"),
                runner_source_sha256=sha(bench), checker_source_sha256=sha(checks),
                variants=[dict(variant=i, source_sha256=sha((directory / f"variant-{i}.cu").read_bytes()), status="generated_not_compiled")
                          for i in (0, variant)])
            save(directory / "manifest.json", manifest)
            if not args.generate_only:
                successful = build_arm(directory, manifest, args.nvcc, args.arch) and successful
            top["arms"].append(dict(directory=directory.name, experiment=experiment, variant=variant,
                                   complete=manifest["complete"], candidate_offline_qualified=manifest["candidate_offline_qualified"],
                                   validation=manifest["validation"]))
            save(out / "manifest.json", top)
    top["complete"] = successful
    save(out / "manifest.json", top)
    print(json.dumps(top, indent=2))
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
