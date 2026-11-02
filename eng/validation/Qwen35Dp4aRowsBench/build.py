#!/usr/bin/env python3
"""Isolated N8 DP4a rows-per-warp experiment. Offline compiler only; never CUDA init.

R1/R2/R3/R4 use private wrappers around the unchanged production template.
The production-style oracle and R1/R2/R3 use 128-thread launches, changing only R.
An additional R4/block32 wrapper is a separate launch-only control.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = HERE.with_name("Qwen35Int8MmaBench")
SPEC = importlib.util.spec_from_file_location("sealed_int8mma_fixture_source_for_dp4a_rows", FROZEN / "build.py")
SEALED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEALED)
SOURCE = SEALED.SOURCE
ORACLE = "ts_q35_gemv_nvfp4_8"
ROWS = (1, 2, 3, 4)
SYMBOLS = (ORACLE,) + tuple(f"ts_bench_dp4a_rows_8_r{r}" for r in ROWS)
PERMIT = "dp4a-rows-resources-permitted.json"
RUNNER = "dp4a-rows-bench"
ARTIFACTS = ("kernels.cubin", "kernels.ptx", "kernels.cu", "plan.h", "plan.json", RUNNER)
TOOLS = ("build.py", "run.py", "bench.cu")
FLAGS = list(SEALED.FLAGS)
FROZEN_FILES = ("build.py", "bench.cu", "run.py", "int8mma.cuh")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def generate(source):
    # Do not rewrite the prefix, helper arithmetic, accumulation or reduction.
    prefix = SEALED.oracle_prefix(source)
    wrappers = []
    for r in ROWS:
        wrappers.append(f'''extern "C" __global__ void ts_bench_dp4a_rows_8_r{r}(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, int outDim, int k,
    const int8_t* __restrict__ xq, const float* __restrict__ xd, float alpha, float* __restrict__ y)
{{ ts_q35_gemv_nvfp4<8, {r}>(codes, scales, outDim, k, xq, xd, alpha, y); }}
''')
    return "#include <cuda_runtime.h>\n" + prefix + "\n// Private experiment wrappers; production source remains unchanged.\n" + "\n".join(wrappers)


def launch_geometry(m, rows, threads):
    if m <= 0 or rows not in ROWS or threads not in (32, 128):
        raise ValueError("unsupported launch")
    per_block = rows * (threads // 32)
    return (m + per_block - 1) // per_block, threads


def parse_resources(log):
    resources = {}
    for symbol in SYMBOLS:
        records = re.findall(rf"Compiling entry function ['\"]{re.escape(symbol)}['\"][\s\S]*?(?=Compiling entry function|\Z)", log)
        if len(records) != 1:
            raise ValueError(f"expected exactly one actual ptxas record for {symbol}")
        body = records[0]
        regs = re.search(r"Used (\d+) registers", body)
        stack = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
        shared = re.search(r"(\d+) bytes smem", body)
        if not regs or not stack:
            raise ValueError(f"incomplete actual ptxas record for {symbol}")
        # No __shared__ object exists in the verbatim template or wrappers;
        # ptxas omits the zero-byte smem field. Driver checks it again.
        entry = dict(registers_per_thread=int(regs[1]), stack_bytes=int(stack[1]),
            spill_store_bytes=int(stack[2]), spill_load_bytes=int(stack[3]),
            static_shared_bytes=int(shared[1]) if shared else 0,
            block_threads=32 if symbol == SYMBOLS[4] else 128)
        entry["offline_resource_qualified"] = (0 < entry["registers_per_thread"] <= 255
            and not any(entry[k] for k in ("stack_bytes", "spill_store_bytes", "spill_load_bytes", "static_shared_bytes")))
        resources[symbol] = entry
    return resources


def evidence_dir(path):
    path = path.resolve()
    if not any(path.is_relative_to(ROOT / p) for p in ("artifacts", "docs/validation")):
        raise ValueError("output must be inside this checkout's ignored artifacts/ or docs/validation/")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path, help="NEW ignored directory; never overwrite/resume")
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--nvcc", default="nvcc")
    ap.add_argument("--cuobjdump", default="cuobjdump")
    ap.add_argument("--arch", default="sm_120a", choices=("sm_120", "sm_120a", "sm_120f"))
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args(argv)
    try:
        out = evidence_dir(args.out)
        if out.exists():
            raise ValueError("--out already exists; preserve earlier failures/build evidence")
    except ValueError as error:
        ap.error(str(error))
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="qwen35-dp4a-rows-build/1", complete=False, generation_complete=False,
        source=str(args.source.resolve()), source_sha256=None,
        tools={n:digest(HERE / n) for n in TOOLS},
        reused_fixture_tools={n:digest(FROZEN / n) for n in FROZEN_FILES},
        build_environment={n:os.environ.get(n) for n in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "CUDA_HOME", "CUDA_PATH")},
        arch=args.arch, flags=FLAGS, activation_rows=8, activation_rows_NOT_SUPPORTED=list(range(1, 8)),
        oracle=dict(symbol=ORACLE, rows_per_warp=4, block_threads=128),
        candidates=[dict(symbol=SYMBOLS[r], rows_per_warp=r, block_threads=32 if r == 4 else 128,
                         role="launch_only_control" if r == 4 else "rows_only_experiment") for r in ROWS],
        resource_policy=dict(max_registers_per_thread=255, stack_bytes=0, spill_bytes=0, static_shared_bytes=0,
                             note="natural no-spill launchability gate; not Int8MMA's unrelated <=64-register budget"),
        cuda_compilation="NOT_RUN", gpu_quality="NOT_RUN", gpu_timing="NOT_RUN", hardware_counters="NOT_RUN",
        production_integration=False, limitations=[
            "N8 only; W4/A8, scale groups and production arithmetic remain unchanged.",
            "R4/block32 isolates launch geometry from the production-style R4/block128 oracle.",
            "All candidates require actual ptxas and driver resource evidence and complete bit/guard/argmax/A-A gates.",
            "Register count alone does not establish a performance bottleneck or speedup.",
            "Synthetic projections are not model quality or end-to-end performance evidence.",
            "Imported oracle is recompiled with recorded flags, not asserted identical to the production fatbin."])
    state_path = out / "manifest.json"
    save(state_path, manifest)
    try:
        if any(os.environ.get(n, "").strip() for n in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")):
            raise ValueError("NVCC_PREPEND_FLAGS/NVCC_APPEND_FLAGS must be unset")
        source_bytes = args.source.read_bytes()
        source = source_bytes.decode()
        manifest["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        generated = generate(source)
        if "__shared__" in generated:
            raise ValueError("unexpected shared object; re-review resource gates")
        (out / "kernels.cu").write_text(generated)
        cases = SEALED.plan()
        (out / "plan.json").write_text(json.dumps(cases, indent=2) + "\n")
        (out / "plan.h").write_text(SEALED.plan_header(cases))
        manifest.update(generation_complete=True, generated_sha256=digest(out / "kernels.cu"),
            oracle_prefix_sha256=hashlib.sha256(SEALED.oracle_prefix(source).encode()).hexdigest(),
            strict_cases=len(cases), timed_shapes=sum(c["timed"] for c in cases),
            planned_strict_candidate_comparisons=len(cases)*len(ROWS),
            plan_sha256=digest(out / "plan.json"))
        if args.generate_only:
            manifest["status"] = "SOURCE_GENERATED_NOT_COMPILED_NOT_GPU_VALIDATED"
            save(state_path, manifest)
            print(json.dumps(manifest, indent=2)); return 0
        version = subprocess.run([args.nvcc, "--version"], capture_output=True, text=True, check=True)
        manifest["nvcc_version"] = version.stdout + version.stderr
        common = [args.nvcc, "-std=c++17", f"-arch={args.arch}", *FLAGS]
        command = common + ["-cubin", "-Xptxas=-v,-warn-spills", str(out / "kernels.cu"), "-o", str(out / "kernels.cubin")]
        result = subprocess.run(command, capture_output=True, text=True)
        log = result.stdout + result.stderr
        (out / "kernels.build.log").write_text(log)
        manifest.update(cuda_compilation="RUN", compile_command=command, compile_exit_code=result.returncode)
        if result.returncode:
            raise RuntimeError("CUDA compilation failed")
        manifest["resources"] = parse_resources(log)
        ptx_command = common + ["-ptx", str(out / "kernels.cu"), "-o", str(out / "kernels.ptx")]
        result = subprocess.run(ptx_command, capture_output=True, text=True)
        (out / "ptx.build.log").write_text(result.stdout + result.stderr)
        manifest.update(ptx_command=ptx_command, ptx_exit_code=result.returncode)
        if result.returncode:
            raise RuntimeError("PTX generation failed")
        dump = shutil.which(args.cuobjdump)
        if dump:
            result = subprocess.run([dump, "--dump-sass", str(out / "kernels.cubin")], capture_output=True, text=True)
            (out / "kernels.sass.txt").write_text(result.stdout + result.stderr)
            manifest["sass_exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError("SASS extraction failed")
        else:
            manifest["sass_status"] = "tool_unavailable; full-output GPU bit gate remains mandatory"
        if not all(r["offline_resource_qualified"] for r in manifest["resources"].values()):
            raise RuntimeError("actual resources rejected; no forced register constraint or candidate pruning")
        command = [args.nvcc, "-std=c++17", "-O3", "-I", str(out), str(HERE / "bench.cu"), "-lcuda", "-o", str(out / RUNNER)]
        result = subprocess.run(command, capture_output=True, text=True)
        (out / "runner.build.log").write_text(result.stdout + result.stderr)
        manifest.update(runner_command=command, runner_exit_code=result.returncode)
        if result.returncode:
            raise RuntimeError("runner compilation failed")
        if digest(args.source) != manifest["source_sha256"]:
            raise RuntimeError("production source changed during build")
        if manifest["tools"] != {n:digest(HERE / n) for n in TOOLS} or manifest["reused_fixture_tools"] != {n:digest(FROZEN / n) for n in FROZEN_FILES}:
            raise RuntimeError("tool/fixture source changed during build")
        manifest.update(complete=True, artifacts={n:digest(out / n) for n in ARTIFACTS},
                        status="OFFLINE_BUILD_RESOURCES_QUALIFIED_GPU_NOT_RUN")
        save(state_path, manifest)
        save(out / PERMIT, dict(schema="qwen35-dp4a-rows-permission/1", manifest_sha256=digest(state_path),
            artifacts=manifest["artifacts"], status="offline_resources_only_driver_and_fullbits_NOT_RUN"))
        print(json.dumps(manifest, indent=2)); return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        manifest.update(complete=False, status="BUILD_OR_RESOURCE_REJECTED_GPU_NOT_VALIDATED", error=str(error))
        save(state_path, manifest)
        print(json.dumps(manifest, indent=2)); return 1


if __name__ == "__main__":
    raise SystemExit(main())
