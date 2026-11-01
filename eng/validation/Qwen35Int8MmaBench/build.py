#!/usr/bin/env python3
"""Offline build/resource gate; never initializes CUDA or runs the GPU executable.

  python eng/validation/Qwen35Int8MmaBench/build.py --out artifacts/qwen38-int8mma --generate-only
  python eng/validation/Qwen35Int8MmaBench/build.py --out artifacts/qwen38-int8mma --nvcc /usr/local/cuda/bin/nvcc

GPU execution is a separate explicit run.py command. N1..7 are NOT_SUPPORTED.
Generated logs/cubins/plans must stay in ignored artifacts/ or docs/validation/.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "TensorSharp.Backends.Cuda/native/kernels/tensorsharp_q35_kernels.cu"
CANDIDATE = "ts_bench_gemv_nvfp4_int8mma_8"
ORACLE = "ts_q35_gemv_nvfp4_8"
ARTIFACTS = ("kernels.cubin", "kernels.ptx", "kernels.cu", "plan.h", "plan.json", "int8mma-bench")
TOOLS = ("build.py", "run.py", "bench.cu", "int8mma.cuh")
FLAGS = ["-O3", "--extra-device-vectorization", "--ftz=false", "--prec-div=true", "--prec-sqrt=true", "--fmad=true", "-lineinfo"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def oracle_prefix(source):
    """Import the production prefix verbatim, including its helpers/template/N8."""
    marker = "// Activation rows -> NVFP4 planes + per-row scale, ggml-cuda's quantize_mmq_nvfp4 numerics."
    if source.count(marker) != 1:
        raise ValueError("production oracle boundary changed; review before building")
    prefix = source[:source.index(marker)]
    required = ["#define TS_Q35_NVFP4_BLOCK 64", "#define TS_Q35_NVFP4_BLOCK_BYTES 36",
                "TS_Q35_GEMV(8, TS_Q35_GEMV_ROWS(8))",
                "#define TS_Q35_GEMV_ROWS(N) ((N) <= 1 ? 1 : (N) <= 3 ? 2 : 4)",
                "acc[n][j] += ds[j] * xs * (float)sumi;",
                "for (int off = 16; off > 0; off >>= 1)", "for (int b = lane; b < nblocks; b += 32)"]
    if any(anchor not in prefix for anchor in required):
        raise ValueError("production GEMV layout/arithmetic changed; review experiment")
    return prefix


def generate(source):
    return "#include <cuda_runtime.h>\n" + oracle_prefix(source) + "\n" + (HERE / "int8mma.cuh").read_text()


def plan():
    cases = []
    def add(k, m, pattern=0, alpha=0x3f800000, timed=False):
        cases.append(dict(k=k, out_dim=m, pattern=pattern, alpha_bits=alpha, timed=timed))
    for k in (5120, 6144, 17408):
        for m in (5120, 14336, 16480, 34816):
            add(k, m, timed=True)
    for k in (64, 128, 1984, 2048, 2112, 5120):
        for m in (1, 7, 8, 15, 16, 17, 31, 33):
            add(k, m, pattern=(k // 64 + m) % 5)
    for m in (5119, 5121, 14335, 14337, 16479, 16481, 34815, 34817):
        add(2112, m, pattern=4)
    for alpha in (0, 0x80000000, 0x3f800000, 0xbf800000, 0x00800000, 0x00000001, 0x7f7fffff, 0xff7fffff):
        for pattern in (1, 2, 3):
            add(2112, 17, pattern, alpha)
    for m in (1, 15, 16, 17):
        add(64, m, pattern=5, alpha=0x80000000)
    return cases


def plan_header(cases):
    rows = [f"    {{{s['k']},{s['out_dim']},{s['pattern']},0x{s['alpha_bits']:08x}u,{'true' if s['timed'] else 'false'}}}" for s in cases]
    return "// Generated validation plan, not production.\nstatic const Shape StrictPlan[] = {\n" + ",\n".join(rows) + "\n};\n"


def parse_resources(log):
    marker = rf"Compiling entry function ['\"]{CANDIDATE}['\"][\s\S]*?(?=Compiling entry function|\Z)"
    records = re.findall(marker, log)
    if len(records) != 1:
        raise ValueError("expected exactly one candidate ptxas resource record")
    body = records[0]
    regs = re.search(r"Used (\d+) registers", body)
    stack = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
    shared = re.search(r"(\d+) bytes smem", body)
    if not regs or not stack or not shared:
        raise ValueError("candidate resource record incomplete; no GPU quality authorization")
    d = dict(registers_per_thread=int(regs[1]), stack_bytes=int(stack[1]),
             spill_store_bytes=int(stack[2]), spill_load_bytes=int(stack[3]),
             static_shared_bytes=int(shared[1]), block_threads=1024)
    d["allocated_registers_per_cta"] = ((d["registers_per_thread"] + 7) // 8 * 8) * 1024
    d["offline_resource_qualified"] = (0 < d["registers_per_thread"] <= 64
        and 16508 <= d["static_shared_bytes"] <= 16512
        and not any(d[key] for key in ("stack_bytes", "spill_store_bytes", "spill_load_bytes")))
    d["status"] = "offline_resources_qualified_GPU_NOT_RUN" if d["offline_resource_qualified"] else "resource_rejected_no_gpu_quality_run"
    return d


def evidence_dir(path):
    path = path.resolve()
    if not any(path.is_relative_to(base) for base in (ROOT / "artifacts", ROOT / "docs/validation")):
        raise ValueError("--out must be inside this checkout's artifacts/ or docs/validation/")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--source", type=Path, default=SOURCE, help="unchanged production kernel source; useful for a separate VM harness checkout")
    ap.add_argument("--nvcc", default="nvcc")
    ap.add_argument("--cuobjdump", default="cuobjdump")
    ap.add_argument("--arch", default="sm_120a", choices=("sm_120", "sm_120a", "sm_120f"))
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args(argv)
    try: out = evidence_dir(args.out)
    except ValueError as error: ap.error(str(error))
    out.mkdir(parents=True, exist_ok=True)
    permit = out / "gpu-quality-permitted.json"
    permit.unlink(missing_ok=True) # stale cubins must never authorize a new failed build
    manifest = dict(schema="qwen35-int8mma-build/1", complete=False, generation_complete=False,
        source=str(args.source.resolve()), source_sha256=None,
        tools={name: digest(HERE / name) for name in TOOLS},
        build_environment={name:os.environ.get(name) for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "CUDA_HOME", "CUDA_PATH")},
        arch=args.arch, flags=FLAGS, logical_scale_group=16, physical_mma_k=32,
        activation_rows=8, activation_rows_NOT_SUPPORTED=list(range(1, 8)),
        cuda_compilation="NOT_RUN", gpu_quality="NOT_RUN", gpu_timing="NOT_RUN", hardware_counters="NOT_RUN",
        limitations=["Synthetic projections, not model-level quality or end-to-end speed evidence.",
          "Imported oracle and candidate are compiled with identical recorded flags; actual production fatbin equivalence is not assumed.",
          "No resource constraint is forced; excessive registers/any stack/spill reject the candidate."])
    save = lambda: (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    try:
        if any(os.environ.get(name, "").strip() for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")):
            raise ValueError("NVCC_PREPEND_FLAGS/NVCC_APPEND_FLAGS must be unset; unrecorded precision/resource overrides are rejected")
        manifest["source_sha256"] = digest(args.source)
        source = args.source.read_text()
        generated = out / "kernels.cu"
        generated.write_text(generate(source))
        cases = plan()
        (out / "plan.json").write_text(json.dumps(cases, indent=2) + "\n")
        (out / "plan.h").write_text(plan_header(cases))
        manifest.update(generation_complete=True, oracle_prefix_sha256=hashlib.sha256(oracle_prefix(source).encode()).hexdigest(),
                        generated_sha256=digest(generated), strict_cases=len(cases), timed_shapes=sum(s["timed"] for s in cases))
        if args.generate_only:
            manifest["status"] = "source_generation_only_NOT_COMPILED_NOT_GPU_VALIDATED"
            save(); print(json.dumps(manifest, indent=2)); return 0
        version = subprocess.run([args.nvcc, "--version"], capture_output=True, text=True, check=True)
        manifest["nvcc_version"] = version.stdout + version.stderr
        common = [args.nvcc, "-std=c++17", f"-arch={args.arch}", *FLAGS]
        command = common + ["-cubin", "-Xptxas=-v,-warn-spills", str(generated), "-o", str(out / "kernels.cubin")]
        result = subprocess.run(command, capture_output=True, text=True)
        log = result.stdout + result.stderr
        (out / "kernels.build.log").write_text(log)
        manifest.update(cuda_compilation="RUN", compile_command=command, compile_exit_code=result.returncode)
        if result.returncode: raise RuntimeError("CUDA compilation failed; see kernels.build.log")
        manifest["resources"] = parse_resources(log)
        # Keep actual PTX/SASS for contraction/layout auditing, even if resources reject it.
        ptxcmd = common + ["-ptx", str(generated), "-o", str(out / "kernels.ptx")]
        ptx = subprocess.run(ptxcmd, capture_output=True, text=True)
        (out / "ptx.build.log").write_text(ptx.stdout + ptx.stderr)
        manifest.update(ptx_command=ptxcmd, ptx_exit_code=ptx.returncode)
        if ptx.returncode: raise RuntimeError("PTX evidence generation failed")
        dump = shutil.which(args.cuobjdump)
        if dump:
            dis = subprocess.run([dump, "--dump-sass", str(out / "kernels.cubin")], capture_output=True, text=True)
            (out / "kernels.sass.txt").write_text(dis.stdout + dis.stderr)
            manifest["sass_exit_code"] = dis.returncode
            if dis.returncode: raise RuntimeError("SASS extraction failed")
        else:
            manifest["sass_status"] = "tool_unavailable; PTX and full-output GPU bit gate still required"
        if not manifest["resources"]["offline_resource_qualified"]:
            raise RuntimeError("candidate exceeds resource budget; no maxrregcount override or GPU quality permit")
        runnercmd = [args.nvcc, "-std=c++17", "-O3", "-I", str(out), str(HERE / "bench.cu"), "-lcuda", "-o", str(out / "int8mma-bench")]
        runner = subprocess.run(runnercmd, capture_output=True, text=True)
        (out / "runner.build.log").write_text(runner.stdout + runner.stderr)
        manifest.update(runner_command=runnercmd, runner_exit_code=runner.returncode)
        if runner.returncode: raise RuntimeError("host GPU runner build failed")
        if digest(args.source) != manifest["source_sha256"]: raise RuntimeError("production source changed during build")
        artifacts = {name: digest(out / name) for name in ARTIFACTS}
        manifest.update(complete=True, artifacts=artifacts, status="offline_build_resources_qualified_GPU_NOT_RUN")
        save()
        permit.write_text(json.dumps({"manifest_sha256": digest(out / "manifest.json"), "artifacts": artifacts,
            "status": "offline_resource_permission_only; actual_driver_launch_and_full_bit_gate_NOT_RUN"}, indent=2) + "\n")
        print(json.dumps(manifest, indent=2)); return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        manifest.update(complete=False, status="build_or_resource_rejected_NOT_GPU_VALIDATED", error=str(error))
        save(); print(json.dumps(manifest, indent=2)); return 1


if __name__ == "__main__":
    raise SystemExit(main())
