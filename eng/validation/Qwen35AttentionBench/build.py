#!/usr/bin/env python3
"""Build isolated attention experiments; never writes production sources or uses a GPU."""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "TensorSharp.Backends.Cuda/native/kernels/tensorsharp_q35_kernels.cu"


def block(text: str, marker: str) -> str:
    """Extract one balanced C++ block, ignoring comments and string literals."""
    start = text.index(marker)
    opening = text.index("{", start)
    depth, i, state = 0, opening, "code"
    while i < len(text):
        char, pair = text[i], text[i:i + 2]
        if state == "line":
            if char == "\n": state = "code"
        elif state == "comment":
            if pair == "*/": state, i = "code", i + 1
        elif state in ('"', "'"):
            if char == "\\": i += 1
            elif char == state: state = "code"
        elif pair == "//": state, i = "line", i + 1
        elif pair == "/*": state, i = "comment", i + 1
        elif char in ('"', "'"): state = char
        elif char == "{": depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                if end < len(text) and text[end] == ";": end += 1
                return text[start:end]
        i += 1
    raise ValueError(f"Unbalanced block: {marker}")


def once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one source anchor: {old[:100]!r}")
    return text.replace(old, new, 1)


def generate(source: str, variant: int) -> str:
    if variant not in range(5): raise ValueError("variant must be0..4")
    for name, value in (("TS_Q35_HD",256),("TS_Q35_KV_BLOCK",256),("TS_Q35_Q8_ROW",272),
                        ("TS_Q35_Q8_SCALES",256),("TS_Q35_AI_KT",32),("TS_Q35_AI_ROWS",128),("TS_Q35_AH_CHAIN",1)):
        if not re.search(rf"^#define\s+{name}\s+{value}\b", source, re.MULTILINE):
            raise ValueError(f"Production constant changed: {name}; review experiment before building")
    if not re.search(r"^#define\s+TS_Q35_ATTN_ROW\s+\(TS_Q35_HD\s*\+\s*8\)", source, re.MULTILINE):
        raise ValueError("Production attention row stride changed")
    raw = 2 if variant in (0, 2) else 1
    half_bytes = 32768 if variant == 4 else 33792
    row_stride = "TS_Q35_HD" if variant == 4 else "(TS_Q35_HD+8)"
    pieces = ["#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <stdint.h>\n",
              f"#define TS_Q35_HD 256\n#define TS_Q35_KV_BLOCK 256\n#define TS_Q35_Q8_ROW 272\n#define TS_Q35_Q8_SCALES 256\n#define TS_Q35_ATTN_ROW {row_stride}\n#define TS_BENCH_COMPACT_SWIZZLE {int(variant == 4)}\n"]
    markers = [
        "__device__ __forceinline__ float ts_q35_sigmoid(",
        "__device__ __forceinline__ uint32_t ts_q35_smem_u32(",
        "__device__ __forceinline__ void ts_q35_cp16(",
        "__device__ __forceinline__ void ts_q35_cp_commit(",
        "template <int N>\n__device__ __forceinline__ void ts_q35_cp_wait(",
        "__device__ __forceinline__ void ts_q35_ldsm_x4(",
        "__device__ __forceinline__ uint32_t ts_q35_pack_h2(",
        "__device__ __forceinline__ void ts_q35_mma_f16h(",
        "__device__ __forceinline__ void ts_q35_ldsm_x4_t(",
        "__device__ __forceinline__ void ts_q35_i8x4_to_h2(",
    ]
    pieces.extend(block(source, m) + "\n" for m in markers)
    pieces += ["#define TS_Q35_AI_KT 32\n#define TS_Q35_AI_TILE (32*TS_Q35_Q8_ROW)\n#define TS_Q35_AI_ROWS 128\n#define TS_Q35_AH_CHAIN 1\n",
               f"#define TS_BENCH_RAW_STAGES {raw}\n"]
    struct = block(source, "struct TsQ35AttnPrefill")
    struct = once(struct, "raw[2][2]", f"raw[{raw}][2]")
    pieces += [struct + "\n", f"static_assert(sizeof(TsQ35AttnPrefill)=={half_bytes + raw * 17408}, \"shared layout changed\");\n",
               block(source, "__device__ __forceinline__ void ts_q35_tile_to_h(") + "\n"]
    if variant < 2:
        kernel = block(source, 'extern "C" __global__ void __launch_bounds__(256) ts_q35_attn_prefill_f32(')
        kernel = once(kernel, "ts_q35_attn_prefill_f32(", "ts_bench_attn_prefill(")
        if raw == 1:
            kernel = once(kernel, "const int st = t & 1;", "const int st = 0;")
            prefetch = "        if (first + TS_Q35_AI_KT < kHi)\n            load(st ^ 1, first + TS_Q35_AI_KT);\n        ts_q35_cp_commit();\n"
            kernel = once(kernel, prefetch, "")
            barrier = "        __syncthreads();   // this tile's K and V are in F16\n"
            kernel = once(kernel, barrier, barrier + "        // Every raw reader is finished. Next DMA may overwrite the same bank.\n" + prefetch.replace("st ^ 1", "0"))
        pieces.append(kernel + "\n")
    else:
        pieces.append((HERE / "attention64.cuh").read_text())
    pieces.append(block(source, 'extern "C" __global__ void ts_q35_attn_combine_f32(')
                  .replace("ts_q35_attn_combine_f32(", "ts_bench_attn_combine(", 1) + "\n")
    result = "\n".join(pieces)
    # No hidden static shared allocation is allowed at the exact50KiB boundary.
    if len(re.findall(r"\b__shared__\b", result)) != 1 or "extern __shared__" not in result:
        raise ValueError("Unexpected static shared allocation in standalone source")
    return result


def parse_resources(log: str, variant: int) -> dict:
    entry = re.search(r"Compiling entry function ['\"]ts_bench_attn_prefill['\"][\s\S]*?(?=Compiling entry function|\Z)", log)
    if entry is None: raise ValueError("Missing ptxas resource record for attention entry")
    body = entry.group()
    regs = re.search(r"Used (\d+) registers", body)
    stack = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
    if not regs or not stack: raise ValueError("Incomplete ptxas resource record; refusing an unverified64-query branch")
    static = re.search(r"(\d+) bytes smem", body)
    raw = 2 if variant in (0, 2) else 1
    result = dict(registers_per_thread=int(regs[1]), stack_bytes=int(stack[1]),
                  spill_store_bytes=int(stack[2]), spill_load_bytes=int(stack[3]),
                  static_shared_bytes=int(static[1]) if static else 0,
                  dynamic_shared_bytes=(32768 if variant == 4 else 33792) + raw * 17408,
                  static_shared_evidence="ptxas plus generated source has no static shared objects")
    result["total_shared_bytes"] = result["dynamic_shared_bytes"] + result["static_shared_bytes"]
    result["no_local_or_spills"] = not any(result[k] for k in ("stack_bytes", "spill_store_bytes", "spill_load_bytes"))
    result["two_cta_resource_gate"] = result["registers_per_thread"] <= 128 and result["total_shared_bytes"] <= 51200 and result["no_local_or_spills"]
    result["gpu_gate_permitted"] = variant < 2 or result["two_cta_resource_gate"]
    result["status"] = "resources_qualified_not_gpu_validated" if result["gpu_gate_permitted"] else "resource_rejected_no_gpu_run"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--arch", default="sm_120a")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--variants", default="0,1,2,3", help="Optional compact+swizzle variant4; use0,1 to explicitly narrow scope")
    args = parser.parse_args()
    try: requested = sorted(set(int(v) for v in args.variants.split(",")))
    except ValueError: parser.error("--variants must be comma-separated0..4")
    if not requested or requested[0] != 0 or any(v not in range(5) for v in requested): parser.error("--variants must include baseline0 and contain only0..4")
    out = args.out.resolve()
    # Generated evidence must be ignored, as required by AGENTS.md.
    check = subprocess.run(["git", "check-ignore", str(out / "manifest.json")], cwd=ROOT, capture_output=True)
    if check.returncode: parser.error("--out must be ignored artifacts/ or docs/validation/ storage")
    out.mkdir(parents=True, exist_ok=True)
    (out / "qualified-variants.txt").unlink(missing_ok=True)  # A failed/re-generated build must not authorize stale cubins.
    source = SOURCE.read_text()
    manifest = dict(complete=False, source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    prototype_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/"build.py",HERE/"bench.cu",HERE/"attention64.cuh",HERE/"checks.h")},
                    arch=args.arch, flags=["-O3", "--extra-device-vectorization", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"],
                    gpu_runs=0, hardware_counters="NOT_RUN", variants_requested=requested, variants=[])
    save = lambda: (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    failed = False
    for variant in requested:
        path = out / f"variant-{variant}.cu"
        path.write_text(generate(source, variant))
        info = dict(variant=variant, query_rows=128 if variant < 2 else 64, raw_banks=2 if variant in (0, 2) else 1,
                    compact_swizzle=variant == 4,
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), status="generated_not_compiled")
        manifest["variants"].append(info)
        if args.generate_only: continue
        command = [args.nvcc, "-std=c++17", "-cubin", f"-arch={args.arch}", *manifest["flags"],
                   "-Xptxas=-v,-warn-spills", str(path), "-o", str(out / f"variant-{variant}.cubin")]
        try: result = subprocess.run(command, capture_output=True, text=True)
        except OSError as error:
            info.update(command=command,exit_code=None,status="compiler_unavailable_not_run",error=str(error))
            (out / f"variant-{variant}.build.log").write_text(str(error)+"\n")
            failed=True;save();break
        log = result.stdout + result.stderr
        (out / f"variant-{variant}.build.log").write_text(log)
        info.update(command=command, exit_code=result.returncode)
        if result.returncode:
            info["status"] = "compile_failed_not_validated"; failed = True
        else:
            try:
                info.update(parse_resources(log, variant))
                info["cubin_sha256"] = hashlib.sha256((out / f"variant-{variant}.cubin").read_bytes()).hexdigest()
            except ValueError as error: info.update(status="resources_unverified_no_gpu_run", error=str(error)); failed = True
        save()
    if not args.generate_only and not failed:
        command = [args.nvcc, "-std=c++17", "-O3", str(HERE / "bench.cu"), "-lcuda", "-o", str(out / "attention-bench")]
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            (out / "runner.build.log").write_text(result.stdout + result.stderr)
            manifest["runner_command"], manifest["runner_exit_code"] = command, result.returncode
            failed |= result.returncode != 0
            if result.returncode == 0: manifest["runner_sha256"] = hashlib.sha256((out / "attention-bench").read_bytes()).hexdigest()
        except OSError as error:
            manifest.update(runner_command=command,runner_exit_code=None,runner_error=str(error))
            (out / "runner.build.log").write_text(str(error)+"\n");failed=True
    manifest["complete"] = not failed
    manifest["validation"] = "source_generation_only" if args.generate_only else "offline_compile_resources_only"
    save()
    # Publish only AFTER all native binaries and the final complete manifest.
    # A failed runner build must not authorize a stale executable from an older run.
    if not args.generate_only and not failed:
        marker=out/"qualified-variants.tmp"
        marker.write_text("".join(f"{v['variant']}\n" for v in manifest["variants"] if v.get("gpu_gate_permitted")))
        marker.replace(out/"qualified-variants.txt")
    print(json.dumps(manifest, indent=2))
    return 1 if failed else 0


if __name__ == "__main__": sys.exit(main())
