#!/usr/bin/env python3
"""Isolated exact F16 KV intermediate experiment. Offline build only; no CUDA init."""
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
FROZEN = HERE.with_name("Qwen35AttentionBench")
SPEC = importlib.util.spec_from_file_location("sealed_attention_for_f16_intermediate", FROZEN / "build.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
SOURCE = BASE.SOURCE
FLAGS = ["-O3", "--extra-device-vectorization", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"]
ORACLE = "ts_bench_attn_prefill"
CANDIDATE = "ts_bench_attn_f16_prefill"
NO_SECOND_BARRIER = "ts_bench_attn_f16_no_second_barrier"
SYMBOLS = (ORACLE, CANDIDATE, NO_SECOND_BARRIER, "ts_bench_decode_plain", "ts_bench_decode_reference", "ts_bench_attn_combine")
PERMIT = "predecoded-attention-permitted.json"
RUNNER = "predecoded-attention-bench"
TOOLS = ("build.py", "run.py", "bench.cu", "predecode.cuh", "half_checks.h")
ARTIFACTS = ("kernels.cu", "kernels.cubin", "kernels.ptx", RUNNER)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def workspace_bytes(keys, heads=4, sequences=1, stride=256):
    if any(type(x) is not int or x <= 0 for x in (keys, heads, sequences, stride)) or stride not in (256, 264):
        raise ValueError("positive integer dimensions and exact256/264 half stride required")
    return sequences * 2 * heads * keys * stride * 2


def generate(source):
    original = BASE.generate(source, 0) # Exact sealed baseline0 remains the whole prefix.
    helper = BASE.block(source, "__device__ __forceinline__ void ts_q35_tile_to_h(")
    plain = BASE.once(helper, "ts_q35_tile_to_h(", "ts_bench_tile_to_plain(")
    plain = BASE.once(plain, "dst + key * TS_Q35_ATTN_ROW + b * 32", "dst + key * TS_Q35_HD + b * 32")
    kernel = BASE.block(original, 'extern "C" __global__ void __launch_bounds__(256) ts_bench_attn_prefill(')
    candidate = BASE.once(kernel, ORACLE + "(", CANDIDATE + "(")
    candidate = BASE.once(candidate,
        "uint8_t* const* __restrict__ blockTable, int maxBlocks, long long kOff, long long vOff,",
        "const half* __restrict__ cache, int cacheKeys,")
    candidate = BASE.once(candidate,
        "TsQ35AttnPrefill& sm = *reinterpret_cast<TsQ35AttnPrefill*>(ts_q35_attn_smem);",
        "TsBenchHalfBanks& sm = *reinterpret_cast<TsBenchHalfBanks*>(ts_q35_attn_smem);")
    candidate = BASE.once(candidate,
        "uint8_t* const* blocks = blockTable + (long long)seq * maxBlocks;",
        "const half* cacheBase = cache + ((long long)seq * 2 * nKv + kvh) * cacheKeys * TS_Q35_HD;")
    old_load = BASE.block(candidate, "auto load = [&](int st, int first)")
    new_load = '''auto load = [&](int st, int first) {
        const int vectorsPerRow = TS_Q35_HD * 2 / 16;
        const int vectorsPerPlane = TS_Q35_AI_KT * vectorsPerRow;
        for (int i = tid; i < 2 * vectorsPerPlane; i += blockDim.x)
        {
            const int isV = i >= vectorsPerPlane, c = isV ? i - vectorsPerPlane : i;
            const int row = c / vectorsPerRow, col = (c % vectorsPerRow) * 8;
            const bool ok = first + row < kHi;
            half* dst = sm.kv[st][isV] + row * TS_Q35_ATTN_ROW + col;
            const half* src = cacheBase + (long long)isV * nKv * cacheKeys * TS_Q35_HD
                + (ok ? (long long)(first + row) * TS_Q35_HD + col : 0);
            ts_q35_cp16(dst, src, ok);
        }
    };'''
    candidate = BASE.once(candidate, old_load, new_load)
    candidate = BASE.once(candidate,
        "        ts_q35_tile_to_h(sm.raw[st][0], sm.k, tid);\n        ts_q35_tile_to_h(sm.raw[st][1], sm.v, tid);\n", "")
    candidate = BASE.once(candidate, "const half* kb = sm.k +", "const half* kb = sm.kv[st][0] +")
    candidate = BASE.once(candidate, "const half* vrow = sm.v +", "const half* vrow = sm.kv[st][1] +")
    fewer_barriers = BASE.once(candidate, CANDIDATE + "(", NO_SECOND_BARRIER + "(")
    fewer_barriers = BASE.once(fewer_barriers, "        __syncthreads();   // this tile's K and V are in F16\n", "")
    return original + "\n" + plain + "\n" + (HERE / "predecode.cuh").read_text() + "\n" + candidate + "\n" + fewer_barriers + "\n"


def parse_resources(log):
    result = {}
    for symbol in SYMBOLS:
        records = re.findall(rf"Compiling entry function ['\"]{re.escape(symbol)}['\"][\s\S]*?(?=Compiling entry function|\Z)", log)
        if len(records) != 1:
            raise ValueError("missing/duplicate actual ptxas record for " + symbol)
        body = records[0]
        regs = re.search(r"Used (\d+) registers", body)
        stack = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
        shared = re.search(r"(\d+) bytes smem", body)
        if not regs or not stack:
            raise ValueError("incomplete actual ptxas resources: " + symbol)
        result[symbol] = dict(registers_per_thread=int(regs[1]), stack_bytes=int(stack[1]),
            spill_store_bytes=int(stack[2]), spill_load_bytes=int(stack[3]),
            static_shared_bytes=int(shared[1]) if shared else 0,
            dynamic_shared_bytes=68608 if symbol == ORACLE else 67584 if symbol in (CANDIDATE,NO_SECOND_BARRIER) else 0,
            note="actual resources, not a2CTA/no-spill occupancy admission gate")
    return result


def evidence_dir(path):
    p = path.resolve()
    if not any(p.is_relative_to(ROOT / x) for x in ("artifacts", "docs/validation")):
        raise ValueError("new output must be inside ignored artifacts/ or docs/validation/")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--nvcc", default="nvcc")
    ap.add_argument("--arch", default="sm_120a")
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args(argv)
    try:
        out = evidence_dir(args.out)
        if out.exists(): raise ValueError("output exists; earlier evidence is never overwritten/resumed")
        if any(os.environ.get(n, "").strip() for n in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")):
            raise ValueError("NVCC_PREPEND_FLAGS/NVCC_APPEND_FLAGS must be unset")
    except ValueError as error: ap.error(str(error))
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="q35-predecoded-attention-build/1", complete=False,
        source=str(args.source.resolve()), tools={n:digest(HERE/n) for n in TOOLS},
        frozen_tools={n:digest(FROZEN/n) for n in ("build.py", "bench.cu", "checks.h", "attention64.cuh")},
        build_environment={n:os.environ.get(n) for n in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "CUDA_HOME", "CUDA_PATH")},
        arch=args.arch, flags=FLAGS, gpu_quality="NOT_RUN", gpu_timing="NOT_RUN", hardware_counters="NOT_RUN",
        cuda_compilation="NOT_RUN", production_integration=False,
        default_gpu_variants=[0,1], optional_gpu_variant=dict(id=2,flag="--with-no-second-barrier",default=False),
        workspace=dict(layout="[sequence][K/V][KVhead][maxKeys][256half]", current_layer_only=True,
            bytes_131072_4kv_1seq=workspace_bytes(131072), bytes_250000_4kv_1seq=workspace_bytes(250000),
            bytes_256000_4kv_1seq=workspace_bytes(256000),
            validation_reference="Temporary264half-stride legacy decode workspace only during strict validation; freed before attention/timing"),
        timing_scope="Every candidate call predecode+attention+split-combine; no reuse/preprocessing outside event interval",
        resource_policy="Record stack/spill/reg/shared facts; require actual driver launchability>=1CTA; NOT64-query occupancy experiment")
    save(out/"manifest.json", manifest)
    try:
        source = args.source.read_bytes()
        generated = generate(source.decode())
        (out/"kernels.cu").write_text(generated)
        manifest.update(source_sha256=hashlib.sha256(source).hexdigest(), generated_sha256=digest(out/"kernels.cu"),
            baseline0_source_sha256=hashlib.sha256(BASE.generate(source.decode(),0).encode()).hexdigest(), generation_complete=True)
        if args.generate_only:
            manifest["status"]="SOURCE_GENERATION_ONLY_NOT_GPU_VALIDATED";save(out/"manifest.json",manifest);print(json.dumps(manifest,indent=2));return 0
        version=subprocess.run([args.nvcc,"--version"],capture_output=True,text=True,check=True)
        manifest["nvcc_version"]=version.stdout+version.stderr
        common=[args.nvcc,"-std=c++17",f"-arch={args.arch}",*FLAGS]
        command=common+["-cubin","-Xptxas=-v,-warn-spills",str(out/"kernels.cu"),"-o",str(out/"kernels.cubin")]
        r=subprocess.run(command,capture_output=True,text=True);log=r.stdout+r.stderr;(out/"kernels.build.log").write_text(log)
        manifest.update(compile_command=command,compile_exit_code=r.returncode,cuda_compilation="RUN")
        if r.returncode:raise RuntimeError("native CUDA compilation failed")
        manifest["resources"]=parse_resources(log)
        command=common+["-ptx",str(out/"kernels.cu"),"-o",str(out/"kernels.ptx")]
        r=subprocess.run(command,capture_output=True,text=True);(out/"ptx.build.log").write_text(r.stdout+r.stderr)
        manifest.update(ptx_command=command,ptx_exit_code=r.returncode)
        if r.returncode:raise RuntimeError("PTX generation failed")
        command=[args.nvcc,"-std=c++17","-O3","-I",str(FROZEN),str(HERE/"bench.cu"),"-lcuda","-o",str(out/RUNNER)]
        r=subprocess.run(command,capture_output=True,text=True);(out/"runner.build.log").write_text(r.stdout+r.stderr)
        manifest.update(runner_command=command,runner_exit_code=r.returncode)
        if r.returncode:raise RuntimeError("runner build failed")
        if digest(args.source)!=manifest["source_sha256"] or manifest["tools"]!={n:digest(HERE/n) for n in TOOLS}:
            raise RuntimeError("source/tool changed during build")
        if manifest["frozen_tools"]!={n:digest(FROZEN/n) for n in manifest["frozen_tools"]}:
            raise RuntimeError("sealed fixture/helper changed during build")
        manifest.update(complete=True,status="OFFLINE_BUILD_ONLY_DRIVER_FULLBITS_AND_TIMING_NOT_RUN",artifacts={n:digest(out/n) for n in ARTIFACTS})
        save(out/"manifest.json",manifest)
        save(out/PERMIT,dict(schema="q35-predecoded-attention-permission/1",manifest_sha256=digest(out/"manifest.json"),artifacts=manifest["artifacts"]))
        print(json.dumps(manifest,indent=2));return 0
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        manifest.update(complete=False,status="BUILD_REJECTED_GPU_NOT_VALIDATED",error=str(error));save(out/"manifest.json",manifest);print(json.dumps(manifest,indent=2));return 1


if __name__=="__main__":raise SystemExit(main())
