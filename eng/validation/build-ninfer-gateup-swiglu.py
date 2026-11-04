#!/usr/bin/env python3
"""Compile the owned main module for one exact BN8 GateUp/SwiGLU component."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

SYMBOL = "ts_q35_gemm_w4a4_8_swiglu_bf16"
POLICY = "ts-ninfer-gateup-bn8-s2-s1-f32-expf-bf16-swiglu-v1"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inventory(ptx):
    result = {}
    for name, parameters in re.findall(r"\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)", ptx, re.S):
        need(name not in result, "Duplicate PTX entry")
        declarations = [re.sub(r"\s+", " ", p.strip()) for p in parameters.split(",")]
        result[name] = {"symbol": name, "parameterDeclarations": declarations}
    need(result, "No exported PTX entries")
    return result


def resources(log, name):
    match = re.search(r"Function properties for " + re.escape(name)
                     + r"\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
                       r"\nptxas info\s*: Used (\d+) registers([^\n]*)", log)
    if not match:
        return None
    static = re.search(r"(\d+) bytes smem", match[5])
    barriers = re.search(r"used (\d+) barriers", match[5])
    return dict(stackBytes=int(match[1]), spillStoreBytes=int(match[2]), spillLoadBytes=int(match[3]),
                registers=int(match[4]), staticSharedBytes=int(static[1]) if static else 0,
                barriers=int(barriers[1]) if barriers else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--baseline-ptx", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    include = root / "TensorSharp.Backends.Cuda/native/kernels"
    source = include / "tensorsharp_q35_kernels.cu"
    helper_path = root / "eng/validation/NinferA4GdnCoreProbe/native/build.py"
    spec = importlib.util.spec_from_file_location("a4_abi_contract", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    paths = [source] + [include / n for n in ("tensorsharp_q35_gdn_prepared.cuh", "tensorsharp_q35_dflash.cuh", "tensorsharp_q35_topk.cuh")]
    inputs = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())} for p in paths]
    helper_native = root / "TensorSharp.Backends.Cuda/native/ninfer/tensorsharp_q35_ninfer.cu"
    def silu(text, name):
        return re.sub(r"\s+", "", re.search(r"float " + name + r"\(float x\)\s*\{([^}]+)\}", text)[1])
    need(silu(source.read_text(), "ts_q35_silu") == silu(helper_native.read_text(), "ts_ni_silu"), "SiLU source arithmetic differs")
    seen = set()
    for block in range(272):
        for tid in range(256):
            gate_row = block * 128 + (tid // 32) * 16 + (tid % 32) // 4
            index = gate_row // 16 * 8 + gate_row % 16
            need((index // 8) * 16 + index % 8 == gate_row, "Gate/up row mapping differs")
            for e in range(2):
                token = 2 * (tid % 4) + e
                need((token, index) not in seen, "Duplicate SwiGLU output")
                seen.add((token, index))
    need(len(seen) == 8 * 17408, "Compact output coverage differs")
    result = dict(schema="ninfer-gateup-swiglu-build/1", complete=False, arithmeticPolicy=POLICY,
                  sources=inputs, builderSha256=sha(Path(__file__).read_bytes()),
                  abiHelperSha256=sha(helper_path.read_bytes()), swigluHelperSourceSha256=sha(helper_native.read_bytes()),
                  CPU=dict(siluSourceEqual=True, uniqueCompactOutputs=len(seen), GPU="NOT_RUN"),
                  fixedTrial=dict(rows=8, outputRows=34816, columns=5120, intermediate=17408,
                                  splits=1, grid=[272, 1, 1], block=[256, 1, 1], dynamicSharedBytes=43520,
                                  outputF32Bytes=557056, originalProjectionF32Bytes=1114112),
                  GPU="NOT_RUN", performance="NOT_RUN", modelQuality="NOT_RUN", servingIntegration="NOT_PRESENT")
    if args.verify_only:
        print(json.dumps(result, indent=2))
        return
    need(args.out is not None and args.out.is_absolute() and not args.out.exists(), "Fresh absolute output directory required")
    compiler = shutil.which(args.nvcc)
    need(compiler is not None, "NVCC not found")
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=True)
    args.out.mkdir(parents=True)
    result.update(nvccVersion=version.stdout + version.stderr, commands=[])
    stem = "tensorsharp_q35_kernels"
    try:
        for kind in ("ptx", "fatbin"):
            target = args.out / (stem + "." + kind)
            command = [compiler, "-std=c++17", "-O3", "--ftz=false", "--fmad=true", "-arch=sm_120a",
                       "-I" + str(include), "--" + kind, str(source), "-o", str(target)]
            if kind == "fatbin":
                command += ["-Xptxas=-v"]
            run = subprocess.run(command, capture_output=True, text=True, timeout=600)
            (args.out / (kind + ".stdout.log")).write_text(run.stdout)
            (args.out / (kind + ".stderr.log")).write_text(run.stderr)
            result["commands"].append({"argv": command, "exit": run.returncode})
            need(run.returncode == 0, "NVCC failed: " + kind)
            result[kind] = {"path": target.name, "bytes": target.stat().st_size, "sha256": sha(target.read_bytes())}
        ptx = (args.out / (stem + ".ptx")).read_text()
        entries = inventory(ptx)
        result["entries"] = [helper.abi(ptx, s) for s in (SYMBOL, "ts_q35_gemm_w4a4_8", "ts_q35_gemm_w4a4_32")]
        log = (args.out / "fatbin.stderr.log").read_text()
        for name in entries:
            entries[name]["resources"] = resources(log, name)
        need(entries[SYMBOL]["resources"] is not None, "Candidate ptxas resource record missing")
        result["entryInventory"] = list(entries.values())
        if args.baseline_ptx is not None:
            before = inventory(args.baseline_ptx.read_text())
            need(set(entries) == set(before) | {SYMBOL}, "Old exported entry inventory differs")
            need(all(entries[n]["parameterDeclarations"] == before[n]["parameterDeclarations"] for n in before), "Old entry parameter ABI changed")
            result["oldEntryAbiComparison"] = dict(passed=True, baselineSha256=sha(args.baseline_ptx.read_bytes()), oldEntries=len(before))
        need(all(sha((root / x["path"]).read_bytes()) == x["sha256"] for x in inputs), "Source changed during build")
        (args.out / (stem + ".fatbin.sha256")).write_text(result["ptx"]["sha256"] + "\n")
        result["complete"] = True
    finally:
        (args.out / ("completed-build.json" if result["complete"] else "failed-build.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
