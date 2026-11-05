#!/usr/bin/env python3
"""AOT original raw NVFP4/BF16 body kernels: four shapes, physical T1..8.

The emitted CUBIN/manifest runtime uses only the CUDA driver. Compilation does
not install packages, patch upstream sources, or enable a serving option.
"""
import argparse
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
import pathlib
import re
import sys
import traceback
import types

REVISION = "61bc8b6dd7049a44d614b2a0cd4a036cb3c46ad0"
SOURCE_MANIFEST_SHA256 = "1eaa7ccd7f7fb35ff917951c1910e8ebd6d42b423221768f2d00beed8149a68c"
SHAPES = (("gate-up", 34816, 5120), ("down", 5120, 17408), ("output", 5120, 6144), ("gqa-input", 14336, 5120))
POLICIES = {family: "flashinfer61bc8-body-raw-nv4-bf16-mma-f32-alpha-post-" + family + "-s1-w4-k128-st2-v1" for family in ("n128", "n64")}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    with path.open("xb") as output:
        output.write(data); output.flush(); os.fsync(output.fileno())


def shared_bytes(family, rows):
    if not 1 <= rows <= 8 or family not in POLICIES:
        raise ValueError("Unsupported family or physical rows")
    tile_n = 128 if family == "n128" else 64
    live = 16 if family == "n128" else min(rows + 1, 16)
    return 2 * (live * 68 * 4 + tile_n * 20 * 4 + 1024)


def verify_sources(folder):
    path = folder / "manifest.json"
    raw = path.read_bytes(); manifest = json.loads(raw)
    expected = {"flashinfer/gemm/kernels/native_bf16_fp4/n64_staged_kernel.py", "flashinfer/gemm/kernels/native_bf16_fp4/staged_kernel.py",
                "flashinfer/gemm/kernels/native_bf16_fp4/kernel.py", "flashinfer/cute_dsl/fp4_common.py", "LICENSE"}
    if sha(raw) != SOURCE_MANIFEST_SHA256 or manifest["revision"] != REVISION or manifest["modifiedUpstreamFiles"] or {x["path"] for x in manifest["files"]} != expected:
        raise ValueError("Pinned original body source contract differs")
    for item in manifest["files"]:
        data = (folder / item["path"]).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if len(data) != item["bytes"] or sha(data) != item["sha256"] or blob != item["gitBlobSha1"]:
            raise ValueError("Original source changed: " + item["path"])
    return {"revision": REVISION, "manifestSha256": sha(raw), "originalFilesVerified": 4, "licenseFilesVerified": 1, "modifiedUpstreamFiles": []}


def parse_entry(ptx):
    entries = re.findall(r"(?:\.visible\s+)?\.entry\s+([A-Za-z0-9_.$]+)\s*\((.*?)\)", ptx, re.S)
    if len(entries) != 1:
        raise ValueError("Each single-split body image must contain one kernel")
    name, text = entries[0]; parameters = []
    for index, declaration in enumerate(text.split(",")):
        declaration = re.sub(r"//[^\n]*", "", declaration).strip()
        scalar = re.fullmatch(r"\.param\s+\.(u64|b64)\s+([A-Za-z0-9_.$]+)", declaration)
        array = re.fullmatch(r"\.param\s+\.align\s+8\s+\.b8\s+([A-Za-z0-9_.$]+)\[8\]", declaration)
        if not scalar and not array:
            raise ValueError("Unknown static body device ABI: " + declaration)
        parameters.append({"index": index, "name": scalar.group(2) if scalar else array.group(1), "offset": index * 8, "size": 8})
    if len(parameters) != 5:
        raise ValueError("Expected precisely A/B/scales/alpha/output device pointers")
    return name, parameters


def stack_versions():
    result = {"python": sys.version, "platform": sys.platform}
    for dist in importlib.metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name in ("torch", "cuda-bindings", "apache-tvm-ffi") or name.startswith("nvidia-cutlass-dsl"):
            result[name] = {"version": dist.version, "recordSha256": sha((dist.read_text("RECORD") or "").encode())}
    return result


def compile_one(family, shape, n, k, rows, folder, output):
    import cutlass
    import cutlass.cute as cute
    alias = "tensorsharp_flashinfer_body_61bc8b6d"
    if alias not in sys.modules:
        package = types.ModuleType(alias); package.__path__ = [str(folder / "flashinfer")]; sys.modules[alias] = package
    suffix = "staged_kernel" if family == "n128" else "n64_staged_kernel"
    upstream = importlib.import_module(alias + ".gemm.kernels.native_bf16_fp4." + suffix)
    if not upstream._DIRECT_BF16_CVT:
        raise RuntimeError("The body contract requires CUDA >=13.2 native FP4/E4M3-to-BF16 conversion")
    fake = cute.runtime.make_fake_compact_tensor
    a = fake(cutlass.Int32, (rows, k // 2), stride_order=(1, 0), assumed_align=16)
    b = fake(cutlass.Int32, (n, k // 8), stride_order=(1, 0), assumed_align=16)
    scales = fake(cutlass.Float8E4M3FN, (n // 128 * (k // 64) * 512,), assumed_align=16)
    alpha = fake(cutlass.Float32, (1,), assumed_align=4)
    out = fake(cutlass.Float32, (rows, n), stride_order=(1, 0), assumed_align=4)
    stream = cute.runtime.make_fake_stream()
    cls = upstream.NativeBf16Fp4StagedKernel if family == "n128" else upstream.NativeN64Bf16Fp4Kernel
    op = cls(splits=1, warps=4, enable_pdl=False, tile_k=128, stages=2)
    compiled = cute.compile(op, a, b, scales, alpha, out, None, stream, options="--gpu-arch sm_120a")
    cubin, ptx = compiled.__cubin__, compiled.__ptx__
    if not isinstance(cubin, bytes) or cubin[:4] != b"\x7fELF" or not isinstance(ptx, str):
        raise RuntimeError("A standalone binary CUBIN and text PTX are mandatory")
    kernel, parameters = parse_entry(ptx)
    stem = "fi_body_" + family + "_" + shape + "_n" + str(n) + "_k" + str(k) + "_t" + str(rows)
    write_new(output / (stem + ".cubin"), cubin); write_new(output / (stem + ".ptx"), ptx.encode())
    ir_module = getattr(compiled, "ir_module", None)
    if ir_module is not None:
        write_new(output / (stem + ".host.mlir"), str(ir_module).encode())
    return {"shape": shape, "outputRows": n, "columns": k, "rows": rows, "cubin": stem + ".cubin", "cubinSha256": sha(cubin),
            "ptx": stem + ".ptx", "ptxSha256": sha(ptx.encode()), "kernel": kernel, "parameters": parameters,
            "argumentOrder": ["inputBf16", "originalCodes", "originalScales", "alphaF32", "outputF32"],
            "grid": [n // (128 if family == "n128" else 64), 1, 1], "block": [128, 1, 1], "dynamicSmemBytes": shared_bytes(family, rows),
            "dynamicSmemBasis": "all three unchanged source SmemAllocator allocations including alignment; emitted host launch metadata retained for actual verification",
            "codeBytes": n * k // 2, "scaleBytes": n * k // 16, "inputBytes": rows * k * 2, "outputBytes": rows * n * 4, "alphaBytes": 4,
            "compilerKernelInfo": repr(getattr(compiled, "kernel_info", None))}


def image_set_sha(policy, specs):
    return sha((policy + "\n" + "".join(str(x["outputRows"]) + "x" + str(x["columns"]) + "/t" + str(x["rows"]) + "=" + x["cubinSha256"].lower() + "\n" for x in specs)).encode())


def self_test():
    entries = [(n, k, t) for _, n, k in SHAPES for t in range(1, 9)]
    assert len(entries) == len(set(entries)) == 32
    assert all(n % 128 == k % 128 == 0 and n * k < 2 ** 31 for n, k, _ in entries)
    assert max(n * k // 2 + n * k // 16 for n, k, _ in entries) == 100270080
    assert shared_bytes("n128", 1) == shared_bytes("n128", 8) == 31232
    assert shared_bytes("n64", 1) == 13376 and shared_bytes("n64", 8) == 17184
    name, parameters = parse_entry(".entry k(" + ",".join(".param .u64 p" + str(i) for i in range(5)) + ") {}")
    assert name == "k" and [x["offset"] for x in parameters] == [0, 8, 16, 24, 32]
    for bad in ("", ".entry k(.param .u32 p){}", ".entry k(.param .u64 p){}"):
        try:
            parse_entry(bad)
        except ValueError:
            continue
        raise AssertionError("Unknown body device ABI admitted")
    return {"uniqueSpecializations": 32, "deviceAbiParser": "PASS", "geometryAndSourceSharedReservations": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(POLICIES), default="n128")
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    folder = root / "TensorSharp.Backends.Cuda/native/ninfer/flashinfer_w4a16_n64/upstream"
    verification = verify_sources(folder); check = self_test()
    if args.verify_only:
        print(json.dumps({"sourceVerification": verification, "selfTest": check, "family": args.family, "nativeCompile": "NOT_RUN", "GPU": "NOT_RUN"}, indent=2)); return
    if args.out is None:
        parser.error("--out required unless --verify-only")
    output = args.out.resolve(); output.mkdir(parents=True, exist_ok=False)
    os.environ["CUTE_DSL_KEEP"] = "ir,ptx,cubin"; os.environ["CUTE_DSL_DUMP_DIR"] = str(output / "compiler-dump"); os.environ["CUTE_DSL_ARCH"] = "sm_120a"
    result = {"schema": "tensorsharp-ninfer-flashinfer-body-aot/1", "family": args.family, "arithmeticPolicy": POLICIES[args.family],
              "createdUtc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "sourceVerification": verification,
              "builderSha256": sha(pathlib.Path(__file__).read_bytes()), "selfTest": check, "stack": stack_versions(),
              "arch": "sm_120a", "splits": 1, "warps": 4, "tileK": 128, "stages": 2, "enablePdl": False, "outputDtype": "Float32",
              "specializations": [], "status": "building", "GPU": "NOT_RUN", "wholeModelQuality": "NOT_RUN", "performance": "NOT_RUN"}
    try:
        for shape, n, k in SHAPES:
            for rows in range(1, 9):
                result["specializations"].append(compile_one(args.family, shape, n, k, rows, folder, output))
        result["imageSetSha256"] = image_set_sha(result["arithmeticPolicy"], result["specializations"])
        result["status"] = "complete"
    except Exception:
        result["status"] = "failed"; result["error"] = traceback.format_exc()
        write_new(output / "failed-build.json", (json.dumps(result, indent=2) + "\n").encode()); raise
    data = (json.dumps(result, indent=2) + "\n").encode()
    write_new(output / "module.json", data); write_new(output / "module.json.sha256", (sha(data) + "  module.json\n").encode())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
