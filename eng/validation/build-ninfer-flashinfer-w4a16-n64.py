#!/usr/bin/env python3
"""Separate N64/F32 AOT trial; leave the qualified N128 tools and images unchanged."""
import argparse
import datetime
import hashlib
import importlib.util
import importlib
import json
import os
import pathlib
import sys
import traceback
import types

REVISION = "61bc8b6dd7049a44d614b2a0cd4a036cb3c46ad0"
POLICY = "flashinfer61bc8-n64-raw-nv4-bf16-mma-f32-alpha-post-s1-w4-k128-st2-v1"
HELPER_SHA256 = "cdc3e1552e56c7569190008e8c354ac312bd8a2afd104d343582244de20b204b"
N, K = 34816, 5120


def digest(data):
    return hashlib.sha256(data).hexdigest()


def helpers():
    path = pathlib.Path(__file__).with_name("build-ninfer-flashinfer-w4a16.py")
    if digest(path.read_bytes()) != HELPER_SHA256:
        raise ValueError("The pinned TensorSharp-owned AOT parsing/artifact helper changed")
    spec = importlib.util.spec_from_file_location("tensorsharp_fi_frozen_aot_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shared_bytes(rows):
    return 2 * (min(rows + 1, 16) * (128 // 2 + 4) * 4 + 64 * (128 // 8 + 4) * 4 + (128 // 64) * 512)


def verify(folder):
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    expected = {"flashinfer/gemm/kernels/native_bf16_fp4/n64_staged_kernel.py",
                "flashinfer/gemm/kernels/native_bf16_fp4/staged_kernel.py",
                "flashinfer/gemm/kernels/native_bf16_fp4/kernel.py", "flashinfer/cute_dsl/fp4_common.py", "LICENSE"}
    if manifest["revision"] != REVISION or manifest["modifiedUpstreamFiles"] or {x["path"] for x in manifest["files"]} != expected:
        raise ValueError("Unexpected original N64 source contract")
    for item in manifest["files"]:
        data = (folder / item["path"]).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if len(data) != item["bytes"] or digest(data) != item["sha256"] or blob != item["gitBlobSha1"]:
            raise ValueError("Pristine original N64 dependency differs: " + item["path"])
    text = (folder / "flashinfer/gemm/kernels/native_bf16_fp4/n64_staged_kernel.py").read_text()
    for token in ("a_rows = min(m + 1, 16)", "self.n_tiles = 64 // (warps * 8)", "out[mi, ni] = out.element_type(value)",
                  "value = value * alpha[0]", "(bn // 2) * (k // 64) * 512", "scale_row = ni + (bn % 2) * 64"):
        if token not in text:
            raise ValueError("The pinned N64 layout/epilogue contract changed")
    if shared_bytes(1) != 13376 or shared_bytes(8) != 17184:
        raise ValueError("Unexpected static N64 shared allocation")
    return {"revision": REVISION, "originalFilesVerified": 4, "licenseFilesVerified": 1,
            "manifestSha256": digest(path.read_bytes()), "modifiedUpstreamFiles": []}


def compile_one(rows, folder, output, common):
    import cutlass
    import cutlass.cute as cute
    alias = "tensorsharp_flashinfer_n64_61bc8b6d"
    package = types.ModuleType(alias)
    package.__path__ = [str(folder / "flashinfer")]
    sys.modules[alias] = package
    upstream = importlib.import_module(alias + ".gemm.kernels.native_bf16_fp4.n64_staged_kernel")
    if not upstream._DIRECT_BF16_CVT:
        raise RuntimeError("Direct native FP4/E4M3-to-BF16 conversion requires CUDA >=13.2")
    fake = cute.runtime.make_fake_compact_tensor
    a = fake(cutlass.Int32, (rows, K // 2), stride_order=(1, 0), assumed_align=16)
    b = fake(cutlass.Int32, (N, K // 8), stride_order=(1, 0), assumed_align=16)
    sf = fake(cutlass.Float8E4M3FN, (N // 128 * (K // 64) * 512,), assumed_align=16)
    alpha = fake(cutlass.Float32, (1,), assumed_align=4)
    out = fake(cutlass.Float32, (rows, N), stride_order=(1, 0), assumed_align=4)
    stream = cute.runtime.make_fake_stream()
    op = upstream.NativeN64Bf16Fp4Kernel(splits=1, warps=4, enable_pdl=False, tile_k=128, stages=2)
    compiled = cute.compile(op, a, b, sf, alpha, out, None, stream, options="--gpu-arch sm_120a")
    cubin, ptx = compiled.__cubin__, compiled.__ptx__
    if not isinstance(cubin, bytes) or cubin[:4] != b"\x7fELF" or not isinstance(ptx, str):
        raise RuntimeError("Standalone N64 CUBIN and PTX artifacts were not retained")
    name, parameters = common.parse_entry(ptx)
    stem = "ninfer_fi_native_w4a16_n64_t" + str(rows)
    common.write_new(output / (stem + ".cubin"), cubin)
    common.write_new(output / (stem + ".ptx"), ptx.encode())
    ir_module = getattr(compiled, "ir_module", None)
    if ir_module is not None:
        common.write_new(output / (stem + ".host.mlir"), str(ir_module).encode())
    return {"rows": rows, "outputRows": N, "columns": K, "cubin": stem + ".cubin", "cubinSha256": digest(cubin),
            "ptx": stem + ".ptx", "ptxSha256": digest(ptx.encode()), "kernel": name, "parameters": parameters,
            "argumentOrder": ["inputBf16", "originalCodes", "originalScales", "alphaF32", "outputF32"],
            "grid": [N // 64, 1, 1], "block": [128, 1, 1], "dynamicSmemBytes": shared_bytes(rows),
            "dynamicSmemBasis": "pinned three N64 SmemAllocator allocations, live rows min(M+1,16); inspect emitted host launch metadata before interpreting performance",
            "inputBytes": rows * K * 2, "codeBytes": N * K // 2, "scaleBytes": N * K // 16,
            "alphaBytes": 4, "outputBytes": rows * N * 4, "compilerKernelInfo": repr(getattr(compiled, "kernel_info", None))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    folder = root / "TensorSharp.Backends.Cuda/native/ninfer/flashinfer_w4a16_n64/upstream"
    common = helpers()
    verification = verify(folder)
    check = common.self_test()
    if args.verify_only:
        print(json.dumps({"sourceVerification": verification, "helperSha256": HELPER_SHA256, "deviceAbiParser": check["deviceAbiParser"],
                          "sourceSharedBytesT1": shared_bytes(1), "sourceSharedBytesT8": shared_bytes(8), "nativeCompilation": "NOT_RUN", "GPU": "NOT_RUN"}, indent=2))
        return
    if args.out is None:
        parser.error("--out required unless --verify-only")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CUTE_DSL_KEEP"] = "ir,ptx,cubin"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(output / "compiler-dump")
    os.environ["CUTE_DSL_ARCH"] = "sm_120a"
    result = {"schema": "tensorsharp-ninfer-flashinfer-a16-n64-aot/1", "arithmeticPolicy": POLICY,
              "createdUtc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "sourceVerification": verification,
              "builderSha256": digest(pathlib.Path(__file__).read_bytes()), "pinnedHelperSha256": HELPER_SHA256,
              "stack": common.stack_versions(), "arch": "sm_120a", "splits": 1, "warps": 4, "tileK": 128,
              "stages": 2, "enablePdl": False, "outputDtype": "Float32", "specializations": [], "status": "building",
              "GPU": "NOT_RUN", "modelQuality": "NOT_RUN", "performance": "NOT_RUN"}
    try:
        for rows in (1, 8):
            result["specializations"].append(compile_one(rows, folder, output, common))
        result["status"] = "complete"
    except Exception:
        result["status"] = "failed"; result["error"] = traceback.format_exc()
        common.write_new(output / "failed-build.json", (json.dumps(result, indent=2) + "\n").encode())
        raise
    data = (json.dumps(result, indent=2) + "\n").encode()
    common.write_new(output / "module.json", data)
    common.write_new(output / "module.json.sha256", (digest(data) + "  module.json\n").encode())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
