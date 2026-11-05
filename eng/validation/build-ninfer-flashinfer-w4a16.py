#!/usr/bin/env python3
"""Build explicit driver-loadable CuTe DSL images from unchanged FlashInfer sources.

Build-time Python/CuTe/Torch dependencies are never used by the serving loader.
This prototype admits three explicit fixed tactics, no PDL, no split-K, F32 output.
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
POLICY = "flashinfer61bc8-native-raw-nv4-bf16-mma-f32-alpha-post-s1-w4-k128-st2-v1"
TACTICS = {
    "s1-w4-st2": (4, 2, POLICY),
    "s1-w8-st2": (8, 2, "flashinfer61bc8-native-raw-nv4-bf16-mma-f32-alpha-post-s1-w8-k128-st2-v1"),
    "s1-w4-st3": (4, 3, "flashinfer61bc8-native-raw-nv4-bf16-mma-f32-alpha-post-s1-w4-k128-st3-v1"),
}
N, K = 34816, 5120
SMEM = 2 * (16 * (128 // 2 + 4) * 4 + 128 * (128 // 8 + 4) * 4 + (128 // 64) * 512)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_sources(folder):
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["revision"] != REVISION or manifest["modifiedUpstreamFiles"]:
        raise ValueError("Unexpected upstream revision or modified source declaration")
    required = {"flashinfer/gemm/kernels/native_bf16_fp4/staged_kernel.py",
                "flashinfer/gemm/kernels/native_bf16_fp4/kernel.py", "flashinfer/cute_dsl/fp4_common.py", "LICENSE"}
    if {x["path"] for x in manifest["files"]} != required:
        raise ValueError("The pristine source manifest does not contain exactly the required sources and license")
    for item in manifest["files"]:
        data = (folder / item["path"]).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if len(data) != item["bytes"] or digest(data) != item["sha256"] or blob != item["gitBlobSha1"]:
            raise ValueError("Pristine source differs: " + item["path"])
    source = (folder / "flashinfer/gemm/kernels/native_bf16_fp4/staged_kernel.py").read_text()
    for text in ("class NativeBf16Fp4StagedKernel:", "out[mi, ni] = out.element_type(value)",
                 "value = value * alpha[0]", "self.splits > 1", "mma_bf16(", "cp.async.cg.shared.global"):
        if text not in source:
            raise ValueError("The static source contract no longer matches the pinned kernel")
    return {"revision": REVISION, "originalFilesVerified": 3, "licenseFilesVerified": 1,
            "manifestSha256": digest(manifest_path.read_bytes()), "modifiedUpstreamFiles": []}


def parse_entry(ptx):
    """Refuse unknown device ABIs rather than guessing CuTe's host-call ABI."""
    entries = re.findall(r"(?:\.visible\s+)?\.entry\s+([A-Za-z0-9_.$]+)\s*\((.*?)\)", ptx, re.S)
    if len(entries) != 1:
        raise ValueError("Expected precisely one CUDA kernel entry for the single-split specialization")
    name, text = entries[0]
    parameters = []
    for index, declaration in enumerate(text.split(",")):
        declaration = re.sub(r"//[^\n]*", "", declaration).strip()
        scalar = re.fullmatch(r"\.param\s+\.(u64|b64)\s+([A-Za-z0-9_.$]+)", declaration)
        array = re.fullmatch(r"\.param\s+\.align\s+8\s+\.b8\s+([A-Za-z0-9_.$]+)\[8\]", declaration)
        if not scalar and not array:
            raise ValueError("Unsupported static tensor device parameter: " + declaration)
        parameters.append({"index": index, "name": scalar.group(2) if scalar else array.group(1),
                           "offset": index * 8, "size": 8})
    if len(parameters) != 5:
        raise ValueError("Expected A, B, scale, alpha, output pointer arguments only")
    return name, parameters


def stack_versions():
    versions = {"python": sys.version, "platform": sys.platform}
    for dist in importlib.metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name in ("torch", "cuda-bindings", "apache-tvm-ffi") or name.startswith("nvidia-cutlass-dsl"):
            versions[name] = {"version": dist.version,
                              "recordSha256": digest((dist.read_text("RECORD") or "").encode())}
    return versions


def import_pristine(folder):
    # A private namespace avoids importing or rewriting the installed FlashInfer
    # package. Relative imports within the byte-exact sources remain unchanged.
    alias = "tensorsharp_flashinfer_61bc8b6d"
    package = types.ModuleType(alias)
    package.__path__ = [str(folder / "flashinfer")]
    sys.modules[alias] = package
    return importlib.import_module(alias + ".gemm.kernels.native_bf16_fp4.staged_kernel")


def write_new(path, data):
    with path.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def compile_one(rows, folder, output, warps=4, stages=2):
    import cutlass
    import cutlass.cute as cute
    upstream = import_pristine(folder)
    if not upstream._DIRECT_BF16_CVT:
        raise RuntimeError("This prototype requires CUDA >=13.2 for the direct native FP4/E4M3 BF16 conversions")
    fake = cute.runtime.make_fake_compact_tensor
    a = fake(cutlass.Int32, (rows, K // 2), stride_order=(1, 0), assumed_align=16)
    b = fake(cutlass.Int32, (N, K // 8), stride_order=(1, 0), assumed_align=16)
    sf = fake(cutlass.Float8E4M3FN, (N // 128 * (K // 64) * 512,), assumed_align=16)
    alpha = fake(cutlass.Float32, (1,), assumed_align=4)
    out = fake(cutlass.Float32, (rows, N), stride_order=(1, 0), assumed_align=4)
    stream = cute.runtime.make_fake_stream()
    op = upstream.NativeBf16Fp4StagedKernel(splits=1, warps=warps, enable_pdl=False, tile_k=128, stages=stages)
    compiled = cute.compile(op, a, b, sf, alpha, out, None, stream, options="--gpu-arch sm_120a")
    cubin, ptx = compiled.__cubin__, compiled.__ptx__
    if not isinstance(cubin, bytes) or cubin[:4] != b"\x7fELF" or not isinstance(ptx, str):
        raise RuntimeError("The compiler did not retain a binary CUBIN and textual PTX; no host object fallback is accepted")
    name, parameters = parse_entry(ptx)
    stem = "ninfer_fi_native_w4a16_t" + str(rows)
    if (warps, stages) != (4, 2):
        stem += "_w" + str(warps) + "_st" + str(stages)
    cubin_path, ptx_path = output / (stem + ".cubin"), output / (stem + ".ptx")
    write_new(cubin_path, cubin)
    write_new(ptx_path, ptx.encode())
    # Preserve the lowered host artifact for inspecting resolved launch metadata;
    # it is evidence only and is never loaded by TensorSharp's serving runtime.
    ir_module = getattr(compiled, "ir_module", None)
    if ir_module is not None:
        write_new(output / (stem + ".host.mlir"), str(ir_module).encode())
    return {"rows": rows, "outputRows": N, "columns": K, "cubin": cubin_path.name,
            "cubinSha256": digest(cubin), "ptx": ptx_path.name, "ptxSha256": digest(ptx.encode()),
            "kernel": name, "parameters": parameters, "argumentOrder": ["inputBf16", "originalCodes", "originalScales", "alphaF32", "outputF32"],
            "grid": [N // 128, 1, 1], "block": [32 * warps, 1, 1], "dynamicSmemBytes": SMEM // 2 * stages,
            "dynamicSmemBasis": "safe reservation from all three pinned SmemAllocator allocations including 16-byte alignment; not a claim about resolved host launch metadata",
            "inputBytes": rows * K * 2, "codeBytes": N * K // 2, "scaleBytes": N * K // 16,
            "alphaBytes": 4, "outputBytes": rows * N * 4,
            "compilerKernelInfo": repr(getattr(compiled, "kernel_info", None))}


def self_test():
    name, args = parse_entry(".visible .entry test(\n" + ",\n".join(".param .u64 p" + str(i) for i in range(5)) + ") { ret; }")
    assert name == "test" and [a["offset"] for a in args] == [0, 8, 16, 24, 32]
    for bad in (".visible .entry k(.param .u32 a){}", ".visible .entry k(.param .u64 a){}", ""):
        try:
            parse_entry(bad)
        except ValueError:
            continue
        raise AssertionError("Unsafe device ABI accepted")
    assert SMEM == 31232
    assert SMEM // 2 * 3 == 46848
    assert len({v[2] for v in TACTICS.values()}) == 3 and TACTICS["s1-w4-st2"] == (4, 2, POLICY)
    return {"deviceAbiParser": "PASS", "sourceSmemReservationBytes": SMEM, "knownExplicitTactics": list(TACTICS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--tactic", choices=tuple(TACTICS), default="s1-w4-st2")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    folder = root / "TensorSharp.Backends.Cuda/native/ninfer/flashinfer_w4a16/upstream"
    verification = verify_sources(folder)
    checks = self_test()
    if args.verify_only:
        print(json.dumps({"verification": verification, "selfTest": checks, "nativeCompilation": "NOT_RUN", "GPU": "NOT_RUN"}, indent=2))
        return
    if args.out is None:
        parser.error("--out is required unless --verify-only")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CUTE_DSL_KEEP"] = "ir,ptx,cubin"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(output / "compiler-dump")
    os.environ["CUTE_DSL_ARCH"] = "sm_120a"
    warps, stages, policy = TACTICS[args.tactic]
    result = {"schema": "tensorsharp-ninfer-flashinfer-a16-aot/1", "arithmeticPolicy": policy, "tactic": args.tactic,
              "createdUtc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "sourceVerification": verification,
              "builderSha256": digest(pathlib.Path(__file__).read_bytes()), "selfTest": checks,
              "stack": stack_versions(), "arch": "sm_120a", "splits": 1, "warps": warps, "tileK": 128,
              "stages": stages, "enablePdl": False, "outputDtype": "Float32", "specializations": [],
              "status": "building", "GPU": "NOT_RUN", "modelQuality": "NOT_RUN", "performance": "NOT_RUN"}
    try:
        for rows in (1, 8):
            result["specializations"].append(compile_one(rows, folder, output, warps, stages))
        result["status"] = "complete"
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        write_new(output / "failed-build.json", (json.dumps(result, indent=2) + "\n").encode())
        raise
    data = (json.dumps(result, indent=2) + "\n").encode()
    write_new(output / "module.json", data)
    write_new(output / "module.json.sha256", (digest(data) + "  module.json\n").encode())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
