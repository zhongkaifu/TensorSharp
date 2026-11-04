#!/usr/bin/env python3
"""Build the isolated, non-serving Ignis BF16 MMA T8/F32-epilogue component."""
import argparse
import datetime
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import traceback

REVISION = "e600a7d9e0d5921d767a6cb464373583b3cdcaf4"
POLICY = "ignis-e600-bf16-t8-original-mma-f32-store-v1"
MANIFEST_SHA256 = "62654b5bddb6f9dc20c0243bfa22835d9ba71f7621257be664f98fb3d4488876"
SHAPES = ((14336, 5120), (5120, 6144))
EXPECTED_FILES = {"ops/linear/bf16/bf16_config.h", "ops/linear/bf16/bf16_gemm_mma.cuh",
                  "ops/common/mma.cuh", "ops/common/memory.cuh", "LICENSE"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def verify_upstream(folder):
    raw = (folder / "manifest.json").read_bytes()
    if sha(raw) != MANIFEST_SHA256:
        raise ValueError("Pinned original-source manifest differs")
    manifest = json.loads(raw)
    if (manifest["revision"] != REVISION or manifest["modifiedUpstreamFiles"] != []
            or {x["path"] for x in manifest["files"]} != EXPECTED_FILES
            or len(manifest["files"]) != len(EXPECTED_FILES)):
        raise ValueError("Original-source inventory differs")
    for item in manifest["files"]:
        data = (folder / item["path"]).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if (len(data) != item["bytes"] or sha(data) != item["sha256"]
                or blob != item["gitBlobSha1"]):
            raise ValueError("Pristine Ignis source differs: " + item["path"])
    return {"revision": REVISION, "manifestSha256": sha(raw), "filesVerified": len(EXPECTED_FILES),
            "modifiedUpstreamFiles": []}


def parse_parameters(text):
    result = []
    offset = 0
    for parameter in text.split(","):
        parameter = parameter.strip()
        match = re.fullmatch(r"\.param\s+(?:\.align\s+(\d+)\s+)?\.(u64|u32|b8)\s+"
                             r"(?:\.ptr(?:\s+\.(?:global|const|shared|local))?\s+"
                             r"(?:\.align\s+\d+\s+)?)?([\w$]+)(?:\[(\d+)\])?", parameter)
        if not match:
            raise ValueError("Unsupported PTX parameter declaration: " + parameter)
        natural = {"u64": 8, "u32": 4, "b8": 1}[match[2]]
        alignment = int(match[1] or natural)
        size = natural if match[2] != "b8" else int(match[4] or 0)
        if alignment not in (1, 2, 4, 8, 16) or not size or (match[2] != "b8" and match[4]):
            raise ValueError("Invalid PTX parameter storage")
        offset = (offset + alignment - 1) // alignment * alignment
        result.append({"name": match[3], "type": match[2], "size": size,
                       "alignment": alignment, "offset": offset})
        offset += size
    return result


def parse_entries(ptx):
    exports = re.findall(r"\.visible\s+\.entry\s+([\w$]+)\s*\((.*?)\)", ptx, re.S)
    if len(exports) != 4:
        raise ValueError("Exactly four original-core instantiations are required")
    result = []
    identities = set()
    for name, params in exports:
        if "bf16_gemm_mma_kernel" not in name:
            raise ValueError("Unexpected exported entry")
        shapes = [(n, k) for n, k in SHAPES if f"Bf16GemvGeometryILi{n}ELi{k}EE" in name]
        if len(shapes) != 1:
            raise ValueError("Unknown original-core geometry")
        n, k = shapes[0]
        candidate = "TsNinferBf16MmaF32Output" in name
        reference = "Bf16MmaContiguousOutput" in name
        if candidate == reference:
            raise ValueError("Unknown or ambiguous output policy")
        policy = "f32" if candidate else "original-bf16"
        key = (n, k, policy)
        if key in identities:
            raise ValueError("Duplicate original-core specialization")
        identities.add(key)
        parameters = parse_parameters(params)
        expected_sizes = [8, 8, 8 if candidate else 16, 4]
        if ([p["size"] for p in parameters] != expected_sizes
                or [p["offset"] for p in parameters] != [0, 8, 16, 24 if candidate else 32]
                or [p["type"] for p in parameters[:2]] != ["u64", "u64"]
                or parameters[2]["type"] != "b8" or parameters[2]["alignment"] != 8
                or parameters[3]["type"] != "u32" or parameters[3]["alignment"] != 4):
            raise ValueError("Original-core device ABI differs")
        result.append({"outputRows": n, "columns": k, "rows": 8, "outputPolicy": policy,
                       "kernel": name, "parameters": parameters, "grid": [n // 32, 1, 1],
                       "block": [256, 1, 1], "dynamicSmemBytes": 98304,
                       "argumentOrder": ["inputBF16", "weightsRowMajorBF16", "outputPolicyByValue", "tokensI32"],
                       "outputPolicyFields": ([{"name": "outputF32", "offset": 0, "size": 8}]
                                              if candidate else [{"name": "outputBF16", "offset": 0, "size": 8},
                                                                 {"name": "outputRowsI32", "offset": 8, "size": 4},
                                                                 {"name": "zeroPadding", "offset": 12, "size": 4}]),
                       "inputBytes": 8 * k * 2, "weightBytes": n * k * 2,
                       "outputBytes": 8 * n * (4 if candidate else 2)})
    if identities != {(n, k, p) for n, k in SHAPES for p in ("f32", "original-bf16")}:
        raise ValueError("Missing original-core specialization")
    return sorted(result, key=lambda x: (x["outputRows"], x["outputPolicy"]))


def self_test():
    def fixture(n, k, candidate):
        policy = "TsNinferBf16MmaF32Output" if candidate else "Bf16MmaContiguousOutput"
        name = f"_Zbf16_gemm_mma_kernel_Bf16GemvGeometryILi{n}ELi{k}EE_{policy}"
        params = f".param .u64 a, .param .u64 .ptr .align 1 b, .param .align 8 .b8 c[{8 if candidate else 16}], .param .u32 tokens"
        return f".visible .entry {name}({params}) {{ ret; }}"
    text = "\n".join(fixture(n, k, c) for n, k in SHAPES for c in (False, True))
    assert len(parse_entries(text)) == 4
    bad = [text.replace(".b8 c[16]", ".b8 c[12]", 1), text.replace(".align 8 .b8", ".align 4 .b8", 1),
           text + fixture(14336, 5120, True), text.replace("14336", "14335"),
           text.replace("TsNinferBf16MmaF32Output", "UnknownPolicy"),
           text.replace(".u64 a", ".u32 a", 1), text.replace(".b8 c[8]", ".b8 c[16]", 1),
           text.replace(".u32 tokens", ".u64 tokens", 1), text.replace(", .param .u32 tokens", "", 1)]
    for malformed in bad:
        try:
            parse_entries(malformed)
        except ValueError:
            continue
        raise AssertionError("Malformed ABI was admitted")
    return {"validFourEntryAbi": True, "rejectedMalformedAbiCases": len(bad),
            "nativeCompilation": "NOT_RUN", "GPU": "NOT_RUN"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    source_dir = root / "TensorSharp.Backends.Cuda/native/ninfer/bf16_mma"
    source = source_dir / "tensorsharp_ninfer_bf16_mma.cu"
    verified = verify_upstream(source_dir / "upstream")
    check = self_test()
    if args.verify_only:
        print(json.dumps({"sourceVerification": verified, "parserSelfTest": check,
                          "sourceSha256": sha(source.read_bytes()), "servingIntegration": "NOT_PRESENT"}, indent=2))
        return
    if args.out is None or not args.out.is_absolute():
        parser.error("--out must be a fresh absolute directory, unless --verify-only")
    nvcc = shutil.which(args.nvcc)
    if not nvcc:
        raise ValueError("NVCC not found")
    version = subprocess.run([nvcc, "--version"], check=True, capture_output=True, text=True, timeout=30)
    args.out.mkdir(parents=True, exist_ok=False)
    stem = "tensorsharp_ninfer_bf16_mma"
    result = {"schema": "tensorsharp-ninfer-bf16-mma-aot/1", "arithmeticPolicy": POLICY,
              "createdUtc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "complete": False,
              "sourceVerification": verified, "sourceSha256": sha(source.read_bytes()),
              "builderSha256": sha(pathlib.Path(__file__).read_bytes()), "parserSelfTest": check,
              "nvcc": nvcc, "nvccVersion": version.stdout, "arch": "sm_120a", "commands": [],
              "GPU": "NOT_RUN", "modelQuality": "NOT_RUN", "performance": "NOT_RUN",
              "servingIntegration": "NOT_PRESENT", "scope": "two fixed BF16-input/BF16-weight T8 projections only",
              "pdl": "original MMA core has no PDL operations; ordinary launch",
              "schedule": [32, 32, 256, 16, 8, 3, 1, "cg", "cg", "PingPong", "TokenFast", "Xor64", 1],
              "scheduleSource": "Observed Ignis binary; owned instantiation of pristine e600 core, not the e600 production alias",
              "fullTokens": False, "splitK": 1}
    try:
        for kind in ("ptx", "fatbin"):
            target = args.out / (stem + "." + kind)
            command = [nvcc, "--std=c++17", "-O3", "--ftz=false", "--fmad=true", "-lineinfo",
                       "--gpu-architecture=sm_120a", "--" + kind, "-I", str(source_dir / "upstream"),
                       str(source), "-o", str(target)]
            if kind == "fatbin":
                command += ["--ptxas-options=-v"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
            write_new(args.out / (kind + ".stdout.log"), completed.stdout.encode())
            write_new(args.out / (kind + ".stderr.log"), completed.stderr.encode())
            result["commands"].append({"argv": command, "exit": completed.returncode})
            if completed.returncode:
                raise RuntimeError("NVCC failed: " + kind)
            if not target.is_file() or not target.stat().st_size:
                raise ValueError("NVCC omitted module bytes")
            result[kind] = {"path": target.name, "bytes": target.stat().st_size, "sha256": sha(target.read_bytes())}
        result["specializations"] = parse_entries((args.out / (stem + ".ptx")).read_text())
        after = verify_upstream(source_dir / "upstream")
        if after != verified or sha(source.read_bytes()) != result["sourceSha256"]:
            raise ValueError("Source changed during build")
        # CUDA's paired-fatbin sidecar contains the associated PTX SHA, not the fatbin SHA.
        write_new(args.out / (stem + ".fatbin.sha256"), (result["ptx"]["sha256"] + "\n").encode())
        result["complete"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        write_new(args.out / "failed-build.json", (json.dumps(result, indent=2) + "\n").encode())
        raise
    raw = (json.dumps(result, indent=2) + "\n").encode()
    write_new(args.out / "completed-build.json", raw)
    write_new(args.out / "completed-build.json.sha256", (sha(raw) + "  completed-build.json\n").encode())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
