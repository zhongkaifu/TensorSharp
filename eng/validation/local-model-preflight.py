#!/usr/bin/env python3
"""Audit local GGUF/safetensors files without loading a model or downloading data.

Full SHA-256 comparison is the integrity gate. File length, sparse flags, valid
headers and completed download ranges are only structural/advisory evidence.
Expected JSON: {"files": [{"path": "...", "sha256": "...", "bytes": 123}]}.
Paths in the manifest may be relative to --model-root. No file is modified.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import time

BLOCKS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
          6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
          10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
          14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74),
          18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
          22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2),
          26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2),
          34: (256, 54), 35: (256, 66), 39: (32, 17), 40: (64, 36)}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gguf_structure(path):
    # Reuse the repository's header reader, but fetch only local bounded bytes.
    spec = importlib.util.spec_from_file_location("gguf_header", Path(__file__).with_name("inspect-remote-gguf.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    size = path.stat().st_size
    with path.open("rb") as source:
        header = module.inspect(source.read(min(size, 64 << 20)), size)
    if header["version"] not in (2, 3):
        raise ValueError("Only GGUF v2/v3 are supported")
    tensors, seen, end = header["tensors"], set(), header["header_bytes"]
    for tensor in tensors:
        name, shape, kind = tensor["name"], tensor["dimensions"], tensor["type"]
        if name in seen or not shape or len(shape) > 8 or any(n <= 0 for n in shape):
            raise ValueError(f"Duplicate tensor or invalid shape: {name}")
        seen.add(name)
        if kind not in BLOCKS:
            raise ValueError(f"Unsupported type {kind}; cannot prove extent of {name}")
        block, width = BLOCKS[kind]
        if shape[0] % block:
            raise ValueError(f"Unaligned first dimension: {name}")
        count = math.prod(shape) // block * width
        begin = tensor["file_offset"]
        if begin < end or begin + count > size:
            raise ValueError(f"Overlapping/truncated tensor: {name}")
        end = begin + count
    metadata = header["metadata"]
    return {"format": "gguf", "tensor_count": len(tensors),
            "architecture": metadata.get("general.architecture"),
            "split_no": metadata.get("split.no"), "split_count": metadata.get("split.count"),
            "split_tensor_count": metadata.get("split.tensors.count"),
            "header_bytes": header["header_bytes"], "payload_end": end,
            "tensor_names": sorted(seen), "types": sorted({t["type"] for t in tensors})}


def safetensors_structure(path):
    size = path.stat().st_size
    with path.open("rb") as source:
        (count,) = struct.unpack("<Q", source.read(8))
        if count <= 1 or count > min(64 << 20, size - 8):
            raise ValueError("Invalid safetensors header length")
        header = json.loads(source.read(count))
    widths = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "F16": 2, "BF16": 2,
              "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    tensors, ranges = [], []
    for name, value in header.items():
        if name == "__metadata__":
            continue
        lo, hi = value["data_offsets"]
        shape, kind = value["shape"], value["dtype"]
        if kind not in widths or any(n < 0 for n in shape):
            raise ValueError(f"Unsupported dtype/shape: {name}")
        if lo < 0 or hi - lo != math.prod(shape) * widths[kind] or hi > size - count - 8:
            raise ValueError(f"Truncated/invalid tensor: {name}")
        tensors.append(name)
        ranges.append((lo, hi))
    end = 0
    for lo, hi in sorted(ranges):
        if lo != end:
            raise ValueError("Safetensors data must be contiguous without holes or overlap")
        end = hi
    if end != size - count - 8:
        raise ValueError("Safetensors trailing/missing data")
    return {"format": "safetensors", "tensor_count": len(tensors), "tensor_names": sorted(tensors)}


def range_evidence(path):
    result = []
    legacy = Path(str(path) + ".ranges.json")
    if legacy.is_file():
        record = json.loads(legacy.read_text(encoding="utf-8-sig"))
        total, prefix = record["total"], record["prefix"]
        starts = sorted(set(record["completed"]))
        # This historical downloader used 64 MiB chunks; require its exact grid.
        chunk = record.get("chunk_bytes", 64 << 20)
        if not 0 <= prefix <= total or chunk <= 0:
            raise ValueError("Invalid download ranges")
        wanted = set(range(prefix, total, chunk))
        valid = set(starts).issubset(wanted)
        result.append({"path": str(legacy), "bytes": total, "prefix_bytes": prefix,
                       "chunk_bytes": chunk, "chunk_size_explicit": "chunk_bytes" in record,
                       "missing_chunk_starts": sorted(wanted - set(starts)),
                       "ranges_cover_file": valid and set(starts) == wanted and total == path.stat().st_size,
                       "note": "Downloader records are not a content integrity proof."})
    provenance = Path(str(path) + ".part.ranges") / "source.json"
    if provenance.is_file():
        record = json.loads(provenance.read_text(encoding="utf-8-sig"))
        result.append({"path": str(provenance), **record,
                       "note": "prefix_bytes is downloader state, not a content integrity proof."})
    return result


def inspect_file(path, expected=None, hash_contents=False):
    expected = expected or {}
    report = {"path": str(path), "exists": path.is_file(), "status": "missing"}
    if not report["exists"]:
        return report
    before = path.stat()
    report.update(bytes=before.st_size, mtime_ns=before.st_mtime_ns)
    if os.name == "nt":
        get_attributes = ctypes.windll.kernel32.GetFileAttributesW
        get_attributes.argtypes, get_attributes.restype = [ctypes.c_wchar_p], ctypes.c_uint32
        attributes = get_attributes(str(path))
        report["windows_sparse_attribute"] = None if attributes == 0xffffffff else bool(attributes & 0x200)
    try:
        report["structure"] = (gguf_structure(path) if path.suffix.lower() == ".gguf" else safetensors_structure(path))
        report["download_records"] = range_evidence(path)
        expected_hash = expected.get("sha256")
        if expected_hash and not re.fullmatch(r"[a-fA-F0-9]{64}", expected_hash):
            raise ValueError("Expected SHA-256 must contain 64 hexadecimal digits")
        expected_bytes = expected.get("bytes")
        report["expected"] = expected
        if expected_bytes is not None and expected_bytes != before.st_size:
            raise ValueError(f"Expected {expected_bytes} bytes, observed {before.st_size}")
        report["status"] = "structure_only"
        if hash_contents:
            started = time.perf_counter()
            report["sha256"] = sha256(path)
            report["hash_seconds"] = time.perf_counter() - started
            report["status"] = "verified" if expected_hash and report["sha256"] == expected_hash.lower() else "identity_only"
            if expected_hash and report["status"] != "verified":
                raise ValueError("SHA-256 does not match the expected checkpoint")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("File changed during inspection")
    except Exception as error:
        report.update(status="invalid", error=str(error))
    return report


def split_checks(files):
    groups = {}
    for item in files:
        match = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", item["path"], re.I)
        if match:
            groups.setdefault((match[1], int(match[2])), []).append(item)
    reports = []
    for (stem, count), items in groups.items():
        shapes = [item.get("structure", {}) for item in items]
        names = [name for shape in shapes for name in shape.get("tensor_names", [])]
        expected_counts = {shape.get("split_tensor_count") for shape in shapes}
        complete = len(items) == count and {s.get("split_no") for s in shapes} == set(range(count))
        complete &= all(s.get("split_count") == count for s in shapes)
        complete &= len(expected_counts) == 1 and None not in expected_counts and len(names) in expected_counts
        complete &= len(set(names)) == len(names)
        reports.append({"stem": stem, "expected_shards": count, "present_shards": len(items),
                        "tensor_count": len(names), "structurally_complete": bool(complete),
                        "integrity_verified": bool(complete) and all(x["status"] == "verified" for x in items)})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="*")
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--model-root", type=Path, default=Path.cwd())
    parser.add_argument("--hash", action="store_true", help="Read each entire file once, sequentially")
    parser.add_argument("--require-verified", action="store_true", help="Exit nonzero unless every file matches an expected SHA-256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expectations = {}
    if args.expected:
        for item in json.loads(args.expected.read_text(encoding="utf-8-sig"))["files"]:
            path = Path(item["path"])
            if not path.is_absolute():
                path = args.model_root / path
            expectations[str(path.resolve())] = item
    paths = [p.resolve() for p in args.paths] or [Path(p) for p in expectations]
    if not paths:
        parser.error("Provide paths or --expected files")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "files": [],
              "scope": "Integrity/structure only; no inference, quality, performance or memory-fit claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in paths:
        item = inspect_file(path, expectations.get(str(path)), args.hash)
        report["files"].append(item)
        report["split_checks"] = split_checks(report["files"])
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: item.get(key) for key in ("path", "status", "bytes", "sha256", "error")}), flush=True)
    invalid = any(item["status"] in ("invalid", "missing") for item in report["files"])
    invalid |= any(not group["structurally_complete"] for group in report["split_checks"])
    unverified = any(item["status"] != "verified" for item in report["files"])
    return 1 if invalid else 2 if args.require_verified and unverified else 0


if __name__ == "__main__":
    raise SystemExit(main())
