#!/usr/bin/env python3
"""Copy complete model artifacts to fast local storage with source/destination SHA256 checks.

No model conversion, sparsification or source-tree mutation. Existing staged
files are reused only after matching their pinned source hash. Run outside
measurement phases; --reserve-gib leaves space for other tasks on the store.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=float, default=10)
    args = parser.parse_args()
    original = json.loads(args.source_manifest.read_text())
    if not isinstance(original, list):
        parser.error("--source-manifest must contain the pinned file identity list")
    pinned = {str(Path(entry["path"]).resolve()): entry for entry in original if entry.get("sha256") and entry.get("path")}
    files = []
    for path in args.sources:
        if path.is_dir():
            files += [(source, args.destination / path.name / source.relative_to(path)) for source in sorted(path.rglob("*"))
                      if source.is_file() and not any(part in (".cache", ".git") for part in source.relative_to(path).parts)]
        elif path.is_file():
            files.append((path, args.destination / path.name))
        else:
            parser.error(f"missing source: {path}")
    args.destination.mkdir(parents=True, exist_ok=True)
    needed = sum(source.stat().st_size for source, target in files if not target.exists())
    if needed + int(args.reserve_gib * 1024 ** 3) > shutil.disk_usage(args.destination).free:
        parser.error("staging would violate the requested free-space reserve")
    staged = []
    for source, target in files:
        started = time.monotonic()
        reference = pinned.get(str(source.resolve()))
        expected = reference["sha256"] if reference else digest(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != source.stat().st_size or digest(target) != expected:
                raise ValueError(f"existing staged file differs; refusing overwrite: {target}")
        else:
            temporary = target.with_name(target.name + f".staging-{os.getpid()}")
            source_hash = hashlib.sha256()
            try:
                with source.open("rb") as reader, temporary.open("xb") as writer:
                    for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                        source_hash.update(block)
                        writer.write(block)
                if source_hash.hexdigest() != expected or digest(temporary) != expected:
                    raise ValueError(f"source/destination SHA256 mismatch: {source}")
                temporary.rename(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        record = {"source": str(source.resolve()), "path": str(target.resolve()), "bytes": target.stat().st_size,
                  "sha256": expected, "pinned_manifest_match": bool(reference), "staging_seconds": time.monotonic() - started}
        staged.append(record)
        print(json.dumps(record), flush=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps({"source_manifest": original, "staged_files": staged,
        "stager_sha256": digest(Path(__file__)), "reserve_gib": args.reserve_gib,
        "free_bytes_after_staging": shutil.disk_usage(args.destination).free}, indent=2) + "\n")


if __name__ == "__main__":
    main()
