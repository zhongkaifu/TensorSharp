#!/usr/bin/env python3
"""Refresh all saved HTTP/oracle comparisons without running a model forward.

Run only after the standard HTTP reports are complete and performance measurement
has stopped. Existing JSON reports and their README are archived before the new
reports are promoted. A numerical gate failure is retained and yields exit 1.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


def plans(results: Path, models: Path):
    for model, filename, tokens in [
        ("snowflake", "snowflake-arctic-embed-l-v2.0-q8_0.gguf", "snowflake-tokenization.json"),
        ("minilm", "all-MiniLM-L6-v2-Q8_0.gguf", "numpy-minilm-tokens.json"),
        ("nomic", "nomic-embed-text-v1.5.Q8_0.gguf", "nomic-tokenization.json"),
    ]:
        for backend in ("managed", "cpu", "metal"):
            for order in ("", "-reverse"):
                stem = f"{model}-{backend}{order}"
                for engine in ("tensorsharp", "llama"):
                    yield {
                        "name": ("llama-" if engine == "llama" else "") + stem + ".json",
                        "model": models / filename,
                        "tokens": results / tokens,
                        "http_results": results / stem / "results.json",
                        "reference_vectors": results / "numpy-oracle" / f"{model}-metal.vectors.npz",
                        "engine": engine,
                    }


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=root / "docs/validation/embeddings-2026-09")
    parser.add_argument("--models", type=Path, default=root.parent / "models/embeddings")
    parser.add_argument("--gguf-py", type=Path, default=root.parent / "llama.cpp/gguf-py")
    parser.add_argument("--archive-label", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned comparisons; no reads, hashes, imports, or writes of evidence.")
    args = parser.parse_args()
    if Path(args.archive_label).name != args.archive_label or args.archive_label in (".", ".."):
        parser.error("--archive-label must be one directory name")
    jobs = list(plans(args.results.resolve(), args.models.resolve()))
    if args.dry_run:
        print(json.dumps(jobs, default=str, indent=2))
        return 0

    # Hash immutable evidence once per process. Reusing a hash requires the same
    # device, inode, size, mtime, and ctime; any concurrent rewrite aborts refresh.
    cache = {}

    def signature(path):
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def sha256(path):
        path = Path(path).resolve()
        before = signature(path)
        if path in cache:
            old_signature, digest = cache[path]
            if before != old_signature:
                raise RuntimeError(f"Evidence changed during refresh: {path}")
            return digest
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if signature(path) != before:
            raise RuntimeError(f"Evidence changed while hashing: {path}")
        cache[path] = (before, digest.hexdigest())
        return cache[path][1]

    output = args.results.resolve() / "numpy-oracle"
    archive = output / "history" / args.archive_label
    staging = output / (".refresh-" + args.archive_label)
    if archive.exists() or staging.exists():
        raise FileExistsError("Archive/staging label already exists; choose a new label.")
    required = {root / "eng/embedding-reference.py"}
    for job in jobs:
        required.update(job[key] for key in ("model", "tokens", "http_results", "reference_vectors"))
        required.add(output / job["name"])
    required.add(output / "README.md")
    for path in sorted(required):
        if not path.is_file():
            raise FileNotFoundError(path)

    archive.mkdir(parents=True)
    staging.mkdir()
    previous = {}
    for name in [job["name"] for job in jobs] + ["README.md", "refresh-manifest.json"]:
        source = output / name
        if not source.exists():
            continue
        digest = sha256(source)
        shutil.copy2(source, archive / name)
        if sha256(archive / name) != digest:
            raise RuntimeError(f"Archive verification failed: {name}")
        previous[name] = digest
    (archive / "archive-manifest.json").write_text(json.dumps(previous, indent=2) + "\n")

    for name in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "1"
    script = root / "eng/embedding-reference.py"
    spec = importlib.util.spec_from_file_location("embedding_reference", script)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    oracle.sha256 = sha256
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Comparison-only refresh using existing independent NumPy gold; no model forward.",
        "refresh_script_sha256": sha256(Path(__file__)),
        "comparison_script_sha256": sha256(script),
        "previous_reports": {"directory": str(archive), "sha256": previous},
        "thresholds": {"min_cosine": 0.999, "max_absolute_error": 0.005},
        "reports": [],
    }
    original_argv = sys.argv
    try:
        for job in jobs:
            destination = staging / job["name"]
            sys.argv = [str(script), "--model", str(job["model"]),
                        "--tokens", str(job["tokens"]), "--indices", "0,1,2,3,7,8",
                        "--http-results", str(job["http_results"]),
                        "--reference-vectors", str(job["reference_vectors"]),
                        "--http-engine", job["engine"], "--gguf-py", str(args.gguf_py.resolve()),
                        "--gelu", "exact", "--min-cosine", "0.999",
                        "--max-absolute-error", "0.005", "--output", str(destination)]
            print(job["name"], flush=True)
            status = oracle.main()
            report = json.loads(destination.read_text())
            if status not in (0, 1) or status != (0 if report["passed"] else 1):
                raise RuntimeError(f"Unexpected oracle status for {job['name']}: {status}")
            manifest["reports"].append({
                "file": job["name"], "sha256": sha256(destination),
                "engine": job["engine"], "passed": report["passed"],
                "minimum_cosine": report["minimum_cosine"],
                "maximum_absolute_error": report["maximum_absolute_error"],
                "retrieval_passed": all(item["same_order"] for item in report["retrieval"]),
                "http_results_sha256": report["http_results_sha256"],
                "reference_vectors_sha256": report["reference_vectors_sha256"],
            })
    finally:
        sys.argv = original_argv

    # All comparisons have finished before canonical reports change. Verify that
    # the inputs still match their hashed snapshots, then promote staged reports.
    for path, (expected, _) in cache.items():
        if signature(path) != expected:
            raise RuntimeError(f"Evidence changed before report promotion: {path}")
    manifest["failed_reports"] = [item["file"] for item in manifest["reports"] if not item["passed"]]
    manifest["tensorsharp_passed"] = all(item["passed"] for item in manifest["reports"]
                                         if item["engine"] == "tensorsharp")
    manifest["all_gates_passed"] = not manifest["failed_reports"]
    for job in jobs:
        (staging / job["name"]).replace(output / job["name"])
    (staging / "refresh-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (staging / "refresh-manifest.json").replace(output / "refresh-manifest.json")
    staging.rmdir()
    print(json.dumps({key: manifest[key] for key in
                      ("tensorsharp_passed", "all_gates_passed", "failed_reports")}, indent=2))
    return 0 if manifest["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
