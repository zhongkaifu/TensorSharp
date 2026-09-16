#!/usr/bin/env python3
"""Pin and download explicit public GGUF release-validation artifacts.

The manifest records immutable Hugging Face revisions, byte sizes and published
LFS SHA256 digests. Downloads are resumable. A download is not a correctness or
quality result; --verify performs an independent complete SHA256 scan.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--include", action="append", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    api = f"https://huggingface.co/api/models/{args.repo}/revision/{args.revision}?blobs=true"
    with urllib.request.urlopen(api, timeout=120) as response:
        metadata = json.load(response)
    entries = [entry for entry in metadata["siblings"]
               if any(fnmatch.fnmatchcase(entry["rfilename"], pattern) for pattern in args.include)]
    for pattern in args.include:
        if not any(fnmatch.fnmatchcase(entry["rfilename"], pattern) for entry in entries):
            raise ValueError(f"Pattern {pattern!r} matched no files in {args.repo}")
    files = [{"path": entry["rfilename"], "size": entry.get("size"),
              "expected_sha256": entry.get("lfs", {}).get("sha256"),
              "expected_git_blob_sha1": None if entry.get("lfs") else entry.get("blobId")}
             for entry in entries]
    manifest = {"repository": args.repo, "revision": metadata["sha"],
                "directory": str(args.directory), "files": files,
                "total_bytes": sum(entry["size"] or 0 for entry in files),
                "status": "planned", "started_at_unix": time.time()}
    save(args.manifest, manifest)
    print(json.dumps(manifest), flush=True)
    if args.inspect_only:
        return
    from huggingface_hub import hf_hub_download
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest["status"] = "downloading"
    save(args.manifest, manifest)

    def download(entry):
        started = time.monotonic()
        filename = hf_hub_download(args.repo, entry["path"], revision=metadata["sha"],
                                   local_dir=args.directory)
        observed = Path(filename).stat().st_size
        if entry["size"] is not None and observed != entry["size"]:
            raise ValueError(f"Size mismatch for {filename}: {observed} != {entry['size']}")
        entry["download_seconds"] = time.monotonic() - started
        entry["observed_size"] = observed
        if args.verify:
            digest = hashlib.sha256()
            git_digest = hashlib.sha1(f"blob {observed}\0".encode())
            with open(filename, "rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
                    if entry["expected_git_blob_sha1"]:
                        git_digest.update(chunk)
            entry["observed_sha256"] = digest.hexdigest()
            if entry["expected_sha256"] and digest.hexdigest() != entry["expected_sha256"]:
                raise ValueError(f"SHA256 mismatch for {filename}")
            if entry["expected_git_blob_sha1"]:
                entry["observed_git_blob_sha1"] = git_digest.hexdigest()
                if git_digest.hexdigest() != entry["expected_git_blob_sha1"]:
                    raise ValueError(f"Git blob SHA1 mismatch for {filename}")
            if not entry["expected_sha256"] and not entry["expected_git_blob_sha1"]:
                raise ValueError(f"Publisher supplied no verifiable digest for {filename}")
        return entry

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, entry): entry for entry in files}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                result = future.result()
                print("Completed " + json.dumps(result), flush=True)
            except Exception as error:
                entry["error"] = str(error)
                errors.append(str(error))
                print(f"ERROR {entry['path']}: {error}", flush=True)
            save(args.manifest, manifest)
    manifest["finished_at_unix"] = time.time()
    manifest["status"] = "failed" if errors else ("verified" if args.verify else "downloaded-size-checked")
    save(args.manifest, manifest)
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()
