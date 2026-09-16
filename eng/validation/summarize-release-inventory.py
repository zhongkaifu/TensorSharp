#!/usr/bin/env python3
"""Reconcile the catalog with files and prior independent SHA verification logs.

This performs metadata checks only. A download-time SHA256 verification is
labelled separately from current existence/size; it is not a new full-file scan.
"""
import argparse
import json
from pathlib import Path
import re
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--logs", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    verified = {}
    for directory in args.logs:
        for path in directory.rglob("*.json"):
            try:
                doc = json.loads(path.read_text())
                if not isinstance(doc, dict) or not doc.get("directory") or not doc.get("repository"):
                    continue
                for item in doc.get("files", []):
                    lfs_match = item.get("expected_sha256") and item.get("observed_sha256") == item["expected_sha256"]
                    git_match = item.get("expected_git_blob_sha1") and item.get("observed_git_blob_sha1") == item["expected_git_blob_sha1"]
                    if ((lfs_match or git_match) and item.get("observed_sha256")
                            and item.get("observed_size") == item.get("size")):
                        filename = str(Path(doc["directory"]) / item["path"])
                        verified[filename] = {"manifest": str(path), "repository": doc["repository"],
                            "revision": doc.get("revision"), "size": item["size"],
                            "download_verified_sha256": item["observed_sha256"],
                            "publisher_digest_kind": "lfs-sha256" if lfs_match else "git-blob-sha1"}
            except (OSError, ValueError, TypeError):
                continue
    report = {"captured_at_unix": time.time(), "verification_scope": "Existing independent download hashes plus current file existence/size; no new whole-file hash scan.",
              "models": [], "unavailable_sources": catalog.get("unprovisioned", [])}
    for model in catalog["models"]:
        directory = args.root / model["id"]
        files, missing = set(), []
        for pattern in model["patterns"]:
            matches = [path for path in directory.glob(pattern) if path.is_file()]
            files.update(matches)
            if not matches:
                missing.append(pattern)
        shard_groups = {}
        for path in files:
            match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name)
            if match:
                key = (str(path.parent), match.group(1), int(match.group(3)))
                shard_groups.setdefault(key, set()).add(int(match.group(2)))
        for (_, prefix, count), ordinals in shard_groups.items():
            absent = sorted(set(range(1, count + 1)) - ordinals)
            if absent:
                missing.append(f"{prefix}: missing shards {absent} of {count}")
        records = []
        for path in sorted(files):
            record = {"path": str(path), "bytes": path.stat().st_size}
            observation = verified.get(str(path))
            if observation and observation["size"] == record["bytes"] and observation["repository"] == model["repo"]:
                record.update(observation)
            records.append(record)
        complete = bool(records) and not missing and all("download_verified_sha256" in record for record in records)
        status = "downloaded-and-verified" if complete else "incomplete-or-unverified" if records else "not-downloaded"
        report["models"].append({**model, "status": status, "files": records, "missing_patterns": missing,
                                 "total_present_bytes": sum(record["bytes"] for record in records)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for model in report["models"]:
        print(model["id"], model["status"], round(model["total_present_bytes"] / 1024 ** 3, 2))


if __name__ == "__main__":
    main()
