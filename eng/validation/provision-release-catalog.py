#!/usr/bin/env python3
"""Resolve the checked-in release model catalog and provision selected entries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--models", help="Comma-separated model IDs; default all catalog entries")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    selected = set(args.models.split(",")) if args.models else None
    models = [model for model in catalog["models"] if selected is None or model["id"] in selected]
    args.logs.mkdir(parents=True, exist_ok=True)

    def inspect(model):
        item = dict(model)
        try:
            with urllib.request.urlopen(f"https://huggingface.co/api/models/{model['repo']}?blobs=true", timeout=90) as response:
                metadata = json.load(response)
            item["revision"] = metadata["sha"]
            files = [entry for entry in metadata["siblings"] if any(fnmatch.fnmatchcase(entry["rfilename"], pattern) for pattern in model["patterns"])]
            item["files"] = files
            item["unmatched_patterns"] = [pattern for pattern in model["patterns"] if not any(fnmatch.fnmatchcase(entry["rfilename"], pattern) for entry in files)]
            item["total_bytes"] = sum(entry.get("size", 0) for entry in files)
            if item["unmatched_patterns"]:
                item["available_candidates"] = [entry["rfilename"] for entry in metadata["siblings"] if entry["rfilename"].endswith((".gguf", ".safetensors"))]
            item["status"] = "missing-artifact" if item["unmatched_patterns"] else "resolved"
        except Exception as error:
            item["status"], item["error"] = "unavailable", str(error)
        return item

    with ThreadPoolExecutor(max_workers=6) as pool:
        inventory = list(pool.map(inspect, models))
    report = {"resolved_at_unix": time.time(), "models": inventory,
              "unprovisioned": catalog.get("unprovisioned", [])}
    (args.logs / "catalog-inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    for item in inventory:
        print(item["id"], item["status"], round(item.get("total_bytes", 0) / 1024**3, 3), item.get("unmatched_patterns", []), flush=True)
    if args.inspect_only:
        return
    failures = []
    for item in inventory:
        if item["status"] != "resolved":
            failures.append(item["id"])
            continue
        command = [sys.executable, str(Path(__file__).with_name("provision-release-models.py")),
                   "--repo", item["repo"], "--revision", item["revision"],
                   "--directory", str(args.root / item["id"]),
                   "--manifest", str(args.logs / (item["id"] + ".json")),
                   "--workers", str(args.workers), "--verify"]
        for pattern in item["patterns"]:
            command += ["--include", pattern]
        print("Provisioning", item["id"], flush=True)
        with (args.logs / (item["id"] + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        print("Finished", item["id"], result.returncode, flush=True)
        if result.returncode:
            failures.append(item["id"])
    if failures:
        raise RuntimeError("Unprovisioned entries: " + ", ".join(failures))


if __name__ == "__main__":
    main()
