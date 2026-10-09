#!/usr/bin/env python3
"""Qualify cache-off/on/on/off file-weight runs with identical complete logits."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics

spec = importlib.util.spec_from_file_location("parallel_evidence", Path(__file__).with_name("compare-parallel-runs.py"))
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)
require = evidence.require


def compare(paths, *, device=False, workspace=False):
    cache = "DeviceWorkspaceCache" if workspace else "DeviceCache" if device else "HostCache"
    option = "--workspace-cache-bytes" if workspace else "--device-cache-bytes" if device else "--host-cache-bytes"
    result = {"ComparableAndBitwiseEqual": False, "Errors": [], "Measurements": {}, "Executions": [],
        "Qualification": "Balanced fresh processes, unchanged complete logits, charged-owner cleanup and successful exits. Logical source bytes are not physical SSD traffic. Warm file cache; not independent arithmetic, semantic quality or engine parity. Idle hardware and thermal controls require external evidence."}
    try:
        require(len(paths) == 4 and len({str(p.resolve()) for p in paths}) == 4, "Require four distinct execution records")
        runs = [evidence.load(path, "serial") for path in paths]
        runs.sort(key=lambda r: evidence.utc(r["execution"]["started_utc"]))
        require(len({r["execution"]["pid"] for r in runs}) == 4, "Process IDs must be distinct")
        require(all(evidence.utc(a["execution"]["finished_utc"]) <= evidence.utc(b["execution"]["started_utc"])
                    for a, b in zip(runs, runs[1:])), "Processes overlap")
        ceilings = [int(r["report"]["RequestedOptions"][option]) for r in runs]
        require(ceilings[0] == ceilings[3] == 0 and ceilings[1] == ceilings[2] > 0, "Require off/on/on/off order")
        baseline = runs[0]["report"]
        keys = ("ModelSha256", "ModelBytes", "ModelsAssemblySha256", "ProbeAssemblySha256", "ManagedAssembliesSha256",
                "ModelGeometry", "Prompt", "Mode", "Generation", "Teacher", "Steps", "Repeats", "Environment")
        options = lambda r: {k: v for k, v in r["RequestedOptions"].items() if k not in ("--output", option)}
        history = lambda r: [(row["Generated"], row["Consumed"], row["LogitsSha256"]) for row in r["Records"]]
        require(len({row["LogitsSha256"] for row in baseline["Records"]}) == 1, "Same requests are not deterministic")
        samples = {"off": [], "on": []}
        for run, ceiling in zip(runs, ceilings):
            report = run["report"]
            require(all(report.get(k) == baseline.get(k) for k in keys), "Identity, environment or geometry differs")
            require(options(report) == options(baseline), "Other requested settings differ")
            require(report["Native"][0]["Sha256"] == baseline["Native"][0]["Sha256"], "Native libraries differ")
            require(history(report) == history(baseline), "Complete logits or token histories differ")
            plan = report.get("Plan") or {}
            require(report["Mode"] == "adaptive" and plan.get("Accepted") is True
                    and plan.get("SelectedCandidate", {}).get("Placement") == 2, "File-weight placement was not exercised")
            require(plan["SelectedCandidate"] == baseline["Plan"]["SelectedCandidate"], "Execution geometry differs")
            previous_reads = previous_hits = previous_device_hits = previous_uploads = 0
            previous_creations = previous_reuses = 0
            for row in report["Records"]:
                usage = row.get("Streaming") or {}
                require(usage.get("LinearTiles", 0) > 0 and usage.get("FileBytesRead", 0) > 0, "No streamed work")
                require(0 <= usage[cache + "Bytes"] <= usage["Peak" + cache + "Bytes"] <= ceiling, "Invalid cache payload accounting")
                if workspace:
                    require(usage["DeviceWorkspaceReuses"] >= 0, "Invalid workspace reuse counter")
                    require((usage[cache + "Bytes"] > 0 and usage["DeviceWorkspaceReuses"] > 0) if ceiling
                            else usage["DeviceWorkspaceReuses"] == 0, "Workspace cache was not exercised as requested")
                elif ceiling:
                    require(usage[cache + "Bytes"] > 0 and usage[cache + "HitBytes"] > 0 and usage[cache + "Hits"] > 0,
                            "Cache was enabled but not exercised")
                else:
                    require(usage[cache + "HitBytes"] == usage[cache + "Hits"] == 0, "Disabled cache reports hits")
                for when in ("Before", "After"):
                    pools = row[when]["Pools"]
                    require(pools and all(0 <= p["Reserved"] + p["Committed"] <= p["Capacity"] for p in pools), "Shared budget exceeded")
                reads, hits = usage["FileBytesRead"] - previous_reads, usage["HostCacheHitBytes"] - previous_hits
                device_hits = usage.get("DeviceCacheHitBytes", 0) - previous_device_hits
                require(reads > 0 and hits >= 0, "Invalid per-request source counters")
                require(device_hits >= 0, "Device reuse counter decreased")
                extra = {"DeviceCacheBytesPerRequest": device_hits}
                if device or workspace:
                    uploads = usage["WeightUploadBytes"] - previous_uploads
                    require(uploads >= 0 and usage["PeakDeviceOwnedBytes"] >= usage["PeakDeviceCacheBytes"],
                            "Invalid device upload/ownership counters")
                    require(usage["PeakDeviceOwnedBytes"] <= int(report["RequestedOptions"]["--device-bytes"]),
                            "Combined device retention and workspace exceeded the ceiling")
                    if device and ceiling and not row["Warmup"]:
                        require(device_hits > 0, "Measured request did not reuse device weights")
                    extra["WeightUploadBytesPerRequest"] = uploads
                    previous_uploads = usage["WeightUploadBytes"]
                if workspace:
                    creations = usage["DeviceSessionCreations"] - previous_creations
                    reuses = usage["DeviceWorkspaceReuses"] - previous_reuses
                    require(creations >= 0 and reuses >= 0, "Workspace lifecycle counter decreased")
                    require(usage["PeakDeviceOwnedBytes"] >= usage["PeakDeviceWorkspaceCacheBytes"], "Invalid workspace ownership")
                    if ceiling and not row["Warmup"]:
                        require(reuses > 0, "Measured request did not reuse workspaces")
                    extra.update(DeviceSessionCreationsPerRequest=creations, DeviceWorkspaceReusesPerRequest=reuses)
                    previous_creations, previous_reuses = usage["DeviceSessionCreations"], usage["DeviceWorkspaceReuses"]
                if not row["Warmup"]:
                    samples["on" if ceiling else "off"].append({**row, "FileBytesPerRequest": reads, "CacheBytesPerRequest": hits, **extra})
                previous_reads, previous_hits = usage["FileBytesRead"], usage["HostCacheHitBytes"]
                previous_device_hits = usage.get("DeviceCacheHitBytes", 0)
            result["Executions"].append({k: v for k, v in run.items() if k not in ("report", "execution")})
        for arm, rows in samples.items():
            result["Measurements"][arm] = {key: {"Median": statistics.median(row[key] for row in rows),
                "Minimum": min(row[key] for row in rows), "Maximum": max(row[key] for row in rows), "Samples": len(rows)}
                for key in ("PrefillTokensPerSecond", "DecodeTokensPerSecond", "FileBytesPerRequest", "CacheBytesPerRequest",
                            "DeviceCacheBytesPerRequest", *(("WeightUploadBytesPerRequest",) if device or workspace else ()),
                            *(("DeviceSessionCreationsPerRequest", "DeviceWorkspaceReusesPerRequest") if workspace else ()))}
        measurements = result["Measurements"]
        consumed_bytes = {row["FileBytesPerRequest"] + row["CacheBytesPerRequest"] + row["DeviceCacheBytesPerRequest"]
                          for rows in samples.values() for row in rows}
        require(len(consumed_bytes) == 1, "Cache/source counters do not describe the same consumed weight bytes")
        if device or workspace:
            projected_bytes = {row["WeightUploadBytesPerRequest"] + row["DeviceCacheBytesPerRequest"]
                               for rows in samples.values() for row in rows}
            require(len(projected_bytes) == 1, "Upload/reuse counters do not describe the same projected weights")
        if workspace:
            operations = {row["DeviceSessionCreationsPerRequest"] + row["DeviceWorkspaceReusesPerRequest"]
                          for rows in samples.values() for row in rows}
            require(len(operations) == 1, "Creation/reuse counters do not describe the same workspace operations")
            for key in ("FileBytesPerRequest", "CacheBytesPerRequest", "DeviceCacheBytesPerRequest", "WeightUploadBytesPerRequest"):
                require(len({row[key] for rows in samples.values() for row in rows}) == 1,
                        "Workspace comparison changed weight transfers or reads")
        reduced = "DeviceSessionCreationsPerRequest" if workspace else "WeightUploadBytesPerRequest" if device else "FileBytesPerRequest"
        require(measurements["on"][reduced]["Median"] < measurements["off"][reduced]["Median"], "No measured allocation/transfer/read reduction")
        result["OnToOffRatio"] = {key: measurements["on"][key]["Median"] / measurements["off"][key]["Median"]
            for key in ("PrefillTokensPerSecond", "DecodeTokensPerSecond", "FileBytesPerRequest",
                        *(("WeightUploadBytesPerRequest",) if device or workspace else ()),
                        *(("DeviceSessionCreationsPerRequest",) if workspace else ()))}
        result["ComparableAndBitwiseEqual"] = True
    except Exception as error:
        result["Errors"].append(str(error))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executions", nargs=4, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.executions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ComparableAndBitwiseEqual"] else 1)
