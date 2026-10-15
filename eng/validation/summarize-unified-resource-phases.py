#!/usr/bin/env python3
"""Read-only phase resource observations from pinned HTTP logs and owned samples.

Usage: python summarize-unified-resource-phases.py <owned-run-directory>
Creates fresh resource-phase-summary.json; original reports are never changed.
Windows .NET process startTicks are used only for this launcher evidence schema.
This is not a memory ownership or OS hard-limit qualification.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def timestamp(value):
    # .NET logs have seven fractional digits. Python 3.10 accepts at most six;
    # truncate sub-microsecond precision, negligible beside 1s observations.
    normalized = re.sub(r"(\.\d{6})\d+(?=[+-]\d{2}:\d{2}$)", r"\1", value.replace("Z", "+00:00"))
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("HTTP timestamps require a timezone")
    return parsed.timestamp()


def request_windows(report, records, mode):
    rows = report.get(mode)
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("Each phase needs two actual HTTP requests")
    targets = [row.get("request_id") for row in rows]
    if any(not isinstance(x, str) or not x for x in targets) or len(set(targets)) != 2:
        raise ValueError("Missing or duplicate HTTP request identity")
    result = []
    for row in rows:
        identity = row["request_id"]
        starts, ends = [], []
        for record in records:
            if record.get("scope", {}).get("RequestId") != identity:
                continue
            event = record.get("event", {}).get("name")
            if event not in ("HttpRequestStarted", "HttpRequestCompleted"):
                continue
            props = record.get("props", {})
            if props.get("Method") != "POST" or props.get("Path") != "/api/chat":
                raise ValueError("Current request boundary is not POST /api/chat")
            if event == "HttpRequestStarted":
                starts.append(timestamp(record["ts"]))
            else:
                if props.get("StatusCode") != 200:
                    raise ValueError("Current HTTP request did not complete successfully")
                ends.append(timestamp(record["ts"]))
        if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
            raise ValueError("Missing, duplicate or reversed HTTP request boundaries")
        result.append({"name": row.get("name"), "request_id": identity,
                       "start_unix": starts[0], "end_unix": ends[0]})
    return result


def observations(samples, start, end):
    rows = [row for row in samples if start <= row["unix"] <= end]
    result = {"sample_count": len(rows), "start_unix": start, "end_unix": end,
              "window_seconds": end - start,
              "first_sample_after_start_seconds": rows[0]["unix"] - start if rows else None,
              "last_sample_before_end_seconds": end - rows[-1]["unix"] if rows else None,
              "maximum_internal_sample_gap_seconds": max(
                  (b["unix"] - a["unix"] for a, b in zip(rows, rows[1:])), default=None)}
    for key in ("rss", "private"):
        values = [row[key] for row in rows if finite(row.get(key))]
        result[f"process_sampled_peak_{key}_bytes"] = max(values, default=None)
        result[f"{key}_sample_count"] = len(values)
    devices = {}
    invalid_board = 0
    for row in rows:
        value = row.get("board_gpu")
        if not isinstance(value, str):
            invalid_board += 1
            continue
        found = False
        for line in value.splitlines():
            fields = [part.strip() for part in line.split(",")]
            try:
                if len(fields) != 4 or not fields[0].startswith("GPU-"):
                    raise ValueError("Unknown nvidia-smi row")
                used = float(fields[1])
                if not finite(used):
                    raise ValueError("Unavailable board observation")
            except ValueError:
                invalid_board += 1
                continue
            entry = devices.setdefault(fields[0], {"sample_count": 0, "sampled_peak_used_mib": 0})
            entry["sample_count"] += 1
            entry["sampled_peak_used_mib"] = max(entry["sampled_peak_used_mib"], used)
            found = True
        if not found and not value.strip():
            invalid_board += 1
    result["whole_board_devices"] = devices
    result["whole_board_sampled_peak_used_mib"] = next(iter(devices.values()))["sampled_peak_used_mib"] if len(devices) == 1 else None
    result["invalid_or_missing_board_observations"] = invalid_board
    return result


def validate_samples(samples, owner):
    pid, ticks = owner.get("pid"), owner.get("startTicks")
    if type(pid) is not int or pid <= 0 or type(ticks) is not int:
        raise ValueError("Missing owned PID/startTicks")
    created = (ticks - 621355968000000000) / 10_000_000
    previous = None
    for row in samples:
        now = row.get("unix")
        if not finite(now) or (previous is not None and now < previous):
            raise ValueError("Missing/nonmonotonic sample timestamp")
        if row.get("pid") != pid or not finite(row.get("created")) or abs(row["created"] - created) > .001:
            raise ValueError("Sample process identity differs from owned PID/start time")
        previous = now


def summarize(directory):
    directory = Path(directory)
    sample_path, owner_path = directory / "memory-samples.jsonl", directory / "owned-process.json"
    inputs = {str(p.resolve()): sha(p) for p in (sample_path, owner_path)}
    samples = [json.loads(line) for line in sample_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    owner = load(owner_path)
    validate_samples(samples, owner)
    phases = []
    for label, folder in (("short", directory), ("extended", directory / "extended")):
        if label == "extended" and not folder.exists():
            continue
        report_path, log_path = folder / "concurrency/report.json", folder / "concurrency/fused-server-log.jsonl"
        report = load(report_path)
        if report.get("fused_evidence", {}).get("server_log", {}).get("sha256") != sha(log_path):
            raise ValueError("HTTP log differs from report's captured log identity")
        inputs.update({str(p.resolve()): sha(p) for p in (report_path, log_path)})
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        for mode in ("concurrent", "serial"):
            windows = request_windows(report, records, mode)
            start, end = min(r["start_unix"] for r in windows), max(r["end_unix"] for r in windows)
            phase = {"phase": f"{label}_{mode}", **observations(samples, start, end)}
            phase["requests"] = [{**r, "observations": observations(samples, r["start_unix"], r["end_unix"])} for r in windows]
            phases.append(phase)
    if any(sha(path) != digest for path, digest in inputs.items()):
        raise ValueError("Resource inputs changed during analysis")
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": inputs, "tool_sha256": sha(__file__), "owned_process": owner,
            "phases": phases, "scope": [
                "Boundaries are server JSONL HTTP start/completion timestamps bound to each actual report RequestId, not SSE frame counts or inferred GPU activity.",
                "A phase is the bounding interval of its two HTTP requests, including any gap between serial controls. Per-request observations overlap and must not be summed.",
                "Only timestamped samples inside the interval are used; no interpolation or lifetime peak_rss is used. Roughly one-second polling can miss shorter peaks; actual gaps and endpoint distances are recorded.",
                "RSS/private are overlapping process observations; GPU values are whole-board observations, not owned VRAM or allocation attribution. Other activity on the same board is not subtracted.",
                "Short and extended requests reuse the same process/model/retained caches; phase peaks are not independent cold starts or per-request incremental memory.",
                "This sidecar does not change quality/performance gates, establish complete physical ownership, or reinterpret the independent semantic review."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    result = summarize(args.directory)
    output = args.directory / "resource-phase-summary.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(output.resolve()), "sha256": sha(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
