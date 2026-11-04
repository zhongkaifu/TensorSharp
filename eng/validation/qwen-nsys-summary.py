#!/usr/bin/env python3
"""Summarize Nsight Systems GPU workload intervals, transfers and host API waits.

Kernel categories are name heuristics; top names remain visible for review.
Durations are inclusive (possibly overlapping) except explicitly marked unions.
Whole-graph spans may contain internal gaps. Neither uncovered wall time nor
CUDA synchronization waits establish CPU compute, host idle or SM utilization.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[2]


def union_ns(intervals):
    total = 0
    end = None
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if end is None or a > end:
            total += b - a
        elif b > end:
            total += b - end
        end = b if end is None else max(end, b)
    return total


def category(name):
    name = name.lower()
    if re.search(r"gated.?delta|gdn|delta.?net", name):
        return "gdn"
    if re.search(r"flash.?attn|flash.?attention|fattn|attention|(?:^|_)attn(?:_|$)", name):
        return "attention"
    if re.search(r"dequantize|quantize|convert", name):
        return "quantization"
    if re.search(r"gemm|gemv|matmul|mul_mat|mul_mm|mmq|cutlass|cublas", name):
        return "matrix_multiply"
    if re.search(r"copy|cpy|memset|set_rows|get_rows", name):
        return "memory_kernel"
    return "other"


def summarize(connection, request=None):
    connection.row_factory = sqlite3.Row
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings = dict(connection.execute('SELECT id, value FROM StringIds')) if 'StringIds' in tables else {}
    all_intervals = []
    detailed_intervals = []
    graph_intervals = []
    warnings = []
    graph_node_kernels = 0
    kernel_events = 0
    copy_memset_events = 0
    graph_launch_api_records = 0
    graph_invalid_intervals = 0
    groups = defaultdict(lambda: {"count": 0, "sum_ms": 0.0, "bytes": 0})
    kernels = defaultdict(lambda: {"count": 0, "sum_ms": 0.0, "category": "other"})
    api_tables = {}
    missing = []
    events = 0

    def columns(table):
        return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}

    def name_for(row, keys):
        for key in keys:
            if key in row.keys() and row[key] is not None:
                return strings.get(row[key], str(row[key]))
        return "unknown"

    kernel_table = next((t for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL") if t in tables), None)
    if kernel_table:
        for row in connection.execute(f'SELECT * FROM "{kernel_table}"'):
            start, end = row["start"], row["end"]
            if end <= start:
                continue
            name = name_for(row, ("demangledName", "shortName", "mangledName", "nameId"))
            kind = category(name)
            duration = (end - start) / 1e6
            groups[kind]["count"] += 1
            groups[kind]["sum_ms"] += duration
            kernels[name]["count"] += 1
            kernels[name]["sum_ms"] += duration
            kernels[name]["category"] = kind
            all_intervals.append((start, end))
            detailed_intervals.append((start, end))
            kernel_events += 1
            if "graphNodeId" in row.keys() and isinstance(row["graphNodeId"], int) and row["graphNodeId"] > 0:
                graph_node_kernels += 1
            events += 1
    else:
        missing.append("GPU kernel table absent")
    copy_labels = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}
    enum_table = "ENUM_CUDA_MEMCPY_OPER"
    if enum_table in tables and {"id", "label"} <= columns(enum_table):
        copy_labels.update(dict(connection.execute(f'SELECT id, label FROM "{enum_table}"')))
    for table, kind_key in (("CUPTI_ACTIVITY_KIND_MEMCPY", "copyKind"), ("CUPTI_ACTIVITY_KIND_MEMSET", None)):
        if table not in tables:
            continue
        for row in connection.execute(f'SELECT * FROM "{table}"'):
            start, end = row["start"], row["end"]
            if end <= start:
                continue
            label = "memset" if kind_key is None else "memcpy_" + copy_labels.get(row[kind_key], str(row[kind_key]))
            groups[label]["count"] += 1
            groups[label]["sum_ms"] += (end - start) / 1e6
            groups[label]["bytes"] += row["bytes"] if "bytes" in row.keys() else 0
            all_intervals.append((start, end))
            detailed_intervals.append((start, end))
            copy_memset_events += 1
            events += 1
    graph_table = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
    graph_schema_supported = graph_table in tables and {"start", "end"} <= columns(graph_table)
    if graph_schema_supported:
        for row in connection.execute(f'SELECT start, end FROM "{graph_table}"'):
            start, end = row["start"], row["end"]
            if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                graph_invalid_intervals += 1
                continue
            graph_intervals.append((start, end))
            all_intervals.append((start, end))
            groups["cuda_graph_aggregate"]["count"] += 1
            groups["cuda_graph_aggregate"]["sum_ms"] += (end - start) / 1e6
            events += 1
        if graph_intervals:
            groups["cuda_graph_aggregate"].update(inclusive=True, additive_with_kernel_or_copy_categories=False,
                attribution="Whole-graph execution spans; internal node gaps may be included. No per-kernel attribution.")
    elif graph_table in tables:
        missing.append("CUDA graph trace table lacks supported start/end interval columns")
        warnings.append("Graph activity is present with an unsupported interval schema; the GPU union may omit graph execution.")
    if graph_invalid_intervals:
        warnings.append("Invalid CUDA graph intervals were omitted; GPU interval coverage may be incomplete.")
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER", "OSRT_API"):
        if table not in tables:
            continue
        buckets = defaultdict(lambda: {"count": 0, "inclusive_ms": 0.0})
        intervals = []
        for row in connection.execute(f'SELECT * FROM "{table}"'):
            start, end = row["start"], row["end"]
            if end <= start:
                continue
            name = name_for(row, ("nameId", "name"))
            if table != "OSRT_API" and "GraphLaunch" in name:
                graph_launch_api_records += 1
            buckets[name]["count"] += 1
            buckets[name]["inclusive_ms"] += (end - start) / 1e6
            intervals.append((start, end))
        api_tables[table] = {"all_threads_interval_union_ms": union_ns(intervals) / 1e6,
            "inclusive_ms": sum(v["inclusive_ms"] for v in buckets.values()),
            "top": [{"name": k, **v} for k, v in sorted(buckets.items(), key=lambda p: -p[1]["inclusive_ms"])[:40]]}
    covered_ms = union_ns(all_intervals) / 1e6
    detailed_ms = union_ns(detailed_intervals) / 1e6
    graph_aggregate_only = bool(graph_intervals) and graph_node_kernels == 0
    graph_activity_missing = graph_launch_api_records > 0 and not graph_intervals and graph_node_kernels == 0
    if graph_aggregate_only:
        graph_attribution = "UNAVAILABLE_AGGREGATE_ONLY"
        warnings.append("CUDA graphs have aggregate spans but no kernel records with positive graphNodeId. Kernel names/categories do not attribute graph bodies; missing node durations are not host idle.")
    elif graph_node_kernels:
        graph_attribution = "NODE_RECORDS_PRESENT_COMPLETENESS_NOT_ESTABLISHED"
    elif graph_activity_missing:
        graph_attribution = "MISSING_GRAPH_DEVICE_COVERAGE"
        warnings.append("Host graph-launch APIs were recorded without graph spans or identified node kernels. The GPU union may omit graph bodies; no host-idle estimate is valid.")
    elif graph_table in tables and not graph_schema_supported:
        graph_attribution = "UNSUPPORTED_GRAPH_INTERVAL_SCHEMA"
    else:
        graph_attribution = "NO_GRAPH_EXECUTION_OBSERVED"
    span_ms = (max(b for _, b in all_intervals) - min(a for a, _ in all_intervals)) / 1e6 if all_intervals else None
    output = {"gpu_events": events, "gpu_activity_union_ms": covered_ms if events else None,
              "gpu_activity_union_scope": "Observed workload-interval union of kernels, copies, memsets and whole-graph spans across devices/streams. Graph spans may contain internal gaps; this is not SM utilization or exact kernel-busy time.",
              "gpu_event_record_counts": {"kernels": kernel_events, "copies_and_memsets": copy_memset_events, "graph_aggregates": len(graph_intervals)},
              "gpu_detailed_activity_union_ms": detailed_ms if detailed_intervals else None,
              "cuda_graph_trace": {"table_present": graph_table in tables, "interval_schema_supported": graph_schema_supported,
                  "valid_aggregate_events": len(graph_intervals), "invalid_intervals_omitted": graph_invalid_intervals,
                  "aggregate_inclusive_sum_ms": sum(b - a for a, b in graph_intervals) / 1e6,
                  "aggregate_interval_union_ms": union_ns(graph_intervals) / 1e6 if graph_intervals else None,
                  "aggregate_interval_without_individual_activity_ms": covered_ms - detailed_ms if graph_intervals else None,
                  "kernel_graph_node_id_column_present": kernel_table is not None and "graphNodeId" in columns(kernel_table),
                  "kernel_records_with_positive_graph_node_id": graph_node_kernels,
                  "host_graph_launch_api_records": graph_launch_api_records,
                  "aggregate_only": graph_aggregate_only, "per_kernel_body_attribution": graph_attribution,
                  "per_kernel_body_attribution_complete": False if graph_aggregate_only or graph_activity_missing else None},
              "kernel_attribution_scope": "Individually recorded kernels only. Presence of some node records does not prove complete attribution of every graph body.",
              "warnings": warnings,
              "gpu_first_to_last_span_ms": span_ms,
              "gpu_gap_between_first_and_last_ms": span_ms - covered_ms if span_ms is not None else None,
              "gpu_categories": dict(groups),
              "top_kernels": [{"name": k, **v} for k, v in sorted(kernels.items(), key=lambda p: -p[1]["sum_ms"])[:40]],
              "host_api_tables": api_tables, "missing": missing,
              "limitations": ["Capture includes the complete single request, including decoding; it is not inherently a TTFT-only window.",
                  "Profiler overhead, graph-node tracing and optional CPU sampling perturb latency; never use these times as unprofiled performance.",
                  "GPU category sums are inclusive and non-additive. Whole-graph spans overlap their node kernels/copies; the combined interval union counts each timestamp once.",
                  "Aggregate graph spans can contain internal gaps/dependencies. Their union is observed GPU-workload coverage, not exact kernel execution time or SM utilization. Per-kernel graph-body attribution requires node records.",
                  "Kernel categories use name heuristics. CUDA runtime/driver/OS API durations can overlap each other and GPU work; they include waiting.",
                  "Process CPU counters are summed across threads and include runtime/background work. Host-to-host memcpy is not individually traced.",
                  "No CPU samples is not zero CPU work. Missing GPU tables is not passing GPU coverage."]}
    if request:
        result = request.get("result", {})
        output["request"] = {k: result.get(k) for k in ("scenario", "step", "ttft_ms", "e2e_ms", "prompt_tokens", "cached_tokens", "prefilled_tokens", "completion_tokens", "finish_reason")}
        output["server_process_cpu"] = request.get("server_process_cpu")
        output["server_process_faults"] = request.get("server_process_faults")
        if events and isinstance(result.get("e2e_ms"), (float, int)):
            # Capture deliberately encloses one request only. This is a gap estimate,
            # not an attempt to align client clocks to CUPTI timestamps.
            output["request_wall_minus_gpu_union_ms"] = result["e2e_ms"] - covered_ms
            output["request_wall_minus_gpu_union_interpretation"] = "Unclassified client-wall minus observed GPU-workload interval union; clocks/window boundaries are not aligned. This residual is not CPU execution, host idle, or a TTFT breakdown."
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not any(output.is_relative_to(ROOT / p) for p in ("docs/validation", "artifacts")):
        parser.error("Output must be in ignored docs/validation/ or artifacts/")
    if output.exists():
        parser.error("Output already exists")
    with sqlite3.connect(args.sqlite.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        report = summarize(connection, json.loads(args.request.read_text()) if args.request else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": str(output), "gpu_events": report["gpu_events"],
                      "gpu_activity_union_ms": report["gpu_activity_union_ms"], "missing": report["missing"]}))


if __name__ == "__main__":
    main()
