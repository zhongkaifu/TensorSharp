#!/usr/bin/env python3
"""Summarize diagnostic Nsight SQLite events; this is not a speed benchmark.

Full-logit D2H copies delimit observable requests. Without NVTX these boundaries
include host sampling/hash work, and cannot distinguish warmup from measured
requests. Repeated modal kernel counts select consistent decode-like intervals;
they do not prove that a trace with dropped events is complete.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--vocab", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.vocab <= 0:
        parser.error("--vocab must be positive")
    connection = sqlite3.connect(args.trace.resolve().as_uri() + "?mode=ro", uri=True)
    # Some Nsight builds export invalid UTF-8 in unrelated demangled names.
    # Preserve the diagnostic with replacement rather than losing all events.
    connection.text_factory = lambda value: value.decode("utf-8", "replace")
    names = dict(connection.execute("select id,value from StringIds"))
    kernels = connection.execute(
        "select start,end,shortName,gridX,gridY,blockX from CUPTI_ACTIVITY_KIND_KERNEL order by start"
    ).fetchall()
    runtime = connection.execute(
        "select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start"
    ).fetchall()
    # CUPTI memcpy kind 2 is device-to-host (validated against the export enum).
    copies = connection.execute(
        "select start,end from CUPTI_ACTIVITY_KIND_MEMCPY where bytes=? and copyKind=2 order by start",
        (args.vocab * 4,),
    ).fetchall()
    intervals = []
    for index in range(1, len(copies)):
        begin, end = copies[index - 1][1], copies[index][1]
        events = [row for row in kernels if begin <= row[0] and row[1] <= end]
        by_name = defaultdict(lambda: {"count": 0, "milliseconds": 0.0})
        shapes = defaultdict(lambda: {"count": 0, "milliseconds": 0.0})
        for start, stop, name_id, gx, gy, bx in events:
            name = names[name_id]
            by_name[name]["count"] += 1
            by_name[name]["milliseconds"] += (stop - start) / 1e6
            if name == "q8_f32_vector":
                key = f"grid={gx},{gy};block={bx}"
                shapes[key]["count"] += 1
                shapes[key]["milliseconds"] += (stop - start) / 1e6
        apis = defaultdict(lambda: {"count": 0, "milliseconds": 0.0})
        for start, stop, name_id in runtime:
            if begin <= start and stop <= end:
                name = names[name_id]
                apis[name]["count"] += 1
                apis[name]["milliseconds"] += (stop - start) / 1e6
        intervals.append({
            "interval_index": index, "start_ns": begin, "end_ns": end,
            "logit_copy_interval_ms": (end - begin) / 1e6,
            "kernel_count": len(events),
            "kernel_sum_ms": sum(row[1] - row[0] for row in events) / 1e6,
            "q8_vector_count": by_name["q8_f32_vector"]["count"],
            "q8_vector_ms": by_name["q8_f32_vector"]["milliseconds"],
            "kernels": dict(by_name), "q8_launch_shapes": dict(shapes), "apis": dict(apis),
        })
    signatures = Counter((row["kernel_count"], row["q8_vector_count"])
                         for row in intervals if row["q8_vector_count"] > 1)
    modal = signatures.most_common(1)[0][0] if signatures else None
    selected = [row for row in intervals
                if (row["kernel_count"], row["q8_vector_count"]) == modal]
    diagnostics = [dict(zip([col[1] for col in connection.execute("pragma table_info(DIAGNOSTIC_EVENT)")], row))
                   for row in connection.execute("select * from DIAGNOSTIC_EVENT")]
    summary = {
        "trace": str(args.trace.resolve()), "logit_copies_observed": len(copies),
        "kernel_events_observed": len(kernels), "modal_signature": modal,
        "consistent_intervals": len(selected),
        "median": {key: statistics.median(row[key] for row in selected) for key in
                   ("logit_copy_interval_ms", "kernel_sum_ms", "q8_vector_ms")} if selected else {},
        "diagnostics": diagnostics,
        "limitations": "Diagnostic trace, not a throughput claim. No NVTX phase markers: copy intervals include host work and may include warmup. Modal counts are a consistency filter, not proof that events were not dropped. Kernel sums may overlap; API durations are not additive with device work. Inspect driver/trace warnings and all excluded intervals.",
        "intervals": intervals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("intervals", "diagnostics")}, indent=2))


if __name__ == "__main__":
    main()
