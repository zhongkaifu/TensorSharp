#!/usr/bin/env python3
"""Compare already-live attention tensors from a labelled teacher diagnostic.

Only the first solo invocation following an explicit serial rebuild label is a
serial reference. Unlabelled solo calls (including round-robin) are excluded.
Native row correspondence comes from the independent residual-trace parser.
This is diagnostic evidence, never an EOS, semantic or performance pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct

SPEC = importlib.util.spec_from_file_location("layer_trace", Path(__file__).with_name("compare-qwen4exp-layer-traces.py"))
LAYER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAYER)
DIAG = re.compile(r"^\[q4e-attn-diagnostic\] path=(\w+) call=(\d+) layer=(\d+) pos=(\d+) caller=(\d+) slot=(\d+) tensor=(\w+) type=(f16|f32) ne=([0-9,]+) bytes=(\d+) hash=([0-9a-f]{16}) file=(.+)$")
NAMES = ("q_pre_norm", "k_pre_norm", "k_cache", "v_cache", "mask")


def decode(row, dump_root):
    path = Path(row["file"]).resolve()
    if not path.is_relative_to(dump_root.resolve()):
        raise ValueError("Dump path leaves the explicitly supplied directory")
    if path.stat().st_size > 64 << 20:
        raise ValueError("Tensor exceeds the native 64 MiB diagnostic bound")
    data = path.read_bytes()
    size = 2 if row["type"] == "f16" else 4
    if len(row["ne"]) != 4 or any(x <= 0 for x in row["ne"]) or math.prod(row["ne"]) * size != row["bytes"] or len(data) != row["bytes"]:
        raise ValueError("Tensor shape, type and actual byte count disagree")
    h = 1469598103934665603
    for b in data:
        h = ((h ^ b) * 1099511628211) & ((1 << 64) - 1)
    if f"{h:016x}" != row["hash"]:
        raise ValueError("Tensor bytes do not match the native hash")
    values = [x[0] for x in struct.iter_unpack("<e" if size == 2 else "<f", data)]
    if any(not math.isfinite(x) and not (row["tensor"] == "mask" and x == -math.inf) for x in values):
        raise ValueError("Invalid nonfinite tensor value")
    return data, values, hashlib.sha256(data).hexdigest()


def compare(left, right, dump_root):
    if (left["type"], left["ne"]) != (right["type"], right["ne"]):
        raise ValueError("Compared tensors have different logical geometry")
    left_root, right_root = dump_root if isinstance(dump_root, tuple) else (dump_root, dump_root)
    a, av, ash = decode(left, left_root)
    b, bv, bsh = decode(right, right_root)
    differences = [(i, x, y) for i, (x, y) in enumerate(zip(av, bv)) if x != y]
    finite = all(math.isfinite(x) and math.isfinite(y) for _, x, y in differences)
    errors = [abs(x - y) for _, x, y in differences] if finite else []
    l2 = math.fsum(x * x for x in av if math.isfinite(x))
    diff2 = math.fsum((x - y) ** 2 for _, x, y in differences) if finite else None
    locations = []
    for flat, x, y in differences[:16]:
        index, remainder = [], flat
        for dim in left["ne"]:
            index.append(remainder % dim)
            remainder //= dim
        locations.append({"index_ne0_first": index, "serial": x if math.isfinite(x) else str(x), "arena": y if math.isfinite(y) else str(y)})
    return {"tensor": left["tensor"], "type": left["type"], "ne": left["ne"], "bytes": len(a),
            "bytes_equal": a == b, "different_values": len(differences),
            "max_abs": max(errors, default=0) if finite else None,
            "relative_l2": math.sqrt(diff2 / l2) if finite and l2 else (0 if diff2 == 0 else None),
            "first_differences": locations, "serial_sha256": ash, "arena_sha256": bsh,
            "serial": left, "arena": right}


def collect(text, *, position, layer, serial_prefix):
    groups, headers, serial, arena = {}, {}, {}, {}
    pending = None
    for line_number, line in enumerate(text.splitlines(), 1):
        marker = LAYER.MARKER.fullmatch(line)
        if marker:
            pending = int(marker[2]) if marker[1].startswith(serial_prefix) and int(marker[4]) == position else None
            continue
        m = DIAG.fullmatch(line)
        if not m or int(m[3]) != layer or int(m[4]) != position:
            # A complete serial span ends its association; later unlabelled
            # calls cannot steal the label if this invocation had no dump.
            span = LAYER.SPAN.fullmatch(line)
            if span and int(span[4]) == position:
                pending = None
            continue
        key = m[1], int(m[2])
        if key not in groups:
            groups[key] = {}
            headers[key] = int(m[5]), int(m[6])
            if m[1] == "solo" and pending is not None:
                if pending in serial:
                    raise ValueError("Duplicate labelled serial invocation")
                serial[pending] = groups[key]
                pending = None
            elif m[1] == "arena":
                caller = int(m[5])
                if caller in arena:
                    raise ValueError("Duplicate arena invocation for one caller")
                arena[caller] = groups[key]
        elif headers[key] != (int(m[5]), int(m[6])):
            raise ValueError("One invocation mixes caller or slot identities")
        group = groups[key]
        if m[7] in group:
            raise ValueError("Duplicate tensor in one invocation")
        group[m[7]] = {"path": m[1], "call": int(m[2]), "layer": layer, "position": position,
            "caller": int(m[5]), "slot": int(m[6]), "tensor": m[7], "type": m[8],
            "ne": [int(x) for x in m[9].split(",")], "bytes": int(m[10]), "hash": m[11], "file": m[12], "line": line_number}
    return serial, arena


def first_prefill(text):
    points = []
    for line in text.splitlines():
        if LAYER.MARKER.fullmatch(line):
            break
        match = LAYER.SPAN.fullmatch(line)
        if match and int(match[3]) > 1:
            points.append(match.groups())
    return points


def analyze(text, dump_root, *, position=1501, layer=39, layers=48, rows=2, serial_prefix="serial-2-0-",
            serial_text=None, serial_dump_root=None, serial_rows=None):
    mapping = LAYER.analyze(text, position=position, layers=layers, rows=rows, serial_id_prefix=serial_prefix)
    if mapping["status"] not in ("localized", "matched"):
        raise ValueError("A unique complete residual-trace row correspondence is required")
    serial, arena = collect(text, position=position, layer=layer, serial_prefix=serial_prefix)
    selected = list(range(rows)) if serial_rows is None else serial_rows
    if not selected or len(set(selected)) != len(selected) or any(type(r) is not int or not 0 <= r < rows for r in selected):
        raise ValueError("Explicit serial rows must be nonempty, unique and within geometry")
    cross_run = serial_text is not None
    if cross_run:
        if serial_dump_root is None:
            raise ValueError("A separate serial dump directory is required for cross-run comparison")
        source = LAYER.analyze(serial_text, position=position, layers=layers, rows=rows, serial_id_prefix=serial_prefix)
        if source["errors"]:
            raise ValueError("Cross-run source has malformed or repeated residual observations")
        a, b = first_prefill(serial_text), first_prefill(text)
        if not a or a != b:
            raise ValueError("Cross-run first-prefill residual traces are missing or differ")
        for row in selected:
            a, b = source["serial"].get(row), mapping["serial"].get(row)
            if not a or not b or (a["request_id"], a["step"], a["token"]) != (b["request_id"], b["step"], b["token"]):
                raise ValueError("Cross-run serial identity or teacher token differs")
            signature = lambda r: [(p["layer"], p["half"], p["hash"], p["l2"]) for p in r["observations"]]
            if len(a["observations"]) != 2 * layers or signature(a) != signature(b):
                raise ValueError("Cross-run serial residuals differ or are incomplete")
        serial, _ = collect(serial_text, position=position, layer=layer, serial_prefix=serial_prefix)
    results = []
    for pair in mapping["selected_mapping"]["pairs"]:
        logical, native = pair["logical_serial_row"], pair["native_sorted_row"]
        if logical not in selected:
            continue
        a, b = serial.get(logical, {}), arena.get(native, {})
        if not all(name in a and name in b for name in NAMES):
            raise ValueError("Missing labelled serial or arena tensor; unlabelled solo is not a substitute")
        tensors = []
        for name in NAMES:
            try:
                # Separate validated directories are intentional: the inputs
                # must not be copied together and lose their provenance.
                tensor = compare(a[name], b[name], (serial_dump_root, dump_root) if cross_run else dump_root)
                tensors.append({"status": "compared", **tensor})
            except ValueError as error:
                # Preserve valid neighbouring observations, but never present
                # a corrupt/unreadable tensor as a numerical match or mismatch.
                tensors.append({"status": "invalid", "tensor": name, "error": str(error),
                                "serial": a[name], "arena": b[name]})
        results.append({"serial_row": logical, "native_sorted_row": native, "tensors": tensors})
    complete = all(t["status"] == "compared" for pair in results for t in pair["tensors"])
    return {"status": "compared" if complete else "incomplete_invalid_tensor", "position": position, "layer": layer, "mapping": mapping,
            "cross_run": cross_run, "cross_run_residual_and_first_prefill_match": True if cross_run else None,
            "pairs": results, "scope": "Only already-live q/k inputs, effective F16 KV and mask. Other attention intermediates are metadata only. Cross-run mode requires separately pinned logs plus matching labelled serial residuals and first-prefill traces, but does not prove all hidden state equal. Diagnostic execution may perturb timing/layout; this is not quality, EOS or performance acceptance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--serial-log", type=Path)
    parser.add_argument("--serial-dump-root", type=Path)
    parser.add_argument("--serial-row", type=int, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not any(output.is_relative_to((root / directory).resolve()) for directory in ("artifacts", "docs/validation")):
        parser.error("Output must resolve within workspace artifacts/ or docs/validation/")
    for path in (args.log, args.serial_log):
        if path is not None and path.stat().st_size > 128 << 20:
            parser.error("Each trace must be within the 128 MiB parser bound")
    data = args.log.read_bytes()
    if (args.serial_log is None) != (args.serial_dump_root is None):
        parser.error("--serial-log and --serial-dump-root must be supplied together")
    serial_data = args.serial_log.read_bytes() if args.serial_log else None
    report = analyze(data.decode("utf-8-sig"), args.dump_root,
        serial_text=serial_data.decode("utf-8-sig") if serial_data is not None else None,
        serial_dump_root=args.serial_dump_root, serial_rows=args.serial_row)
    report.update(log=str(args.log.resolve()), log_sha256=hashlib.sha256(data).hexdigest())
    if serial_data is not None:
        report.update(serial_log=str(args.serial_log.resolve()), serial_log_sha256=hashlib.sha256(serial_data).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "pairs": len(report["pairs"])}))
    return 0 if report["status"] == "compared" else 2


if __name__ == "__main__":
    raise SystemExit(main())
