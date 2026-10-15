#!/usr/bin/env python3
"""Localize residual-hash differences in a Qwen4Exp teacher trace.

This is diagnostic evidence, not a numerical, language-quality or performance
acceptance gate. Native arena rows are sorted by holder pointer. Never assume
that their row numbers are the labelled serial request numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re


MARKER = re.compile(r"^diagnostic serial rebuild: id=([^,]+), row=(\d+), step=(\d+), position=(\d+), token=(\d+)$")
SPAN = re.compile(r"^\[q4e-trace\] span (\d+)\.\.(\d+) T=(\d+) pos=(\d+):(.*)$")
POINT = re.compile(r"(\d+):([^@]+)@([0-9a-fA-F]{16})")
ARENA = re.compile(r"^\[q4e-arena-trace\] row=(\d+) pos=(\d+) l=(\d+) half=(\d+) l2=(\S+) hash=(\S+)(?: head=.*)?$")
SCOPE = ("Residual hashes and printed L2 norms only. Trace/rebuild can change graph layout; "
         "this does not establish full-tensor tolerance, model quality, EOS or performance. "
         "Original execution and numerical assertion results must be reported separately.")


def analyze(text: str, *, position: int, layers: int = 48, rows: int = 2,
            serial_id_prefix: str | None = None) -> dict:
    if position < 0 or layers < 1 or not 1 <= rows <= 8:
        raise ValueError("Require position >= 0, layers >= 1 and 1..8 rows")
    expected = [(layer, half) for layer in range(layers) for half in (0, 1)]
    serial, arena, errors = {}, {}, []
    active = None

    def observation(layer, half, norm, hash_value, line):
        try:
            value = float(norm)
        except ValueError:
            value = float("nan")
        if not math.isfinite(value) or not re.fullmatch(r"[0-9a-fA-F]{16}", hash_value):
            errors.append(f"line {line}: malformed or nonfinite observation")
            return None
        if (layer, half) not in expected:
            errors.append(f"line {line}: layer/half outside the requested geometry")
            return None
        return {"layer": layer, "half": half, "l2": value, "hash": hash_value.lower(), "line": line}

    def append_point(points, point):
        if point is None:
            return
        key = point["layer"], point["half"]
        if any((p["layer"], p["half"]) == key for p in points):
            errors.append(f"line {point['line']}: duplicate layer/half")
        points.append(point)

    for number, line in enumerate(text.splitlines(), 1):
        marker = MARKER.fullmatch(line)
        if marker:
            # An unrelated marker must not lend its following trace to an
            # incomplete selected serial block.
            active = None
            if int(marker[4]) != position or (serial_id_prefix is not None and not marker[1].startswith(serial_id_prefix)):
                continue
            row = int(marker[2])
            if row >= rows:
                errors.append(f"line {number}: serial row outside the requested geometry")
                continue
            if row in serial:
                errors.append(f"line {number}: duplicate serial marker; select one cohort with --serial-id-prefix")
                continue
            serial[row] = {"request_id": marker[1], "step": int(marker[3]), "token": int(marker[5]),
                           "marker_line": number, "observations": []}
            active = row
            continue
        span = SPAN.fullmatch(line)
        if span and active is not None and int(span[3]) == 1 and int(span[4]) == position:
            begin, end = int(span[1]), int(span[2])
            words = span[5].split()
            if not 0 <= begin < end <= layers or len(words) != 2 * (end - begin):
                errors.append(f"line {number}: incomplete or invalid serial span")
            for word in words:
                point = POINT.fullmatch(word)
                if point is None:
                    errors.append(f"line {number}: malformed serial point")
                    continue
                index = int(point[1])
                if not 0 <= index < 2 * (end - begin):
                    errors.append(f"line {number}: serial index outside span")
                    continue
                append_point(serial[active]["observations"], observation(
                    begin + index // 2, index % 2, point[2], point[3], number))
            if len(serial[active]["observations"]) >= len(expected):
                active = None
            continue
        match = ARENA.fullmatch(line)
        if match and int(match[2]) == position:
            row = int(match[1])
            if row >= rows:
                errors.append(f"line {number}: native row outside the requested geometry")
                continue
            append_point(arena.setdefault(row, []), observation(
                int(match[3]), int(match[4]), match[5], match[6], number))

    complete = (set(serial) == set(arena) == set(range(rows))
                and all([(p["layer"], p["half"]) for p in group["observations"]] == expected for group in serial.values())
                and all([(p["layer"], p["half"]) for p in group] == expected for group in arena.values()))
    if len({s["request_id"] for s in serial.values()}) != len(serial):
        errors.append("Serial labels repeat the same request identity")

    candidates = []
    if complete and not errors:
        for permutation in itertools.permutations(range(rows)):
            pairs = []
            for logical, native in enumerate(permutation):
                left, right = serial[logical]["observations"], arena[native]
                prefix, first = 0, None
                for a, b in zip(left, right):
                    if a["hash"] != b["hash"] or a["l2"] != b["l2"]:
                        first = {"serial": a, "arena": b}
                        break
                    prefix += 1
                pairs.append({"logical_serial_row": logical, "native_sorted_row": native,
                              "equal_prefix_count": prefix, "first_difference": first})
            candidates.append({"score": sum(pair["equal_prefix_count"] for pair in pairs), "pairs": pairs})
        candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    unique = bool(candidates and (len(candidates) == 1 or candidates[0]["score"] > candidates[1]["score"]))
    selected = candidates[0] if unique else None
    status = ("invalid" if errors else "incomplete" if not complete else "ambiguous" if not unique
              else "localized" if any(pair["first_difference"] for pair in selected["pairs"]) else "matched")
    return {"schema_version": 1, "status": status, "position": position, "layers": layers, "rows": rows,
            "trace_complete": complete, "mapping_unique": unique, "errors": errors,
            "serial": serial, "arena": arena, "selected_mapping": selected,
            # Keep the runner-up to make uniqueness reviewable without a large
            # 8! report. Every permutation was considered before selection.
            "best_candidates": candidates[:2], "mapping_candidates_evaluated": len(candidates),
            "mapping_basis": "Unique longest exact residual-hash/L2 prefix, not request-number equality.", "scope": SCOPE}


def output_path(path: Path, workspace: Path) -> Path:
    resolved = path.resolve()
    allowed = [(workspace / "artifacts").resolve(), (workspace / "docs/validation").resolve()]
    if not any(resolved.is_relative_to(root) and resolved != root for root in allowed):
        raise ValueError("Generated output must be inside workspace artifacts/ or docs/validation/")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--serial-id-prefix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    destination = output_path(args.output, args.workspace_root)
    if args.log.stat().st_size > 128 << 20:
        raise ValueError("Trace exceeds the 128 MiB parser bound")
    raw = args.log.read_bytes()
    report = analyze(raw.decode("utf-8", errors="replace"), position=args.position, layers=args.layers,
                     rows=args.rows, serial_id_prefix=args.serial_id_prefix)
    report["source_log"] = {"path": str(args.log.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(destination), "selected_mapping": report["selected_mapping"]}))
    # Finding a mismatch is successful diagnostic execution, never a model PASS.
    return 0 if report["status"] in ("localized", "matched") else 2


if __name__ == "__main__":
    raise SystemExit(main())
