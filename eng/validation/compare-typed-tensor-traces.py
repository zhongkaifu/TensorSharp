#!/usr/bin/env python3
"""Validate pinned typed tensor snapshots and localize common-prefix differences.

Native trace output retention can change graph liveness/fusion. This tool reports
diagnostic observations, never numerical, semantic, performance or hard-cap PASS.
The run-pin contract and packed/original-stride distinction are documented in
typed-tensor-traces.md. Requires numpy; no model/backend execution is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

SCHEMAS = {"tensorsharp.tensor-trace.v1", "tensorsharp.glm-trace.v1"}
MANIFEST = re.compile(r"pass-(\d{6})\.json\Z")
DATA = re.compile(r"pass-(\d{6})\.tensor-(\d{4})\.bin\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
SCOPE = ("Typed intermediate diagnostics with explicit common input history; trace retention changes liveness/fusion. "
         "Equality is not a model quality, EOS, performance, shared-ledger or physical-hard-cap acceptance gate.")


def integer(value, name, minimum=0, maximum=(1 << 63) - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid integer {name}")
    return value


def vector(value, name, minimum=0):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} requires four dimensions")
    return [integer(item, name, minimum) for item in value]


def text(value, name):
    if not isinstance(value, str) or not value or len(value) > 512 or "\0" in value:
        raise ValueError(f"Invalid {name}")
    return value


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fnv1a64(data):
    result = 14695981039346656037
    for byte in data:
        result = ((result ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{result:016x}"


def load_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Nonfinite JSON number")
        return result
    try:
        value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=invalid, parse_float=number)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Malformed UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def regular_file(root, name):
    if not isinstance(name, str) or "/" in name or "\\" in name or ":" in name or name in {".", ".."}:
        raise ValueError("Trace files require plain relative filenames")
    path = root / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
        raise ValueError("Trace file escapes its root or is not a regular file")
    return path


def read_capture(directory, pin_path, *, manifest_name=None, expected_pin_sha256=None,
                 maximum_tensor_bytes=256 << 20, maximum_capture_bytes=512 << 20):
    integer(maximum_tensor_bytes, "maximum_tensor_bytes", 4)
    integer(maximum_capture_bytes, "maximum_capture_bytes", 4)
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Trace root must be a directory")
    pin_path = Path(pin_path).resolve(strict=True)
    if pin_path.is_relative_to(root):
        raise ValueError("External run pin must stay outside the immutable trace root")
    with pin_path.open("rb") as stream:
        pin_data = stream.read((16 << 20) + 1)
    if len(pin_data) > 16 << 20:
        raise ValueError("Run pin exceeds parser bound")
    if expected_pin_sha256 is not None and (not SHA.fullmatch(expected_pin_sha256) or sha256(pin_data) != expected_pin_sha256):
        raise ValueError("External run-pin SHA256 mismatch")
    pin = load_json(pin_data)
    if pin.get("schema") != "tensorsharp.tensor-trace-pins.v1" or pin.get("capture_root") != str(root):
        raise ValueError("Run pin schema/root mismatch")
    if pin.get("fresh_directory") is not True or integer(pin.get("controller_exit_code"), "controller_exit_code") != 0:
        raise ValueError("Successful execution controller and a fresh directory are required")
    producer_exit = integer(pin.get("producer_exit_code"), "producer_exit_code", -(1 << 31), (1 << 32) - 1)
    expected_exit = integer(pin.get("expected_producer_exit_code"), "expected_producer_exit_code", -(1 << 31), (1 << 32) - 1)
    if producer_exit != expected_exit:
        raise ValueError("Producer exit differs from the controller's explicit expected exit")
    for key in ("producer_binary_sha256", "producer_build_receipt_sha256", "model_identity_sha256",
                "probe_report_sha256", "execution_receipt_sha256"):
        if not isinstance(pin.get(key), str) or not SHA.fullmatch(pin[key]):
            raise ValueError("Missing identity SHA256: " + key)
    ids = pin.get("input_ids")
    if not isinstance(ids, list) or not ids or len(ids) > 1 << 20:
        raise ValueError("Run pin requires actual input IDs")
    for token in ids:
        integer(token, "input_id", maximum=(1 << 31) - 1)
    files = sorted(path.name for path in root.iterdir())
    if any(name.endswith(".incomplete") for name in files):
        raise ValueError("Incomplete native trace marker remains")
    if not files or any(not (MANIFEST.fullmatch(name) or DATA.fullmatch(name)) for name in files):
        raise ValueError("Unknown or temporary file in fresh capture")
    if sum(regular_file(root, name).stat().st_size for name in files) > maximum_capture_bytes:
        raise ValueError("Capture exceeds parser aggregate byte bound")
    frozen = pin.get("files")
    if not isinstance(frozen, dict) or set(frozen) != set(files) or any(not isinstance(h, str) or not SHA.fullmatch(h) for h in frozen.values()):
        raise ValueError("Run-pin files do not cover exactly the fresh capture")
    snapshots = {}
    def read_file(name, limit):
        path = regular_file(root, name)
        if path.stat().st_size > limit:
            raise ValueError("Trace file exceeds parser bound: " + name)
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("Trace file exceeds parser bound: " + name)
        digest = sha256(raw)
        if digest != frozen[name]:
            raise ValueError("Pinned trace SHA256 mismatch: " + name)
        snapshots[name] = {"bytes": len(raw), "sha256": digest}
        return raw
    manifests = []
    referenced = set()
    for name in files:
        match = MANIFEST.fullmatch(name)
        if not match:
            continue
        meta = load_json(read_file(name, 16 << 20))
        if meta.get("schema") not in SCHEMAS or integer(meta.get("pass_id"), "pass_id") != int(match[1]):
            raise ValueError("Manifest schema/pass filename mismatch")
        integer(meta.get("model_id"), "model_id", 1)
        nt = integer(meta.get("nt"), "nt", 1)
        integer(meta.get("slot_id"), "slot_id", -1)
        if meta.get("kind") not in {"trunk", "batched", "mtp"} or meta.get("endian") not in {"little", "big"}:
            raise ValueError("Invalid pass kind or endian")
        n_kv = integer(meta.get("n_kv"), "n_kv", 0 if meta["kind"] == "batched" else 1)
        if "p0" not in meta:
            raise ValueError("Pass must explicitly declare its context p0")
        if meta["p0"] is not None:
            integer(meta["p0"], "p0")
        sequences = meta.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise ValueError("Pass requires actual sequence identities")
        for seq in sequences:
            if not isinstance(seq, dict):
                raise ValueError("Malformed sequence identity")
            integer(seq.get("slot_id"), "sequence slot_id")
            integer(seq.get("position"), "sequence position")
        if n_kv == 0:
            # A combined joint-graph extent can be undefined. This sentinel is
            # metadata, never a measurement of zero allocated KV memory.
            if meta["p0"] is not None or len(sequences) != nt or len({(s["slot_id"], s["position"]) for s in sequences}) != nt:
                raise ValueError("Undefined joint n_kv requires explicit unique sequence identity for every token row")
        rows_limit = integer(meta.get("rows_limit"), "rows_limit", 1)
        maximum = integer(meta.get("max_bytes"), "max_bytes", 1)
        overrides = meta.get("token_axis_overrides")
        if not isinstance(overrides, dict):
            raise ValueError("Pass requires explicit token-axis override dictionary")
        for tag, axis in overrides.items():
            text(tag, "override tag"); integer(axis, "override axis", maximum=3)
        rows, keys, total = [], set(), 0
        if not isinstance(meta.get("traces"), list) or not meta["traces"] or len(meta["traces"]) > 10000:
            raise ValueError("Missing or excessive trace records")
        for ordinal, row in enumerate(meta["traces"]):
            if not isinstance(row, dict) or integer(row.get("ordinal"), "ordinal") != ordinal:
                raise ValueError("Trace records must preserve unique contiguous registration order")
            tag = text(row.get("tag"), "tag"); text(row.get("name"), "name")
            layer = integer(row.get("layer"), "layer", -1); rank = integer(row.get("rank"), "rank")
            key = (tag, layer, rank)
            if key in keys:
                raise ValueError("Duplicate trace tag/layer/rank")
            keys.add(key)
            integer(row.get("type_code"), "type_code")
            if (row.get("type"), row.get("type_code")) not in {("f32", 0), ("i32", 26)}:
                raise ValueError("Unsupported or inconsistent tensor type")
            ne, nb, shape = vector(row.get("ne"), "ne", 1), vector(row.get("nb"), "nb"), vector(row.get("shape"), "shape", 1)
            if any(stride % 4 for stride in nb):
                raise ValueError("Tensor strides are not element-aligned")
            source_bytes = integer(row.get("source_bytes"), "source_bytes", 4)
            span = 4 + sum((extent - 1) * stride for extent, stride in zip(ne, nb))
            if span > source_bytes or source_bytes % 4:
                raise ValueError("Original tensor strides leave their source byte span")
            candidates = [axis for axis, extent in enumerate(ne) if extent == nt]
            declared_candidates = row.get("candidate_token_axes")
            if not isinstance(declared_candidates, list) or any(type(a) is not int for a in declared_candidates) or declared_candidates != candidates:
                raise ValueError("Token-axis candidates do not match the complete original shape")
            axis = integer(row.get("token_axis"), "token_axis", maximum=3)
            if axis not in candidates:
                raise ValueError("Token axis does not represent this pass's nt")
            source = row.get("token_axis_source")
            if source == "override":
                if overrides.get(tag) != axis:
                    raise ValueError("Token-axis override is not declared in this pass")
            elif source != "unique-extent" or len(candidates) != 1 or tag in overrides:
                raise ValueError("Ambiguous token axis or inconsistent override source")
            wanted = ne.copy(); wanted[axis] = min(nt, rows_limit)
            if shape != wanted or integer(row.get("slice_start"), "slice_start") != 0 or integer(row.get("token_count"), "token_count", 1) != wanted[axis]:
                raise ValueError("Tensor slice is not the declared initial token prefix")
            elements = math.prod(shape)
            if integer(row.get("logical_elements"), "logical_elements", 1) != elements or integer(row.get("bytes"), "bytes", 1) != elements * 4:
                raise ValueError("Logical shape/elements/full byte count disagree")
            if "snapshot_checked_elements" in row or "snapshot_source" in row:
                checked = integer(row.get("snapshot_checked_elements"), "snapshot_checked_elements")
                snapshot = row.get("snapshot_source")
                if checked not in {0, elements} or (snapshot is None) != (checked == 0):
                    raise ValueError("Snapshot check must cover the complete dumped logical tensor or be absent")
                if snapshot is not None:
                    if not isinstance(snapshot, dict):
                        raise ValueError("Malformed independent snapshot descriptor")
                    text(snapshot.get("kind"), "snapshot kind"); text(snapshot.get("op"), "snapshot op")
                    tensor_name = snapshot.get("tensor_name")
                    if not isinstance(tensor_name, str) or len(tensor_name) > 512 or "\0" in tensor_name:
                        raise ValueError("Invalid snapshot tensor_name")
                    integer(snapshot.get("layer"), "snapshot layer", -1)
                    integer(snapshot.get("rank"), "snapshot rank")
                    snapshot_ne, snapshot_nb = vector(snapshot.get("ne"), "snapshot ne", 1), vector(snapshot.get("nb"), "snapshot nb")
                    snapshot_bytes = integer(snapshot.get("source_bytes"), "snapshot source_bytes", 4)
                    if snapshot_ne != ne or any(s % 4 for s in snapshot_nb) or snapshot_bytes % 4 or 4 + sum((n - 1) * s for n, s in zip(snapshot_ne, snapshot_nb)) > snapshot_bytes:
                        raise ValueError("Snapshot source geometry/strides/byte span disagree")
            data_name = row.get("file")
            data_match = DATA.fullmatch(data_name) if isinstance(data_name, str) else None
            if not data_match or int(data_match[1]) != meta["pass_id"] or int(data_match[2]) != ordinal or data_name in referenced:
                raise ValueError("Tensor requires a fresh relative pass/ordinal filename")
            referenced.add(data_name)
            raw = read_file(data_name, maximum_tensor_bytes)
            if len(raw) != elements * 4:
                raise ValueError("Truncated or oversized packed tensor")
            if not isinstance(row.get("content_fnv1a64"), str) or fnv1a64(raw) != row["content_fnv1a64"]:
                raise ValueError("Native tensor content FNV1a64 mismatch")
            dtype = ("<" if meta["endian"] == "little" else ">") + ("f4" if row["type"] == "f32" else "i4")
            values = np.frombuffer(raw, dtype=dtype).reshape(shape, order="F")
            if row["type"] == "f32" and not np.isfinite(values).all():
                raise ValueError("Nonfinite F32 trace tensor")
            rows.append({"metadata": row, "key": key, "values": values})
            total += len(raw)
        if integer(meta.get("total_bytes"), "total_bytes", 1) != total or total > maximum:
            raise ValueError("Pass aggregate bytes exceed or disagree with its bound")
        manifests.append({"file": name, "metadata": meta, "rows": rows})
    if referenced != {name for name in files if DATA.fullmatch(name)}:
        raise ValueError("Orphan tensor has no committed final manifest")
    choices = [m for m in manifests if manifest_name is None or m["file"] == manifest_name]
    if len(choices) != 1:
        raise ValueError("Select exactly one completed pass; missing/ambiguous manifest")
    return {"root": root, "pin_path": pin_path, "pin": pin, "pin_sha256": sha256(pin_data),
            "external_pin_sha256_verified": expected_pin_sha256 is not None, "files": snapshots, "selected": choices[0]}


def unchanged(capture):
    with capture["pin_path"].open("rb") as stream:
        pin_data = stream.read((16 << 20) + 1)
    if sha256(pin_data) != capture["pin_sha256"]:
        raise ValueError("Run pin changed during comparison")
    if sorted(path.name for path in capture["root"].iterdir()) != sorted(capture["files"]):
        raise ValueError("Capture file inventory changed during comparison")
    for name, pin in capture["files"].items():
        with regular_file(capture["root"], name).open("rb") as stream:
            raw = stream.read(pin["bytes"] + 1)
        if len(raw) != pin["bytes"] or sha256(raw) != pin["sha256"]:
            raise ValueError("Trace changed during comparison: " + name)


def compare_captures(reference, candidate, *, prefix_tokens=512):
    integer(prefix_tokens, "prefix_tokens", 1)
    a, b = reference["selected"], candidate["selected"]
    am, bm = a["metadata"], b["metadata"]
    if am["p0"] is None or bm["p0"] is None or len(am["sequences"]) != 1 or len(bm["sequences"]) != 1:
        raise ValueError("Joint sequence token axes require a separate explicit row mapping")
    if am["p0"] != bm["p0"] or am["kind"] != bm["kind"] or min(am["nt"], bm["nt"]) < prefix_tokens:
        raise ValueError("Pass contexts/kinds cannot cover the same common token prefix")
    for meta in (am, bm):
        if meta["sequences"][0]["position"] != meta["p0"] or meta["sequences"][0]["slot_id"] != meta["slot_id"]:
            raise ValueError("Solo pass sequence identity disagrees with its context")
    history = am["p0"] + prefix_tokens
    ap, bp = reference["pin"], candidate["pin"]
    if ap["model_identity_sha256"] != bp["model_identity_sha256"] or min(len(ap["input_ids"]), len(bp["input_ids"])) < history or ap["input_ids"][:history] != bp["input_ids"][:history]:
        raise ValueError("Model identity or common consumed input history differs")
    ar, br = {r["key"]: r for r in a["rows"]}, {r["key"]: r for r in b["rows"]}
    ordered = [r["key"] for r in a["rows"]] == [r["key"] for r in b["rows"]]
    observations = []
    for row in a["rows"]:
        key = row["key"]
        if key not in br:
            continue
        other = br[key]
        xmeta, ymeta = row["metadata"], other["metadata"]
        axis = xmeta["token_axis"]
        xshape, yshape = xmeta["shape"].copy(), ymeta["shape"].copy()
        if axis != ymeta["token_axis"] or xmeta["type"] != ymeta["type"] or min(xshape[axis], yshape[axis]) < prefix_tokens:
            raise ValueError("Matched tensor type/token-axis/prefix extent differs")
        xshape[axis] = yshape[axis] = prefix_tokens
        if xshape != yshape:
            raise ValueError("Matched tensors have incompatible non-token geometry")
        take = [slice(None)] * 4; take[axis] = slice(0, prefix_tokens)
        dtype = "<f4" if xmeta["type"] == "f32" else "<i4"
        x = np.array(row["values"][tuple(take)], dtype=dtype, order="F").ravel(order="F")
        y = np.array(other["values"][tuple(take)], dtype=dtype, order="F").ravel(order="F")
        different = x.view("<u4") != y.view("<u4")
        indices = np.flatnonzero(different)
        item = {"tag": key[0], "layer": key[1], "rank": key[2], "type": xmeta["type"],
                "reference_ordinal": xmeta["ordinal"], "candidate_ordinal": ymeta["ordinal"],
                "shape": xshape, "token_axis": axis, "elements": int(x.size),
                "bitwise_differences": int(indices.size), "first_differences": []}
        item["producer_snapshot_checks"] = [
            {"checked_elements": meta.get("snapshot_checked_elements", 0), "source": meta.get("snapshot_source")}
            for meta in (xmeta, ymeta)]
        for index in indices[:16]:
            coordinates = [int(v) for v in np.unravel_index(int(index), xshape, order="F")]
            item["first_differences"].append({"index_ne0_first": coordinates,
                "absolute_token": am["p0"] + coordinates[axis], "reference": x[index].item(), "candidate": y[index].item()})
        if xmeta["type"] == "i32":
            item["exact_ids_equal"] = bool(indices.size == 0)
        else:
            xf, yf = x.astype(np.float64), y.astype(np.float64)
            delta = yf - xf; error = np.abs(delta)
            aa, bb, ab, dd = float(xf @ xf), float(yf @ yf), float(xf @ yf), float(delta @ delta)
            item.update(numeric_differences=int(np.count_nonzero(delta)), max_absolute=float(error.max()),
                mean_absolute=float(error.mean()), mean_signed=float(delta.mean()), rmse=math.sqrt(dd / x.size),
                relative_l2=math.sqrt(dd / aa) if aa else (0.0 if dd == 0 else None),
                cosine=ab / math.sqrt(aa * bb) if aa and bb else float(aa == bb),
                absolute_error_quantiles={str(q): float(v) for q, v in zip((0.5, 0.9, 0.99, 0.999), np.quantile(error, (0.5, 0.9, 0.99, 0.999)))})
        observations.append(item)
    unchanged(reference); unchanged(candidate)
    first = next((r for r in observations if r["bitwise_differences"]), None)
    first_id = next((r for r in observations if r["type"] == "i32" and not r["exact_ids_equal"]), None)
    return {"schema": "tensorsharp.tensor-trace-comparison.v1", "scope": SCOPE,
        "status": "incomplete" if not ordered else "localized" if first else "matched",
        "diagnostic_complete": ordered, "numerical_quality_passed": None, "semantic_quality_passed": None,
        "performance_accepted": None, "prefix_tokens": prefix_tokens, "p0": am["p0"],
        "verified_input_history_extent": history,
        "capture_pass_extents": [{"p0": m["p0"], "nt": m["nt"]} for m in (am, bm)],
        "registration_order_identical": ordered, "common_input_history_verified": True,
        "producer_binary_identical": ap["producer_binary_sha256"] == bp["producer_binary_sha256"],
        "first_divergent_observation": first if ordered else None, "first_differing_i32_observation": first_id if ordered else None,
        "missing_from_candidate": [list(k) for k in ar if k not in br], "missing_from_reference": [list(k) for k in br if k not in ar],
        "observations": observations,
        "sources": [{"root": str(c["root"]), "manifest": c["selected"]["file"], "run_pin": str(c["pin_path"]),
            "run_pin_sha256": c["pin_sha256"], "external_pin_sha256_verified": c["external_pin_sha256_verified"],
            "run_bindings": c["pin"], "files_before_after_identical": True, "file_pins": c["files"]} for c in (reference, candidate)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for side in ("reference", "candidate"):
        parser.add_argument("--" + side, type=Path, required=True)
        parser.add_argument("--" + side + "-pin", type=Path, required=True)
        parser.add_argument("--" + side + "-pin-sha256")
        parser.add_argument("--" + side + "-manifest")
    parser.add_argument("--prefix-tokens", type=int, default=512)
    parser.add_argument("--max-tensor-bytes", type=int, default=256 << 20)
    parser.add_argument("--max-capture-bytes", type=int, default=512 << 20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    for directory in (args.reference.resolve(), args.candidate.resolve()):
        if destination.is_relative_to(directory):
            raise ValueError("Comparison report must not modify an immutable trace root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("Never overwrite an executed comparison report")
    try:
        captures = [read_capture(getattr(args, side), getattr(args, side + "_pin"),
            manifest_name=getattr(args, side + "_manifest"), expected_pin_sha256=getattr(args, side + "_pin_sha256"),
            maximum_tensor_bytes=args.max_tensor_bytes, maximum_capture_bytes=args.max_capture_bytes)
            for side in ("reference", "candidate")]
        report = compare_captures(*captures, prefix_tokens=args.prefix_tokens)
    except (ValueError, OSError, KeyError) as error:
        report = {"schema": "tensorsharp.tensor-trace-comparison.v1", "status": "invalid", "scope": SCOPE,
            "numerical_quality_passed": None, "semantic_quality_passed": None, "performance_accepted": None, "error": str(error)}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(destination)}))
    return 0 if report["status"] in {"matched", "localized"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
