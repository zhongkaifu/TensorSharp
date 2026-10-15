#!/usr/bin/env python3
"""Localize differences in opt-in Gemma4 resident/streaming tensor dumps.

Requires numpy. Use TS_GEMMA4_TENSOR_DUMP and optionally
TS_GEMMA4_TENSOR_DUMP_LAYERS when running UnifiedMemory.WeightModelProbe.
Snapshots synchronize and retain intermediates, so their run timings are not
performance evidence. This report diagnoses stages; only the model probe can
qualify end-to-end logits. Store reports in artifacts/ or docs/validation/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


STAGES = [
    "embeddingScaled", "pleTokenScaled", "pleProjectionRaw", "pleProjectionScaled",
    "pleProjectionNormed", "pleCombined", "attnNorm", "qProjected", "kProjected",
    "vProjected", "qNormRope", "kNormRope", "vNormed", "attention",
    "attentionProjected", "attentionResidual", "ffnNorm", "ffnActivation",
    "ffnResidual", "output",
]
NAME = re.compile(r"^(native|stream)\.(p\d+\.n\d+\.t[0-9a-fA-F]+)\.(.+)\.f32$")


def order(key):
    tag, stage = key
    layer = re.match(r"layer(\d+)\.(.+)", stage)
    name = layer[2] if layer else stage
    return tag, int(layer[1]) if layer else -1, STAGES.index(name) if name in STAGES else len(STAGES), name


def identity(path):
    meta = path.with_suffix(".json")
    return {
        "path": str(path.resolve()), "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "metadata": json.loads(meta.read_text(encoding="utf-8-sig")) if meta.exists() else None,
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(reference, observed):
    if reference.stat().st_size == 0 or reference.stat().st_size % 4:
        raise ValueError(f"Invalid F32 byte count: {reference}")
    if reference.stat().st_size != observed.stat().st_size:
        raise ValueError(f"Tensor byte counts differ: {reference} / {observed}")
    a32 = np.fromfile(reference, dtype="<f4")
    b32 = np.fromfile(observed, dtype="<f4")
    if not np.isfinite(a32).all() or not np.isfinite(b32).all():
        return {"finite": False, "elements": int(a32.size), "within_logits_error_thresholds": False}
    a, b = a32.astype(np.float64), b32.astype(np.float64)
    delta = b - a
    aa, bb, ab = float(a @ a), float(b @ b), float(a @ b)
    relative_l2 = float(np.sqrt(float(delta @ delta) / max(aa, 1e-300)))
    cosine = ab / np.sqrt(aa * bb) if aa and bb else float(aa == bb)
    worst = int(np.argmax(np.abs(delta)))
    return {
        "finite": True, "elements": int(a32.size),
        "bitwise_differences": int(np.count_nonzero(a32.view("<u4") != b32.view("<u4"))),
        "relative_l2": relative_l2, "cosine": float(cosine),
        "max_absolute": float(abs(delta[worst])), "worst_flat_index": worst,
        "reference_at_worst": float(a[worst]), "streamed_at_worst": float(b[worst]),
        "within_logits_error_thresholds": bool(relative_l2 <= 0.001 and cosine >= 0.999999),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--tag", help="Compare just one p<position>.n<count>.t<tokenhash> tag")
    args = parser.parse_args()
    files = {"native": {}, "stream": {}}
    for path in args.directory.glob("*.f32"):
        match = NAME.match(path.name)
        if match and (args.tag is None or match[2] == args.tag):
            files[match[1]][(match[2], match[3])] = path
    keys = files["native"].keys() & files["stream"].keys()
    if not keys:
        parser.error("No matching native/stream tensor pairs found.")
    observations, errors = [], []
    first_changed, first_outside = {}, {}
    for key in sorted(keys, key=order):
        tag, stage = key
        reference, streamed = files["native"][key], files["stream"][key]
        try:
            item = {"tag": tag, "stage": stage, "reference": identity(reference),
                    "streamed": identity(streamed), **compare(reference, streamed)}
            observations.append(item)
            if item.get("bitwise_differences", 1) and tag not in first_changed:
                first_changed[tag] = stage
            if not item["within_logits_error_thresholds"] and tag not in first_outside:
                first_outside[tag] = stage
            print(f"{tag} {stage:28s} relL2={item.get('relative_l2', float('nan')):.9g} "
                  f"cos={item.get('cosine', float('nan')):.12g} max={item.get('max_absolute', float('nan')):.9g}")
        except (ValueError, OSError) as error:
            errors.append({"tag": tag, "stage": stage, "error": str(error)})
    missing = {
        "streamed": [".".join(k) for k in sorted(files["native"].keys() - keys, key=order)],
        "reference": [".".join(k) for k in sorted(files["stream"].keys() - keys, key=order)],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "scope": "Intermediate diagnostics, not end-to-end parity or a performance benchmark. Missing stages are unverified.",
        "first_bitwise_difference": first_changed, "first_outside_logits_error_thresholds": first_outside,
        "compared_pairs": len(observations), "missing": missing, "errors": errors,
        "observations": observations,
    }, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Compared {len(observations)} pairs; errors={len(errors)}; missing={sum(map(len, missing.values()))}")
    print("First stages outside logits error thresholds:", first_outside)
    return int(bool(errors) or any(not item["finite"] for item in observations))


if __name__ == "__main__":
    raise SystemExit(main())
