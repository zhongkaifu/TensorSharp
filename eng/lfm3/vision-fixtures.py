#!/usr/bin/env python3
"""Pack llama.cpp's LFM3-Omni vision dumps (eng/lfm3/vision-reference.sh, one directory per mmproj type) into the
compact fixture InferenceWeb.Tests/Fixtures/Lfm3Vision/reference.json that Lfm3VisionTests checks against: per case
the chunk layout, the SHA-256 of each image chunk's graph input (float32 little-endian, as llama.cpp fed it), and per
mmproj type the row count, the mean / RMS over all rows, three sample rows (first, middle, last) as F16 and the encode
time.

usage: eng/lfm3/vision-fixtures.py <out json> <case spec>... -- <type>=<ref dir>...
  case specs as given to the harness (synth:..., synthvideo:..., file:NAME:PATH with PATH under
  InferenceWeb.Tests/Fixtures, stored relative to it)
"""
import base64
import hashlib
import json
import os
import sys

import numpy as np


def main(argv):
    if "--" not in argv or len(argv) < 4:
        print(__doc__)
        return 2
    sep = argv.index("--")
    out, specs, dirs = argv[1], argv[2:sep], dict(a.split("=", 1) for a in argv[sep + 1:])
    fixtures_root = os.path.join(os.path.dirname(os.path.abspath(out)), "..")
    cases = []
    for spec in specs:
        parts = spec.split(":")
        kind, name = parts[0], parts[1]
        case = {"name": name, "kind": kind}
        if kind == "synth":
            case.update(width=int(parts[2]), height=int(parts[3]), seed=int(parts[4]))
        elif kind == "synthvideo":
            case.update(width=int(parts[2]), height=int(parts[3]), frames=int(parts[4]), fps=float(parts[5]))
        elif kind == "file":
            path = spec[len(kind) + len(name) + 2:]
            case["file"] = os.path.relpath(os.path.abspath(path), os.path.abspath(fixtures_root)).replace(os.sep, "/")
        else:
            raise SystemExit(f"case kind {kind} does not go into fixtures")
        layout = None
        for qtype, ref in dirs.items():
            meta = json.load(open(os.path.join(ref, name + ".json")))
            pixels = np.fromfile(os.path.join(ref, name + ".pixels.f32"), dtype="<f4")
            embd = np.fromfile(os.path.join(ref, name + ".embd.f32"), dtype="<f4").reshape(-1, 2048)
            chunks, at = [], 0
            for c in meta["chunks"]:
                if c["type"] == "text":
                    chunks.append({"text": c["text"], "tokens": c["tokens"]})
                else:
                    px = pixels[at:at + c["pixels"]]
                    at += c["pixels"]
                    chunks.append({"rows": c["rows"], "pixels": c["pixels"],
                                   "pixels_sha256": hashlib.sha256(px.astype("<f4").tobytes()).hexdigest()})
            if layout is None:
                layout = chunks
                if "served_frames" in meta:
                    case["served_frames"] = meta["served_frames"]
            elif layout != chunks:
                raise SystemExit(f"{name}: the layouts of the mmproj types differ")
            rows = embd.shape[0]
            sample = [0, rows // 2, rows - 1]
            case.setdefault("mmproj", {})[qtype] = {
                "rows": rows,
                "mean": float(embd.astype(np.float64).mean()),
                "rms": float(np.sqrt((embd.astype(np.float64) ** 2).mean())),
                "encode_ms": round(meta["encode_ms_total"], 2),
                "sample_rows": sample,
                "sample_f16": base64.b64encode(embd[sample].astype("<f2").tobytes()).decode("ascii"),
            }
        case["chunks"] = layout
        cases.append(case)
    doc = {
        "about": "llama.cpp mtmd (LFM3-Omni-2B mmproj, lfm3vl) on ggml-metal: layouts, graph-input hashes and sample "
                 "rows for Lfm3VisionTests; regenerate with eng/lfm3/vision-reference.sh + eng/lfm3/vision-fixtures.py",
        "llama_cpp": os.environ.get("LLAMA_CPP_REV", ""),
        "mmproj": os.environ.get("MMPROJ_REV", ""),
        "cases": cases,
    }
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"{out}: {len(cases)} cases, {os.path.getsize(out)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
