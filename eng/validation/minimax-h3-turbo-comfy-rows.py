#!/usr/bin/env python3
"""Capture reference rows of the LightX2V MiniMax-H3 Turbo ComfyUI twins, by HTTP range reads.

The Diffusers-format adapters TensorSharp loads (minimax_h3_fl2v_turbo_*_768p_bf16.safetensors)
name to_q/to_k/to_v and ff.net.0.proj, and leave two layout facts to the loader: diffusers'
SwiGLU reads ff.net.0.proj as [value; gate] while the checkpoint's fc1 is [gate; value]
(single_file_utils.py:4319-4371, convert_minimax_h3_gated_ff), and the checkpoint's qkv_proj
is stacked [q; k; v]. The publisher's ComfyUI twin of the same adapter is already in checkpoint
layout (its metadata: swi_glu_mapping 'Diffusers [value;gate] -> ComfyUI [gate;value]',
qkv_fusion 'block diagonal B; concat A; alpha multiplied by 3'), so its rows are an
independent oracle for TensorSharp's swap and its q/k/v placement:

  * mlp.fc1.lora_B rows 0 and ffn (gate and value) must equal TensorSharp's conditioned fc1
    up rows 0 and ffn, divided by the scale alpha / rank;
  * qkv_proj.lora_B rows 0, inner and 2 inner must be zero outside column blocks 0, 1 and 2,
    and those blocks must equal row 0 of TensorSharp's q, k and v up factors.

Only the 8-byte prefix, the JSON header, the alpha scalars and the selected rows are read
(a few KiB of each 1.9 GB file). The output is the test fixture
InferenceWeb.Tests/Fixtures/MiniMaxH3/turbo-comfy-reference-rows.json; it records the
repository, the immutable revision, every file's size and publisher sha256, the header length
and the absolute byte range of every row. Stdlib only; run with `python3 -I`:

  python3 -I eng/validation/minimax-h3-turbo-comfy-rows.py \
      --output InferenceWeb.Tests/Fixtures/MiniMaxH3/turbo-comfy-reference-rows.json
"""
import argparse
import json
import re
import struct
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY = "lightx2v/Minimax-h3-Turbo"
REVISION = "a29e57d195f80ead07c89f89e55ecf2911e893c5"
BLOCK = 10
# (Diffusers file TensorSharp loads, its ComfyUI twin). A twin missing at the revision is
# recorded as absent, never skipped silently.
PAIRS = (
    ("minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors",
     "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"),
    ("minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors",
     "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"),
)
MAX_HEADER = 32 * 1024 * 1024


def fetch(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read(MAX_HEADER + 1)
    if len(data) > MAX_HEADER:
        raise ValueError("Metadata exceeds 32 MiB bound")
    return data


def byte_range(url, first, stop, total):
    """Bytes [first, stop) of a file of `total` bytes; the server must honor the range."""
    request = urllib.request.Request(url, headers={
        "Range": f"bytes={first}-{stop - 1}", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {first}-{stop - 1}/{total}":
            raise ValueError(f"Server did not honor the byte range {first}-{stop - 1} of {url}")
        data = response.read(stop - first + 1)
    if len(data) != stop - first:
        raise ValueError("Byte range length differs")
    return data


def tensor(header, name, dtype, rank):
    entry = header.get(name)
    if entry is None:
        raise ValueError(f"{name} is missing from the ComfyUI header")
    if entry["dtype"] != dtype or len(entry["shape"]) != rank:
        raise ValueError(f"{name} is {entry['dtype']}{entry['shape']}, expected a {rank}-D {dtype}")
    return entry


def capture(base, size, sha256, diffusers_file, comfy_file):
    url = base + comfy_file
    prefix = byte_range(url, 0, 8, size)
    header_bytes, = struct.unpack("<Q", prefix)
    if not 0 < header_bytes <= MAX_HEADER:
        raise ValueError("Invalid safetensors header length")
    header = json.loads(byte_range(url, 8, 8 + header_bytes, size))
    data_start = 8 + header_bytes
    stem = f"diffusion_model.blocks.{BLOCK}."
    record = dict(diffusers_file=diffusers_file, comfyui_file=comfy_file, comfyui_file_bytes=size,
                  comfyui_sha256=sha256, header_bytes=header_bytes, data_start=data_start,
                  metadata=header.get("__metadata__", {}), tensors={}, rows=[])

    def alpha(name):
        entry = tensor(header, name, "F32", 0)
        first = data_start + entry["data_offsets"][0]
        value, = struct.unpack("<f", byte_range(url, first, first + 4, size))
        return value, [first, first + 3]

    for key, module in (("fc1", "mlp.fc1"), ("qkv", "attn.qkv_proj")):
        b = tensor(header, stem + module + ".lora_B.weight", "BF16", 2)
        out, rank = b["shape"]
        a_value, a_range = alpha(stem + module + ".alpha")
        record["tensors"][key] = dict(name=stem + module + ".lora_B.weight", dtype=b["dtype"], shape=b["shape"],
                                      data_offsets=b["data_offsets"], alpha=a_value, alpha_name=stem + module + ".alpha",
                                      alpha_byte_range=a_range)
        row_bytes = rank * 2
        # fc1: the gate and value halves; qkv: the first row of q, k and v.
        picked = (0, out // 2) if key == "fc1" else (0, out // 3, 2 * out // 3)
        for row in picked:
            first = data_start + b["data_offsets"][0] + row * row_bytes
            raw = byte_range(url, first, first + row_bytes, size)
            record["rows"].append(dict(tensor=key, row=row, byte_range=[first, first + row_bytes - 1],
                                       bf16_le_hex=raw.hex()))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.revision):
        parser.error("Revision must be an immutable commit SHA")
    api = json.loads(fetch(f"https://huggingface.co/api/models/{REPOSITORY}/tree/{args.revision}"))
    files = {row["path"]: row for row in api if row.get("type") == "file"}
    base = f"https://huggingface.co/{REPOSITORY}/resolve/{args.revision}/"
    result = dict(repository=REPOSITORY, revision=args.revision, block=BLOCK,
                  tool="eng/validation/minimax-h3-turbo-comfy-rows.py", files=[], absent=[])
    for diffusers_file, comfy_file in PAIRS:
        row = files.get(comfy_file)
        if row is None:
            result["absent"].append(comfy_file)
            print(f"absent at {args.revision}: {comfy_file}")
            continue
        try:
            result["files"].append(capture(base, row["size"], (row.get("lfs") or {}).get("oid"), diffusers_file, comfy_file))
        except urllib.error.HTTPError as error:
            raise SystemExit(f"{comfy_file}: HTTP {error.code}") from error
        print(f"captured {comfy_file}")
    if not result["files"]:
        raise SystemExit("No ComfyUI twin found at the revision")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
