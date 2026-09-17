#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""Reference Qwen 3.5 M-RoPE positions from SGLang, for Qwen35MRopeReferencePositionTests.

The fixture this writes (InferenceWeb.Tests/Fixtures/Qwen35MRope/reference_positions.json)
holds, for synthetic prompts with one and two images (and a two-frame video), the
positions SGLang assigns:

* prompt positions: ``get_rope_index`` with ``model_type="qwen3_5"`` (the same function
  HF Qwen3-VL uses), the per-token (T, H, W) table and ``mrope_position_delta``;
* decode positions: ``ForwardBatch._compute_mrope_positions_decode``, i.e.
  ``delta - 1 + seq_len`` for the token that makes the sequence ``seq_len`` long;
* a follow-up turn: ``get_rope_index`` over prompt + generated tokens + new text, whose
  tail must equal the decode positions (the property that makes a cache built by
  decoding continue exactly like a re-prefill).

Nothing here needs SGLang to be installed: the one module that implements the layout
(``python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py``) imports only torch
and is loaded by path, so the reference is SGLang's own code, not a re-implementation.

Usage:
    python3 generate_reference_positions.py --sglang /path/to/sglang --sglang-revision <commit> \
        --out InferenceWeb.Tests/Fixtures/Qwen35MRope/reference_positions.json
"""
import argparse
import hashlib
import importlib.util
import json
import os

import torch

# Synthetic ids: the layout keys only on these three.
VISION_START = 900001
IMAGE_PAD = 900002
VIDEO_PAD = 900003
SPATIAL_MERGE = 2
DECODE_STEPS = 6


def load_rope_index(sglang_root):
    path = os.path.join(sglang_root, "python", "sglang", "srt", "layers",
                        "rotary_embedding", "mrope_rope_index.py")
    spec = importlib.util.spec_from_file_location("sglang_mrope_rope_index", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return module, path, digest


def text(start, n):
    return list(range(start, start + n))


def build(parts):
    """parts: ("text", n) | ("image", t, h, w) | ("video", t, h, w). Returns expanded ids,
    image grids, video grids and the spans (pad id, merged h, merged w, count) in order."""
    ids, images, videos, spans = [], [], [], []
    next_text = 1000
    for p in parts:
        if p[0] == "text":
            ids += text(next_text, p[1])
            next_text += p[1]
            continue
        kind, t, h, w = p
        mh, mw = h // SPATIAL_MERGE, w // SPATIAL_MERGE
        pad = IMAGE_PAD if kind == "image" else VIDEO_PAD
        (images if kind == "image" else videos).append([t, h, w])
        # Qwen3.5 splits a video into one span per temporal step (repeat_interleave
        # of grid t with t := 1), each opened by its own vision_start - the layout
        # TensorSharp renders for a clip, one span per frame pair.
        for _ in range(t if kind == "video" else 1):
            ids.append(VISION_START)
            ids += [pad] * (mh * mw)
            spans.append({"padTokenId": pad, "mergedHeight": mh, "mergedWidth": mw, "tokenCount": mh * mw})
            if kind == "video":
                ids += text(next_text, 2)   # the per-frame timestamp text between pairs
                next_text += 2
    return ids, images, videos, spans


def rope_index(module, ids, images, videos):
    input_ids = torch.tensor([ids], dtype=torch.long)
    pos, delta = module.get_rope_index(
        spatial_merge_size=SPATIAL_MERGE,
        image_token_id=IMAGE_PAD,
        video_token_id=VIDEO_PAD,
        vision_start_token_id=VISION_START,
        model_type="qwen3_5",
        input_ids=input_ids,
        image_grid_thw=torch.tensor(images, dtype=torch.long) if images else None,
        video_grid_thw=torch.tensor(videos, dtype=torch.long) if videos else None,
    )
    table = pos[:, 0, :].tolist()   # [3][N] (both branches: [3, batch, N])
    flat = []
    for i in range(len(ids)):
        flat += [int(table[0][i]), int(table[1][i]), int(table[2][i])]
    return flat, int(delta.reshape(-1)[0])


def decode_positions(prompt_len, delta, steps):
    # forward_batch_info._compute_mrope_positions_decode: (delta - 1) + seq_lens, where
    # seq_lens counts the token being decoded.
    return [delta - 1 + (prompt_len + k + 1) for k in range(steps)]


SCENARIOS = {
    "one_image": [("text", 5), ("image", 1, 8, 12), ("text", 7)],
    "two_images": [("text", 3), ("image", 1, 6, 4), ("text", 4), ("image", 1, 10, 10), ("text", 6)],
    "image_first_tall": [("image", 1, 16, 4), ("text", 9)],
    "text_image_text_wide": [("text", 11), ("image", 1, 4, 20), ("text", 3)],
    "two_frame_video": [("text", 4), ("video", 2, 6, 8), ("text", 5)],
    "text_only": [("text", 12)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang", required=True, help="SGLang checkout root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sglang-revision", required=True,
                    help="the SGLang commit the module comes from (recorded in the fixture)")
    args = ap.parse_args()

    module, path, digest = load_rope_index(args.sglang)
    rev = args.sglang_revision

    out = {
        "source": "sglang python/sglang/srt/layers/rotary_embedding/mrope_rope_index.py get_rope_index(model_type='qwen3_5') "
                  "+ forward_batch_info._compute_mrope_positions_decode",
        "sglangRevision": rev,
        "sourceSha256": digest,
        "torch": torch.__version__,
        "spatialMergeSize": SPATIAL_MERGE,
        "visionStartTokenId": VISION_START,
        "imagePadTokenId": IMAGE_PAD,
        "videoPadTokenId": VIDEO_PAD,
        "scenarios": [],
    }
    for name, parts in SCENARIOS.items():
        ids, images, videos, spans = build(parts)
        # A text-only prompt takes get_rope_index's arange branch (delta 0).
        positions, delta = rope_index(module, ids, images, videos)
        generated = list(range(700000, 700000 + DECODE_STEPS))
        follow_text = list(range(800000, 800004))
        follow_ids = ids + generated + follow_text
        follow_positions, follow_delta = rope_index(module, follow_ids, images, videos)
        out["scenarios"].append({
            "name": name,
            "promptTokens": ids,
            "spans": spans,
            "positions": positions,
            "delta": delta,
            "decodePositions": decode_positions(len(ids), delta, DECODE_STEPS),
            "followUpTokens": follow_ids,
            "followUpPositions": follow_positions,
            "followUpDelta": follow_delta,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {len(out['scenarios'])} scenarios from {path} (sha256 {digest[:12]}, sglang {rev[:12]})")


if __name__ == "__main__":
    main()
