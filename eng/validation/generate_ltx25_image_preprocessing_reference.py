#!/usr/bin/env python3
"""Generate LTX image preprocessing fixtures from an unchanged official checkout.

Requires CPU torch, einops, numpy and PyAV. Does not load a model or use a GPU.
Example:
  python eng/validation/generate_ltx25_image_preprocessing_reference.py \
    --reference-root artifacts/ltx25/reference-original \
    --output InferenceWeb.Tests/Fixtures/Ltx25/image-preprocessing.json
"""
import argparse
import ast
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

import av
import numpy as np
import torch
from einops import rearrange


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.reference_root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(args.reference_root), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("The official reference checkout must be unchanged")
    media = args.reference_root / "packages/ltx-pipelines/src/ltx_pipelines/utils/media_io"
    namespace = dict(av=av, np=np, torch=torch, BytesIO=io.BytesIO, math=math, rearrange=rearrange)
    sources = {}
    # Execute the original function ASTs without importing unrelated pipeline/model
    # dependencies. Function bodies and constants are not rewritten or reimplemented.
    for filename, names in {
        "decode.py": {"encode_single_frame", "decode_single_frame", "preprocess"},
        "resize.py": {"resize_and_center_crop"},
        "range_map.py": {"normalize_images"},
    }.items():
        source = (media / filename).read_bytes()
        sources[filename] = hashlib.sha256(source).hexdigest()
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if len(tree.body) != len(names):
            raise RuntimeError(f"Missing official functions in {filename}")
        exec(compile(ast.fix_missing_locations(tree), str(media / filename), "exec"), namespace)

    def pattern(width, height):
        return np.fromfunction(lambda y, x, c: (x * 17 + y * 29 + c * 71 + x * y * 3) % 256,
                               (height, width, 3)).astype(np.uint8)

    def resized(source, width, height):
        result = namespace["resize_and_center_crop"](torch.tensor(source, dtype=torch.float32), height, width)
        result = namespace["normalize_images"](result, torch.device("cpu"), torch.float32)
        return rearrange(result, "1 c 1 h w -> h w c").flatten().tolist()

    resize_cases = []
    for name, sw, sh, dw, dh, fractional in [
        ("upscale_odd_landscape", 9, 5, 7, 9, False),
        ("downscale_landscape", 19, 11, 7, 5, False),
        ("downscale_portrait", 11, 19, 5, 7, False),
        ("fractional_values", 7, 9, 11, 5, True),
        ("singleton_width", 1, 7, 3, 5, False),
        ("singleton_target", 5, 3, 1, 1, False),
    ]:
        source = pattern(sw, sh).astype(np.float32)
        if fractional:
            source = (source * np.float32(.9713) + np.float32(.3271)).astype(np.float32)
        resize_cases.append(dict(name=name, width=sw, height=sh, target_width=dw, target_height=dh,
                                 source=source.flatten().tolist(), normalized=resized(source, dw, dh)))

    codec_cases = []
    for sw, sh in [(35, 27), (67, 131)]:
        source = pattern(sw, sh)
        compressed = namespace["preprocess"](source, 18)
        codec_cases.append(dict(width=sw, height=sh, compressed_width=compressed.shape[1], compressed_height=compressed.shape[0],
                               source_rgb_base64=base64.b64encode(source.tobytes()).decode(),
                               compressed_rgb_base64=base64.b64encode(compressed.tobytes()).decode(),
                               target_width=9, target_height=7, normalized=resized(compressed, 9, 7)))
    # Actual uploaded-image geometry after the official odd-edge clip. Sample
    # border/interior pixels without embedding megabytes of synthetic source data.
    large_resize_cases = []
    for size in (512, 1024):
        source = pattern(1252, 836)
        result = namespace["resize_and_center_crop"](torch.tensor(source, dtype=torch.float32), size, size)
        result = namespace["normalize_images"](result, torch.device("cpu"), torch.float32)
        flat = rearrange(result, "1 c 1 h w -> h w c").flatten()
        points = [0, 1, 2, size // 3, size // 2, size - 3, size - 2, size - 1]
        indices = [(y * size + x) * 3 + c for y in points for x in points for c in range(3)]
        large_resize_cases.append(dict(width=1252, height=836, target_width=size, target_height=size,
                                       source_formula="(x*17 + y*29 + c*71 + x*y*3) % 256",
                                       indices=indices, normalized=flat[indices].tolist()))
    output = dict(source_revision=revision, source_sha256=sources, torch_version=torch.__version__,
                  pyav_version=av.__version__, av_library_versions=av.library_versions,
                  recipe="libx264, rate=1, crf=18, preset=veryfast, RGB24->YUV420P PyAV reformat, default slice threading; float32 bilinear align_corners=False then center crop and /127.5-1",
                  resize_cases=resize_cases, codec_cases=codec_cases, large_resize_cases=large_resize_cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {args.output} from unchanged official revision {revision}")


if __name__ == "__main__":
    main()
