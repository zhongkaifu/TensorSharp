#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""Regenerate the llama.cpp image goldens that InferenceWeb.Tests/MuseGlimmerVisionParityTests.cs checks Muse-Glimmer's
vision tower and its image prompts against.

The script starts a llama-server on the given GGUF and mmproj, asks it for greedy continuations of image prompts (the
four eng/minimax-m3-bench-media questions, descriptions of the two non-square ones, and left/right and top/bottom
questions about two synthetic pictures this script draws), records each prompt, its image, llama.cpp's prompt length
(the image's rows included) and every generated token with its top log-probabilities, and writes
InferenceWeb.Tests/Fixtures/MuseGlimmer/LlamaCpp/ref_vision.json (the pictures go to
InferenceWeb.Tests/Fixtures/MuseGlimmer/Vision/). The prompts are rendered as muse-glimmer-llamacpp-goldens.py renders
them (the template's rules without its date line), the image part first, with the direct-answer suffix (thinking off);
llama.cpp gets them with its media marker in
place of the template's <|patch|>, which mtmd expands to <|image_start|> + the image's rows + <|image_end|>, the layout
TensorSharp's injector builds.

    python3 eng/validation/muse-glimmer-llamacpp-vision-goldens.py \\
        --llama-cpp ~/work/llama.cpp-lfm3 \\
        --model ~/work/models/muse-glimmer-30b/Muse-Glimmer-30B-UD-IQ2_XXS.gguf \\
        --mmproj ~/work/models/muse-glimmer-30b/mmproj-Muse-Glimmer-30B-Q8_0.gguf

Only the standard library is used. Run it under the Mac job lock and memguard like any real-weight run.
"""
import argparse
import base64
import datetime
import json
import os
import struct
import subprocess
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib

goldens = importlib.import_module("muse-glimmer-llamacpp-goldens")

REPO = goldens.REPO
# The chat protocol's direct-answer suffix (ChatProtocolRegistry's muse-glimmer DirectAnswerGenerationSuffix): the reply
# goes to the user, not to the reasoning channel, as TensorSharp renders a turn with thinking off.
DIRECT = " to=user<|message|>"
# llama-server randomizes its media marker unless LLAMA_MEDIA_MARKER pins it (tools/server/server-common.cpp).
MARKER = "<__media__>"
MEDIA = os.path.join("eng", "minimax-m3-bench-media")
SYNTHETIC = os.path.join("InferenceWeb.Tests", "Fixtures", "MuseGlimmer", "Vision")

# (name, image path relative to the repository, question, tokens to generate)
CASES = [
    ("number", os.path.join(MEDIA, "number-7319.png"), "What number is written in this image? Reply with the digits only.", 16),
    ("shapes", os.path.join(MEDIA, "shapes-red-circle.jpg"),
     "This image shows a circle and a square. What color is the circle? Reply with one word.", 16),
    ("ocean", os.path.join(MEDIA, "word-ocean.jpg"), "What word is written in this image? Reply with the word only.", 16),
    ("count", os.path.join(MEDIA, "count-green-squares.png"), "How many green squares are in this image? Reply with a number.", 16),
    ("number-describe", os.path.join(MEDIA, "number-7319.png"), "Describe this image in detail.", 64),
    ("ocean-describe", os.path.join(MEDIA, "word-ocean.jpg"), "Describe this image in detail.", 64),
    ("wide-side", os.path.join(SYNTHETIC, "wide-left-right.png"),
     "Is the red circle on the left side or the right side of this image? Reply with one word.", 16),
    ("wide-describe", os.path.join(SYNTHETIC, "wide-left-right.png"),
     "Describe where each shape is in this image and what colour it is.", 64),
    ("tall-side", os.path.join(SYNTHETIC, "tall-top-bottom.png"),
     "Is the blue square at the top or at the bottom of this image? Reply with one word.", 16),
]


def png(path, width, height, pixels):
    """An 8-bit RGB PNG from raw rows (stdlib only)."""
    raw = b"".join(b"\x00" + pixels[y * width * 3:(y + 1) * width * 3] for y in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def draw(width, height, shapes):
    """White canvas with filled circles ('c', cx, cy, r, rgb) and squares ('s', x, y, side, rgb)."""
    px = bytearray(b"\xff\xff\xff" * (width * height))
    for shape in shapes:
        kind, a, b, size, rgb = shape
        for y in range(height):
            for x in range(width):
                inside = ((x - a) ** 2 + (y - b) ** 2 <= size * size) if kind == "c" else (a <= x < a + size and b <= y < b + size)
                if inside:
                    px[(y * width + x) * 3:(y * width + x) * 3 + 3] = bytes(rgb)
    return bytes(px)


def synthesize():
    """The two spatial pictures: 448 x 168 (a 32 x 12 patch grid: both sides within the 32-patch position table, so only the
    table's order matters) with a red circle left and a blue square right, and 168 x 448 with them top and bottom."""
    out = os.path.join(REPO, SYNTHETIC)
    os.makedirs(out, exist_ok=True)
    red, blue = (220, 30, 30), (30, 60, 220)
    png(os.path.join(out, "wide-left-right.png"), 448, 168,
        draw(448, 168, [("c", 84, 84, 56, red), ("s", 308, 28, 112, blue)]))
    png(os.path.join(out, "tall-top-bottom.png"), 168, 448,
        draw(168, 448, [("c", 84, 84, 56, red), ("s", 28, 308, 112, blue)]))


def complete(port, prompt_string, image_path, n_predict):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = {
        "prompt": {"prompt_string": prompt_string, "multimodal_data": [b64]},
        "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
        "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0,
        "cache_prompt": False, "return_tokens": True, "n_probs": goldens.N_PROBS, "post_sampling_probs": False,
        "samplers": ["top_k"],
    }
    return goldens.post(port, "/completion", body)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-cpp", required=True, help="llama.cpp checkout with a built build/bin/llama-server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "InferenceWeb.Tests", "Fixtures", "MuseGlimmer", "LlamaCpp"))
    ap.add_argument("--log-dir", default=os.path.join(REPO, "artifacts", "glimmer-goldens"))
    ap.add_argument("--port", type=int, default=8898)
    a = ap.parse_args()

    synthesize()
    os.environ["LLAMA_MEDIA_MARKER"] = MARKER
    binary = os.path.join(a.llama_cpp, "build", "bin", "llama-server")
    commit = subprocess.check_output(["git", "-C", a.llama_cpp, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", a.llama_cpp, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    version = subprocess.run([binary, "--version"], capture_output=True, text=True).stderr.strip().splitlines()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.log_dir, exist_ok=True)

    base_args = ["-m", a.model, "--mmproj", a.mmproj, "-c", "8192", "-ngl", "99", "-fa", "on", "-np", "1", "--no-webui"]
    records = []
    with goldens.Server(binary, base_args, a.port, os.path.join(a.log_dir, "llama-server-vision.log")):
        for name, image, question, n_predict in CASES:
            user = "<|patch|>" + question
            prompt = goldens.render(user) + DIRECT
            res = complete(a.port, prompt.replace("<|patch|>", MARKER), os.path.join(REPO, image), n_predict)
            rec = goldens.record(name, user, prompt, [], res, with_probs=True)
            del rec["prompt_tokens"]
            rec["image"] = image.replace(os.sep, "/")
            rec["image_sha256"] = goldens.sha256(os.path.join(REPO, image))
            rec["prompt_n"] = res.get("timings", {}).get("prompt_n")
            records.append(rec)
            print(f"[goldens] {name}: {rec['prompt_n']} prompt positions, {len(res['tokens'])} generated: {res['content'][:80]!r}",
                  flush=True)

    provenance = {
        "llama_cpp": {"commit": commit, "dirty": bool(dirty), "version": version, "server_args": goldens.portable(base_args[2:])},
        "model": {"file": os.path.basename(a.model), "sha256": goldens.known_sha(a.model),
                  "source": "unsloth/Muse-Glimmer-30B-GGUF@faa5b025c584459c13febfa5c59883516710ae39"},
        "mmproj": {"file": os.path.basename(a.mmproj), "sha256": goldens.known_sha(a.mmproj)},
        "sampling": "greedy (temperature 0, top_k 1); logprobs are log-softmax over the whole vocabulary of the raw logits "
                    "(post_sampling_probs false), after the model's logit scale and softcap",
        "prompt": "the user turn is <|patch|> then the question, the assistant header ends with the direct-answer suffix "
                  "' to=user<|message|>' (thinking off); llama.cpp received its media marker in place of <|patch|> "
                  "(LLAMA_MEDIA_MARKER=" + MARKER + ")",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "eng/validation/muse-glimmer-llamacpp-vision-goldens.py",
    }
    goldens.write(os.path.join(a.out, "ref_vision.json"), dict(provenance, records=records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
