#!/usr/bin/env python3
"""Which PNGs the TensorAgent apps (Apple media provider) decode to other bytes than the CLI and server.

The CLI and server decode with ImageMagick, which keeps a PNG's stored 8-bit values. The Apple
provider (TensorSharp.Models/Media/Apple/AppleMediaProvider.cs) reads a PNG that carries alpha and
embeds no colour profile with its own codec, also as stored, and draws every other PNG through
Core Graphics into an sRGB bitmap. Where those bytes differ, every consumer of the decoded
pixels differs between the hosts; for Qwen-Image-2.1 edits, the reference-keyed noise stream does.

This writes one picture as PNG variants (opaque and with alpha; no colour chunks, sRGB, gAMA and
cHRM for sRGB, a linear gAMA, Adobe RGB primaries in cHRM, the chunks ImageMagick writes, an
embedded Display P3 profile), adds any --png files given
(for example a screenshot or a CLI output), decodes each with apple-png-decode.swift and compares
with the stored values (Pillow, which applies no colour management). macOS only.

  python3 eng/validation/apple-png-decode-check.py --out artifacts/png-decode \\
      [--magick-sample cli-output.png] [--png screenshot.png ...] [--json report.json]
"""
import argparse
import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DISPLAY_P3 = Path("/System/Library/ColorSync/Profiles/Display P3.icc")


def chunk(kind, payload):
    body = kind.encode("latin-1") + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def chunks(path):
    data = Path(path).read_bytes()
    offset, found = 8, []
    while offset + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[offset:offset + 8])
        found.append((kind.decode("latin-1"), data[offset + 8:offset + 8 + length]))
        if kind == b"IEND":
            break
        offset += 12 + length
    return found


def write_png(path, pixels, extra):
    """8-bit RGB or RGBA, filter 0, with the ancillary chunks in extra before IDAT."""
    height, width, channels = pixels.shape
    header = struct.pack(">IIBBBBB", width, height, 8, 6 if channels == 4 else 2, 0, 0, 0)
    rows = b"".join(b"\x00" + pixels[y].tobytes() for y in range(height))
    body = chunk("IHDR", header) + b"".join(chunk(k, v) for k, v in extra)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + body + chunk("IDAT", zlib.compress(rows, 9)) + chunk("IEND", b""))


def picture(width=96, height=64):
    """A photo-like test picture: smooth gradients with noise, every channel spanning 0-255."""
    y, x = np.mgrid[0:height, 0:width]
    rng = np.random.default_rng(1)
    rgb = np.stack([x / (width - 1), y / (height - 1), (x + y) / (width + height - 2)], axis=-1) * 255
    rgb = np.clip(rgb + rng.normal(0, 12, rgb.shape), 0, 255).round().astype(np.uint8)
    return rgb


def variants(out, magick_sample):
    rgb = picture()
    rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=-1)
    srgb = [("sRGB", b"\x00")]
    gama_chrm = [("gAMA", struct.pack(">I", 45455)),
                 ("cHRM", struct.pack(">8I", 31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000))]
    # gAMA 1.0 and Adobe RGB (1998) primaries: colour chunks that do not describe sRGB.
    linear = [("gAMA", struct.pack(">I", 100000))]
    adobe = [("gAMA", struct.pack(">I", 45470)),
             ("cHRM", struct.pack(">8I", 31270, 32900, 64000, 33000, 21000, 71000, 15000, 6000))]
    p3 = [("iCCP", b"ICC Profile\x00\x00" + zlib.compress(DISPLAY_P3.read_bytes()))]
    sets = {"plain": [], "srgb": srgb, "gama-chrm": gama_chrm, "gama-linear": linear, "adobe-rgb-chrm": adobe,
            "display-p3": p3}
    if magick_sample:
        sets["imagemagick-chunks"] = [(k, v) for k, v in chunks(magick_sample) if k not in ("IHDR", "IDAT", "IEND")]
    files = []
    for name, extra in sets.items():
        for layout, pixels in (("rgb", rgb), ("rgba-opaque", rgba)):
            path = out / f"{layout}-{name}.png"
            write_png(path, pixels, extra)
            files.append(path)
    return files


def compare(path, decoded_dir, decoder):
    stored = np.asarray(Image.open(path).convert("RGBA"), dtype=np.int16)
    record = {"file": path.name, "chunks": [k for k, _ in chunks(path) if k not in ("IDAT", "IEND")],
              "apple_decoder": decoder.split()[0]}
    if record["apple_decoder"] != "coregraphics":
        record.update(values_differing=0, max_difference=0, note="read as stored")
        return record
    apple = np.frombuffer((decoded_dir / (path.name + ".rgba")).read_bytes(), np.uint8).astype(np.int16)
    apple = apple.reshape(stored.shape)
    difference = np.abs(apple[..., :3] - stored[..., :3])
    record.update(colour_space=" ".join(decoder.split()[1:]), values_differing=int((difference > 0).sum()),
                  values=int(difference.size), max_difference=int(difference.max()))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="directory for the generated PNGs and decodes")
    parser.add_argument("--magick-sample", type=Path, help="a PNG ImageMagick wrote (a CLI output): its ancillary chunks are copied")
    parser.add_argument("--png", type=Path, nargs="*", default=[], help="more PNG files to check as they are")
    parser.add_argument("--json", type=Path, help="write the records here")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("Core Graphics is macOS only.")
    args.out.mkdir(parents=True, exist_ok=True)
    files = variants(args.out, args.magick_sample) + list(args.png)
    result = subprocess.run(["swift", str(HERE / "apple-png-decode.swift"), str(args.out), *map(str, files)],
                            check=True, capture_output=True, text=True)
    decoders = dict(line.split(" ", 1) for line in result.stdout.splitlines() if " " in line)
    records = [compare(Path(f), args.out, decoders[Path(f).name]) for f in files]
    for r in records:
        print(f"{r['file']:<34} {r['apple_decoder']:<12} {r['values_differing']:>6} values differ (max {r['max_difference']})  "
              f"chunks {' '.join(r['chunks'])}")
    if args.json:
        args.json.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
