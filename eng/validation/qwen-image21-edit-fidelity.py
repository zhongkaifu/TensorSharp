#!/usr/bin/env python3
"""How much of a source picture an image edit kept, outside what it was asked to change.

For each edited PNG this reports, against --source: its SHA-256; the PSNR of the whole picture;
the PSNR of the pixels outside --changed (white where the instruction applies, for example the
object a recolor names) and more than --margin pixels away from it, since edges move with the
object; and the mean absolute difference inside --changed, which shows the edit did change
something there. With --selection (the mask the edit was given, white = editable) it also
counts protected pixels that changed, which must be zero, and the PSNR inside the selection
outside the changed area.

None of this measures whether the edit followed its instruction: look at the pictures. PSNR is
on 8-bit RGB (alpha dropped); identical regions report inf.

  python3 eng/validation/qwen-image21-edit-fidelity.py --source teapot.png \\
      --changed teapot-object.png --selection teapot-selection.png edit-a.png edit-b.png
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def region(path, shape):
    mask = np.asarray(Image.open(path).convert("L")) > 127
    if mask.shape != shape:
        raise ValueError(f"{path} is {mask.shape[1]}x{mask.shape[0]}; the source is {shape[1]}x{shape[0]}.")
    return mask


def grow(mask, margin):
    """The mask and every pixel within margin pixels of it (a square neighbourhood)."""
    if margin <= 0:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * margin + 1))
    return np.asarray(image) > 0


def psnr(a, b, where=None):
    squared = (a - b) ** 2
    if where is not None:
        if not where.any():
            return None
        squared = squared[where]
    mse = float(squared.mean())
    return math.inf if mse == 0 else round(10 * math.log10(255 ** 2 / mse), 2)


def measure(source, edited, changed, margin, selection=None):
    """The record for one edited picture: arrays in, numbers out."""
    if edited.shape != source.shape:
        raise ValueError("The edit and the source differ in size; compare at the source's size.")
    unchanged = ~grow(changed, margin)
    record = {
        "psnr_whole_db": psnr(source, edited),
        "psnr_outside_changed_db": psnr(source, edited, unchanged),
        "changed_mean_abs_diff": round(float(np.abs(source - edited)[changed].mean()), 2) if changed.any() else None,
        "unchanged_fraction": round(float(unchanged.mean()), 4),
    }
    if selection is not None:
        record["protected_pixels_changed"] = int(np.any(source[~selection] != edited[~selection], axis=-1).sum())
        record["psnr_selection_outside_changed_db"] = psnr(source, edited, selection & unchanged)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="The picture that was edited.")
    parser.add_argument("--changed", type=Path, required=True,
                        help="Mask at the source's size: white where the instruction asks for a change.")
    parser.add_argument("--selection", type=Path, help="The mask the edits were given (white = editable), if any.")
    parser.add_argument("--margin", type=int, default=16, help="Pixels around --changed left out of the kept area.")
    parser.add_argument("--json", type=Path, help="Also write the records here.")
    parser.add_argument("edits", type=Path, nargs="+", help="Edited PNGs at the source's size.")
    args = parser.parse_args()

    source = rgb(args.source)
    shape = source.shape[:2]
    changed = region(args.changed, shape)
    selection = region(args.selection, shape) if args.selection else None
    records = []
    for path in args.edits:
        record = {"png": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        record.update(measure(source, rgb(path), changed, args.margin, selection))
        records.append(record)
        print(json.dumps(record))
    if args.json:
        args.json.write_text(json.dumps({"source": str(args.source), "changed": str(args.changed),
                                         "selection": str(args.selection) if args.selection else None,
                                         "margin": args.margin, "edits": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
