#!/usr/bin/env python3
"""Check a masked edit's kept pixels against its source, independently of TensorSharp's mask code (stage S6 gate).

  mask_kept_pixels.py --source src.png --output edited.png --mask mask.png [--invert] [--json out.json]

A grayscale mask (white edits, black keeps): the kept pixels are the black ones (with --invert, the white ones); a feather
only lowers weights inward, so it keeps the same set. Every kept pixel of the output must be the source's bytes (RGBA when
either has alpha), the output the source's size, and some selected pixel must have changed. Prints a JSON summary; the exit
code is 0 when the gate holds. Pillow only.
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mask', type=Path, required=True)
    ap.add_argument('--invert', action='store_true')
    ap.add_argument('--json', type=Path)
    args = ap.parse_args()
    src, out, mask = Image.open(args.source), Image.open(args.output), Image.open(args.mask).convert('L')
    summary = {'source': str(args.source), 'output': str(args.output), 'mask': str(args.mask), 'invert': args.invert,
               'source_size': src.size, 'output_size': out.size}
    if src.size != out.size or src.size != mask.size:
        summary['ok'] = False
        print(json.dumps(summary))
        return 1
    mode = 'RGBA' if 'A' in src.getbands() or 'A' in out.getbands() else 'RGB'
    s, o, m = src.convert(mode).tobytes(), out.convert(mode).tobytes(), mask.tobytes()
    n = len(mode)
    keep_value = 255 if args.invert else 0
    kept = differ = selected = changed = 0
    for i, weight in enumerate(m):
        same = s[i * n:(i + 1) * n] == o[i * n:(i + 1) * n]
        if weight == keep_value:
            kept += 1
            differ += not same
        else:
            selected += 1
            changed += not same
    summary.update(mode=mode, kept_pixels=kept, kept_differ=differ, selected_pixels=selected, selected_changed=changed)
    summary['ok'] = kept > 0 and differ == 0 and changed > 0
    print(json.dumps(summary))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2))
    return 0 if summary['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
