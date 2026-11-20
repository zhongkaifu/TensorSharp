#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""Voice-consistency proxy for spoken replies: per WAV, the median F0 (autocorrelation over voiced 40 ms frames, 70-400
Hz) of each 2 s window; per directory and voice (a file name holding "male" but not "female" is the male voice), the
median F0 across replies, its p10-p90 across replies, and the median within-reply spread (p90-p10 of the windows). A
female and a male voice should sit far apart (TTS: about 200 vs 115 Hz); two voices whose F0 match say the same voice.

    speech-f0.py DIR... [--json OUT.json]

Needs numpy (e.g. the speech-whisper.py venv).
"""
import json
import os
import statistics
import sys
import wave

import numpy as np


def f0_track(x, rate):
    hop, win = int(0.02 * rate), int(0.04 * rate)
    out = []
    for s in range(0, len(x) - win, hop):
        f = x[s:s + win]
        if np.sqrt(np.mean(f * f)) < 0.02:
            out.append(None)
            continue
        f = f - f.mean()
        ac = np.correlate(f, f, mode="full")[win - 1:]
        lo, hi = int(rate / 400), int(rate / 70)
        k = lo + int(np.argmax(ac[lo:hi]))
        out.append(rate / k if ac[k] > 0.3 * ac[0] else None)
    return out


def main():
    args = sys.argv[1:]
    out_json = None
    if "--json" in args:
        i = args.index("--json")
        out_json = args[i + 1]
        args = args[:i] + args[i + 2:]
    report = {}
    for d in args:
        by_voice = {}
        for name in sorted(os.listdir(d)):
            if not name.endswith(".wav"):
                continue
            with wave.open(os.path.join(d, name)) as w:
                rate = w.getframerate()
                x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
            track = f0_track(x, rate)
            windows = []
            for s in range(0, len(track), 100):
                voiced = [f for f in track[s:s + 100] if f]
                if len(voiced) >= 20:
                    windows.append(statistics.median(voiced))
            voice = "male" if "male" in name and "female" not in name else "female"
            if windows:
                spread = float(np.percentile(windows, 90) - np.percentile(windows, 10)) if len(windows) > 2 else 0.0
                by_voice.setdefault(voice, []).append((statistics.median(windows), spread, len(windows)))
        report[d] = {}
        for voice, rows in sorted(by_voice.items()):
            medians = [r[0] for r in rows]
            spreads = [r[1] for r in rows if r[2] > 2]
            entry = {"replies": len(rows), "f0": statistics.median(medians),
                     "f0p10": float(np.percentile(medians, 10)), "f0p90": float(np.percentile(medians, 90)),
                     "withinReplySpread": statistics.median(spreads) if spreads else None}
            report[d][voice] = entry
            print("%-40s %-6s replies=%2d F0 %6.1f Hz (p10-p90 %6.1f-%6.1f); within-reply spread %s Hz" % (
                d[-40:], voice, entry["replies"], entry["f0"], entry["f0p10"], entry["f0p90"],
                "%.1f" % entry["withinReplySpread"] if entry["withinReplySpread"] is not None else "n/a"))
    if out_json:
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
