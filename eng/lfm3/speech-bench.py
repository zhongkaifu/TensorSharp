#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""Streaming TTS speed of OpenAI-compatible servers, measured by one client: time to the first audio byte (after a WAV
stream's 44-byte header), total time, audio seconds, and the real-time factor (audio seconds / generation seconds),
for each server in turn (A/B/A/B...) so a shared machine's load falls on every arm alike.

    speech-bench.py OUT.json REPEATS NAME=URL [NAME=URL ...]

Each server gets POST {URL}/v1/audio/speech {"input", "voice", "language", "seed", "stream": true, "response_format": "pcm"}
for every text below and every repeat. Run with `python3 -I` (stdlib only).
"""
import json
import statistics
import sys
import time
import urllib.request

TEXTS = [
    ("en1", "en", "female", "Hello! This is a short test of speech output, eighty milliseconds at a time."),
    ("en2", "en", "male", "The quick brown fox jumps over the lazy dog, and then it runs back home before the rain starts."),
    ("zh1", "zh", "female", "今天天气很好，我们一起去公园散步，然后在湖边喝一杯咖啡。"),
]
RATE = 24000


def speak(url, text, lang, voice, seed):
    body = json.dumps({"model": "x", "input": text, "voice": voice, "language": language_field(lang), "seed": seed,
                       "stream": True, "response_format": "pcm"}).encode()
    req = urllib.request.Request(url + "/v1/audio/speech", data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    total = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        while True:
            chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(4096)
            if not chunk:
                break
            if first is None:
                first = time.perf_counter() - start
            total += len(chunk)
    elapsed = time.perf_counter() - start
    return first, elapsed, total / 2 / RATE


def language_field(lang):
    return lang


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    out_path, repeats = sys.argv[1], int(sys.argv[2])
    servers = [arg.split("=", 1) for arg in sys.argv[3:]]
    runs = []
    # Warm every server once (compiled programs, codec streams).
    for name, url in servers:
        speak(url, TEXTS[0][3], TEXTS[0][1], TEXTS[0][2], 1)
    for r in range(repeats):
        for ident, lang, voice, text in TEXTS:
            for name, url in servers:
                first, elapsed, seconds = speak(url, text, lang, voice, 100 + r)
                runs.append({"server": name, "text": ident, "repeat": r, "first_audio_s": first, "total_s": elapsed,
                             "audio_s": seconds, "rtf": seconds / elapsed if elapsed > 0 else 0})
                print("%-10s %-4s r%d  first audio %6.1f ms  total %5.2f s  audio %5.2f s  %5.1fx real time"
                      % (name, ident, r, 1000 * first, elapsed, seconds, seconds / elapsed), flush=True)
    summary = {}
    for name, _ in servers:
        mine = [x for x in runs if x["server"] == name]
        summary[name] = {
            "first_audio_ms_median": 1000 * statistics.median(x["first_audio_s"] for x in mine),
            "rtf_median": statistics.median(x["rtf"] for x in mine),
            "ms_per_frame_median": statistics.median(1000 * x["total_s"] / (x["audio_s"] / 0.08) for x in mine if x["audio_s"] > 0),
        }
    print(json.dumps(summary, indent=1))
    json.dump({"runs": runs, "summary": summary}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
