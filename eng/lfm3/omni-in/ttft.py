#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Cold time to first token of LFM3-Omni media prompts (decode + encode + prefill), llama-server vs TensorSharp, from the
client: every repetition sends media neither engine has seen (a pixel, a sample or the container metadata changed),
llama.cpp with cache_prompt false, and the two servers alternate A/B then B/A so machine noise falls on both.

    python3 eng/lfm3/omni-in/ttft.py --media artifacts/lfm3/omni-in/media/manifest.json \
        --llama http://127.0.0.1:8181 --ts http://127.0.0.1:5181 --out artifacts/lfm3/omni-in/ttft/metal --reps 4

Concurrency (TensorSharp, optionally llama.cpp): --concurrent measures a short text request's TTFT alone and while a
media request is being prepared (sent --delay seconds after it)."""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time

from PIL import Image, PngImagePlugin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omni_client as oc  # noqa: E402

CASES = [
    ("image_512", "image", "photo_zebra", "Describe this image in one sentence."),
    ("image_receipt", "image", "receipt", "What is the total?"),
    ("image_2048", "image", "aerial_large", "Describe this landscape photo in one sentence."),
    ("audio_6s", "audio", "ticket", None),
    ("audio_47s", "audio", "long_harvard", None),
    ("video_10s_sound", "video", "test3", "What happens in this video?"),
    ("video_11s_speech", "video", "recipe_narrated", "What are the steps?"),
    # Controls: text prompts as long as the media prompts above, so a media case's prefill share can be told apart.
    ("text_2k", "text", 2000, "Summarize the text above in one sentence."),
    ("text_6k", "text", 5850, "Summarize the text above in one sentence."),
]

_WORDS = ("the birch canoe slid on the smooth planks glue the sheet to the dark blue background it is easy to tell the "
          "depth of a well these days a chicken leg is a rare dish rice is often served in round bowls").split()


def text_prompt(tokens, k):
    """About `tokens` tokens of text (one word is about one token here), opened by a nonce so no prefix is shared."""
    words = [f"Request {k}:"] + [_WORDS[(i * 5) % len(_WORDS)] for i in range(tokens)]  # the same text for every k
    return " ".join(words)


def variant(kind, src, k, work):
    """A copy of src with other bytes (so no cache keyed by content can answer it) and the same decoded media: a JPEG
    comment segment, a PNG text chunk, one PCM sample's LSB, or the container's comment metadata."""
    base, ext = os.path.splitext(os.path.basename(src))
    dst = os.path.join(work, f"{base}.v{k}{ext}")
    if kind == "image" and ext.lower() in (".jpg", ".jpeg"):
        with open(src, "rb") as f:
            data = f.read()
        text = f"variant {k}".encode()
        com = b"\xff\xfe" + (len(text) + 2).to_bytes(2, "big") + text
        with open(dst, "wb") as f:
            f.write(data[:2] + com + data[2:])  # right after SOI
    elif kind == "image":
        info = PngImagePlugin.PngInfo()
        info.add_text("variant", str(k))
        Image.open(src).save(dst, pnginfo=info)
    elif kind == "audio":
        shutil.copy(src, dst)
        with open(dst, "r+b") as f:  # flip the LSB of one sample past the 44-byte header
            f.seek(100 + 2 * k)
            b = f.read(1)
            f.seek(100 + 2 * k)
            f.write(bytes([b[0] ^ 1]))
    else:
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src, "-map", "0", "-c", "copy",
                        "-metadata", f"comment=variant {k}", dst], check=True)
    return dst


def request(kind, path, text):
    if kind == "text":
        return oc.body([{"role": "user", "content": path + "\n\n" + text}], 4, extra={"cache_prompt": False})
    msgs = []
    if kind == "audio":
        msgs.append({"role": "system", "content": "Perform ASR in English."})
    msgs.append(oc.user([(kind, path)], text))
    return oc.body(msgs, 4, extra={"cache_prompt": False})


def cold(args, manifest):
    work = os.path.join(args.out, "variants")
    os.makedirs(work, exist_ok=True)
    groups = {"image": manifest["images"], "audio": manifest["audio"], "video": manifest["video"]}
    results = {}
    for name, kind, key, text in CASES:
        if args.only and name not in args.only.split(","):
            continue
        src = groups[kind][key]["path"] if kind != "text" else None

        def variant_of(k):
            return text_prompt(key, k) if kind == "text" else variant(kind, src, k, work)
        # warm both servers on this shape (program compile, first-size kernels) with its own variant
        warm = variant_of(99)
        for base in (args.llama, args.ts):
            oc.chat(base, request(kind, warm, text))
        t = {"llama": [], "ts": [], "prompt": {}}
        for rep in range(args.reps):
            order = [("llama", args.llama), ("ts", args.ts)] if rep % 2 == 0 else [("ts", args.ts), ("llama", args.llama)]
            for who, base in order:
                path = variant_of(rep * 2 + (0 if who == "llama" else 1))
                r = oc.chat(base, request(kind, path, text))
                t[who].append(r["ttft"])
                t["prompt"][who] = (r.get("usage") or {}).get("prompt_tokens")
        t["median"] = {w: statistics.median(t[w]) for w in ("llama", "ts")}
        results[name] = t
        print(f"{name:<18} prompt {t['prompt'].get('llama')}/{t['prompt'].get('ts')}  TTFT median llama {t['median']['llama']*1000:7.0f} ms"
              f"  TS {t['median']['ts']*1000:7.0f} ms  (llama {[round(x*1000) for x in t['llama']]}, TS {[round(x*1000) for x in t['ts']]})", flush=True)
    return results


def concurrent(args, manifest):
    """A short text request's TTFT alone, then sent while a media request is being prepared."""
    work = os.path.join(args.out, "variants")
    os.makedirs(work, exist_ok=True)
    groups = {"image": manifest["images"], "audio": manifest["audio"], "video": manifest["video"]}
    text_req = oc.body([{"role": "user", "content": "Name three primary colours."}], 4)
    out = {}
    for base_name, base in (("ts", args.ts),) + ((("llama", args.llama),) if args.concurrent_llama else ()):
        alone = []
        for _ in range(args.reps + 1):
            alone.append(oc.chat(base, dict(text_req, messages=[{"role": "user", "content": f"Name three primary colours. ({len(alone)})"}]))["ttft"])
        alone = alone[1:]
        out[base_name] = {"text_alone_ms": [round(x * 1000) for x in alone]}
        for name, kind, key, text in CASES:
            if name not in args.concurrent_cases.split(","):
                continue
            rows = []
            for rep in range(args.reps):
                path = variant(kind, groups[kind][key]["path"], 50 + rep + (100 if base_name == "llama" else 0), work)
                media_result = {}

                def media():
                    media_result.update(oc.chat(base, request(kind, path, text)))
                th = threading.Thread(target=media)
                t0 = time.perf_counter()
                th.start()
                time.sleep(args.delay)
                r = oc.chat(base, dict(text_req, messages=[{"role": "user", "content": f"Name three primary colours. ({name} {rep})"}]))
                th.join()
                rows.append({"text_ttft_ms": round(r["ttft"] * 1000), "media_ttft_ms": round(media_result["ttft"] * 1000),
                             "text_sent_at_ms": round(args.delay * 1000)})
            out[base_name][name] = rows
            print(f"[{base_name}] {name:<18} text alone {statistics.median(alone)*1000:5.0f} ms; during media prep: text TTFT "
                  f"{[x['text_ttft_ms'] for x in rows]} ms, media TTFT {[x['media_ttft_ms'] for x in rows]} ms", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--llama", required=True)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--concurrent", action="store_true")
    ap.add_argument("--concurrent-llama", action="store_true")
    ap.add_argument("--concurrent-cases", default="image_2048,audio_47s,video_11s_speech")
    ap.add_argument("--delay", type=float, default=0.15)
    ap.add_argument("--only", default=None, help="comma-separated case names (cold mode)")
    a = ap.parse_args()
    with open(a.media) as f:
        manifest = json.load(f)
    os.makedirs(a.out, exist_ok=True)
    res = concurrent(a, manifest) if a.concurrent else cold(a, manifest)
    with open(os.path.join(a.out, "concurrent.json" if a.concurrent else "ttft.json"), "w") as f:
        json.dump(res, f, indent=1)
    shutil.rmtree(os.path.join(a.out, "variants"), ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
