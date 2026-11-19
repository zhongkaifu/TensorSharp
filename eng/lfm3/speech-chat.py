#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""Spoken chat replies from an OpenAI-compatible server (`modalities` ["text","audio"]): each reply's WAV and transcript
are saved, and a manifest for speech-asr.py pairs the WAV with its own transcript, so the ASR's word error rate measures
whether the speech says what the text says.

    speech-chat.py URL OUT_DIR [MODE]        MODE: interleaved (default) or audio

Writes OUT_DIR/<id>.wav, OUT_DIR/replies.json and OUT_DIR/manifest.tsv. Run with `python3 -I` (stdlib only).
"""
import base64
import json
import os
import sys
import time
import urllib.request

QUESTIONS = [
    ("q1", "en", "female", "What is the capital of France?"),
    ("q2", "en", "male", "Give me one tip for sleeping better."),
    ("q3", "en", "female", "Why is the sky blue? Answer in two sentences."),
    ("q4", "en", "male", "Say hello to my friend Anna and wish her a happy birthday."),
    ("q5", "en", "female", "What should I pack for a weekend trip to the beach?"),
    ("q6", "en", "male", "How many days are there in a leap year?"),
    ("q7", "zh", "female", "请用一句话介绍一下北京。"),
    ("q8", "zh", "male", "今天下雨了，我应该带什么？"),
]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    url, out = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "interleaved"
    modalities = ["text", "audio"] if mode == "interleaved" else ["audio"]
    os.makedirs(out, exist_ok=True)
    # The hosted model's id (servers check the requested model against it).
    model = json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]
    replies = []
    with open(os.path.join(out, "manifest.tsv"), "w", encoding="utf-8") as manifest:
        for ident, lang, voice, question in QUESTIONS:
            body = {"model": model, "messages": [{"role": "user", "content": question}], "modalities": modalities,
                    "audio": {"voice": voice, "format": "wav", "language": lang, "seed": 7}, "max_tokens": 600}
            req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            start = time.perf_counter()
            reply = json.load(urllib.request.urlopen(req, timeout=600))
            elapsed = time.perf_counter() - start
            choice = reply["choices"][0]
            audio = choice["message"]["audio"]
            wav = os.path.join(out, ident + ".wav")
            data = base64.b64decode(audio["data"])
            open(wav, "wb").write(data)
            seconds = (len(data) - 44) / 48000
            replies.append({"id": ident, "question": question, "transcript": audio["transcript"], "audio_s": seconds,
                            "finish": choice["finish_reason"], "usage": reply["usage"], "elapsed_s": elapsed})
            print("%s %5.2f s audio in %5.2f s  %s" % (ident, seconds, elapsed, audio["transcript"]), flush=True)
            if audio["transcript"].strip():
                manifest.write("%s\t%s\t%s\t%s\n" % (ident, lang, wav, audio["transcript"].replace("\t", " ").replace("\n", " ")))
    json.dump(replies, open(os.path.join(out, "replies.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
