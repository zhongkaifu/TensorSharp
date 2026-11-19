#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""Spoken replies in several languages from a running TensorSharp.Server.Host with LFM3-Omni and its mmproj loaded, through
the three surfaces that speak, each request as its client sends it:

  webchat  /api/chat with "speech" (true, or {"voice": ...}), as the bundled Web UI's Speak replies switch sends it
  chat     /v1/chat/completions, stream, modalities ["text","audio"], audio {voice, format pcm16}
  audio    the same with modalities ["audio"] (speech only; the transcript comes as audio.transcript)
  speech   /v1/audio/speech, stream, response_format pcm (fixed texts, not the model's replies)
  text     /v1/chat/completions without audio (the text reply alone; no WAV)

    speech-lang-eval.py URL OUT_DIR SURFACE [--voices female,male] [--sets zh,en] [--language] [--max-tokens N] [--seed S]

--language also sends each prompt's language (speech.language, audio.language, language), --language-code X sends X
instead; without either the request names none, as the Web UI and an OpenAI client do. Each reply's WAV (24 kHz PCM16) is written as OUT_DIR/<id>-<voice>.wav with
replies.json (transcript, time to the first text and the first audio, total time, audio seconds, real-time factor = audio
seconds / generation seconds, tokens) and manifest.tsv for speech-whisper.py / speech-asr.py, which pairs each WAV with its
own transcript and the transcript's language (by its script: Han -> zh, otherwise en). Run with `python3 -I` (stdlib only).
"""
import argparse
import base64
import json
import os
import re
import struct
import time
import unicodedata
import urllib.request

PROMPTS = {
    "zh": [
        ("zh01", "请用一句话介绍一下北京。"),
        ("zh02", "今天下雨了，我应该带什么？"),
        ("zh03", "给我讲一个关于小猫的简短故事。"),
        ("zh04", "一年有多少天？闰年呢？"),
        ("zh05", "推荐三种健康的早餐。"),
        ("zh06", "中国的国庆节是几月几日？"),
        ("zh07", "一杯咖啡二十五元，三杯一共多少钱？"),
        ("zh08", "什么是machine learning？请用简单的话解释。"),
        ("zh09", "我想学习Python编程，应该从哪里开始？"),
        ("zh10", "怎样才能睡得更好？给我两个建议。"),
        ("zh11", "请帮我写一句生日祝福，送给我的妈妈。"),
        ("zh12", "长城大约有多长？"),
    ],
    "en": [
        ("en01", "What is the capital of France?"),
        ("en02", "Give me one tip for sleeping better."),
        ("en03", "Tell me a very short story about a cat."),
        ("en04", "How many days are there in a leap year?"),
        ("en05", "What should I pack for a weekend trip to the beach?"),
        ("en06", "Explain what machine learning is in one sentence."),
    ],
}

# /v1/audio/speech texts: everyday sentences, a story, numbers and dates, Chinese with English terms; English controls.
TEXTS = {
    "zh": [
        ("tz01", "北京是中国的首都，拥有丰富的历史文化和现代化的城市风貌。"),
        ("tz02", "今天下雨了，出门记得带一把雨伞和一件防水外套。"),
        ("tz03", "从前有一只小猫，它每天早上都会跑到花园里追蝴蝶。"),
        ("tz04", "一年有三百六十五天，闰年有三百六十六天。"),
        ("tz05", "2024年10月1日是国庆节，我们放假七天。"),
        ("tz06", "这件衣服原价199元，现在打八折，只要159.2元。"),
        ("tz07", "我今天用Python写了一个machine learning的小程序。"),
        ("tz08", "请把这份PDF文件发到我的email邮箱。"),
        ("tz09", "早睡早起身体好，每天至少要睡七个小时。"),
        ("tz10", "祝妈妈生日快乐，身体健康，天天开心！"),
        ("tz11", "长城全长约两万一千公里，是世界上最长的城墙。"),
        ("tz12", "会议定在下午三点半，请大家准时参加。"),
    ],
    "en": [
        ("te01", "The capital of France is Paris, a city famous for its art and food."),
        ("te02", "Remember to bring an umbrella, because it might rain this afternoon."),
        ("te03", "Once upon a time, a small cat chased butterflies in the garden every morning."),
        ("te04", "A leap year has 366 days, one more than a regular year."),
        ("te05", "The meeting starts at 3:30 PM on October 1st, 2024."),
        ("te06", "Machine learning lets computers learn patterns from data."),
    ],
}

RATE = 24000


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def sse(resp):
    for raw in resp:
        line = raw.decode("utf-8").rstrip("\n")
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                return
            yield json.loads(payload)


def write_wav(path, pcm16):
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(pcm16)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16)
                + b"data" + struct.pack("<I", len(pcm16)) + pcm16)


def spoken_reference(text):
    """The transcript as speech can say it: emoji and other symbols (Unicode So/Sk) and markdown markers dropped."""
    text = "".join(c for c in text if unicodedata.category(c) not in ("So", "Sk", "Cs"))
    return re.sub(r"[*_#`|>]+", " ", text).replace("\t", " ").replace("\n", " ").strip()


def script_language(text):
    return "zh" if re.search(r"[一-鿿]", text) else "en"


def run_webchat(base, prompt, voice, lang, args):
    session = json.load(post(base + "/api/sessions", {}))["sessionId"]
    speech = {"voice": voice}
    if args.language:
        speech["language"] = lang
    body = {"sessionId": session, "messages": [{"role": "user", "content": prompt}], "maxTokens": args.max_tokens, "speech": speech}
    start = time.perf_counter()
    first_text = first_audio = last_text = None
    text, pcm, done = [], bytearray(), {}
    with post(base + "/api/chat", body) as resp:
        for frame in sse(resp):
            now = time.perf_counter() - start
            if "token" in frame:
                first_text = first_text if first_text is not None else now
                last_text = now
                text.append(frame["token"])
            if "audio" in frame:
                first_audio = first_audio if first_audio is not None else now
                pcm += base64.b64decode(frame["audio"])
            if frame.get("done"):
                done = frame
    total = time.perf_counter() - start
    extra = {k: done.get(k) for k in ("promptTokens", "tokenCount", "elapsed", "error", "audioId", "audioSeconds", "tokPerSec",
                                      "kvReusedTokens")}
    extra["last_text_s"] = last_text
    return "".join(text), bytes(pcm), first_text, first_audio, total, extra


def run_chat(base, model, prompt, voice, lang, args, modalities=("text", "audio")):
    audio = {"voice": voice, "format": "pcm16", "seed": args.seed}
    if args.language:
        audio["language"] = lang
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "modalities": list(modalities), "audio": audio,
            "stream": True, "max_tokens": args.max_tokens, "stream_options": {"include_usage": True}}
    start = time.perf_counter()
    first_text = first_audio = last_text = None
    text, pcm, usage = [], bytearray(), {}
    with post(base + "/v1/chat/completions", body) as resp:
        for chunk in sse(resp):
            now = time.perf_counter() - start
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                a = (choice.get("delta") or {}).get("audio") or {}
                if a.get("transcript"):
                    first_text = first_text if first_text is not None else now
                    last_text = now
                    text.append(a["transcript"])
                if a.get("data"):
                    first_audio = first_audio if first_audio is not None else now
                    pcm += base64.b64decode(a["data"])
    total = time.perf_counter() - start
    usage = dict(usage)
    usage["last_text_s"] = last_text
    return "".join(text), bytes(pcm), first_text, first_audio, total, usage


def run_text(base, model, prompt, args):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens, "temperature": 0}
    start = time.perf_counter()
    reply = json.load(post(base + "/v1/chat/completions", body))
    total = time.perf_counter() - start
    return reply["choices"][0]["message"].get("content") or "", b"", None, None, total, reply.get("usage", {})


def run_speech(base, text, voice, lang, args):
    body = {"model": "x", "input": text, "voice": voice, "seed": args.seed, "stream": True, "response_format": "pcm"}
    if args.language:
        body["language"] = lang
    start = time.perf_counter()
    first = None
    pcm = bytearray()
    with post(base + "/v1/audio/speech", body) as resp:
        while True:
            chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(4096)
            if not chunk:
                break
            first = first if first is not None else time.perf_counter() - start
            pcm += chunk
    total = time.perf_counter() - start
    return text, bytes(pcm), None, first, total, {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("out")
    p.add_argument("surface", choices=["webchat", "chat", "audio", "speech", "text"])
    p.add_argument("--voices", default="female,male")
    p.add_argument("--sets", default="zh,en")
    p.add_argument("--language", action="store_true")
    p.add_argument("--language-code", default=None, help="send this language code instead of each prompt's (implies --language)")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--only", default=None, help="comma-separated ids")
    args = p.parse_args()
    if args.language_code:
        args.language = True
    base = args.url.rstrip("/")
    os.makedirs(args.out, exist_ok=True)
    model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
    voices = args.voices.split(",") if args.surface != "text" else ["-"]
    only = set(args.only.split(",")) if args.only else None
    rows = []
    for lang in args.sets.split(","):
        items = TEXTS[lang] if args.surface == "speech" else PROMPTS[lang]
        for ident, prompt in items:
            if only and ident not in only:
                continue
            for voice in voices:
                lang_sent = args.language_code or lang
                if args.surface == "webchat":
                    result = run_webchat(base, prompt, voice, lang_sent, args)
                elif args.surface == "chat":
                    result = run_chat(base, model, prompt, voice, lang_sent, args)
                elif args.surface == "audio":
                    result = run_chat(base, model, prompt, voice, lang_sent, args, modalities=("audio",))
                elif args.surface == "text":
                    result = run_text(base, model, prompt, args)
                else:
                    result = run_speech(base, prompt, voice, lang_sent, args)
                text, pcm, first_text, first_audio, total, extra = result
                name = ident if voice == "-" else "%s-%s" % (ident, voice)
                seconds = len(pcm) / 2 / RATE
                if pcm:
                    write_wav(os.path.join(args.out, name + ".wav"), pcm)
                row = {"id": name, "prompt_id": ident, "set": lang, "voice": voice, "prompt": prompt, "transcript": text,
                       "transcript_language": script_language(text) if text else None, "audio_s": round(seconds, 3),
                       "first_text_s": first_text, "first_audio_s": first_audio, "total_s": total,
                       "rtf": seconds / total if total > 0 and seconds > 0 else 0, "extra": extra}
                rows.append(row)
                print("%-14s %5.2fs audio, first audio %s, total %5.2fs: %s" % (
                    name, seconds, "%.0f ms" % (1000 * first_audio) if first_audio is not None else "-", total,
                    text.replace("\n", " ")[:90]), flush=True)
    json.dump(rows, open(os.path.join(args.out, "replies.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "manifest.tsv"), "w", encoding="utf-8") as m:
        for r in rows:
            if r["audio_s"] > 0:
                spoken = spoken_reference(r["transcript"])
                m.write("%s\t%s\t%s.wav\t%s\n" % (r["id"], r["transcript_language"] or r["set"], r["id"], spoken))


if __name__ == "__main__":
    main()
