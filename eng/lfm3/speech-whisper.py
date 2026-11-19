#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""Independent intelligibility and language check of speech output: Whisper (faster-whisper, not the model that spoke)
identifies each WAV's language and transcribes it in the language it should be in, and the transcript is scored against
the text that was spoken (WER, or CER for Chinese/Japanese/Korean), with speech-asr.py's normalization.

    speech-whisper.py MANIFEST.tsv OUT.json [--model medium] [--device cpu] [--threads N]

MANIFEST.tsv lines (speech-asr.py's): id <TAB> language code <TAB> wav path (relative to the manifest) <TAB> reference.
Each row of OUT.json: the detected language and its probability (Whisper's language ID over the first 30 s), the
probability of the expected language, the transcript (decoded in the expected language, beam 5, temperature 0) and its
error rate. Chinese text is folded to simplified characters on both sides (Whisper writes either), and its numerals are
read out as Mandarin says them on both sides ("365" and "三百六十五" agree).

Needs a Python with faster-whisper and opencc-python-reimplemented, e.g.
    uv venv ~/work/venvs/whisper && uv pip install --python ~/work/venvs/whisper/bin/python faster-whisper \\
        opencc-python-reimplemented
Run it with `python -I` (models download to the Hugging Face cache; medium is 1.4 GB).
"""
import importlib.util
import json
import os
import re
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("speech_asr", os.path.join(HERE, "speech-asr.py"))
speech_asr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(speech_asr)

# Whisper's code of each manifest language (en-us / en-gb are en).
WHISPER_CODES = {"en": "en", "en-us": "en", "en-gb": "en", "zh": "zh", "fr": "fr", "de": "de", "es": "es", "it": "it",
                 "ja": "ja", "ko": "ko", "pt": "pt", "nl": "nl", "ar": "ar", "ru": "ru"}
# Whisper's prompt that keeps Mandarin in simplified characters with punctuation.
INITIAL_PROMPTS = {"zh": "以下是普通话的句子。"}


# Chinese numerals for scoring: the reference and the transcript may write a number either way ("365" / "三百六十五"),
# so both sides are read out as Mandarin says them before the character error rate (the reading of
# TensorSharp.Models.Audio.ChineseNumberReading, enough for scoring).
_ZH = "零一二三四五六七八九"
_MEASURES = "个只位本件条张次天杯种份双台辆块人家点岁周名样句步页层门所座节套颗"


def _digits(s):
    return "".join(_ZH[int(c)] for c in s)


def _four(g):
    out, any_, zero = "", False, False
    for d, unit in zip((g // 1000, g // 100 % 10, g // 10 % 10, g % 10), ("千", "百", "十", "")):
        if d == 0:
            zero = zero or any_
            continue
        if zero:
            out += "零"
            zero = False
        out += _ZH[d] + unit
        any_ = True
    return out


def _number(s):
    if len(s) >= 10 or (len(s) > 1 and s[0] == "0"):
        return _digits(s)
    n = int(s)
    if n == 0:
        return "零"
    out, started, zero = "", False, False
    for g, unit in ((n // 100000000, "亿"), (n // 10000 % 10000, "万"), (n % 10000, "")):
        if g == 0:
            zero = zero or started
            continue
        if started and (zero or g < 1000):
            out += "零"
        out += _four(g) + unit
        started, zero = True, False
    return out[1:] if out.startswith("一十") else out


def _value(s):
    whole, _, frac = s.partition(".")
    return _number(whole) + ("点" + _digits(frac) if frac else "")


def _signs(text):
    for a, b in (("!=", "不等于"), ("==", "等于"), ("≠", "不等于"), ("≈", "约等于"), ("≤", "小于等于"), ("≥", "大于等于"),
                 ("×", "乘"), ("÷", "除以"), ("=", "等于")):
        text = text.replace(a, b)
    return text


def chinese_numerals(text):
    text = "".join(chr(ord(c) - 0xFF10 + 0x30) if "０" <= c <= "９" else c for c in text)
    if not re.search(r"\d", text):
        return _signs(text)
    text = re.sub(r"(?<!\d)(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})(?!\d)\s*日?",
                  lambda m: _digits(m[1]) + "年" + _number(str(int(m[2]))) + "月" + _number(str(int(m[3]))) + "日", text)
    text = re.sub(r"(?<!\d)(\d{4})(?=\s*年)", lambda m: _digits(m[1]), text)
    text = re.sub(r"(?<![\d.])(\d{1,2}):(\d{2})(?![\d.])",
                  lambda m: _number(str(int(m[1]))) + "点" + ("" if int(m[2]) == 0 else ("零" if int(m[2]) < 10 else "") + _number(str(int(m[2]))) + "分"), text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*[%％]", lambda m: "百分之" + _value(m[1]), text)
    text = re.sub(r"([¥￥$€£])\s*(\d+(?:\.\d+)?)", lambda m: _value(m[2]) + {"$": "美元", "€": "欧元", "£": "英镑"}.get(m[1], "元"), text)
    text = re.sub(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?![\d,])", lambda m: m[0].replace(",", ""), text)
    text = re.sub(r"(?<=\d)\s*[-~～—–]\s*(?=\d)", "到", text)
    text = re.sub(r"(?<=\d)\s*\+\s*(?=\d)", "加", text)
    text = re.sub(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])", lambda m: _number(m[1]) + "点" + _digits(m[2]), text)

    def integer(m):
        after = text[m.end():m.end() + 1]
        ordinal = m.start() > 0 and text[m.start() - 1] == "第"
        return "两" if m[0] == "2" and not ordinal and after and after in _MEASURES else _number(m[0])
    return _signs(re.sub(r"\d+", integer, text))


def load_pcm16k(path):
    with wave.open(path, "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        data = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: {8 * width}-bit samples, expected 16-bit PCM")
    x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    seconds = len(x) / rate
    if rate != 16000:
        # Linear resampling is enough for recognition (Whisper's own front end low-passes at 8 kHz).
        n = int(round(len(x) * 16000 / rate))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x, seconds


def main():
    args = sys.argv[1:]
    model_name, device, threads = "medium", "cpu", 0
    if "--model" in args:
        i = args.index("--model"); model_name = args[i + 1]; del args[i:i + 2]
    if "--device" in args:
        i = args.index("--device"); device = args[i + 1]; del args[i:i + 2]
    if "--threads" in args:
        i = args.index("--threads"); threads = int(args[i + 1]); del args[i:i + 2]
    if len(args) != 2:
        sys.exit(__doc__)
    manifest, out_path = args
    from faster_whisper import WhisperModel
    from opencc import OpenCC
    t2s = OpenCC("t2s")
    model = WhisperModel(model_name, device=device, compute_type="int8", cpu_threads=threads)
    base = os.path.dirname(os.path.abspath(manifest))
    rows = []
    for line in open(manifest, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        ident, lang, wav, reference = line.rstrip("\n").split("\t", 3)
        path = wav if os.path.isabs(wav) else os.path.join(base, wav)
        if not os.path.exists(path) and os.path.exists(wav):
            path = wav
        audio, seconds = load_pcm16k(path)
        code = WHISPER_CODES.get(lang.lower(), lang.lower())
        if len(audio) < 1600:
            rows.append({"id": ident, "language": lang, "wav": wav, "reference": reference, "audio_s": seconds,
                         "detected": None, "detected_p": 0.0, "expected_p": 0.0, "asr": "", "unit": "CER"
                         if lang in speech_asr.CHARACTER_LANGUAGES else "WER", "errors": None, "length": None,
                         "rate": 1.0})
            print("%-28s no audio" % ident, flush=True)
            continue
        detected, detected_p, probs = model.detect_language(audio=audio)
        expected_p = dict(probs).get(code, 0.0)
        segments, _ = model.transcribe(audio, language=code, beam_size=5, temperature=0.0,
                                       initial_prompt=INITIAL_PROMPTS.get(code), condition_on_previous_text=False)
        hypothesis = "".join(s.text for s in segments).strip()
        ref_text, hyp_text = reference, hypothesis
        if code == "zh":
            ref_text, hyp_text = chinese_numerals(t2s.convert(ref_text)), chinese_numerals(t2s.convert(hyp_text))
        ref, hyp = speech_asr.normalize(ref_text, lang), speech_asr.normalize(hyp_text, lang)
        errors = speech_asr.edit_distance(ref, hyp)
        unit = "CER" if lang in speech_asr.CHARACTER_LANGUAGES else "WER"
        top = sorted(probs, key=lambda p: -p[1])[:3]
        rows.append({"id": ident, "language": lang, "wav": wav, "reference": reference, "audio_s": round(seconds, 3),
                     "detected": detected, "detected_p": round(float(detected_p), 4), "expected_p": round(float(expected_p), 4),
                     "top_languages": [[c, round(float(p), 4)] for c, p in top], "asr": hypothesis, "unit": unit,
                     "errors": errors, "length": len(ref), "rate": errors / max(1, len(ref))})
        print("%-28s lid %s %.2f (%s %.2f)  %s %5.1f%%  %s" % (ident, detected, detected_p, code, expected_p, unit,
                                                             100 * rows[-1]["rate"], hypothesis), flush=True)
    json.dump(rows, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
