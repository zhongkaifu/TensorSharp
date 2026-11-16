#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Convert hexgrad/Kokoro-82M (Apache-2.0) into the "kokoro" GGUF TensorSharp loads.

Input: a snapshot directory with config.json, kokoro-v1_0.pth and voices/*.pt. Then, in an environment with torch
(and, for Mandarin text input, misaki[zh], pypinyin and jieba, the libraries Kokoro's own pipeline uses):

    python eng/audio-models/kokoro_to_gguf.py --input DIR --out kokoro-82m-v1.0.gguf

Weights: every tensor is written F32 under "<module>.<name>" with the checkpoint's "module." prefix removed, and weight
norm folded (w = g * v / ||v||, the norm over every axis but the first, which is torch.nn.utils.weight_norm's dim=0).
Voices: each voices/<name>.pt pack [510, 1, 256] becomes "voice.<name>" [510, 256]; the row picked is the phoneme
count minus one, as KPipeline does.

Text front end (Mandarin, Kokoro-82M v1.0's misaki.zh.ZHG2P legacy path). The tables the C# port needs, all generated
by calling the reference libraries so the port reproduces them exactly:

    kokoro.zh.jieba.words / .freqs                 jieba's dictionary (DAG + max-probability route)
    kokoro.zh.jieba.hmm.start / .trans             finalseg's BMES start and transition log probabilities
    kokoro.zh.jieba.hmm.emit.<B|M|E|S>.chars/.logp finalseg's emission log probabilities
    kokoro.zh.pinyin.chars / .readings             lazy_pinyin(char, TONE3, neutral_tone_with_five) per character
    kokoro.zh.pinyin.phrases / .readings           the same for each pypinyin phrase (space-separated syllables)
    kokoro.zh.ipa.syllables / .ipa                 ZHG2P.py2ipa(syllable) for every syllable above

cn2an's an2cn number reading and ZHG2P.map_punctuation are reimplemented in C# and checked against the
reference in the tests.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_writer import F32, GgufWriter  # noqa: E402


def fold_weight_norm(state):
    out = {}
    for k, v in state.items():
        if k.endswith(".weight_v"):
            base = k[: -len(".weight_v")]
            g = state[base + ".weight_g"].double()
            vv = v.double()
            norm = vv.reshape(vv.shape[0], -1).norm(dim=1).reshape([-1] + [1] * (vv.dim() - 1))
            out[base + ".weight"] = (g * vv / norm).float()
        elif k.endswith(".weight_g"):
            continue
        else:
            out[k] = v
    return out


def finite32(values):
    """jieba's MIN_FLOAT (-3.14e100) is beyond float32: written as -inf, read back as MIN_FLOAT."""
    return [float("-inf") if v < -3.0e38 else v for v in values]


def zh_tables(w):
    import jieba
    import jieba.finalseg as finalseg
    from pypinyin import Style, lazy_pinyin
    from pypinyin.constants import PHRASES_DICT, PINYIN_DICT
    from misaki.zh import ZHG2P

    # jieba's dictionary, as Tokenizer.gen_pfdict reads it.
    dict_path = os.path.join(os.path.dirname(jieba.__file__), "dict.txt")
    words, freqs = [], []
    with open(dict_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 2:
                words.append(parts[0])
                freqs.append(int(parts[1]))
    w.strs("kokoro.zh.jieba.words", words)
    w.i32s("kokoro.zh.jieba.freqs", freqs)
    states = "BMES"
    w.f32s("kokoro.zh.jieba.hmm.start", finite32([finalseg.start_P[s] for s in states]))
    trans = []
    for a in states:
        for b in states:
            trans.append(finalseg.trans_P[a].get(b, -3.14e100))
    w.f32s("kokoro.zh.jieba.hmm.trans", finite32(trans))
    for s in states:
        items = sorted(finalseg.emit_P[s].items())
        w.strs(f"kokoro.zh.jieba.hmm.emit.{s}.chars", [c for c, _ in items])
        w.f32s(f"kokoro.zh.jieba.hmm.emit.{s}.logp", finite32([p for _, p in items]))

    def tone3(text):
        return lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)

    chars, char_readings = [], []
    for code in sorted(PINYIN_DICT):
        ch = chr(code)
        py = tone3(ch)
        if len(py) == 1 and py[0] != ch:
            chars.append(ch)
            char_readings.append(py[0])
    w.strs("kokoro.zh.pinyin.chars", chars)
    w.strs("kokoro.zh.pinyin.readings", char_readings)

    phrases, phrase_readings = [], []
    for phrase in sorted(PHRASES_DICT):
        py = tone3(phrase)
        if len(py) == len(phrase):
            phrases.append(phrase)
            phrase_readings.append(" ".join(py))
    w.strs("kokoro.zh.pinyin.phrases", phrases)
    w.strs("kokoro.zh.pinyin.phrase_readings", phrase_readings)

    syllables = sorted(set(char_readings) | {s for r in phrase_readings for s in r.split(" ")})
    ok_syllables, ipas = [], []
    for syl in syllables:
        try:
            ipa = ZHG2P.py2ipa(syl)
        except Exception:
            continue
        ok_syllables.append(syl)
        ipas.append(ipa)
    w.strs("kokoro.zh.ipa.syllables", ok_syllables)
    w.strs("kokoro.zh.ipa.ipa", ipas)
    print(f"zh tables: {len(words)} jieba words, {len(chars)} characters, {len(phrases)} phrases, "
          f"{len(ok_syllables)}/{len(syllables)} syllables with IPA")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="hexgrad/Kokoro-82M snapshot directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voices", nargs="*", help="voice names to include (default: every voices/*.pt present)")
    ap.add_argument("--no-zh", action="store_true", help="leave out the Mandarin text front end tables")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import torch

    cfg = json.load(open(os.path.join(args.input, "config.json"), encoding="utf-8"))
    ckpt = os.path.join(args.input, "kokoro-v1_0.pth")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)

    w = GgufWriter("kokoro")
    w.str("general.name", "Kokoro-82M v1.0")
    w.str("kokoro.config", json.dumps(cfg, ensure_ascii=False))
    vocab = sorted(cfg["vocab"].items(), key=lambda kv: kv[1])
    w.strs("kokoro.vocab.symbols", [k for k, _ in vocab])
    w.i32s("kokoro.vocab.ids", [v for _, v in vocab])

    for top in ("bert", "bert_encoder", "predictor", "text_encoder", "decoder"):
        state = {k[7:] if k.startswith("module.") else k: v for k, v in sd[top].items()}
        state = fold_weight_norm(state)
        for name in sorted(state):
            if top == "bert" and name.startswith("pooler."):
                continue  # CustomAlbert returns the last hidden state only
            w.tensor(f"{top}.{name}", state[name].float().numpy(), F32)

    voice_files = sorted(glob.glob(os.path.join(args.input, "voices", "*.pt")))
    if args.voices:
        wanted = set(args.voices)
        voice_files = [f for f in voice_files if os.path.splitext(os.path.basename(f))[0] in wanted]
    names = []
    for f in voice_files:
        name = os.path.splitext(os.path.basename(f))[0]
        pack = torch.load(f, map_location="cpu", weights_only=True).float().reshape(510, -1).numpy()
        w.tensor(f"voice.{name}", pack, F32)
        names.append(name)
    if not names:
        sys.exit("error: no voices found (voices/*.pt)")
    w.strs("kokoro.voices", names)
    print(f"voices: {', '.join(names)}")

    if not args.no_zh:
        try:
            zh_tables(w)
        except ImportError as e:
            sys.exit(f"error: the Mandarin tables need misaki[zh], pypinyin and jieba ({e}); pass --no-zh to skip them")
    w.write(args.out, quiet=args.quiet)


if __name__ == "__main__":
    main()
