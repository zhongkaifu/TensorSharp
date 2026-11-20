#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""ASR word error rate of LFM3-Omni through one OpenAI-compatible server, for a sampling-preset comparison: each clip
with a known text (the omni-in media manifest's speech, plus artifacts/loop/media/repeats.wav, whose text repeats
words on purpose: "no, no, no", "very, very, very", "step by step, step by step") is sent with the system prompt
"Perform ASR in English." (as eng/lfm3/omni-in/compare.py sends it) and scored against the spoken text.

  python3 eng/lfm3/loop/asr.py --url http://127.0.0.1:5191 --manifest MEDIA/manifest.json --extra artifacts/loop/media \\
      [--sampling '{"temperature":0}'] --out artifacts/loop/runs/asr-ts.json

--sampling adds request fields (none by default: the server's own defaults; a TensorSharp config pins its values).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "omni-in"))
import omni_client as oc  # noqa: E402

REPEATS = ("No, no, no, I did not say that. The meeting is very, very, very important, so please, please be on time. "
           "We will go over the plan step by step, step by step.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--extra", help="directory holding repeats.wav")
    ap.add_argument("--sampling", default="{}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    extra = json.loads(a.sampling)
    man = json.load(open(a.manifest))
    root = os.path.dirname(os.path.abspath(a.manifest))
    clips = [(k, os.path.join(root, v["path"]), v["text"]) for k, v in man["audio"].items() if v.get("text")]
    if a.extra and os.path.exists(os.path.join(a.extra, "repeats.wav")):
        clips.append(("repeats", os.path.join(a.extra, "repeats.wav"), REPEATS))
    rows = []
    for name, path, ref in clips:
        msgs = [{"role": "system", "content": "Perform ASR in English."}, oc.user([("audio", path)], None)]
        req = oc.body(msgs, 400 if name == "long_harvard" else 256)
        for k in list(req):
            if k in ("temperature", "top_k", "top_p", "min_p", "repeat_penalty", "seed"):
                del req[k]
        req.update(extra)
        r = oc.chat(a.url, req)
        w = oc.wer(ref, r["text"])
        rows.append({"id": name, "wer": w, "text": r["text"], "ref": ref, "finish": r.get("finish")})
        print(f"{name:<18} WER {w:.3f}  {r['text'][:120]!r}", flush=True)
    mean = sum(r["wer"] for r in rows) / max(1, len(rows))
    print(json.dumps({"clips": len(rows), "mean_wer": round(mean, 4)}))
    with open(a.out, "w") as f:
        json.dump({"mean_wer": mean, "rows": rows}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
