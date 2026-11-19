#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""LFM3-Omni end to end: the same greedy OpenAI chat requests (images, videos with their soundtrack, ASR, audio QA,
mixed multi-turn) to llama-server and TensorSharp.Server.Host; per case both replies, prompt tokens, the
token-identical prefix (tokenized by llama-server), and for ASR the WER against the spoken text and between the two.

    python3 eng/lfm3/omni-in/compare.py --media artifacts/lfm3/omni-in/media/manifest.json \
        --llama http://127.0.0.1:8181 --ts http://127.0.0.1:5181 --out artifacts/lfm3/omni-in/e2e/metal [--only img_,asr_]

Start the servers with eng/lfm3/omni-in/run-e2e.sh (or by hand: llama-server -m ... -mm ... --jinja; TensorSharp.Server.Host
--model ... --mmproj ... --no-skills --no-multi-agent)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omni_client as oc  # noqa: E402

ASR_SYSTEM = "Perform ASR in English."
AUDIO_QA_SYSTEM = "Answer questions about the provided audio clip. Base every answer only on what can be heard."
VIDEO_SYSTEM = "Answer questions about the video using what you see and hear."


def cases(m):
    img, aud, vid = m["images"], m["audio"], m["video"]
    c = []

    def one(cid, media, text, max_tokens=160, system=None, ref=None, kind=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [oc.user(media, text)]
        c.append({"id": cid, "kind": kind or cid.split("_")[0], "messages": msgs, "max_tokens": max_tokens, "ref": ref})

    one("img_zebra", [("image", img["photo_zebra"]["path"])], "What animal is in this photo? Describe it in two sentences.")
    one("img_parrot", [("image", img["photo_parrot"]["path"])], "Describe this image in one sentence.")
    one("img_newspaper", [("image", img["newspaper"]["path"])], "What is the main headline on this newspaper front page, and what date is printed?")
    one("img_diagram", [("image", img["diagram"]["path"])], "Describe the steps of this flowchart in order. What happens when the accuracy is not above 90%?")
    one("img_receipt", [("image", img["receipt"]["path"])], "What is the name of the cafe and the total amount? List every item with its price.")
    one("img_large", [("image", img["aerial_large"]["path"])], "Describe this landscape photo in two sentences.")
    one("img_two", [("image", img["photo_zebra"]["path"]), ("image", img["photo_parrot"]["path"])], "What animal is in each of the two images?")
    one("vid_test3", [("video", vid["test3"]["path"])], "What happens in this video? Describe what you see and what you hear.", 200, VIDEO_SYSTEM)
    one("vid_recipe", [("video", vid["recipe_narrated"]["path"])],
        "What are the steps shown in this video? For how long and at what temperature should the bread bake?", 200, VIDEO_SYSTEM)
    if "signal" in vid:
        one("vid_signal", [("video", vid["signal"]["path"])], "Describe what you see in this short video.", 120)
    for name, a in aud.items():
        if name == "long_harvard":
            continue
        one(f"asr_{name}", [("audio", a["path"])], None, 256, ASR_SYSTEM, a.get("text"), "asr")
    one("asr_long_harvard", [("audio", aud["long_harvard"]["path"])], None, 400, ASR_SYSTEM, aud["long_harvard"]["text"], "asr")
    one("aqa_story", [("audio", aud["story_moira"]["path"])], "Why was the train delayed, and what did Maria bring?", 120, AUDIO_QA_SYSTEM)
    if "ticket" in aud:
        one("aqa_ticket", [("audio", aud["ticket"]["path"])], "How does the speaker feel, and what do they want?", 120, AUDIO_QA_SYSTEM)
    # Mixed image + text multi-turn: turn 1's reply is llama.cpp's (filled in at run time) so both servers see the same
    # history; turn 2 is compared.
    c.append({"id": "mt_zebra", "kind": "multiturn", "max_tokens": 160, "ref": None,
              "turn1": [oc.user([("image", img["photo_zebra"]["path"])], "What animal is this?")],
              "turn2": "What colours are its stripes, and where in the wild might it live?"})
    c.append({"id": "mt_receipt", "kind": "multiturn", "max_tokens": 160, "ref": None,
              "turn1": [oc.user([("image", img["receipt"]["path"])], "What is the total on this receipt?")],
              "turn2": "Which single item was the most expensive, and how much was the tax?"})
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--llama", required=True)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated case id prefixes")
    ap.add_argument("--reuse-llama", default=None,
                    help="llama.json of an earlier run: reuse its llama.cpp replies (compare another TensorSharp backend)")
    a = ap.parse_args()
    with open(a.media) as f:
        manifest = json.load(f)
    os.makedirs(a.out, exist_ok=True)
    previous = {}
    if a.reuse_llama:
        with open(a.reuse_llama) as f:
            previous = {r["id"]: r for r in json.load(f)}
    results = []
    for c in cases(manifest):
        if a.only and not any(c["id"].startswith(p) for p in a.only.split(",")):
            continue
        r = {"id": c["id"], "kind": c["kind"], "ref": c.get("ref")}
        if c["kind"] == "multiturn":
            if c["id"] in previous:
                first = previous[c["id"]]["turn1_reply"]
            else:
                first = oc.chat(a.llama, oc.body(c["turn1"], c["max_tokens"]))["text"]
            r["turn1_reply"] = first
            msgs = c["turn1"] + [{"role": "assistant", "content": first}, {"role": "user", "content": c["turn2"]}]
        else:
            msgs = c["messages"]
        req = oc.body(msgs, c["max_tokens"], extra={"cache_prompt": False})
        if c["id"] in previous:
            r["llama"] = previous[c["id"]]["llama"]
        else:
            r["llama"] = oc.chat(a.llama, req)
        r["ts"] = oc.chat(a.ts, req)
        lt, tt = oc.tokenize(a.llama, r["llama"]["text"]), oc.tokenize(a.llama, r["ts"]["text"])
        r["tokens"] = {"llama": len(lt), "ts": len(tt), "common_prefix": oc.common_prefix(lt, tt), "identical": lt == tt}
        if c["kind"] == "asr":
            if c.get("ref"):
                r["wer_ref"] = {"llama": oc.wer(c["ref"], r["llama"]["text"]), "ts": oc.wer(c["ref"], r["ts"]["text"])}
            r["wer_ts_vs_llama"] = oc.wer(r["llama"]["text"], r["ts"]["text"])
        results.append(r)
        pl = (r["llama"].get("usage") or {}).get("prompt_tokens")
        pt = (r["ts"].get("usage") or {}).get("prompt_tokens")
        print(f"{r['id']:<18} prompt {pl}/{pt}  prefix {r['tokens']['common_prefix']}/{r['tokens']['llama']},{r['tokens']['ts']}"
              f"{' IDENTICAL' if r['tokens']['identical'] else ''}  ttft {r['llama']['ttft']:.2f}/{r['ts']['ttft']:.2f}s"
              + (f"  WER ref {r['wer_ref']['llama']:.3f}/{r['wer_ref']['ts']:.3f}" if 'wer_ref' in r else "")
              + (f"  WER ts-vs-llama {r['wer_ts_vs_llama']:.3f}" if 'wer_ts_vs_llama' in r else ""), flush=True)
        print(f"   llama [{r['llama']['finish']}]: {r['llama']['text'][:300]!r}\n   ts    [{r['ts']['finish']}]: {r['ts']['text'][:300]!r}", flush=True)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    if not a.reuse_llama:
        with open(os.path.join(a.out, "llama.json"), "w") as f:
            json.dump([{"id": r["id"], "llama": r["llama"], "turn1_reply": r.get("turn1_reply")} for r in results], f, indent=1)
    write_markdown(results, os.path.join(a.out, "results.md"))


def write_markdown(results, path):
    rows = ["| case | prompt tokens llama / TS | token-identical prefix | identical | TTFT llama / TS (s) | WER vs spoken (llama / TS) | WER TS vs llama |",
            "|---|---|---|---|---|---|---|"]
    for r in results:
        pl = (r["llama"].get("usage") or {}).get("prompt_tokens")
        pt = (r["ts"].get("usage") or {}).get("prompt_tokens")
        t = r["tokens"]
        wref = f"{r['wer_ref']['llama']:.3f} / {r['wer_ref']['ts']:.3f}" if "wer_ref" in r else ""
        wts = f"{r['wer_ts_vs_llama']:.3f}" if "wer_ts_vs_llama" in r else ""
        rows.append(f"| {r['id']} | {pl} / {pt} | {t['common_prefix']} of {t['llama']} / {t['ts']} | {'yes' if t['identical'] else 'no'} | "
                    f"{r['llama']['ttft']:.2f} / {r['ts']['ttft']:.2f} | {wref} | {wts} |")
    rows.append("")
    for r in results:
        rows.append(f"### {r['id']}\n\n- llama.cpp: {r['llama']['text']!r}\n- TensorSharp: {r['ts']['text']!r}\n")
    with open(path, "w") as f:
        f.write("\n".join(rows))


if __name__ == "__main__":
    sys.exit(main())
