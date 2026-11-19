#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""The same LFM3-Omni media prompts through TensorSharp's three chat surfaces - /v1/chat/completions, /v1/responses
(input_image / input_audio / input_video) and the Web UI path TensorAgent uses (/api/upload, then /api/chat with
imagePaths / audioPaths / videoFilePaths) - greedy: the replies should be the same text, and a follow-up turn in one
/api/chat session should reuse the cached prompt through the media (kvReusedTokens), a turn about another image not.

    python3 eng/lfm3/omni-in/surfaces.py --media artifacts/lfm3/omni-in/media/manifest.json \
        --ts http://127.0.0.1:5181 --out artifacts/lfm3/omni-in/surfaces"""
import argparse
import base64
import json
import os
import sys
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omni_client as oc  # noqa: E402

GREEDY = {"temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0, "repeat_penalty": 1.0}


def post(base, path, payload, stream=False):
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=1200)
    if not stream:
        return json.loads(resp.read())
    return resp


def upload(base, path):
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        data = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{os.path.basename(path)}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(base.rstrip("/") + "/api/upload", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def web_chat(base, messages, session, max_tokens):
    payload = dict(GREEDY, messages=messages, maxTokens=max_tokens, sessionId=session, think=False, tools=[])
    text, done = [], None
    with post(base, "/api/chat", payload, stream=True) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:].strip())
            if "token" in ev and ev["token"]:
                text.append(ev["token"])
            if ev.get("done"):
                done = ev
    return "".join(text), done


def responses(base, parts, text, max_tokens, system=None):
    content = []
    for kind, path in parts:
        if kind == "image":
            content.append({"type": "input_image", "image_url": oc.data_uri(path)})
        elif kind == "audio":
            with open(path, "rb") as f:
                content.append({"type": "input_audio", "input_audio": {"data": base64.b64encode(f.read()).decode(),
                                                                       "format": path.rsplit(".", 1)[1]}})
        else:
            content.append({"type": "input_video", "input_video": {"url": oc.data_uri(path)}})
    if text:
        content.append({"type": "input_text", "text": text})
    payload = dict(GREEDY, model=oc.model_id(base), input=[{"type": "message", "role": "user", "content": content}],
                   max_output_tokens=max_tokens, stream=False)
    if system:
        payload["instructions"] = system
    r = post(base, "/v1/responses", payload)
    out = "".join(c.get("text", "") for item in r.get("output", []) if item.get("type") == "message" for c in item["content"])
    return out, r.get("usage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama", default=None, help="unused (run-e2e.sh passes it to every tool)")
    a = ap.parse_args()
    with open(a.media) as f:
        m = json.load(f)
    os.makedirs(a.out, exist_ok=True)
    ts = a.ts
    cases = [
        ("image", m["images"]["receipt"]["path"], "What is the total on this receipt?", None, 60),
        ("audio", m["audio"]["harvard_daniel"]["path"], None, "Perform ASR in English.", 120),
        ("video", m["video"]["recipe_narrated"]["path"], "For how long and at what temperature should the bread bake?",
         "Answer questions about the video using what you see and hear.", 60),
    ]
    results = []
    for kind, path, text, system, n in cases:
        msgs = ([{"role": "system", "content": system}] if system else []) + [oc.user([(kind, path)], text)]
        chat = oc.chat(ts, oc.body(msgs, n))
        resp_text, resp_usage = responses(ts, [(kind, path)], text, n, system)
        up = upload(ts, path)
        wmsg = {"role": "user", "content": text or ""}
        if kind == "image":
            wmsg["imagePaths"] = [up["file"]]
        elif kind == "audio":
            wmsg["audioPaths"] = [up["file"]]
        else:
            wmsg["isVideo"] = True
            wmsg["imagePaths"] = up.get("frames") or []
            wmsg["videoFilePaths"] = [up["file"]]
        wmsgs = ([{"role": "system", "content": system}] if system else []) + [wmsg]
        session = post(ts, "/api/sessions", {})["sessionId"]
        web_text, web_done = web_chat(ts, wmsgs, session, n)
        row = {"kind": kind, "chat": chat["text"], "chat_prompt": (chat.get("usage") or {}).get("prompt_tokens"),
               "responses": resp_text, "responses_prompt": (resp_usage or {}).get("input_tokens"),
               "web": web_text, "web_prompt": (web_done or {}).get("promptTokens"),
               "same_text": chat["text"] == resp_text == web_text}
        results.append(row)
        print(f"{kind:<6} prompt chat/responses/web {row['chat_prompt']}/{row['responses_prompt']}/{row['web_prompt']}  "
              f"same text: {row['same_text']}\n  chat: {chat['text'][:160]!r}\n  resp: {resp_text[:160]!r}\n  web:  {web_text[:160]!r}", flush=True)

    # Prefix reuse through the Web UI session: a follow-up about the same image, then a new chat with another image.
    img = upload(ts, m["images"]["photo_zebra"]["path"])["file"]
    other = upload(ts, m["images"]["photo_parrot"]["path"])["file"]
    session = post(ts, "/api/sessions", {})["sessionId"]
    t1 = [{"role": "user", "content": "What animal is this?", "imagePaths": [img]}]
    a1, d1 = web_chat(ts, t1, session, 40)
    t2 = t1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "What colours are its stripes?"}]
    a2, d2 = web_chat(ts, t2, session, 40)
    session2 = post(ts, "/api/sessions", {})["sessionId"]
    t3 = [{"role": "user", "content": "What animal is this?", "imagePaths": [other]}]
    a3, d3 = web_chat(ts, t3, session2, 40)
    reuse = {"turn1": {k: d1.get(k) for k in ("promptTokens", "kvReusedTokens")},
             "followup": {k: d2.get(k) for k in ("promptTokens", "kvReusedTokens")},
             "other_image": {k: d3.get(k) for k in ("promptTokens", "kvReusedTokens")},
             "replies": [a1, a2, a3]}
    print("web session reuse:", json.dumps({k: v for k, v in reuse.items() if k != "replies"}), flush=True)
    print("  ", [r[:80] for r in reuse["replies"]], flush=True)
    with open(os.path.join(a.out, "surfaces.json"), "w") as f:
        json.dump({"cases": results, "reuse": reuse}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
