#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Send the repetition-study prompt set (prompts.json) through one chat surface and record every reply.

Surfaces (each sends what its real client sends, nothing more, unless --sampling adds fields):
  webui   TensorSharp's Web UI: POST /api/sessions, then /api/chat {messages, maxTokens, think, replaceActive,
          sessionId, newChat on the first turn}; media through /api/upload (imagePaths / audioPaths).
  tagent  TensorAgent's page: the same endpoints with the host's cookie (--cookie or --connection
          connection.json from eng/validation/TensorAgentHost) and its chatBody() fields
          {messages, maxTokens, think, knowledgeCollections, sessionId}.
  openai  POST /v1/chat/completions, streaming, media as image_url data URIs / input_audio parts. --think on|off
          sends chat_template_kwargs.enable_thinking; omit sends nothing (the server's default).
  llama   llama-server's /v1/chat/completions, the same body as openai.

Every reply is analysed by loopdetect.analyze; one JSON line per turn goes to --out:
  {id, lang, cat, turn, surface, setting, text, thinking, finish_reason, tokens, seconds, tok_s, notes, loop...}
The host's own notes (the repetition guard's italic sentences) are cut from `text` and kept in `notes`.

  python3 eng/lfm3/loop/run.py --surface webui --base http://127.0.0.1:5191 --think on \\
      --media artifacts/loop/media --out artifacts/loop/runs/webui-think.jsonl
"""
import argparse
import base64
import http.client
import json
import mimetypes
import os
import re
import sys
import time
import urllib.parse
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loopdetect  # noqa: E402

HOST_NOTE = re.compile(r"\n*_\((?:The model's output started repeating itself|The model ended this turn without)[^)]*\)_", re.S)
GUARD_NOTE = re.compile(r"repeat(?:ing|ed) itself|started repeating", re.I)


class Http:
    def __init__(self, base, cookie=None, timeout=1800):
        u = urllib.parse.urlparse(base.rstrip("/"))
        self.host, self.port, self.timeout = u.hostname, u.port, timeout
        self.cookie = cookie

    def _headers(self, ctype="application/json"):
        h = {"Content-Type": ctype}
        if self.cookie:
            h["Cookie"] = self.cookie
        return h

    def request(self, method, path, body=None, ctype="application/json"):
        c = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
        c.request(method, path, data, self._headers(ctype))
        return c, c.getresponse()

    def json(self, method, path, body=None):
        c, r = self.request(method, path, body)
        data = r.read()
        c.close()
        if r.status >= 400:
            raise RuntimeError(f"{method} {path}: HTTP {r.status}: {data[:300]!r}")
        return json.loads(data) if data else {}

    def sse(self, path, body):
        c, r = self.request("POST", path, body)
        if r.status >= 400:
            raise RuntimeError(f"POST {path}: HTTP {r.status}: {r.read()[:500]!r}")
        buf = b""
        while True:
            chunk = r.readline()
            if not chunk:
                break
            line = chunk.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue
        c.close()

    def upload(self, path):
        boundary = uuid.uuid4().hex
        with open(path, "rb") as f:
            data = f.read()
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{os.path.basename(path)}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        c, r = self.request("POST", "/api/upload", body, f"multipart/form-data; boundary={boundary}")
        out = r.read()
        c.close()
        if r.status >= 400:
            raise RuntimeError(f"upload {path}: HTTP {r.status}: {out[:300]!r}")
        return json.loads(out)


def kind_of(path):
    ext = path.rsplit(".", 1)[-1].lower()
    return "audio" if ext in ("wav", "mp3", "flac", "ogg", "m4a") else "image"


def data_uri(path):
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


# ---------------------------------------------------------------------------------------------- surfaces

def web_turn(http, history, session, args, first):
    body = {"messages": history, "maxTokens": args.max_tokens, "think": args.think == "on"}
    if args.surface == "webui":
        body["replaceActive"] = True
        if first:
            body["newChat"] = True
    else:
        body["knowledgeCollections"] = []
    body["sessionId"] = session
    if args.no_skills:
        body["skills_discovery"] = False
    if args.speech:
        body["speech"] = True
    body.update(args.sampling)
    text, thinking, done, t0, ttft = [], [], None, time.time(), None
    for ev in http.sse("/api/chat", body):
        tok = ev.get("token")
        if tok:
            if ttft is None:
                ttft = time.time() - t0
            text.append(tok)
        if ev.get("thinking"):
            thinking.append(ev["thinking"])
        if ev.get("done"):
            done = ev
    secs = time.time() - t0
    done = done or {}
    return {"raw": "".join(text), "thinking": "".join(thinking), "finish_reason": None,
            "tokens": done.get("tokenCount") or done.get("tokens"), "seconds": secs, "ttft": ttft,
            "tok_s": done.get("tokPerSec"), "prompt_tokens": done.get("promptTokens"),
            "kv_reused": done.get("kvReusedTokens"), "truncated": done.get("truncated"), "error": done.get("error"),
            "audio_url": done.get("audioUrl"), "audio_seconds": done.get("audioSeconds"),
            "done": {k: v for k, v in done.items() if k not in ("token",)}}


def oai_turn(http, messages, args, model):
    body = {"model": model, "messages": messages, "stream": True, "max_tokens": args.max_tokens,
            "stream_options": {"include_usage": True}}
    if args.think != "omit":
        body["chat_template_kwargs"] = {"enable_thinking": args.think == "on"}
    body.update(args.sampling)
    text, thinking, finish, usage, t0, ttft, timings = [], [], None, None, time.time(), None, None
    for ev in http.sse("/v1/chat/completions", body):
        if ev.get("usage"):
            usage = ev["usage"]
        if ev.get("timings"):
            timings = ev["timings"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                text.append(d["content"])
            if d.get("reasoning_content"):
                thinking.append(d["reasoning_content"])
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    secs = time.time() - t0
    toks = (usage or {}).get("completion_tokens")
    tok_s = None
    if timings and timings.get("predicted_per_second"):
        tok_s = timings["predicted_per_second"]
    elif toks and ttft is not None and secs > ttft:
        tok_s = (toks - 1) / (secs - ttft) if toks > 1 else None
    return {"raw": "".join(text), "thinking": "".join(thinking), "finish_reason": finish, "tokens": toks,
            "seconds": secs, "ttft": ttft, "tok_s": tok_s, "prompt_tokens": (usage or {}).get("prompt_tokens")}


def oai_user(text, media_paths):
    if not media_paths:
        return {"role": "user", "content": text}
    parts = []
    for p in media_paths:
        if kind_of(p) == "audio":
            with open(p, "rb") as f:
                parts.append({"type": "input_audio", "input_audio": {"data": base64.b64encode(f.read()).decode(),
                                                                     "format": p.rsplit(".", 1)[1].lower()}})
        else:
            parts.append({"type": "image_url", "image_url": {"url": data_uri(p)}})
    parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}


def run_case(http, case, args, model):
    media = [os.path.join(args.media, m) for m in case.get("media", [])]
    rows, history = [], []
    session = None
    if args.surface in ("webui", "tagent"):
        q = "?conversation=new" if args.surface == "tagent" else ""
        session = http.json("POST", "/api/sessions" + q, {}).get("sessionId")
    for ti, user_text in enumerate(case["turns"]):
        if args.surface in ("webui", "tagent"):
            msg = {"role": "user", "content": user_text}
            if ti == 0 and media:
                ups = [http.upload(p) for p in media]
                imgs = [u["file"] for u, p in zip(ups, media) if kind_of(p) == "image"]
                auds = [u["file"] for u, p in zip(ups, media) if kind_of(p) == "audio"]
                if imgs:
                    msg["imagePaths"] = imgs
                if auds:
                    msg["audioPaths"] = auds
            history.append(msg)
            r = web_turn(http, history, session, args, ti == 0)
        else:
            history.append(oai_user(user_text, media if ti == 0 else None))
            r = oai_turn(http, history, args, model)
        raw = r.pop("raw")
        notes = HOST_NOTE.findall(raw)
        text = HOST_NOTE.sub("", raw)
        guard = any(GUARD_NOTE.search(n) for n in notes)
        finish = r.get("finish_reason") or ("repetition" if guard else None)
        a = loopdetect.analyze(text, finish)
        # The repetition guard's retry note, when the skills loop retried, sits in the answer text too.
        row = dict(id=case["id"], lang=case["lang"], cat=case["cat"], turn=ti, surface=args.surface,
                   setting=args.setting, text=text, notes=notes, **r)
        row["finish_reason"] = finish
        row.update({"loop": a["loop"], "why": a["why"], "dup_frac": a["dup_frac"],
                    "max_unit_count": a["max_unit_count"], "max_run": a["max_run"]})
        if r.get("thinking"):
            ta = loopdetect.analyze(r["thinking"])
            row["thinking_loop"] = ta["loop"]
            row["thinking_why"] = ta["why"]
        if r.get("audio_url") and args.speech_dir:
            os.makedirs(args.speech_dir, exist_ok=True)
            wav = os.path.join(args.speech_dir, f"{case['id']}-{ti}.wav")
            c, resp = http.request("GET", r["audio_url"])
            with open(wav, "wb") as wf:
                wf.write(resp.read())
            c.close()
            row["wav"] = wav
        rows.append(row)
        assistant = {"role": "assistant", "content": text}
        if r.get("done", {}).get("audioId"):
            assistant["audioId"] = r["done"]["audioId"]  # the Web UI sends a spoken turn back by its id
        history.append(assistant)
        flag = "LOOP " + "; ".join(a["why"]) if a["loop"] else "ok"
        if r.get("audio_seconds"):
            flag += f" audio={r['audio_seconds']:.1f}s"
        print(f"{case['id']}#{ti} {flag} tokens={r.get('tokens')} {r.get('seconds', 0):.1f}s "
              f"tok/s={r.get('tok_s') and round(r['tok_s'], 1)} | {text[:90]!r}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", required=True, choices=["webui", "tagent", "openai", "llama"])
    ap.add_argument("--base", help="server URL (tagent: read from --connection)")
    ap.add_argument("--connection", help="TensorAgentHost connection.json (baseUrl + cookie)")
    ap.add_argument("--cookie")
    ap.add_argument("--think", choices=["on", "off", "omit"], default="omit")
    ap.add_argument("--sampling", default="{}", help="JSON object of extra request fields (sampling)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--no-skills", action="store_true", help="tagent: send skills_discovery false")
    ap.add_argument("--speech", action="store_true", help="webui/tagent: spoken replies (speech: true)")
    ap.add_argument("--speech-dir", help="download each spoken reply's WAV here")
    ap.add_argument("--media", default="artifacts/loop/media")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json"))
    ap.add_argument("--cases", default="", help="comma-separated id prefixes")
    ap.add_argument("--no-media", action="store_true", help="skip cases with media")
    ap.add_argument("--setting", default="", help="label stored with each row")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.sampling = json.loads(args.sampling)
    if args.connection:
        conn = json.load(open(args.connection))
        args.base = args.base or conn["baseUrl"]
        args.cookie = args.cookie or conn.get("cookie")
    http = Http(args.base, args.cookie)
    model = None
    if args.surface in ("openai", "llama"):
        model = http.json("GET", "/v1/models")["data"][0]["id"]
    cases = json.load(open(args.prompts))["cases"]
    if args.cases:
        pre = tuple(args.cases.split(","))
        cases = [c for c in cases if c["id"].startswith(pre)]
    if args.no_media:
        cases = [c for c in cases if not c.get("media")]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for case in cases:
            try:
                rows = run_case(http, case, args, model)
            except Exception as e:  # keep going: one failed case must not lose the run
                print(f"{case['id']} ERROR {e}", flush=True)
                rows = [dict(id=case["id"], lang=case["lang"], cat=case["cat"], turn=0, surface=args.surface,
                             setting=args.setting, error=str(e), text="", loop=False)]
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    sys.exit(main())
