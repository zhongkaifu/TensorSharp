# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Shared helpers of the LFM3-Omni end-to-end tools: OpenAI chat requests with media parts (identical bodies for
llama-server and TensorSharp.Server.Host), streamed with TTFT, tokenization through llama-server, WER."""
import base64
import json
import mimetypes
import re
import time
import urllib.request

MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".wav": "audio/wav", ".mp3": "audio/mpeg",
        ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}


def data_uri(path):
    ext = path[path.rfind("."):].lower()
    mime = MIME.get(ext) or mimetypes.guess_type(path)[0]
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def part(kind, path):
    """One OpenAI content part. Both servers read image_url {url}, input_audio {data, format} and video_url {url}."""
    if kind == "image":
        return {"type": "image_url", "image_url": {"url": data_uri(path)}}
    if kind == "audio":
        with open(path, "rb") as f:
            raw = base64.b64encode(f.read()).decode()
        return {"type": "input_audio", "input_audio": {"data": raw, "format": path[path.rfind(".") + 1:].lower()}}
    if kind == "video":
        return {"type": "video_url", "video_url": {"url": data_uri(path)}}
    raise ValueError(kind)


def user(media, text):
    """A user message: media parts first (TensorSharp renders a message's media before its text), then the text."""
    content = [part(k, p) for k, p in media]
    if text:
        content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def body(messages, max_tokens, stream=True, extra=None):
    b = {"model": "lfm3-omni", "messages": messages, "max_tokens": max_tokens, "stream": stream,
         "temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0, "repeat_penalty": 1.0, "seed": 0,
         "chat_template_kwargs": {"enable_thinking": False}}
    if stream:
        b["stream_options"] = {"include_usage": True}
    if extra:
        b.update(extra)
    return b


_MODEL_IDS = {}


def model_id(base):
    """The id the server hosts (TensorSharp refuses any other name; llama-server takes any)."""
    if base not in _MODEL_IDS:
        with urllib.request.urlopen(base.rstrip("/") + "/v1/models", timeout=30) as r:
            _MODEL_IDS[base] = json.loads(r.read())["data"][0]["id"]
    return _MODEL_IDS[base]


def chat(base, request, timeout=1200):
    """Stream one chat completion. Returns text, TTFT and total seconds (from the request), usage."""
    request = dict(request, model=model_id(base))
    data = json.dumps(request).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    t0 = time.perf_counter()
    ttft = None
    text, reasoning, usage, finish = [], [], None, None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = d.get("content")
                rpiece = d.get("reasoning_content")
                if (piece or rpiece) and ttft is None:
                    ttft = time.perf_counter() - t0
                if piece:
                    text.append(piece)
                if rpiece:
                    reasoning.append(rpiece)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    total = time.perf_counter() - t0
    return {"text": "".join(text), "reasoning": "".join(reasoning), "ttft": ttft, "total": total, "usage": usage,
            "finish": finish}


def tokenize(llama_base, text):
    req = urllib.request.Request(llama_base.rstrip("/") + "/tokenize",
                                 data=json.dumps({"content": text, "add_special": False}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["tokens"]


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


_NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
        "8": "eight", "9": "nine"}


def words(s):
    s = s.lower().replace("'", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return [w for w in s.split() if w]


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def wait_ready(base, path, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base.rstrip("/") + path, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False
