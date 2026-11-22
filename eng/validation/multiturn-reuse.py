#!/usr/bin/env python3
"""Measure cross-turn KV reuse of one multi-turn conversation on a running TensorSharp.Server.Host.

    multiturn-reuse.py --url http://127.0.0.1:5120 --surface openai --think off --turns 5 \
        --out artifacts/kvreuse/openai-off.json

Each turn's prompt should reuse the whole previous turn (its prompt and its reply) except
what the template legitimately re-renders. Per turn the report shows

  prompt   the prompt tokens the server counted
  reused   the tokens it served from the prefix cache
  prefill  prompt - reused (what the turn forwarded again)
  carry    the previous prompt + the previous reply's tokens (what a perfect cache holds)
  new      prompt - carry (the new user message and template framing; negative or small
           numbers mean the template re-rendered history shorter, e.g. stripped reasoning)
  waste    prefill - max(new, 0): tokens re-forwarded that the previous turn already held

Surfaces (each re-sends the client's history the way that client does):
  webui      POST /api/chat with a Web UI session (SSE; assistant turns carry `thinking`)
  openai     POST /v1/chat/completions, non-streamed, stateless history
  openai-stream  the same, streamed (SSE deltas)
  responses  POST /v1/responses, stateless input items (assistant output_text items)
  ollama     POST /api/chat/ollama, non-streamed

Scenarios beyond plain chat:
  --task tools        every turn needs the client's get_weather tool; the client answers each call and sends
                      the round back (rows "2.1", "2.2" are those rounds)
  --task code         every turn needs the server's own shell (start the server with --code-exec)
  --image P --image-turns 1,3     attach an image to those turns (webui upload, OpenAI data URI, Ollama images)
  --speech-turns 1,3  webui: ask for a spoken reply on those turns (LFM3-Omni; the reply goes back with its
                      audioId as the page sends it), so mixed spoken/text conversations can be measured
  --attach-file F     a long document, --chunk-chars of it per turn (a few thousand tokens a turn)
  --tensoragent --cookie C [--reopen-before 3]   TensorAgent's loopback host: sessions bound to a saved
                      conversation, and the chat reopened (a new session) before the named turns

--replay REPORT re-sends a finished run's last turn (its whole history up to the last user message, as
the client sent it) once and compares the reply with the run's: run it against a FRESH server (a new
process, --no-prefix-cache) to check that the cached conversation's last turn answers exactly as a
re-prefill of the same conversation does. The report keeps the run's history for that ("history").

--echo-reasoning sends the previous replies' reasoning back (OpenAI `reasoning_content`,
Ollama `thinking`, Responses `reasoning` items); the Web UI always sends `thinking`.
--tools adds a client tool declaration (OpenAI/Responses/Ollama) so tool-capable templates
render their tool block; no tool is required to answer.

Prompt/reuse numbers come from each response (done frame, usage.prompt_tokens_details /
input_tokens_details.cached_tokens, Ollama prompt_eval_count) and, with --log, also from the
server's JSONL chat.complete lines (all rounds of a turn; TTFT). Greedy sampling is sent
(temperature 0, top_k 1) so a rerun on another build is comparable; the server must accept
request sampling (--sampling-precedence request) for that to hold.
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

DOC_A = """TensorSharp serves large language models through one engine that batches requests continuously.
A request's prompt is tokenized, matched against a radix tree of cached prefixes, and only the part the
cache does not hold is forwarded through the model. The tree keys token ids and media content ids, so
two conversations that start with the same system prompt and tool declarations share that prefix, while
each conversation's own turns stay in its own scope. When a turn finishes, the engine keeps the
sequence's end state - its attention cache rows and, for recurrent families, the recurrent state - so the
next turn of the same conversation resumes exactly where the previous reply stopped. Families whose
caches cannot be truncated, such as gated delta networks or short convolutions, keep checkpoints at the
end of the shared prefix and at the end of long prompts instead, because they cannot rewind into a
longer state. Sliding-window families keep circular caches and may rewind only a few tokens.

The chat layer renders each request through the model's own chat template. Because templates often
render a past assistant message differently from the way the model generated it - stripping its
reasoning, dropping an empty thinking block, or normalizing whitespace - the server records the raw
tokens of every reply it produced and, when a client sends that reply back, splices the raw tokens into
the new prompt in place of the re-rendered text. A client that edits its history, or a template that
changes earlier content when a new message arrives, makes the new prompt diverge from the cached one,
and everything after the divergence is forwarded again. The longest common prefix decides the reuse.

Operators run the server with a model file, a backend and a port. The ggml_metal backend runs on Apple
GPUs, ggml_cuda and the direct cuda backend on NVIDIA GPUs, ggml_vulkan on other GPUs, and ggml_cpu or
the managed cpu backend anywhere. Options control the context length, the KV cache type (f16, q8_0 or
q4_0), the number of concurrent sequences, speculative decoding with a draft model or multi-token
prediction heads, the thinking budget, and whether skills, code execution and sub-agents are offered.
Clients talk to it through an OpenAI-compatible chat completions route, the OpenAI Responses route, an
Ollama-compatible route, and the bundled Web UI, which keeps a session per conversation. Every route
reports the prompt tokens it counted and how many of them came from the cache.
"""

DOC_B = """Measurements from a recent serving round. On an Apple M5 Pro the first message of a process paid
the whole shared prefix once, about six thousand tokens of skills and tool declarations, which took
twenty seconds on a twenty-seven billion parameter model. Warming that prefix at load moved the first
message to under a second. A follow-up turn in the same conversation reused the previous turn whole and
took a few hundred milliseconds. A new conversation reused only the shared prefix. Under eight parallel
conversations three silent caps limited retention: a fixed budget of four retained end states, a family
that declared at most one retained native slot, and a page size of two hundred fifty six tokens for
families that reuse whole pages only. Each was replaced by a budget that follows the configured number
of concurrent sequences, by reclaimable slots, and by sixteen-token pages.
"""

TURNS = [
    lambda: ("Here is a design note about a model server:\n\n" + DOC_A +
             "\nSummarize the note's main points in about 150 words."),
    lambda: "Which parts of the note explain why a new turn might have to prefill again? Answer in about 120 words.",
    lambda: ("Here are some measurements that go with the note:\n\n" + DOC_B +
             "\nHow do these measurements relate to the design? About 120 words."),
    lambda: "Write a short checklist (five items) an operator could follow to keep reuse high.",
    lambda: "Give a one-paragraph conclusion.",
    lambda: "Now say the same in two sentences.",
]

TOOL = {"type": "function", "function": {
    "name": "get_weather",
    "description": "Return the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                   "required": ["city"]}}}

# --task tools: every turn needs the client's get_weather tool (OpenAI/Responses/Ollama --tools).
TOOL_TURNS = [
    lambda: ("Here is a design note about a model server:\n\n" + DOC_A +
             "\nFirst summarize the note in about 100 words. Then call get_weather for Paris and tell me if "
             "it is a good day to visit a data center there."),
    lambda: "Now check the weather in Tokyo with the tool and compare it with Paris in two sentences.",
    lambda: ("Here are some measurements:\n\n" + DOC_B + "\nSummarize them in 80 words, then call get_weather "
             "for Berlin."),
    lambda: "Which of the three cities had the best weather? One sentence.",
    lambda: "Give a one-paragraph conclusion about the note.",
]

# --task code: every turn needs the server's own shell (server started with --code-exec).
CODE_TURNS = [
    lambda: ("Here is a design note about a model server:\n\n" + DOC_A +
             "\nUse the shell to count the words of the sentence 'the longest common prefix decides the reuse' "
             "with wc -w, and tell me the number. Then summarize the note in 80 words."),
    lambda: "Use the shell to print the SHA-256 of the text 'radix' (printf radix | shasum -a 256). Report the hex.",
    lambda: ("Here are some measurements:\n\n" + DOC_B + "\nUse the shell to compute 6459 * 3 with "
             "python3 -c, then explain the measurements in 80 words."),
    lambda: "Without running anything, which of your earlier results was the largest number? One sentence.",
    lambda: "Give a one-paragraph conclusion.",
]

WEATHER = {"paris": "18 C, light rain", "tokyo": "24 C, sunny", "berlin": "12 C, overcast"}


COOKIE = None   # TensorAgent's loopback server authenticates by cookie (--cookie)


def http(url, body=None, timeout=900, stream=False):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"}
    if COOKIE:
        headers["Cookie"] = COOKIE
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET", headers=headers)
    r = urllib.request.urlopen(req, timeout=timeout)
    if stream:
        return r
    with r:
        return json.loads(r.read().decode("utf-8") or "{}")


def sse(resp):
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                return
            yield json.loads(payload)


def log_lines(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.readlines()


def completes_since(path, start):
    rows = []
    for line in log_lines(path)[start:]:
        if '"ChatCompleted"' not in line:
            continue
        try:
            d = json.loads(line)
            p = d["props"]
        except Exception:
            continue
        # Only the engine requests' lines: a spoken reply logs a second ChatCompleted line of its own.
        if not d.get("message", "").startswith("chat.complete"):
            continue
        rows.append({"prompt": p.get("PromptTokens"), "reused": p.get("KvReusedTokens"),
                     "ttftMs": p.get("TimeToFirstTokenMs"), "tokens": p.get("Tokens"),
                     "finish": p.get("FinishReason")})
    return rows


def upload_webui(url, path):
    import uuid
    boundary = uuid.uuid4().hex
    with open(path, "rb") as f:
        data = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{os.path.basename(path)}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if COOKIE:
        headers["Cookie"] = COOKIE
    req = urllib.request.Request(url + "/api/upload", data=body, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        reply = json.loads(r.read().decode("utf-8"))
    return reply["file"]


def with_images(args, history):
    """The history as the surface sends it: a user message's `_images` (local paths) become the surface's own
    image form."""
    import base64
    out = []
    for m in history:
        imgs = m.get("_images")
        m = {k: v for k, v in m.items() if not k.startswith("_")}
        if imgs:
            if args.surface == "webui":
                m["imagePaths"] = [args.uploaded.setdefault(p, upload_webui(args.url, p)) for p in imgs]
            elif args.surface in ("openai", "openai-stream"):
                parts = [{"type": "text", "text": m["content"]}]
                for p in imgs:
                    b64 = base64.b64encode(open(p, "rb").read()).decode()
                    parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}})
                m["content"] = parts
            elif args.surface == "ollama":
                m["images"] = [base64.b64encode(open(p, "rb").read()).decode() for p in imgs]
        out.append(m)
    return out


def run_turn(args, history, sampling):
    s = args.surface
    if s == "webui":
        payload = {"messages": history, "maxTokens": args.max_tokens, "think": args.think == "on",
                   "sessionId": args.session, "newChat": len(history) == 1 + (1 if args.system else 0)}
        if args.plain:
            payload.update({"skills": [], "skills_discovery": False, "multi_agent": False})
        payload.update({"temperature": 0, "topK": 1, "topP": 1, "minP": 0, "repeatPenalty": 1})
        if args.tensoragent:
            payload.pop("newChat", None)
            payload["knowledgeCollections"] = []
        if args.turn_index in args.speech_turns:
            payload["speech"] = True
        thinking, text, done = [], [], {}
        for d in sse(http(args.url + "/api/chat", payload, stream=True)):
            if d.get("error"):
                raise RuntimeError(d["error"])
            if d.get("thinking"):
                thinking.append(d["thinking"])
            if d.get("token"):
                text.append(d["token"])
            if d.get("replace") is not None:
                text = [d["replace"]]
            if d.get("done"):
                done = d
        msg = {"role": "assistant", "content": "".join(text)}
        if thinking:
            msg["thinking"] = "".join(thinking)
        # A spoken reply goes back as the page sends it: its transcript plus the audio id that splices it.
        for key in ("audioId", "audioUrl", "audioSeconds"):
            if done.get(key) is not None:
                msg[key] = done[key]
        return msg, done.get("promptTokens"), done.get("kvReusedTokens"), done.get("tokenCount")
    if s in ("openai", "openai-stream"):
        body = {"model": args.model, "messages": history, "max_tokens": args.max_tokens,
                "temperature": 0, "top_k": 1, "top_p": 1, "repeat_penalty": 1,
                "chat_template_kwargs": {"enable_thinking": args.think == "on"}}
        if args.think == "on":
            body["reasoning_effort"] = args.reasoning_effort
        if args.tools:
            body["tools"] = [TOOL]
        calls = []
        if s == "openai":
            r = http(args.url + "/v1/chat/completions", body)
            m = r["choices"][0]["message"]
            u = r.get("usage", {})
            content, reasoning = m.get("content") or "", m.get("reasoning_content") or m.get("reasoning") or ""
            calls = m.get("tool_calls") or []
        else:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
            content, reasoning, u = [], [], {}
            for d in sse(http(args.url + "/v1/chat/completions", body, stream=True)):
                if d.get("usage"):
                    u = d["usage"]
                for c in d.get("choices", []):
                    delta = c.get("delta", {})
                    for tc in delta.get("tool_calls") or []:
                        ix = tc.get("index", 0)
                        while len(calls) <= ix:
                            calls.append({"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            calls[ix]["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        calls[ix]["function"]["name"] += fn.get("name") or ""
                        calls[ix]["function"]["arguments"] += fn.get("arguments") or ""
                    if delta.get("content"):
                        content.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    elif delta.get("reasoning"):
                        reasoning.append(delta["reasoning"])
            content, reasoning = "".join(content), "".join(reasoning)
        msg = {"role": "assistant", "content": content}
        if calls:
            msg["tool_calls"] = calls
        if args.echo_reasoning and reasoning:
            msg["reasoning_content"] = reasoning
        details = u.get("prompt_tokens_details") or {}
        return msg, u.get("prompt_tokens"), details.get("cached_tokens"), u.get("completion_tokens")
    if s == "responses":
        items = []
        for m in history:
            if m["role"] == "assistant":
                if args.echo_reasoning and m.get("_reasoning"):
                    items.append({"type": "reasoning", "summary": [],
                                  "content": [{"type": "reasoning_text", "text": m["_reasoning"]}]})
                items.append({"type": "message", "role": "assistant",
                              "content": [{"type": "output_text", "text": m["content"]}]})
            else:
                items.append({"type": "message", "role": m["role"],
                              "content": [{"type": "input_text", "text": m["content"]}]})
        body = {"model": args.model, "input": items, "max_output_tokens": args.max_tokens,
                "temperature": 0, "top_p": 1, "store": False}
        if args.think == "on":
            body["reasoning"] = {"effort": args.reasoning_effort}
        if args.tools:
            body["tools"] = [dict(type="function", **TOOL["function"])]
        r = http(args.url + "/v1/responses", body)
        text, reasoning = [], []
        for item in r.get("output", []):
            if item.get("type") == "message":
                text += [c.get("text", "") for c in item.get("content", []) if c.get("type") == "output_text"]
            elif item.get("type") == "reasoning":
                reasoning += [c.get("text", "") for c in (item.get("content") or []) + (item.get("summary") or [])]
        u = r.get("usage", {})
        msg = {"role": "assistant", "content": "".join(text), "_reasoning": "".join(reasoning)}
        return msg, u.get("input_tokens"), (u.get("input_tokens_details") or {}).get("cached_tokens"), \
            u.get("output_tokens")
    if s == "ollama":
        body = {"model": args.model, "messages": history, "stream": False, "think": args.think == "on",
                "options": {"num_predict": args.max_tokens, "temperature": 0, "top_k": 1, "top_p": 1,
                            "repeat_penalty": 1}}
        if args.tools:
            body["tools"] = [TOOL]
        r = http(args.url + "/api/chat/ollama", body)
        if os.environ.get("MTR_DUMP"):
            print(json.dumps(r, ensure_ascii=False)[:3000], file=sys.stderr)
        m = r.get("message", {})
        msg = {"role": "assistant", "content": m.get("content", "")}
        if m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        if args.echo_reasoning and m.get("thinking"):
            msg["thinking"] = m["thinking"]
        return msg, r.get("prompt_eval_count"), r.get("prompt_cache_hit_tokens"), r.get("eval_count")
    raise SystemExit(f"unknown surface {s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:5120")
    ap.add_argument("--surface", required=True, choices=["webui", "openai", "openai-stream", "responses", "ollama"])
    ap.add_argument("--think", choices=["on", "off"], default="off")
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--echo-reasoning", action="store_true")
    ap.add_argument("--tools", action="store_true")
    ap.add_argument("--task", choices=["chat", "tools", "code"], default="chat",
                    help="chat: plain turns; tools: turns that need the client's get_weather tool (the client "
                         "answers each call and sends the round back); code: turns that need the server's shell "
                         "(start the server with --code-exec)")
    ap.add_argument("--plain", action="store_true", help="webui: no skills, no discovery, no multi-agent")
    ap.add_argument("--system", help="a system message (text) the conversation starts with")
    ap.add_argument("--log", help="the server's JSONL log, for per-round chat.complete lines")
    ap.add_argument("--nonce", default=None,
                    help="text the first user message starts with, so two runs on one server are different "
                         "conversations (default: the --out file name). Identical greedy conversations in "
                         "two scopes make the transcript owner ambiguous, and such a turn rightly inherits "
                         "neither scope (ConversationTranscriptStore)")
    ap.add_argument("--image", help="an image attached to the user turns named by --image-turns (webui, openai, ollama)")
    ap.add_argument("--image-turns", default="1", help="comma-separated 1-based turns that carry --image (default 1)")
    ap.add_argument("--attach-file", help="a text file whose k-th --chunk-chars characters are prepended to turn k "
                                          "(long conversations: a few thousand tokens per turn)")
    ap.add_argument("--chunk-chars", type=int, default=8000)
    ap.add_argument("--speech-turns", default="", help="webui: comma-separated 1-based turns that ask for a spoken "
                                                         "reply (speech: true; LFM3-Omni)")
    ap.add_argument("--cookie", help="Cookie header (TensorAgent's loopback server: the connection.json cookie)")
    ap.add_argument("--tensoragent", action="store_true",
                    help="webui against TensorAgent's host: sessions bound to a saved conversation, the app page's body")
    ap.add_argument("--reopen-before", default="", help="--tensoragent: comma-separated 1-based turns before which the "
                                                         "chat is reopened (a new session bound to the conversation)")
    ap.add_argument("--pause", type=float, default=0.3, help="seconds to wait between turns")
    ap.add_argument("--model", help="the hosted model name (default: the first of /api/tags)")
    ap.add_argument("--label", default="")
    ap.add_argument("--replay", help="a finished run's report: re-send its last turn once and compare the reply (see above)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    global COOKIE
    COOKIE = args.cookie
    if args.model is None and args.surface != "webui":
        tags = http(args.url + "/api/tags")
        args.model = tags["models"][0]["name"]
    args.session = None
    args.conversation = None
    args.uploaded = {}
    args.image_turns = {int(x) - 1 for x in args.image_turns.split(",") if x.strip()}
    args.reopen_before = {int(x) - 1 for x in args.reopen_before.split(",") if x.strip()}
    args.speech_turns = {int(x) - 1 for x in args.speech_turns.split(",") if x.strip()}
    if args.surface == "webui" and args.tensoragent:
        # TensorAgent's page: a session bound to a saved conversation (WebUiRoutes /api/sessions?conversation=).
        created = http(args.url + "/api/sessions?conversation=new", {})
        args.session, args.conversation = created.get("sessionId"), created.get("conversationId")
    elif args.surface == "webui":
        args.session = http(args.url + "/api/sessions", {}).get("sessionId")
    if args.replay:
        return replay(args)
    history = [{"role": "system", "content": args.system}] if args.system else []
    turns = {"chat": TURNS, "tools": TOOL_TURNS, "code": CODE_TURNS}[args.task]
    if args.task == "tools":
        args.tools = True
    rows, prev_carry = [], None
    attach = open(args.attach_file, encoding="utf-8").read() if args.attach_file else ""
    nonce = args.nonce if args.nonce is not None else os.path.splitext(os.path.basename(args.out))[0]
    for k in range(min(args.turns, len(turns))):
        text = turns[k]()
        if args.attach_file:
            chunk = attach[k * args.chunk_chars:(k + 1) * args.chunk_chars]
            text = f"Part {k + 1} of a longer document follows.\n\n{chunk}\n\n" + text
        if k == 0 and nonce:
            text = f"(conversation {nonce})\n" + text
        args.turn_index = k
        if args.tensoragent and k in args.reopen_before:
            # The app reopens the chat (another chat picked and this one picked again, or the page reloaded): a new
            # session bound to the same conversation, which sends the saved history from then on.
            reopened = http(args.url + "/api/sessions?conversation=" + urllib.parse.quote(args.conversation), {})
            args.session = reopened.get("sessionId")
            saved = reopened.get("messages")
            if isinstance(saved, list) and saved:
                history[:] = saved
            print(f"reopened conversation {args.conversation}: session {args.session}, {len(history)} saved messages",
                  flush=True)
        user = {"role": "user", "content": text}
        if args.image and k in args.image_turns:
            user["_images"] = [args.image]
        history.append(user)
        for client_round in range(6):
            start = len(log_lines(args.log))
            t0 = time.time()
            sent = with_images(args, history) if args.surface != "responses" else history
            msg, prompt, reused, completion = run_turn(args, sent, None)
            elapsed = time.time() - t0
            time.sleep(args.pause)
            rounds = completes_since(args.log, start)
            # The server's own tool loop runs several engine requests per HTTP request: the first one's
            # reuse is the turn's, the last one's prompt + reply is what the next turn continues.
            if rounds:
                prompt, reused = rounds[0]["prompt"], rounds[0]["reused"]
                carry_next = (rounds[-1]["prompt"] or 0) + (rounds[-1]["tokens"] or 0)
                inner = [(r["prompt"], r["reused"]) for r in rounds[1:]]
            else:
                carry_next = (prompt or 0) + (completion or 0)
                inner = []
            history.append(msg)
            prefill = (prompt - reused) if prompt is not None and reused is not None else None
            new = (prompt - prev_carry) if prompt is not None and prev_carry is not None else None
            waste = (prefill - max(new, 0)) if prefill is not None and new is not None else None
            label = f"{k + 1}" if client_round == 0 else f"{k + 1}.{client_round}"
            row = {"turn": label, "prompt": prompt, "reused": reused, "prefill": prefill, "carry": prev_carry,
                   "new": new, "waste": waste, "completion": completion, "elapsedSec": round(elapsed, 2),
                   "ttftMs": rounds[0]["ttftMs"] if rounds else None, "rounds": rounds,
                   "toolCalls": len(msg.get("tool_calls") or []),
                   "answer": msg.get("content", ""), "reasoning": msg.get("thinking") or msg.get("reasoning_content")
                   or msg.get("_reasoning") or ""}
            rows.append(row)
            print(f"turn {label}: prompt={prompt} reused={reused} prefill={prefill} carry={prev_carry} new={new} "
                  f"waste={waste} completion={completion} ttftMs={row['ttftMs']} serverRounds={len(rounds)} "
                  f"inner={inner} clientCalls={row['toolCalls']}", flush=True)
            prev_carry = carry_next
            calls = msg.get("tool_calls") or []
            if not calls:
                break
            for i, c in enumerate(calls):
                fn = c.get("function") or {}
                a = fn.get("arguments")
                try:
                    a = json.loads(a) if isinstance(a, str) else (a or {})
                except ValueError:
                    a = {}
                city = str(a.get("city", "")).strip().lower()
                tool_msg = {"role": "tool", "content": json.dumps({"city": city, "weather": WEATHER.get(city, "unknown")})}
                if c.get("id"):
                    tool_msg["tool_call_id"] = c["id"]
                if args.surface == "ollama":
                    tool_msg["tool_name"] = fn.get("name")
                history.append(tool_msg)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        config = {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(args).items() if k not in ("system", "uploaded")}
        json.dump({"configuration": config, "turns": rows, "history": history}, f,
                  ensure_ascii=False, indent=1)
    return 0


def replay(args):
    with open(args.replay, encoding="utf-8") as f:
        report = json.load(f)
    history = report.get("history")
    if not history:
        print(f"{args.replay} keeps no history (written before --replay existed)", file=sys.stderr)
        return 2
    last_user = max(i for i, m in enumerate(history) if m.get("role") == "user")
    sent = history[:last_user + 1]
    want = report["turns"][-1]
    cfg = report["configuration"]
    for key in ("think", "max_tokens", "plain", "reasoning_effort", "tools"):
        if key in cfg:
            setattr(args, key, cfg[key])
    args.turn_index = sum(1 for m in sent if m.get("role") == "user") - 1
    if args.surface == "webui":
        args.session = http(args.url + "/api/sessions", {}).get("sessionId")
    t0 = time.time()
    msg, prompt, reused, completion = run_turn(args, sent, None)
    got, expected = msg.get("content", ""), want.get("answer", "")
    same = got == expected
    first = next((i for i in range(min(len(got), len(expected))) if got[i] != expected[i]), min(len(got), len(expected)))
    print(f"replay of turn {want['turn']} ({len(sent)} messages): prompt={prompt} reused={reused} completion={completion} "
          f"elapsedSec={time.time() - t0:.1f}; reply {'IDENTICAL to' if same else f'differs at character {first} from'} "
          f"the cached run's ({len(got)} vs {len(expected)} characters)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"replayOf": os.path.abspath(args.replay), "turn": want["turn"], "prompt": prompt, "reused": reused,
                   "completion": completion, "identical": same, "firstDifference": None if same else first,
                   "answer": got, "cachedAnswer": expected}, f, ensure_ascii=False, indent=1)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
