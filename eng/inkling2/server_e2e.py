"""End-to-end checks of a TensorSharp server hosting Inkling 2 over the OpenAI chat API, standard library only.

    python -I eng/inkling2/server_e2e.py http://127.0.0.1:4100 MODEL_ID

Checks that the reasoning and the answer come back apart with no message framing left in either, that a stream
carries the same, that a tool call comes back as an OpenAI tool call and its result is answered, and that a
follow-up turn reuses the cached prefix of the conversation. Prints one line a check and exits non-zero on failure.
"""

from __future__ import annotations

import http.client
import json
import sys
import time
import urllib.parse

FRAMING = ("<|message_", "<|content_", "<|end_message|>")
WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def post(base: str, body: dict, stream: bool = False):
    url = urllib.parse.urlsplit(base)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=900)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:400]!r}")
    if not stream:
        return json.loads(response.read())
    events = []
    for raw in response:
        line = raw.decode("utf-8").strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
    return events


def clean(text: str | None) -> bool:
    return not any(tag in (text or "") for tag in FRAMING)


def main() -> int:
    base, model = sys.argv[1], sys.argv[2]
    failures = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")

    ask = [{"role": "user", "content": "What is 17 * 23? Reply with the number only."}]
    t0 = time.time()
    reply = post(base, {"model": model, "messages": ask, "max_tokens": 512, "temperature": 0, "think": True})
    message = reply["choices"][0]["message"]
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    check("answer and reasoning apart", message.get("content", "").strip() == "391" and len(reasoning) > 0
          and clean(message.get("content")) and clean(reasoning),
          f"content={message.get('content')!r} reasoning={len(reasoning)} chars, {time.time() - t0:.1f} s, usage={reply.get('usage')}")

    quiet = post(base, {"model": model, "messages": ask, "max_tokens": 512, "temperature": 0})
    m = quiet["choices"][0]["message"]
    check("thinking off shows the answer only", m.get("content", "").strip() == "391" and clean(m.get("content")),
          f"content={m.get('content')!r} reasoning={len(m.get('reasoning_content') or '')} chars")

    events = post(base, {"model": model, "messages": ask, "max_tokens": 512, "temperature": 0, "think": True, "stream": True}, stream=True)
    content = "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e.get("choices"))
    thought = "".join((e["choices"][0]["delta"].get("reasoning_content") or e["choices"][0]["delta"].get("reasoning") or "")
                      for e in events if e.get("choices"))
    check("stream", content.strip() == "391" and len(thought) > 0 and clean(content) and clean(thought),
          f"content={content!r} reasoning={len(thought)} chars in {len(events)} events")

    tools_turn = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
    first = post(base, {"model": model, "messages": tools_turn, "tools": WEATHER, "max_tokens": 1024, "temperature": 0})
    choice = first["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    check("tool call", choice.get("finish_reason") == "tool_calls" and len(calls) == 1
          and calls[0]["function"]["name"] == "get_weather" and "paris" in str(args.get("city", "")).lower()
          and clean(choice["message"].get("content")),
          f"finish={choice.get('finish_reason')} calls={[(c['function']['name'], c['function']['arguments']) for c in calls]}")
    if calls:
        follow = tools_turn + [choice["message"], {"role": "tool", "tool_call_id": calls[0]["id"], "content": "18 C, light rain"}]
        second = post(base, {"model": model, "messages": follow, "tools": WEATHER, "max_tokens": 1024, "temperature": 0})
        answer = second["choices"][0]["message"].get("content") or ""
        cached = (second.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        check("tool result answered", "18" in answer and clean(answer),
              f"answer={answer[:120]!r} prompt={second['usage']['prompt_tokens']} cached={cached}")

    turn1 = [{"role": "user", "content": "Name one planet. One word."}]
    r1 = post(base, {"model": model, "messages": turn1, "max_tokens": 512, "temperature": 0})
    turn2 = turn1 + [r1["choices"][0]["message"], {"role": "user", "content": "And one more, different. One word."}]
    r2 = post(base, {"model": model, "messages": turn2, "max_tokens": 512, "temperature": 0})
    u = r2.get("usage", {})
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    check("follow-up reuses the conversation", cached >= r1["usage"]["prompt_tokens"],
          f"turn 1 prompt {r1['usage']['prompt_tokens']}, turn 2 prompt {u.get('prompt_tokens')} with {cached} cached; "
          f"answers {r1['choices'][0]['message'].get('content')!r}, {r2['choices'][0]['message'].get('content')!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
