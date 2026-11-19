#!/usr/bin/env python3
"""Regenerate the LFM3 chat-protocol fixture in InferenceWeb.Tests/Fixtures/Lfm3/chat_reference.json.

The expected prompts come from llama.cpp's own Jinja engine: every case is POSTed to a running
llama-server's /apply-template endpoint, the same code path /v1/chat/completions uses to build its
prompt. The C# test (Lfm3ChatProtocolTests) parses the SAME OpenAI request body with TensorSharp's
OpenAI message and tool parsers, renders it through the GGUF template with ChatTemplate, and asserts
byte equality with "<|startoftext|>" + the TensorSharp render (TensorSharp renders bos_token as ""
and lets the tokenizer add the BOS; llama.cpp renders the token text).

Usage (stdlib only):

  LLAMA_MEDIA_MARKER='<__media__>' llama-server -m LFM3-1.7B-Q4_K_M.gguf --port 18431 -c 4096 &
  LLAMA_MEDIA_MARKER='<__media__>' llama-server -m LFM3-Omni-2B-Q4_K_M.gguf \
      --mmproj mmproj-LFM3-Omni-2B-Q8_0.gguf --port 18432 -c 4096 &
  python3 eng/lfm3/chat-template-fixtures.py --text http://127.0.0.1:18431 \
      --omni http://127.0.0.1:18432 --text-gguf LFM3-1.7B-Q4_K_M.gguf --omni-gguf LFM3-Omni-2B-Q4_K_M.gguf \
      --llama-revision $(git -C llama.cpp rev-parse HEAD) \
      --out InferenceWeb.Tests/Fixtures/Lfm3/chat_reference.json

Every case pins chat_template_kwargs.enable_thinking (TensorSharp's request default is thinking
off, the templates' defaults differ: 1.7B on, Omni off) and preserve_reasoning=false: llama-server
turns the template's preserve_thinking ON by default (--reasoning-preserve), while HF, vLLM, SGLang
and TensorSharp render the template default (off), which strips earlier turns' reasoning.

`expected_tokens` is llama.cpp's tokenization of `expected` (its /tokenize, special tokens parsed):
the gated Lfm3ChatProtocolTests case compares TensorSharp's tokenizer with it.

Media parts: llama-server replaces each image_url / input_audio part with its media marker and
mtmd later wraps the encoder rows in <|vision_start|> ... <|vision_end|>. The fixture stores the raw
llama prompt and an `expected` in which every marker is replaced by the template's own placeholder
for that part (<|vision_start|><|image|><|vision_end|> or <|vision_start|><|audio|><|vision_end|>),
which is TensorSharp's placeholder contract for LFM3 (see docs/models/lfm3.md).
"""
import argparse, base64, json, os, struct, sys, urllib.request, zlib

MEDIA_MARKER = "<__media__>"
IMAGE_SPAN = "<|vision_start|><|image|><|vision_end|>"
AUDIO_SPAN = "<|vision_start|><|audio|><|vision_end|>"


def tiny_png():
    raw = b"\x00\xff\x00\x00"  # one red pixel row (filter byte + RGB)
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def tiny_wav():
    samples = b"\x00\x00" * 1600  # 0.1 s of 16 kHz mono silence
    header = b"RIFF" + struct.pack("<I", 36 + len(samples)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16) + b"data" + struct.pack("<I", len(samples))
    return base64.b64encode(header + samples).decode()


IMAGE_PART = {"type": "image_url", "image_url": {"url": tiny_png()}}
AUDIO_PART = {"type": "input_audio", "input_audio": {"data": tiny_wav(), "format": "wav"}}

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "description": "Temperature unit", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search_docs", "description": "Search the documentation — returns \"quotes\" & <tags>.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "What to look for"},
        "limit": {"type": "integer", "description": "Maximum hits"},
        "exact": {"type": "boolean", "description": "Exact match only"}},
        "required": ["query"]}}}
PING = {"type": "function", "function": {
    "name": "ping", "description": "Check the service.",
    "parameters": {"type": "object", "properties": {}, "required": []}}}


def call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def case(name, messages, think, tools=None, add_generation_prompt=True):
    body = {"messages": messages, "add_generation_prompt": add_generation_prompt,
            "chat_template_kwargs": {"enable_thinking": think, "preserve_reasoning": False}}
    if tools:
        body["tools"] = tools
    return {"name": name, "request": body}


SYS = {"role": "system", "content": "You are a concise assistant."}
U1 = {"role": "user", "content": "What is 2+2?"}


def text_cases():
    cases = []
    for think in (True, False):
        t = "on" if think else "off"
        cases.append(case(f"user_only_{t}", [{"role": "user", "content": "Hello"}], think))
        cases.append(case(f"system_user_{t}", [SYS, U1], think))
        cases.append(case(f"multi_turn_{t}", [
            SYS, U1,
            {"role": "assistant", "content": "4", "reasoning_content": "2 plus 2 is 4."},
            {"role": "user", "content": "And 3+3?"}], think))
        cases.append(case(f"tools_declared_{t}", [SYS, {"role": "user", "content": "Weather in Paris?"}],
                          think, [WEATHER, SEARCH]))
        cases.append(case(f"tool_round_{t}", [
            SYS, {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [call("call_1", "get_weather", {"city": "Paris", "unit": "celsius"})]},
            {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 20, \"sky\": \"clear\"}"}],
            think, [WEATHER]))
        cases.append(case(f"parallel_calls_{t}", [
            {"role": "user", "content": "Compare Paris and Oslo."},
            {"role": "assistant", "content": None, "tool_calls": [
                call("call_a", "get_weather", {"city": "Paris"}), call("call_b", "get_weather", {"city": "Oslo"})]},
            {"role": "tool", "tool_call_id": "call_a", "content": "Paris: 20C"},
            {"role": "tool", "tool_call_id": "call_b", "content": "Oslo: 5C"}],
            think, [WEATHER]))
    cases.append(case("tools_no_system_on", [{"role": "user", "content": "Ping it."}], True, [PING]))
    cases.append(case("tool_round_with_reasoning_on", [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "I need the weather tool.",
         "tool_calls": [call("call_1", "get_weather", {"city": "Paris"})]},
        {"role": "tool", "tool_call_id": "call_1", "content": "20C"}], True, [WEATHER]))
    cases.append(case("past_tool_round_then_user_on", [
        SYS, {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "Use the tool.",
         "tool_calls": [call("call_1", "get_weather", {"city": "Paris"})]},
        {"role": "tool", "tool_call_id": "call_1", "content": "20C"},
        {"role": "assistant", "content": "It is 20C in Paris.", "reasoning_content": "Report it."},
        {"role": "user", "content": "Thanks! And Oslo?"}], True, [WEATHER]))
    cases.append(case("typed_arguments_on", [
        {"role": "user", "content": "Search."},
        {"role": "assistant", "content": "", "tool_calls": [call("call_1", "search_docs", {
            "query": "line one\nline two", "limit": 5, "exact": True, "score": 0.5,
            "tags": ["a", "b"], "filter": {"field": "x", "value": None}, "note": None})]},
        {"role": "tool", "tool_call_id": "call_1", "content": ""}], True, [SEARCH]))
    cases.append(case("think_tags_in_past_content_on", [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "<think>greeting</think>\n\nHello there!"},
        {"role": "user", "content": "Bye"}], True))
    cases.append(case("unicode_on", [{"role": "user", "content": "翻译：你好，世界 🌍 — «ça va?»"}], True))
    # No add_generation_prompt=false case: llama-server treats a trailing assistant message as an
    # assistant prefill to continue (it renders the open turn with its own <think> framing), which
    # is not the template's text; TensorSharp renders such histories only as KV-cache prefixes.
    return cases


def omni_cases():
    cases = [c for c in text_cases() if c["name"] in (
        "user_only_on", "user_only_off", "system_user_off", "multi_turn_off", "tools_declared_off",
        "tool_round_off", "parallel_calls_off", "past_tool_round_then_user_on", "typed_arguments_on")]
    cases.append(case("image_off", [{"role": "user", "content": [IMAGE_PART, {"type": "text", "text": "Describe it."}]}], False))
    cases.append(case("two_images_off", [{"role": "user", "content": [IMAGE_PART, IMAGE_PART, {"type": "text", "text": "Compare."}]}], False))
    cases.append(case("audio_off", [{"role": "user", "content": [AUDIO_PART, {"type": "text", "text": "Transcribe."}]}], False))
    cases.append(case("image_and_audio_off", [{"role": "user", "content": [
        IMAGE_PART, AUDIO_PART, {"type": "text", "text": "Answer in one sentence."}]}], False))
    cases.append(case("image_history_off", [
        SYS, {"role": "user", "content": [IMAGE_PART, {"type": "text", "text": "What color?"}]},
        {"role": "assistant", "content": "Red."},
        {"role": "user", "content": "Are you sure?"}], False))
    return cases


def media_parts(request):
    kinds = []
    for message in request["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    kinds.append(IMAGE_SPAN)
                elif part.get("type") == "input_audio":
                    kinds.append(AUDIO_SPAN)
    return kinds


def render(base, request):
    data = json.dumps(request).encode()
    req = urllib.request.Request(base.rstrip("/") + "/apply-template", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["prompt"]


def tokenize(base, text):
    """llama.cpp's token ids for a rendered prompt: special tokens parsed, BOS taken from the text."""
    data = json.dumps({"content": text, "add_special": False, "parse_special": True}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/tokenize", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["tokens"]


def run(base, cases, template):
    out = []
    for c in cases:
        prompt = render(base, c["request"])
        expected = prompt
        for span in media_parts(c["request"]):
            if MEDIA_MARKER not in expected:
                sys.exit(f"{template}/{c['name']}: llama prompt has fewer media markers than media parts "
                         f"(start llama-server with LLAMA_MEDIA_MARKER='{MEDIA_MARKER}')")
            expected = expected.replace(MEDIA_MARKER, span, 1)
        if MEDIA_MARKER in expected:
            sys.exit(f"{template}/{c['name']}: unmatched media marker")
        out.append({"template": template, "name": c["name"], "request": c["request"],
                    "llama_prompt": prompt, "expected": expected,
                    "expected_tokens": tokenize(base, expected)})
    return out


def gguf_string(path, wanted):
    """Read one string key from a GGUF header (v2/v3), without numpy or llama.cpp's gguf-py."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as f:
        def u32(): return struct.unpack("<I", f.read(4))[0]
        def u64(): return struct.unpack("<Q", f.read(8))[0]
        def text(): return f.read(u64()).decode("utf-8")
        def skip(kind):
            if kind == 8:
                f.seek(u64(), 1)
            elif kind == 9:
                inner, count = u32(), u64()
                if inner == 8:
                    for _ in range(count): f.seek(u64(), 1)
                else:
                    f.seek(sizes[inner] * count, 1)
            else:
                f.seek(sizes[kind], 1)
        if f.read(4) != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        u32(); u64()  # version, tensor count
        for _ in range(u64()):
            key, kind = text(), u32()
            if key == wanted and kind == 8:
                return text()
            skip(kind)
    sys.exit(f"{path}: no string key {wanted}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True, help="llama-server URL serving LFM3-1.7B (its GGUF template)")
    ap.add_argument("--omni", required=True, help="llama-server URL serving LFM3-Omni-2B with its mmproj")
    ap.add_argument("--text-gguf", required=True, help="the LFM3-1.7B GGUF the text server loaded")
    ap.add_argument("--omni-gguf", required=True, help="the LFM3-Omni-2B GGUF the omni server loaded")
    ap.add_argument("--llama-revision", required=True)
    ap.add_argument("--out", required=True, help="chat_reference.json; the two templates are written beside it")
    args = ap.parse_args()
    out_dir = os.path.dirname(os.path.abspath(args.out))
    for gguf, name in ((args.text_gguf, "text_chat_template.jinja"), (args.omni_gguf, "omni_chat_template.jinja")):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="") as f:
            f.write(gguf_string(gguf, "tokenizer.chat_template"))
    fixture = {
        "generator": "eng/lfm3/chat-template-fixtures.py",
        "source": "llama-server /apply-template",
        "llama_cpp_revision": args.llama_revision,
        "models": {"text": "LiquidAI/LFM3-1.7B-GGUF@cd0c74c575b5c416f3011b5b4c298077f26ca8c2 (tokenizer.chat_template)",
                   "omni": "LiquidAI/LFM3-Omni-2B-GGUF@d508b88f08a5b4286f2d2805a6a8d69a0ff44c72 (tokenizer.chat_template)"},
        "cases": run(args.text, text_cases(), "text") + run(args.omni, omni_cases(), "omni"),
    }
    # One line per token list, so the fixture stays reviewable.
    token_lists = []
    for case in fixture["cases"]:
        token_lists.append(case["expected_tokens"])
        case["expected_tokens"] = f"@@TOKENS{len(token_lists) - 1}@@"
    text = json.dumps(fixture, ensure_ascii=False, indent=1)
    for i, tokens in enumerate(token_lists):
        text = text.replace(f'"@@TOKENS{i}@@"', json.dumps(tokens, separators=(",", ":")))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"wrote {len(fixture['cases'])} cases to {args.out}")


if __name__ == "__main__":
    main()
