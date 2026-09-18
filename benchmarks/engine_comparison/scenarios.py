#!/usr/bin/env python3
"""
Scenario -> request payload builders.

Each builder returns a dict describing one `/v1/chat/completions` request:
  { "messages", "tools", "response_format", "checker", "followups" }
`checker(metrics) -> Optional[bool]` is used by correctness-bearing scenarios
(function_call, agentic, code_edit) to record whether the model did the expected
thing.

`followups` turns a cell into a client-driven multi-turn workflow. It is a list
of callables, one per FOLLOW-UP request; each is handed the metrics of the turn
that just finished plus the messages that produced it, and returns the request
overrides for the next turn ({"messages": [...], optionally "tools",
"response_format", "extra_body"}). Raising from a follow-up ends the
conversation there and the reason is recorded on the cell — a scenario must
never fabricate the next turn out of a response that did not contain what that
turn needs, because the cell would then silently be measuring something else.

Everything is driven from the client on purpose. TensorSharp's own code-exec
surface (`--code-exec`: shell / read_file / write_file / apply_patch) is answered
*inside the server* and never handed back to the API
client, it is off by default, and its request workspace is destroyed when the
response ends — so it can neither be observed round-trip-by-round-trip nor
carry a file from one request to the next, and llama.cpp/vLLM have no
equivalent at all. A client-driven loop is therefore the only shape that is
both measurable and identical on every engine.

Image is sent in the portable OpenAI `image_url` form to every engine. Audio
and video differ per engine (TensorSharp accepts a message-level base64 array /
a sampled-frame image sequence; llama.cpp uses the OpenAI `input_audio` part),
so those builders take the engine id.
"""
from __future__ import annotations

import base64
import functools
import json
from pathlib import Path
from typing import Optional

import config

ASSETS = config.ASSETS_DIR


def _read_asset(name: str, fallback: str) -> str:
    p = ASSETS / name
    if p.exists():
        return p.read_text(encoding="utf-8")
    return fallback


def _b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{_b64_file(path)}"


# ---------------------------------------------------------------------------
# Shared prompt-context preamble
# ---------------------------------------------------------------------------
# The interactive text scenarios (text_short / text_long / multi_turn /
# function_call / json_mode) are timed by TTFT. At tiny prompt lengths TTFT is
# dominated by fixed per-request overhead (HTTP, scheduling, cold-graph launch,
# first-token sampling) rather than prefill compute, so the numbers are noisy
# and engine-to-engine differences are unreliable. To measure on a realistic,
# stable footing every one of these scenarios is prefixed with the same
# ~2k-token reference document; the scenario's actual task then follows.
_PREFILL_CHARS_PER_TOKEN = 4.6   # ~English prose under these models' tokenizers
_CONTEXT_TOKENS = 2048           # ~2k-token context preamble for text scenarios


@functools.lru_cache(maxsize=8)
def _sliced_corpus(target_tokens: int) -> str:
    """Tile `prefill_corpus.txt` to ~`target_tokens` tokens (char-budget approx).

    Length is controlled by slicing to a target character budget; the *reported*
    `prompt_tokens` (from each engine's own tokenizer) is what the throughput math
    actually uses, so this approximation only needs to land in the right ballpark.
    """
    corpus = _read_asset("prefill_corpus.txt",
                         "The quick brown fox jumps over the lazy dog. " * 400)
    target_chars = int(target_tokens * _PREFILL_CHARS_PER_TOKEN)

    # Tile the corpus (with a numbered separator, so adjacent blocks are not
    # byte-identical) until it is long enough, then truncate to the budget.
    body = corpus
    section = 2
    while len(body) < target_chars:
        body += f"\n\n--- continued (part {section}) ---\n\n{corpus}"
        section += 1
    return body[:target_chars].rstrip()


def _context_preamble(tag: str) -> str:
    """A ~2k-token reference-document preamble to prepend to a text scenario.

    `tag` is embedded in a unique header at position 0 so that scenarios sharing
    this identical body cannot hit a server's prompt/prefix cache off one another
    (which would report a near-zero TTFT and a wildly inflated prefill_tps). The
    differing tag busts any shared-prefix match within the first few tokens.
    """
    doc = _sliced_corpus(_CONTEXT_TOKENS)
    header = (f"[context:{tag}] Reference document, provided for context; keep it "
              f"in mind when answering the request that follows.\n\n")
    return f"{header}{doc}\n\n---\n\n"


@functools.lru_cache(maxsize=4)
def _video_frames_b64(path_str: str, n: int = 4) -> tuple:
    """Sample `n` evenly-spaced frames from a video, return JPEG base64 strings."""
    import cv2  # available in this environment
    cap = cv2.VideoCapture(path_str)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    if total <= 0:
        cap.release()
        return tuple()
    idxs = [int(total * (i + 1) / (n + 1)) for i in range(n)]
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            frames.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    cap.release()
    return tuple(frames)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _text_short(engine, model):
    return {"messages": [
        {"role": "user",
         "content": _context_preamble("text_short") +
                    "Explain what a transformer neural network is in three concise sentences."}]}


def _text_long(engine, model):
    doc = _read_asset("long_text.txt", "Lorem ipsum dolor sit amet. " * 200)
    return {"messages": [
        {"role": "user",
         "content": _context_preamble("text_long") +
                    doc + "\n\nSummarize the passage immediately above in two sentences."}]}


# ---------------------------------------------------------------------------
# Prefill (prompt-processing) dataset
# ---------------------------------------------------------------------------
# Dedicated long-prompt scenarios used to measure prefill throughput accurately.
# The interactive text scenarios carry only a ~2k-token context preamble (see
# `_context_preamble`), enough to keep TTFT out of the fixed-overhead floor but
# still short enough that per-token prefill cost does not fully dominate. The
# prefill scenarios below instead drive the prompt to several thousand /
# tens-of-thousands of tokens at controlled lengths so the per-token prefill
# cost separates cleanly from that fixed overhead.
#
# Length is controlled by slicing `prefill_corpus.txt` to a target character
# budget (see `_sliced_corpus`); the *reported* `prompt_tokens` (from each
# engine's own tokenizer) is what `prefill_tps = prompt_tokens / ttft` actually
# uses, so this approximation only needs to land each scenario in the right
# ballpark, not hit an exact count.


def _parse_prefill_target(scenario_id: str) -> int:
    """`prefill_4k` / `prefill_4096` / `prefill_512` -> target token count."""
    suffix = scenario_id.split("prefill_", 1)[-1].strip().lower()
    if suffix.endswith("k"):
        return int(round(float(suffix[:-1]) * 1024))
    return int(suffix)


def _prefill(target_tokens: int, engine, model):
    body = _sliced_corpus(target_tokens)

    # A unique-per-length header at position 0 so the longer prompts cannot hit
    # the server's prompt/prefix cache off the shorter ones (which would report a
    # near-zero TTFT and a wildly inflated prefill_tps). The differing token
    # count busts any shared-prefix match within the first few tokens.
    header = (f"[prefill-benchmark target={target_tokens} tokens] The following is a "
              f"long technical document, provided so the system can be timed while it "
              f"processes the prompt.\n\n")
    instruction = ("\n\nIn one sentence, state the single most important theme of the "
                   "document above.")
    return {"messages": [
        {"role": "user", "content": header + body + instruction}]}


def _multi_turn(engine, model):
    # Prepend the ~2k-token context to the first user turn so the final (timed)
    # turn re-processes a realistic-length conversation prefix.
    return {"messages": [
        {"role": "user", "content": _context_preamble("multi_turn") +
                                    "I'm planning a trip to Japan. My budget is $3000."},
        {"role": "assistant", "content": "Great! Japan is a wonderful destination. With a $3000 budget, "
                                         "you can have a comfortable week-long trip. What time of year are you thinking?"},
        {"role": "user", "content": "Cherry blossom season. Which two cities should I prioritize and why?"},
        {"role": "assistant", "content": "For cherry blossom season, Kyoto and Tokyo are the top picks. "
                                         "Kyoto offers classic temple-and-blossom scenery; Tokyo adds variety and nightlife."},
        {"role": "user", "content": "Given my budget, roughly how should I split spending between those two cities?"}]}


def _function_call(engine, model):
    tools = json.loads(_read_asset("tools/weather.json", _DEFAULT_WEATHER_TOOL))

    def checker(metrics):
        # Accept either a structured tool-call name (llama.cpp/vLLM streaming) or
        # finish_reason=tool_calls (TensorSharp streams the call as content text
        # but still flags the turn as a tool call).
        if "get_weather" in (metrics.get("tool_calls") or []):
            return True
        return metrics.get("finish_reason") == "tool_calls"

    return {"messages": [
        {"role": "user",
         "content": _context_preamble("function_call") +
                    "What is the current weather in Paris in celsius? Use the available tool."}],
        "tools": tools,
        "checker": checker}


def _json_mode(engine, model):
    return {"messages": [
        {"role": "user",
         "content": _context_preamble("json_mode") +
                    "Return a JSON object describing the planet Mars with keys "
                    "'name', 'diameter_km' (number), and 'has_moons' (boolean)."}],
        "response_format": {"type": "json_object"}}


def _image(engine, model):
    uri = _data_uri(config.MEDIA_IMAGE, "image/jpeg")
    return {"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe what you see in this image."},
            {"type": "image_url", "image_url": {"url": uri}}]}]}


def _audio(engine, model):
    # The portable OpenAI `input_audio` content part, sent identically to every
    # engine (llama.cpp and TensorSharp both accept it), so the audio cells are
    # as apples-to-apples as the text ones.
    b64 = _b64_file(config.MEDIA_AUDIO)
    fmt = config.MEDIA_AUDIO.suffix.lstrip(".").lower() or "mp3"
    return {"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "Transcribe and summarize this audio."},
            {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}]}]}


def _video(engine, model):
    frames = _video_frames_b64(str(config.MEDIA_VIDEO), n=4)
    if not frames:
        # No frames decoded; signal an empty build so the caller records a skip.
        return {"messages": None, "detail": "could not decode video frames"}
    parts = [{"type": "text",
              "text": "These are sampled frames from a video. Describe what is happening."}]
    for fr in frames:
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{fr}"}})
    return {"messages": [{"role": "user", "content": parts}]}


# ---------------------------------------------------------------------------
# Agentic tool loop (multi-step, client-driven)
# ---------------------------------------------------------------------------
# Two DEPENDENT tool calls and then a final answer, over three round trips:
#
#   1. read_invoice("INV-472")        -> {unit_price: 13.75, quantity: 5}
#   2. calculate_total(13.75, 5)      -> {total: 74.25}   (a handling fee the
#                                        model was never told about is in there)
#   3. final answer                   -> {"invoice_id": "INV-472", "total": 74.25}
#
# The fixtures are chosen so that a correct final answer PROVES the tool results
# were used, rather than merely correlating with them: 13.75 and 5 appear
# nowhere in the prompt, so turn 2's arguments can only have come from turn 1's
# result, and 74.25 is deliberately NOT unit_price * quantity (68.75), so the
# total can only have come from turn 2's result. A model that ignores the tools
# and answers from arithmetic lands on 68.75 and is marked wrong.
#
# The tool-call shape is checked the way `validate_deepseek41_tools.py` checks
# it: a structured `tool_calls` entry with a non-empty id, `type: "function"`,
# the declared name, and arguments that are a complete JSON string. Nothing is
# executed — every tool result is a fixed fixture.
_AGENTIC_INVOICE_ID = "INV-472"
_AGENTIC_INVOICE_RESULT = {"invoice_id": _AGENTIC_INVOICE_ID,
                           "unit_price": 13.75, "quantity": 5}
_AGENTIC_TOTAL_RESULT = {"total": 74.25,
                         "note": "includes the 5.50 handling fee on this account"}
_AGENTIC_FINAL = {"invoice_id": _AGENTIC_INVOICE_ID, "total": 74.25}


def _tool(name: str, description: str, properties: dict, required: list) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False}}}


_AGENTIC_TOOLS = [
    _tool("read_invoice", "Read an invoice by its identifier.",
          {"invoice_id": {"type": "string"}}, ["invoice_id"]),
    _tool("calculate_total", "Calculate the payable total for an invoice line.",
          {"unit_price": {"type": "number"}, "quantity": {"type": "integer"}},
          ["unit_price", "quantity"]),
]


def _structured_call(metrics, expect_name: str, expect_args: dict) -> dict:
    """The one structured tool call `metrics` must carry, or raise.

    Raising is the point: it stops the conversation instead of inventing a tool
    result for a call the model never made, which would leave the remaining
    turns measuring a different workload than the one the cell claims to be."""
    if metrics.get("finish_reason") != "tool_calls":
        raise ValueError(f"expected finish_reason=tool_calls for {expect_name}, "
                         f"got {metrics.get('finish_reason') or 'none'}")
    calls = metrics.get("tool_call_details") or []
    if len(calls) != 1:
        raise ValueError(f"expected exactly one structured tool call for "
                         f"{expect_name}, got {len(calls)}")
    call = calls[0]
    if not isinstance(call.get("id"), str) or not call["id"] or call.get("type") != "function":
        raise ValueError("tool call needs a non-empty id and type=function")
    fn = call.get("function") or {}
    if fn.get("name") != expect_name:
        raise ValueError(f"expected {expect_name}, got {fn.get('name') or 'no name'}")
    try:
        args = json.loads(fn.get("arguments") or "")
    except ValueError as ex:
        raise ValueError(f"{expect_name} arguments are not complete JSON: {ex}")
    if args != expect_args:
        raise ValueError(f"expected {expect_name}({expect_args}), got {args}")
    return call


def _tool_turn(metrics, messages, result: dict, expect_name: str,
               expect_args: dict, follow_up: str, final: bool) -> dict:
    """Append the assistant's own call plus its fixture result, then ask the
    next question. The call id is echoed back exactly as the model produced it,
    so a server that loses track of its own ids fails here rather than later."""
    call = _structured_call(metrics, expect_name, expect_args)
    nxt = list(messages) + [
        metrics["assistant_message"],
        {"role": "tool", "tool_call_id": call["id"],
         "content": json.dumps(result, sort_keys=True)},
        {"role": "user", "content": follow_up}]
    out = {"messages": nxt}
    if final:
        # Ask for the answer, not another call. `tool_choice: "none"` keeps the
        # tool catalogue in the conversation (so the history still parses) while
        # forbidding a new call, which is how validate_deepseek41_tools.py ends
        # its own workflow.
        out["extra_body"] = {"tool_choice": "none"}
    return out


def _json_answer(metrics):
    """The assistant's final content parsed as JSON, or None.

    One tolerance, and only one: a single fenced ```json block is unwrapped
    first. The fence is an artifact of some chat templates rather than a wrong
    answer, and unwrapping it is what keeps this checkable on every engine."""
    msg = metrics.get("assistant_message")
    text = (msg or {}).get("content") if isinstance(msg, dict) else None
    if not isinstance(text, str):
        return None
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    try:
        return json.loads(body.strip())
    except ValueError:
        return None


def _agentic(engine, model):
    def checker(metrics):
        # Three round trips, and the last one carries the tool-derived answer.
        # `turns` < 3 means a follow-up refused to continue (the detail on the
        # cell says which), which is a failed workflow, not a passed one.
        if int(metrics.get("turns", 1) or 1) != 3:
            return False
        return _json_answer(metrics) == _AGENTIC_FINAL

    followups = [
        lambda m, msgs: _tool_turn(
            m, msgs, _AGENTIC_INVOICE_RESULT, "read_invoice",
            {"invoice_id": _AGENTIC_INVOICE_ID},
            "Now call calculate_total with the unit_price and quantity that "
            "read_invoice returned.", final=False),
        lambda m, msgs: _tool_turn(
            m, msgs, _AGENTIC_TOTAL_RESULT, "calculate_total",
            {"unit_price": _AGENTIC_INVOICE_RESULT["unit_price"],
             "quantity": _AGENTIC_INVOICE_RESULT["quantity"]},
            "Report the result. Return only a JSON object with exactly the keys "
            "invoice_id and total, taking total from calculate_total's result. "
            "Do not recompute it and do not call any tool.", final=True),
    ]
    return {"messages": [
        {"role": "user",
         "content": _context_preamble("agentic") +
                    f"Read invoice {_AGENTIC_INVOICE_ID} with read_invoice. Call "
                    f"exactly one tool in this turn and wait for its result."}],
        "tools": _AGENTIC_TOOLS,
        "followups": followups,
        "checker": checker}


# ---------------------------------------------------------------------------
# Code generation + edit (multi-step, client-driven)
# ---------------------------------------------------------------------------
# Turn 1 asks for a small program; turn 2 asks for one specific, mechanically
# checkable change to THAT program and the checker parses the result to see
# whether the change actually landed. Both turns are ordinary chat completions,
# so this runs on every engine and every registered model.
_CODE_FN = "slugify"
_CODE_EDITED_FN = "slugify_title"
_CODE_LIMIT_PARAM = "max_length"
_CODE_LIMIT_DEFAULT = 40


def _python_source(metrics) -> str:
    """The Python source in an assistant reply: the first fenced block when the
    reply has one, otherwise the whole reply (models that were told "code only"
    often skip the fence)."""
    msg = metrics.get("assistant_message")
    text = (msg or {}).get("content") if isinstance(msg, dict) else None
    if not isinstance(text, str):
        return ""
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            block = parts[1]
            # Drop the language tag on the fence's opening line.
            return block.split("\n", 1)[1] if "\n" in block else ""
    return text


def _function_def(source: str, name: str):
    """The `ast.FunctionDef` called `name` in `source`, or None when the source
    does not parse or does not define it."""
    import ast
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _code_edit(engine, model):
    def checker(metrics):
        import ast
        turns = metrics.get("turn_metrics") or []
        if len(turns) != 2:
            return False
        # The first turn must really have produced the program being edited,
        # otherwise turn 2 was asked to edit nothing and a fresh function that
        # happens to match would pass.
        first = _function_def(_python_source(turns[0]), _CODE_FN)
        if first is None or [a.arg for a in first.args.args] != ["text"]:
            return False
        edited = _function_def(_python_source(turns[-1]), _CODE_EDITED_FN)
        if edited is None:
            return False
        if [a.arg for a in edited.args.args] != ["text", _CODE_LIMIT_PARAM]:
            return False
        defaults = edited.args.defaults
        if len(defaults) != 1 or not isinstance(defaults[0], ast.Constant):
            return False
        if defaults[0].value != _CODE_LIMIT_DEFAULT:
            return False
        # A parameter that is declared but never read is a rename, not the edit
        # that was asked for: the truncation has to be in the body.
        return any(isinstance(n, ast.Name) and n.id == _CODE_LIMIT_PARAM
                   for n in ast.walk(ast.Module(body=edited.body, type_ignores=[])))

    def edit_request(metrics, messages):
        return {"messages": list(messages) + [
            metrics["assistant_message"],
            {"role": "user", "content":
                f"Make exactly this change and return the complete updated "
                f"function the same way: rename it to `{_CODE_EDITED_FN}`, and give "
                f"it a second parameter `{_CODE_LIMIT_PARAM}` that defaults to "
                f"{_CODE_LIMIT_DEFAULT} and truncates the returned slug to at most "
                f"that many characters. Change nothing else."}]}

    return {"messages": [
        {"role": "user",
         "content": _context_preamble("code_edit") +
                    f"Write a Python function `{_CODE_FN}(text)` that lowercases "
                    f"the text, replaces every run of non-alphanumeric characters "
                    f"with a single hyphen, and strips leading and trailing "
                    f"hyphens. Reply with only the function in one ```python code "
                    f"block."}],
        "followups": [edit_request],
        "checker": checker}


_BUILDERS = {
    "text_short": _text_short,
    "text_long": _text_long,
    "multi_turn": _multi_turn,
    "function_call": _function_call,
    "json_mode": _json_mode,
    "agentic": _agentic,
    "code_edit": _code_edit,
    "image": _image,
    "audio": _audio,
    "video": _video,
}


def build_request(scenario_id: str, engine: str, model: config.ModelSpec) -> dict:
    if scenario_id.startswith("prefill_"):
        req = _prefill(_parse_prefill_target(scenario_id), engine, model)
    else:
        builder = _BUILDERS[scenario_id]
        req = builder(engine, model)
    req.setdefault("tools", None)
    req.setdefault("response_format", None)
    req.setdefault("checker", None)
    req.setdefault("followups", None)
    return req


_DEFAULT_WEATHER_TOOL = """[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a city.",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string", "description": "City name"},
          "units": {"type": "string", "enum": ["c", "f"], "description": "Temperature units"}
        },
        "required": ["city"]
      }
    }
  }
]"""
