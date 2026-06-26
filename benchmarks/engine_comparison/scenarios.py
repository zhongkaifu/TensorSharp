#!/usr/bin/env python3
"""
Scenario -> request payload builders.

Each builder returns a dict describing one `/v1/chat/completions` request:
  { "messages", "tools", "response_format", "checker" }
`checker(metrics) -> Optional[bool]` is used by correctness-bearing scenarios
(function_call) to record whether the model did the expected thing.

Image is sent in the portable OpenAI `image_url` form to every engine. Audio
and video differ per engine (TensorSharp accepts a message-level base64 array /
a sampled-frame image sequence; llama.cpp uses the OpenAI `input_audio` part),
so those builders take the engine id.
"""
from __future__ import annotations

import base64
import functools
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
         "content": "Explain what a transformer neural network is in three concise sentences."}]}


def _text_long(engine, model):
    doc = _read_asset("long_text.txt", "Lorem ipsum dolor sit amet. " * 200)
    return {"messages": [
        {"role": "user",
         "content": doc + "\n\nSummarize the passage above in two sentences."}]}


def _multi_turn(engine, model):
    return {"messages": [
        {"role": "user", "content": "I'm planning a trip to Japan. My budget is $3000."},
        {"role": "assistant", "content": "Great! Japan is a wonderful destination. With a $3000 budget, "
                                         "you can have a comfortable week-long trip. What time of year are you thinking?"},
        {"role": "user", "content": "Cherry blossom season. Which two cities should I prioritize and why?"},
        {"role": "assistant", "content": "For cherry blossom season, Kyoto and Tokyo are the top picks. "
                                         "Kyoto offers classic temple-and-blossom scenery; Tokyo adds variety and nightlife."},
        {"role": "user", "content": "Given my budget, roughly how should I split spending between those two cities?"}]}


def _function_call(engine, model):
    import json
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
         "content": "What is the current weather in Paris in celsius? Use the available tool."}],
        "tools": tools,
        "checker": checker}


def _json_mode(engine, model):
    return {"messages": [
        {"role": "user",
         "content": "Return a JSON object describing the planet Mars with keys "
                    "'name', 'diameter_km' (number), and 'has_moons' (boolean)."}],
        "response_format": {"type": "json_object"}}


def _image(engine, model):
    uri = _data_uri(config.MEDIA_IMAGE, "image/jpeg")
    return {"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe what you see in this image."},
            {"type": "image_url", "image_url": {"url": uri}}]}]}


def _audio(engine, model):
    b64 = _b64_file(config.MEDIA_AUDIO)
    fmt = config.MEDIA_AUDIO.suffix.lstrip(".").lower() or "mp3"
    if engine == "llamacpp":
        # llama.cpp / OpenAI input_audio content part.
        return {"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Transcribe and summarize this audio."},
                {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}]}]}
    # TensorSharp: message-level base64 `audios` array.
    return {"messages": [
        {"role": "user",
         "content": "Transcribe and summarize this audio.",
         "audios": [b64]}]}


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


_BUILDERS = {
    "text_short": _text_short,
    "text_long": _text_long,
    "multi_turn": _multi_turn,
    "function_call": _function_call,
    "json_mode": _json_mode,
    "image": _image,
    "audio": _audio,
    "video": _video,
}


def build_request(scenario_id: str, engine: str, model: config.ModelSpec) -> dict:
    builder = _BUILDERS[scenario_id]
    req = builder(engine, model)
    req.setdefault("tools", None)
    req.setdefault("response_format", None)
    req.setdefault("checker", None)
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
