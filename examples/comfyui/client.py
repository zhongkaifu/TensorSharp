# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#
# TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
"""HTTP client for a TensorSharp server: model info, uploads, the streaming generation
routes, the audio routes and downloads. It imports nothing from ComfyUI, so it can be
tested on its own."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import uuid
import wave
from typing import AsyncIterator, Callable, Mapping

import aiohttp
import numpy as np

# The TensorSharp seed is a signed 64-bit integer; ComfyUI's seed widget goes to 2^64 - 1.
MAX_SEED = 2**63 - 1

VIDEO_MODES = ("auto", "t2v", "i2v", "fl2v", "ref")

TIME_SIGNATURES = ("auto", "2", "3", "4", "6")


class TensorSharpError(RuntimeError):
    """The server refused or failed a request; the message is the server's own."""


class Interrupted(Exception):
    """The caller asked to stop while a generation was streaming."""


def parse_sse_data(line: bytes | str) -> dict | None:
    """Return the JSON payload of one SSE `data:` line, or None for anything else."""
    text = line.decode("utf-8", "replace") if isinstance(line, bytes) else line
    text = text.strip()
    if not text.startswith("data:"):
        return None
    payload = text[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


class SseBuffer:
    """Splits arbitrary byte chunks into complete SSE lines, keeping any partial line."""

    def __init__(self) -> None:
        self._pending = b""

    def feed(self, chunk: bytes) -> list[bytes]:
        self._pending += chunk
        *lines, self._pending = self._pending.split(b"\n")
        return lines

    def flush(self) -> list[bytes]:
        rest, self._pending = self._pending, b""
        return [rest] if rest.strip() else []


def build_video_body(
    prompt: str,
    *,
    negative_prompt: str = "",
    mode: str = "auto",
    width: int = 0,
    height: int = 0,
    frames: int = 0,
    steps: int = 0,
    cfg: float = 0.0,
    seed: int | None = None,
    fps: int = 0,
    generate_audio: bool = True,
    first_frame_png: bytes | None = None,
    end_image: str | None = None,
    reference_images: list[str] | None = None,
) -> dict:
    """The JSON body of /api/video-generate/stream. Zero or empty values are left out, so
    the hosted model's own defaults (and a LoRA plug-in's recipe) apply to them."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt is required")
    if mode not in VIDEO_MODES:
        raise ValueError(f"mode must be one of {', '.join(VIDEO_MODES)}")
    body: dict = {"prompt": prompt, "generateAudio": bool(generate_audio)}
    if negative_prompt:
        body["negativePrompt"] = negative_prompt
    if mode != "auto":
        body["videoMode"] = mode
    for key, value in (("width", width), ("height", height), ("frames", frames), ("steps", steps), ("fps", fps)):
        if value:
            body[key] = int(value)
    if cfg:
        body["cfg"] = float(cfg)
    if seed is not None:
        body["seed"] = int(seed) % (MAX_SEED + 1)
    if first_frame_png:
        body["image"] = "data:image/png;base64," + base64.b64encode(first_frame_png).decode("ascii")
    if end_image:
        body["endImage"] = end_image
    if reference_images:
        body["referenceImages"] = list(reference_images)
    return body


def build_image_body(
    prompt: str,
    *,
    negative_prompt: str = "",
    width: int = 0,
    height: int = 0,
    steps: int = 0,
    cfg: float = 0.0,
    seed: int | None = None,
    image_paths: list[str] | None = None,
) -> dict:
    """The JSON body of /api/image-generate/stream, or of /api/image-edit/stream when
    `image_paths` names uploaded references."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt is required")
    body: dict = {"prompt": prompt}
    if negative_prompt:
        body["negativePrompt"] = negative_prompt
    for key, value in (("width", width), ("height", height), ("steps", steps)):
        if value:
            body[key] = int(value)
    if cfg:
        body["cfg"] = float(cfg)
    if seed is not None:
        body["seed"] = int(seed) % (MAX_SEED + 1)
    if image_paths:
        body["imagePaths"] = list(image_paths)
    return body


def build_speech_body(text: str, *, voice: str = "", speed: float = 1.0) -> dict:
    """The JSON body of /v1/audio/speech (OpenAI's shape), asking for a WAV."""
    if not text or not text.strip():
        raise ValueError("text is required")
    body: dict = {"input": text, "response_format": "wav"}
    if voice and voice.strip():
        body["voice"] = voice.strip()
    if speed and speed != 1.0:
        body["speed"] = float(speed)
    return body


def build_music_body(
    tags: str,
    *,
    lyrics: str = "",
    duration: float = 30.0,
    bpm: int = 0,
    key: str = "",
    time_signature: str = "auto",
    language: str = "en",
    steps: int = 0,
    seed: int | None = None,
) -> dict:
    """The JSON body of /v1/audio/music. Zero or empty values are left to the planner."""
    if not tags or not tags.strip():
        raise ValueError("tags are required (a style caption such as 'lofi hip hop, piano')")
    if time_signature not in TIME_SIGNATURES:
        raise ValueError(f"time_signature must be one of {', '.join(TIME_SIGNATURES)}")
    body: dict = {"prompt": tags, "lyrics": lyrics if lyrics and lyrics.strip() else "[Instrumental]",
                  "duration": float(duration), "language": (language or "en").strip(), "response_format": "wav"}
    if bpm:
        body["bpm"] = int(bpm)
    if key and key.strip():
        body["key"] = key.strip()
    if time_signature != "auto":
        body["time_signature"] = time_signature
    if steps:
        body["steps"] = int(steps)
    if seed is not None:
        body["seed"] = int(seed) % (MAX_SEED + 1)
    return body


def wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Planar float samples [channels, n] in -1..1 as a 16-bit PCM WAV."""
    samples = np.atleast_2d(np.asarray(samples, dtype=np.float32))
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).round().astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(samples.shape[0])
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(np.ascontiguousarray(pcm.T).tobytes())
    return buffer.getvalue()


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """A 16-bit PCM WAV (what the audio routes return) as planar floats [channels, n] and its rate."""
    with wave.open(io.BytesIO(data), "rb") as w:
        if w.getsampwidth() != 2:
            raise TensorSharpError(f"expected a 16-bit WAV, got {8 * w.getsampwidth()}-bit")
        channels, rate = w.getnchannels(), w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return pcm.reshape(-1, channels).T.copy(), rate


def srt_from_segments(segments: list[dict]) -> str:
    """SubRip text from verbose_json segments ({start, end, text})."""
    def stamp(seconds: float) -> str:
        ms = int(round(max(0.0, float(seconds)) * 1000))
        return f"{ms // 3_600_000:02}:{ms // 60_000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
    blocks = []
    for i, seg in enumerate(segments, start=1):
        text = str(seg.get("text", "")).strip().replace("-->", "->")
        blocks.append(f"{i}\n{stamp(seg.get('start', 0))} --> {stamp(seg.get('end', 0))}\n{text}\n")
    return "\n".join(blocks)


def describe_error(payload, status: int | None = None) -> str:
    """The server's message from an error body in any of its envelopes."""
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            err = err.get("message")
        if err:
            return str(err)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return f"HTTP {status}: {text}" if status else text


class TensorSharpClient:
    """One aiohttp session against one server. Use as `async with TensorSharpClient(url)`."""

    def __init__(self, base_url: str, headers: Mapping[str, str] | None = None, connect_timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self._headers = dict(headers or {})
        self._timeout = aiohttp.ClientTimeout(total=None, sock_connect=connect_timeout)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "TensorSharpClient":
        self._session = aiohttp.ClientSession(timeout=self._timeout, headers=self._headers)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _url(self, path: str) -> str:
        return path if path.startswith("http") else self.base_url + path

    async def models(self) -> dict:
        async with self._session.get(self._url("/api/models"), timeout=aiohttp.ClientTimeout(total=15)) as resp:
            payload = await resp.json(content_type=None)
            if resp.status != 200:
                raise TensorSharpError(describe_error(payload, resp.status))
            return payload

    async def upload_png(self, png: bytes, name_hint: str = "comfyui") -> str:
        """Upload one PNG and return the server's file name for it."""
        form = aiohttp.FormData()
        form.add_field("file", png, filename=f"{name_hint}-{uuid.uuid4().hex[:12]}.png", content_type="image/png")
        async with self._session.post(self._url("/api/upload"), data=form) as resp:
            payload = await resp.json(content_type=None)
            if resp.status != 200 or not payload.get("file"):
                raise TensorSharpError("upload failed: " + describe_error(payload, resp.status))
            return payload["file"]

    async def stream(
        self,
        path: str,
        body: dict,
        *,
        should_stop: Callable[[], bool] = lambda: False,
        poll_seconds: float = 0.5,
    ) -> AsyncIterator[dict]:
        """POST `body` to a streaming route and yield its events until the final one.

        `should_stop` is checked at least every `poll_seconds`. When it returns True the
        connection is closed, which the server treats as a cancellation, and Interrupted
        is raised. A `{done, error}` event raises TensorSharpError."""
        async with self._session.post(self._url(path), json=body) as resp:
            if resp.status != 200 or "text/event-stream" not in resp.headers.get("Content-Type", ""):
                text = await resp.text()
                try:
                    payload = json.loads(text)
                except ValueError:
                    payload = text
                raise TensorSharpError(describe_error(payload, resp.status))
            buffer = SseBuffer()
            finished = False
            while not finished:
                if should_stop():
                    resp.close()
                    raise Interrupted()
                try:
                    # readany() takes nothing from the buffer until data is there, so a
                    # timeout never drops part of an event.
                    chunk = await asyncio.wait_for(resp.content.readany(), timeout=poll_seconds)
                except asyncio.TimeoutError:
                    continue
                lines = buffer.feed(chunk) if chunk else buffer.flush()
                for line in lines:
                    event = parse_sse_data(line)
                    if event is None:
                        continue
                    if event.get("done"):
                        if event.get("error"):
                            raise TensorSharpError(str(event["error"]))
                        finished = True
                    yield event
                    if finished:
                        break
                if not chunk and not finished:
                    # The server ends a cancelled clip without a final event.
                    raise TensorSharpError("the server closed the stream without a result")

    async def _until_done(self, request, should_stop: Callable[[], bool], poll_seconds: float):
        """Await `request` (a coroutine), checking `should_stop` every `poll_seconds`. On a stop
        the request is cancelled, which closes its connection, and Interrupted is raised."""
        task = asyncio.ensure_future(request)
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=poll_seconds)
                if done:
                    return task.result()
                if should_stop():
                    task.cancel()
                    raise Interrupted()
        finally:
            if not task.done():
                task.cancel()

    @staticmethod
    async def _read_audio_reply(resp) -> bytes:
        data = await resp.read()
        if resp.status != 200:
            try:
                payload = json.loads(data.decode("utf-8", "replace"))
            except ValueError:
                payload = data.decode("utf-8", "replace")
            raise TensorSharpError(describe_error(payload, resp.status))
        return data

    async def post_audio(self, path: str, body: dict, *, should_stop: Callable[[], bool] = lambda: False,
                         poll_seconds: float = 0.5) -> bytes:
        """POST a JSON body to /v1/audio/speech or /v1/audio/music and return the WAV it answers."""
        async def request():
            async with self._session.post(self._url(path), json=body) as resp:
                return await self._read_audio_reply(resp)
        return await self._until_done(request(), should_stop, poll_seconds)

    async def transcribe(self, wav: bytes, *, language: str = "", translate: bool = False, prompt: str = "",
                         word_timestamps: bool = False, should_stop: Callable[[], bool] = lambda: False,
                         poll_seconds: float = 0.5) -> dict:
        """POST a WAV to /v1/audio/transcriptions (or /translations) and return its verbose_json reply."""
        async def request():
            form = aiohttp.FormData()
            form.add_field("file", wav, filename="comfyui.wav", content_type="audio/wav")
            form.add_field("response_format", "verbose_json")
            if language and language.strip() and not translate:
                form.add_field("language", language.strip())
            if prompt and prompt.strip():
                form.add_field("prompt", prompt)
            if word_timestamps:
                form.add_field("timestamp_granularities[]", "word")
            route = "/v1/audio/translations" if translate else "/v1/audio/transcriptions"
            async with self._session.post(self._url(route), data=form) as resp:
                data = await self._read_audio_reply(resp)
            return json.loads(data.decode("utf-8"))
        return await self._until_done(request(), should_stop, poll_seconds)

    async def download(self, url: str, destination: str) -> str:
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        async with self._session.get(self._url(url)) as resp:
            if resp.status != 200:
                raise TensorSharpError(f"download of {url} failed: HTTP {resp.status}")
            with open(destination, "wb") as out:
                async for chunk in resp.content.iter_chunked(1 << 20):
                    out.write(chunk)
        return destination
