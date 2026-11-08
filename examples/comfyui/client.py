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
routes and downloads. It imports nothing from ComfyUI, so it can be tested on its own."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from typing import AsyncIterator, Callable, Mapping

import aiohttp

# The TensorSharp seed is a signed 64-bit integer; ComfyUI's seed widget goes to 2^64 - 1.
MAX_SEED = 2**63 - 1

VIDEO_MODES = ("auto", "t2v", "i2v", "fl2v", "ref")


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

    async def download(self, url: str, destination: str) -> str:
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        async with self._session.get(self._url(url)) as resp:
            if resp.status != 200:
                raise TensorSharpError(f"download of {url} failed: HTTP {resp.status}")
            with open(destination, "wb") as out:
                async for chunk in resp.content.iter_chunked(1 << 20):
                    out.write(chunk)
        return destination
