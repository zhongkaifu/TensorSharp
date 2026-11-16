# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#
# TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
"""ComfyUI nodes that run video, image and audio generations on a TensorSharp server.

Each generation node sends one request to TensorSharp's streaming route, turns its step
events into ComfyUI's progress bar, closes the connection when the prompt is cancelled
(TensorSharp stops the clip when its caller goes away), and downloads the result into
ComfyUI's output folder.
"""

from __future__ import annotations

import atexit
import base64
import io as _bytes_io
import json
import logging
import os
import wave

import numpy as np
import torch
from PIL import Image
from typing_extensions import override

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, Input, InputImpl, io, ui

from . import client as ts
from . import server_manager as sm

log = logging.getLogger("comfyui_tensorsharp")

ServerType = io.Custom("TENSORSHARP_SERVER")
LOG_DIR = os.path.join(folder_paths.base_path, "logs")
DEFAULT_URL = "http://127.0.0.1:5090"
CATEGORY = "TensorSharp"

atexit.register(sm.stop_all)


# ---- helpers -------------------------------------------------------------------------

def _free_comfy_memory() -> None:
    """Move ComfyUI's own models off the GPU so the TensorSharp server has the memory."""
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache(True)


def _trace_headers() -> dict[str, str]:
    """W3C trace context of the running ComfyUI node, when ComfyUI exports telemetry,
    so the TensorSharp request joins the same trace."""
    try:
        from opentelemetry.propagate import inject
    except ImportError:
        return {}
    headers: dict[str, str] = {}
    inject(headers)
    return headers


def _png(image: torch.Tensor) -> bytes:
    """One ComfyUI image, [H, W, C] floats in 0..1, as PNG bytes."""
    array = (image[..., :3].clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    buffer = _bytes_io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


# The node's own preview is at most this many pixels on its long side. ComfyUI writes a
# preview as a full PNG (compress level 1): 1.1 s for one 4096x4096 picture, more than
# the decode, where 1024 pixels takes about 0.1 s. The node's output stays full size.
PREVIEW_MAX_SIDE = 1024


def _image_and_preview(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The picture at `path` as a ComfyUI image, and a copy reduced for the node's preview
    (the picture itself when it is small already)."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
    image = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)[None, ...]
    factor = -(-max(rgb.size) // PREVIEW_MAX_SIDE)   # ceiling division
    if factor <= 1:
        return image, image
    small = rgb.reduce(factor)                       # box filter over factor x factor blocks
    return image, torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0)[None, ...]


def _wav_audio(path: str) -> dict | None:
    """A 16-bit PCM WAV as ComfyUI AUDIO, for a server that could not mux the track."""
    try:
        with wave.open(path, "rb") as w:
            if w.getsampwidth() != 2:
                return None
            channels, rate = w.getnchannels(), w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    except (wave.Error, OSError) as ex:
        log.warning("TensorSharp: could not read %s: %s", path, ex)
        return None
    waveform = torch.from_numpy(pcm.reshape(-1, channels).T.copy())[None, ...]
    return {"waveform": waveform, "sample_rate": rate}


def _output_path(prefix: str, extension: str) -> tuple[str, str, str]:
    """A new numbered file under ComfyUI's output folder: (full path, file name, subfolder)."""
    folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(prefix, folder_paths.get_output_directory())
    file = f"{name}_{counter:05}_.{extension}"
    return os.path.join(folder, file), file, subfolder


async def _stream_with_progress(client: ts.TensorSharpClient, route: str, body: dict, expected_steps: int) -> dict:
    """Run one streaming generation and return its final event."""
    bar = comfy.utils.ProgressBar(max(expected_steps, 1))
    result = None
    try:
        async for event in client.stream(route, body, should_stop=comfy.model_management.processing_interrupted):
            if event.get("done"):
                result = event
                continue
            total = event.get("total") or 0
            step = event.get("step") or 0
            if total > 0:
                bar.update_absolute(min(step, total), total, _preview(event.get("image")))
    except ts.Interrupted:
        log.info("TensorSharp: %s cancelled from ComfyUI", route)
        comfy.model_management.throw_exception_if_processing_interrupted()
        raise
    if result is None:
        raise ts.TensorSharpError("the server ended the stream without a result")
    return result


def _preview(data_url) -> tuple | None:
    """A denoising preview the image routes send (a PNG data URL), in the form ComfyUI's
    progress bar shows on the node."""
    if not isinstance(data_url, str) or "," not in data_url:
        return None
    try:
        image = Image.open(_bytes_io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
        image.load()
        return ("PNG", image.convert("RGB"), 512)
    except (ValueError, OSError) as ex:
        log.debug("TensorSharp: unreadable preview: %s", ex)
        return None


def _seed_input() -> io.Int.Input:
    return io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                        tooltip="Noise seed. TensorSharp takes it modulo 2^63.")


def _audio_from_wav(data: bytes) -> dict:
    """The WAV an audio route answers as ComfyUI AUDIO ({waveform [1, C, N], sample_rate})."""
    samples, rate = ts.read_wav(data)
    return {"waveform": torch.from_numpy(samples)[None, ...], "sample_rate": rate}


def _wav_from_audio(audio: dict) -> bytes:
    """The first clip of a ComfyUI AUDIO batch as a 16-bit WAV."""
    waveform = audio["waveform"]
    if waveform.dim() == 3:
        waveform = waveform[0]
    return ts.wav_bytes(waveform.float().cpu().numpy(), int(audio["sample_rate"]))


def _require_audio(info: dict, kind: str, url: str, what: str) -> dict:
    """The /api/models audio block of a server hosting a `kind` model, or a clear refusal."""
    audio = info.get("audio") or {}
    if audio.get("kind") != kind:
        raise ValueError(f"{url} hosts {sm.hosted_model(info) or 'a model'}, which does not {what}; "
                         f"pick a profile for a {kind} model")
    return audio


# ---- nodes ---------------------------------------------------------------------------

class TensorSharpServer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TensorSharpServer",
            display_name="TensorSharp Server",
            category=CATEGORY,
            description=(
                "A TensorSharp server to generate on. With a profile, starts a server for that "
                "profile's model on the URL's port, or reuses one already hosting it; a server "
                "this ComfyUI started for another model is stopped first. Without one, connects "
                "to a server that is already running."),
            inputs=[
                io.String.Input("url", default=DEFAULT_URL, tooltip="Base URL of the TensorSharp server."),
                io.Combo.Input("profile", options=[sm.CONNECT_ONLY, *sm.list_profiles()], default=sm.CONNECT_ONLY,
                               tooltip="Launch profile from this node pack's profiles folder."),
            ],
            outputs=[
                ServerType.Output(display_name="server"),
                io.String.Output(display_name="model"),
            ],
        )

    @classmethod
    async def execute(cls, url: str, profile: str) -> io.NodeOutput:
        info = await sm.ensure_server(url, profile, LOG_DIR, before_start=_free_comfy_memory)
        model = sm.hosted_model(info)
        return io.NodeOutput({"url": url.rstrip("/"), "profile": profile}, model)


class TensorSharpVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TensorSharpVideo",
            display_name="TensorSharp Video",
            category=CATEGORY,
            is_output_node=True,
            description=(
                "Generates one clip, with its soundtrack when the model makes one (LTX 2.5), "
                "on a TensorSharp server and saves the MP4 to the output folder. "
                "Zero leaves a setting to the hosted model's defaults."),
            inputs=[
                ServerType.Input("server"),
                io.String.Input("prompt", multiline=True, default=""),
                io.Combo.Input("mode", options=list(ts.VIDEO_MODES), default="auto",
                               tooltip="auto: inferred from the images given. t2v text only, i2v first frame, "
                                       "fl2v last frame (and optional first), ref reference images."),
                io.Int.Input("width", default=0, min=0, max=4096, step=16, tooltip="0: the model's default for the mode."),
                io.Int.Input("height", default=0, min=0, max=4096, step=16, tooltip="0: the model's default for the mode."),
                io.Int.Input("frames", default=0, min=0, max=1024, tooltip="0: the server's --video-frames."),
                io.Int.Input("steps", default=0, min=0, max=200, tooltip="0: the model's or LoRA plug-in's schedule."),
                io.Float.Input("cfg", default=0.0, min=0.0, max=30.0, step=0.1, tooltip="0: the model's default."),
                _seed_input(),
                io.Int.Input("fps", default=0, min=0, max=120, tooltip="0: the model's frame rate."),
                io.Boolean.Input("generate_audio", default=True),
                io.Boolean.Input("free_comfyui_memory", default=True,
                                 tooltip="Unload ComfyUI's models from the GPU before the generation."),
                io.String.Input("filename_prefix", default="tensorsharp/video"),
                io.String.Input("negative_prompt", multiline=True, default="", optional=True, advanced=True),
                io.Image.Input("first_frame", optional=True, tooltip="The clip starts on this picture (first of a batch)."),
                io.Image.Input("last_frame", optional=True, tooltip="The clip ends on this picture (first of a batch)."),
                io.Image.Input("reference_images", optional=True,
                               tooltip="Each picture of the batch is one reference (mode ref)."),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
                io.Image.Output(display_name="last_frame"),
                io.Audio.Output(display_name="audio"),
                io.String.Output(display_name="path"),
            ],
        )

    @classmethod
    async def execute(cls, server: dict, prompt: str, mode: str, width: int, height: int, frames: int, steps: int,
                      cfg: float, seed: int, fps: int, generate_audio: bool, free_comfyui_memory: bool,
                      filename_prefix: str, negative_prompt: str = "", first_frame: Input.Image | None = None,
                      last_frame: Input.Image | None = None,
                      reference_images: Input.Image | None = None) -> io.NodeOutput:
        info = await sm.ensure_server(server["url"], server["profile"], LOG_DIR, before_start=_free_comfy_memory)
        if not info.get("video"):
            raise ValueError(f"{server['url']} hosts {sm.hosted_model(info) or 'a model'} that does not generate video")
        if free_comfyui_memory:
            _free_comfy_memory()

        async with ts.TensorSharpClient(server["url"], headers=_trace_headers()) as client:
            end_image = await client.upload_png(_png(last_frame[0]), "last-frame") if last_frame is not None else None
            references = None
            if reference_images is not None:
                references = [await client.upload_png(_png(img), f"reference-{i}") for i, img in enumerate(reference_images)]
            body = ts.build_video_body(
                prompt, negative_prompt=negative_prompt, mode=mode, width=width, height=height, frames=frames,
                steps=steps, cfg=cfg, seed=seed, fps=fps, generate_audio=generate_audio,
                first_frame_png=_png(first_frame[0]) if first_frame is not None else None,
                end_image=end_image, reference_images=references)
            result = await _stream_with_progress(client, "/api/video-generate/stream", body, steps)
            path, file, subfolder = _output_path(filename_prefix, "mp4")
            await client.download(result["url"], path)
            sidecar = None
            if result.get("audioUrl") and not result.get("audioMuxed"):
                sidecar = await client.download(result["audioUrl"], os.path.splitext(path)[0] + ".wav")

        log.info("TensorSharp: %sx%s, %s frames at %s fps, seed %s, %.1f s -> %s", result.get("width"),
                 result.get("height"), result.get("frames"), result.get("fps"), result.get("seed"),
                 result.get("elapsedSeconds") or 0.0, path)
        video = InputImpl.VideoFromFile(path)
        components = video.get_components()
        audio = components.audio if components.audio is not None else (_wav_audio(sidecar) if sidecar else None)
        return io.NodeOutput(video, components.images[-1:].clone(), audio, path,
                             ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)]))


class TensorSharpImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TensorSharpImage",
            display_name="TensorSharp Image",
            category=CATEGORY,
            description=(
                "Generates a picture on a TensorSharp server hosting Qwen-Image-2.1, or edits the "
                "given pictures when references are connected. Zero leaves a setting to the model."),
            inputs=[
                ServerType.Input("server"),
                io.String.Input("prompt", multiline=True, default=""),
                io.Int.Input("width", default=0, min=0, max=8192, step=32, tooltip="0: the model's default (MiniMax-H3-Image: 2048, up to about 16 MP such as 6272x2688)."),
                io.Int.Input("height", default=0, min=0, max=8192, step=32, tooltip="0: the model's default."),
                io.Int.Input("steps", default=0, min=0, max=200, tooltip="0: 40 steps, or the LoRA plug-in's recipe."),
                io.Float.Input("cfg", default=0.0, min=0.0, max=30.0, step=0.1, tooltip="0: the model's default."),
                _seed_input(),
                io.Boolean.Input("free_comfyui_memory", default=True),
                io.String.Input("negative_prompt", multiline=True, default="", optional=True, advanced=True),
                io.Image.Input("images", optional=True, tooltip="Pictures to edit; each one of the batch is a reference."),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(cls, server: dict, prompt: str, width: int, height: int, steps: int, cfg: float, seed: int,
                      free_comfyui_memory: bool, negative_prompt: str = "",
                      images: Input.Image | None = None) -> io.NodeOutput:
        await sm.ensure_server(server["url"], server["profile"], LOG_DIR, before_start=_free_comfy_memory)
        if free_comfyui_memory:
            _free_comfy_memory()
        async with ts.TensorSharpClient(server["url"], headers=_trace_headers()) as client:
            paths = None
            if images is not None:
                paths = [await client.upload_png(_png(img), f"edit-{i}") for i, img in enumerate(images)]
            body = ts.build_image_body(prompt, negative_prompt=negative_prompt, width=width, height=height,
                                       steps=steps, cfg=cfg, seed=seed, image_paths=paths)
            route = "/api/image-edit/stream" if paths else "/api/image-generate/stream"
            result = await _stream_with_progress(client, route, body, steps or 40)
            path = os.path.join(folder_paths.get_temp_directory(), "tensorsharp", os.path.basename(result["url"]))
            await client.download(result["url"], path)
        image, preview = _image_and_preview(path)
        return io.NodeOutput(image, ui=ui.PreviewImage(preview, cls=cls))


class TensorSharpSpeech(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TensorSharpSpeech",
            display_name="TensorSharp Speech",
            category=CATEGORY,
            description=(
                "Speaks text on a TensorSharp server hosting a speech model (Kokoro-82M: Mandarin text, "
                "24 kHz). Long text is split at newlines and sentence ends, as Kokoro's own pipeline does."),
            inputs=[
                ServerType.Input("server"),
                io.String.Input("text", multiline=True, default=""),
                io.String.Input("voice", default="",
                                tooltip="A voice the model has (zm_yunjian, zf_xiaoxiao, ...); empty for its default. "
                                        "The node lists them in the log when the name is unknown."),
                io.Float.Input("speed", default=1.0, min=0.25, max=4.0, step=0.05),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    async def execute(cls, server: dict, text: str, voice: str, speed: float) -> io.NodeOutput:
        info = await sm.ensure_server(server["url"], server["profile"], LOG_DIR, before_start=_free_comfy_memory)
        caps = _require_audio(info, "speech", server["url"], "speak text")
        if voice.strip() and caps.get("voices") and voice.strip() not in caps["voices"]:
            raise ValueError(f"the hosted model has no voice {voice!r}; it has {', '.join(caps['voices'])}")
        body = ts.build_speech_body(text, voice=voice, speed=speed)
        async with ts.TensorSharpClient(server["url"], headers=_trace_headers()) as client:
            try:
                data = await client.post_audio("/v1/audio/speech", body,
                                               should_stop=comfy.model_management.processing_interrupted)
            except ts.Interrupted:
                comfy.model_management.throw_exception_if_processing_interrupted()
                raise
        audio = _audio_from_wav(data)
        return io.NodeOutput(audio, ui=ui.PreviewAudio(audio, cls=cls))


class TensorSharpTranscribe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TensorSharpTranscribe",
            display_name="TensorSharp Transcribe",
            category=CATEGORY,
            description=(
                "Transcribes audio of any length on a TensorSharp server hosting a speech recognition "
                "model (Whisper), with segment timestamps and, when asked, word timestamps."),
            inputs=[
                ServerType.Input("server"),
                io.Audio.Input("audio"),
                io.String.Input("language", default="", tooltip="An ISO 639-1 code such as en or zh; empty detects it."),
                io.Combo.Input("task", options=["transcribe", "translate"], default="transcribe",
                               tooltip="translate: to English, on checkpoints trained for it (large-v3-turbo is not)."),
                io.Boolean.Input("word_timestamps", default=False,
                                 tooltip="Also time each word (a second decoder pass per window)."),
                io.String.Input("prompt", multiline=True, default="", optional=True, advanced=True,
                                tooltip="Text before the audio: names and spellings the speech uses."),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                io.String.Output(display_name="srt"),
                io.String.Output(display_name="json", tooltip="OpenAI's verbose_json: segments, and words when timed."),
            ],
        )

    @classmethod
    async def execute(cls, server: dict, audio: dict, language: str, task: str, word_timestamps: bool,
                      prompt: str = "") -> io.NodeOutput:
        info = await sm.ensure_server(server["url"], server["profile"], LOG_DIR, before_start=_free_comfy_memory)
        _require_audio(info, "transcription", server["url"], "transcribe speech")
        wav = _wav_from_audio(audio)
        async with ts.TensorSharpClient(server["url"], headers=_trace_headers()) as client:
            try:
                result = await client.transcribe(wav, language=language, translate=task == "translate", prompt=prompt,
                                                 word_timestamps=word_timestamps,
                                                 should_stop=comfy.model_management.processing_interrupted)
            except ts.Interrupted:
                comfy.model_management.throw_exception_if_processing_interrupted()
                raise
        text = str(result.get("text", "")).strip()
        srt = ts.srt_from_segments(result.get("segments") or [])
        log.info("TensorSharp: transcribed %.1f s (%s), %d segments", result.get("duration") or 0.0,
                 result.get("language"), len(result.get("segments") or []))
        return io.NodeOutput(text, srt, json.dumps(result, ensure_ascii=False), ui=ui.PreviewText(text))


class TensorSharpMusic(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TensorSharpMusic",
            display_name="TensorSharp Music",
            category=CATEGORY,
            description=(
                "Writes a song (48 kHz stereo) from a style caption and lyrics on a TensorSharp server hosting "
                "a music model (ACE-Step 1.5). Zero or empty settings are left to the model's planner."),
            inputs=[
                ServerType.Input("server"),
                io.String.Input("tags", multiline=True, default="", tooltip="Style caption: genre, instruments, mood, voice."),
                io.String.Input("lyrics", multiline=True, default="[Instrumental]",
                                tooltip="Lyrics with [verse] / [chorus] section tags, or [Instrumental]."),
                io.Float.Input("duration", default=30.0, min=1.0, max=600.0, step=0.1, tooltip="Seconds."),
                io.Int.Input("bpm", default=0, min=0, max=300, tooltip="0: the planner chooses."),
                io.String.Input("key", default="", tooltip="Such as 'C major' or 'A minor'; empty: the planner chooses."),
                io.Combo.Input("time_signature", options=list(ts.TIME_SIGNATURES), default="auto"),
                io.String.Input("language", default="en", tooltip="Language of the lyrics (en, zh, ja, ...)."),
                io.Int.Input("steps", default=0, min=0, max=200, tooltip="0: the checkpoint's own (8 for turbo)."),
                _seed_input(),
                io.Boolean.Input("free_comfyui_memory", default=True),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    async def execute(cls, server: dict, tags: str, lyrics: str, duration: float, bpm: int, key: str,
                      time_signature: str, language: str, steps: int, seed: int,
                      free_comfyui_memory: bool) -> io.NodeOutput:
        info = await sm.ensure_server(server["url"], server["profile"], LOG_DIR, before_start=_free_comfy_memory)
        _require_audio(info, "music", server["url"], "generate music")
        if free_comfyui_memory:
            _free_comfy_memory()
        body = ts.build_music_body(tags, lyrics=lyrics, duration=duration, bpm=bpm, key=key,
                                   time_signature=time_signature, language=language, steps=steps, seed=seed)
        async with ts.TensorSharpClient(server["url"], headers=_trace_headers()) as client:
            try:
                data = await client.post_audio("/v1/audio/music", body,
                                               should_stop=comfy.model_management.processing_interrupted)
            except ts.Interrupted:
                comfy.model_management.throw_exception_if_processing_interrupted()
                raise
        audio = _audio_from_wav(data)
        return io.NodeOutput(audio, ui=ui.PreviewAudio(audio, cls=cls))


class TensorSharpStopServer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TensorSharpStopServer",
            display_name="TensorSharp Stop Server",
            category=CATEGORY,
            description=(
                "Stops the TensorSharp server this ComfyUI started, once the value it receives (a "
                "clip, a picture, anything) is made, and passes that value on, so later nodes get "
                "the GPU memory back. A server ComfyUI did not start is left running. The next "
                "TensorSharp node starts it again."),
            inputs=[ServerType.Input("server"),
                    io.AnyType.Input("value", tooltip="The result to wait for, passed on unchanged.")],
            outputs=[io.AnyType.Output(display_name="value")],
        )

    @classmethod
    async def execute(cls, server: dict, value) -> io.NodeOutput:
        if not await sm.stop_server(server["url"]):
            log.info("TensorSharp: %s was not started by this ComfyUI; left running", server["url"])
        return io.NodeOutput(value)


class TensorSharpExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TensorSharpServer, TensorSharpVideo, TensorSharpImage, TensorSharpSpeech, TensorSharpTranscribe,
                TensorSharpMusic, TensorSharpStopServer]


async def comfy_entrypoint() -> TensorSharpExtension:
    return TensorSharpExtension()
