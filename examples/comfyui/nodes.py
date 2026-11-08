# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
#
# TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
"""ComfyUI nodes that run whole video and image generations on a TensorSharp server.

Each generation node sends one request to TensorSharp's streaming route, turns its step
events into ComfyUI's progress bar, closes the connection when the prompt is cancelled
(TensorSharp stops the clip when its caller goes away), and downloads the result into
ComfyUI's output folder.
"""

from __future__ import annotations

import atexit
import base64
import io as _bytes_io
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


def _image_tensor(path: str) -> torch.Tensor:
    with Image.open(path) as img:
        array = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None, ...]


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
                io.Int.Input("width", default=0, min=0, max=4096, step=32, tooltip="0: the model's default."),
                io.Int.Input("height", default=0, min=0, max=4096, step=32, tooltip="0: the model's default."),
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
        image = _image_tensor(path)
        return io.NodeOutput(image, ui=ui.PreviewImage(image, cls=cls))


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
        return [TensorSharpServer, TensorSharpVideo, TensorSharpImage, TensorSharpStopServer]


async def comfy_entrypoint() -> TensorSharpExtension:
    return TensorSharpExtension()
