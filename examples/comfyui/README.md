# ComfyUI nodes

These [ComfyUI](https://github.com/Comfy-Org/ComfyUI) custom nodes run TensorSharp
generations from a ComfyUI workflow: LTX 2.5 video with its soundtrack, and Qwen-Image-2.1
pictures and edits. Each generation node sends one request to a TensorSharp server's
streaming route and returns ComfyUI types, so the result feeds ComfyUI's own nodes: Save
Video, Get Video Components, upscalers, frame interpolation, or the next clip.

ComfyUI needs no change. The pack is an ordinary custom node that uses ComfyUI's public
node API (`comfy_api.latest`), built the same way as ComfyUI's own API nodes for hosted
models, but pointed at a TensorSharp server on your machine.

## Requirements

- ComfyUI with the `comfy_api.latest` node API. It was tested with v0.40.0. The pack uses
  only packages ComfyUI already installs: aiohttp, NumPy, Pillow and PyAV.
- A built `TensorSharp.Server.Host` (see [DEVELOPMENT.md](../../DEVELOPMENT.md)). The pack
  looks for `TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll` in the checkout it
  sits in; `TENSORSHARP_SERVER_DLL` names another build.
- The model files of the preset each profile names, under `TENSORSHARP_MODELS`; see
  [LTX 2.5](../../docs/models/ltx-2.5.md) and [Qwen-Image-2.1](../../docs/models/qwenimage21.md).
  A preset downloads missing files when its server starts, which for LTX 2.5's 43 GB
  takes longer than a profile waits (`startup_timeout`). Fetch them first, for example
  with one CLI run of the preset.
- For MP4 export, FFmpeg on `PATH` or in `TS_FFMPEG`. On Windows, the CUDA runtime as
  described in DEVELOPMENT.md.

A server the pack starts inherits ComfyUI's environment, so set these variables before
starting ComfyUI.

## Install

Link this folder into ComfyUI's `custom_nodes` and restart ComfyUI:

```powershell
# Windows
New-Item -ItemType Junction -Path <ComfyUI>\custom_nodes\comfyui_tensorsharp -Target <TensorSharp>\examples\comfyui
```

```sh
# Linux and macOS
ln -s <TensorSharp>/examples/comfyui <ComfyUI>/custom_nodes/comfyui_tensorsharp
```

A link rather than a copy keeps the pack next to the server build and the `config/`
presets it uses. ComfyUI Manager installs node packs from a repository's root, so it
cannot install this one. The nodes appear under **TensorSharp** in the node library.

## Nodes

| Node | What it does |
|---|---|
| **TensorSharp Server** | Names the server (`url`) and a launch `profile`. With a profile, it starts a server for that profile on the URL's port, or reuses one already hosting its model. A server it started for another profile is stopped first. Without a profile, it connects to a server that is already running |
| **TensorSharp Video** | Makes one clip from text, a first frame (`first_frame`), a last frame (`last_frame`) or reference pictures (`reference_images`, one per picture of the batch). It outputs the `video` (an MP4 with sound, saved under `output/tensorsharp/`), its `last_frame` for chaining the next clip, the `audio` and the file `path` |
| **TensorSharp Image** | Generates a picture, or edits the connected pictures (`images`, each picture of the batch one reference). It shows the server's denoising previews on the node, and the finished picture reduced to at most 1024 pixels on its long side (the `image` output stays full size) |
| **TensorSharp Stop Server** | Stops the server this ComfyUI started once the value it receives (a clip, a picture, anything) exists, and passes that value on, so later nodes get the GPU memory back. The next TensorSharp node starts it again |

A setting left at zero is decided by the hosted model, its preset or its LoRA plug-in.
Each node shows its progress in ComfyUI's progress bar, one step per denoising step.
Cancelling closes the connection, and TensorSharp stops the generation at its next step or
phase boundary. Before each generation the node unloads ComfyUI's own models from the GPU
(`free_comfyui_memory`).

## Profiles

A TensorSharp server hosts only the model it was started with, so another model means
another process. A profile is a JSON file in [`profiles/`](profiles) that says how to
start one:

| Key | Use |
|---|---|
| `config` | A preset passed as `--config`; a relative path is taken from the repository root |
| `model` | A GGUF passed as `--model`, instead of or with `config`; a relative path is taken from the profile's folder |
| `args` | Further server arguments, such as `--backend` |
| `env` | Extra environment variables for the server |
| `path_prepend` | Folders put in front of the server's `PATH` |
| `server_dll` | The `TensorSharp.Server.Host.dll` to run, when not the default |
| `service_name` | The server's `OTEL_SERVICE_NAME`; default `tensorsharp-comfyui` |
| `startup_timeout` | Seconds to wait for the server to answer; default 900 |

Values may use environment variables (`${NAME}`; also `%NAME%` on Windows). A variable
that is not set is an error, not an empty string.

| Profile | Preset |
|---|---|
| `ltx25-distilled` | [`config/ltx-2.5-distilled.json`](../../config/ltx-2.5-distilled.json) on `ggml_cuda` |
| `minimax-h3` | [`config/minimax-h3-fl2va.json`](../../config/minimax-h3-fl2va.json) on `ggml_cuda`, 20 steps |
| `minimax-h3-turbo` | The same with the eight-step Turbo plug-in ([`config/lora/minimax-h3-fl2v-turbo-8step.json`](../../config/lora/minimax-h3-fl2v-turbo-8step.json)) |
| `minimax-h3-image` | [`config/minimax-h3-image.json`](../../config/minimax-h3-image.json) on `ggml_cuda`, for the **TensorSharp Image** node (six steps, 2048x2048 unless the node sets a size) |
| `minimax-h3-image-int8` | The same with `TS_H3_INT8_MATMUL=1`: int8 projections, ComfyUI's INT8 template's precision class and speed |
| `qwen-image-2.1` | [`config/qwen-image-2.1.json`](../../config/qwen-image-2.1.json) on `ggml_cuda` |
| `qwen-image-2.1-turbo` | [`config/qwen-image-2.1-turbo.json`](../../config/qwen-image-2.1-turbo.json) on `ggml_cuda` |

On Apple Silicon, change `--backend` to `mlx` or `ggml_metal`. Any other preset in
[`config/`](../../config), such as the Wan video presets, takes a profile of the same
shape. `TENSORSHARP_COMFYUI_PROFILES` adds folders of your own profiles, searched before
this one, so local profiles stay outside the repository.

Servers of two profiles can run side by side on their own ports: each keeps its Video
Studio store in `<ComfyUI>/logs/tensorsharp-studio-<port>` (`TS_STUDIO_DIRECTORY`, unless
the profile's `env` sets it), since one store admits one server.

Each started server writes to `<ComfyUI>/logs/tensorsharp-<port>.log`. A server ComfyUI
did not start is never stopped; the node reports the conflict instead.

A started server ends with ComfyUI. On Windows it is in a job object that is closed when
ComfyUI's process ends. On Linux it is started through `setpriv --pdeathsig` (util-linux),
which stops it when the ComfyUI thread that started it ends. macOS has neither, so after a
crash there, stop the server yourself.

## Telemetry

A started server inherits ComfyUI's environment, so it exports OpenTelemetry whenever
TensorSharp's variables are set there (see [Telemetry](../../USAGE.md#telemetry-opentelemetry)).
Each server reports under its profile's `service_name`. When an OpenTelemetry span is current
in ComfyUI, for example from a telemetry custom node, each request carries its W3C trace
context. TensorSharp's request span then becomes a child of the ComfyUI node span.

## Limits

- **Whole generations.** ComfyUI's sampler, LoRA and ControlNet nodes cannot act inside a
  TensorSharp generation; the preset, the server arguments and the request decide them.
- **One GPU.** ComfyUI's models and the TensorSharp server share it. The node unloads
  ComfyUI's models first, and Stop Server frees TensorSharp's afterwards.
- **Compressed frames.** The server returns H.264 MP4, so `last_frame` and decoded frames
  carry its compression.
- **RGB pictures.** ComfyUI's `IMAGE` type has no alpha channel, and Qwen-Image-2.1 writes
  one. An edit's noise depends on its reference pictures, alpha included. So editing a
  generated picture inside ComfyUI starts from different noise than editing its PNG file
  with the CLI. It matches an edit of the RGB file exactly.
- **One server per port.** A workflow that switches between two profiles on one port
  restarts the server each time; give each profile its own port to keep both loaded, if
  memory allows.

## Tests

```sh
python tests/test_client.py -v
```

Run them with the Python that runs ComfyUI, which has aiohttp. They cover the request
bodies, the stream reader against a fake server (split events, error events, a refused
request, a stream that ends without a result, and cancellation closing the connection),
profiles, and the server lookup. They need neither ComfyUI nor a GPU.

## Validation

Windows on Arm with an NVIDIA GPU (CUDA 13.4.3). An unmodified ComfyUI v0.40.0 started each
server from the shipped profiles. Each result below has the same settings as a
TensorSharp CLI run of the same preset:

| Workflow | Result |
|---|---|
| LTX 2.5 text-to-video, 1024x1024x17 | MP4 byte-identical to the CLI's, video and audio; 103 s including the server start |
| LTX 2.5 from a first frame (Load Image), 49 frames, then a 17-frame clip from its `last_frame`, saved by Save Video | First clip byte-identical. The second starts on the first's last frame (37.1 dB PSNR, against 16.9 dB for unrelated frames) |
| LTX 2.5 from a last frame only (`fl2v`) | Byte-identical to the CLI's `--end-image` run |
| Qwen-Image-2.1 Turbo text-to-image, 1024x1024 | PNG byte-identical to the CLI's |
| Qwen-Image-2.1 Turbo edit of that picture | Byte-identical to a CLI edit of the RGB picture (see **RGB pictures** above) |

The behaviour around the generations was also checked:
- **Cancel.** Cancel at step 3 of 11 showed in ComfyUI within 0.6 s. The server ended the
  clip about 2.6 s later and kept serving.
- **Switching.** Switching LTX 2.5 to Qwen-Image to a chat model stopped each server
  cleanly (exit code 0). The Video node refused the chat model with a message naming it.
- **Stopping.** Stop Server after the edit freed the GPU.
- **Crash.** Force-killing ComfyUI took its server down within 0.1 s.
- **Linux.** In a `python:3.13-slim` container, a child started through `setpriv` died
  with its parent; without `setpriv` it survived.
- **Telemetry.** Every request reached Application Insights under its profile's
  `service_name`.

Not tested yet: macOS, Linux end to end on a GPU, the Wan presets, and other ComfyUI
versions.
