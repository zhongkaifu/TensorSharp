# LTX 2.5 Distilled

[English](ltx-2.5.md) | [中文](ltx-2.5_zh-cn.md)

TensorSharp implements native text-to-video and image-conditioned video with jointly
generated 48 kHz stereo audio for the LTX 2.5 Distilled checkpoint. The experimental TensorAgent entry is
**LTX 2.5 Distilled — UD-Q4_K_XL + BF16**; Download installs the five pinned base
files, and Use activates the video model. The optional Ingredients IC-LoRA enables
identity/appearance references and is included when TensorAgent's optional downloads
are enabled (the default).

## Files and startup

[`config/ltx-2.5-distilled.json`](../../config/ltx-2.5-distilled.json) pins the
denoiser, dedicated Gemma 4 text encoder/projections, convolutional video VAE,
x2 spatial latent upscaler and audio VAE by commit and SHA-256. They total
43,077,008,276 bytes (43.08 GB), all from the ungated
[Unsloth repository](https://huggingface.co/unsloth/LTX-2.5-GGUF), revision
`ac3c540c6025903784336e99b2dc179385c4c3b7`. No Hugging Face account is needed.
The companions are byte-identical to the Lightricks files; the config also lists
their gated originals as fallback URLs.

The config also lists the optional `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors`
at `${ltxRoot}/loras/`, pinned to Lightricks revision
`12040e4091ac2008d3906a594e31a7fb1ab9d546` (1,308,787,472 bytes; SHA-256
`ff873a5beada3c579a8137c7c53343916f78bcc9a529ba910143073fe8715e95`). This repo is
gated. Accept the LTX-2.x terms and configure `HF_TOKEN` or a Hugging Face token file
to download it. Config expansion fetches it before model loading when authorized and
continues without it otherwise; Video Studio then enables reference mode automatically
when the file is present.

Use `distilled/ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf`, installed directly in
`ltx25/` beside its companion folders. The repository root holds the **dev**
checkpoint, which needs a different sampler recipe (30 steps, CFG 3).
TensorSharp refuses dev-named files; renaming one does not turn it into a distilled model.
TensorAgent's current id is `ltx-2.5-distilled-q4kxl`. For the default model cache, launch migration preserves
the previous entry's complete companions and user-added adapter folders.

The config resolves files under `${TENSORSHARP_MODELS:-../models}/ltx25`.
Manual installations can instead set `TS_VIDEO_TEXT_ENCODER`, `TS_VIDEO_VAE`,
`TS_VIDEO_AUDIO_VAE` and `TS_VIDEO_UPSCALER` (CLI/server `--video-upscaler`); the
upscaler otherwise loads from `latent_upscale_models/` beside the denoiser and is
required. The custom encoder embeds its tokenizer; the ordinary Gemma
4 checkpoint is not a substitute. Use the **conv** video VAE named in the preset.
The publisher's diffusion decoder is optional; see [Video decoders](#video-decoders).

```sh
TENSORSHARP_MODELS=/path/to/models MLX_ENABLE_TF32=0 \
dotnet run --project TensorSharp.Cli -- \
  --config config/ltx-2.5-distilled.json --backend mlx \
  --prompt "A red toy car drives slowly along a quiet road in warm afternoon light." \
  --diffusion-seed 42 --output ltx25.mp4
```

Use `--backend cuda` or `ggml_cuda` on NVIDIA GPUs (the CUDA tensor executor).
`mlx` and `ggml_metal` select the MLX tensor executor on Apple Silicon. `cpu` and
`ggml_cpu` select the managed CPU tensor executor; GPU generation measurements do
not establish CPU throughput.
Vulkan and multi-GPU execution are unsupported. Desktop MP4 audio requires
FFmpeg on `PATH` or `TS_FFMPEG`; the CLI also writes the original PCM as a WAV.
The Mac app encodes audio/video through AVFoundation.

## Measured Unsloth preset (2026-10-18)

On a 48 GB M5 Pro, MLX with `MLX_ENABLE_TF32=0`, two-stage 1024×1024×17,
seed 42, the new UD-Q4_K_XL took 109.7 / 107.1 seconds for two different prompts
in one process, versus 115.3 / 115.2 seconds with the previous Abiray file.
Peak process footprint was 17.6 GB versus 27.0 GB. The mixed Q2_K/Q3_K/BF16
weights stay mapped and dequantize to F32 per projection; expanding them all at
load had caused a 56 GB footprint. This fix preserves every raw video and audio bit.

Against Unsloth Q8_0 as a higher-precision reference, the fox/lighthouse samples
scored 26.6/30.8 dB PSNR, versus 25.1/23.0 dB with the previous quantization.
These two samples support this preset choice, not a universal quality ranking.
A 1280×704×121 sample completed in 513.5 seconds plus 36.2 seconds of encoding,
with a 20.2 GB MLX allocation peak. CLI config downloads and the real TensorAgent
HTTP/SSE generation and media download paths were exercised. CUDA, iOS and the
native MAUI controls were not revalidated in this update. See the
[validation tools and limits](../../eng/validation/README-ltx25.md#unsloth-distilled-gguf-validation-2026-10-18).

## Generation recipe and limits

Generation follows the publisher's two-stage `DistilledPipeline`. Width and height
are the output size. The first pass samples the eight published sigmas at half
that size. The x2 spatial latent upscaler then doubles the latent, and three
refinement sigmas (0.909375, 0.725, 0.421875) run at full size. The refinement mixes
the upscaled latent with fresh noise at 0.909375 and re-applies each image guide,
re-encoded at full size; the audio latent is re-noised and refined with it. Each
pass draws its own noise, like the publisher's seed offsets (+10000 and +20000).
Ingredients identity sheets (`Mode = "ref"`) keep one full-size pass.

Both passes use CFG 1, with no flow-shift
override, sampled with ancestral Euler (`euler_ancestral`, eta 1). The publisher's
`DistilledPipeline` selects that sampler for every LTX 2.5 and newer checkpoint.
Each step re-noises the latent from a separate seeded stream before the next
prediction; the final step returns the denoised prediction. `--sampler euler`
selects deterministic Euler as an explicit diagnostic, and the log says it is not the
publisher's recipe. Defaults are provided by the model; other explicit sampler, step
or guidance requests fail. `steps: 8` in a request names the published first-pass
schedule. The preset and the model default are 1024×1024, 17 frames, 24 fps.
Width and height must be multiples of 64, so that each pass stays on the 32-pixel
latent grid, and frames must follow `8k+1`.
The experimental catalog tier is 48 GB. This is a conservative supported starting
point, not a proven minimum or a promise for arbitrary duration/resolution.

Image conditioning accepts a first frame, a last frame, both endpoints, or multiple
images at explicit output pixel-frame positions. `fl2v` pins the last frame and
optionally the first; a last frame alone becomes one appended guide at the final
frame (strength 1.0). `i2v` needs a frame-zero image and refuses an end image alone. Ordered reference-image lists
become evenly spaced keyframes; one image anchors frame zero. These references
control the timeline. Character/prop/location identity references use `Mode = "ref"` and the optional
Ingredients IC-LoRA described below. For video continuation, `ContinuationVideoPaths`
(or the HTTP `continueFromVideos` field) uses the exact last frame of the first clip as
frame zero; a second clip pins the new take's last frame. `ReferenceVideoPaths` (HTTP
`referenceVideos`) in `Mode = "ref"` instead supplies a full temporal reference clip.
Reference audio, retake/refinement, HDR/EXR, generated
keyframe slots and multi-GPU splitting remain unavailable. Disabling audio decode still runs the trained joint
AV denoiser.
The native audio length can be slightly shorter than the video duration; the
original decoder/export rule crops excess samples and does not pad short tracks.

Launch Apple processes with `MLX_ENABLE_TF32=0` for the Float32 denoiser/VAE
quality path. LTX sets this default before its first MLX operation, but upstream
MLX caches the choice when a process first performs GEMM; an already initialized
process needs the launch setting. TensorAgent's Mac app sets this default before
starting the host, so switching from another model preserves this precision.
Explicit user overrides are respected.

CUDA prompt projections use BF16 Tensor Core GEMM on compute capability 8.0 or
newer, with Float32 accumulation and the original BF16 activation boundaries.
Older GPUs use Float32. CUDA also reads each mapped projection range through
one bounded 8 MiB buffer per reader before upload; Linux skips resident ranges
and uses up to eight concurrent 1 MiB reads. This reduces
small page faults on network-backed model storage without loading the complete
checkpoint into host memory.

Every backend also reads weights ahead of their use, in the order the layers run,
within a 4 GiB window: the Gemma encoder, the prompt connectors and the first
CUDA denoiser upload. The upscaler, video decoder and audio VAE are read one
stage before their first use. Safetensors and GGUF files store tensors
alphabetically (layers 0, 1, 10, …). Without the read-ahead, a cold mapping
faults pages in at 0.9 GiB/s on an M5 Pro SSD, against 12 GiB/s for one
sequential reader. On an A40 pod's network volume it reaches 0.3 GiB/s,
against 2.3 GiB/s for sixteen concurrent readers. Chunks already in the file
cache are skipped.

When RAM, or the container's memory limit, is below the text encoder plus the
denoiser plus 16 GiB, the encoder's cached pages are released after each new
prompt, so they do not compete with the denoiser. A later new prompt reads them
again. `TS_VIDEO_READAHEAD=0` disables both the read-ahead and the release.
`TS_VIDEO_READAHEAD_WORKERS` (1–32) sets the readers: 16 on Linux, 1 elsewhere.
The conditioning log line reports the bytes read, the bytes already resident
and whether the cache is released.
CUDA self-attention tiles queries and heads within a 256 MiB score workspace,
including the 10,752-token Ingredients recipe. Each query attends to the complete
key sequence with pedantic Float32 GEMM; independent large-sequence numerical
checks cover tile boundaries and incomplete head groups.

## Clear faces and hands

Three earlier TensorSharp deviations from the publisher's pipeline contributed
to distorted faces and hands and have been corrected. These implementation fixes
do not guarantee anatomically correct output for every prompt, seed or duration.

1. **Sampler.** Generation used deterministic Euler where the publisher samples
   LTX 2.5 with ancestral Euler.
2. **Audio-video gate.** The audio-video cross-attention gates received sigma × 1,
   but this checkpoint's `config` metadata declares `av_ca_timestep_scale_multiplier`
   1000. The multipliers are now read from the GGUF `config` exactly as the original
   `LTXModelConfigurator.from_metadata` reads them. A GGUF without that metadata uses
   the published values and logs a warning. Earlier validation oracles built the
   official model with its keyword default (× 1), so they shared this deviation.
3. **One pass instead of two.** The publisher's `DistilledPipeline` plans motion at
   half the output size, upscales the latent x2 and regenerates detail in three
   refinement steps. TensorSharp ran a single full-size pass. The video VAE
   compresses 32×32 pixels into one latent token, so at 512×512 a ~70-pixel face is
   about 2×2 tokens. A hand crossing it then came out as a translucent smear, with
   eyes and mouth showing through the fingers. More pixels in one pass did not cure
   this: a single 1152×768 pass still smeared the hand over the face. The
   refinement pass, which restarts from the upscaled latent mixed with 91% noise,
   improved the earlier samples below.

Measured on an M5 Pro (MLX) with `imgs/banner_1.png` as the first frame, the
detailed Chinese prompt below, seed 42 and audio. These are historical sampled
visual observations, not an anatomy acceptance test:

| Recipe | Output | First pass | Frames | Observation |
| --- | --- | --- | --- | --- |
| One pass (before) | 512×512 | — | 97, 121 | Hand over the face is a translucent smear; eyes and mouth show through the fingers |
| One pass | 1152×768 | — | 97 | Sharper face; the hand is still translucent over it |
| Two-stage | 512×512 | 256×256 | 97 | Face cleaner, but the hand is a ghost: an 8×8-token first pass is too coarse to plan it |
| Two-stage | 1024×1024 | 512×512 | 97 | Coherent five-finger palm, crisp eyes and smile; faint traces of nostrils through the palm in a few frames |
| Two-stage | 1024×1024 | 512×512 | 121 | Opaque, well-formed hand that occludes the face; clean pursed lips and smile |

The 1024×1024 first pass draws the same noise from the same inputs as the old
512×512 single pass, so the improvement in that row comes from the refinement. The 121-frame two-stage
run took 1,600 s (first pass 401 s, upscale 5 s, refinement 924 s, decode 243 s),
with an MLX peak of 41.5 GB. The machine was thermally loaded after back-to-back
runs: the same 512×512 single pass measured 18.4 s per step cool and 63 s hot, so
treat these times as an upper bound. The upscaler adds seconds; the refinement
steps dominate.

Checks that ruled out other causes:

- **Denoiser numerics at full size.** A 48-block forward at 512×512×97 with 1,024
  context tokens matches the unchanged official PyTorch model with relative L2
  4.2e-5 (video) and 1.9e-6 (audio). Earlier parity runs covered only 64×64×9.
- **Decoder.** The publisher's diffusion decoder (`ltx-2.5-video-vae-bf16.safetensors`,
  a neighborhood-attention transformer) decoding the same 512×512 single-pass
  latents keeps the smear: the defect is in the latents. See
  [Video decoders](#video-decoders) for how the two decoders compare.

For clear faces and hands:

1. **Keep the first pass at 512 or more on each side.** That means an output of
   1024×1024 (the default) or larger. Width and height are output sizes and must be
   multiples of 64. A square request center-crops a landscape upload; a 3:2 source
   preserves its composition at, for example, 1536×1024, if memory allows.
   Larger outputs than 1024×1024×121 were not measured in the table above.
2. **Frame the face large.** An upper-body or close-up upload gives the face more
   latent tokens at any size.
3. **Describe the shot and the face.** Name the shot type, the face and the action.
4. Keep `euler_ancestral`, eight steps and CFG 1. Distilled LTX has no
   negative-prompt or guidance lever.

Temporal image preprocessing now includes the publisher's CRF 18 compression and
Float32 bilinear resize-before-crop, and fractional-strength refinement preserves
the noised prior-stage latent. Both Studio and chat use this shared pipeline.
For ordinary single first-image uploads, the first stage now uses strength 0.7
and refinement uses 1.0, following the publisher's ComfyUI recipe. Explicit
`ImageConditions` strengths, ordered/end guides and other modes retain their
requested strengths. Each stage logs its effective guide strengths.

In the reported 145-frame banner sample (seed `3206401772581117598`), every reviewed
first-second frame avoided the earlier blank-white-eye artifact after the
preprocessing correction. Strength 0.7/1.0 then substantially improved the hand:
frames 63–84 show five distinct, opaque digits. Finger/face blending remains near
frames 56–61 and 88–94, so this is not an artifact-free result. Switching decoders
alone did not correct the earlier full-strength guide's hand errors. Media
equality between Studio and chat is a separate check from anatomy quality.

Remaining differences from the publisher's default ComfyUI workflow: TensorSharp
decodes with the convolutional VAE by default (the diffusion decoder is opt-in),
passes the prompt through without the optional image-aware enhancer, and the
preset denoiser is Unsloth's mixed UD-Q4_K_XL quantization (the earlier captures
below used Abiray Q4_K_M). The new shorthand image
strengths match the [official ComfyUI image-to-video example](https://docs.ltx.io/open-source-model/usage-guides/image-to-video);
the other differences remain comparison dimensions, not proven fixes for extra
fingers. The GGUF publisher's own ComfyUI workflow uses the same model file.
Evidence is in `artifacts/ltx25/diffvae-diag/` and `docs/validation/ltx25/face-fix/`
(historical evidence), and `artifacts/studio-anatomy/` (the 145-frame follow-up).
Generated validation evidence is local and not committed.

## Video decoders

Lightricks publishes two decoders for the same latents. Both share one encoder:
the encoder tensors and latent statistics in the two files are byte-identical.

- **Convolutional** (`ltx-2.5-video-vae-conv-bf16.safetensors`, `--video-vae`): the
  default on every backend.
- **Diffusion** (`ltx-2.5-video-vae-bf16.safetensors`, `--video-diffusion-vae` or
  `TS_VIDEO_DIFFUSION_VAE`): the publisher's `NADiffusionDecoder`, opt-in on MLX and
  CPU. Four neighborhood-attention stages upsample the latent into a context volume.
  A fifth, diffusion stage predicts the pixels in one step (the checkpoint's
  `default_num_inference_steps` is 1) from seeded Gaussian pixel noise. CUDA has no
  neighborhood-attention kernel and refuses it. The load line names the decoder in
  use, for example `[ltx25] video decoder: diffusion (ltx-2.5-video-vae-bf16.safetensors)`.

On the banner sample (two-stage 1024×1024×97, seed 42), both decoders were scored
against the uploaded image, which conditions frame 0:

| Measure | Convolutional | Diffusion |
| --- | --- | --- |
| Frame 0 PSNR / SSIM against the source | 33.4 dB / 0.930 | 30.2 dB / 0.915 |
| Face edge energy (Laplacian variance; source 1.24e-2) | 1.37e-2 | 0.88e-2 |
| Frame-to-frame flicker (temporal high-pass, 8-bit levels) | 1.62 | 1.41 |
| MLX decode of 1024×1024×97 on an M5 Pro | 172 s, inside a thermally loaded generation | 99 s, standalone; MLX peak 17.2 GB |

The diffusion decoder renders cleaner frames, with no fine grain in flat areas and
less flicker, but it is softer: lashes, teeth and fabric edges lose about a third
of the source's edge energy. The convolutional decoder stays the default for that
reason. The 512×512 single pass shows the same: frame 0 is 27.7 dB against 26.9 dB, and
the diffusion decoder has 30% less edge energy than the source. These are one
subject and one seed; choose the diffusion decoder if grain or flicker matters more
than edge detail. The extra 1.47 GB file is not part of the preset or the
TensorAgent download.

TensorSharp follows the original's default (`CHUNKED_EAGER`) untiled decode: the
last latent frame is repeated twice for the trailing window and cropped again, and
latents smaller than the windows are edge padded. Stage-five attention reproduces
the original's width slabs, whose halos at the volume edges are edge-replicated
with continuing rotary positions. Activations stay Float32 and BF16 weights widen
exactly. Attention runs on one frame-sized tile of queries at a time against
cached keys, so no full-volume attention operand exists. The MLX kernel stages
each key row once per 8×8 query tile in threadgroup memory: the 1024×1024×97 decode
went from 484 s (one thread per query) to 99 s, with outputs within 1.3e-7.

## First, last and multiple reference images

Uploading one image in TensorAgent or the server video UI animates it as the first
frame. The prompt describes the desired motion; for example,
`请让图中的女人来个飞吻` with `imgs/banner_1.png`. Uploading multiple images treats
them as ordered keyframes, from the first frame to the last. Use explicit image
conditions when the images belong at particular times or require different
conditioning strengths.

The model API offers the following inputs:

| Input | Timeline meaning |
| --- | --- |
| `Image`, `ImageBytes`, `ImagePath` | First frame; decoded image takes precedence over bytes, then path. |
| `EndImage`, `EndImagePath` | Last output frame. |
| `ReferenceImagePaths` | Ordered, evenly spaced keyframes by default; identity sheet panels in `Mode = "ref"`. Cannot be combined with the other image inputs, except one frame-0 `ImageConditions` guide in `Mode = "ref"` ([starting on a first frame](#starting-on-a-first-frame)). |
| `ImageConditions` | `VideoImageCondition` entries with an image, `FrameIndex`, and `Strength`. `-1` selects the last frame. |
| `ContinuationVideoPaths` | One or two video files. Their exact final frames pin the new take's first frame and, optionally, its last frame. |

```csharp
var request = new VideoGenerationParams
{
    Width = 512, Height = 512, Frames = 17, Fps = 24, Seed = 42,
    ImageConditions = new[]
    {
        new VideoImageCondition { ImagePath = "first.png", FrameIndex = 0 },
        new VideoImageCondition { ImagePath = "middle.png", FrameIndex = 8, Strength = 0.8f },
        new VideoImageCondition { ImagePath = "last.png", FrameIndex = -1 },
    },
};
var video = model.GenerateVideo("The woman smiles and blows a kiss.", request);
```

Each frame position must be unique and within the requested clip. Strength is
finite and between zero and one, with one preserving the encoded guide latent.
Explicit `ImageConditions` keep their strengths in both stages. An ordinary
single `Image`, `ImageBytes` or `ImagePath` upload in inferred/`i2v` mode instead
uses the model's first-stage default of 0.7 and restores strength 1.0 in refinement.
Temporal image guides follow the publisher's LTX 2.5 preprocessing: crop an odd
bottom/right edge to even dimensions, recompress through libx264 at CRF 18 with
the `veryfast` preset and `yuv420p`, then resize the full image with Float32
bilinear interpolation (`align_corners=False`) and center-crop to each stage's
size. The intermediate resize retains fractional pixels. RGB/YUV conversion and
slice threading match PyAV's settings; exact lossy-codec pixels can vary with
the installed x264 version and CPU thread count.

Desktop temporal image conditioning requires FFmpeg with **libx264** on `PATH`
or at `TS_FFMPEG`. Capability discovery checks for that encoder; a missing or
unsupported codec fails before image encoding and denoising. There is no OS-codec
fallback for conditioning. Managed and Apple-only media providers do not yet
implement this transform, so they do not advertise temporal image conditioning;
text-to-video and the separate Ingredients identity-reference path are unaffected.
The CPU resize oracle and an optional local PyAV codec comparison are covered by
`Ltx25ImagePreprocessingTests`; regenerate the local reference with
`eng/validation/generate_ltx25_image_preprocessing_reference.py` and set
`TS_LTX25_IMAGE_ORACLE` to its output path to run the codec comparison.

At most 16 images are accepted. More guides increase the denoiser token
count, memory use, and runtime. The decoded first frame is a VAE reconstruction,
so exact pixel equality with the uploaded file is not expected.

The implementation follows the publisher's combined conditioning: frame zero
replaces the first latent frame; later guides append clean tokens with their
pixel-time intervals. The first canvas latent frame always receives the model's keyframe marker,
including text-only generation. Appended supplied-image guide tokens remain
unmarked; generated keyframe slots are a separate feature. See the pinned original
[combined image helper](https://github.com/Lightricks/LTX-2/blob/be6d6d0a69d11a0a399c462ed92606692e5f8697/packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py)
and [keyframe conditioning](https://github.com/Lightricks/LTX-2/blob/be6d6d0a69d11a0a399c462ed92606692e5f8697/packages/ltx-core/src/ltx_core/conditioning/types/keyframe_cond.py).

## Identity and appearance reference sheets

The publisher's [Ingredients adapter](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients)
uses a composite reference sheet to preserve characters, props, and locations.
Install `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors` in the denoiser's sibling
`loras/` directory, or set `TS_LTX25_INGREDIENTS_LORA` to its path. The shipped
config downloads it there before the first model load when Hugging Face access is
approved. Accept the gated repository terms and configure `HF_TOKEN` or a Hugging
Face token file; see the [official Lightricks Ingredients repo](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients).
The adapter is 1,308,787,472 bytes with SHA-256
`ff873a5beada3c579a8137c7c53343916f78bcc9a529ba910143073fe8715e95`.
TensorAgent downloads the optional adapter by default when access is approved and
offers **Identity references (Ingredients)** after loading it. The ordinary upload
flow continues to use timeline image guides. A running server reads this config at
startup, so apply the updated preset on the next server launch; after that, reference
images are available on the initial model load.

```csharp
var request = new VideoGenerationParams
{
    Mode = "ref", Width = 768, Height = 448, Frames = 121, Fps = 24, Seed = 42,
    ReferenceImagePaths = new[] { "character.png", "outfit.png" },
};
var video = model.GenerateVideo(
    "The woman smiles, touches her fingertips to her lips and blows a kiss toward the camera.",
    request);
```

One image becomes the complete reference sheet. Multiple images form ordered
panels on a black background. Each complete image is fitted into its cell without
cropping or stretching; transparency is composited on black. The encoder
processes a static RGB reference video on the same temporal grid as generation,
and all encoded reference frames are appended as clean guide tokens. The
Ingredients adapter applies its separate low-rank updates to the AV transformer;
it does not replace the generated first frame. The trained default is
768×448, 121 frames, 24 fps, the eight distilled sigmas at CFG 1. At least 121 frames are
required. Smaller dimensions are diagnostics rather than evidence of the trained
reference recipe's quality.

Ingredients prompts use a reference-sheet description followed by a generated
video description. TensorSharp retains an ordinary prompt verbatim under those
labels and adds only the known sheet panel inventory. Explicit structured prompts
are preserved. Describe the subjects and requested action clearly; preserving
appearance and following the requested motion are separate quality checks.

One full reference clip can also be supplied with `ReferenceVideoPaths` in
`Mode = "ref"`. Studio exposes existing project clips as references when the
Ingredients adapter is loaded. The clip is encoded as temporal reference latents;
it is not reduced to a still. To continue from a clip's ending frame, use
`ContinuationVideoPaths` (HTTP `continueFromVideos`) instead. Studio keeps that
continuation action separate because the validated Ingredients workflow does not
reliably keep a pinned first frame while reference conditioning is active.

For better reference sheets, use clean subject or object images without text,
logos or unrelated collage elements. Describe each panel by its actual position
and explain which views show the same identity. For example:

```text
Reference sheet: Left panel shows a bust view of Maya in a blue embroidered
outfit. Right panel shows a close-up of the same Maya, with the same hairstyle
and earrings.

Generated video: One continuous full-frame close-up of Maya in a blue starry
setting. She touches her fingertips to her lips, then extends her palm toward
the camera to blow a kiss.
```

For multiple characters or props, name them by panel position before describing
their interactions. These publisher-oriented guidelines do not guarantee motion
or composition. A same-seed test with a branded banner still copied the panel
layout after the caption was clarified. The same seed also retained two panels
after both inputs were changed to text-free bust/face crops and the caption was
made position-aware. One reference image gives more predictable composition;
multiple reference images can still copy the reference sheet layout.

The official family also includes MotionTrack, Union depth/canny/pose, layout,
inpainting, HDR and AlphaGen workflows. These require their specialized pipelines,
adapters or decoders and are not generic photo-reference inputs. Native
audio-to-video is a base-model capability; it does not inherently require another
LoRA, but its input path remains unimplemented here. Reference video, retake and
generated keyframe workflows also remain unsupported. See the official
[conditioning documentation](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/docs/conditioning.md)
for these distinctions.

### Starting on a first frame

An Ingredients request can also start on a pinned frame: the composition the
publisher's IC-LoRA pipeline runs when it is given images and reference
conditioning together. Video Studio does **not** offer it (see the validation
result below and [Keep a character across segments](../video-studio.md#keep-a-character-across-segments));
it is reachable through the video API. Add one `ImageConditions` entry at
`FrameIndex = 0`. In `Mode = "ref"`, `Image`
and `ImagePath` remain a single identity image, not a frame.

```csharp
var request = new VideoGenerationParams
{
    Mode = "ref", Width = 768, Height = 448, Frames = 121, Fps = 24, Seed = 42,
    ReferenceImagePaths = new[] { "character.png" },
    ImageConditions = new[]
    {
        new VideoImageCondition { ImagePath = "previous-last-frame.png", FrameIndex = 0, Strength = 1 },
    },
};
```

The recipe stays one full-size pass. The guide goes through the same CRF 18
preprocessing as the other image guides, so it needs FFmpeg with libx264, and is
encoded once at the output size. It is pinned at its strength: canvas latent
frame 0 takes the clean guide latent with denoise mask 1 − strength, adding no
tokens. The reference sheet's tokens are appended after the canvas as before.
Because the guide and the references occupy disjoint tokens, applying the guide
after the references equals upstream's images-then-references order. The
upstream equivalent is
`ICLoraPipeline(stages=(ICLoraStageConfig(1, True, DISTILLED_SIGMA_VALUES),))(images=[(frame, 0, 1.0)], video_conditioning=[(sheet, 1.0)])`.
At most one guide is accepted, and only at frame 0. A last-frame image or timed
guides beside references are refused: no publisher workflow uses them. The
denoise log line reports the guide (`first-frame guide strength: frame 0=1`).
The composition matches the unchanged upstream functions bit for bit at strength 1
(`FrameZeroGuideBesideReferenceVideoMatchesUnchangedOfficialImagesThenReferences`,
fixture from `eng/validation/generate-ltx25-reference-first-frame-oracle.py`).

The publisher trained Ingredients with `first_frame_conditioning_p` 0.0, so the
adapter never saw a pinned first frame in training, and the publisher has not
validated this use. On 2026-10-17 (Q4_K_M, MLX, 768×448×121, seed 42) every take
that started on a frame cut away from it at its second frame: frame 0 matched the
pinned frame (27–29 dB PSNR), frame 1 was 9.6–9.7 dB from frame 0 and showed the
reference sheet's composition, with a portrait reference letterboxed on the sheet
and with a full-width one alike. A plain Ingredients take with the portrait
reference reproduced the letterboxed sheet too. See the
[validation record](../video-studio-validation.md#video-chains-and-character-references-2026-10-17).

## Native graph and memory

[`Ltx25PromptEncoder`](../../TensorSharp.Models/Models/Ltx25/Ltx25PromptEncoder.cs)
runs the authentic 48-layer Gemma 4 Unified architecture and all 49 hidden states,
preserving BF16 arithmetic boundaries, partial global RoPE and original padded
positions. V2 feature normalization and learned aggregate projections feed eight
video and eight audio connector blocks. Learned registers make the complete
1024-token contexts valid; removing padding positions would change conditioning.

LTX 2.5 runs on the shared [diffusion runtime](../design/diffusion-runtime.md#312-ltx-25-on-the-runtime-as-built):
[`Ltx25Family`](../../TensorSharp.Models/Models/Ltx25/Ltx25Family.cs) drives
`DiffusionVideoPipeline` through both passes, and
[`Ltx25Denoiser`](../../TensorSharp.Models/Models/Ltx25/Ltx25Denoiser.cs) runs
all 48 AV transformer blocks at every sigma as one graph program a step on the
backend's tensor executor. The pipeline then runs the convolutional video
decoder and the audio VAE, BigVGAN vocoder and learned bandwidth extension. [`Ltx25LatentUpsampler`](../../TensorSharp.Models/Models/Ltx25/Ltx25LatentUpsampler.cs)
is the publisher's `LatentUpsampler` (1024 mid channels, four ResBlocks before and
after a Conv2d + PixelShuffle(2)): Conv3d layers zero-pad every axis, GroupNorm(32)
statistics span the whole clip and accumulate in Float64, and BF16 weights widen
exactly to Float32. It runs between the passes on the stage-one latent, which is
small, so it adds seconds rather than minutes.
The pipeline uses the original ties-to-even audio token rounding. SD.cpp uses
ceil; matched 17-frame/24-fps validation has 18 audio tokens in both.

Prompt weights stream per projection and video VAE weights per convolution. The
model retains mapped prompt/video checkpoints, denoiser weight/cache identities
and the audio decoder’s converted weight caches across requests. Connector device
weights are released after each new prompt, and at most four final prompt context
pairs are cached. `TrimIdleMemory` releases all retained components and unused
device allocations. This avoids repeated checkpoint opens and weight conversion
while keeping subsequent uncached prompts bounded.

## Validation

### Diffusion runtime migration (2026-11-27)

LTX 2.5 and its Ingredients IC-LoRA run only on the shared diffusion runtime
([design](../design/diffusion-runtime.md#312-ltx-25-on-the-runtime-as-built)):
`Ltx25Family` drives the video pipeline's two passes (or the single Ingredients
pass), `Ltx25Denoiser` runs each step as one graph program on the backend's
tensor executor, and `Ltx25LoraSet` is the Ingredients adapter as the runtime's
adapter set, applied unmerged in exact F32. The retired `Ltx25Pipeline`,
`Ltx25Sampling`, the direct denoiser and `Ltx25LoraAdapter` were deleted after
their outputs were frozen.

Evidence, on an NVIDIA A40 (48 GB, driver 595.112, CUDA 13.2, .NET 10.0.404) and an Intel Xeon
Gold 6342 in a 50 GB container, ggml `bb877c0c` unchanged, the retired path built
from `c178bfc` with the same ggml and checkpoint (Unsloth UD-Q4_K_XL):

- **Denoiser steps, frozen.** The retired denoiser's outputs on the synthetic
  checkpoint (the official oracle's four cases, each also with a synthetic
  rank-2 Ingredients adapter over every adapted projection) are frozen per lane
  in `InferenceWeb.Tests/Fixtures/Ltx25/LegacyGoldens/`. The runtime gives their
  bits on both lanes (CPU and CUDA, 16 outputs each), and
  `Ltx25DenoiserTests.StepsGiveTheRetiredDirectDenoisersBitsOnItsLane` holds it
  there.
- **Real-checkpoint steps** (synthetic latents and contexts at generation
  geometries; a synthetic rank-128 BF16 adapter with the Ingredients layout over
  all 480 projections, because the gated adapter was not available): bit-identical
  on CUDA for a 512×512×17 first pass, an image guide's per-token timesteps, a
  1024×1024×17 refinement, 384×256×49 with appended keyframe guides and a
  768×448×121 Ingredients pass. Steady-state steps: 820/825, 897/893,
  2954/2954, 919/931 and 21838/21919 ms (retired/runtime). A clip's first step
  after the device copies were released: 2.45–2.63 s retired, 2.37–2.49 s runtime
  (first pass); 4.65–4.89 s and 4.51–4.72 s (refinement). On the CPU (two
  blocks) both cases were bit-identical, at 2829/2548 and 3478/3154 ms.
- **Whole clips** through the public model API (`Ltx25AbProbe`), raw RGB and both
  PCM channels compared: bit-identical in every round of every case. Wall times in
  alternating rounds with prebuilt binaries (the final code, two rounds each):

  | Clip (A40, `cuda`) | Retired | Runtime | Runtime / retired |
  |---|---:|---:|---:|
  | t2v 1024×1024×17 | 33.86, 32.43 s | 32.39, 32.08 s | 0.973 |
  | i2v 1024×1024×17 | 32.26, 31.87 s | 32.03, 31.96 s | 0.998 |
  | keyframes 768×512×49 (three guides) | 34.89, 34.68 s | 34.68, 34.61 s | 0.996 |
  | t2v 1024×576×121 | 91.32, 89.98 s | 90.39, 90.13 s | 0.996 |
  | Ingredients 768×448×121 (two references) | 203.49 s | 204.03 s | 1.003 |

  Every phase matches or is faster (text and image encodes, both passes, the
  upsampler and the decodes). Three changes made that so: projections upload at
  their first product, overlapping the device work queued before them, as the
  retired denoiser did; Philox noise and the rotary tables are drawn on several
  cores (same bits); and the Ingredients update is added into each projection in
  place. Earlier rounds, before those changes, ran up to 1.8% slower.
- **Memory.** Peak live device memory of a step is lower: 2,063 vs 2,345 MiB
  (refinement), 6,192 vs 6,684 MiB (Ingredients), weights aside. The CUDA pool
  keeps 0.3–0.4 GB more freed blocks cached at that peak, which it returns when an
  allocation would otherwise fail. So nvidia-smi read 23.9 vs 23.4 GB during an
  Ingredients pass, while both trees peaked at 18.2–22.9 GB elsewhere, the 121-frame
  decode's transient. Host memory of a two-clip process: anonymous 3.56 vs 3.61 GiB,
  file-backed 38.31 GiB both.
- **CPU.** A 256×256×9 clip on the Xeon (7.65-core quota) was bit-identical in
  two pairs run in opposite orders: 411.7/405.3 s retired, 415.5/403.9 s runtime
  (1.003× over both).
- **Tests.** The LTX, diffusion-runtime, graph and LoRA filters passed: 1,637 on
  the CPU (34 skipped) and 1,628 on CUDA (34 skipped). The skips are MLX- and
  Metal-only tests, GGML-backend region tests, and the oracles that need the
  gated adapter, the tokenizer reference or the PyAV image reference.

Not validated: MLX and Metal hardware (no Apple device was available; the MLX
executor's regions run their reference ops where it has no kernel); the real
gated Ingredients adapter (the synthetic adapter has its layout, shapes and
dtype); other GPUs and operating systems; perceptual quality beyond bit
identity, which makes the outputs the same clips.

Removed with the direct denoiser (the server and CLI refuse each at startup,
naming what replaced it): `TS_LTX25_CUDA_ATTN`, `TS_LTX25_CUDA_FUSED_HEAD_GATE`,
`TS_LTX25_FUSED_ROPE`, `TS_LTX25_CUDA_TILE`, `TS_LTX25_CUDA_REQUIRE_TILE`,
`TS_LTX25_CUDA_F16_DEQUANT`, `TS_LTX25_STAGE_TIMING`, `TS_LTX25_CUDA_SOURCE_UPLOAD`,
and the prompt encoder's `TS_LTX25_CUDA_PREFAULT`, `TS_LTX25_CUDA_PREFAULT_WORKERS`,
`TS_LTX25_DEVICE_PROMPT_MATH`, `TS_LTX25_CUDA_BF16_GEMM` and
`TS_LTX25_MLX_BF16_GEMM`. Each kept the default it had, so outputs are
unchanged. `TS_LTX25_INGREDIENTS_LORA` stays. The latent probe's
`--first-stage-image-strength` and the prompt probe's switch-alternating round
modes were removed with them (`Ltx25PromptProbe --rounds N` repeats an encode).

### Latest exact performance work

The 2026-10-17 work removed weight loading and host overheads from the default
1024×1024×17 recipe (seed 42). Every output is unchanged. The M5 Pro rows use
the end-to-end probe, comparing raw Float32 RGB and both PCM channels with
`ltx25-compare-native-generations.py --require-bitwise`. The A40 rows use
complete CLI processes in two alternating rounds; their decoded RGB and PCM
were identical in all four runs. Model files started cold on the M5 Pro and
were uncontrolled on the A40.

| Host / backend | Before | After | Output |
|---|---:|---:|---|
| M5 Pro `mlx`, new process | 168.9 s | 108.0 s | Bitwise RGB and PCM |
| M5 Pro `mlx`, a second, uncached prompt in the same process | 142.8 s | 110.3 s | Bitwise RGB and PCM |
| A40 `ggml_cuda` (direct CUDA), CLI process | 163.9 s | 72.9 s | Identical decoded RGB and PCM |

The changes:

- **Read-ahead in compute order** (see [Video decoders](#video-decoders) and
  the weight-reading paragraph above). On the M5 Pro, text-encode went from
  42.3 to 5.5 s in a new process and from 30.6 to 5.2 s for a second prompt.
  On the A40 pod's network volume it went from 61.7–65.2 to 13.0–13.5 s. The
  A40's first-pass denoise, which includes the first denoiser upload, went from
  39–44 to 15–19 s.
- **Prompt connectors.** RoPE tables are now computed once with the same
  expressions, and the connector weights are read ahead in layer order. On the
  A40 the connectors went from 12.3 to 1.8 s.
- **Video VAE and upsampler weights.** Weights are widened across cores, and
  CUDA host buffers are written without first reading back their never-written
  device copy. On the A40, upsampling went from 9.0–11.6 to 2.5 s and video
  decoding from 12.7–15.4 to 6.9 s; on the M5 Pro, upsampling went from 2.3 to
  0.5 s.
- **Gemma cache release.** On a 48 GB Mac, releasing Gemma's cached pages
  after each new prompt keeps 22 GiB of stale pages from slowing the first
  1024×1024 denoise step. Together with the read-ahead, the M5 Pro's first-pass
  denoise in a new process went from 64.8 to 43.5 s.
- **Weight-widening loops.** These now compile straight to optimized code and
  use block copies.

The previous exact performance work: current source runs Gemma prompt BF16 rounding and split-half RoPE on CUDA/MLX
devices while retaining the original host-computed trigonometric values and
arithmetic order. CUDA also fuses the existing per-head `2 * sigmoid(logit)`
gating without changing Float32 rounding boundaries. Lossless MLX Q5_K loading
packs rows directly with bounded CPU parallelism; `TS_MLX_Q5K_REPACK=0` retains
the scalar control. The switches that selected the previous kernels were
removed with the direct denoiser (see
[Diffusion runtime migration](#diffusion-runtime-migration-2026-11-27)).
These changes preserve quantized weights, BF16 activation boundaries and the
complete generation recipe. Component tests require native dispatch and exact
outputs; older optional PTX can fall back to the host path, so generation alone
does not prove dispatch.

An experiment extending CUDA batched attention to extra rectangular shapes
passed component checks but failed complete-generation RGB and stereo PCM
equivalence. That extension and its selector were removed. With only the
accepted prompt math and head-gating changes active, all three tested remote
generation cases again produced bitwise-identical RGB and both PCM channels.
Component timing and packer microbenchmarks are not overall model speed claims;
the [validation workflow](../../eng/validation/README-ltx25.md) defines the
separate numerical, media and timing gates. Skipped GPUs do not count as coverage.

### Recorded model and application coverage

Reusable tools and commands are described in
[`eng/validation/README-ltx25.md`](../../eng/validation/README-ltx25.md). Independent
fixtures and actual-weight checks compare prompt stages, video decoder stages and
the complete audio decoder against the original implementation. Video/audio
decoder arithmetic uses exactly widened BF16 weights and Float32 activations;
those checks do not claim bitwise BF16 activation parity. Full pretrained prompt
conditioning has small backend-dependent BF16 reduction differences.

Complete generation probes separately cover the registered model, MP4 soundtrack
export, TensorAgent activation and HTTP/SSE/download routes. Benchmark tools link
an unchanged clean stable-diffusion.cpp export and record its source/library
identity. Generated evidence stays in ignored `docs/validation/ltx25/` and
`artifacts/ltx25/`. Unavailable devices, skipped scenarios and failed attempts are
recorded separately from passing coverage; short clips do not establish broad
perceptual quality or performance equivalence.

Image-conditioning validation uses unchanged ggml
`3612c0af93f2cfc8f048295300a61187723b12ae` and the publisher's reference
`be6d6d0a69d11a0a399c462ed92606692e5f8697`. The unchanged SD.cpp comparison
is `d637cfb935feb2254898a8f1524046c958f3df80`, with its ggml
`d25b121ce1d6ae0c7e62ad080460d9fc1e46dd93`. Source/build manifests, actual
model/binary hashes, Linux file-page residency, failed attempts and unavailable
scenarios are retained with the generated evidence.

The real image tests cover first-only, last-only, both endpoints, and three
uploaded image guides through TensorAgent HTTP/SSE and normal chat on Apple
Silicon and an NVIDIA A40. The trained 512×512 image encoder matches the
unchanged official Float32 latent reference within relative L2 error `4.2e-6`
on both GPUs. These numerical checks establish encoder fidelity, while
full-generation samples separately establish media and application integration.
The following historical timing samples use the earlier 512×512 single-pass
recipe, before the ancestral-sampler/two-stage fixes; they do not benchmark the
current default. Warm same-prompt image generation was faster than the measured
SD.cpp baseline, while fresh-process CUDA loading remained slower. Apple
Silicon 17-frame first-image generation took 126.85 seconds first and 39.81
seconds repeated, versus SD.cpp's 221.34 and 160.63 seconds. A40 measurements
were 90.47/12.22 seconds versus SD.cpp's 56.67/43.43 seconds. These are measured
model-generation times for one image, prompt, seed and geometry; file residency,
media encoding and longer clips have separate costs.

An opt-in Linux CUDA source-upload path (cold mapped ranges through pinned
buffers) improved one reversed-order A40 first-generation pair by 8.45% but was
never made the default; it was removed with the direct denoiser. Weights are
uploaded from the mapped checkpoint after the file-cache read-ahead.

Matched 97-frame clips with an explicit English blowing-kiss prompt visibly
perform the hand-to-mouth and outward gesture in both engines; the short Chinese
prompt does not establish that gesture in either engine. A separate real
TensorAgent/server upload with the detailed Chinese prompt below, 97 frames at
24 fps and seed 42, visibly touches fingertips to lips, extends the palm toward
the camera, and lowers the hand while smiling. That Apple Silicon HTTP/SSE run
completed in 263.93 seconds with H.264 video and generated audio:

```text
请让图中的女人来个飞吻。她面向镜头微笑，先将右手指尖轻触嘴唇，噘嘴，再把手掌向镜头伸出，做一个清晰的飞吻动作，然后放下手并微笑。保持人物的面容、发型和服装，镜头稳定。
```

Use the normal image-to-video mode with the banner as its first image. That
512×512 sample predates the fixes described under
[Clear faces and hands](#clear-faces-and-hands): it ran one deterministic Euler pass,
and its face smears where the hand crosses it and drifts in later frames. This single subject,
seed and prompt does not establish broad perceptual parity. Ingredients identity
validation is recorded separately because the pinned SD.cpp baseline does not
implement that adapter.

Ingredients validation exercises the trained 768×448×121 recipe with one image
and two views of the same subject on both GPUs, including real TensorAgent
uploads and normal chat. The single-reference samples retain the subject and
complete the requested kiss. Multiple-reference samples retain recognizable
appearance, but observed failures include copying the sheet panel layout and
changing camera framing. Multi-character fidelity and broad prompt adherence
are not established. The strict adapter checks validate all 480 factor pairs
and actual rank-128 attention and feed-forward updates; these numerical checks
do not remove the model's perceptual limitations.
