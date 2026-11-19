# MiniMax-H3-Image

[English](minimax-h3-image.md) | [中文](minimax-h3-image_zh-cn.md)

## License and legal disclaimer

> **Important — MiniMax-H3-Image is subject to a separate third-party license.** Before downloading, running, deploying, modifying, redistributing, or using MiniMax-H3-Image or its outputs with TensorSharp, you must review and comply with the [MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE), its Acceptable Use Policy, and all applicable laws and regulations.
>
> The community license expressly excludes the **United States, European Union, United Kingdom, and Republic of Korea (South Korea)** from its licensed territory. It does **not** authorize covered use in those territories. It also requires separate written authorization from MiniMax when the annualized revenue of you and your affiliates exceeds USD 1 million or when you provide hosted services to third parties, and it prohibits using outputs to improve other AI models. Consult the official license for the full terms and any changes.
>
> TensorSharp is an independent inference engine. Its technical compatibility with MiniMax-H3-Image does **not** grant a license, waive restrictions, or imply authorization or endorsement by MiniMax. **You are solely responsible for determining whether your intended use is permitted and for all risks, consequences, claims, penalties, damages, and other liabilities arising from your use or noncompliance. To the fullest extent permitted by applicable law, TensorSharp, its authors, contributors, and maintainers disclaim any responsibility or liability for your use of MiniMax-H3-Image, its outputs, or any violation of its license or applicable law.** This notice does not override the MiniMax license or limit liability where such limitation is prohibited by law.

MiniMax-H3-Image is MiniMax's text-to-image and image-editing model built from the MiniMax-H3 video
model: the same 33B single-stream DiT (50 blocks, hidden 5376, 56 heads of 128) with
its audio stream removed, the same Qwen3-VL-32B text encoder (layer-50 hidden
states), and a new image VAE whose 2-D ViT decoder reads the H3 latent space. It is
distilled to **six Euler steps with guidance 1**: the final layer carries one output
head per step of the checkpoint's own sigma grid, and there is no negative prompt.
It works from about 4 MP to 20 MP and is best around 16 MP (4096x4096, 4736x3552,
5024x3360, 5472x3072, 6272x2688).

TensorSharp runs text to image and editing (up to 16 reference pictures, [Editing](#editing))
on its shared diffusion runtime (`TensorSharp.Models.SharedDiffusion`: the MiniMax-H3-Image
family, its text encoder with the vision tower, denoiser and VAE encoder and decoder, run
phase by phase by one pipeline) on `ggml_cuda`, `ggml_metal` and `ggml_cpu`, end to end. It
has been run with real weights on `ggml_cuda` (an H100 and an A40) and on `ggml_metal` (an
M5 Pro); `ggml_cpu` runs the same programs (validated on small synthetic checkpoints; with
real weights it takes tens of minutes a picture). Masked edits are not implemented yet.

| Component | File | Source |
|---|---|---|
| Diffusion transformer (bf16, as released) | `minimax_h3_image_pruned_bf16.safetensors`, 40.2 GB | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/diffusion_models) |
| Text encoder (shared with MiniMax-H3) | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf`, 16.97 GiB, with `vocab.json`, `merges.txt`, `tokenizer_config.json` | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF), [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor) |
| Image VAE | `minimax_h3_image_vae_fp16.safetensors`, 1.17 GB | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/vae) |

The denoiser is read from the safetensors file directly, in its original tensor
names: there is no GGUF to make. A GGUF holding the same tensors (for a smaller
quantization) loads as well; it is recognised by its tensors (the H3 DiT with no
`audio_patch_proj`). That includes the GGUFs ComfyUI-GGUF's converter writes, such
as [unsloth/MiniMax-H3-Image-GGUF](https://huggingface.co/unsloth/MiniMax-H3-Image-GGUF):
they declare `general.architecture = hyvid`, a label that never names this model, so
the tensors decide. ComfyUI's own quantized repacks (`*_int8_convrot`, `*_w4a8`)
store the projections in formats ggml has no type for and are refused with a
message naming the bf16 file.

## Run

```bash
TensorSharp.Cli --config config/minimax-h3-image.json \
  --prompt "A red paper lantern glowing in a snowy courtyard at dusk, a wooden sign that reads \"冬至\"" \
  --width 4096 --height 4096 --diffusion-seed 7 --output lantern.png

TensorSharp.Server.Host --config config/minimax-h3-image.json
```

[`config/minimax-h3-image.json`](../../config/minimax-h3-image.json) downloads the
three files on first use (about 58 GB) and pins their revisions and SHA-256. The
text encoder is the file [`config/minimax-h3-fl2va.json`](../../config/minimax-h3-fl2va.json)
downloads, so the two configurations share it. The text encoder and VAE are found
beside the denoiser or passed with the MiniMax-H3 companion flags,
`--video-text-encoder` and `--video-vae` (`TS_VIDEO_TEXT_ENCODER`, `TS_VIDEO_VAE`).

The server serves it on the image routes (`/api/image-generate`, `/api/image-edit` and their
`/stream` forms) like Qwen-Image-2.1, and on the OpenAI-shaped `/v1/images/generations` and
`/v1/images/edits` in the shapes SGLang serves (MiniMax's `examples/t2i.py` and
`examples/edit.py` work unchanged against them; [API examples](../../TensorSharp.Server.Host/API_EXAMPLES.md#openai-image-routes-v1imagesgenerations-v1imagesedits)).
The ComfyUI node pack has a
launch profile for it, `minimax-h3-image` (`examples/comfyui/profiles/`), for the
**TensorSharp Image** node, and `minimax-h3-image-int8` with int8 projections
(`TS_H3_INT8_MATMUL=1`, [Performance](#performance)).

Request settings:

- **Size**: 2048x2048 when a request names none (what ComfyUI's template opens at);
  sides must be multiples of 32.
- **Steps**: 6 by default, the distilled count. Other counts run (a step spanning
  several grid intervals takes the dt-weighted mean of their heads, as ComfyUI
  does) but are not what the checkpoint was trained for.
- **Guidance**: fixed at 1. A `cfg` above 1 or a negative prompt is refused.
- **Seed**: the initial latent is drawn exactly as ComfyUI's KSampler draws it
  (CPU `torch.randn` with `torch.manual_seed(seed)`), so the same seed starts from
  the same latent in both.

## Editing

```bash
TensorSharp.Cli --config config/minimax-h3-image.json \
  --image studio.png --image poster.png --width 2048 --height 2048 --diffusion-seed 1234 \
  --prompt "Hang <Picture 2> as a framed print on the wall behind the two hosts of <Picture 1>." \
  --output edit.png
```

An edit takes up to 16 reference pictures (`--image`, repeated; `imagePaths` or the
`image` parts on the server). The prompt names them `<Picture 1>`, `<Picture 2>`, ... in the
order they are given, and is used verbatim. TensorSharp follows MiniMax's own pipeline as
diffusers implements it (main d5df05ed), which conditions on every picture twice:

- **Copies.** Each picture is read upright (EXIF orientation applied) as RGB. A side
  under 32 pixels is refused. The *VAE copy* keeps the picture when its short edge is
  1024 to 2048 pixels, otherwise it is scaled to the nearer bound with both sides rounded
  to 32; the *vision copy* is scaled to a short edge of exactly 1024 and then fitted by the
  Qwen3-VL processor's rule (sides multiples of 32). Both use Pillow's Lanczos on 8 bits
  (the processor's own step uses Pillow's bicubic), byte for byte. References are never
  resized to the output.
- **Text encoder.** For each picture the label `Picture i: `, then its vision block
  (`<|vision_start|>`, one `<|image_pad|>` per merged 32x32 patch, `<|vision_end|>`),
  then the prompt; label and prompt rows are tagged text, the vision blocks image. The
  vision tower's output replaces the pads and its DeepStack taps join after the first
  three layers. A 1024x1024 vision copy is 1,024 rows, so many pictures make a long
  sequence; the trunk then runs flash attention.
- **VAE encoder.** The VAE copy (cropped to whole 16-pixel cells) goes through the image
  VAE's encoder (the H3 causal 3-D CNN at one frame) in 256-pixel tiles overlapping by at
  least 64; the tiles' moments are blended as diffusers blends them and form the
  posterior. A sample of it (rounded to float16, cropped to whole 2x2 patches, normalized)
  is the reference's latent, noised to t = 0.999 and held there for every step.
- **Packed sequence.** `[text | references | target]`: reference k is one step of the time
  axis after the text with its own aspect-normalized grid, the target after the last
  reference. Text and target run at t = 1 - sigma, the references at 0.999.
- **Seed.** One CPU torch generator per request draws, in MiniMax's order, each
  reference's posterior sample, then the target's noise, then each reference's noise. An
  edit therefore never restarts from its source picture's noise.
- **Size.** `--width`/`--height` (multiples of 32, at most 20 MP); `--keep-source-size`
  returns `<Picture 1>`'s exact size; otherwise `<Picture 1>`'s aspect at the request's
  area (the server's 2048x2048 default), or on the CLI its VAE copy's size. An edit below
  2048x2048 samples on a larger canvas of the same aspect (diffusers'
  `min_generation_area`) and is resized back with Lanczos. Text to image keeps its own
  size (no floor).
- **Refusals.** `cfg` above 1, a negative prompt, a mask, more than 16 references, a side
  under 32 pixels, and an edit whose references make its sequence longer than the
  device's memory holds (the message names the rows; fewer or smaller references fit).

Each reference costs rows twice: about 1,024 text rows for its vision block and, in the
DiT, (VAE copy / 32)² rows (1,024 for a 1024x1024 copy, 4,096 for 2048x2048). Sixteen
1024x1024 references beside a 2048x2048 target are about 37,000 DiT rows, nine times a
text-to-image step's; attention grows with the square.

## How it runs

- **Phases.** Each picture runs three phases: the text (the encoder), the denoise (six
  steps of the DiT) and the decode (the VAE). The pipeline keeps every component on the
  device between phases and between pictures when they fit, and otherwise releases one
  between phases ([Memory](#memory)).
- **Prompt.** Raw prompt tokens, no chat template, through the MiniMax-H3 Qwen3-VL
  trunk (the native encoder MiniMax-H3 video uses); the conditioning is the layer-50
  hidden state.
- **Packed sequence.** `[text | target image]` with the H3 3-axis RoPE: text token i
  at (i, 0, 0), the image's 2x2 patches one step later on the time axis over the
  aspect-normalized spatial grid. Text and image run at timestep t = 1 - sigma, with
  no flow shift.
- **Schedule.** ComfyUI's template samples KSampler's `simple` scheduler over the
  checkpoint's grid (`MiniMaxH3ImageModelSampling`); TensorSharp reproduces those
  sigmas in float32 (1, 0.96765, 0.88771, 0.75, 0.53364, 0.23261, 0 at six steps).
- **PDD heads.** `final_layer.video_out` stacks six heads; the step from grid point
  k to k+1 predicts with head 0 plus head k (head 0 alone for the first step). The
  combined head is built once per interval and passed to the step as an input, so the
  six steps share one compiled step.
- **DiT.** One graph program a step: the token refiner, the 50 blocks and the final
  layer, with no audio stream. On `ggml_cuda` the modulations, gated residuals,
  attention (cuDNN) and the BF16 SwiGLU-in-GEMM run as fused stages and the wide
  products on cuBLAS (or int8); elsewhere the same program runs ggml's own kernels.
  The program is the graph MiniMax-H3's native DiT builds, node for node, so the
  velocities are the same bit for bit.
- **VAE.** The image decoder (24 blocks of 1024, 16 heads, RoPE over (h, w) on half of
  each head, 16x16 RGB patches) on 256-pixel tiles overlapping by at least 64 pixels,
  blended into the canvas exactly as ComfyUI's `MiniMaxH3ImageVAE` does. A row of tiles
  (up to 64) is one graph program; on `ggml_cuda` its attention is one cuDNN call over
  the whole row. Any row decodes on every backend: the decoder graph is sized from the
  program, where the earlier native decode aborted on `ggml_metal` and `ggml_cpu` past
  four tiles a call (`GGML_ASSERT(cgraph->n_nodes < cgraph->size)`, which every
  2048x2048 or larger picture hit). `TS_H3_IMAGE_VAE_BATCH` was that decoder's
  workaround and is no longer read.

## Memory

The pipeline plans each picture against the device's budget: on `ggml_cuda` the free
VRAM less a headroom (1/16 of the card, at least 512 MiB); on `ggml_metal` the GPU's
recommended working set less what is allocated, less the same headroom (on unified
memory a command buffer past the working set fails). When the text encoder, the
denoiser, the VAE and their working memory fit, everything stays resident between
phases and between pictures. Otherwise a component the next phase does not use is
released before it: its device copies are dropped and its weights stay mapped on the
host, so a later phase uploads them again from memory or the page cache. `ggml_cpu`
keeps everything. The decision is printed once per size, for example:

```
[diffusion-residency] 2048x2048, 0 reference(s): before Denoise: release the conditioner (peak 24.7 GB, budget 37.7 GB)
```

Measured (the device's peak use over a picture, CUDA context included; on Metal the GPU's allocated size and the
system's wired memory):

| Device | DiT | Size | Plan | Peak |
|---|---|---|---|---:|
| A40 46 GB | Q4_K_M | 2048x2048 | everything resident | 30.0 GiB |
| A40 46 GB | Q4_K_M | 4096x4096 | everything resident | 33.9 GiB |
| A40 46 GB | bf16 | 2048x2048 | the text encoder released before the denoise (15.6 GB back in 0.8 s) | 39.9 GiB |
| A40 46 GB | bf16 | 4096x4096 | that, and the DiT's workspace released before the decode | 42.6 GiB |
| M5 Pro 48 GB | Q8_0 | 2048x2048 | the text encoder released before the denoise; from the second picture on also the DiT before the text | 24.3 GiB allocated, 29.9 GB wired |
| M5 Pro 48 GB | Q4_K_M | 512x512 | everything resident | 33.1 GB wired |

Without these releases bf16 does not fit the A40 beside the text encoder (about 60 GB),
and Q8_0 does not fit the 48 GB Mac: with the text encoder kept, the earlier path was
stopped by the wired-memory watchdog at 43.5 GB wired while it loaded the DiT. With
bf16 on the A40 each later picture uploads the text encoder and the DiT again (about 4 s
and 15 s from disk on that VM, whose page cache cannot hold both). bf16 does not fit a
48 GB Mac at all (40.2 GB of weights against a 40.2 GB working set): use Q8_0 there. An
H100-class card keeps everything, bf16 at 16 MP included.

## Performance

### A40 and M5 Pro

One A40 46 GB (`ggml_cuda`, sm_86, a shared VM) and one M5 Pro 48 GB (`ggml_metal`),
the prompts of the validation set (40 to 160 tokens), six steps. Text is the encode; a
step is the median of steps 2 to 6; the picture is the whole request. The first picture
of a process reads the files cold (on the A40 they were evicted from the page cache first)
and includes the uploads; later pictures are warm.

| A40, 2048x2048 | Text | Per step | Decode | Picture |
|---|---:|---:|---:|---:|
| Q4_K_M, first picture | 3.25 s | 2.23 s | 1.22 s | 21.0 s |
| Q4_K_M, later pictures | 0.08 s | 2.24 s | 0.69 s | 14.3 s |
| Q4_K_M, `TS_H3_INT8_MATMUL=1`, later pictures | 0.08 s | 1.88 s | 0.68 s | 12.1 s |
| bf16, first picture (the text encoder released before the denoise) | 3.07 s | 1.92 s | 1.20 s | 29.7 s |
| bf16, later pictures (both reloaded from disk each time) | 4.27 s | 1.91 s | 0.73 s | 34.2 s |

| A40, 4096x4096 | Text | Per step | Decode | Picture |
|---|---:|---:|---:|---:|
| Q4_K_M, first picture | 3.25 s | 11.42 s | 2.91 s | 77.8 s |
| Q4_K_M, later pictures | 0.07 s | 11.47 s | 2.40 s | 71.5 s |
| Q4_K_M, `TS_H3_INT8_MATMUL=1`, later pictures | 0.07 s | 9.96 s | 2.48 s | 62.5 s |
| bf16, first picture | 3.36 s | 10.36 s | 3.09 s | 82.6 s |

| M5 Pro, 2048x2048 | Text | Per step | Decode | Picture |
|---|---:|---:|---:|---:|
| Q8_0, first picture | 1.71 s | 12.82 s | 3.45 s | 82.6 s |
| Q8_0, later pictures | 1.70 s | 12.67 s | 2.72 s | 81.1 s |

On the same machines, against the earlier native path in alternating processes, no
phase took more than 1.4% longer and most took the same or less (the decode up to 13%
less; the first picture's text 11 to 37% less, because the encoder's file is now mapped
before that phase starts), the peak device memory was within 0.5% (1.3% lower for bf16
at 16 MP), and
every picture was identical, bit for bit: 8 prompts each for Q4_K_M, Q4_K_M with int8
products and bf16 at 2048x2048 on the A40 and for Q8_0 on the M5 Pro, 2 prompts for
Q4_K_M and int8 at 4096x4096, 1 for bf16 at 4096x4096. On the M5 Pro the earlier path
needed its text encoder released and its decoder held to 4 tiles a call to run at all.

### Editing

One A40 46 GB (`ggml_cuda`) and one M5 Pro 48 GB (`ggml_metal`, Q8_0), 2048x2048 edits, six steps, one process an edit
with its files read cold (evicted from the page cache first on the A40), so these are first-edit numbers. Text is the
encode without the vision tower; vision is the tower over every picture; VAE encode is every reference's tiles; a step
is the median of steps 2 to 6. The 1, 4 and 16 references are 1024x1024 pictures (1,024 DiT rows and about 1,024 text
rows each); the text edit has one 2048x2048 reference (4,096 rows), the composition two.

| A40, Q4_K_M | Rows (text + references + target) | Text | Vision | VAE encode | Per step | Decode | Edit | Peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 reference | 1,076 + 1,024 + 4,096 | 3.98 s | 1.03 s | 0.84 s (25 tiles) | 3.43 s | 1.66 s | 31.9 s | 32.7 GiB |
| 4 references | 4,189 + 4,096 + 4,096 | 8.20 s | 2.01 s | 2.73 s (100) | 9.36 s | 1.16 s | 75.5 s | 35.8 GiB |
| 16 references | 16,635 + 16,384 + 4,096 | 33.20 s | 6.05 s | 9.97 s (400) | 49.1 s | 1.17 s | 349.5 s | 42.5 GiB |
| text edit (1 at 2048x2048) | 1,091 + 4,096 + 4,096 | 4.29 s | 0.98 s | 3.12 s (121) | 5.47 s | 1.11 s | 46.6 s | 33.6 GiB |
| composition (2 at 2048x2048) | 2,125 + 8,192 + 4,096 | 5.00 s | 1.39 s | 6.09 s (242) | 9.48 s | 1.15 s | 74.7 s | 35.5 GiB |

A text-to-image step at 2048x2048 is 2.2 s on the same card ([above](#a40-and-m5-pro)); a step's cost grows with the
rows, and attention with their square. Sixteen 1024x1024 references keep everything resident on the A40 with Q4_K_M; the
planner drops only the text encoder's workspace before the denoise.

| M5 Pro, Q8_0 | Rows | Text | Vision | VAE encode | Per step | Decode | Edit | Peak (allocated) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 reference | 1,076 + 1,024 + 4,096 | 6.48 s | 1.38 s | 3.55 s (25) | 20.7 s | 3.82 s | 140.3 s | 28.4 GB |
| 4 references | 4,189 + 4,096 + 4,096 | 19.23 s | 3.92 s | 14.05 s (100) | 52.2 s | 3.88 s | 355.0 s | 33.0 GB |
| text edit (1 at 2048x2048) | 1,091 + 4,096 + 4,096 | 6.44 s | 1.24 s | 16.93 s (121) | 34.6 s | 3.80 s | 236.6 s | 30.7 GB |

### H100

The H100 numbers below were measured on the earlier native path; the runtime runs the
same graphs (above).

One H100 NVL 94 GB (`ggml_cuda`, sm_90), warm (models resident, second image of a
size), the same prompt (156 tokens) and seeds, six steps. ComfyUI is 0.45.0 running
its own template ("MiniMax-H3-Image: Text to Image": the pruned INT8 DiT, the INT8
Qwen3-VL-32B encoder, the FP16 VAE); TensorSharp reads the bf16 DiT and the Q4_K_M
encoder. ComfyUI's sampling time is its progress bar; TensorSharp's phases are the
`[h3-image]` line the model prints.

| 2048x2048 (4 MP) | Text | Denoise, 6 steps | VAE decode | ComfyUI job |
|---|---:|---:|---:|---:|
| ComfyUI template | | 2.6 s (0.43 s/step) | | 3.97 s |
| TensorSharp, bf16 projections | 0.10 s | 3.09 s (0.51 s/step) | 0.28 s (121 tiles) | 4.38 s |
| TensorSharp, `TS_H3_INT8_MATMUL=1` (`minimax-h3-image-int8`) | 0.09 s | 2.55 s (0.43 s/step) | 0.27 s | 3.88 s |

| 4096x4096 (16 MP) | Text | Denoise, 6 steps | VAE decode | ComfyUI job |
|---|---:|---:|---:|---:|
| ComfyUI template | | 14.7 s (2.45 s/step) | | 19.9 s |
| TensorSharp, bf16 projections | 0.09 s | 15.8 s (2.63 s/step) | 0.95 s (441 tiles) | 19.6 s |
| TensorSharp, `TS_H3_INT8_MATMUL=1` (`minimax-h3-image-int8`) | 0.09 s | 13.1 s (2.18 s/step) | 0.9 s | 16.9 s |

The job is the ComfyUI prompt through the TensorSharp Image node: TensorSharp's picture
plus its PNG (under 0.1 s at 4 MP, 0.1 to 0.4 s at 16 MP), the node's download and
decode, and ComfyUI's SaveImage, which a ComfyUI-only job pays as well (2.0 s for a
16 MP PNG). The first job of a server (weights from the page cache) took 9.1 s with int8
and 9.5 s with bf16 at 4 MP, against 55.1 s for ComfyUI's first job.

- **Projections.** A BF16 DiT's wide products (qkv, out, fc1, fc2) run on cuBLAS with
  the weight as stored; `TS_H3_INT8_MATMUL=1` rounds each weight row and each token
  to int8 per call and multiplies on int8 tensor cores, the precision class of
  ComfyUI's INT8 checkpoint. It is opt-in because it changes the picture: against
  the bf16 picture of the same seed it measured 17.5 to 19.8 dB PSNR (same
  composition, different detail). fc2 reads fc1's [gate | value] output and applies
  the SwiGLU as it narrows (or rounds) its input, so the SwiGLU's own pass over a
  [14336, tokens] tensor is gone (1.5 ms a block at 16 MP).
- **Attention.** cuDNN's SDPA runs the 56 x 128 attention over 16,540 tokens in 18 ms
  a block (21 ms with its prep and widening), the rate PyTorch's cuDNN backend reaches
  on the same GPU (its FlashAttention-2 took 30 ms); it is about half of a 16 MP step.
- **VAE.** The decoder runs a row of 256-pixel tiles in one program, and the row's
  attention in one cuDNN call over a batch of sequences rather than one call and one
  concat per tile: 0.7 s of decoder time at 16 MP, where per-tile attention took 1.29 s
  and per-tile calls 1.84 s.
- **PNG.** PNGs are written by the managed codec on every media provider: the Up
  filter, and bands of about 1 MiB of rows deflated at zlib level 4 in parallel (each
  band its own IDAT chunk, the pigz construction). The encode of a 16 MP picture
  takes about 0.1 s for 19.9 MB, where ImageMagick took 8.3 s at its default level 7
  and 0.9 s at level 1 (21.3 MB for the same picture); from the end of the decode to
  the written file, 1.6 s became 0.1 to 0.4 s.
- **Preview.** The TensorSharp Image node previews the finished picture at most
  1024 pixels on its long side; ComfyUI writes a preview as a full PNG, which took
  1.1 s for a 16 MP picture. The node's `image` output is the full picture.
- **Pictures, not pixels.** The two engines start from the same latent (the noise
  is ComfyUI's), and at 16 MP give the same composition, but they do not agree
  pixel for pixel: the Q4_K_M encoder against INT8, bf16 or per-call int8 against
  ComfyUI's INT8 weights and kernels move a six-step distilled sample by more than
  a few levels (9 to 12 dB PSNR between the engines; at 4 MP the composition can
  differ). Prompt adherence and the rendered text (a calendar reading "十一月 11")
  held in both.

## Not implemented

- Masked edits (inpainting) and the multi-turn planner for this model.
- The prompt expander (MiniMaxAI/MiniMax-H3-Image-PE). ComfyUI's text-to-image
  template leaves it off by default.
- The `mlx`, `cuda` and `cpu` (pure C#) backends.
