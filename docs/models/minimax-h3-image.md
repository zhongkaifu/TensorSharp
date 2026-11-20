# MiniMax-H3-Image

[English](minimax-h3-image.md) | [中文](minimax-h3-image_zh-cn.md)

## License and legal disclaimer

> **Important — MiniMax-H3-Image is subject to a separate third-party license.** Before downloading, running, deploying, modifying, redistributing, or using MiniMax-H3-Image or its outputs with TensorSharp, you must review and comply with the [MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE), its Acceptable Use Policy, and all applicable laws and regulations.
>
> The community license expressly excludes the **United States, European Union, United Kingdom, and Republic of Korea (South Korea)** from its licensed territory. It does **not** authorize covered use in those territories. It also requires separate written authorization from MiniMax when the annualized revenue of you and your affiliates exceeds USD 1 million or when you provide hosted services to third parties, and it prohibits using outputs to improve other AI models. Consult the official license for the full terms and any changes.
>
> TensorSharp is an independent inference engine. Its technical compatibility with MiniMax-H3-Image does **not** grant a license, waive restrictions, or imply authorization or endorsement by MiniMax. **You are solely responsible for determining whether your intended use is permitted and for all risks, consequences, claims, penalties, damages, and other liabilities arising from your use or noncompliance. To the fullest extent permitted by applicable law, TensorSharp, its authors, contributors, and maintainers disclaim any responsibility or liability for your use of MiniMax-H3-Image, its outputs, or any violation of its license or applicable law.** This notice does not override the MiniMax license or limit liability where such limitation is prohibited by law.

MiniMax-H3-Image is MiniMax's text-to-image model built from the MiniMax-H3 video
model: the same 33B single-stream DiT (50 blocks, hidden 5376, 56 heads of 128) with
its audio stream removed, the same Qwen3-VL-32B text encoder (layer-50 hidden
states), and a new image VAE whose 2-D ViT decoder reads the H3 latent space. It is
distilled to **six Euler steps with guidance 1**: the final layer carries one output
head per step of the checkpoint's own sigma grid, and there is no negative prompt.
It works from about 4 MP to 20 MP and is best around 16 MP (4096x4096, 4736x3552,
5024x3360, 5472x3072, 6272x2688).

TensorSharp runs text to image on its shared diffusion runtime
(`TensorSharp.Models.SharedDiffusion`: the MiniMax-H3-Image family, its text encoder,
denoiser and VAE decoder, run phase by phase by one pipeline) on `ggml_cuda`,
`ggml_metal` and `ggml_cpu`, end to end. It has been run with real weights on
`ggml_cuda` (an H100 and an A40) and on `ggml_metal` (an M5 Pro); `ggml_cpu` runs the
same programs (validated on small synthetic checkpoints; with real weights it takes tens
of minutes a picture). Editing with reference images (`<Picture 1>` ...) is not
implemented yet.

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

The server serves it on the image routes (`/api/image-generate`,
`/api/image-generate/stream`) like Qwen-Image-2.1, and the ComfyUI node pack has a
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

## LoRA plug-ins

MiniMax-H3-Image takes LoRA adapters of type `minimax-h3-image-lora` through the hosts'
`--lora`, `--lora-scale` and `--lora-config` (usage: [USAGE](../../USAGE.md#minimax-h3-image-lora-plug-ins))
and `TS_LORAS`: any number of style adapters, each at its own strength, and at most one speed adapter, whose config fixes
the sampling.

| Plug-in | Source | Kind | Names | Rank / alpha |
|---|---|---|---|---|
| [`minimax-h3-image-pdmd-2step.json`](../../config/lora/minimax-h3-image-pdmd-2step.json) | linoyts/MiniMax-H3-Image-pDMD-2steps @c07d96e5 | speed: 2 steps on `[1, 12/13, 0]`, `by-step` heads | diffusers PEFT | 128 / 128 |
| [`minimax-h3-image-pdmd-2step-comfy.json`](../../config/lora/minimax-h3-image-pdmd-2step-comfy.json) | akhaliq/minimax-h3-image-pdmd-2step @ef1e3b9a (the ComfyUI conversion) | speed: the same sigmas, `grid-interval` heads | ComfyUI native, block-diagonal fused qkv | 384 / 384 on qkv, 128 / 128 |
| [`minimax-h3-image-seedreamstyle.json`](../../config/lora/minimax-h3-image-seedreamstyle.json) | Nebsh/Seedreamstyle-H3-6K @5e1b48cd | style (`seedreamstyle.` trigger) | kohya | 16 / 16 |

The two pDMD files are two trainings, not one: linoyts' and akhaliq's (pdmd2026's) factors
are uncorrelated. A bare `.safetensors` in any of the three grammars is a style adapter.

**How an update is applied.** Unmerged: `y = W x + up (down x)` on the projection the
adapter names, the base weights as stored. diffusers' `to_q`, `to_k`, `to_v` land on the
stacked thirds of `attn.qkv_proj`, its `ff.net.0.proj` with the two halves of `lora_B`
swapped from `[value; gate]` into the checkpoint's `[gate; value]`; ComfyUI and kohya names
are the checkpoint's own layout (a block-diagonal fused qkv term splits exactly into its
thirds). Strength × alpha / rank is folded into `up` in F32; each rank component is balanced
and stored as F16 (F32 on `ggml_cpu`). Several adapters on one projection are concatenated
along the rank, so a projection runs one shrink and one expand a slot however many there are.
On `ggml_cuda` a wide product (Q4_K, Q6_K or BF16 weight, 1,024 rows or more) takes its update
inside its own cuBLAS GEMM, `[W | up] · [x ; down x]` K + R deep (a BF16 weight is copied
into those rows below K = 2 × rows; above, its parts' expands accumulate into its product).
On `ggml_metal` a wide product over a Q8_0, F16 or F32 weight is built the same way from
ggml's own operations (the weight concatenated with the ups each step, never changed) where
that moves fewer bytes than adding the update beside it: qkv, out and fc1 at 4 MP, not fc2.
Elsewhere the region's products and adds run on ggml.
A request with other adapters than the last drops the last set's compiled steps and device
copies first.

**pDMD's heads.** By-step heads run head 0 for step 1 and head 0 plus head 1 for step 2:
TensorSharp's combined head over one grid interval `[i, i + 1)` equals diffusers'
`pdd_proj_out[i]` (MiniMaxAI's `pdd_fused_heads.safetensors`) to 3.8e-10 relative, all six
heads, every row.

**Validation** (`eng/validation/MiniMaxH3ImageLoraRows`, `eng/validation/MiniMaxH3ImageAB`):

- *Rows against an independent merge.* pDMD (linoyts), akhaliq's pDMD in both its diffusers
  and ComfyUI files, Seedreamstyle (kohya), waryusya (ComfyUI names, no alpha) and domokun
  (ComfyUI, fused qkv with `.alpha`): every projection an adapter touches in blocks 0, 25
  and 49 and the refiner's block 0 ran as the backend runs it and was compared on sampled
  rows with NumPy's `(W + scale · B A) x`, the adapter read from the Hub by HTTP range
  requests and mapped by the published conversions, not by TensorSharp's code. `y` agrees to
  3.6e-3 relative on the A40's Q4_K_M, 2.1e-3 on bf16 and 6.4e-4 on the M5 Pro's Q8_0
  (`ggml_metal`, beside the product at 1,024 rows and inside it at 4,096), the base product's
  own rounding; the update alone (the same region run with zero ups subtracted) to 5.8e-3,
  5.7e-3, and 1.3e-3 beside and 5.4e-3 inside the product (its ups rounded to Q8_0). akhaliq's two
  files give the same update to the last bit, in NumPy and in TensorSharp.
- *No adapter, no change.* The 2048x2048 portrait is bit-identical to the one before LoRA
  support: Q4_K_M and bf16 on the A40, Q8_0 on the M5 Pro.
- *Cost of a step* at 2048x2048, the median over warm pictures of every step but each
  picture's first (on the A40 a bf16 picture's first step re-uploads the weights):

  | Step, 2048x2048 | No adapter | pDMD (rank 128) | Seedreamstyle (rank 16) |
  |---|---:|---:|---:|
  | A40, Q4_K_M | 2.188 s | 2.087 s (−4.6%) | 2.045 s (−6.5%) |
  | A40, bf16 | 1.907 s | 2.104 s (+10.3%) | 2.062 s (+8.1%) |
  | M5 Pro, Q8_0 | 12.53 s | 12.94 s (+3.3%) | 12.65 s (+0.9%) |

  On Q4_K_M a step with an adapter is faster than one without: the K-concat GEMM also
  takes the projections a Q4_K_M file keeps at Q6_K (qkv and fc2 of half the blocks) onto
  cuBLAS BF16, where the plain path runs them on ggml's own kernels; rank 128 costs 2.1% more than
  rank 16 on the same route. On bf16 the cost hardly depends on the rank: it is the copy
  of each weight into the K + R rows (4 × K × N bytes a projection, about 71 GB a step),
  which a quantized weight's dequantization pays anyway. Running the copy on a second
  stream and adding the expands before the product with beta 1 were both slower (2.149 s
  and 2.166 s for pDMD); a GEMM that reads the update's columns from their own pointer is
  what would remove it. On the M5 Pro the ops beside the product cost pDMD 10.8% and
  Seedreamstyle 8.2%; with the update inside the product the two together cost 5.7%.
- *pDMD end to end.* On the A40 Q4_K_M a warm 2048x2048 picture takes 5.1 s against 14.1 s
  (2.8×), the first picture of a process 13.0 s against 21.0 s; on bf16 24.5 s against
  34.4 s (1.4×: the 46 GB A40 re-uploads the bf16 weights in every picture); on the M5 Pro
  30.8 s against 79.8 s (2.6×).
- *Pictures.* The 8 prompts of the validation set at 2048x2048, pDMD on the A40 (Q4_K_M,
  bf16) and the M5 Pro (Q8_0), against the 6-step pictures: the same subjects and
  compositions, sharp and clean; small type suffers (the tea house's English sign is
  garbled while 冬至 and the departure board hold), and a few details go wrong where the
  6-step picture is right (the product mug gets a second handle, the dog under the picnic
  table blurs). diffusers' own pipeline (`diffusers_pdmd.py`: the official bf16
  transformer with the LoRA fused into its weights, fed TensorSharp's text embeddings, the
  same seed and noise) draws the same portrait, the same pose, hat and window, with fine
  detail apart: 17.2 dB PSNR against TensorSharp's bf16 picture, where the 6-step picture
  is 14.0 dB away. The ComfyUI conversion (akhaliq, pdmd2026's training) draws its
  subjects well but lays a fine mesh pattern over skin and flat areas, with by-step heads
  as well as the grid-interval heads it ships with (both tried on the M5 Pro); its own
  card's pictures came from Pruna's P-H3-Image stack (a 4B text encoder), not the official
  one. Its update matches its diffusers twin bit for bit, so this is the adapter on this
  stack, not its loading. For pictures use `minimax-h3-image-pdmd-2step.json`.

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

- LoRA merging (the updates always run unmerged), DoRA, LoCon, LoHa and LoKr adapters,
  and a per-module alpha or rank pattern in a PEFT config.
- Editing and reference images (the VAE encoder path and the `Picture n:` vision
  blocks).
- The prompt expander (MiniMaxAI/MiniMax-H3-Image-PE). ComfyUI's text-to-image
  template leaves it off by default.
- The `mlx`, `cuda` and `cpu` (pure C#) backends.
