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

TensorSharp runs text to image through `MiniMaxH3ImageModel` on the ggml backends.
It has been run on `ggml_cuda` (an H100, below); `ggml_metal` and `ggml_cpu` take the
same native graphs but have not been run with it yet. Editing with reference images
(`<Picture 1>` ...) is not implemented yet.

| Component | File | Source |
|---|---|---|
| Diffusion transformer (bf16, as released) | `minimax_h3_image_pruned_bf16.safetensors`, 40.2 GB | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/diffusion_models) |
| Text encoder (shared with MiniMax-H3) | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf`, 16.97 GiB, with `vocab.json`, `merges.txt`, `tokenizer_config.json` | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF), [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor) |
| Image VAE | `minimax_h3_image_vae_fp16.safetensors`, 1.17 GB | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/vae) |

The denoiser is read from the safetensors file directly, in its original tensor
names: there is no GGUF to make. A GGUF holding the same tensors (for a smaller
quantization) loads as well; it is recognised by its tensors (the H3 DiT with no
`audio_patch_proj`). ComfyUI's own quantized repacks (`*_int8_convrot`, `*_w4a8`)
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

- **Prompt.** Raw prompt tokens, no chat template, through the MiniMax-H3 Qwen3-VL
  trunk; the conditioning is the layer-50 hidden state. The encoder stays resident
  between images.
- **Packed sequence.** `[text | target image]` with the H3 3-axis RoPE: text token i
  at (i, 0, 0), the image's 2x2 patches one step later on the time axis over the
  aspect-normalized spatial grid. Text and image run at timestep t = 1 - sigma, with
  no flow shift.
- **Schedule.** ComfyUI's template samples KSampler's `simple` scheduler over the
  checkpoint's grid (`MiniMaxH3ImageModelSampling`); TensorSharp reproduces those
  sigmas in float32 (1, 0.96765, 0.88771, 0.75, 0.53364, 0.23261, 0 at six steps).
- **PDD heads.** `final_layer.video_out` stacks six heads; the step from grid point
  k to k+1 predicts with head 0 plus head k (head 0 alone for the first step). The
  combined head is built once per interval and bound like any other weight.
- **DiT.** The H3 native graph (`TSGgml_MiniMaxH3DitForward`) with no audio stream,
  cuDNN attention and the fused modulation and residual stages.
- **VAE.** The H3 video VAE's native ViT decode with the image decoder's geometry
  (24 blocks of 1024, 16 heads, RoPE over (h, w) on half of each head, 16x16 RGB
  patches), on 256-pixel tiles overlapping by at least 64 pixels and blended into
  the canvas exactly as ComfyUI's `MiniMaxH3ImageVAE` does.

## Performance

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
- **VAE.** The decoder runs a row of 256-pixel tiles in one native call
  (`TSGgml_MiniMaxH3VitDecodeBatch`), and the row's attention in one cuDNN call over a
  batch of sequences rather than one call and one concat per tile: 0.7 s of native
  time at 16 MP, where per-tile attention took 1.29 s and per-tile calls 1.84 s.
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

- Editing and reference images (the VAE encoder path and the `Picture n:` vision
  blocks).
- The prompt expander (MiniMaxAI/MiniMax-H3-Image-PE). ComfyUI's text-to-image
  template leaves it off by default.
- The `mlx`, `cuda` and `cpu` (pure C#) backends.
