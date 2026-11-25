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
real weights it takes tens of minutes a picture). An edit can be limited to a selection of `<Picture 1>` with a mask
([Masked edits](#masked-edits)), and the chat hosts edit across turns ([Multi-turn](#multi-turn)).

| Component | File | Source |
|---|---|---|
| Diffusion transformer (bf16, as released) | `minimax_h3_image_pruned_bf16.safetensors`, 40.2 GB | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/diffusion_models) |
| Text encoder (shared with MiniMax-H3) | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf`, 16.97 GiB, with `vocab.json`, `merges.txt`, `tokenizer_config.json` | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF), [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor) |
| Image VAE | `minimax_h3_image_vae_fp16.safetensors`, 1.17 GB | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/vae) |

Smaller denoisers, Unsloth's GGUFs of the same pruned DiT, run with the same text encoder and
VAE: [Quantizations](#quantizations) has their measured quality and speed and the config for
each, and the TensorAgent app lists two of them ([TensorAgent](#tensoragent)).

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
(`--diffusion-products int8`, [Performance](#performance)).

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

## Quantizations

[unsloth/MiniMax-H3-Image-GGUF](https://huggingface.co/unsloth/MiniMax-H3-Image-GGUF/tree/0d25c7349e007529691bbc0f4426b579fce7de01)
(at `0d25c734`) quantizes Comfy-Org's pruned DiT; a GGUF loads as the bf16 file does, beside the
same text encoder and VAE. Four DiTs have a config, each pinned by revision and SHA-256 and sharing
the other files (same paths):

| DiT | File, bytes | Config | Runs on (measured) |
|---|---|---|---|
| bf16 | `minimax_h3_image_pruned_bf16.safetensors`, 40,234,657,162 | [`minimax-h3-image.json`](../../config/minimax-h3-image.json) | A40 46 GB to 4096x4096, H100; not a 48 GB Mac |
| Q8_0 | `minimax_h3_image_pruned-Q8_0.gguf`, 22,026,760,928 | [`minimax-h3-image-q8.json`](../../config/minimax-h3-image-q8.json) | A40 46 GB, M5 Pro 48 GB; refused on 20 GB |
| Q5_K_M | `minimax_h3_image_pruned-Q5_K_M.gguf`, 14,599,107,296 | [`minimax-h3-image-q5km.json`](../../config/minimax-h3-image-q5km.json) | RTX A4500 20 GB to 3840x3840 |
| Q4_K_M | `minimax_h3_image_pruned-Q4_K_M.gguf`, 12,721,119,968 | [`minimax-h3-image-q4km.json`](../../config/minimax-h3-image-q4km.json) | RTX A4500 20 GB to 4480x4480, A40, M5 Pro |

**Quality.** The same prompt, seed and size (2048x2048, six steps) against the bf16 DiT: four
prompts (a portrait, a sign and banner with Chinese and English text, a landscape, a watercolour
illustration), PSNR of the 8-bit RGB picture and SSIM.

| DiT | RTX A4500 (`ggml_cuda`) against bf16 on `ggml_cpu`, mean of four | A40 (`ggml_cuda`) against bf16 on the A40: portrait / text / landscape / illustration |
|---|---|---|
| Q8_0 | does not fit 20 GB | 27.7 / 22.9 / 28.4 / 17.6 dB |
| Q6_K | 21.84 dB / 0.840 | |
| Q5_K_M | 21.81 dB / 0.835 | |
| UD-Q4_K_XL | 19.65 dB / 0.791 | |
| Q4_K_M | 17.29 dB / 0.725 | 18.9 / 16.2 / 21.6 / 13.3 dB |
| Q3_K_M | 17.06 dB / 0.703 | |

A six-step distilled sampler turns a small difference in a step into a moved composition, so even
the best quantizations sit at 17-28 dB: Q8_0 against bf16 on the same CPU backend is 26.7 dB, and
the same Q6_K weights on CUDA against CPU are 29.3 dB. Differences of 1-2 dB between the top
quantizations are within that; the 4-5 dB down to Q4_K_M and Q3_K_M are not. Every picture is
coherent and follows its prompt. Q8_0 keeps bf16's composition in all four; Q5_K_M and Q6_K are
near copies of the portrait and write the banner cleanly; Q4_K_M changes faces and objects and, in
the text prompt, adds a second sign of garbled lettering; Q3_K_M shrinks the banner and adds a
garbled line.

**Speed and memory on 20 GB.** One RTX A4500 (`ggml_cuda`, sm_86), 2048x2048, six steps, the
runtime's defaults (the text encoder is released before the denoise there). A step is the median of
steps 2 to 6 over three fresh processes; the CLI picture is the whole process; the warm picture is
a server's second and third request (each re-uploads the DiT, which cannot stay beside the
encoder); the peak is the process's, CUDA context included.

| DiT | Per step | CLI picture | Warm server picture | Peak | Largest square that runs |
|---|---:|---:|---:|---:|---|
| Q3_K_M | 3.68 s | 30.4 s | 29.5 s (a noisy spell on the shared host) | 15.2 GiB | 4480x4480 |
| Q4_K_M | 3.32 s | 28.6 s | 24.8 s | 15.2 GiB | 4480x4480 (18.9 GiB) |
| UD-Q4_K_XL | 3.66 s | 31.5 s | 26.4 s | 15.6 GiB | 4096x4096 |
| Q5_K_M | 3.88 s | 32.0 s | 27.6 s | 16.6 GiB | 3840x3840 (19.0 GiB) |
| Q6_K | 4.22 s | 34.5 s | 30.2 s | 17.4 GiB | 3328x3328 (19.2 GiB) |
| Q8_0 | refused | | | | none: its DiT alone is 22.0 GB |

Every picture was the same, bit for bit, as the earlier native path's at the same seed, and a size
the card cannot hold is refused before any phase runs ([Memory](#memory)). The A40 and M5 Pro
timings are under [Performance](#performance).

**Which one.** 20-24 GB GPU: Q5_K_M (the closest that fits, clean text), or Q4_K_M for 16-20 MP
output. 32 GB Mac: Q4_K_M (the text encoder's phase, 18.9 GB with its working memory, is the
largest; Q8_0's denoise is past a 32 GB Mac's GPU working set). 46-48 GB GPU or 48 GB Mac: Q8_0.
80 GB and up: bf16.

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
  sequence; past about 2,000 rows the trunk computes its attention scores a block of
  query rows at a time (the same scores, in pieces of at most 1 GiB).
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
  `min_generation_area`) and is resized back with Lanczos; a speed plug-in's own floor
  replaces it (pDMD's is 1024 pixels: its edits sample at the asked size, as its diffusers
  pipeline does). Text to image keeps its own size (no floor); a masked edit's selection
  has its own rule (below).
- **Refusals.** `cfg` above 1, a negative prompt, more than 16 references, a side under
  32 pixels, and, before anything runs, an edit whose references make its text encoding
  or its denoising step need more than the device's free memory (the message names the
  rows; fewer or smaller references fit). This is the one check every request gets, a
  picture from words included: a step that fits the free memory runs even when the plan
  says it uses some of the headroom.

Each reference costs rows twice: about 1,024 text rows for its vision block and, in the
DiT, (VAE copy / 32)² rows (1,024 for a 1024x1024 copy, 4,096 for 2048x2048). Sixteen
1024x1024 references beside a 2048x2048 target are about 37,000 DiT rows, nine times a
text-to-image step's; attention grows with the square.

### Masked edits

```bash
TensorSharp.Cli --config config/minimax-h3-image.json \
  --image shop.png --mask banner.png --keep-source-size --diffusion-seed 42 \
  --prompt "In <Picture 1>, change the painted banner above the door so it reads \"SPRING EQUINOX CAFE\"." \
  --output shop-edit.png
```

A mask limits the edit to a selection of `<Picture 1>`, the picture being edited; the
other pictures stay references. MiniMax's pipeline has no masks (its README: "not yet
supported: mask-based inpainting"), so this is TensorSharp's own method, the one its
Qwen-Image path uses, and the flags and fields are the same:

- **The mask** has `<Picture 1>`'s exact size. `--mask-mode grayscale` (the default):
  white edits, black keeps, gray blends; `alpha`: transparent edits, opaque keeps.
  `--mask-invert` swaps them, `--mask-feather N` softens the edge inward by N source
  pixels (never past the selection), and `--mask-crop` with `--mask-crop-padding N`
  (default 64) samples only the selection's box and that much context around it. The
  server takes `maskPath` (an upload), `maskMode`, `maskInvert`, `maskFeather`, `maskCrop`
  and `maskCropPadding`, or a `mask` file on the multipart `/api/image-edit`; the OpenAI
  `/v1/images/edits` takes a `mask` part, read [OpenAI's way](../../TensorSharp.Server.Host/API_EXAMPLES.md#openai-image-routes-v1imagesgenerations-v1imagesedits).
- **Size.** The result is always `<Picture 1>`'s own canvas. The whole picture (or a
  crop's box) samples at the density of the canvas the same edit without a mask would use
  (the rules above), but never above its own resolution, and the edit's floor does not raise
  it: a whole picture's latent mask, denoise and decode are at most the size of the pixels
  they are composited into, snapped upward to the 32-pixel grid. A picture larger than that
  canvas still samples below its own resolution, and the decode is resized back with
  Lanczos. (Before, the 2048x2048 floor applied here too: a 512x512 picture's masked edit
  sampled 2048x2048, 16 times its pixels.)
- **A crop's floor.** A crop smaller than `<Picture 1>` samples at least 2048x2048 pixels
  (the edit's own floor) at its own aspect, held to 4 times the whole picture's pixels; a
  speed plug-in's floor does not change it. A crop above 4 MP samples at its own size, and a crop whose
  padding reaches every edge is the whole picture, at its own size. The 448x512 crop of a
  1024x1024 picture samples 1920x2176, as before masked edits were capped. Smaller floors
  were tried first on the A40 (Q4_K_M, a red vase in that crop, seeds 7, 11 and 23):

  | crop sampled at | total | a step | peak | the vases |
  |---|---|---|---|---|
  | its own size, 448x512 | 16.9-17.5 s | 1.71 s | 32.5 GB | plain; one seed drew stray white shapes |
  | 1 MP, 960x1088 | 19.9-20.4 s | 2.1 s | 32.8 GB | clean, but plain |
  | 4 MP, 1920x2176 (the floor) | 33.5-34.6 s | 3.9 s | 33.8 GB | relief or painted surfaces for two seeds |

  At 4 MP the three PNGs are byte-identical to the ones from before the cap. The peak is
  mostly the 32B text encoder's. A small picture's crop is held to 4 times the picture. A
  crop of a 512x512 picture (a green bottle in place of a vase, seeds 7, 11 and 23) samples
  736x1440, 1 MP, in 28.3-28.9 s (3.1 s a step, 34.1 GB). At 4 MP, 16 times the picture,
  it took 42.9-43.4 s (5.1 s a step, 35.1 GB). At 4 MP two seeds kept most of the vase, and
  the third drew an illustrated bottle and redrew the cat's edge as a drawing. At 1 MP two
  seeds drew a clean bottle in the picture's light, and the third (seed 7) kept the vase
  beside a thin bottle.
- **The latent.** The mask goes to the latent grid by the largest weight in each 16x16
  cell, so a thin stroke stays editable. `<Picture 1>` (or its crop) at the sampling canvas
  is encoded to its posterior mean, normalized as the references are (its reference copy is
  reused when it has that size; otherwise it costs one more VAE encode). After every step
  the kept cells are set back to that latent noised to the next sigma with the request's own
  noise, so the last step leaves them exactly on it: the edited area is denoised with its
  surroundings in view, the picture's own pixels around it. `<Picture 1>` is still a
  reference, so the model sees what it blends with.
- **Composite.** The decode, resized to the selection, is blended into `<Picture 1>` by the
  mask's weights: every pixel of weight 0 is the source's own bytes (alpha included), every
  pixel of weight 1 the edit's. An empty mask returns `<Picture 1>` without running the
  model; a full one is an unmasked edit on the source's canvas.
- **With pDMD.** Masks and the pDMD plug-in compose: its two steps reset the kept cells
  after each step like the six do, and the composite is the same, so the kept pixels are
  still the source's own bytes; the selection samples as above (at most its own
  resolution, a crop at least the crop floor). Checked on the M5 Pro before that cap, at the plug-in's floor (Q8_0, the mug replaced by a
  glass teapot, a box mask): 2,960,883 kept pixels, none differing; 34.9 s a step against
  the six-step masked edit's 33.6 s, the edit in 99 s.

The cost is the unmasked edit's, plus the extra VAE encode when `<Picture 1>`'s reference
copy is not the sampling canvas (a 2048x2048 picture's is; a 1024x1024 one's is not). The
selection is kept as drawn: an inverted rectangle around an object keeps the whole box,
background included, which then meets the new surroundings at a hard edge; a selection
drawn along the object, or a feather, blends it.

M5 Pro 48 GB, `ggml_metal`, Q8_0, 2048x2048 pictures, one process, six steps (a step is the
median of steps 2 to 6; the first edit in the process cold, the rest warm):

| Masked edit | Selection | Text | Vision | VAE encode | Per step | Decode | Edit | Peak (allocated) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sign rewritten ("SPRING EQUINOX CAFE") | the banner, 10.0% | 3.81 s | 1.24 s | 16.94 s (121) | 33.6 s | 2.86 s | 232.2 s | 30.6 GB |
| mug replaced by a glass teapot | a box around the mug, 29.4% | 6.44 s | 1.15 s | 16.95 s (121) | 33.6 s | 2.71 s | 229.9 s | 30.6 GB |
| inverted: everything but the mug box | 70.6% | 6.27 s | 1.16 s | 16.93 s (121) | 33.6 s | 2.69 s | 229.6 s | 30.6 GB |

The same picture unmasked costs the same (S5's text edit: 33.8 s a step, 231.0 s, 30.7 GB):
the reference copy is the sampling canvas, so nothing more is encoded. Every unselected pixel
was the source's own bytes (2,960,883 to 3,774,333 of them a picture, none differing).

### Multi-turn

TensorAgent's chat (its `/api/chat` turns) reads each message against the pictures before
it, as it does for Qwen-Image-2.1; the server's Web UI sends each message to the image
routes as it is (the photos attached are the edit's, Edit again restores a turn's photos,
selection and words). In TensorAgent:

- A follow-up with no picture attached changes the newest picture in the chat. Qwen-Image
  asks its own language model which picture a follow-up means; MiniMax-H3-Image's encoder
  has no output head to answer that, so the turn changes the newest one and the page says
  it guessed (its other reading, a new picture, is one tap).
- Photos attached to a message are what it edits (the first is `<Picture 1>`, the rest
  references, a selection the first's). A message the user said changes an earlier
  picture of the chat (the reading chosen on the page) edits that picture with the
  attached photos as its references: `[picture, ...attached]`, its selection the picture's.
- Every edit keeps its source's size, and a turn asks for the model's own 2048x2048 area
  (TensorAgent's other image model, Qwen-Image-2.1, keeps its 1024x1024 budget).
- Pages know an image model by `/api/models`' `image` block (`family`, `supportsEdit`,
  `supportsMask`, `minimumArea`), so the edit area (Edit again, Select area, several
  attachments) works for MiniMax-H3-Image as it does for Qwen-Image.

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
  workaround; it went with it (stage S10), and a host refuses to start with it set.
- **One path.** The runtime is the only MiniMax-H3-Image path: the earlier native model
  (`MiniMaxH3ImageModel`, its decoder and its one-graph-a-step DiT forward) was the tests'
  oracle until stage S10 deleted it. Its outputs on the synthetic fixtures are frozen per
  backend (`InferenceWeb.Tests/Fixtures/MiniMaxH3Image/LegacyGoldens/`) and the runtime is
  held to them: byte for byte on `ggml_metal` (M5 Pro) and `ggml_cuda` (RTX PRO 4500), within
  rounding elsewhere. On ARM64 `ggml_cpu` the runtime takes the CPU products main adopted
  (weights repacked for the i8mm and dot-product kernels, BF16 products over unrounded
  activations), which the earlier model did not: up to 2.5e-3 of the largest value from it
  on the fixtures (Q4_K), a rounding difference.
- **Products.** `--diffusion-products bf16|int8|int8-all` (the config key
  `diffusion-products`; the CLI and the server) chooses the wide products' arithmetic on
  `ggml_cuda`: `bf16`, the default, runs them as stored (BF16 and Q4_K on cuBLAS BF16, other
  quantizations on ggml's kernels); `int8` puts the DiT's on int8 tensor cores (below);
  `int8-all` adds the VAE decoder's F16 projections. It replaces `TS_H3_INT8_MATMUL` (`1` is
  `int8`, `all` is `int8-all`), which a host now refuses by name. Other backends run the
  products as stored, as before, and the host says so once.

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
| RTX A4500 20 GB | Q4_K_M | 2048x2048 | the text encoder released before the denoise | 15.2 GiB |
| RTX A4500 20 GB | Q4_K_M | 4480x4480 | that, and the DiT's workspace released before the decode | 18.9 GiB |
| RTX A4500 20 GB | Q6_K | 3328x3328 | the same | 19.2 GiB |

Without these releases bf16 does not fit the A40 beside the text encoder (about 60 GB),
and Q8_0 does not fit the 48 GB Mac: with the text encoder kept, the earlier path was
stopped by the wired-memory watchdog at 43.5 GB wired while it loaded the DiT. With
bf16 on the A40 each later picture uploads the text encoder and the DiT again (about 4 s
and 15 s from disk on that VM, whose page cache cannot hold both). bf16 does not fit a
48 GB Mac at all (40.2 GB of weights against a 40.2 GB working set): use Q8_0 there. An
H100-class card keeps everything, bf16 at 16 MP included.

A picture the device cannot hold is refused before any phase runs: when its most
demanding phase needs more than the device has free even with every other component
released (that component's weights and its working memory), the CLI exits `2` with
`error: request refused: <reason>` and the server answers `507`, and the model stays
loaded for the next picture. A phase that runs out of memory inside a ggml-cuda kernel
would end the process instead. On the RTX A4500 (20 GiB, of which 20.8 GB is free after the CUDA context) the largest square
sizes that run are 4480x4480 (the model's limit) for Q3_K_M and Q4_K_M, 4096x4096 for
UD-Q4_K_XL, 3840x3840 for Q5_K_M and 3328x3328 for Q6_K; the next sizes measured past
those (UD-Q4_K_XL 4480, Q5_K_M 4096, Q6_K 3584) failed inside the denoise before the
refusal and are refused now. Q8_0 is refused at every size there: its DiT alone is
22.0 GB.

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
  and 2.166 s for pDMD); a GEMM that reads the LoRA columns from their own pointer is
  what would remove it. This is a known gap, left for a follow-up. On the M5 Pro the ops beside the product cost pDMD 10.8% and
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
| Q4_K_M, `--diffusion-products int8`, later pictures | 0.08 s | 1.88 s | 0.68 s | 12.1 s |
| Q8_0, first picture (earlier native path, the text encoder released; the runtime's pictures are the same) | 3.6-4.0 s | 2.39 s | 1.18-1.21 s | 24.1-24.8 s |
| bf16, first picture (the text encoder released before the denoise) | 3.07 s | 1.92 s | 1.20 s | 29.7 s |
| bf16, later pictures (both reloaded from disk each time) | 4.27 s | 1.91 s | 0.73 s | 34.2 s |

| A40, 4096x4096 | Text | Per step | Decode | Picture |
|---|---:|---:|---:|---:|
| Q4_K_M, first picture | 3.25 s | 11.42 s | 2.91 s | 77.8 s |
| Q4_K_M, later pictures | 0.07 s | 11.47 s | 2.40 s | 71.5 s |
| Q4_K_M, `--diffusion-products int8`, later pictures | 0.07 s | 9.96 s | 2.48 s | 62.5 s |
| bf16, first picture | 3.36 s | 10.36 s | 3.09 s | 82.6 s |

| M5 Pro, 2048x2048 | Text | Per step | Decode | Picture |
|---|---:|---:|---:|---:|
| Q8_0, first picture | 1.71 s | 12.82 s | 3.45 s | 82.6 s |
| Q8_0, later pictures | 1.70 s | 12.67 s | 2.72 s | 81.1 s |

On the same Mac the TensorAgent host (`eng/validation/TensorAgentHost`, the app's `AgentAppHost`) with the Q4_K_M
entry, every component resident, made from chat messages: a 2048x2048 picture (the app's turn area) in 89.1 s (text
3.08 s, 13.64 s a step, decode 3.93 s); with the pDMD plug-in on, a 1024x1024 picture in 7.8 s (2 steps of 3.47 s)
and a masked edit of it in 30.6 s (2 steps of 10.6 s; every unselected pixel kept). Peak 39.2 GB wired, the
process's peak footprint 4.38 GB.

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
rows each); the text edit has one 2048x2048 reference (4,096 rows), the composition two (or two 1024x1024 ones), and
MiniMax's example is `examples/edit.py`'s picture and prompt at 2048x2048.

| A40, Q4_K_M | Rows (text + references + target) | Text | Vision | VAE encode | Per step | Decode | Edit | Peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 reference | 1,076 + 1,024 + 4,096 | 3.19 s | 1.12 s | 0.93 s (25 tiles) | 3.44 s | 1.15 s | 33.0 s | 32.8 GiB |
| 4 references | 4,189 + 4,096 + 4,096 | 8.47 s | 1.98 s | 2.63 s (100) | 9.35 s | 1.13 s | 74.2 s | 36.1 GiB |
| 16 references | 16,635 + 16,384 + 4,096 | 38.88 s | 6.15 s | 9.91 s (400) | 49.15 s | 1.54 s | 356.3 s | 42.5 GiB |
| text edit (1 at 2048x2048) | 1,091 + 4,096 + 4,096 | 4.07 s | 0.93 s | 3.05 s (121) | 5.48 s | 1.24 s | 46.0 s | 33.7 GiB |
| composition (2 at 2048x2048) | 2,125 + 8,192 + 4,096 | 5.44 s | 1.53 s | 6.60 s (242) | 9.62 s | 1.19 s | 76.3 s | 36.1 GiB |
| composition (2 at 1024x1024) | 2,125 + 2,048 + 4,096 | 5.63 s | 1.30 s | 1.38 s (50) | 4.87 s | 1.15 s | 42.5 s | 34.3 GiB |
| MiniMax's example (1 at 1024x1024) | 1,459 + 1,024 + 4,096 | 4.82 s | 0.98 s | 0.80 s (25) | 3.70 s | 1.09 s | 33.6 s | 33.2 GiB |

| A40, bf16 | Rows | Text | Vision | VAE encode | Per step | Decode | Edit | Peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 reference | 1,076 + 1,024 + 4,096 | 4.43 s | 1.03 s | 0.75 s (25) | 3.01 s | 1.19 s | 41.1 s | 41.0 GiB |
| 4 references | 4,189 + 4,096 + 4,096 | 8.37 s | 2.20 s | 2.77 s (100) | 8.64 s | 1.36 s | 83.2 s | 41.8 GiB |
| 16 references | 16,635 + 16,384 + 4,096 | refused before it runs: a step needs about 52.1 GB of the 44.4 GB budget | | | | | | |
| text edit (1 at 2048x2048) | 1,091 + 4,096 + 4,096 | 4.64 s | 0.94 s | 3.14 s (121) | 4.89 s | 1.14 s | 55.0 s | 40.4 GiB |
| composition (2 at 2048x2048) | 2,125 + 8,192 + 4,096 | refused before it runs: a step needs about 44.9 GB of the 44.4 GB budget | | | | | | |
| composition (2 at 1024x1024) | 2,125 + 2,048 + 4,096 | 5.94 s | 1.32 s | 1.37 s (50) | 4.29 s | 1.17 s | 51.4 s | 41.1 GiB |
| MiniMax's example (1 at 1024x1024) | 1,459 + 1,024 + 4,096 | 4.96 s | 1.05 s | 0.76 s (25) | 3.20 s | 1.21 s | 42.2 s | 41.1 GiB |

bf16's first step includes the 40 GB DiT's upload (the text encoder is released before it, as for text to image), which
the edit times include. A text-to-image step at 2048x2048 is 2.2 s (Q4_K_M) and 1.9 s (bf16) on the same card
([above](#a40-and-m5-pro)); a step's cost grows with the rows, and attention with their square. Sixteen 1024x1024
references keep everything resident on the A40 with Q4_K_M (the planner drops only the text encoder's workspace before
the denoise); with bf16 the same edit is refused before it runs, by the row budget, with the message naming its rows,
as is a composition of two 2048x2048 references (two 1024x1024 ones fit). The text encoder materializes an edit's
attention scores a block of query rows at a time once they pass 1 GiB (about 2,000 rows): 1,001 rows a block at four
references, 252 at sixteen.

| M5 Pro, Q8_0 | Rows | Text | Vision | VAE encode | Per step | Decode | Edit | Peak (allocated) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 reference | 1,076 + 1,024 + 4,096 | 3.59 s | 1.01 s | 3.54 s (25) | 19.98 s | 3.74 s | 134.3 s | 28.3 GB |
| 4 references | 4,189 + 4,096 + 4,096 | 19.63 s | 3.91 s | 14.11 s (100) | 51.55 s | 3.83 s | 352.4 s | 33.0 GB |
| 16 references | 16,635 + 16,384 + 4,096 | refused before it runs: a step needs about 49.8 GB of the 37.7 GB budget | | | | | | |
| text edit (1 at 2048x2048) | 1,091 + 4,096 + 4,096 | 5.18 s | 1.25 s | 16.91 s (121) | 33.77 s | 3.81 s | 231.0 s | 30.7 GB |
| composition (2 at 2048x2048) | 2,125 + 8,192 + 4,096 | 10.60 s | 2.15 s | 33.90 s (242) | 62.57 s | 3.78 s | 427.4 s | 34.5 GB |
| MiniMax's example (1 at 1024x1024) | 1,459 + 1,024 + 4,096 | 7.77 s | 1.25 s | 3.52 s (25) | 21.47 s | 3.81 s | 146.3 s | 28.4 GB |

The refusals above are stage S5's, which checked an edit against the plan's budget (free memory less a headroom).
Since then an edit is checked as every request is, against the device's free memory, and an edit's step estimate counts
two more F32 rows of the hidden width a packed row (calibrated to the A40 peaks above: bf16 with four references measured
1.08x the earlier estimate, Q4_K_M with sixteen 1.1 GB over its plan). By that check bf16 with sixteen references still
needs about 53.7 GB of the A40's 48.0 GB and Q8_0 with sixteen about 51.4 GB of the M5 Pro's 40.2 GB, so both are refused;
bf16's composition of two 2048x2048 references needs about 45.5 GB, is no longer refused, and has not been run since.

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
| TensorSharp, `--diffusion-products int8` (`minimax-h3-image-int8`) | 0.09 s | 2.55 s (0.43 s/step) | 0.27 s | 3.88 s |

| 4096x4096 (16 MP) | Text | Denoise, 6 steps | VAE decode | ComfyUI job |
|---|---:|---:|---:|---:|
| ComfyUI template | | 14.7 s (2.45 s/step) | | 19.9 s |
| TensorSharp, bf16 projections | 0.09 s | 15.8 s (2.63 s/step) | 0.95 s (441 tiles) | 19.6 s |
| TensorSharp, `--diffusion-products int8` (`minimax-h3-image-int8`) | 0.09 s | 13.1 s (2.18 s/step) | 0.9 s | 16.9 s |

The job is the ComfyUI prompt through the TensorSharp Image node: TensorSharp's picture
plus its PNG (under 0.1 s at 4 MP, 0.1 to 0.4 s at 16 MP), the node's download and
decode, and ComfyUI's SaveImage, which a ComfyUI-only job pays as well (2.0 s for a
16 MP PNG). The first job of a server (weights from the page cache) took 9.1 s with int8
and 9.5 s with bf16 at 4 MP, against 55.1 s for ComfyUI's first job.

- **Projections.** A BF16 DiT's wide products (qkv, out, fc1, fc2) run on cuBLAS with
  the weight as stored; `--diffusion-products int8` (measured as `TS_H3_INT8_MATMUL=1`, the
  same route, before stage S10 renamed it) rounds each weight row and each token
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

## TensorAgent

The TensorAgent app lists two entries, under MiniMax H3 on its Models page:

| Entry | DiT | Download | Offered from |
|---|---|---|---|
| `minimax-h3-image-q4km` | Q4_K_M | 12,721,119,968-byte DiT + 18,218,065,024-byte text encoder + 1,168,505,312-byte VAE + `vocab.json`, `merges.txt`, `tokenizer_config.json` (4.46 MB); 32.1 GB | 32 GB of system memory |
| `minimax-h3-image-q8-0` | Q8_0 | 22,026,760,928-byte DiT + the same companions; 41.4 GB | 48 GB |

The files are those of `minimax-h3-image-q4km.json` and `minimax-h3-image-q8.json`, at the same
pinned revisions and SHA-256. The text encoder and tokenizer files are the MiniMax-H3 video entry's,
byte for byte: an installation that has one entry links them from it instead of downloading 18.2 GB
again. With the model loaded, a message is a description of a picture, or, with photos attached,
an edit that names them `<Picture 1>`, `<Picture 2>`, ... in order, as for Qwen-Image 2.1.

- **Tiers.** Q4_K_M's largest phase is the text encoder's (18.2 GB of weights and 0.6 GB of working
  memory); the pipeline releases it before the denoise wherever the GPU's working set cannot hold
  both, as on a 32 GB Mac (about 22.9 GB). Q8_0's denoise needs 22.0 GB of weights and 3.1 GB of
  working memory at 2048x2048, past that working set, so it starts at 48 GB (measured on an M5 Pro
  48 GB: 29.9 GB wired).
- **Not marked experimental.** Their quality against the bf16 DiT is measured
  ([Quantizations](#quantizations)) and stated on each row: Q4_K_M is the furthest of the four and
  can add stray lettering; Q8_0 is the closest.
- **Pictures and the plug-in.** A turn asks for the model's own 2048x2048 area ([Multi-turn](#multi-turn)). The LoRA
  sheet offers one plug-in for both entries, pDMD 2-step (linoyts @c07d96e5, under the Community License): with it a
  picture takes 2 steps and the turn asks for 1024x1024, the size it was distilled at. It is a Model Derivative under
  the same license, so it downloads only once that license is accepted: an acceptance of the current text for either
  entry covers it (the one their own download asked for), and without one its download is refused before anything is
  fetched (`POST /api/agent/loras/{id}/download` answers `409` with `licenseRequired`, the entry to accept it for and
  its terms; a decline of that entry withdraws the cover). The other MiniMax-H3-Image
  plug-in configs (akhaliq's ComfyUI conversion, the Seedreamstyle adapter) are for the CLI and server only (decision
  D7).
- **License acceptance.** MiniMax-H3-Image's license limits where and by whom it may be used, so
  before an entry's first download the Models page shows its terms: a summary in the interface
  language (the territory exclusion of the United States, the European Union, the United Kingdom
  and the Republic of Korea; MiniMax's written authorization past USD 1 million of annual revenue or
  for a hosted service; "MiniMax-H3-Image" shown in a commercial product; the use restrictions; a
  copy of the license and a NOTICE file with a redistribution; Hong Kong law), then the license
  text itself, verbatim, and a link to it on Hugging Face. **Accept and download** starts the
  download; **Decline**, or closing the page, starts nothing.
  - The answer is kept in `license-acceptances.json` in the app's data folder (not its cache), per
    entry and per license revision: the SHA-256 of the text the app ships
    (`0ec3bb5e...`, MiniMaxAI/MiniMax-H3-Image at `8ced6006`; Unsloth's repository carries the
    version before MiniMax's 2026-11-11 change, which differs in the authorization link only).
    Each entry asks for itself; a catalog that ships a new text asks again and says the terms
    changed; a decline withdraws an acceptance of the current text.
  - The download manager refuses an entry whose terms are not accepted, whoever starts it: the
    Models page's buttons, a resume when the app comes back, the desktop host's HTTP route.
    There, `POST /api/agent/catalog/{id}/download` answers `409` with `licenseRequired` and the
    terms, `GET /api/agent/catalog/{id}/license` returns them with the text, and
    `POST /api/agent/catalog/{id}/license` with `{"revision": "...", "accept": true}` records an
    acceptance (`409` when the revision is not the current one) and `"accept": false` a decline
    (which also stops the entry's running download). `/api/agent/catalog` describes each entry's
    `licenseTerms` (null for every entry that asks nothing).
  - Every other entry downloads as before, with no question.

## Not implemented

- A LoRA on bf16 weights on `ggml_cuda` at the cost of a quantized one: it costs about 10%
  a step now (each weight's copy into the K + R rows); the fix is a GEMM that reads the LoRA
  columns from their own pointer.
- LoRA merging (the updates always run unmerged), DoRA, LoCon, LoHa and LoKr adapters,
  and a per-module alpha or rank pattern in a PEFT config.
- The chat planner's question for this model (which picture a follow-up changes): its
  text encoder has no output head, so a follow-up changes the newest picture.
- Mask-aware modulation of the kept rows (modulating them as clean, as ComfyUI's H3 video
  does): designed as an off-by-default experiment, not built or measured.
- The prompt expander (MiniMaxAI/MiniMax-H3-Image-PE). ComfyUI's text-to-image
  template leaves it off by default.
- The `mlx`, `cuda` and `cpu` (pure C#) backends.
