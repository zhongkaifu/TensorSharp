# MiniMax-H3

MiniMax-H3 generates video **and native stereo audio together**, up to 15 s at
24 fps with 32 kHz stereo. It is not a video model with audio bolted on: one
diffusion transformer denoises a packed video+audio latent in a single token
sequence, so the soundtrack is part of the model output rather than something
added afterwards.

TensorSharp runs it as **seven** native whole-network ggml graphs — text encode,
vision encode, the DiT forward, and encode + decode for each of the two VAEs —
with weights bound resident straight from the GGUF/safetensors mmap. The public
entry point is `MiniMaxH3Model.GenerateVideo(prompt, VideoGenerationParams)`,
behind the shared `IVideoGenerationModel` seam, so the CLI and the server drive
H3 and Wan down one path instead of type-testing each concrete model.

```sh
TensorSharp.Cli --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --prompt "a red fox trotting through falling snow, cinematic" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output fox.mp4
```

That writes `fox.mp4` plus `fox.wav` with the generated soundtrack.

## Status

| Component | State |
|---|---|
| Text-to-video (`t2v`) | working |
| Image-to-video (`i2v`) — animate a photo | working |
| First/last frame (`fl2v`) | working |
| Reference-to-video (`ref`) — images, clips and soundtracks | working |
| Joint stereo audio | working |
| Video VAE encode + decode | working, tiled |
| Clips past 22 frames | working (the VAE decodes 5 latent frames at a time) |
| `--backend cpu` (100% pure C#) | working: t2v, i2v, fl2v and reference conditioning |

### The pure-C# backend

`--backend cpu` runs H3 with no ggml at all, through
`MiniMaxH3Direct{DiT,TextEncoder,VideoVae,AudioVae}` on the shared
`TensorSharp.Models/Direct` primitives. That covers the whole text-to-video path:
prompt encoding, denoising, video VAE decode and the audio vocoder.

The vision tower, the causal 3-D video VAE encoder and the audio VAE encoder are
ported too, so every conditioning route works: `i2v`, `fl2v`, and reference
images, clips and soundtracks. Everything runs at F32 on the host, so expect it to be slower than `ggml_cuda`;
it is there for correctness, portability and machines with no accelerator. A
256x160x5f single-step t2v measured 69 s against 14 s on `ggml_cpu`; the same
i2v measured 70 s against 176 s, because the managed 3-D encode costs far less
than the GGML one (single sample - do not read too much into the direction).

Measured against the GGML path on identical inputs (256x160, 5 frames, one step,
fixed `--diffusion-seed`):

| stage | cosine | relative RMS |
|---|---|---|
| text encoder (64 layers, 32B Q4_K_M) | 0.99999899 | 0.147% |
| DiT, audio velocity | 0.9994631 | 3.31% |
| DiT, video velocity | 0.9975865 | 7.60% |
| i2v (vision tower + 3-D VAE encode + DiT) | 0.9998974 | - |
| ref-audio (audio VAE encode + DiT) | 0.9994101 | 4.32% |
| ref-image (vision tower + DiT) | 0.9985535 | - |
| vision tower output, on its own | 0.9999189 | 1.28% |

The vision tower is the one stage whose residual is LARGER than GGML's own
internal spread rather than smaller: measured on the tower output directly, the
managed path sits at 1.28% relative against GGML's flash kernel and 1.40% against
its explicit-softmax fallback, while those two GGML kernels differ from each other
by 0.99%. Turning flash off makes the managed agreement slightly WORSE, so the F16
K/V cast does not explain it. At cosine 0.9999 over 737k elements the tower is
structurally right - a wrong tower does not land near 1 - but the residual is
roughly 1.4x the control rather than below it, and it has not been chased further.
It is worth knowing before trusting reference-image conditioning to the managed
path for anything precision-sensitive.

On the finished render, measured as PSNR against the GGML path: t2v 31.95 dB,
i2v 34.87 dB, reference-audio 35.27 dB. The control that makes those numbers
readable is GGML against ITSELF - its flash kernel vs `TS_H3_NO_FLASH=1` renders
at 28.17 dB, i.e. every managed route agrees with GGML more closely than GGML's
two attention kernels agree with each other.

The trunk is not bit-identical to GGML and cannot be, because the denoiser
amplifies tiny differences. `TS_H3_DIT_LAYERS` truncates the trunk on BOTH paths,
which turns the velocity comparison into an error-vs-depth curve:

| trunk depth | 1 | 10 | 25 | 30 | 35 | 40 | 44 | 47 | 50 |
|---|---|---|---|---|---|---|---|---|---|
| 1 - cosine | 1.4e-6 | 3.8e-6 | 9.4e-6 | 2.8e-5 | 2.8e-4 | 1.15e-3 | 1.5e-4 | 2.0e-4 | 1.26e-3 |

Through 25 layers the two agree to ~1e-5, i.e. the implementation is right. Past
that the difference amplifies, and NON-MONOTONICALLY - it is larger at depth 40
than at 44. That shape is what rules out a bug: an error introduced at some layer
would leave the curve non-decreasing for every depth beyond it. The second half of
this trunk simply has high gain.

For scale, GGML disagrees with ITSELF by more than it disagrees with the managed
path. At full depth, `h3_attend`'s flash kernel against its own explicit-softmax
fallback (`TS_H3_NO_FLASH=1`) gives 1 - cosine = 2.97e-3, against 1.26e-3 for
managed-vs-either. The text encoder, which is shallower in effect and has no such
kernel on either side, agrees to 1e-6 - that is what says the shared machinery
(quantized matmul, RMSNorm, rotate-half RoPE, GQA attention, SwiGLU) is right.

`TS_H3_DUMP_TE`, `TS_H3_DUMP_VEL_V` and `TS_H3_DUMP_VEL_A` write those tensors to
disk so the two paths can be compared on ONE forward, and `TS_H3_DIT_LAYERS`
truncates the trunk so that comparison becomes a curve. Do not compare finished
clips: the sampler amplifies whatever the first step differed by, so a few-step
render says nothing about whether an implementation is correct. Note also that
`--seed` is the SAMPLING seed; video generation draws its noise from
`--diffusion-seed`, and comparing two runs without it silently compares different
noise.

### Long clips and the FP16 attention ceiling

H3 attends bidirectionally over ONE packed sequence, so its key count is the whole
clip rather than a window: a 640x384 request is 2364 packed tokens at 22 frames but
8646 at 107. ggml's flash-attention kernels keep the softmax numerator in FP16 and
give it three bits of headroom, so that key count is a numeric budget, and H3 is the
model that spends it - `v_norm` does not exist in the checkpoint, so the value stream
carries the same unbounded magnitudes that `h3_mm` guards against one op later. Left
alone, a 107-frame clip overflowed to NaN on the first denoise step and came back
with every pixel black and every audio sample clamped, while 73 frames of the same
request rendered correctly.

`h3_attend` now pre-scales V by a power of two derived from the key count and undoes
it on the output - exact, because attention is linear in V - so the accumulator sees
a bounded number of effective keys at any clip length. The sampler also refuses to
write a diverged clip: a non-finite velocity fails the request naming the step it
appeared at, rather than saving a black file. Set `TS_H3_TRACE=1` to print the latent
and velocity magnitudes for every step.

## Performance

Two machines, two answers. Both are 22 frames, 8 steps, identical seed, against
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) at its
best-performing configuration on that machine, with the same weight files on both
engines. They are reported separately and deliberately not averaged — the CUDA machine is
memory-starved against this model set in a way the Metal one is not, and that is where
they part company.

### M5 Pro / Metal

| Resolution | stable-diffusion.cpp | TensorSharp | |
|---|---|---|---|
| 256×256 | 49.3 s | **20.9 s** | 2.4× faster |
| 640×384 | 108.5 s | **63.1 s** | 1.7× faster |

### RTX 3080 Laptop 16 GB / CUDA

Shipped build on `ggml_cuda`, best of three. Driver 566.36, CUDA 12.6, i7-11800H,
31.7 GB RAM, Windows 11, .NET 10.0.204.

| Resolution | stable-diffusion.cpp | TensorSharp | |
|---|---|---|---|
| 256×256 | **37.8 s** | 43.6 s | sd.cpp 1.15× faster |
| 640×384 | **59.8 s** | 63.7 s | sd.cpp 1.07× faster |

| Engine | Peak VRAM, 640×384 |
|---|---|
| TensorSharp | 15 780 MiB |
| stable-diffusion.cpp | 12 035 MiB |

The card is 16 384 MiB. stable-diffusion.cpp was `97d2990` with ggml `8e800ce`, rebuilt
from source with `SD_CUDA=ON` for arch 86 and run with
`--auto-fit --stream-layers --diffusion-fa --rng cpu` — its default `--offload-to-cpu`
path cannot run this model on this machine at all, because it tries to pin 17.7 GB into
12.3 GB of free RAM.

### Why the two disagree

Per **denoise step** TensorSharp is ahead on CUDA as well — 3.325 s against 3.338 s by
the 8-versus-16-step slope, and 3.00–3.11 s measured directly per step. What it loses on
CUDA is fixed setup cost, and roughly 3 s of the remaining 3.9 s is not inference at all:
TensorSharp encodes H.264 where sd.cpp writes MJPEG + PCM into an AVI, and a .NET process
starts against a native binary.

The two hardware points differ because the CUDA machine is a deliberately hostile one —
16 GB of VRAM and 31.7 GB of RAM against a ~35.5 GB model set, so neither the weights nor
the page cache fit and setup dominates in a way it would not on a card that holds the
model. Both numbers are real. Which one describes your run depends on how much memory the
machine has, so check the hardware label before quoting either.

### What this round bought — RTX 3080 Laptop 16 GB / CUDA

Same machine, same workload, before and after the load-time work in
[VRAM across four networks](#vram-across-four-networks):

| Resolution | before | after | |
|---|---|---|---|
| 256×256 | 67.2 s | **43.6 s** | 1.54× |
| 640×384 | 89.0 s | **63.7 s** | 1.40× |

## VRAM across four networks

Four networks run in sequence, so peak VRAM is `max(...)` rather than the sum — but only
if each one is handed back before the next arrives, and only if getting the next one onto
the card is not itself the slow part. On a 16 GB card neither was free. Everything below
was measured on the RTX 3080 Laptop 16 GB above; `TS_H3_PHASE=1` prints the per-stage
breakdown these numbers come from.

### The denoiser is released before the video VAE loads

The denoiser is ~10.6 GB at Q4_K and the video VAE another ~5.2 GB. Resident together on
a 16 GB card that is 15.8 GB of weights before a single activation, and Windows/WDDM does
not fail the allocation — it silently backs the overflow with shared host memory, so the
whole decode runs at PCIe speed. The decode window sat pinned at 16 041 MiB of a
16 384 MiB card for its entire duration.

The finished denoiser's device residency is now handed back first, when the VAE would not
fit beside it. Peak VRAM during decode fell to about 5 600 MiB, worth **22 s** at
640x384. It is a cap, not a policy change: the decision reads actual free VRAM, so a card
with room keeps the denoiser resident and behaves exactly as before, and non-CUDA GGML
backends are left alone — a unified-memory device has nothing to hand back. The cost is
re-uploading the DiT on the *next* request only, since its weights are mmapped GGUF pages
that are still in RAM. The sibling Qwen-Image pipeline already worked this way.

### The denoiser file is read before it is uploaded

Weights are bound as pointers into the mmapped GGUF, so the first upload faulted every
page in from disk *from inside* the host-to-device copy — the driver's worst case:

| Path | Rate | 10.6 GB would take |
|---|---|---|
| H2D from pinned host memory | 19.09 GB/s | 0.56 s |
| H2D from pageable resident memory | 5.97 GB/s | 1.78 s |
| Plain sequential read of the file | 2.70 GB/s | 3.93 s |
| **H2D faulting pages in as it copies** | **0.91 GB/s** | **11.6 s** |

So the denoiser file is now read through sequentially first. Two details make it pay:

* **It starts early.** The read is kicked off the moment the text trunk produces its
  hidden states, so it runs alongside the encoder's teardown — handing back 551
  per-tensor device buffers is driver work and the read is disk, and they contend for
  nothing.
* **It is not joined before the upload.** The read and the upload walk the file in the
  same order and the read is the faster of the two, so leaving it running keeps it ahead
  and the copy lands on pages that are already there. Joining first would serialize a
  whole read in front of a whole copy.

First denoise step went from 14.87 s to about 10.2 s, and denoiser weight movement from
11.8 s to 8.25 s. Output is **byte-identical** with the read on and off.

That setup cost was confirmed token-independent before the fix — 11.77 s at 256x256
against 11.55 s at 640x384 — which is what established it as weight movement rather than
graph construction. `TS_H3_PREFAULT` selects the mode; see
[Environment variables](#environment-variables).

## Files

Four networks cooperate, so a run needs four files. The two denoisers are
**separate checkpoints, not settings** — which one you load decides what
conditioning it accepts.

| File | Size | Source |
|---|---|---|
| `minimax_h3_fl2va_pruned-Q4_K.gguf` | 10.64 GiB | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) — text + keyframes |
| `minimax_h3_ref2va_pruned-Q4_K.gguf` | 10.60 GiB | same repo — text + references |
| `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | 16.97 GiB | same repo — the text encoder, shared (`-Q2_K_M.gguf`, 12.20 GiB, pairs with a smaller denoiser) |
| `minimax_h3_video_vae_fp16.safetensors` | 5.21 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) (also mirrored under `vae/` in the unsloth repo) |
| `minimax_h3_audio_vae_fp32.safetensors` | 0.61 GB | same repo — omit for silent video |

About 35.5 GB for one set (35.4 GB for Ref2VA). Each network is loaded and **released** in turn, so
peak VRAM is `max(...)`, not the sum; adding the second denoiser later costs only
its own ~10.6 GiB, because the encoder and both VAEs are shared.

Companions resolve automatically next to the denoiser, or with `--video-vae`,
`--video-text-encoder`, `--audio-vae`.

Two shipped configs download the lot on first run and only differ in their `model`
entry, so the encoder and the VAEs are fetched once:

```sh
TensorSharp.Server.Host --config config/minimax-h3-fl2va.json # keyframes
TensorSharp.Cli         --config config/minimax-h3-ref2va.json \
  --ref-image person.png --prompt "…" --output out.mp4        # references
```

Both name `"backend": "ggml_cuda"` (a `--backend` on the command line wins) and set
width 640, height 384, 22 frames, 24 fps. Neither sets steps or guidance: a shipped
config has to parse on both hosts, and the two spell steps differently
(`--video-steps N` on the server, `--diffusion-steps N` on the CLI), while the
server takes no `--cfg` at all — so the model's own defaults apply.

> **The text-encoder GGUF carries no tokenizer**, and that is the one thing a config
> cannot fetch for you: auto-download fills in options that are **flags**, and the
> tokenizer is not one. Put `vocab.json` and `merges.txt`
> from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/processor)
> beside it, or point `TS_VIDEO_TOKENIZER` at them. Neither published GGUF has any
> metadata at all, so TensorSharp identifies H3 by its tensors rather than by an
> architecture string (arch keys `minimax-h3` / `minimax_h3` are accepted when a
> file does declare one).

> **`--cfg 1.0` is required.** H3 is CFG-distilled; above 1.0 it degrades badly and
> TensorSharp refuses. 4–8 steps is the operating point, so iteration is fast.

## Conditioning modes

An image can mean two completely different things to H3, and they use different
checkpoints. `--video-mode` makes the choice explicit; without it the mode is
inferred from what you pass.

| What you want | Mode | Checkpoint |
|---|---|---|
| "animate this photo" | `i2v` | FL2VA |
| "go from photo A to photo B" | `fl2v` | FL2VA |
| "use this person, brand-new scene" | `ref` | Ref2VA |
| "reference this product, new angle and background" | `ref` | Ref2VA |
| text only | `t2v` | either |

Which partition is active is read off the **file name** — `ref2va` anywhere in it,
case-insensitively — so keep that substring if you rename or requantize. Asking one
checkpoint for the other's mode does not silently drop the input: the request fails
with a message naming the file to load instead. Keyframes and named references in
the same request are refused outright, on either checkpoint.

### Text to video

```sh
TensorSharp.Cli --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --prompt "a red fox trotting through falling snow, cinematic" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output fox.mp4
```

### Image to video — animate a photo

The image becomes the **first frame** and the prompt drives what happens next.

A keyframe is conditioned **twice**, and both halves matter: the VAE-encoded latent
pins the frame it is anchored to, and the same picture goes through the Qwen3-VL
vision tower into the prompt, where it describes the scene for the whole clip. The
latent alone fades — the clip starts on the image and wanders off within a second
or two, which is invisible at 22 frames and obvious at 124.

```sh
TensorSharp.Cli --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --image portrait.jpg \
  --prompt "the person turns toward the camera and smiles, subtle handheld motion" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output animated.mp4
```

`--video-mode i2v` states it explicitly. The image is fitted to the generation
canvas; when its aspect ratio differs from `--width`/`--height` it is
centre-cropped (the run prints a `[h3] conditioning image … centre-cropping to
fit` line), so leave the size unset to keep the whole image.

### First and last frame

Both ends are pinned and the model fills in the motion between them.

```sh
TensorSharp.Cli --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --image start.png --end-image end.png \
  --prompt "a slow cinematic push-in" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output morph.mp4
```

`--end-image` alone also works: the clip ends on that frame.

### Reference to video — keep the subject, change everything else

Ref2VA treats an image as an identity and appearance **reference**, not as a frame.
The first frame need not resemble it at all: the person, product or object carries
over while the camera, background and composition come entirely from the prompt.

This needs the **Ref2VA** checkpoint — `i2v`/`fl2v` and `ref` are separate files,
not settings.

```sh
TensorSharp.Cli --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-image person.jpg \
  --prompt "the same woman sits at a table in a sunlit cafe by a window, drinking coffee, wide shot" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output cafe.mp4
```

Pass `--ref-image` more than once for several references (up to nine), for example
a person and the product they are holding:

```sh
TensorSharp.Cli --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-image person.jpg --ref-image bottle.png \
  --prompt "she holds the bottle up to the light on a rooftop at golden hour, slow orbit" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output rooftop.mp4
```

Notes:

* References are only ever scaled **down**, to the generated clip's area, and keep
  their own aspect ratio. Nothing is stretched to the output canvas, and the output
  canvas is *not* taken from the reference — say what you want with
  `--width`/`--height`.
* **Each reference is extra tokens in the same packed sequence**, so the cost is
  linear and easy to predict: a 640x384 reference is 240 tokens, one at 576x448 is
  252. Measured on an RTX 3080 Laptop against a 640x384 22-frame clip, a denoise step
  goes from 4.37 s with no references to 9.38 s with eight — about 626 ms per
  reference per step, flat from one to eight.
* **Past about four references the Qwen3-VL pass dominates, not the denoiser.** Every
  reference adds roughly 250 vision placeholder tokens to the prompt, and that prompt
  is prefilled through all 50 layers: two references make a 548-token prompt and eight
  make 2086, so text conditioning grows from ~65 s to ~447 s while the whole 8-step
  denoise costs ~75 s. Reach for fewer, better references before reaching for fewer
  steps.
* TensorSharp caps reference images at nine. The reference implementation has no cap;
  this one exists because the packed sequence is attended over unmasked and its length
  is a numeric budget as well as a time one.
* References are **labelled in the order you pass them** before the language model
  sees them — `<Picture 1>`, `<Picture 2>` … for stills, `<Video 1>` … for clips,
  `<Audio 1>` … for soundtracks — so a prompt can name one: "the jacket from
  `<Picture 2>`". A clip that arrived with its own soundtrack is presented as
  *two* items and takes an `<Audio n>` label as well as a `<Video n>` one.
* Say who is in frame. The reference supplies identity; the prompt still has to
  describe the shot, or you get a well-rendered scene with the subject missing.
* A reference can also be a **clip** or a **soundtrack**, not just a still — see
  below.

> **A plain `--image` on the Ref2VA checkpoint is treated as a reference.** Clients
> that only know how to attach "an image" — the Web UI among them — therefore work
> against Ref2VA without any extra field. Pass `--video-mode ref` to be explicit.

### Reference clips and soundtracks

`--ref-video` takes a video **file** or a directory of frames, and `--ref-audio` a
WAV/MP3/Ogg. A clip's own soundtrack goes in separately with `--ref-video-audio`,
paired by position with `--ref-video` — a container's audio track is not readable
through the frame decoder, so the two arrive as separate inputs.

```sh
TensorSharp.Cli --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-video walk.mp4 --ref-video-audio walk.wav \
  --prompt "the same woman walks along a beach at sunset, wide shot, waves behind her" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output beach.mp4
```

A standalone soundtrack, with no picture attached, is a reference too:

```sh
TensorSharp.Cli --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-image singer.jpg --ref-audio song.wav \
  --prompt "she performs on a small club stage under a single spotlight" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output stage.mp4
```

What happens to a reference clip:

* It is resampled onto H3's own **24 fps** and pulled down to the 17k+5 frame grid,
  capped at the number of frames being generated. A clip shorter than 5 frames at
  24 fps is rejected.
* Its canvas comes from its own aspect ratio around a nominal 768 px, capped by
  area and snapped to 32. A source **smaller** than that keeps its own size:
  upscaling invents no detail to reference and costs a great deal — a 448x320 clip
  blown up to 1088x768 would be 5712 conditioning tokens against 1680 for the clip
  being generated.
* The language model sees it at **2 fps**, two frames at a time, each pair labelled
  with the time it sits at. The denoiser sees all of it, as latents.
* A soundtrack is resampled to the audio VAE's 32 kHz stereo and truncated to the
  generated clip's duration.

> **A reference clip is the most expensive input H3 takes.** A 22-frame 448x320
> reference adds 980 conditioning tokens on top of the 1680 the output itself
> needs, and the VAE has to encode all 22 frames first. Measured on an M5 Pro:
> 14 s to encode the reference, then about 9 s per denoising step against 5 s
> without it.

## Getting good quality

Two settings dominate; everything else is a rounding error next to them.

| Lever | Default | What it does |
|---|---|---|
| `--width` / `--height` | 640×384, or the conditioning image's aspect at that area | **The single biggest factor.** Faces need pixels: at 256×256 a face is a few dozen pixels across and comes back blurry and malformed no matter what else you set. |
| `--diffusion-steps` | 20 | Removes chromatic fringing around moving subjects. 8 is the fast lane, ~20 is clean, past ~30 gains little. Time scales roughly linearly. |
| `--diffusion-seed` | random | Some seeds simply compose better. Cheapest thing to retry. |
| `--flow-shift` | 12 | Moves where the schedule spends its steps. Rarely needs touching. |
| model quant | — | `-Q8_0` over `-Q4_K` for the denoiser, and `qwen3vl_32b_minimax_h3-Q4_K_M` over `-Q2_K_M`, both help if you have the memory. |

### Frame the subject — this dominates everything else

**How much of the frame your subject occupies matters more than every setting
below combined.** The model renders what it is given at the resolution it is
given; if a face is 40 pixels across in the conditioning image, no number of
steps and no output size will recover it.

Measured on one photo of a park pass with two small printed portraits, same
prompt, same seed, 20 steps:

| Input | Faces in the output |
|---|---|
| the photo as shot, 640x384 | ~40 px across — mushy, unrecognizable |
| cropped to the two figures, 768x480 | ~250 px across — hair, skin texture, lapel pin all legible |

Nothing changed but the crop. If the thing you care about is a face, crop the
source photo to it before generating.

> **A conditioning image is never stretched.** When the canvas aspect differs
> from the image's, the image is centre-cropped to fit and the run says so. A 4:3
> photo forced into 640x384 (5:3) would otherwise be squeezed 25% horizontally,
> and since the keyframe *is* the first frame, that distortion is baked into
> every frame after it. Leave `--width`/`--height` unset to take the canvas from
> the image instead, and crop the source yourself when you want to control what
> is in shot.

`--cfg` is **not** a quality lever here — H3 is CFG-distilled and only accepts 1.0.
`--negative-prompt` does nothing for the same reason: there is no unconditional
pass to steer away from, and `--cfg-cache-stride`, which caches that pass, has
nothing to cache. `--sampler` is a Wan-family knob; H3 runs its own flow-match
schedule and ignores it.

Measured on the same image-to-video request (M5 Pro, `ggml_metal`, seed 42, 22 frames):

| Setting | Time | Result |
|---|---|---|
| 256×256, 8 steps | 26 s | blurry, faces unrecognizable |
| 640×384, 8 steps | 79 s | sharp, but coloured fringing around moving hands |
| 640×384, 24 steps | 212 s | clean |
| 576×448 (image aspect), 20 steps | 187 s | best — nothing stretched |

> **On the server, size is a startup flag.** The Web UI sends only the prompt and
> the image, so requests inherit the server's defaults. Start it with
> `--video-width 640 --video-height 384 --video-steps 20 --video-frames 22` (or omit
> width and height and let each request pick the aspect from its image; give only
> one of the two and H3 takes the other from the conditioning image). `--width` /
> `--height` are accepted as aliases, and `--video-mode` pins the conditioning mode
> for a deployment that only offers one. The server has **no `--cfg`** — there is
> nothing to set, since H3 enforces 1.0 itself.

> **Forcing a size that does not match your image crops it.** A 4:3 photo forced
> into 640×384 loses about 20% of its height, split between top and bottom. Leave
> width/height unset for image-to-video and the aspect is taken from the image.

> **The soundtrack changed in this revision.** The denoiser emits a whitened audio
> latent and the decoder wants the VAE's own scale; the un-whitening step was
> missing, so every generated track was decoded from a latent about 1.9x too small.
> It was audible only as a wrong level and a slightly wrong timbre, never as
> silence or noise. Verified against the reference implementation's own decode of
> the same latent: cosine 0.99998 at 1.0000x gain, from 0.81x before.

## Sizes and lengths

* Width and height are rounded **up** to a multiple of 32. Default 640×384, or
  that area at the conditioning image's aspect ratio.
* Frame count is rounded up onto the **17k+5** grid — 5, 22, 39, 56, 73, 90, 107,
  124 … — which comes from the video VAE's temporal chunking, not from an
  arbitrary choice: each 5-latent-frame chunk yields 17 pixel frames, on top of a
  5-frame lead-in. Default 22.
* fps is pinned to 24; any other value is overridden.
* Clips of any grid length decode correctly: the VAE runs 5 latent frames at a
  time with a 2-frame look-ahead and cross-fades the seams, exactly as the
  reference does. Decoding a long clip in one call instead washes detail out
  progressively — measured against the conditioning photo, frame 0 fell from 0.97
  correlation at 22 frames to 0.86 at 90 — so the chunking is a correctness
  requirement, not an optimization.

> **Long clips keep the conditioning image, not necessarily the framing.** A
> keyframe pins the *start* of the clip. Over 124 frames (5.2 s) a prompt that
> describes a different shot wins: "a medium shot of two people arguing, handheld"
> on a photo of an ID card pushed into the card and had become that medium shot by
> about frame 60. With a prompt that asks for no camera move, the same 124-frame
> clip stayed at 0.94–0.97 correlation with the source photo throughout. To keep
> the photo in shot, say so in the prompt or generate a shorter clip.

## HTTP API

Three routes share one parser: `POST /api/video-generate`,
`POST /api/video-generate/stream` (same body, SSE progress ticks) and the
OpenAI-shaped `POST /v1/videos/generations`.

```sh
curl -s localhost:5000/api/video-generate -H 'content-type: application/json' -d '{
  "prompt": "a red fox trotting through falling snow, cinematic",
  "width": 640, "height": 384, "frames": 22, "steps": 8, "cfg": 1.0,
  "imagePath": "card.jpeg", "videoMode": "i2v"
}'
```

Reference conditioning uses the same route against the Ref2VA checkpoint:

```sh
curl -s localhost:5000/api/video-generate -H 'content-type: application/json' -d '{
  "prompt": "the same woman sits at a table in a sunlit cafe by a window, wide shot",
  "width": 640, "height": 384, "frames": 22, "steps": 20, "cfg": 1.0,
  "referenceImages": ["person.jpg", "bottle.png"], "videoMode": "ref"
}'
```

Returns `{ ok, url, audioUrl, width, height, frames, fps, seed, codec, elapsedSeconds }`.
`audioUrl` is null when the model produced no track. The full field set is `prompt`,
`width`, `height`, `frames`, `steps`, `cfg`, `fps`, `seed`, `flowShift`,
`imagePath` (or inline base64 `image`), `videoMode`, `generateAudio`, `endImage`,
`referenceImages`, `referenceVideos`, `referenceAudios`, `referenceVideoAudios`;
the Wan fields `negativePrompt`, `sampler`, `cfgCacheStride` and `cfg2` are
accepted and ignored by H3. Most of them are **camelCase only** —
`videoMode`, `generateAudio`, `endImage` and the four `reference*` lists are the
only ones that also accept a snake_case spelling (`video_mode`, `generate_audio`,
`end_image`, `reference_images`, …), and camelCase wins when both are present.
Sending `image_path` or `flow_shift` is silently ignored rather than rejected, so
prefer camelCase everywhere. `referenceVideoAudios` pairs **by index** with
`referenceVideos`, the same positional rule `--ref-video-audio` follows. `imagePath` / `endImage` /
`referenceImages` name files previously uploaded through `/api/upload`; paths
outside the upload directory are rejected. `/v1/videos/generations` shares the same
parser, so the same seven fields accept snake_case there too (`video_mode`,
`reference_images`, …); it returns `audio_url` rather than `audioUrl`.

A model rejection — the wrong checkpoint for the mode, or keyframes together with
references — comes back as a **400 carrying the model's own message**, not a generic
500, so the file to load instead is in the response body.

`GET /api/models` answers with a `video` object (null for every non-video model)
carrying `family` (`minimax-h3`), `supportsAudio`, `supportsImageConditioning`,
`supportsEndImageConditioning`, `supportsReferenceConditioning` and
`maxReferenceImages`. That is how the Web UI decides whether to offer a first frame,
a last frame or up to nine references, instead of pattern-matching an architecture
string.

Audio is written as a sidecar WAV rather than muxed into the MP4, because muxing
needs an encoder that may not be installed. To combine them:

```sh
ffmpeg -i fox.mp4 -i fox.wav -c:v copy -c:a aac fox_with_audio.mp4
```

`--no-audio` (`"generateAudio": false`) skips the audio decode entirely, saving the
audio VAE's time and memory; video-only models ignore it.

## Architecture

Recorded here because none of it is inferable from the checkpoints, which carry no
metadata. Cross-checked against stable-diffusion.cpp, the upstream reference
PyTorch, and the GGUF tensor tables.

### Diffusion transformer

50 blocks, ~19.3 B parameters, single-stream: text, conditioning frames, target
audio and target video are ONE sequence under full bidirectional attention. There
is **no cross-attention**.

| | |
|---|---|
| hidden | 5376 |
| attention | 56 heads × 128 = **7168 inner**, wider than the 5376 residual stream |
| MLP | SwiGLU, gate first |
| patch | (t, h, w) = (1, 2, 2); video 24 ch → 96-element patch |
| audio | 32 ch, one token per latent frame per stereo channel |

Within a token the 96 video values are **channel-major, patch-minor**
(`c*4 + ky*2 + kx`). The transposed order has identical statistics and silently
scrambles every token.

**AdaLN** uses a learned curve table, not a timestep MLP: `adaln_t_table` is
`[8, 1025]`, interpolated at the timestep, and one `Linear(8 → 96768)` per block
emits 6 modulation vectors for each of 3 modalities (0 = visual, 1 = text,
2 = audio). A token run picks `timestep*3 + modality`.

**RoPE** is 3-axis with a learned 16-entry `inv_freq`, so 96 of 128 head dims
rotate. Positions are **continuous floats**: time is measured in audio-latent units
(1/40 s) so both streams share one timeline, and the spatial axes are normalized by
the frame's geometric mean extent.

### The dual video/audio shift

Video wants timestep shift 12, audio wants 3. Rather than run two samplers, the
sampler stays on the video schedule and the model converts: the audio stream is
conditioned on the sigma a shift-3 schedule would have had at the same underlying
time, and its velocity is pre-multiplied by `d(sigma_a)/d(sigma_v)` so the shared
Euler step integrates it along its own trajectory. Both schedules start at 1 and
end at 0, so the streams land together. `--flow-shift` moves the video stream only.

### Text encoder

Qwen3-VL-32B truncated to 50 language layers **with the final norm removed** — the
DiT consumes the raw layer-50 hidden state, whose massive-activation outliers
(absmax ~15 000) are load-bearing rather than a bug. **There is no chat template**:
the prompt is the raw text. Applying one shifts every hidden state.

### Video VAE

16× spatial, 4× temporal. The decoder is a **pure 36-layer transformer** — no
deconvolutions. Each latent voxel is one token, and a final `Linear(2048 → 3072)`
plus depth-to-space does the whole upsample in one step.

Decoding is **tiled at 256 px, and that is a correctness requirement, not an
optimization**: the decoder's RoPE coordinates are length-normalized over whatever
extent it is handed, and it only works over the extent it was trained on. Decoding
a 640×384 frame in one call produces visibly sheared output even though every
individual tile is exact.

The encoder is a causal 3-D CNN, but for the single frame image conditioning needs,
the causal padding is two leading *zero* frames — so only the last temporal slice
of each kernel contributes and the network reduces **exactly** to 2-D convolutions.

### Audio VAE

DAC/BigVGAN, 32 latent channels → stereo 32 kHz. Upsample rates {5,5,2,2,2,2,2}
multiply to 800, and 32000/800 = the 40 Hz latent rate. Uses alias-free snake
activations: every nonlinearity is wrapped in a 2× upsample / activate / 2×
downsample sandwich so it cannot fold high frequencies back into the band.

Unlike the video latent, the audio latent is decoded **as-is** — `latents_mean` /
`latents_std` are *not* applied. Applying them costs about 15× amplitude and yields
a track that is spectrally plausible but inaudible.

## Environment variables

| Variable | Effect |
|---|---|
| `TS_H3_PREFAULT` | Sequential read of the denoiser file before its first upload. `0` off, `1` serial (read, then join before the upload), `2` overlapped with text conditioning, `3` **default** — pipelined with the upload |
| `TS_H3_PREFAULT_THREADS` | Read streams for that prefault, default `1` |
| `TS_H3_PHASE` | `1` = per-stage breakdown — encoder open / trunk / teardown, the prefault, every denoise step, VAE open / decode. The one-line summaries say a phase was slow; this says which half of it |
| `TS_H3_TE_GROUP` | `<n>` = run the 50-layer text-encoder trunk in groups of `n` layers, handing each group's device copy back after it runs. **Off by default**; any `n` ≥ the layer count reproduces the single whole-trunk call exactly |
| `TS_H3_TRACE` | `1` = latent and velocity magnitudes for every denoise step |

**Why `TS_H3_PREFAULT=3` is the default.** Mode 3 leaves the read running while the
upload proceeds, and wins because both walk the file in the same order and the read is
the faster of the two, so it stays ahead and the copy finds its pages already placed.
Mode 1 joins first and serializes a whole read in front of a whole copy. Mode 2 starts
the read earlier still, overlapping it with text conditioning, and loses: the text
encoder streams its own 17 GB through the same page cache and evicts the pages the
prefault has just placed. Mode 2 is kept because a host with enough RAM to hold both
files would flip that result, and mode 0 is the A/B — the output is identical either way.

**Why one read stream.** This is the opposite of `GgufReader.PrefaultFileCache`, which
uses sixteen, and the difference is deliberate: that warm-up runs before any other work
with the machine to itself, while this one runs concurrently with the encoder teardown
and with the upload it exists to help. Best of three at 640x384 on the RTX 3080 Laptop
16 GB was 1 stream 63.9 s, 4 streams 64.9 s, 16 streams 66.6 s, and the encoder teardown
alone rose from 2.2 s to 4.0 s at sixteen. Raise it only where storage is not the contended resource.

**Why `TS_H3_TE_GROUP` is off.** On a 16 GB card the 17 GB Qwen3-VL trunk does overflow
into shared host memory, and grouping does remove the overflow — peak device use fell
from 16 041 MiB to 12 981 MiB at 640x384, bit-identical output — but it was **3 s
slower**. The trunk is a one-shot prefill over a short prompt (ten tokens in the
measurement), so every weight is read exactly once and the overflowed ~1.3 GB costs a
single PCIe crossing; grouping cannot make that cheaper, still moves all 17 GB, and adds
an allocate/invalidate cycle per group. Spill compounds only when weights are re-read per
step, which is the denoiser, not this trunk. It is kept for the case where device memory
rather than time is the binding constraint — a smaller card, or another process wanting
the VRAM.

### What was tried and did not work

Measured on the RTX 3080 Laptop 16 GB, recorded so nobody has to run them again:

| Attempt | Result |
|---|---|
| Text-encoder layer grouping (`TS_H3_TE_GROUP`) | Removes the encoder's own spill, 16 041 → 12 981 MiB, bit-identical — and 3 s slower. Off by default |
| Prefaulting the video VAE too | Straight loss: 1.9 s of read and the decode does not move (9.4 s against 9.1 s). At 4.85 GB it is not fault-bound, and its decode is compute-bound |
| Multi-stream prefault (`TS_H3_PREFAULT_THREADS`) | Worse here: 63.9 / 64.9 / 66.6 s at 1 / 4 / 16 streams, 640x384 best of three, and the encoder teardown alone 2.2 s → 4.0 s |
| Overlapping the prefault with text conditioning (`TS_H3_PREFAULT=2`) | Worse than pipelining it with the upload: the encoder evicts the pages just placed |
| Disabling CUDA graphs | No effect: 65.6 s against 65.8 s |
| `TS_GGML_ASYNC_COMPUTE=1` | No effect: 67.4 s against 66.2 s. It governs graph submission, not weight upload |
| ffmpeg in place of the OS H.264 encoder | Slower: 98.2 s. The sidecar encoder was left alone |

## Verification

Every network is checked against the reference implementation, not just against
itself. Fixtures come from the upstream PyTorch and from stable-diffusion.cpp's own
intermediate tensors.

| Component | Result |
|---|---|
| Text encoder, all 50 layers | cos 0.999999 vs sd.cpp's conditioning |
| DiT step, all 50 blocks | video cos 0.9983, audio cos 0.9998 |
| Video VAE decode | cos 1.000000 |
| Video VAE encode | cos 1.000000 |
| Audio VAE decode | cos 0.999995 |

Regenerate the fixtures with:

```sh
python InferenceWeb.Tests/tools/minimax_h3_oracle.py --ref-dir <reference py> \
    --weights-dir ~/work/models/minimax-h3 --out-dir ~/work/models/minimax-h3/fixtures
python InferenceWeb.Tests/tools/minimax_h3_dit_oracle.py --gguf <denoiser> --out-dir <fixtures>
python InferenceWeb.Tests/tools/minimax_h3_te_oracle.py --gguf <text encoder> --out-dir <fixtures>
```

Then run the gated tests:

```sh
export TS_MINIMAX_H3_DIR=~/work/models/minimax-h3
export TS_TEST_GGML_BACKEND=metal
dotnet test InferenceWeb.Tests -c Release --filter "FullyQualifiedName~MiniMaxH3"
```
