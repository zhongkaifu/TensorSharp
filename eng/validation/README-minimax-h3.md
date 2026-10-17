# MiniMax-H3 validation

Use the same checkpoint files and tokenizer for both engines. Keep upstream
ggml and the stable-diffusion.cpp baseline unchanged; the paired runner rejects
dirty or unavailable dependency checkouts. Put outputs in ignored `artifacts/`
or `docs/validation/`. A structural media pass does not establish perceptual
quality or benchmark parity.

## 2026-10-17 1280×768 speed and Turbo plug-in validation

Evidence is under ignored `artifacts/minimax-h3/speed/` (attention benchmarks and
base runs) and `artifacts/minimax-h3/turbo-*` and `artifacts/minimax-h3/lora-native/`
(Turbo plug-in). All runs use unchanged upstream ggml
`a05c9922613c6a2230dbb1d0de679519ecafb4f0` on the Apple M5 Pro (48 GiB, Metal 4
tensor API live, 38.3 GB recommended working set). Every timing is a single run under
`caffeinate` with OS caches uncontrolled.

**Why 1280×768 was slow.** A 1280×768×107 FL2VA clip packs 34,991 tokens, and about
85% of a ~570 s step was flash attention. ggml-metal takes its tensor-API flash
kernel only when the key count is a multiple of 64 (34,991 leaves 47), so H3 ran the
8-queries-per-threadgroup simdgroup kernel. `ggml_ops_metal_flash_pad.h` pads K/V
with zero rows to a multiple of 64 and masks the pads, and passes the queries in
64-row chunks over ggml's grouped-query broadcast. One 4 MB mask then serves every
row, where a per-query mask was 2.45 GB. ggml is not modified.

| Measurement | Before | After |
|---|---:|---:|
| One flash call, 34,991 tokens, 56×128 heads (`GgmlOpsMetalFlashKvPadTest metal`) | 8.05 s | 3.24 s |
| One flash call, 9,298 tokens (640×384×107) | 0.456 s | 0.233 s |
| DiT step at 1280×768×107 | ~570 s | 262.8 s |
| Full 20-step clip at 1280×768×107 (first/last keyframes, seed 42) | 11,553.8 s | 5,438.5 s |

- **Correctness.** On CPU the chunked padded call is bit-identical to unpadded
  attention. On Metal it differs only by FP16 rounding (rel-rms ≤ 6e-4), and its error
  against an F32 explicit-softmax reference matches the simdgroup kernel's.
- **First-step velocities.** At 384×256×22 the chunked and full-mask versions are
  bit-identical. Against the F32 reference, video and audio rel-rms are 2.6e-3 and
  3.6e-3, against 2.5e-3 and 4.1e-3 unpadded.
- **Full-clip drift.** The 20-step clips before and after agree at a mean whole-frame
  PSNR of 41.6 dB.
- **H3 oracle tests on Metal.** The VAE, vision, long-clip and checkpoint tests pass
  on Metal with the padding active. 31 run; several of them return early without
  their external fixtures, and those are not counted.
- **The gate.** The padding runs only where the tensor API is predicted live. With
  the tensor API disabled, the padded call was 5% slower. `TS_METAL_FLASH_KV_PAD=0`
  disables it.
- **Memory fix.** At this size the DiT's reuse allocator grows to 23.9 GB. With the
  DiT weights still wired, only 3.3 GB of the 38.3 GB budget stayed free, and the
  VAE's first command buffer failed with `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
  The pipeline now releases both on Metal before the decode (33.1 GB free, decode
  146–148 s).
- **MLX's fused attention** at the same shape: F32 4.08 s and F16 1.68 s per call.
  That is 1.93× the padded ggml kernel, below the 2× bar set for writing a
  TensorSharp-owned Metal kernel, so none was written.

**Turbo plug-ins** (LightX2V FL2V 8-step v1.0 and 4-step v1.2, 768p, applied
unmerged). The same keyframes, prompt and seed were run at 1280×768×107:

| Run | s/step | Total |
|---|---:|---:|
| Base 20 steps | 262.8 | 5,438.5 s |
| Turbo-8 | 264.7 | 2,298.7 s |
| Turbo-4 | 264.7 | 1,238.6 s |
| Turbo-4 at video shift 12 (A/B, not shipped) | 264.9 | 1,240.1 s |

- **Identity.** On `ggml_metal` with the real weights, strength 0 gives a first-step
  video velocity bit-identical to base.
- **Faces.** They stay sharp and undistorted in every Turbo clip, by eye on full-size
  crops.
- **Drift from base 20.** Mean whole-frame PSNR is 17.5 dB for Turbo-8 and 19.8 dB for
  Turbo-4. Most of it is the transition between two dissimilar keyframes: base
  dissolves, Turbo cuts.
- **Character.** Turbo is a little more saturated (0.20 against 0.17), moves more, and
  its audio is louder (RMS 0.21–0.25 against 0.15).
- **Shift A/B.** Shift 6 and 12 came out alike, so the 4-step recipe stays inferred.

Not covered:
- peak memory with an adapter;
- text-to-video, more seeds, or other canvases;
- `ggml_cuda`, direct `cuda` or `mlx` with an adapter, apart from MLX on a tiny
  fixture;
- M1–M4 Macs, where the padding is gated off;
- a Studio or TensorAgent Turbo take;
- the 32 GB TensorAgent tier.

## 2026-10-15 performance and quality validation

Evidence is under ignored `docs/validation/minimax-h3/performance-20261015/`.
The main full-generation comparisons use FL2VA Q4_K, 640×384×22 at 24 fps,
20 Euler steps, seed 42, CFG 1 and flow shift 12. Both frozen binaries use
unchanged upstream ggml `3612c0af93f2cfc8f048295300a61187723b12ae`.
The local host is an Apple M5 Pro with 48 GiB unified memory; the remote
worker is an NVIDIA A40 with 46,068 MiB reported VRAM and CUDA 12.8.
Two alternating rounds include loading, generation and codecs. OS caches
remain uncontrolled; these are not warm-model service benchmarks.

| Backend / host | Baseline median | Candidate median | Confirmed result |
|---|---:|---:|---|
| Native Metal / local | 151.246 s | 143.622 s | Exact decoded RGB/audio; 1.053×, 5.0% lower latency |
| Native CUDA / VM | 68.028768 s | 67.173625 s | Exact decoded RGB/audio; 1.3% lower latency does not establish a meaningful gain |
| Direct MLX / local, final candidate 6 | 143.264 s | 125.426 s | Exact decoded RGB/audio; 1.142×, 12.5% lower latency |
| Direct CUDA / VM, equal precision | 220.456 s | 178.667 s | Exact decoded RGB/audio; 1.234×, 19.0% lower latency |

The reports are `local-metal-final-ab/report.json`,
`local-candidate6-mlx-ab/report.json`, `remote/final-remote-ggml-cuda-ab/report.json`
and `remote/final-remote-cuda-ab/report.json`. Earlier Ref2VA single-pair runs
showed large, opposing VAE timing swings. The completed candidate 6 Ref2VA
comparison used image plus canonical audio, 256×160×5, eight steps and seed 44:
82.037165 → 43.630105 s, 1.880× and 46.8% lower latency, with exact RGB/audio
in both alternating pairs (`local-candidate6-reference-ab/report.json`).
First/last-keyframe FL2VA at the same geometry/eight steps, seed 43, measured
83.466 → 38.493 s with exact RGB/audio (`local-candidate6-keyframe-ab/report.json`);
its single pair is diagnostic only and does not establish a repeatable gain.

The implementation reduces repeated work and memory use:

- Borrowed contiguous token-row views preserve source ownership and avoid
  copies. Exact RoPE tables use a position snapshot with atomic publication
  of the key, cosine and sine arrays for concurrent readers.
- MLX packed modulation/gated residual kernels preserve expression order;
  device SwiGLU avoids activation splitting/copies in DiT and the MLX VAE.
- The MLX VAE uploads dense F32 weights directly into owned MLX arrays,
  structurally avoiding 9,689,041,152 bytes (9.02 GiB) of duplicate retained
  host weights for this decoder; this is not a measured peak-RSS reduction.
  Host reads still materialize an exact copy when needed; precision is unchanged.
- Direct CUDA text weights use bounded CPU prefaulting before first upload.
  Native CUDA uses a per-model stable graph arena and cached weight bindings;
  variable audio scaling stays in a separate exact ggml SCALE graph so the
  main graph can be captured without changing floating-point semantics.

An isolated MLX VAE comparison before the host-storage optimization measured
0.4303285 → 0.4003488 s (1.075×), with exact output floats across six warm
measurements per version in two reversed-order processes. It uses a fixed
synthetic latent producing a 256×160×5 clip with production padding and excludes
the pipeline. See `local-vae-isolate-summary.json`; this stage measurement
predates candidate 6's host-storage optimization.

The subsequent fixed-latent memory check preserved all output bytes and reduced
median sampled warm process RSS from 23.048 to 14.024 GiB versus candidate 5.
MLX active memory and allocator peak were effectively unchanged. This is sampled
process residency, not peak unified/GPU memory; candidate 6 has one process and
three warm runs. See `local-vae-memory-summary.json` for hashes and limitations.

Final local candidate 7 passed 36 focused CPU/MLX tests and all nine component oracles
on each of direct MLX and native Metal, with no skips. See
`local-candidate7-mlx-oracles/report.json` and
`local-candidate7-ggml-metal-oracles/report.json`. Its Ref2VA preservation pair
also reproduced candidate 6's decoded RGB/audio exactly, and a complete 20-step
native Metal pair reproduced the earlier benchmark candidate exactly. On the
VM, final candidate 7 passed 29 focused tests, the default-precision full50
gate, a complete 20-step direct CUDA preservation pair and a native CUDA
first/last-keyframe pair. All nine component oracles on each of direct/native
CUDA and the native exact-output graph-capture test also passed. Component
DiT slices are not full-model coverage.

The final-build preservation pairs are additional quality checks, not new
two-round speed measurements. Reports are `local-candidate7-preservation/`,
`local-candidate7-native-preservation/`, `remote/candidate7-cuda-preservation/`
and `remote/candidate7-native-preservation/`; the final independent prediction
comparison is `remote/candidate7-full50-default-comparison.json`.

Independent all-50-block validation exposed TF32 activation rounding in large
BF16-weight CUDA projections. Pedantic F32 is now the default; the historical
approximation requires `TS_H3_CUDA_BF16_PEDANTIC=0`. With the default override
unset, raw-prediction relative L2 errors improved as follows:

| Full50 output | Previous default | Fixed default | Strict gate |
|---|---:|---:|---:|
| Audio | 0.0070538 | 0.0001258906 | ≤ 0.001: pass |
| Video | 0.0046911 | 0.0000847537 | ≤ 0.001: pass |

This is an independent PyTorch F32 oracle over a synthetic 19-token packed
input at sigma one. `MiniMaxH3DitProbe`, `minimax-h3-full-dit-reference.py`
and `minimax-h3-prediction-compare.py` use `--fixtures`, verify input/output
hashes, and recompute all 50 blocks from raw inputs. They do not reuse the
two-block expected outputs. This does not establish multi-step oracle or
natural-media perceptual quality.

Experimental batched attention passed the synthetic full50 check but **failed
the complete-generation quality gate**: RGB PSNR 40.792195 dB was below 60 dB,
and audio relative L2 0.01725139 exceeded 0.001. Its model opt-in was removed;
the probe rejects obsolete `TS_H3_CUDA_BATCHED_ATTENTION` overrides. Only the
standalone backend helper/tests/microprobe remain for research. The rejected
model patch and failure record are archived as
`rejected-cuda-batched-attention-model-optin.patch` and `.json` in the evidence
directory. No kernel speedup from this experiment is a production speed claim.

Reproduce a comparison with separately frozen CLI directories:

```sh
python3 eng/validation/minimax-h3-ab.py --backend ggml_metal \
  --models /path/to/minimax-h3 --steps 20 --rounds 2 \
  --baseline-cli /path/to/baseline/TensorSharp.Cli.dll \
  --candidate-cli /path/to/candidate/TensorSharp.Cli.dll \
  --expected-ggml-revision 3612c0af93f2cfc8f048295300a61187723b12ae \
  --output docs/validation/minimax-h3/new-metal-ab
```

For direct CUDA use `--backend cuda --bf16-pedantic`: this explicitly applies
the fixed precision policy to both binaries, including the older baseline.
Label it an equal-precision comparison. The runner rejects inherited H3
overrides and dirty ggml, checks geometry/step count and frozen binary hashes,
and requires exact decoded RGB/audio. Failed, skipped or unfinished runs do
not count as successful validation.

Research references: [vLLM-Omni H3](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py)
for reusable RoPE and indexed modulation; [H3-Flash](https://github.com/fluxmatrix/h3-flash)
for separating systems/fusion optimizations from approximate Turbo modes;
[MLX array ownership](https://github.com/ml-explore/mlx/blob/main/mlx/array.h#L543)
for copied F32 storage; and [cuBLAS strided batched GEMM](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex)
for the experimental attention implementation. Their reported speeds are not
TensorSharp benchmark results.

## Earlier recorded validation

The recorded 2026-10-13–14 evidence uses an M5 Pro with 48 GB locally and an A40
remotely. ggml is unchanged at `3612c0af93f2cfc8f048295300a61187723b12ae`;
the clean SD baseline is `d637cfb935feb2254898a8f1524046c958f3df80`, with pinned
ggml `d25b121ce1d6ae0c7e62ad080460d9fc1e46dd93`.

| Recorded case | TensorSharp | SD / preservation evidence |
|---|---:|---|
| Native Metal, 640×384×22, eight steps | 70.019 s | 102.262 s in the paired run |
| Direct MLX, same request, before packing → final | 158.333 → 87.054 s | Decoded RGB and WAV bit-identical across before-packing, after-packing and final-reader binaries |
| Earlier native CUDA binary, 640×384×22, 20 steps, warm-cache repeat | 108.299 s | SD 47.932 s; initial near-parity inference is superseded |
| Earlier bulk-prefault experiment, same 20-step request | about 66–69 s | SD comparison runs ranged 47.92–79.08 s; no general CUDA parity established |
| Final automatic-prefault native CUDA default, same request | 71.899 / 64.904 s | Corresponding alternating SD rounds: 145.020 / 50.238 s; exact RGB/WAV preservation across earlier and final binaries |
| Direct CUDA, same 20-step request | 228.98 s | Earlier same-input SD process 112.21 s |

These are new-process measurements with uncontrolled file/JIT caches, not
warm-model service benchmarks. Final MLX versus saved SD uses the same inputs
but was not a new alternating timing pair. Reader widening alone did not
establish an additional process improvement: 87.054 s final versus 86.318 s after
packing. Tiny smoke settings and successful file exports do not establish
perceptual quality. The inspected llama.cpp and vLLM trees have no identified
H3 AV diffusion execution route, so they are not executable generation baselines.

The selected native CUDA helper uses CPU-only bulk prefault of immutable pending
weights, automatically bounded by half the smaller of physical RAM and the
cgroup memory limit. Unknown capacity disables automatic warming.
`TS_H3_UPLOAD_PREFAULT=0` disables it and `1` forces it;
`TS_H3_UPLOAD_WORKERS` selects 1–8 workers (default 8). The independent managed denoiser read
retains `TS_H3_PREFAULT=3`; removing it did not establish a repeatable process
gain. Retain the exact policy/environment in each frozen-binary report.

The final native default passed all nine actual component oracle cases and
both alternating generation pairs with no skips. Eight before/experimental/final
TensorSharp generations have identical decoded RGB and WAV hashes. The final
warm SD repetition remains faster, so CUDA whole-process parity is not claimed.
`TS_H3_UPLOAD_PROFILE=1` reports the memory decision and separate prefault/upload
costs; it was enabled only for diagnostic oracle verification, not the final
generation timings.

Build the Release CLI and tests first. On the A40 worker, expose the installed
CUDA toolkit and compile compatible PTX rather than using the committed sm120
baseline:

```sh
TENSORSHARP_GGML_NO_UPDATE=1 PATH=/usr/local/cuda/bin:$PATH dotnet build InferenceWeb.Tests/InferenceWeb.Tests.csproj \
  -c Release -p:CudaArch=compute_86
```

Freeze the resulting CLI directory for each before/after comparison. Record
the native library, managed assemblies and PTX from that same directory. The
paired runner records those binary hashes, hardware, dependency revisions,
explicit sigma schedule, seed, conditioning hashes, media shape and waveform
finiteness. It alternates engine order across rounds. Whole-process timing
includes model loading and codecs; neither file-cache coldness nor warm-model
service latency is established by this runner.

```sh
python3 eng/validation/minimax-h3-oracles.py --backend cuda \
  --models /workspace/models/minimax-h3 --primitives \
  --output artifacts/h3/cuda-primitives
python3 eng/validation/minimax-h3-oracles.py --backend cuda \
  --models /workspace/models/minimax-h3 --output artifacts/h3/cuda-oracles
python3 eng/validation/minimax-h3-generation.py --backend ggml_cuda \
  --models /workspace/models/minimax-h3 --case quality --steps 20 --rounds 2 \
  --tensorsharp-cli /path/to/frozen/TensorSharp.Cli.dll \
  --sdcpp-cli /path/to/clean-sd-build/bin/sd-cli \
  --sdcpp-source /path/to/clean-sd-source --output artifacts/h3/paired-20step
```

The oracle runner rejects missing fixtures, skips and unexpected case counts.
Its text/DiT fixtures cover slices; the vision tower and VAEs have separate
coverage. `--case quality` defaults to eight steps on the base checkpoint;
specify `--steps 20` for the TensorSharp default. Dedicated FastH3 checkpoints
have separate trained schedules and are not interchangeable with base H3.

For synthetic conditioning coverage:

```sh
python3 eng/validation/generate-minimax-h3-conditioning.py \
  --output artifacts/h3/conditioning
```

Pass `--image` and `--end-image` for FL2VA, or `--checkpoint ref2va` together
with repeatable `--ref-image`/`--ref-audio` for Ref2VA. These small synthetic cases
exercise routing and output validity; they are not a perceptual dataset.
`minimax-h3-compare.py` rejects mismatched checkpoint, prompt, schedule, seed or
conditioning identities before comparing saved media. Conditioning paths may
differ across hosts if their roles/order and content hashes agree.

Reference audio needs an additional preprocessing check. TensorSharp caps or
pads standalone audio to the generated duration and floors to an 800-sample
hop; the inspected SD version preserves the full file and ceil-pads it. The
original one-second soundtrack versus a five-frame output therefore supplied
8 reference latents to TensorSharp and 40 to SD. Its output difference is not
evidence of an inference arithmetic bug or a matched-input parity failure.
SGLang also caps reference material by the output duration.

Canonicalize a common 32 kHz stereo PCM16 file explicitly for the five-frame
comparison; leave the original file and report intact:

```sh
python3 eng/validation/minimax-h3-canonicalize-reference-audio.py \
  --input artifacts/h3/conditioning/reference.wav \
  --output artifacts/h3/conditioning/reference-canonical-6400.wav \
  --frames 5 --fps 24
python3 eng/validation/minimax-h3-generation.py --backend ggml_cuda \
  --models /workspace/models/minimax-h3 --checkpoint ref2va --case smoke \
  --steps 8 --seed 44 --ref-image artifacts/h3/conditioning/first.png \
  --ref-audio artifacts/h3/conditioning/reference-canonical-6400.wav \
  --prompt "A glossy red ball sits on a blue table. The ball slowly rolls to the right. A soft rolling sound." \
  --tensorsharp-cli /path/to/frozen/TensorSharp.Cli.dll \
  --sdcpp-cli /path/to/clean-sd-build/bin/sd-cli \
  --sdcpp-source /path/to/clean-sd-source --output artifacts/h3/ref2va-canonical
```

The canonicalizer refuses existing outputs and empty/sub-hop results, preserves
the selected PCM bytes, and writes an adjacent `.wav.json` manifest containing
source/output hashes, sample counts and the scope **explicit benchmark
canonicalization, not pipeline change**. A source shorter than the selected
duration can still be padded differently by the engines; inspect effective
lengths rather than treating matching file hashes as sufficient.

The actual image-plus-canonical-audio eight-step pair achieved RGB PSNR
**37.674 dB** and stereo waveform cosine **0.999423**. This five-frame synthetic
request and the FL2VA first/last-frame case establish conditioning execution
and diagnostic agreement, not natural-media perceptual quality or speed parity.

`MiniMaxH3DitProbe` rejects inherited `TS_H3_DIT_LAYERS`. Its trace-free mode
measures complete DiT forwards including returned host arrays, separately from
text encoding, sampling and VAEs. Use one warmup and at least three repeats.
`minimax-h3-prediction-compare.py` binds saved velocities to the probe's digest,
checks exact input hashes against saved independent full50 IEEE F32 truth, and
compares the recovered sigma-one raw predictions. This does not independently
validate a multi-step generation. Trace callbacks/downloads change execution
synchronization and must be excluded from performance claims.

Pure CPU evidence-gate regressions:

```sh
python3 -m unittest discover -s eng/validation/tests \
  -p test_minimax_h3_prediction_compare.py -v
```

`MiniMaxH3MlxGroupNormProbe` measures the prior expression versus the fused
per-frame normalization with synchronized device evaluation.
`MiniMaxH3MlxQ4KPackProbe` compares lossless CPU packing independently of paging,
device upload and inference. Both are stage measurements. Full-generation
quality preservation additionally requires decoded RGB and waveform identity
for before/after frozen binaries when the intended change is bit preserving.

`H3MmapProbe.cpp` compares read-only shared/private mappings on one opened file,
with fresh mappings, reversed trial order, page touches or bounded copies, and
fault/CPU counters. `H3MmapNetProbe` inspects the actual .NET mapping flags through
Linux `smaps`. The measured cached 2 GiB NFS slice showed no meaningful mapping
flag benefit; production mapping semantics remain unchanged. These CPU probes
neither drop caches nor establish cold-file or GPU performance.

```sh
mkdir -p artifacts/h3
c++ -O2 -std=c++17 -pthread eng/validation/H3MmapProbe.cpp -o artifacts/h3/mmap-probe
artifacts/h3/mmap-probe /workspace/models/minimax-h3/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --offset 65184 --bytes 2147483648 --trials 4 --workers 8
dotnet run -c Release --project eng/validation/H3MmapNetProbe -- \
  /workspace/models/minimax-h3/qwen3vl_32b_minimax_h3-Q4_K_M.gguf
```

Current ignored evidence lives under `docs/validation/minimax-h3/optimization-current/`:

- `local-ggml-metal-quality/report.json` and `inspection.json`: actual native Metal pair and sampled-frame inspection.
- `local-mlx-quality-final/three-version-identity.json`: 158.333 / 86.318 / 87.054 s processes and exact decoded RGB/WAV hashes.
- `remote/generation-ggml-cuda-20step-balanced/report.json`: alternating two-round timing and matched-output comparisons, including the 47.932 versus 108.299 s result.
- `remote/generation-ref2va-canonical/report.json`: actual common 6,400-sample mixed-reference generation; `ref2va-reference-preparation-audit.json` explains the original mismatched context.
- `remote/generation-ggml-cuda-final-default-20step/report.json` and `remote/native-prefault-version-identity.json`: final automatic policy, both alternating pairs, unchanged dependency checks and exact before/after media preservation.
- `remote/ggml-cuda-final-oracles/report.json`: nine final native CUDA component cases; `remote/cuda-reader-final/report.json`: four affected direct CUDA VAE cases after vectorized reader conversion.
- `canonicalize-reference-audio-validation.json`: eight functional checks for exact PCM preservation, format/duration rejection, fresh-output protection and manifest integrity.

Skipped, unavailable model/device scenarios, long clips and unrun perceptual workloads must remain
explicit limitations, not passing coverage. For architecture and API details,
see [the model documentation](../../docs/models/minimax-h3.md).
