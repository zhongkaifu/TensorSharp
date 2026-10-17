# LTX 2.5 validation

Keep generated evidence under ignored `docs/validation/ltx25/` or `artifacts/ltx25/`.
The tools never patch SD.cpp or ggml sources.

`Ltx25Model` is a registered native `IVideoGenerationModel`. Its text-to-video/
audio path runs the authentic 48-layer Gemma 4 Unified encoder, V2 BF16 hidden-state
aggregation, eight video/eight audio connector blocks, the publisher's two-stage
distilled recipe (the complete 48-block AV denoiser over the published eight sigmas
with ancestral Euler at half the output size, the x2 spatial latent upscaler, then
three refinement sigmas at full size), the convolutional video VAE and the stereo
audio VAE/vocoder/bandwidth extension. Ingredients identity sheets keep one
full-size pass. CUDA and MLX are native GPU
paths; CPU component tests are separate coverage. Required independent tiny
fixtures live in `InferenceWeb.Tests/Fixtures/Ltx25/` and are embedded in the test
assembly. The pipeline supports first/last/timed image guides and optional
Ingredients identity sheets with its separate adapter. Reference video/audio,
retake/refinement, specialized control pipelines and multi-GPU remain unsupported.

`config/ltx-2.5-distilled.json` pins five verified files and starts at 1024×1024
output (512×512 first pass), 17 frames, 24 fps. TensorAgent enables Download/Use for
`ltx-2.5-distilled-q4km`. The conservative 48 GB catalog tier is not an established
minimum for arbitrary shapes. Both downloaders support current-host Hugging Face
credentials for gated official files; tokens never go to mirrors/CDN redirects.
The preset/catalog test keeps all five names, URLs and hashes aligned.

Prompt conditioning caches at most four final context pairs. The model retains
its denoiser's mapped/device weight identity across requests to avoid repeated
MLX Q4_K repacking. Prompt/video checkpoint mappings and audio decoder converted
weight caches are retained; prompt connector device weights are released after
each new prompt. `TrimIdleMemory` releases all retained components. Prompt weights
stream per projection, and video decoder weights per convolution. Complete-generation memory
and cache behavior must be measured separately from component measurements.

## Complete generation and application probes

`Ltx25EndToEndProbe` calls the real architecture registry and generation model,
writes raw RGB/stereo PCM, then requires a soundtrack in the encoded MP4. It
records factory, inference and codec time separately and identifies prompt cache
hits. `Ltx25TensorAgentProbe` separately activates the real catalog entry through
`AgentAppHost.UseModel`, sends an HTTP/SSE video job and downloads the generated
MP4/WAV through the real upload routes, using isolated pinned cache symlinks.
Fixture download tests exercise all four roles without redownloading 43.77 GB.

`ltx25-maui-probe.py` exercises the actual Release Mac Catalyst app: the native
Models page Use action, its loopback HTTP/SSE route and downloaded MP4/WAV. Launch
with an isolated `TENSORAGENT_VALIDATION_ROOT` containing the verified cache,
then pass its private `connection.json` and actual `.app` path. The tool reads the
loopback cookie locally and never exports it. It requires Metal activation, H.264
video with muxed audio and the original-length 48 kHz stereo WAV. CoreCLR tests
alone do not cover Apple AOT: MLX native callbacks use static Cdecl entry points
because runtime-generated delegate trampolines are unavailable there.

The Mac Catalyst app sets `MLX_ENABLE_TF32=0` in both managed and native
environments at its entry point when the variable is unset. MLX caches this
choice on its first GEMM, so setting it only when LTX is selected is too late
after another MLX model has run. Explicit launch overrides remain unchanged;
other TensorSharp hosts keep their existing defaults. For an app startup-default
check, remove this variable from the launch environment and retain the startup
log before activating a model.

CUDA prompt BF16 GEMM defaults on for SM 8.0 or newer; its opt-out is
`TS_LTX25_CUDA_BF16_GEMM=0`. Bounded mapped projection reads default on; their
opt-out is `TS_LTX25_CUDA_PREFAULT=0`. The latter retains one file handle and an
8 MiB buffer per reader, checks Linux residency and records bytes/ranges/time/failures in
component probes. Whole-checkpoint prefault probes and skipped RAM-budget checks
are separate diagnostics. Never describe them as production coverage. Default
CUDA video attention uses the precise owned path for query/key lengths at least
512; the 768-video-token preset has a matched numerical/control timing gate.
Linux warming uses at most eight concurrent 1 MiB slices of that same buffer;
`TS_LTX25_CUDA_PREFAULT_WORKERS=1` selects the sequential control. Other operating
systems default to one worker. Completed slice reads are joined before buffer
reuse, and partial/failed reads remain visible in the telemetry.

```sh
dotnet publish eng/validation/Ltx25EndToEndProbe -c Release \
  -o artifacts/ltx25/end-to-end -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
MLX_ENABLE_TF32=0 TS_FFMPEG=/path/to/ffmpeg \
TENSORSHARP_MLX_LIBRARY_DIR="$PWD/TensorSharp.Backends.MLX/Native/dist" \
dotnet artifacts/ltx25/end-to-end/Ltx25EndToEndProbe.dll \
  --model "$DENOISER" --backend Mlx --output "$NEW_REPORT" \
  --width 512 --height 512 --frames 17 --fps 24 --seed 42 --repeats 2 --audio 1
```

Pass each option and value as separate arguments (for example `--width 512`).
Use `--backend Cuda` on the VM and install the verified compute 86 primary PTX
after publishing; the checked-in primary asset may target another GPU. Preserve
failed attempts. A requested unavailable device or a missing codec is not a pass.
`VideoAudioExportProbe` can retry the codec alone from saved actual raw output;
it explicitly excludes a model rerun and inference timing.

Build the clean SD.cpp C-API full-generation runner with
`build-ltx25-sdcpp-denoiser.py --kind generation` and the verified source/build
manifests. `ltx25-generation-runner.py` accepts an explicit argv JSON file and
samples process/device/container memory. Compare completed matched reports with
`ltx25-compare-generation.py`; it includes factory-plus-first and repeated-prompt
timings. Both runners use the same pinned assets, 512×512×17@24, seed 42, eight fixed
sigmas, CFG 1 and CUDA Philox RNG. 17 frames gives 18 audio tokens in both the original
rounding rule and SD.cpp's ceil rule; other frame/fps combinations may differ.
SD.cpp stays unmodified. Pixel/audio differences are descriptive and do not
establish a broad perceptual quality ranking. First requests include model loading
and JIT work; repeated prompts use each implementation's conditioning cache.

## Exact prompt math and alternating benchmarks

Gemma prompt conditioning retains its existing BF16 boundaries. CUDA can round
activations on the device, and CUDA/MLX can apply split-half Gemma RoPE without
reading the activation back to the CPU. The host still computes each rotary
cosine/sine with the original `MathF` operations and BF16 rounding, once per
token/frequency and shared across heads. Device kernels round each multiply and
sum in the original order; unrotated channels remain unchanged. This does not
lower precision, reduce prompt tokens, skip layers or change the conditioning
cache. MLX's existing BF16 rounding path is unchanged.

`TS_LTX25_DEVICE_PROMPT_MATH=0` selects the host control for Gemma RoPE and CUDA
BF16 rounding. Unset or `1` requests the device paths. An older optional
`tensorsharp_ltx25.ptx` that lacks either prompt export keeps the host fallback;
successful generation alone therefore does not establish native dispatch. The
new native tests assert that dispatch returned true, then compare every output
bit against the host arithmetic, including full/partial rotary geometry,
offset views and untouched guards. CUDA rounding tests cover ties, signed zero,
subnormals, infinities and NaN bit patterns. Missing GPUs remain skipped coverage.

Build tests first; on CUDA install and verify the intended PTX **after** the
build with `ltx25-deploy-cuda-modules.py` as described below. Run only the native
filter for the device actually available:

```sh
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true \
  --filter 'FullyQualifiedName~Ltx25PromptMathTests.Cpu|FullyQualifiedName~CpuStorageBulkElementIoTests|FullyQualifiedName~TensorApiTests'

MLX_ENABLE_TF32=0 TS_TEST_LTX25_BACKEND=mlx \
  dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release --no-build \
  --filter 'FullyQualifiedName~Ltx25PromptMathTests.Mlx|FullyQualifiedName~Ltx25PromptEncoderTests'

TS_TEST_LTX25_BACKEND=cuda \
  dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release --no-build \
  --filter 'FullyQualifiedName~Ltx25PromptMathTests.Cuda|FullyQualifiedName~Ltx25PromptEncoderTests'
```

The offset-view tests also cover the general CPU storage correction: bulk
Float32/Int32 reads honor nested view offsets and reject out-of-storage ranges.
This is independent of LTX and has direct `CpuStorageBulkElementIoTests` coverage.

`Ltx25PromptProbe --device-math-rounds` alternates host/device requests on one
encoder instance. It supports `CUDA` or `MLX` with `--stage full` and cannot be
combined with the existing `--cuda-bf16-rounds` GEMM experiment. Every round saves
full video/audio contexts, their hashes, elapsed time and Gemma/aggregation/
connector phases. Device-math rounds fail if either context differs bitwise
from the first round. The report says `device_requested` rather than claiming
native dispatch; use the strict tests and matching module hashes above to
establish dispatch.

```sh
dotnet publish eng/validation/Ltx25PromptProbe -c Release \
  -o artifacts/ltx25/prompt-math-probe \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true

MLX_ENABLE_TF32=0 TENSORSHARP_MLX_LIBRARY_DIR="$PWD/TensorSharp.Backends.MLX/Native/dist" \
dotnet artifacts/ltx25/prompt-math-probe/Ltx25PromptProbe.dll \
  --encoder "$TEXT_ENCODER" --model "$DENOISER" --backend MLX --stage full \
  --prompt "A red ball rolls across a wooden table." \
  --device-math-rounds 0,1,1,0,0,1 --output "$NEW_PROMPT_REPORT"
```

For CUDA, select `--backend CUDA` and deploy verified compatible PTX into the
published output before running. The first round includes first-use loading/JIT
and any explicit prefault work; report it separately from subsequent rounds.
Per-round timings exclude saved-array export, while total elapsed includes it.
Repeated `Encode` bypasses the model's final prompt cache and is a prompt
component measurement, not end-to-end generation throughput. Require separately
matched complete-generation RGB and stereo PCM comparisons, full media decoding
and visual inspection before making any quality or overall-speed claim.

## Exact CUDA head gating

CUDA denoiser head gating computes the existing `2 * sigmoid(logit)` once per
token/head and multiplies its channels in place, preserving the separate Float32
rounding boundaries. This avoids the expanded gate tensor and several launches.
`TS_LTX25_CUDA_FUSED_HEAD_GATE=0` restores the separate operations. An older
optional LTX PTX module without the gate export retains the original fallback;
the `OptimizedCudaHeadGates` counter records successful native dispatches.

`Ltx25CudaHeadGateTests` requires bitwise equality to the original device
operations for offset views, special values and 131,072 varied finite logits.
`Ltx25DenoiserTests.FusedHeadGatesExactlyPreserveCompleteOfficialGraph` compares
complete tiny official graphs with fusion off/on. Run both filters on CUDA
with verified compatible PTX. Component parity must still be followed by matched
complete-generation RGB/stereo comparisons and media validation.

## Lossless MLX Q5_K loading

MLX's Q5_K loader preserves every five-bit quant and the established F16
scale/bias bytes while packing eight quants directly into five output bytes.
It assigns every output byte, avoiding the former whole-buffer clear, and
processes independent rows with at most eight CPU workers for sufficiently
large matrices. The later exact F32 scale/bias preparation is unchanged. This
changes checkpoint loading work, not quantization, activation precision or
matmul arithmetic. Q6_K continues to use its existing raw representation.

`TS_MLX_Q5K_REPACK=0` keeps the original scalar Q5_K converter, including its
zero-fill pass. Unset or `1` selects the optimized converter. The choice is read
when a weight is first converted; cached weights must be released before an
alternating comparison can exercise both converters. Separate fresh processes
are appropriate for complete-generation controls, with file residency and
first-use compilation reported independently.

`MlxQ5KRepackTests` independently constructs logical quant values and encodes
the GGML fields, then compares all packed bytes, F16 scale/bias bits, source
immutability and output guards for serial and parallel sizes. Its native tests
force production converter modes `0,1,0` with cache eviction between calls and
require bitwise-identical full matmul/gather outputs for both the ordinary
activation policy and explicit F32 preservation. Existing MLX K-quant tests and
LTX official tiny fixtures cover the consuming operations separately.

```sh
MLX_ENABLE_TF32=0 TS_TEST_LTX25_BACKEND=mlx \
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true \
  --filter 'FullyQualifiedName~MlxQ5KRepackTests|FullyQualifiedName~MlxBackendTests.MlxQuantizedMatmul|FullyQualifiedName~MlxBackendTests.MlxQuantizedRows|FullyQualifiedName~Ltx25PromptEncoderTests|FullyQualifiedName~Ltx25DenoiserTests'

dotnet build eng/validation/MlxQ5KPackProbe -c Release
dotnet eng/validation/MlxQ5KPackProbe/bin/Release/net10.0/MlxQ5KPackProbe.dll 7 \
  > "$NEW_Q5_REPACK_REPORT"
```

The CPU-only microbenchmark checks exact packed/affine bytes and alternates
the original converter, optimized serial converter and bounded parallel
converter over representative LTX matrix shapes. It records every timing and
median, includes the original converter's required zeroing, and excludes
allocation, file paging, device upload, F32 parameter preparation and inference.
Its speedup is not an end-to-end speedup. Complete-generation comparisons must
still require exact RGB/PCM outputs and independent media validation.

## Convolutional video decoder

`Ltx25VideoVaeDecoder(path, allocator)` loads the convolutional checkpoint's
embedded architecture and validates every decoder tensor shape. `DecodeRaw`
accepts channel-major `[128,T,H,W]` normalized latents and returns channel-major
`[3,8*(T-1)+1,32*H,32*W]` unclamped output. `Decode` returns frame-major RGB floats
in `[0,1]`. The optional trace callback receives independent channel-major stage
snapshots. The caller owns the allocator; disposing the decoder releases its
mapped checkpoint and tensors.

BF16 checkpoint values are widened exactly to Float32; activations and
convolutions use Float32. CUDA uses existing zero-padded spatial columns and
pedantic Float32 GEMM. MLX uses a separate TensorSharp-owned zero-column kernel
and device matmul. Convolution weights are streamed per operation and MLX frame
boundaries evaluate pending device graphs without copying pixels to the host.
Checkpoint conversion writes directly into each backend's owned host weight
buffer; spatial column bands are bounded to 32 MiB. Weights are not retained
between decode calls.
CUDA shuffles are composed from supported three-/four-axis native copies;
unsupported layouts fail rather than falling back to host copies.
The decoder preserves the original replicated symmetric temporal padding,
pixel-normalization epsilon, first-frame removal at each temporal upsample and
the final width-before-height patch order. Unsupported diffusion, causal,
stochastic, conditioned or residual-upsample variants fail explicitly.

The decoder is the latent-to-RGB component. The separate causal video encoder
and generation pipeline implement image guides and Ingredients conditioning;
decoder temporal tiling remains unsupported. The decoder numerical reference
uses original Float32 operations with the same widened checkpoint and does not
claim bitwise BF16 activation parity.

```sh
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj \
  --filter FullyQualifiedName~Ltx25VideoVaeTests \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true

MLX_ENABLE_TF32=0 TS_TEST_LTX25_VAE_BACKEND=mlx \
  dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj --no-build \
  --filter FullyQualifiedName~Ltx25VideoVaeTests

TS_TEST_LTX25_VAE_BACKEND=cuda \
  dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj --no-build \
  --filter FullyQualifiedName~Ltx25VideoVaeTests

dotnet publish eng/validation/Ltx25VideoVaeProbe/Ltx25VideoVaeProbe.csproj \
  -o artifacts/ltx25/video-vae-probe \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true

python eng/validation/ltx25-video-vae-reference.py \
  --source artifacts/ltx25/reference-original --checkpoint "$CONV_VIDEO_VAE" \
  --shape 3 16 16 --threads 4 --output "$NEW_REFERENCE_DIRECTORY"

MLX_ENABLE_TF32=0 \
TENSORSHARP_MLX_LIBRARY_DIR="$PWD/TensorSharp.Backends.MLX/Native/dist" \
  dotnet artifacts/ltx25/video-vae-probe/Ltx25VideoVaeProbe.dll \
  --model "$CONV_VIDEO_VAE" --backend mlx \
  --latent-frames 3 --latent-height 16 --latent-width 16 \
  --input "$NEW_REFERENCE_DIRECTORY/input.f32" --trace 1 \
  --warmup 0 --repeats 1 --output "$NEW_PROBE_DIRECTORY"

python eng/validation/ltx25-compare-video-vae.py \
  --probe "$NEW_PROBE_DIRECTORY" --reference "$NEW_REFERENCE_DIRECTORY" \
  --output "$NEW_QUALITY_REPORT"
```

The reference tool requires PyTorch, safetensors, NumPy, einops and the original
source's PyAV/colour import dependencies; it records their actual versions.
Omitting `--checkpoint` produces the synthetic BF16-stored tiny fixture.

The causal video encoder has independent, full-stage official fixtures for an
image and a nine-frame video. They validate the trained graph's patch packing,
causal padding, residual space-to-depth channel averages and normalized means.
Run them with the same `Ltx25VideoVae` test filter and backend environment above.
The encoder streams Float32 convolution weights and folds identical temporal
taps for a still image. Its MLX residual downsample is a TensorSharp-owned Metal
kernel that computes the grouped skip average and layout without CPU fallback.
CUDA computes grouped residual averages with the existing TensorSharp native
adaptive channel-average kernel using one head, avoiding generic Reduce's CPU
fallback. Both implementations preserve Float32 latent range and accumulation.

For a trained image encoder comparison, publish and run the reusable probe:

```sh
dotnet publish eng/validation/Ltx25VideoVaeEncoderProbe/Ltx25VideoVaeEncoderProbe.csproj \
  -c Release -o artifacts/ltx25/video-vae-encoder-probe \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true

MLX_ENABLE_TF32=0 TENSORSHARP_MLX_LIBRARY_DIR="$PWD/TensorSharp.Backends.MLX/Native/dist" \
  dotnet artifacts/ltx25/video-vae-encoder-probe/Ltx25VideoVaeEncoderProbe.dll \
  --model "$CONV_VIDEO_VAE" --image imgs/banner_1.png --backend mlx \
  --height 512 --width 512 --trace 1 --output "$NEW_ENCODER_PROBE_DIRECTORY"

python eng/validation/ltx25-video-vae-encoder-reference.py \
  --source artifacts/ltx25/reference-original --checkpoint "$CONV_VIDEO_VAE" \
  --pixels "$NEW_ENCODER_PROBE_DIRECTORY/input.f32" --shape 1 512 512 \
  --threads 4 --output "$NEW_ENCODER_REFERENCE_DIRECTORY"

python eng/validation/ltx25-compare-video-vae.py \
  --probe "$NEW_ENCODER_PROBE_DIRECTORY" --reference "$NEW_ENCODER_REFERENCE_DIRECTORY" \
  --stages encoded --max-absolute 0.0001 --relative-l2 0.00001 \
  --output "$NEW_ENCODER_LATENT_QUALITY_REPORT"

python eng/validation/ltx25-compare-video-vae.py \
  --probe "$NEW_ENCODER_PROBE_DIRECTORY" --reference "$NEW_ENCODER_REFERENCE_DIRECTORY" \
  --max-normalized-absolute 0.00002 --relative-l2 0.00001 \
  --output "$NEW_ENCODER_STAGE_QUALITY_REPORT"
```

The final normalized latent gate is absolute error at most `1e-4` and relative
L2 error at most `1e-5`. For intermediate stages the additional diagnostic gate
uses relative L2 at most `1e-5` and maximum absolute error divided by the stage's
maximum reference magnitude at most `2e-5` (20 ppm). This matches the scale of
the synthetic stage fixture tolerance, while accounting for trained residual
downsample activations whose magnitudes exceed `2e5` before output pixel RMS
normalization. Keep the default unscaled absolute report too: it records these
large absolute differences instead of concealing them behind a changed bound.
For performance, use a separate probe run with `--trace 0 --warmup 1 --repeats 3`;
its synchronized timings cover the encoder plus normalized latent readback only.
Checkpoint hashing precedes timing and warms the file cache; image decoding,
resizing, hashing and encoder construction are excluded. A first invocation in
a fresh probe process therefore does not measure cold-file-cache latency.
These component checks do not establish generated-video quality or comparison
with stable-diffusion.cpp; run the complete generation checks separately.

Ingredients identity mode (`ref`) is a separate optional IC-LoRA path. It
requires the licensed adapter and accepts identity images through a single
image or a reference-image list; temporal anchors cannot be mixed into it.
Defaults are 768×448, 121 frames and 24 fps, with at least 121 output/reference
frames on the 8n+1 grid. A grid of whole reference images is fitted onto black
without labels; alpha is composited onto black before resizing. Its static RGB
video is encoded with `EncodeStaticVideo`: the validated causal encoder's
repeat-first temporal padding, channel-wise pixel normalization, pointwise
activations and space-to-depth blocks preserve identical frames. Evaluating
one image graph and repeating its normalized channel-major output therefore
computes this static video's full reference latents, up to the Float32 temporal
tap-folding rounding already present in image encoding. This specialization
avoids repeated RGB allocation and convolution; arbitrary videos retain the
full temporal `Encode` path. MLX convolutions also evaluate separate frame
buffers and concatenate once, avoiding whole-clip copies on each band update.
The implementation appends all normalized reference
video latents, using the unchanged official `VideoConditionByReferenceLatent`
causal positions with spatial/temporal scale one, clean mask zero and unmarked
reference tokens. The generated canvas retains its first-frame marker and only
canvas tokens are decoded as output. Stock image guides keep their original path.
For reference sheets, use clean subject/object views without captions, logos or
unrelated collage elements, and describe the actual positions and identities.
For two same-subject panels, distinguish the left full-body/bust view from the
right face view, state that both depict the same character, then describe the
generated scene and action separately. Explicit captions are important for
multi-character/object sheets. These are publisher-oriented input guidelines,
not a guarantee of layout or motion: the same-seed position-caption-only test
with a branded banner still copied its sheet layout. The same seed also copied
two panels after both inputs became clean text-free bust/face crops and the
caption was clarified. A single input gives more predictable composition in
tested samples; multiple identity views can preserve the sheet layout.

`generate-ltx25-reference-conditioning-oracle.py` generates an independent
official fixture for full reference latents, coordinates, masks, noise and
masked Euler. The synthetic encoder reference supports `--static-frame --shape
121 16 32` to verify genuine 121-frame RGB encoding against the original
temporal network, and the same fixture checks the static specialization.
Use the encoder probe's `--static 1 --frames 121` with one RGB frame as input
to measure the specialized path; the report records its logical video shape.
The official oracle's `--terminal-only` suppresses large intermediate snapshot
files but does not reduce the original model's working memory. Keep large
CPU oracles and GPU benchmarks in exclusive lanes to avoid shared-memory swap.
The unchanged official trained Conv VAE at 121×64×64 produced bitwise identical
latent time slices; repeating its one-frame output differed by maximum
1.1921e-6 and relative L2 1.9142e-7. This checks trained weights alongside the
independent tiny 121-frame fixture and the graph invariance argument. The full
121×512×512 official run was interrupted under memory contention and is not
passing coverage; the existing actual 512 image oracle covers tap-fold rounding.
These fixtures are embedded in the test project, outside the
generated evidence directories. Run `Ltx25ReferenceConditioning` together with
`Ltx25VideoVae`, `Ltx25ImageConditioning` and `Ltx25GenerationSettings` tests on
each available backend. LoRA projection numerics and real single/multiple-image
Ingredients generation require separate validation; these conditioning/encoder
oracles alone do not establish adapter quality or end-to-end performance.

For independent CPU-only review of decoded videos, use the exact preprocessed
RGB file from the encoder probe as the source reference, and the raw RGB outputs
from each generation probe. For example:

```sh
python eng/validation/ltx25-review-video.py \
  --reference-rgb "$NEW_ENCODER_PROBE_DIRECTORY/input.f32" \
  --video TensorSharp "$NATIVE_RAW_RGB" f32 17 \
  --video stable-diffusion.cpp "$BASELINE_RAW_RGB" u8 17 \
  --width 512 --height 512 --fps 24 --samples 9 \
  --crop 0 0 224 280 --output "$NEW_VISUAL_REVIEW_DIRECTORY"
```

The optional crop above isolates the face and hands in the uploaded banner;
choose a suitable region for a different reference. The tool exports full and
detail contact sheets, full-resolution sampled frames, source/frame PSNR and
MAE, and adjacent-frame changes. These statistics have no quality pass bound:
source similarity does not measure intended movement, and greater or smaller
motion does not establish a quality winner. Record visual observations of
identity, composition and the requested gesture separately; a short clip with
slight motion cannot demonstrate a complete blowing-kiss gesture. Use longer
clips and inspect early, middle and final frames before claiming that behavior.

Traced probes include intermediate readbacks and are quality diagnostics, not
performance measurements. The comparator verifies checkpoint/input/stage
hashes and dimensions, then requires every stage to pass both recorded bounds.
For a production-size raw-output check without saving intermediate snapshots,
run the probe with `--trace 0` and the comparator with `--stages decoded`. The
report explicitly lists the original intermediate stages not compared in that
run. CUDA probe deployment requires the verified device-compatible primary PTX
at `cuda_kernels/tensorsharp_kernels.ptx`; a publish may contain a different
architecture's bundled asset, so install the validated asset after copying the
managed bundle and retain its hash.
For warm decoder timing, use a fresh probe directory with `--trace 0 --warmup 1
--repeats 3`. Those samples include streamed weight conversion/upload and the
final synchronized raw-output readback, but exclude checkpoint construction,
the warm-up decode, prompt conditioning, denoising, audio and media encoding.

## Tools

- `ltx25-download-remote.py` resolves pinned official companion metadata using
  local cached Hugging Face authentication. Temporary signed download locations
  travel through SSH stdin; tokens and locations are excluded from argv and
  reports. Copy the helper to the VM before invoking local mode with
  `--host`, `--port`, `--identity`, `--remote-script`, `--models`, `--output`,
  `--public-manifest` and `--companion-manifest`. The remote report includes
  actual size/SHA256 validation for every file; an interrupted download retains
  its `.ltx25-download-partial` file.
- `ltx25-sdcpp-bench.py` checks a clean source export manifest and the compiled
  CLI hash, verifies the complete pinned model set, runs generation, and decodes
  video/audio with FFmpeg. `--identity-only` checks source/binary provenance and
  explicitly records that inference was not executed. Successful generation
  records dimensions, frame count, audio rate/duration, decoded hashes and
  sampled GPU telemetry. The pinned official audio VAE includes bandwidth
  extension and emits 48 kHz stereo. Use the **conv** video VAE; the default
  diffusion video decoder is a different architecture.
- `build-ltx25-sdcpp-denoiser.py` compiles the owned
  `ltx25-sdcpp-denoiser.cpp` against an already built, unchanged SD.cpp tree.
  Supply `--build`, a fresh `--output`, `--source-manifest` and `--build-manifest`.
  It verifies every source export entry and the original CLI, then records
  compiler/link arguments and source/library/binary hashes.
- The resulting `ltx25-sdcpp-denoiser` executable accepts `--model`, `--output`,
  `--backend`, `--blocks`, `--warmup`, `--repeats`, `--latent-width`,
  `--latent-height`, `--latent-frames`, `--audio-tokens`, `--context-tokens` and
  `--sigma`. Use actual device names from SD.cpp `--list-devices`, such as
  `CUDA0` or `MTL0`. Defaults match `Ltx25DenoiserProbe`: 2×2×2 latent video,
  16 audio tokens, 8 projected context tokens, sigma 0.725, one warmup and five
  measured forwards. Deterministic input and output F32 files allow independent
  hash/layout checks.
- `Ltx25CudaDequantProbe` compares the unchanged primary CUDA PTX's generic
  weight-to-F16 conversion with the optional owned K-quant block converter.
  It uses resident synthetic weights, requires exact finite half-byte parity
  before and after timing, alternates route order, and records CUDA-event and
  synchronized wall samples plus actual PTX/managed-binary hashes. It excludes
  upload, allocation, readback, GEMM and model execution; its speedup cannot be
  interpreted as a model or generation speedup.

For the pinned distilled checkpoint, a representative complete SD.cpp run uses
512×512, 33 frames, 24 fps, 8 Euler steps, CFG 1 and explicit sigmas:

```text
1,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0
```

Whole CLI timings include model loading and codecs; new processes do not measure
warm-model latency or establish cold-file-cache behavior. Denoiser timings exclude
the text encoder, connectors, sampler, VAEs, media codecs and applications. The
owned SD.cpp denoiser wrapper bypasses caption projection/connectors and absolute
keyframe embedding at runtime to match already projected, unmasked synthetic
TensorSharp inputs. SD.cpp's independent rotary/timestep conventions mean its
output differences are diagnostic; the original LTX oracle provides numerical
correctness coverage. Structural media validity and component speed do not prove
perceptual quality or end-to-end TensorSharp/TensorAgent performance parity.

## Pinned inputs

| Input | Revision or identity |
|---|---|
| Original Lightricks/LTX-2 source | `be6d6d0a69d11a0a399c462ed92606692e5f8697` |
| stable-diffusion.cpp source | `d637cfb935feb2254898a8f1524046c958f3df80` |
| Baseline's committed ggml submodule | `d25b121ce1d6ae0c7e62ad080460d9fc1e46dd93` |
| TensorSharp ggml checkout used for this work | `3612c0af93f2cfc8f048295300a61187723b12ae` |
| `Abiray/LTX-2.5-Distilled-GGUF` | `7b0c2025441f1bf12c18eac375ad21f5e3d3c9e0` |
| `LTX-2.5-Distilled-Q4_K_M.gguf` SHA256 | `0f51eb0d82b19bddbfb3b0371a65217844ea03750f27dd733528f22152e0e0d0` |
| Official `Lightricks/LTX-2.5` companions | `49b093a39c19169a8a81721dcc541e07e945232f` |

Use the conv video VAE, not `ltx-2.5-video-vae-bf16.safetensors`, which is a distinct
diffusion decoder unimplemented by the pinned baseline. The authentic custom
`gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` includes LTX's text projection;
stock Gemma 4 is not equivalent. The smaller official convrot int8 encoder needs
its rotation transform and cannot be treated as plain scaled int8.

`ltx25-download-companions.py --video-vae conv` selects the conv decoder from an
authenticated, pinned official access manifest. Default `diffusion` retains the
original distinct file choice. A selected decoder requires its own unique
authenticated metadata entry. Existing files are size/hash checked; only a
selected file's matching resume data is credited by the disk preflight.

## Tests and independent fixtures

Run from the repository root with .NET 10. The skip properties below disable
native rebuilding, not assertions. Build native dependencies normally first if
unavailable.

```sh
dotnet test TensorAgent/tests/TensorAgent.Tests/TensorAgent.Tests.csproj \
  --filter FullyQualifiedName~VideoTurnsTests \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true

dotnet test TensorAgent/tests/TensorAgent.Tests/TensorAgent.Tests.csproj \
  --filter 'FullyQualifiedName~Ltx25CatalogTests|FullyQualifiedName~CatalogTests|FullyQualifiedName~ModelBrowserTests|FullyQualifiedName~LocalizationTests' \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true

dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj \
  --filter 'FullyQualifiedName~Ltx25DenoiserTests|FullyQualifiedName~Ltx25SamplingTests|FullyQualifiedName~Ltx25HiddenStateFeaturesTests' \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true

python3 -m unittest discover -s eng/validation/tests -p 'test_ltx25_*.py' -v
```

The two page-localization JavaScript fixtures use JavaScriptCore on Apple hosts
and Node.js elsewhere. On Linux, provide Node on `PATH` or set
`TENSORAGENT_TEST_NODE` to a verified portable executable; a missing runtime is
reported as skipped coverage. A VM run of a locally compiled managed test bundle
via `dotnet vstest` validates its managed behavior on that VM, without claiming a
VM source build or native backend compilation. Record bundle and source hashes.

Denoiser tests default to CPU. Explicit device runs select the actual backend;
an unavailable requested device is not passing execution coverage. Upstream MLX
uses TF32 matrix multiplication by default; LTX and TensorAgent set an unset
variable to `MLX_ENABLE_TF32=0` before first GEMM. Set it at launch for LTX quality: the
strict Float32 original gate requires it. Full-depth actual-weight validation
showed substantially smaller original-reference errors with negligible warm
latency cost in the tested small synthetic case (8 video, 16 audio and 8 context
tokens). Larger geometry needs its own cost measurement. Performance runs must
record the setting and compare the same setting before and after.

```sh
MLX_ENABLE_TF32=0 TS_TEST_LTX25_BACKEND=mlx \
  dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj --no-build \
  --filter FullyQualifiedName~Ltx25DenoiserTests

TS_TEST_LTX25_BACKEND=cuda \
  dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj --no-build \
  --filter FullyQualifiedName~Ltx25DenoiserTests

dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj --no-build \
  --filter FullyQualifiedName~Ltx25CudaProjectionTests

dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj --no-build \
  --filter FullyQualifiedName~Ltx25CudaF16DequantTests

TS_CUDA_QMM_F16GEMM_MIN_ROWS=8 TS_CUDA_QMM_F16GEMM_MAX_MB=768 \
  dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj --no-build \
  --filter FullyQualifiedName~ConfiguredF16ThresholdKeepsItsProjectionRouteAndTargetUnchanged

TS_LTX25_CUDA_ATTN=1 dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj \
  --no-build --filter FullyQualifiedName~Ltx25CudaAttentionTests
```

Optional projection tests visibly skip without CUDA. On a CUDA host they require
the owned `tensorsharp_ltx25.ptx` module, actual native execution counters, and
parity with independently dequantized GGML weights and a double dot-product
oracle. A missing module or fallback does not count as a passing kernel test.
The owned projection module is optional and enabled when available; set
`TS_LTX25_CUDA_TILE=0` to opt out or `TS_LTX25_CUDA_REQUIRE_TILE=1` to require it
for the denoiser. It handles rows 2..31 and preserves the existing eligible F16
route configured by `TS_CUDA_QMM_F16GEMM_MIN_ROWS` and
`TS_CUDA_QMM_F16GEMM_MAX_MB`. Those settings are process-static: run the threshold
fact alone in a fresh process for a nondefault threshold.

The bounded precise CUDA attention route is enabled by default for video
self/prompt attention with at least 512 queries and 512 keys. Set
`TS_LTX25_CUDA_ATTN=0` to opt out or `=1` to force supported smaller shapes in
focused tests. It uses pedantic Float32 cuBLAS products, Float32 softmax and a
bounded score workspace (at most 256 MiB). Long sequences tile queries and heads
while every query keeps the complete key sequence. The 10,752-token Ingredients
oracle covers query/head tails and double-precision samples, plus a second
workspace tiling across every output element. Unsupported geometry retains the
existing route. Execution counters and observed workspace capacity are recorded
by the denoiser and complete-generation probes; a setting alone is not evidence
that the route executed.

For the preset's 768 video tokens, compare an explicit attention opt-out with
the unset/default policy in separate processes and fresh report paths:

```sh
TS_LTX25_CUDA_ATTN=0 dotnet "$DENOISER_PROBE/Ltx25DenoiserProbe.dll" \
  --model "$DENOISER" --backend CUDA --frames 17 --height 512 --width 512 \
  --audio-tokens 18 --context-tokens 1024 --blocks 48 \
  --warmup 1 --repeats 3 --schedule forward --frequency-precision Float64 \
  --output "$NEW_REPORT"
```

Repeat with `TS_LTX25_CUDA_ATTN` unset. An explicit `=1` additionally exercises
supported smaller video shapes. Verify identical model/input/PTX hashes,
finite outputs, zero repeat drift and positive native attention counters.
These inputs are deterministic synthetic projected contexts and latents;
component timings exclude prompt encoding, sampling and decoding.

The owned blockwise weight converter is enabled by default when its PTX export
is available, for the existing eligible F16 GEMM route. Set
`TS_LTX25_CUDA_F16_DEQUANT=0` to opt out. Its operands retain that route's
Float16 policy; this is a conversion optimization rather than an additional
activation-precision change. The converter's exact-half tests compare
with unchanged GGML CPU decoding followed by nearest-even half rounding,
including zero/subnormal scales, ragged block counts and output-view guards.
An older tile-only PTX without the converter export retains the existing route
in default automatic mode. Explicit F16 enablement combined with required-native
mode fails if the converter export is absent.

Run the isolated conversion benchmark on an available CUDA device after unit
parity checks, with no concurrent timing workloads:

```sh
dotnet build eng/validation/Ltx25CudaDequantProbe/Ltx25CudaDequantProbe.csproj \
  -c Release -o artifacts/ltx25/dequant-probe

dotnet artifacts/ltx25/dequant-probe/Ltx25CudaDequantProbe.dll \
  --main-ptx /path/to/validated/cuda_kernels/tensorsharp_kernels.ptx \
  --inner 4096 --columns 16384 --types 12,13,14 --warmup 2 --repeats 10 \
  --output docs/validation/ltx25/dequant-microbench.json
```

The tool also loads `tensorsharp_ltx25.ptx` through the session's normal discovery
and records its resolved path/hash. Deploy a compatible module next to the tool;
missing CUDA, incompatible PTX or a missing converter fails explicitly.
Its `--inspect-module 1 --output /fresh/report.json` mode instead inspects the
actual owned-module resolver without initializing a CUDA allocator or driver.
It records a resolved path/hash or null plus application/assembly directories;
use it in the same isolated deployment to substantiate missing-module controls.
Keep benchmark binaries frozen and build later diagnostics into a separate
directory so their recorded identities remain recoverable.

Regenerate required fixtures only with an unchanged original checkout at the
pinned revision and Python with torch, numpy, einops, safetensors and gguf.
Fixture JSON records actual generator versions. Regeneration changes reviewed
test inputs; routine validation uses the committed fixtures.

```sh
ltx_reference="$PWD/artifacts/ltx25/reference-original"
ltx_python="$PWD/artifacts/ltx25/denoiser-oracle-env/bin/python"

"$ltx_python" eng/validation/generate-ltx25-denoiser-oracle.py \
  --reference "$ltx_reference" --output InferenceWeb.Tests/Fixtures/Ltx25

"$ltx_python" eng/validation/ltx25-sampling-reference.py \
  --source "$ltx_reference" \
  --output InferenceWeb.Tests/Fixtures/Ltx25/euler-float32.json

"$ltx_python" eng/validation/generate-ltx25-hidden-state-features.py \
  --source "$ltx_reference" \
  --output InferenceWeb.Tests/Fixtures/Ltx25/hidden-state-features.json

# Imports ltx_pipelines, whose package imports need the reference's full
# dependency set: torch/torchaudio/torchvision, av, colour-science (not the
# unrelated PyPI "colour"), transformers<5.15, accelerate, scipy, OpenImageIO.
# Use a separate venv such as artifacts/ltx25/sampling-oracle-env.
artifacts/ltx25/sampling-oracle-env/bin/python eng/validation/ltx25-ancestral-sampling-reference.py \
  --source "$ltx_reference" \
  --output InferenceWeb.Tests/Fixtures/Ltx25/euler-ancestral-float32.json
```

The tiny checkpoint covers the complete two-block AV graph, independent sigma
and per-token timesteps, keyframe embedding, Float32/Float64 RoPE, coordinate
changes and cache residency rebuilding. Its GGUF declares the published
`config.transformer` multipliers (`timestep_scale_multiplier` 1000,
`av_ca_timestep_scale_multiplier` 1000), and the generator builds the official
model with them. Constructing `LTXModel` with keyword defaults would silently
use an a2v/v2a gate multiplier of 1, which is not what the pinned checkpoint
declares; earlier fixtures and reference reports shared that assumption.
`ltx25-official-denoiser-reference.py` likewise reads the multipliers from the
probe's GGUF and records them in its report. Sampling tests check original
`to_denoised`/`EulerDiffusionStep` Float32 arithmetic and reject unsupported
conditioning. The ancestral fixture runs the unchanged
`euler_ancestral_denoising_loop` over the eight distilled sigmas with recorded
noise and clean, partial-strength and generated video tokens, so both
`post_process_latent` points, the renoise and the noise-free terminal step are
checked at every step. Synthetic fixtures do not establish pretrained video quality.

The hidden-state fixture executes original normalization/rescaling functions
and verbatim AST nodes for the standalone right-pad helpers, avoiding unrelated
optional Gemma imports. It covers left/right/scattered/all-valid/all-pad masks,
zero states, different per-stream widths and nonzero projected padding rows.
The required input state tuple order includes the embedding state. Packing
`[tokens, hiddenSize, stateCount]` varies **state fastest**; concatenating whole
layer vectors yields a different learned-projection input. Rescaling divides by
the encoder hidden size, not that flattened width. Reordering happens after
the learned aggregate projections, where biases can make pad rows nonzero.

For the pinned custom Gemma file, hidden size is 3840 with 48 transformer
layers, giving 49 states and 188160 packed features per token. It embeds its
tokenizer assets and `gemma4-12b-ltx-v1` config; the original text path strips the
prompt, ensures BOS, appends no EOS and pads to 1024 tokens. The learned video
and audio aggregate matrices/biases are in that file (4096/2048 output widths).
The pinned denoiser GGUF separately contains **eight** connector blocks per
stream and 128 learned registers, so constructor defaults cannot determine its
connector geometry. Existing TensorSharp Gemma 4 generation returns last-token
logits; an all-token/all-state extraction/import/tokenizer path and independent
authentic-checkpoint parity remain necessary before these helpers can be wired
into generation.

## Face fidelity: sampler and cross-attention gate A/B

The publisher's `DistilledPipeline` samples LTX 2.5+ checkpoints with ancestral
Euler (eta 1), and the pinned checkpoint's config sets the a2v/v2a gate multiplier
to 1000. TensorSharp now does both; `euler_ancestral` is the default sampler and
`euler` remains an explicit deterministic diagnostic. Step noise is one Philox
draw per step (word 0 = step index) keyed by seed + 10000, video tokens first and
audio after; bit parity with Torch's generator is not claimed.

To compare against the earlier behavior, build the end-to-end probe from the
unmodified earlier commit in a separate worktree. Its pinned `euler` and
default gate reproduce the old output. Then run the current probe with
`--sampler euler` (gate fix only) and without it (both fixes), using identical
model, image, prompt, geometry and seed. The current probe records `sampler` and
`settings.Ancestral` in its report. For a one-block real-checkpoint check of the
gate, run `Ltx25DenoiserProbe` and `ltx25-official-denoiser-reference.py` as
described below; the reference report records the multipliers it used.

The recorded Apple Silicon comparison (512×512, 97 frames, banner first frame,
detailed Chinese prompt, seeds 42 and 7) reproduced the earlier distorted server
sample bit for bit. With both fixes, faces stayed coherent through the gesture
and kept their identity in later frames. The one-block MLX output matched the
corrected official reference within relative L2 3.3e-6 (video) and 2.7e-6
(audio). Framing experiments are summarized in `docs/models/ltx-2.5.md`
(Clear faces). The visual conclusions come from two seeds and one subject. They
are not a broad perceptual benchmark. Evidence is under `docs/validation/ltx25/face-fix/`.

## Two-stage recipe and latent upscaler

`Ltx25LatentUpsampler` ports the publisher's `LatentUpsampler` used between the
half-size and full-size passes. `generate-ltx25-latent-upsampler-oracle.py` builds
the tiny embedded fixture (`latent-upsampler-tiny.safetensors`,
`latent-upsampler.json`: 16 latent / 64 mid channels, two channels per GroupNorm
group, three frames for both temporal edges) with the unchanged original class.
`Ltx25LatentUpsamplerTests` compares CPU, or MLX/CUDA through
`TS_TEST_LTX25_VAE_BACKEND`, with it. For the actual checkpoint,
`Ltx25LatentUpsamplerProbe` upsamples saved `[128,T,H,W]` latents and
`ltx25-latent-upsampler-reference.py` runs the unchanged original on the same input:

```sh
dotnet build eng/validation/Ltx25LatentUpsamplerProbe -c Release -o artifacts/ltx25/upsampler-probe \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
MLX_ENABLE_TF32=0 TENSORSHARP_MLX_LIBRARY_DIR="$PWD/TensorSharp.Backends.MLX/Native/dist" \
dotnet artifacts/ltx25/upsampler-probe/Ltx25LatentUpsamplerProbe.dll --backend mlx \
  --upsampler "$ltx_models/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  --vae "$ltx_models/vae/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --latents latents.f32 --output "$ltx_evidence/upsampler-mlx"
"$ltx_python" eng/validation/ltx25-latent-upsampler-reference.py --source "$ltx_reference" \
  --upsampler ... --vae ... --latents latents.f32 --probe "$ltx_evidence/upsampler-mlx" \
  --output "$ltx_evidence/upsampler-mlx-reference.json"
```

`latents.f32` holds Float32 `[128,T,H,W]` with its shape in `latents.f32.json`
(`{"shape": [128, T, H, W]}`). On the 512×512×97 first pass of the banner sample
the MLX upsampler matched the original within relative L2 1.9e-5.
`ltx25-decode-latents-reference.py` decodes such latents with either published
decoder (`--vae` selects the diffusion or convolutional checkpoint) for decoder
A/B comparisons. Without NATTEN or Triton the original falls back to eager tiled
SDPA neighborhood attention, which takes about two minutes for 512×512×97 on MPS
and over an hour for a full 1024×1024 clip. A temporal crop of the latents is not
a fair decoder test: the crop's first latent frame is decoded as a one-frame causal
latent.

The hand-over-face comparison (banner, detailed Chinese prompt, seed 42) is
summarized in `docs/models/ltx-2.5.md` (Clear faces and hands). The 1024×1024 run
and the old 512×512 single pass share their first-pass noise and inputs, so it
isolates the refinement. Back-to-back GPU runs throttled the M5 Pro by about 3.4×
(18.4 s to 63 s per 512×512 step with identical output); rerun a baseline before
attributing slowdowns to code.

## Diffusion video decoder

`Ltx25DiffusionVideoDecoder` ports the publisher's opt-in `NADiffusionDecoder`
(`ltx-2.5-video-vae-bf16.safetensors`, `--video-diffusion-vae`).
`generate-ltx25-diffusion-vae-oracle.py` builds the tiny embedded fixture
(`diffusion-vae-tiny.safetensors`, `diffusion-vae.json`/`.bin`) through the unchanged
original configurator. It traces the stage-four input, the cropped context and the
output of an edge-padded 2×1×2 latent with saved pixel noise.
`Ltx25DiffusionVideoDecoderTests` compares CPU, or MLX through
`TS_TEST_LTX25_VAE_BACKEND=mlx`, stage by stage within relative L2 2e-5. CUDA is
refused and has no coverage.

For the actual checkpoint, `ltx25-diffusion-vae-reference.py` decodes saved
`[128,T,H,W]` latents with the original's untiled `_decode_pixels` and saves the pixel
noise it drew. `Ltx25DiffusionVaeProbe --noise` decodes the same canvas:

```sh
dotnet build eng/validation/Ltx25DiffusionVaeProbe -c Release -o artifacts/ltx25/diffvae-probe \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
"$ltx_python" eng/validation/ltx25-diffusion-vae-reference.py --source "$ltx_reference" \
  --vae "$ltx_models/vae/ltx-2.5-video-vae-bf16.safetensors" --latents latents.f32 \
  --output "$ltx_evidence/diffvae-ref" --device mps --dtype float32 --seed 42
MLX_ENABLE_TF32=0 TENSORSHARP_MLX_LIBRARY_DIR="$PWD/TensorSharp.Backends.MLX/Native/dist" \
dotnet artifacts/ltx25/diffvae-probe/Ltx25DiffusionVaeProbe.dll --backend mlx \
  --vae "$ltx_models/vae/ltx-2.5-video-vae-bf16.safetensors" --latents latents.f32 \
  --noise "$ltx_evidence/diffvae-ref.noise.f32" --output "$ltx_evidence/diffvae-mlx"
"$ltx_python" eng/validation/ltx25-diffusion-vae-reference.py ... --probe "$ltx_evidence/diffvae-mlx"
```

Without `--noise` the probe decodes from `--seed`, as generation does, and `--mp4`
writes a video. MLX matched the original Float32 decode (MPS, eager SDPA fallback)
within relative L2 9.3e-7 on the 512×512×97 banner first pass and 7.7e-7 on the
top-left 512×512 crop of the two-stage 1024×1024×97 latents. The original needs more
than 64 GB of MPS memory for the full 1024×1024×97 decode, so that size was checked
by crop, and TensorSharp's whole-clip decode was compared between its two attention
kernels (relative L2 1.3e-7). Decoder comparisons are summarized in
`docs/models/ltx-2.5.md` (Video decoders): one subject and one seed.

## Actual-checkpoint probe and official reference

```sh
ltx_models="/Users/zhongkaifu/work/models/ltx25"
ltx_probe="$PWD/artifacts/ltx25/probe"
ltx_evidence="$PWD/docs/validation/ltx25"

dotnet build eng/validation/Ltx25DenoiserProbe/Ltx25DenoiserProbe.csproj \
  -c Release -o "$ltx_probe" \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true

MLX_ENABLE_TF32=0 dotnet "$ltx_probe/Ltx25DenoiserProbe.dll" \
  --model "$ltx_models/LTX-2.5-Distilled-Q4_K_M.gguf" \
  --backend MLX --blocks 1 --frames 9 --width 64 --height 64 \
  --audio-tokens 16 --context-tokens 8 --frequency-precision Float64 \
  --warmup 1 --repeats 3 --schedule forward \
  --output "$ltx_evidence/mlx-precise-one-block.json"

"$ltx_python" eng/validation/ltx25-official-denoiser-reference.py \
  --reference "$ltx_reference" \
  --probe "$ltx_evidence/mlx-precise-one-block.json" \
  --output "$ltx_evidence/mlx-precise-one-block-official.json" --threads 8
```

Use fresh output names: the tools refuse to overwrite evidence. The probe
records checkpoint/managed-binary hashes, input/output hashes, frequency
precision, environment settings, executed depth and synchronized first-call/warm
timings. `--blocks 48` runs all pretrained blocks. `--schedule distilled` runs
the eight-step latent sampler; its sampled latents are refused by the
single-forward velocity oracle.

The official CPU reference independently dequantizes with gguf-py and runs
unchanged Lightricks forward methods. It defaults to four blocks. For explicit
full-depth execution supply `--max-blocks 48`. Full materialization needs
over 70 GiB for this pinned denoiser's Float32 parameters alone, before decoding
temporaries: inspect actual cgroup v1/v2 limits rather than host RAM.
`--stream-block-weights` loads one block through standard module pre-hooks and
returns its parameters to meta storage after its unchanged forward. Non-block
weights persist. `--dequant-workers 8` parallelizes independent tensor decoding,
using additional temporary RAM and CPU quota. The report records actual cgroup
limits, residency mode and every streamed block. Tiny one/two-block streamed
and materialized equivalence is validated separately.

```sh
"$ltx_python" eng/validation/ltx25-official-denoiser-reference.py \
  --reference "$ltx_reference" --probe /path/to/full-forward-probe.json \
  --output "$ltx_evidence/full-original-streamed.json" \
  --max-blocks 48 --stream-block-weights --dequant-workers 8 --threads 8
```

A completed metric report is not a perceptual-quality pass. Streamed forward
timing includes decoding/loading and must not be compared with warm GPU
inference. Coordinate CPU quota/disk as well as GPU use; parallel reference work
can slow GPU command submission. Retain process-control evidence if a reference
is temporarily stopped for exclusive benchmarks; its interrupted elapsed time
is not throughput evidence.

`Ltx25LatentProbe --operation forward` captures the first actual sampler prediction
and stops before its update. Pass its `forward.json` to the same CPU reference
command above. With `input_mode: captured`, the reference verifies every array's
Float32 dtype, shape, byte length, SHA256 and finite values before loading weights.
It uses the captured latent/context arrays, exact position bounds, scalar sigmas,
per-token timesteps and keyframe markers in the unchanged official `Modality`.
Null timestep arrays mean the stream's scalar sigma; captured sampler clean-latent
and denoise-mask metadata are validated but are not applied again inside the
transformer. Ingredients LoRA captures are unsupported and explicitly rejected.
The report records actual process peak RSS separately from retained weight bytes.
A 512×512×145 first pass has 4,864 video tokens; its single Float32 32-head
attention score matrix is about 2.82 GiB. At 1024×1024×145 it is about 45.1 GiB,
before weights or temporary arrays, so full-resolution CPU math attention is not
a suitable reference run on a 48 GB machine. This replay checks the captured
denoiser call, not text encoding, image preprocessing, the sampler trajectory or
rendered anatomy.

To isolate rotary geometry on the captured positions without loading model
weights or a GPU runtime, compile the production geometry file into the small
standalone probe, then compare its tables and intermediate phases with the
unchanged official patchifiers and Float64-frequency RoPE. Supply the capture's
actual FPS explicitly (older forward manifests do not record it):

```sh
dotnet run --project eng/validation/Ltx25RopeProbe -- \
  --capture /path/to/forward.json --output /path/to/rope-native.json
"$ltx_python" eng/validation/ltx25-rope-capture-reference.py \
  --reference "$ltx_reference" --probe /path/to/rope-native.json \
  --fps 24 --output /path/to/rope-original.json
```

The comparison includes all position bounds, frequencies, normalized coordinates,
phases and sine/cosine entries, with per-latent-frame results and Float32 ULP
distances. It does not test the device kernel that applies those tables to Q/K.

Choose `--backend CUDA` for TensorSharp on the requested VM. Connect using:

```sh
ssh root@69.30.85.123 -p 22110 -i ~/.ssh/id_ed25519
```

Copy unchanged original source including `.git` so revision/clean status can be
verified. A remote venv may reuse installed torch with `--system-site-packages`
and install missing einops/safetensors/gguf; record actual package versions.

## Build the unchanged baseline component probe

Build a clean checkout at pinned source/submodule revisions with
`CMAKE_EXPORT_COMPILE_COMMANDS=ON`, `SD_BUILD_EXAMPLES=ON`, and `SD_METAL=ON` on
macOS or `SD_CUDA=ON` on Linux. Preserve user checkouts. The owned C++ probe links
existing unchanged libraries and verifies source/build manifests:

```sh
python3 eng/validation/build-ltx25-sdcpp-denoiser.py \
  --build /path/to/clean/sdcpp-build \
  --source-manifest /path/to/generated/clean-source-export-manifest.json \
  --build-manifest /path/to/generated/verified-build-manifest.json \
  --output artifacts/ltx25/sdcpp-denoiser-probe

artifacts/ltx25/sdcpp-denoiser-probe/ltx25-sdcpp-denoiser \
  --model "$ltx_models/LTX-2.5-Distilled-Q4_K_M.gguf" \
  --backend MTL0 --latent-width 2 --latent-height 2 --latent-frames 2 \
  --audio-tokens 16 --context-tokens 8 --sigma 0.725 \
  --blocks 48 --warmup 1 --repeats 5 --threads 8 \
  --output "$ltx_evidence/sdcpp-full-forward.json"
```

Match geometry, depth, context counts and sigma with TensorSharp's **forward**
probe. Keep loading separate from synchronized warm forwards. Small synthetic
token counts do not predict production-resolution latency/memory. Structural
media validity and component speed cannot establish end-to-end parity, which
requires matched real generation through prompt conditioning and both decoders.

If FFmpeg is unavailable on a host, its media check remains incomplete even when
a clip was produced. A later check of the same hashed media elsewhere is separate
evidence; preserve the original incomplete report and identify where validation
actually ran. Sampled memory telemetry is not an allocator peak measurement.

## Reference-image generation validation

The end-to-end native probe accepts `--image PATH` (first frame),
`--end-image PATH` (last frame), `--reference-images paths.json` (ordered array
of image paths), and `--image-conditions conditions.json` for explicit guides:

```json
[
  { "image_path": "/absolute/first.png", "frame_index": 0, "strength": 1 },
  { "image_path": "/absolute/middle.png", "frame_index": 8, "strength": 0.8 },
  { "image_path": "/absolute/last.png", "frame_index": 16, "strength": 1 }
]
```

For the reported user request, run the full eight-step/48-block pipeline at the
preset geometry on each supported GPU. Use fresh output locations:

```sh
dotnet build eng/validation/Ltx25EndToEndProbe/Ltx25EndToEndProbe.csproj \
  -c Release -o artifacts/ltx25/reference-e2e \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true

MLX_ENABLE_TF32=0 dotnet artifacts/ltx25/reference-e2e/Ltx25EndToEndProbe.dll \
  --model /path/to/models/ltx25/LTX-2.5-Distilled-Q4_K_M.gguf \
  --backend GgmlMetal --width 512 --height 512 --frames 17 --fps 24 \
  --seed 42 --repeats 2 --prompt "请让图中的女人来个飞吻" \
  --image "$PWD/imgs/banner_1.png" \
  --output docs/validation/ltx25/references/local-first.json
```

Deploy the correct owned native libraries/modules beside the probe; skipping
native builds does not make unavailable libraries a passing device test. Use
`--backend CUDA` on the A40 VM. The generated report records source-image
identities, anchor positions, strengths, binaries, progress, raw decoded media,
audio muxing, and first/warm timings. Repeat with last-only, first+last, explicit
interior and three ordered images. Check cancellation, invalid/duplicate positions,
invalid strength, cached prompt reuse, and different images with the same prompt.

The owned `ltx25-sdcpp-generation` C API probe and `ltx25-sdcpp-bench.py`
accept matching `--image` and `--end-image`. They use strength one. Build the
owned probe against the verified clean source/library export as described above;
neither baseline nor ggml files are patched. The pinned SD.cpp LTX implementation
has no C API mapping for arbitrary interior or multiple guiding images. Record
those baseline cases as unavailable, and validate them against the official
conditioning oracle instead. The generation comparison now refuses mismatched
image identities, strengths, and positions.

```sh
python3 eng/validation/ltx25-compare-generation.py \
  --native docs/validation/ltx25/references/local-first.json \
  --baseline docs/validation/ltx25/references/local-sdcpp-first.json \
  --output docs/validation/ltx25/references/local-first-comparison.json
```

First-frame reconstruction metrics, RGB/audio validity, or one prompt's visual
inspection cannot establish broad perceptual parity. Benchmark first generation
and repeated requests separately; image VAE encoding and extra guide-token work
belong in the measured request. Keep unavailable scenarios distinct from failures
and passes. Do not compare whole CLI-process time with codec-excluded C API time.

`ltx25-generation-runner.py` records Linux model-file page residency with a
read-only `mincore` query immediately before and after generation. It does not
evict or warm file pages. Use these observations when comparing first requests
on network-backed storage; warm same-prompt inference and first-request loading
remain separate measurements. Mac file residency is reported unavailable.

`ltx25-generation-suite.py` accepts a JSON array of named argv cases and runs
them serially in one device lane. Each case has `name`, `argv`, and optional
`timeout_seconds`; generated command files, process reports and logs are saved
under a fresh `--output` directory. Failed attempts remain failures.

`ltx25-reference-contact-sheet.py` renders evenly spaced samples from completed
native/baseline raw generation reports for visual inspection:

```sh
python3 eng/validation/ltx25-reference-contact-sheet.py \
  --native docs/validation/ltx25/references/local-first.json \
  --baseline docs/validation/ltx25/references/local-sdcpp-first.json \
  --image imgs/banner_1.png \
  --output docs/validation/ltx25/references/local-first-preview.png
```

The reference row is a Pillow crop preview, and is not an exact preprocessing
oracle. The image never counts as an automated perceptual-quality pass.

`Ltx25TensorAgentProbe` also accepts `--image`, `--end-image`, and
`--reference-images` (path-array JSON). It uploads every image through the real
multipart `/api/upload`, verifies the served upload hash, exercises video
HTTP/SSE, and by default sends a normal chat with the uploaded attachments.
`ltx25-validate-agent-media.py` independently checks the downloaded H264/AAC
streams and PCM sidecar. A chat seed is chosen by the product and must not be
presented as a matched baseline comparison.

For Ingredients identity conditioning, both native and TensorAgent probes accept
`--mode ref --reference-images paths.json`. Use the trained
`--width 768 --height 448 --frames 121 --fps 24` recipe and install the pinned
Ingredients adapter beside the denoiser under `loras/`, or set
`TS_LTX25_INGREDIENTS_LORA`. The report records its hash and source panels. One
image forms a sheet; multiple images become ordered black-background panels.
These inputs condition appearance rather than selecting output frame positions.
The pinned SD.cpp baseline does not implement Ingredients; record its comparison
as unavailable. Test single and multiple inputs through the real model and
TensorAgent separately, inspect subject preservation and the requested motion,
and state whether panels are views of one subject or different subjects.

The TensorAgent probe accepts `--repeats 0 --chat-turn true` for a chat-only run
that still performs real uploads, mode routing, generation and downloaded-media
checks. This avoids generating a second full identity clip solely to exercise
chat. It does not establish prompt-cache reuse.

`Ltx25VideoVaeEncoderProbe --static 1 --frames 121` tests the optimized static
reference encoder with a single RGB frame input. The full temporal `Encode`
implementation remains independently testable. The static shortcut relies on
causal repeat padding and is checked against an independently produced genuine
121-frame official fixture; it is not a substitution of repeated pixels for an
unvalidated temporal encoder. Keep component oracle evidence separate from full
Ingredients generation and perceptual assessment.

`ltx25-compare-static-video-vae.py` checks a completed static probe against the
independently validated official one-frame latent, with a separate genuine
full121 official temporal-invariance proof prerequisite. It verifies model/input
and latent checksums, channel-major temporal layout, finiteness, numerical gates
and exact static slice equality. Its report explicitly excludes an official
full-resolution121-frame execution when that execution was interrupted.

`ltx25-media-contact-sheet.py` fully decodes the exported video stream and saves
17 sampled frames with uploaded panel previews. It supports identity sheets whose
output composition can change. It records media/source/tool hashes and expected
frame geometry; previews are not exact VAE preprocessing, and audio validation
and semantic conclusions are separate checks.

`ltx25-validate-native-media.py` independently checks a completed native probe's
full H264/AAC streams, exact video geometry/count and finite Float32 stereo PCM
sidecars. Like the Agent validator, it supports an explicit `--media-root` for
unchanged copied media. Preserve source and destination hashes when validating
media on another machine; record which host supplied FFmpeg/ffprobe.

`ltx25-compare-native-generations.py` compares matched before/after native reports
and raw RGB with verified report/output hashes, settings, reference identities,
strengths, positions, prompt, seed and adapter. It reports descriptive maximum
error, RMSE, relative L2, PSNR and generation latency; none is a perceptual pass.
The surrounding model manifest must establish unchanged checkpoint bytes because
equal model paths alone cannot establish model identity. End-to-end native reports
also capture existing CUDA attention counters/workspace through read-only probe
reflection, proving that a measured graph actually dispatched the optimized route.

The real HTTP/SSE 97-frame first-image regression also uses the detailed Chinese
motion prompt in `server-detailed-chinese-prompt.txt`: fingertips touch pursed
lips, an open palm extends toward the camera, then the arm lowers. With the user
banner, 512×512, 24 fps, seed 42 and eight Euler steps/CFG 1, the inspected Apple
Silicon sample shows that sequence and preserves source appearance; the actual
request took 263.93 seconds including H.264/audio media production. The original
short Chinese prompt did not clearly complete the gesture in matched 97-frame
SD.cpp/native samples. This is one verified prompt/seed/subject, not a broad
semantic-quality pass.

Multiple Ingredients views can leak the sheet's panel layout into output. The
fixed failing Apple seed retained two panels after a position-aware caption was
introduced, and again after both inputs were changed to clean text-free bust and
face crops. These failures are retained independently of playable-media checks.
A single reference image gives more predictable composition in tested samples;
clean images and panel descriptions are publisher guidance, not a verified
universal mitigation.

After every managed build/publish, use `ltx25-deploy-cuda-modules.py` to install
explicitly verified, device-compatible PTX into each test/probe output before
running `dotnet test --no-build` or the generation suite. A forced managed rebuild
can recopy repository PTX for a newer architecture over a validated deployment.
The tool checks source SHA256 and `.target` before any copy, checks all deployed
bytes afterward, and writes a receipt under the ignored evidence directory.
For the recorded A40 modules:

```bash
python3 eng/validation/ltx25-deploy-cuda-modules.py \
  --source artifacts/ltx25/end-to-end-parallel-final/cuda_kernels \
  --destination InferenceWeb.Tests/bin/Release/net10.0/cuda_kernels \
  --destination artifacts/ltx25/reference-cold-source-e2e-cuda-final/cuda_kernels \
  --module tensorsharp_kernels.ptx:b0178aab4c57a7f31109373afae4e324a4d25f2d0d5ee9fdd902ca10ef8b0f22:sm_86 \
  --module tensorsharp_ltx25.ptx:2e44927a4c6c59cb059d200c51d0990e1277ae1c716ed90a2bdb642ede9845d3:sm_61 \
  --output docs/validation/ltx25/references/deployed-cuda-modules.json
```

Use `--verify-only` to assert a deployment immediately before a later test or
benchmark without changing files. The smaller LTX module targets `sm_61` and the
main module targets `sm_86`; both were executed on the A40. This receipt covers
deployment bytes, while source manifests and numerical tests cover correctness.
Do not rebuild between that receipt and `--no-build` execution.

`ltx25-run-cuda-validation.py --config <explicit.json> --output <ignored-dir>`
combines the required build/deployment order with strict tests and a serial case
manifest. Its JSON supplies `module_source`, pinned `modules`, `probe_output`,
`test_filter`, `environment` and `cases_json`. It forces managed rebuilds, publishes
without rebuilding, installs and verifies PTX in both outputs, runs tests with
`--no-build`, verifies PTX again, then launches the serial generation suite.
Every argv, exit status, test log and module receipt is retained. Use
`ltx25-compare-native-generations.py --require-bitwise` to additionally require
exact RGB and both raw PCM channels for data-movement changes; each side's actual
checkpoint manifest must still prove model identity.

The source-aware CUDA uploader remains an explicit opt-in:
`TS_LTX25_CUDA_SOURCE_UPLOAD=1`; unset/`0` preserves the established mapped path.
The first candidate's single-threaded source reads regressed fresh generation
79.85→95.91 seconds despite exact outputs. Matching the existing eight-worker
1 MiB read slices within each pinned slot improved a candidate-first/control-
second pair 77.015→70.507 seconds (8.45%), with exact RGB, both PCM channels and
complete H.264/AAC checks. It reserves two 8 MiB pinned buffers per used component
and runs only for cold Linux weight ranges; device-resident weights keep the
existing cache. This one mixed-residency A40/storage pair does not establish a
universal speedup, and 70.507 seconds still trails the recorded SD.cpp fresh
56.67 seconds. The default is unchanged; no further loader trials were used.

### Rectangular attention experiment: rejected after full generation

`Ltx25AttentionProbe` compares the existing streaming attention path with the
existing pedantic Float32 batched attention session for representative video/text
and audio/video shapes. It uses deterministic inputs, separate warmups, alternating
synchronized samples, CUDA-event and wall timings, output hashes, and maximum
error/RMSE/relative L2 checks. Both measured paths allocate their output; input
upload, first initialization, projection, sampling, decoding, and export are
excluded. Build without rebuilding upstream native dependencies:

```bash
dotnet build eng/validation/Ltx25AttentionProbe -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
# Deploy and verify device-compatible PTX as described above before execution.
dotnet eng/validation/Ltx25AttentionProbe/bin/Release/net10.0/Ltx25AttentionProbe.dll \
  --warmup 2 --repeats 7 --output docs/validation/ltx25/attention-component.json
```

An experiment extended the batched dispatch to additional rectangular attention
shapes. It retained full dense Float32 attention but changed reduction order.
Component numerical tests passed, yet matched remote end-to-end generations
failed the RGB and stereo PCM equivalence checks. Disabling only this extension
restored bitwise RGB and both PCM channels in all three tested cases while device
prompt math and fused head gating remained enabled. The extension and its selector
were removed; production attention routing is unchanged. The probe remains a
reusable investigation tool. Its component tolerances and timings do not establish
full-generation parity, perceptual quality, or an accepted model speedup.
