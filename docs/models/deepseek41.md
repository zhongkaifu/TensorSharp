# DeepSeek V4.1 Flash (`deepseek41`)

> **Multi-GPU selection:** use `--layer-split N` for whole-layer placement or a supported `--tp N` tensor-parallel mode. With neither mode configured, the default is one device. Older commands and measurements below predate that default: migrate multi-GPU launches by adding `--layer-split N`. Layer split is single-node only.

[← back to model index](README.md) | [中文](deepseek41_zh-cn.md)

TensorSharp has a dedicated **V4.1 inference graph on `ggml_cuda`**, with an
optional native vision encoder. It uses the DeepSeek whole-model loader and
scheduler, with V4.1-specific attention, Engram, residual connections, and chat handling. This card describes
the implemented path and its limits. Model-quality and performance claims need
the measured artifacts tracked in the [validation report](../deepseek41_validation.md).
The same graph also loads on `--backend ggml_cpu`, which exists to run and check
the architecture without a GPU rather than to serve it — see
[Running on the ggml CPU backend](#running-on-the-ggml-cpu-backend). A second
GPU-free option carries no ggml and no native library at all: `--backend cpu`
runs V4.1 on the pure-C# `DeepSeek4CpuExecutor`, which implements this same
graph in managed code and is held to the PyTorch oracle. Both are correctness
and portability paths, not serving paths — see [Backends](#backends).

The [official model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
declares `DeepseekV41ForCausalLM`. Its text network has 40 layers, hidden size
5120, 64 query heads with 512 components, 384 routed experts with top-6 routing,
one shared expert, and a 2304-wide expert intermediate. It declares a
1,048,576-token context. V4.1 differs from V4 in ways that affect every forward
pass; changing the GGUF architecture name to `deepseek4` is invalid.

## Download the repaired Q2_K/Q5_K checkpoint

For new downloads, use `Q2_K-Q5/` from
[smalinin/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/tree/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5),
pinned to `d1de55c19f95172c882906cc83c0e55932d26a63`. Keep all ten shards
together and give TensorSharp the first shard. The files total **335,382,014,624
bytes (312.349 GiB)**: mixed Q2_K/Q3_K backbone and experts, Q5_K Engram tables,
80 F32 mHC matrices and four BF16 Engram gates.

The original seven-shard `vcruz305` Q2_K artifact quantized those 84 sensitive
tensors to Q2_K. It is no longer recommended for quality validation; the
[publisher's repair report](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/blob/2c525d63b9ba5319185c93637f00d70fea55b44f/Q2_K/Q2_REPAIR_REPORT.md)
explains the affected tensors. Adding Engram metadata to an old shard does not
repair its tensor precision.

The repaired package's headers have been checked for all 1,046 tensor names and
shapes, the sensitive tensor types, and matching tokenizer/Engram metadata.
All ten downloaded shards passed full-file SHA-256 verification. Bounded plain
and DSpark HTTP probes passed with both `--layer-split 2` and experimental
routed-expert TP (`--tp 2`) on `ggml_cuda`. Each of the four
processes passed three text checks and one image OCR/color check through EOS,
then shut down cleanly. Separate plain/DSpark text and image pairs matched all 24
token IDs and `max_tokens` finishes in both modes, with active DSpark and clean
exit. These bounded continuations are separate from the HTTP EOS checks. Image
DSpark was slower in both measured pairs. Short repeated warm text controls
(one warmup and three measured 69-prompt/24-output-token pairs per mode) also
completed with exact token IDs and finishes, active DSpark, and clean exit.
**These small-sample, paging-constrained checks do not establish broad quality or
performance qualification, a speedup, multi-node execution, or full-model TP.**
The historical measurements below do not qualify this quantization. Local evidence
is under `docs/validation/model-matrix-20260927/deepseek41/`
(not committed). Verify the hashes below for each new download.

**The GGUF already includes Engram.** TensorSharp reads its token map, hash
multipliers, bucket primes and offsets, and padding ID directly from the GGUF
metadata, alongside the learned Engram weight tensors. No Engram generation,
separate Engram file, or tokenizer/config download is needed for text inference.

The loader validates the embedded layout against the Engram tensor dimensions
and rejects missing or malformed constants. There is no legacy file fallback or
Engram path override. Engram table placement and page warming still apply to
the learned weights; they do not generate hash constants.

From the repository root, on the machine storing the model:

```bash
python3 -m venv /workspace/dsv41-tools
/workspace/dsv41-tools/bin/python -m pip install huggingface_hub
/workspace/dsv41-tools/bin/hf download smalinin/DeepSeek-V4.1-Flash-GGUF \
  --revision d1de55c19f95172c882906cc83c0e55932d26a63 \
  --include "Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-*.gguf" \
  --local-dir /workspace/models/deepseek41-q2-q5
```

`hf download` preserves the `Q2_K-Q5/` subdirectory. Verify every complete file
before inference; the SHA-256 values below are the pinned publisher LFS identities.
Run this outside timed benchmarks, since it reads all 312.349 GiB:

```bash
(cd /workspace/models/deepseek41-q2-q5/Q2_K-Q5 && sha256sum --check - <<'SHA256'
8126b49dfcfde02cb3db24b6f98d56b031f0a34eaca2509fc7b2b9d362cf0ac8  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf
22bb293aee509a348ce32a739e006fa41f2348c6bcfafa3be76a3ee079eadf96  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00002-of-00010.gguf
db894848b4f14d42c39e18faa907c737cd4850f2fcb9deff9d5ebd86f580aaf7  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00003-of-00010.gguf
655a3400f2c092d6e3c11b8b18bf319b29563e59b960337115266574321953e3  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00004-of-00010.gguf
4b9378c6819d1130517e8719026b34e5257f1f73bd1d50cf6f30243982100bac  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00005-of-00010.gguf
04f161084d82032c65247c02e6169784a757be7db9e068b0baba6833125b6bb8  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00006-of-00010.gguf
b1b1bf3cfbbc7388ce49b5ced69c42a13c7c5c5d3d609ac86902897a08d3c88a  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00007-of-00010.gguf
fae7f35123ae3557034a541507bb9fc24fb62c2e16ff8441c32e8e0477743d5d  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00008-of-00010.gguf
781fd69e9dd17c09676865830523a2d077a97544d6a004429aab95d2570f8534  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00009-of-00010.gguf
fa5affb1f971cd6e7effad5684b73780472b486a46330cd7c5776f87c38fea97  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00010-of-00010.gguf
SHA256
)
```

Stop if any hash fails. `eng/dsv41-verify-download.py` is a historical verifier
for the original seven-shard layout; it does not verify this ten-shard package.
The two Q5_K Engram tables total about 125.889 GiB. They cannot stay resident in
a 57.74 GiB RAM allowance, even before CPU experts; use
`TS_DSV41_ENGRAM_DEVICE=0 TS_DSV41_ENGRAM_WARM=0` on such a host and report
paging costs. Choose CPU expert offload from actual device capacity, including
the optional drafter and vision encoder.

**Historical benchmark provenance:** earlier results in this card and the complete-file SHA-256 record
`docs/validation/deepseek41/checkpoint-sha256.json` (local validation evidence, not committed)
refer to revision `8e0c4de3cb6519bfc11ed69dc87184b457a57bb5` and its older
first shard in the original `vcruz305` release. Later seven-shard examples used
`58d8ac86298fdf85a2440defee08b1abcad32e45`; the old Q4_K_M placement evidence is
also separate. Retained historical launch commands identify those files, not
the repaired package. Do not attribute their hashes, 246.35 GiB size, approximately
60 GiB Engram warming, quality results or throughput to Q2_K-Q5. Record the
revision and all shard hashes with every new validation result.

## Prepare the optional vision companion

The supplied GGUF conversion omits the vision tower, aligner, learned image
delimiters and per-layer visual routing biases. The official release places
the approximately 970 MB vision/aligner weights in an isolated shard, so these
can be prepared without downloading the original text weights:

```bash
/workspace/dsv41-tools/bin/python -m pip install numpy==2.0.2 gguf
/workspace/dsv41-tools/bin/python eng/dsv41-prepare-vision.py \
  /workspace/models/deepseek41-q2-q5/Q2_K-Q5 \
  --parent-model /workspace/models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
  --repository deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277
```

This creates `deepseek41.vision.gguf` and `deepseek41.vision.json` beside the text
shards. The `--parent-model` argument supplies the GGUF tokenizer fingerprint;
existing companions remain compatible when the tokenizer matches.
The preparer downloads the isolated vision shard and small byte ranges for
the delimiter/router tensors and preserves BF16/F32 storage. It verifies the
complete isolated vision shard against its LFS SHA-256 and records individual
range hashes for delimiter/router weights; it does not download or verify the
entire original text shards. The companion provenance record
`docs/validation/deepseek41/vision-companion.json` (local validation evidence, not committed)
contains all 306 tensors, source revisions, ranges, and the output digest.
The native loader checks the parent tokenizer fingerprint and model
dimensions before attachment. To enable images, add
`--mmproj /workspace/models/deepseek41-q2-q5/Q2_K-Q5/deepseek41.vision.gguf` to the server
command below. Image capability remains disabled without an attached companion.

Vision uses dense F32 attention by default, matching the official tower's
attention arithmetic. On Ampere-or-newer NVIDIA CUDA, BF16 matrix inputs retain F32
accumulation and output until bias addition and BF16 activation rounding.
`TS_DSV41_VISION_BF16_GEMM=0` selects the diagnostic F32-promoted matrix path.
`TS_DSV41_VISION_FA=1` selects faster flash attention with F16 intermediates;
the real-image reference comparison showed larger feature differences on
that path. These options affect the image encoder, independently of text
attention. Exact bounds and measured tradeoffs are in the
[validation report](../deepseek41_validation.md#independent-vision-and-mixed-modality-reference).

For numerical investigations, `TS_DSV41_VISION_TRACE_DIR=/absolute/directory`
writes F32 patch, block, norm, and projector outputs. Tracing retains
intermediate tensors and adds device transfers, so leave it unset for normal
inference and benchmarks.

The encoder uses 32 bidirectional transformer layers, two-dimensional rotary
positions, a padded 3-by-3 spatial merge, and a two-layer projector. It emits
the complete image span, including learned start/end and per-row newline
embeddings. Image tokens use the visual MoE bias, suppress Engram injection,
and break Engram n-grams across image spans. Mixed image/text spans can cross
prefill microbatch boundaries. Existing video frame extraction uses the same
image path. CPU/CUDA numerical fixtures and exact preprocessing checks pass.
The complete-checkpoint CUDA layer-split run passed 25 image/video requests
across concurrency 1/4 and a separate image request after long text. The final
routed-TP profile also passed all 25 image/video requests across concurrency 1/4.
Strict encoder parity is tracked separately.
The official configuration does not provide an audio decoder. V4.1 rejects
audio-bearing requests, including mixed image/audio requests, instead of
ignoring the audio. Chat Completions, Responses and Web UI return HTTP 400.
The OpenAI parsers reject audio parts before reading their payloads, including
missing or malformed audio, so a later audio attachment cannot leave earlier
image uploads behind.

The OpenAI chat endpoint accepts `image_url` parts containing base64 image
data URIs, including multiple images and images in earlier turns. Its V4.1
`video_url` extension samples a base64 MP4, WebM, or MOV through the existing
video decoder. For example, a message's content array can contain:

```json
{"type":"video_url","video_url":{"url":"data:video/mp4;base64,...","fps":1,"max_frames":3}}
```

Each sampled frame becomes an image span with its source time in seconds,
computed from frame index divided by the probed frame rate. Time labels are
approximate for variable-frame-rate clips.
Frames remain in source order; this is frame sampling rather than a native
temporal encoder. `fps` must be greater than zero and at most 60, and
`max_frames` must be 1–64. The defaults use `VIDEO_SAMPLE_FPS` and a positive
`VIDEO_MAX_FRAMES`, otherwise 1 fps and 16 frames. Frames above the cap are
sampled across the clip. Remote HTTP image/video URLs are not fetched; send
data URIs. For V4.1, both `/v1/chat/completions` (`image_url`) and
`/v1/responses` (`input_image`) reject remote or malformed image URLs and
invalid/empty image base64 with HTTP 400 before streaming or writing any images
from that request.
Valid image data URIs retain their existing decoding and upload path. The text
context budget still applies after expanding every image.
The final routed-TP host (managed stage 3,651, native `6b3b5ab3…`) passed all
eight image rejection checks
`docs/validation/deepseek41/full-checkpoint/tp8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-image-input-rejections.json`
and four Responses audio rejection checks
`docs/validation/deepseek41/full-checkpoint/tp8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-audio-input-rejections.json`
(local validation evidence, not committed).
Streaming and non-streaming requests returned JSON 400 without SSE for remote
image URLs, malformed image base64, and valid-shaped or malformed audio parts.
The earlier CPU-offload host's eight image checks remain preserved separately.

## Prepare the optional DSpark companion

V4.1 requires a `deepseek41-dspark` artifact; V4 drafters are incompatible.
The official revision below isolates DSpark in shards 44–46 (7,933,129,808 bytes),
so the original text model need not be downloaded:

```bash
/workspace/dsv41-tools/bin/hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277 \
  --include config.json model.safetensors.index.json \
    model-00044-of-00048.safetensors model-00045-of-00048.safetensors \
    model-00046-of-00048.safetensors \
  --local-dir /workspace/models/deepseek41-source/DeepSeek-V4.1-Flash

(cd /workspace/models/deepseek41-source/DeepSeek-V4.1-Flash && sha256sum --check - <<'SHA256'
8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879  config.json
74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8  model.safetensors.index.json
9a6b39fb88a2510487a8efaef77aa7864e8061f6b62c95a0f010e9dd538f3b05  model-00044-of-00048.safetensors
0cc9d5f6ca3a2158ccc63ce2c70c76aeda8177d54913340481af566680329eb5  model-00045-of-00048.safetensors
e625902027b9d23d416f8818c665fab4704e0b96dc1bc778321601b700475a9d  model-00046-of-00048.safetensors
SHA256
)
```

Only after all five checks pass, convert the companion:

```bash
/workspace/dsv41-tools/bin/python -m pip install numpy==2.0.2
mkdir -p /workspace/models/deepseek41-dspark
/workspace/dsv41-tools/bin/python eng/dsv4-dspark-to-gguf.py \
  --checkpoint /workspace/models/deepseek41-source/DeepSeek-V4.1-Flash \
  --expert-type mxfp4 \
  --out /workspace/models/deepseek41-dspark/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf
```

The audited conversion has 81 tensors, including all three visual routing biases,
and occupies 7,940,628,416 bytes. Source identity and conversion checks are
complete. Initial trained DSpark text/image HTTP probes passed with two-GPU
layer split and experimental routed-expert TP on `ggml_cuda`, including complete
image answers and clean shutdown. Broad quality and throughput remain unqualified
under heavy paging. Separate 24-token text/image pairs matched plain token IDs
and `max_tokens` finishes in both modes with active DSpark and clean exit; these
are bounded continuations. No speedup, multi-node execution, or full-model TP is
claimed.
Add `--draft-model /workspace/models/deepseek41-dspark/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf --spec`
to a launch using the repaired ten-shard text checkpoint. Keep the same loaded
companion and use `--no-spec` for a plain-decode comparison. The head occupies
the output-head GPU and affects placement; vision needs additional space.

## Run the implemented path

Install the .NET 10 SDK, CMake, a C++ compiler, and the CUDA toolkit with `nvcc`
on `PATH`. Build from the repository root. This example uses A40 CUDA architecture
8.6; adjust both architecture values for other GPUs. It references the repaired
download, whose capacity and inference behavior must be checked on your hardware:

```bash
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON \
  TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES=86 \
  bash TensorSharp.GGML.Native/build-linux.sh
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release \
  -p:CudaArch=compute_86 -p:TensorSharpSkipGgmlNative=true

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_CONTEXT=65536 \
  TS_CPU_MOE_THREADS=32 TS_DSV4_UBATCH=256 \
  TS_DSV41_ENGRAM_WARM=0 \
  TS_DSV41_COMPACT_RAW_GATHER=0 KV_CACHE_DTYPE=f16 \
  TS_SCHED_MAX_RUNNING_SEQS=4 TS_SCHED_MAX_BATCHED_TOKENS=4096 \
  TS_SCHED_PREFILL_CHUNK=256 TS_SCHED_SOLO_PREFILL_CHUNK=8192 \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /workspace/models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
  --backend ggml_cuda --layer-split 8 --port 5000
```

The host build copies the native library beside the server DLL. This launch
uses explicit microbatch and scheduler settings. It is a launch example, not a
measured repaired-checkpoint profile. The historical profiles in the validation
report used different weights and sometimes different settings.
Sparse prefill attention needs no flag: it is the default on this path, and
`TS_DSV41_SPARSE_FA=0` turns it off. `TS_DSV4_UBATCH=256` pins the example's width;
leave it unset to let the loader choose (see
[Backends](#backends)). Choose `TS_CPU_MOE_THREADS` for the available CPU quota
and record it for each run.
Set it in the launch environment, including for GPU-only placements: native
CPU graph work and host reduction can still affect latency. The current CLI
also accepts `--cpu-moe-threads N`; use the same value if supplying both, since
the positive environment value takes precedence in the native loader.

**Historical reproduction only:** the following eight-A40 launch preserves the
original seven-shard checkpoint paths and measured settings. It is not the
recommended new download or a measurement of Q2_K-Q5:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_CONTEXT=65536 \
  TS_DSV4_UBATCH=1024 KV_CACHE_DTYPE=f16 TS_CPU_MOE_THREADS=32 \
  TS_DSV41_SPARSE_FA=1 TS_DSV41_COMPACT_RAW_GATHER=1 \
  TS_DSV41_ENGRAM_WARM=1 TS_DSV41_ENGRAM_THREADS=16 TS_DSV4_PERF=1 \
  TS_SCHED_MAX_RUNNING_SEQS=4 TS_SCHED_MAX_BATCHED_TOKENS=4096 \
  TS_SCHED_PREFILL_CHUNK=1024 TS_SCHED_SOLO_PREFILL_CHUNK=1024 \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /workspace/models/deepseek41-q2/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
  --mmproj /workspace/models/deepseek41-q2/deepseek41.vision.gguf \
  --backend ggml_cuda --layer-split 8 --n-cpu-moe 0 --cpu-moe-threads 32 \
  --host 127.0.0.1 --port 5000 --max-tokens 2048
```

The measured launch record
`docs/validation/deepseek41/full-checkpoint/layer8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-launch.json` (local validation evidence, not committed)
contains the original VM paths, binary hashes and environment. This profile
passed 138/138 inference cases; its throughput and limits are recorded below.
Compact gathering remains opt-in, with its documented floating-point
differences. That record's `TS_DSV41_SPARSE_FA=1` selected ggml's
mask-compacted flash-attention kernel: its results were recorded with the initial
V4.1 support (`3347b06b`), before TensorSharp's owned F32 attention existed,
whose sparse prefill is now the default (see [Backends](#backends)). Its
19.985/80.240 s long-prompt times are therefore not a measurement of the current
attention path. Startup and page warming are excluded from the inference
measurements.

Without an explicit thread setting, a GPU-only V4.1 load uses the caller's
`TS_DSV4_THREADS` value, defaulting to at most 32; CPU expert offload instead
uses the detected available CPU parallelism. Earlier native builds did not
receive the CLI thread override. In particular, the first measured TP run's
`--cpu-moe-threads 48` did **not** configure 48 native threads: its native pool
is inferred to have used the default 32 from the loader source, absent native
environment override, and the VM override probe. That original host did not
expose a pool-width getter, so 32 was not measured directly there. Keep that
baseline distinct from later explicit-thread experiments.

`--layer-split 8` requests **eight GPUs using layer split**. The
startup diagnostic states the placement mode. TensorSharp distributes whole
layers according to available VRAM by default. Set `CUDA_VISIBLE_DEVICES` to the exact
devices intended for the run.

`--tp 8 --backend ggml_cuda` enables **routed-MoE tensor parallelism** on eight
GPUs; `--tp` accepts every degree from `2` through `8`, including `3`, `5`, and
`6`. The degree reaches the native loader as a per-model argument and must equal
the number of GPUs it selects. `--tp` and `--layer-split` cannot be combined.

In this mode, routed-expert gate/up matrices are partitioned along the FFN
intermediate dimension, and down matrices along their output rows. Each rank
receives the full SwiGLU activation before its down projection; the disjoint
output rows are then gathered without summing partial dot products. This keeps
the down projection's reduction order, avoiding rounding changes that later
activation quantization can amplify. Quantization blocks are never split, and
uneven strips rotate across layers to balance storage. CUDA uses reusable pinned
buffers for input/output transfers. Attention,
shared experts, and caches retain their layer placement. This is a partial
tensor-parallel implementation; it does not shard attention or enable
distributed tensor-parallel groups. Host transfers can limit throughput, so
this option does not establish a speedup over layer split. The qualified full-checkpoint
run below was slower than layer split; see also
the [measured placement profiles](../deepseek41_validation.md#full-checkpoint-routed-moe-tp).

Before uploading a TP layer, its three routed-expert source ranges are read
sequentially into the page cache with the existing `TS_DSV4_LOAD_THREADS` policy
(default 16, 64 MiB scratch per reader). Already resident ranges are skipped.
Only the current layer is prepared, avoiding small, strided network-file page
faults during rank uploads. `TS_DSV4_WARM_PREAD=0` keeps direct mapped reads for
this stage. This preparation does not warm the complete checkpoint.

A September 29, 2026 same-build TP6 comparison on the six-A40 VM used
`TS_DSV4_WARM_PREAD=0,1,1,0` in model-start order. Load times excluding kernel
warmup were **546.26, 149.61, 157.35, and 510.88 seconds**: disabled/enabled
medians **528.57/153.48 s**, a descriptive **3.444×** ratio. Each launch required
zero client-kernel residency across 183.25 GiB of complete pages in all 120
routed-expert tensors. Settings, checkpoint, source, native library and managed
runtime stayed fixed; all 129,280 first-prefill logits and the one-token greedy
check matched exactly, and every process exited cleanly. The fourth cache
precondition initially left 135 pages resident and prevented launch. A separately
recorded continuation passed the same zero-page gate on its first attempt and
supplied the fourth observation; the failed attempt remains preserved. This is
an interrupted comparison with two starts per setting, not an uninterrupted
ABBA trial. Partial boundary pages and MooseFS userspace/network/server caches
were uncontrolled, so the ratio is not a cold-storage or universal speedup.

When NCCL selects `NCCL_P2P_DISABLE=1`, batches of at most 16 tokens use the
pinned host activation gather; larger batches use the private F32 NCCL gather.
This threshold follows paired measurements on six PCIe A40s and is not applied
to P2P-enabled configurations. `TS_DSV41_TP_HOST_TOKENS=0` forces the available
device gather for comparison; values from 0 through 4096 set the host threshold.
`TS_GGML_TP_F32_NCCL=0` selects the host fallback for every batch. Both transports
preserve F32 bits. Device gathering remains ordered on the rank CUDA streams,
without a separate gate/up completion fence; errors drain every rank before
returning control to the caller. Full-model throughput must still be measured
for the chosen placement and transport.

The September 29, 2026 UTC check used six PCIe A40s, the repaired ten-shard
EngramQ5/Q2_K checkpoint, context 4096, F16 KV, host Engram tables with warming
disabled, and `NCCL_P2P_DISABLE=1`. Each placement had one process start and five
identical fixed-input rows: 512 prefill tokens and 128 decode tokens. Candidate native
`473ee64d…` passed the runtime-file integrity checks; the upstream ggml checkout
remained unchanged at `353b63b4…`. All three runs produced the same complete
128-token untimed greedy chain as the original layer-split baseline.

| Placement | First row prefill / decode, tok/s | Median rows 2–5 prefill / decode, tok/s | All five prefill rows, tok/s range | All five decode rows, tok/s range |
|---|---:|---:|---:|---:|
| Original `--layer-split 6` | 53.3 / 13.1 | 363.05 / 30.20 | 53.3–502.6 | 13.1–30.3 |
| `--tp 6`, adaptive host threshold 16 | 19.7 / 9.8 | 209.75 / 19.95 | 19.7–280.7 | 9.8–26.2 |
| `--tp 6`, forced device gather, threshold 0 | 37.7 / 10.3 | 196.75 / 18.20 | 37.7–302.0 | 10.3–22.8 |

Use `--layer-split 6` for throughput on this VM and workload. The adaptive
transport's decode median was 1.096× the forced-device median, but the row
variation and single start per configuration do not establish a stable or
general speedup. Both TP runs stayed at a sampled 1740 MHz SM clock and P0;
later rows had no measured major faults. The remaining timing variation was
not attributed to a specific cause.

During these timing rows, container memory was about 255 GiB for TP versus
70 GiB for layer split. File cache accounted for about 252 versus 68 GiB;
anonymous memory was about 1.6–1.7 versus 0.8–0.9 GiB. TP upload scratch is
bounded, but uploaded routed-weight pages remain in reclaimable file cache:
`TS_DSV4_LOAD_DROP_CACHE` currently applies to ordinary layer uploads, not
the private TP uploader. GPU capacity alone therefore does not establish that
a host-memory limit is sufficient; the observed 255 GiB is not a minimum RAM
requirement. These sequential loads had different
source-page residency and are not a controlled loading-latency comparison.

The current output-row implementation passed numerical fixtures on two and six
A40s, including F32, BF16, F16, Q2_K, Q3_K, Q4_K and Q6_K on two GPUs and real
repaired-checkpoint expert weights on six. CPU fixtures cover degrees 2 through
8; seven- and eight-GPU runs were unavailable on the six-GPU validation VM.
The final `473ee64d…` runtime passed six-GPU HTTP qualification with the repaired
ten-shard Q2_K/Q5 checkpoint: 16 text responses matched the unchanged layer-split
baseline exactly, all four strict tool cases and three image cases passed, and
all 129,280 first-prefill logits were bitwise identical. DSpark and ngram each
preserved all 96 greedy tokens and the finish reason on text and image inputs.
Recorded drafted/accepted/verify counts were 49/49/10 for each ngram scenario,
83/73/19 for text DSpark and 40/22/13 for image DSpark; DSpark exercised seven
and nine rollbacks respectively. Both HTTP and speculation exited cleanly with
no runtime-file changes. These counters establish active speculative coverage,
not a speculative throughput improvement.
These checks do not establish that the full checkpoint fits on two or four A40s.
When the historical seven-shard Q2_K Engram tables use host mappings, synchronous warming consumes
approximately 60 GiB of host page cache before readiness. GPU-resident tables
skip this warm. Record cold-load and warming time separately from warm throughput.

On CUDA, quantized gate/up and down output strips run through TensorSharp's owned
quantized strip kernel (`ggml_ops_matmul_quant_strip.cuh`,
`tsg_matmul_id_quant_pair`): it reads only the rank's weight strip but keeps
the unsplit launch's stream-k partitions and reduction order, so each strip's
gate/up rows equal the full tensor's bit for bit. Before it, ggml's batched
MMQ path grouped a strip's F32 sums differently from the unsplit launch and
the down projection's Q8 activation requantization amplified that into a
checkpoint-shaped Q2_K/Q3_K failure at 16 tokens (relative L2 `3.9e-5`
against the `1e-5` full-weight tolerance). `GgmlOpsDsv41TpTest` keeps the
strict full-weight reference and its original tolerances as the pass
criterion, records the same-device partitioned evaluation beside it, and
`--cuda 1 --quant-strip-only` checks bitwise gate/up equality plus scratch
growth/failure recovery. Nonaligned strip shapes stay on ggml's route. The
recorded 15-36% narrow-strip MoE slowdown describes the earlier split-down
implementation, not the current two-gather
implementation; see
`docs/validation/qualification-2026-09-16/numerical-tp-chosen-r1/README.md` (local validation evidence, not committed).

On `ggml_cuda`, an unspecified CPU offload policy now asks the native capacity
planner for the fewest leading host expert layers that fit the visible devices,
context and workspace; a fitting model keeps every expert on the GPU. Use
`--n-cpu-moe 0` to require full GPU residency, a positive `--n-cpu-moe N` to
choose the count, or `--cpu-moe` for all routed experts. Other backends retain
explicit offload. Attention, routing, and the shared expert remain on the GPU.
This load-time plan is not a unified request budget: host mappings can exceed
the RAM allowance and incur paging, as the loader reports.
[Engram table placement](#where-the-engram-tables-live) is selected separately;
when the tables use host mappings, only selected embedding rows are read and
transferred for each input batch. CPU MoE offload and layer split are implemented,
but their throughput must be measured for the chosen hardware
and context. When combined with `--tp N`, CPU-offloaded leading layers
retain whole CPU experts; the remaining layers use the routed-expert shards.

Native `6b3b5ab3…` explicitly assigns shared gate/up/down projections to the
layer device. This corrects earlier scheduler placement that could send shared
gate/up work to CPU after a CPU-offloaded or TP routed branch. The
placement and numerical checks
(`docs/validation/deepseek41/shared-expert-placement/README.md`, local validation
evidence, not committed) passed 597/597 on two GPUs. Earlier full-checkpoint placement benchmarks retain
their original binaries and results. Final CPU-offload, routed-TP and layer-split
profiles using the corrected placement have completed. See the placement
records in `docs/validation/deepseek41/final-placements/README.md` (local validation evidence, not committed).

For a warm CPU-offload benchmark, read the offloaded expert pages separately
after model loading and before starting the timed requests:

```bash
/workspace/dsv41-tools/bin/python eng/dsv41-warm-experts.py \
  /workspace/models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
  --layers 4 \
  --report cpu-expert-warming.json
```

This requires the `gguf` Python package and reads all three routed matrices
in each selected layer. Record its I/O time separately. Pages remain evictable;
Engram warming alone does not warm CPU expert weights.

### Where the Engram tables live

By default the two Engram tables are **GPU-resident**: each is loaded onto the
device that owns its layer, and the graph gathers rows with `get_rows` over the
quantized table. The host never reads a row, so nothing is dequantized on the
CPU and only the row ids cross the link. Startup prints
`Engram lookup: GPU-resident tables, gathered in-graph`.

This is possible because TensorSharp keeps the tables in the checkpoint's
quantization. A Q2_K row of 256 values is 84 bytes, so both 384M-row tables
together are 60.1 GiB. vLLM and SGLang store the same rows as FP8 values plus
per-32 block scales, 264 bytes a row, which is why their default is a host
table with an FP8 gather kernel.

Placement is automatic and conservative: the tables are priced into the
layer-split packing, and if putting them on GPUs would force any routed-expert
CPU offload, they stay host mappings instead and startup says so. Set
`TS_DSV41_ENGRAM_DEVICE=0` to force host mappings or `=1` to require GPU
residency and fail if it does not fit. `TS_DSV41_ENGRAM_THREADS`,
`TS_DSV41_ENGRAM_WARM` and `TS_DSV41_ENGRAM_RANDOM` only affect host mappings;
startup notes when one is set on the GPU-resident path.

Measured on eight A40s with a cold prompt each time, so every prompt selects
rows it has not touched before:

| Engram placement | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Host mapping, no warming | 207-221 | 28.0-29.2 |
| Host mapping, `TS_DSV41_ENGRAM_WARM=1` | 506-528 | 34.5-35.2 |
| GPU-resident (default) | 532-541 | 35.6-35.9 |

The GPU-resident path also removes the 130-second whole-table warm from
startup and the 60.1 GiB of host page cache the warm path depends on. All three
paths produced identical text at temperature 0 on the determinism check.

That is argmax stability rather than bitwise agreement: ggml's CPU and CUDA
Q2_K dequantizers evaluate the same expression, but the device one may contract
a multiply-subtract into an FMA and differ by up to one ulp. Use
`TS_DSV41_ENGRAM_DEVICE=0` for a run that must match the CPU oracle bit for
bit.

### Host-mapped Engram tables

These options apply only when the tables stay on the host, which happens when
the GPUs cannot hold them or `TS_DSV41_ENGRAM_DEVICE=0` is set.

On network-backed storage, first access to sparse Engram rows dominates prefill
and decode latency. Each token hashes 24 row ids per table, and a row that is
not already page cache is one storage round trip: on the eight-A40 VM's
network-mounted Q4_K_M checkpoint that is **1,435-2,565 ms of input preparation
per 1024-token prefill chunk**, against 56-81 ms once the pages are resident.
Decode input preparation is 5-15 ms a token cold and 0.9 ms warm.

So **warming is the default** whenever the tables are host-mapped and the pages
can stay resident. It runs on its own thread after the model is serving, not
during load, so startup is unchanged and requests work (more slowly) while it
proceeds; startup prints how much it is warming and prints again when it
finishes. A model load reads the whole checkpoint through the page cache and
evicts the previous run's Engram pages, which is why this has to happen after
every load rather than once per machine.

Warming reads only those tables, creates no private copy or pinned allocation,
and leaves the pages evictable. It is skipped with a diagnostic when the mapped
host weights plus 8 GiB do not fit the detected host/cgroup memory allowance, or
when `MemAvailable` could not keep the tables cached anyway.

Warming reads each table with `pread` in 64 MiB blocks, one contiguous run of
the table per reader thread (`TS_DSV4_LOAD_THREADS`, default 16), and skips a
block whose pages `mincore` already reports resident, so a reload whose tables
are still cached reads almost nothing. It used to touch the mapping one byte per
4 KiB page, and on a network filesystem every such fault is a synchronous read
capped at the mount's readahead (128 KiB on the A40 VMs' MooseFS mount): the
seven-A40 lane's synchronous warm of the 103 GiB Q4_K_M tables took 311.3 s.
Measured on that VM, 8 GiB of an evicted table reads at 2.38-2.54 GiB/s with
`pread` and 0.63-0.69 GiB/s with the page walk; see [Load time](#load-time) for
the method. `TS_DSV4_WARM_PREAD=0` restores the page walk for both forms below.
A sync-mode read error fails the load with the file and offset; in the
background it is one log line and the model keeps serving.

| `TS_DSV41_ENGRAM_WARM` | behaviour |
|---|---|
| unset (default) | warm in the background once the model is serving; the finish line reads `[dsv41] warmed ... Engram pages in ...s (background, ...)` |
| `1` | warm synchronously during load, as before; startup takes as long as reading the tables (311.3 s for 103 GiB with the page walk on the seven-A40 lane; at the measured 2.24-2.54 GiB/s `pread` rate that is 41-46 s of reads) |
| `0` | never warm |

Sparse-read mapping advice (`MADV_RANDOM`) is applied only after warming
finishes, whichever form it took: the advice turns off the readahead the warm
pass depends on. The `pread` warm leaves the pages in the page cache without
mapping them into the process; a lookup's first touch of a row is then a minor
fault, not a storage read. 2,000 random 144-byte rows through a `MADV_RANDOM`
mapping of a table warmed that way averaged 0.0037-0.0056 ms, with no major
faults.

`TS_DSV41_ENGRAM_THREADS=1..32` controls the persistent lookup workers;
the default is the smaller of 16 and the hardware thread count. Both prefill
and decode use parallel row lookup: a single token selects 24 independent
rows per Engram table. Setting one worker keeps reads serial. This avoids
serialized page faults during decode without changing the embedding values.
The executor interleaves reads from both tables in one worker-pool job and
uploads each table's rows after the workers finish. Staging is grouped within
64 MiB; a single larger table retains the previous one-table allocation bound.

On Linux, parallel lookup automatically requests random-access advice for
the mapped Engram ranges to reduce unnecessary readahead. The hint is applied
after optional whole-table warming and leaves other tensor ranges at their
existing policy, except for shared boundary pages. Set
`TS_DSV41_ENGRAM_RANDOM=0` to disable it or `=1` to force it. When unset, a
one-worker configuration retains the default mapping policy. Unsupported
platforms and rejected OS hints remain nonfatal. The source comparison and
paired scratch-file results are in the Engram investigation,
`docs/validation/deepseek41/cli-gpu-execution/README.md` (local validation evidence, not committed).

### Backends

`--backend ggml_cuda` is the serving path: it is the only one with kernels for
this architecture's fused ops.

`--backend ggml_cpu` runs the whole model on one CPU device, with the scalar
implementations those fused ops fall back to. It exists so the architecture can
be run and checked without a GPU. It is not a serving path: this checkpoint
reads six of 384 routed experts per layer per token out of 246 GiB.

`TS_DSV41_ALLOW_NON_CUDA_GPU=1` additionally permits `ggml_vulkan` and
`ggml_metal`. There the ordinary graph runs on the GPU and only the
architecture-specific ops fall to the CPU backend, at a host round trip per
occurrence. It is opt-in rather than automatic because what the refusal
originally closed was a silent fallback onto whichever GPU enumerated first, not
an explicit request; startup names each device it applies to. Treat it as a
portability and correctness path until it has been measured on your hardware.

`--backend cpu` is not a ggml backend at all: it is the pure-C#
`DeepSeek4CpuExecutor`, with no native library and no GPU, and it implements the
whole V4.1 graph — the ratio-1 and ratio-2 block compressors, the shared
compressed and indexer caches with the lightning indexer's top-k, candidate
block pruning, the Engram tables, the delayed hyper-connection gates, the shared
expert, and the checkpoint's trained cache quantization (FP8 E4M3 raw rows,
MXFP4 indexer, NVFP4 compressed). It is held to the independent PyTorch oracle
`eng/dsv41-reference.py` at atol=rtol=2e-5, plus exact greedy-argmax agreement,
across one-shot prefill, chunk sizes 1/3/5/8 and reset — at fixture scale
rather than on the released weights. `--backend cuda`, the direct-CUDA engine,
also runs V4.1 through its own kernels with no ggml, and has no numerical gate
yet. Like `ggml_cpu`, both are correctness and portability paths, not serving
ones; the end of
[Running on the ggml CPU backend](#running-on-the-ggml-cpu-backend) has the
detail. `--backend mlx` remains refused.

### One backend per GPU

The architecture-specific DeepSeek ops (compressors, attention prologue and
epilogue, MoE routing and reduction, clamped SwiGLU, hyper-connection gates,
top-k masks) are emitted as `GGML_OP_CUSTOM` nodes and run by a TensorSharp
backend. That backend **wraps** its GPU's CUDA backend and takes its place in
`ggml_backend_sched`: it claims the CUDA device's ops and buffer types as well
as its own, forwards ordinary nodes to CUDA as graph views, and launches the
fused kernels on the same stream.

Registering the two backends side by side instead splits the graph wherever the
backend changes, which a V4.1 layer does about fourteen times. A decode graph
was cut into 565 splits of roughly six nodes each, and the scheduler performs a
blocking host synchronization at every boundary. Wrapping brings that to one
split per GPU:

| | Splits per decode graph | Decode compute | Decode tok/s |
|---|---:|---:|---:|
| Side-by-side backends | 565 / 577 | 26.6-27.1 ms | 35.6 |
| Wrapping backend | 8 | 23.2-23.5 ms | 41.1 |

Prefill is unchanged by this, as expected: a prefill split already does
milliseconds of work, so a per-boundary synchronization was noise there. Set
`TS_DSV4_FUSED=0` to fall back to stock CUDA kernels for these ops entirely.

Startup reports each initialized compute device and the routed-expert CPU
offload count. With `--backend ggml_cuda` and no CPU-offload option, all 40
layers run on CUDA devices. The `auxiliary CPU worker pool` message describes
the scheduler's host pool; it does not indicate CPU-only inference. Engram
lookups use host memory only when the tables are host-mapped. Layer split executes
successive layers on successive GPUs, so low per-device utilization alone does not establish a
CPU fallback. `TS_DSV4_PERF=2` reports input preparation and graph-compute times;
`TS_DSV4_PERF=3` additionally logs actual scheduler backend transitions. These
are diagnostic modes whose logging overhead affects throughput. See the CLI
execution investigation, `docs/validation/deepseek41/cli-gpu-execution/README.md`
(local validation evidence, not committed).

On `ggml_cuda`, V4.1 attention runs TensorSharp's owned F32 kernels (ggml's CUDA
flash attention narrows Q and the softmax weights to F16, and the cache
quantization can amplify those lost bits). **Prefill is sparse by default** once
a launch is wide and long: a chunk of more than 8 queries over at least 8,192
keys (the raw window ring plus the visible compressed rows) attends only to each
query's sliding window and indexer selection, at most 128 + 512 keys, through a
mask-compacted kernel. Everything else keeps the dense kernels: single-token
decode and every DSpark verify (6 rows) the split-key kernel, so a verify still
commits exactly the cache rows decode would; shorter prefill the tiled one, so a
prompt whose attention stays below 8,192 keys is unchanged bit for bit. A row
with more visible keys than the bound falls back to a full scan, so the bound
never drops a key.

Measured on one A40 (`GgmlOpsCudaAttentionPrecisionTest --benchmark-dsv41-prefill
512 33536 64 5`, which selects the kernel through the production gate and the
variable): 512 queries over 33,536 keys with 64 heads took **34.1-34.3 ms** per
launch sparse against **1,547-1,549 ms** tiled. Against a decomposed F32
reference the sparse kernel's maximum absolute error was 1.1e-7 (relative L2
7.4e-7) and the tiled kernel's 8.9e-8 (4.8e-7). A sparse query is also
independent of the other queries in its launch -- query 0 alone, in 9 queries
and in 512 is bit-identical -- so a prompt's result does not depend on how
prefill chunked it. `TS_DSV41_SPARSE_FA=0` restores tiled prefill.

This owned gate (more than 8 queries, at least 8,192 keys, F32 compacted kernel)
is not the gate of ggml's flash-attention kernel, which the non-owned attention
path uses (non-CUDA GPUs, the CPU backend). There `TS_DSV41_SPARSE_FA=1` still
opts into ggml's mask-compacted flash attention for single-token batches or at
least 16,384 keys, whose F16 operands measured up to 7.8e-4 relative L2 against
the CPU oracle; that hint stays opt-in. Sparse attention reduces attention work;
it does not eliminate cross-GPU copies of the shared compressed cache during
prefill.

`TS_DSV41_COMPACT_RAW_GATHER=1` opts into raw-window compaction for sparse
single-token decode. It gathers the 128 visible raw rows on their owning GPU
before moving them to the shared compressed-cache GPU. Masked duplicate rows
pad the raw prefix to 256 rows, so the combined 768-row K tensor satisfies
CUDA's 512-component attention alignment. The physical ring, prefill path,
and non-gathered decode stay unchanged. This option defaults off. The qualified
Q2_K comparison at an approximately 8k prompt improved sustained decode by
13.7% at concurrency 1 and 4; strict flash-arithmetic differences and the
complete measurement settings remain in the validation report.

The default context allocation is capped at 65,536 tokens unless `MAX_CONTEXT`
is supplied. `TS_DSV4_UBATCH` controls the forward microbatch. Unset, V4.1 on a
ggml GPU backend lets the loader choose it: 1024, 512 or 256, the widest that
needs no more routed-expert CPU layers than 256 would, logged as
`[dsv4] prefill ubatch: N (auto; ...)` (see
[Device memory held back for the graph](#device-memory-held-back-for-the-graph)).
The CPU executors and the direct-CUDA engine keep 256. Any explicit value is used
verbatim; `TS_DSV4_UBATCH=256` restores the previous fixed default. A larger
advertised model window does not establish that a particular GPU configuration
can allocate or efficiently serve it.

### Token-batched decode

When several sequences decode at once, all of their tokens run in **one graph**
instead of one graph each. A decode step is dominated by reading the weights
(~9.8 GiB a token at Q4_K_M) and by the ~2,200 small kernels that read them, and
batching pays both once: only the parts that touch a sequence's own state fork
per slot, which for V4.1 means the sliding-window ring, the compressor state,
the lightning indexer's selection and the attention itself. The Engram lookup
needs no fork at all -- its staged rows are already one column per token, so a
slot is just a column, hashed against that slot's own history.

The saving is bounded by routing: each token picks its own 6 of 384 experts, so
the routed-expert reads do not overlap between slots and only the dense
projections, the shared expert and the output head are shared. Measured on eight
A40s at Q4_K_M, aggregate decode throughput:

| concurrent requests | serial decode | token-batched decode |
|---:|---:|---:|
| 1 | 22.8 | 28.9 |
| 2 | 24.8 | 39.3 |
| 4 | 24.3 | 48.9 |
| 8 | 26.5 | 48.5 |

Batching changes GEMM shapes, so a batched step and a solo step are not
bit-identical and a near-tie in the logits can pick a different token. Solo
decode repeated its own output on 6 of 6 greedy prompts; a batched step matched
the solo text on 2-3 of 6. The same prompt run at batch width 2 and at batch
width 4 — the same code path, only wider GEMMs — disagrees at the same rate, so
what changes the output is which requests share a step, not the per-slot wiring.
Set `TS_BATCHED_FUSED_DECODE=0` for a serial path and its determinism.

Four concurrent 10,836-token documents, each hiding a different secret, were
answered with 4/4 correct secrets and no answer containing another slot's
secret, which is the check that the per-slot rings, compressed caches and sparse
selections really are separate.

### Device memory held back for the graph

`TS_DSV4_VRAM_RESERVE_MB` overrides the per-device headroom the layer-split
packer leaves unspent. The default prices the indexer's top-k transients, one
microbatch of activations and a 2 GiB floor. Holding back too much is not free:
on the eight-A40 VM at Q4_K_M, 5,240 MiB forced three layers of routed experts
onto the host and 3,174 MiB needs one, worth 350 -> 480 prefill tok/s.

Both that reserve and the raw sliding-window ring grow with the prefill
micro-batch, so with `TS_DSV4_UBATCH` unset the loader prices the split once per
candidate width. At 65,536 context the default reserve is 2,318 / 2,588 / 3,128
MiB per device for 256 / 512 / 1024, and the ring 512 / 768 / 1,280 rows. It
takes the widest candidate that needs no more routed-expert CPU layers than 256
would -- or than an explicit `--n-cpu-moe` the run pays anyway -- and that does
not move GPU-resident Engram tables to the host. A wider chunk is cheaper per
prefill token: one Q4_K_M-shaped routed-expert layer (`GgmlOpsDsv4MoeWidthBench`,
uniform top-6 routing, one A40) took 35.3-35.7 / 36.9-37.2 / 38.9-39.1 ms a chunk
at 256 / 512 / 1024 tokens resident on the GPU, 3.6x cheaper per token at 1024,
and 112.7-117.9 / 185.0-189.1 / 349.9-353.3 ms on 32 host threads (0.44-0.46
against 0.34 ms a token). But an extra host layer is paid on every decoded
token, so that trade is never made. The log line names the width and, when a
wider one was declined, why. A `--n-cpu-moe` below what 256 needs is refused with that number
(`Re-run with --n-cpu-moe N`). 2048 is not a candidate: the reserve was
validated at 1024 (a 57,424-token prefill peaked with 1,522 MiB free against a
3,072 MiB reserve) and a ggml device OOM ends the process. The choice is exported
(`TSGgml_Dsv4UBatch`) so speculative prefill chunks to the same width.

Priced with the Q4_K_M release's tensor sizes and the per-device budgets its
seven-A40 run implies (free memory after load plus what the load placed, 45,091-
45,123 MiB per device), all three widths need 6 routed-expert CPU layers at 65,536
context, so the loader picks 1024; at 131,072 context the same budgets also give
6 at every width. These are computed plans (`GgmlOpsDsv4UbatchPlanTest`), not
measured loads.

The graph cache is bounded by bytes as well as by entry count. An entry's
compute buffers scale with its shape, and concurrent sequences at different
positions produce many distinct shapes: four concurrent 10.8k-token prefills
used to fill all twelve entries and run a device out of memory, which is not a
survivable error -- ggml's allocator frees a buffer before reallocating it, so a
failed reserve leaves a null buffer behind and the process dies rather than the
request failing. Least-recently-used entries are now freed before a new one is
built, until every device has room for another entry as large as the largest one
cached plus a floor. `TS_DSV4_GRAPH_CACHE_HEADROOM_MB` sets that floor (default
1024); `0` restores the pure count cap.

### Load time

The weights are streamed to the GPUs by a pool of reader threads
(`TS_DSV4_LOAD_THREADS`, default 16) in `TS_DSV4_LOAD_CHUNK_MB` chunks (default
64). Each thread walks ONE CONTIGUOUS RUN of the job list. That matters more than
anything else about the loader on a network filesystem: the jobs are sorted by
(shard, file offset), so handing them out from a shared cursor - which is what
this loader used to do - makes every file descriptor read one chunk and then jump
`threads x chunk`, 1 GiB at the defaults. Readahead is per descriptor, so none of
the sixteen streams is sequential.

Measured on eight A40s with the Q4_K_M release on a MooseFS mount, cold (every
run preceded by evicting all 414 GiB of shards from the page cache, alternating
the two orders twice each):

| job order | weight upload, 294.8 GiB | total model load |
|---|---:|---:|
| one contiguous run per thread | **141-147 s** (2.0-2.1 GiB/s) | **144-155 s** |
| shared cursor (`TS_DSV4_LOAD_CONTIGUOUS=0`) | 360-377 s (0.78-0.82 GiB/s) | 363-382 s |

**2.5x.** `TensorSharp.Runtime/GgufReader.cs:330` records the same finding for the
managed GGUF prefault ("~3x slower on MooseFS"). Ranges are split by BYTES rather
than job count, because a tensor's last chunk is a partial one; a thread that runs
out steals from the BACK of the furthest-behind range so its victim keeps reading
forwards.

Three things that look like the fix and are not, each measured on this box:

* **Page-locking the staging buffers.** The host-to-device copies are 87 s of
  thread time against 5,539 s in `fread` - 1.6% of the loader's work. Pinning is
  worth several times that on the copy itself and almost nothing on the load.
* **More reader threads.** Throughput is not monotonic in thread count on this
  filesystem; `TensorSharp.Backends.Cuda/Dsv4/Dsv4CudaEngine.cs:744` records
  2.4 GB/s at 16 threads against 1.0 GB/s at 96.
* **`MADV_WILLNEED` on the host-expert prefault.** 29.7 s and 31.3 s against a
  27.7 s mean for the plain fault-in walk, i.e. no better.

The passes after the upload read with `pread` too. The prefault of the
host-resident experts (`--n-cpu-moe`) and the Engram warm (see
[Host-mapped Engram tables](#host-mapped-engram-tables)) merge their tensors
into file ranges, split the bytes into one contiguous run per thread
(`TS_DSV4_LOAD_THREADS`), and read 64 MiB blocks on a descriptor per thread,
skipping a block whose pages `mincore` already reports resident. They used to
touch the mapping one byte per 4 KiB page. On a network filesystem each of
those faults is a synchronous read capped at the mount's readahead
(`read_ahead_kb`, 128 KiB on the A40 VMs), which is why the seven-A40 lane
(`--n-cpu-moe 6`, Q4_K_M) logged 129.7 s to prefault 48.2 GiB of experts and
311.3 s to warm 103 GiB of Engram tables.

Measured on that VM with `GgmlOpsDsv4FileWarmBench` (built on Linux with the
native tests), 16 threads, 8 GiB ranges evicted before every arm with
`mincore` = 0 checked, three to five repeats, arms alternated:

| pass over 8 GiB | Engram table, shard 00002 @ 20 GiB | experts, shard 00003 @ 9,002,135,936 |
|---|---:|---:|
| `pread` (default) | 2.38-2.54 GiB/s | 2.24-2.47 GiB/s |
| prefault page walk, 256 MiB spans (`TS_DSV4_WARM_PREAD=0`) | 0.62-0.66 GiB/s | 0.68-0.74 GiB/s |
| Engram page walk, 8 MiB chunks (`TS_DSV4_WARM_PREAD=0`) | 0.63-0.69 GiB/s | 0.65-0.68 GiB/s |
| the same range again, already resident (all blocks skipped) | 64-159 GiB/s | 139-187 GiB/s |

`pread` fills the page cache but not the process's page tables, which the walk
also did, and the first prefill reads the experts densely. So the prefault also
reads one byte per page of each block once it is cached: on resident pages that
walk costs 0.004-0.006 s/GiB at 16 threads, against 0.019-0.023 s/GiB for
`madvise(MADV_POPULATE_READ)` on the same ranges. The host-mapped Engram tables
are not prefaulted into the process's page tables; their rows are read a few at a time.

Read-ahead hints are no substitute on this mount. `MADV_WILLNEED`,
`POSIX_FADV_WILLNEED` and `readahead(2)` over an evicted 8 GiB range each left
128 KiB of it (0.0015%) resident ten seconds later. Do not retry them.

A read error in the prefault or the synchronous Engram warm fails the load and
names the shard and offset. `TS_DSV4_WARM_PREAD=0` restores both page walks
exactly.

The offloaded experts are not page-locked unless `TS_HOST_MOE_PIN=1`. Every node
of an offloaded layer's routed experts is assigned to the CPU backend
(`build_moe_host`), and `ggml_backend_sched` never overrides that assignment, so
its op-offload rule never streams those weights to a GPU: only the
`[n_embd, n_tokens]` activations cross the bus, and a pinned expert is never a
DMA source. On the seven-A40 lane (`--n-cpu-moe 6`) the page-lock added 20.4 s
to the load for 48.2 GiB and made those pages unevictable in the same cgroup
that holds the page cache. The load now prints one line saying the offloaded
experts stay pageable. `TS_HOST_MOE_PIN=1` restores the page-lock and its
`page-locked ... GiB of host experts` line; `TS_HOST_MOE_PIN=0` still disables
pinning for every architecture. The other MoE architectures, whose prefill does
stream offloaded experts, keep pinning by default.

The loader never reads an uploaded chunk again, but it reads the host-mapped
weights right after the upload and serves them from the page cache for the rest
of the run, and page cache is charged to the cgroup. So by default each uploaded
chunk's page cache is released once the chunk is on the device exactly when the
upload plus the host-mapped weights plus 8 GiB exceed the host allowance (the
cgroup limit), and kept otherwise or when the allowance is unknown. The load
prints the decision with its numbers:

```text
[dsv4] load page cache: dropping each uploaded chunk's page cache (automatic: 263.0 GiB upload + 151.2 GiB host-mapped + 8.0 GiB headroom exceeds the 326.9 GiB allowance; TS_DSV4_LOAD_DROP_CACHE=0 overrides)
```

Those are the seven-A40 lane's numbers: 414 GiB of reads into a 326.9 GiB
cgroup. Its load logged the expert prefault at 0.37 GiB/s and the Engram warm at
0.33 GiB/s, about half the rate the same page walks measured on that VM with the
cgroup roughly half full (table above). Dropping cannot speed the upload itself,
which already reads at the storage rate, and costs 5.9-7.3 ms per resident
64 MiB chunk on that mount (`GgmlOpsDsv4FileWarmBench --drop-cost`, about 25-30 s
of thread time for 263 GiB). Its gain is expected on the stages after the upload
and has not yet been measured on a full load; the check is a cold load with the
default against one with `TS_DSV4_LOAD_DROP_CACHE=0`. The rule stays conditional
because dropping every time would make each reload of a checkpoint that lives
entirely on the GPUs cold. `TS_DSV4_LOAD_DROP_CACHE=0` never drops (the previous
default) and `=1` always drops. On the eight-A40 box, `=1` did not change the
upload's read thread-time (5,374 s against 5,539 s, inside the run-to-run spread)
and ended the load with ~39 GiB of page cache instead of ~330 GiB.

One caveat when timing this yourself: on a box whose page cache is already full of
the checkpoint, a load can be SLOWER than one that starts with an empty cache,
because the cgroup is at its limit before the first read and every subsequent read
contends with reclaim. Compare like with like - evict the shards first.

### Multi-turn KV reuse

A second turn's rendered prompt is not a continuation of the first turn's cache.
Ordinary chat drops the previous assistant turn's reasoning (see the history
policy above), so the render diverges from the cache exactly one token after the
previous turn's `<｜Assistant｜>`: the cache holds `<think>` there, the render
`</think>`. Everything before that point still matches, and from the third turn
on that is the WHOLE of the previous turn's prompt, because that prompt already
rendered the earlier turns with their reasoning removed.

The native executor therefore supports partial reuse: `TSGgml_Dsv4Truncate`
moves a slot's head back so the matching prefix is kept and only the new suffix
is forwarded. Per-turn prefill is then a function of the newest answer rather
than of the whole conversation. Two conditions bound it, both stated in
`dsv41_truncate.h`:

* **Alignment.** The target must be a multiple of the widest compression ratio
  (2 for the released checkpoint), so no compression block straddles the new
  head. Callers align their reuse length down; it costs at most one token.
* **Depth.** The raw sliding window lives in a ring of
  `pad64(n_swa + n_ubatch, 256)` positions - 512 for the released checkpoint,
  whose window is 128 - so a rewind reaches 385 positions back from the state it
  rewinds from. Generating an answer moves the head thousands of positions past
  the prompt boundary the next turn wants, which is why every slot keeps a
  **rewind checkpoint**: a shadow copy of the two modularly-addressed rings (the
  raw window and the compressor state), taken at the end of every multi-token
  forward, i.e. at a prompt boundary. Decode steps deliberately do not move it.
  The shadow costs `n_embd_head x ring_raw x 2` bytes per layer per slot - about
  21 MiB for the released checkpoint - and `TS_DSV41_REWIND_CHECKPOINT=0` turns
  it off, after which a rewind deeper than the live ring is declined.

Measured on eight A40s with the Q4_K_M release (`--n-cpu-moe 2`, greedy, the
reported prompt then two `continue` turns, `TS_KV_DEBUG=1`), on 2026-09-11, when
the CLI still planned its own reuse (`KVCache.PlanReuse`, which is what
`TS_KV_DEBUG` prints). Since 2026-09-17 the CLI and the server both go through the
engine's radix prefix cache, described below, and `TS_KV_DEBUG` prints nothing
there. The divergence lands exactly where the policy puts it - in both turns the
cache holds token 128821 (`<think>`) where the render holds 128822 (`</think>`),
one token past `<｜Assistant｜>`:

| turn | prompt tokens | matching prefix | plan | prefill |
|---:|---:|---:|---|---:|
| 1 | 38 | 0 (cold) | Reset | 970 ms |
| 2 | 2,056 | 37, aligned to 36 | PartialReuse | 8,399 ms |
| 3 | 5,841 | 2,055, aligned to 2,054 | PartialReuse | 17,058 ms |

Turn 2 recovers only a constant - the system block, the question and the
assistant header - because the cache past that point is the first answer WITH its
reasoning, which the prompt no longer contains. Turn 3 recovers the whole of turn
2's prompt, and the share grows from there: the matching prefix grows with the
conversation while the re-forwarded suffix stays the size of one answer. Both
rewinds are far outside the live ring (6,529 positions for turn 3) and are served
by the checkpoint.

A decline is a normal outcome, not an error: the caller resets and re-prefills,
which is what happened for every turn before this existed. Plain `deepseek4`
does not truncate at all - its compressor overlaps blocks, so a boundary still
reads the previous block's state rows and aligning the head is not sufficient -
and neither do V4.1's direct-CUDA and pure-C# executors, which have no
checkpoint. `--think` off needs none of this: without the reasoning drop the
render is a pure extension of the cache and reuse needs no rewind.

Across requests this reuse is driven by the radix prefix cache, the default mode
and the path both the CLI and the server take. A finished turn stays resident as
the model's primary cache. The next thinking turn keeps it by rewinding it past
the whole previous answer, and the tree asks the model at admission whether the
slot reaches that far (`CanRewindPrimary`, which reads `TSGgml_Dsv4SlotCanReuse`:
the live ring or the prompt-boundary checkpoint). A refusal means a full prefill,
and the admission line says why, for example `Radix prompt reuse for …: 0/2056
tokens; 2056 token(s) to prefill (rewinding the cached conversation is declined by
the model).` Placement plays no part: `--tp`, `--layer-split` and a single GPU plan
alike. A turn that forwards at least two prompt tokens after its reuse leaves a new
checkpoint behind (the capability's `MinTailPrefillTokens`), so a regenerated turn
does not cost the turn after it its reuse.

That holds for a turn that ran alone and is followed by its own conversation's next
turn, which is how the CLI runs. A turn that overlapped another request ran on a
per-request slot, which is released when it finishes, and the first step the
engine runs for any other request discards a resident primary. So on a server whose
conversations overlap, a thinking turn re-prefills its prompt unless its previous
turn finished with nothing else running and nothing else was admitted before it.

Until 2026-09-29 the tree's donation rule refused every rewind longer than 16
tokens, this one included, so every thinking turn after the first reused nothing
(`kvPlan=Prefill` in the CLI; first reported with `--tp 6`). The rule keeps a deep
cached state for a later request rather than handing it to a request that shares
only its beginning. Declining does not keep a primary cache - the next step the
engine runs discards it either way - so the rule no longer binds one.
`DeepSeek41ThinkingTurnReuseTests` drives such a conversation through the engine
with the real V4.1 chat template. The matching prefix is the same as in the table
above by construction, but the table has not been re-measured through the engine
on the real model.

Finished requests' native slots are retained as well, so more than one
conversation can continue without a full re-prefill, including
conversations whose turns overlap. A retained slot serves at most one later
request, and the tree lets its own conversation rewind it past the 16-token rule
wherever the slot can (`CanMaterialize`, the same checkpoint check): a thinking
turn always has to rewind past the previous answer, so under the rule a retained
slot could serve no thinking turn at all. Another conversation never reaches a
scoped slot, so it cannot take one. If the native side refuses to retain a slot (its budget or
device headroom), the finished turn is not reused and is not advertised either -
before 2026-09-29 the failed attempt left an emptied primary registered, which an
exact continuation then decoded from.
Retention is always on for the native executor, the one that can rewind, except
while a DSpark drafter is loaded. `TS_DSV41_RETAINED_CACHE_MB` (default 2048)
sizes the budget of retained slots; a value that is not a positive number keeps
the default. Measured 2026-09-29 on 6x A40 (`--tp 6`, Q2_K) with four overlapping
three-turn Web UI conversations (`eng/validation/parallel-multiturn-webui.py`):
without retention every turn reused 0 tokens, because each finished slot was
released before the conversation's next turn; with it every conversation reused its own
previous prompt on every turn - thinking on, turn 2 reused 54-62 tokens of 167-210
(the answer before the dropped reasoning is re-rendered) and turn 3 the whole
turn-2 prompt; thinking off, 171-196 of 175-200 and 287-417 of 307-437. All eight
conversations passed their answer checks in both modes.

Native V4.1 requests own independent KV slots. The scheduler therefore sizes
its metadata-only block pool for one context per allowed running request.
The scheduler default permits 16 running requests; the example explicitly
sets `TS_SCHED_MAX_RUNNING_SEQS=4` for four live slots. At context 65,536 and
block size 256, their automatic accounting capacity is 1,024 blocks. This
does not preallocate extra GPU caches or enlarge a request's context limit.
Native slots still require enough device memory when allocated. An explicit
positive `TS_SCHED_NUM_BLOCKS` remains a hard aggregate accounting limit;
setting it below the live requests' combined needs can force prompt
recomputation.

For four concurrent prefills, `TS_SCHED_MAX_BATCHED_TOKENS=4096` permits
1,024 tokens per request in an all-prefill scheduler step. Once any request
starts decoding, `TS_SCHED_PREFILL_CHUNK` caps each remaining prefill; its
default is 256. `TS_SCHED_SOLO_PREFILL_CHUNK` controls a lone request; its
configured default is 8,192, bounded by the total batched-token limit (4,096
by default).
Explicitly setting both chunk limits to 1,024 can avoid a short first request
changing the other requests' prefill sizes, but longer mixed steps also delay
active decoders. Record these scheduler settings alongside
`TS_DSV4_UBATCH` when comparing performance.

The [benchmark matrix](../../benchmarks/engine_comparison/benchmark_config_deepseek41.json)
uses this conservative profile and inherits settings omitted from its per-backend
environment. Set `TS_CPU_MOE_THREADS` explicitly and
`TS_DSV41_COMPACT_RAW_GATHER=0` before launching the matrix to reproduce that
baseline. Set `BENCH_DSV41_GGUF` to the first shard if using the model directory
in this card; the matrix's default directory name differs. Its text scenarios
do not substitute for the separate strict tool, JSON, image/video, and reasoning
checks in the validation report.

### Running on the ggml CPU backend

`--backend ggml_cpu` selects the loader's CPU-only branch: one CPU compute
device instead of enumerated accelerators, every layer on it, and every
V4.1-specific op running the scalar CPU implementation in
`ggml_ops_dsv4_fused_cpu.cpp` rather than a CUDA kernel. Startup prints
`compute devices initialized: 1 CPU device(s)` and
`routed-expert placement: all 40 layer(s) on the explicitly selected CPU device`.

**This is a correctness and portability path, not a serving path.** Every
decoded token reads six of 384 routed experts in each of 40 layers, out of a
selected checkpoint, on general-purpose cores. Those reference kernels run
one worker per node (the V4.1 quantize, candidate-score and candidate-mask
kernels are the exceptions), and the wrapping backend of
[One backend per GPU](#one-backend-per-gpu) is a CUDA object that is not built
here at all, so none of that section's numbers carry over. Treat this backend
as a way to run the architecture where there is no CUDA device — to check a
change, to compare a CUDA result against a host one, or to bring the model up on
a machine that cannot host it otherwise. Do not put it behind a serving
endpoint and do not quote it as a throughput number: this card reports no CPU
throughput because none has been measured.

The server's default backend is `ggml_cpu` on everything but macOS, so omitting
`--backend` on a GPU box selects this path. That used to be a refusal naming
`ggml_cuda`; it now loads, and the load says so once on stderr before any weight
is read (`[dsv41] --backend ggml_cpu: DeepSeek V4.1 will run on ONE CPU
device...`). If you see that line on a machine with GPUs, you wanted
`--backend ggml_cuda`.

What the CPU path *does* have is agreement evidence, and it is per-op rather
than per-checkpoint. The validation report's
[fixture comparison](../deepseek41_validation.md#independent-numerical-reference) records 41/41 elementwise
checks at `atol=rtol=2e-5` against the independent oracle on a local CPU, with
maximum absolute error 5.1633e-6 and 41/41 greedy-token agreement. That covers
the fused ops and the fixture-sized graph. A full-checkpoint CPU run has not
been measured, so CPU/CUDA parity on the real weights is not established here.

```bash
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
  --backend ggml_cpu --port 5000
```

The CPU backend reads the same embedded Engram metadata as CUDA. Download the
current GGUF shards; no separate Engram preparation is needed.

`TS_DSV4_THREADS` sets the compute thread count, defaulting to at most 32. That
cap was chosen for GPU runs, where those threads only do auxiliary host work; on
a CPU-only run they are the whole engine, so set it to the cores the run may
actually use. `TS_DSV4_UBATCH` (fixed at 256 here unless set, where `ggml_cuda`
picks the width automatically) and `MAX_CONTEXT` (default 65,536) both cost host
memory rather than VRAM here.

The options that name GPUs behave as follows:

- `--tp N` shards routed-expert dimensions across GPUs. Combined with
  `ggml_cpu` it is refused before the checkpoint is opened, not ignored.
- Multi-GPU `--tp N` and `--layer-split N` requests are refused on CPU backends.
- The Engram tables stay host-mapped: the GPU-resident placement described
  above needs a device to place them on. `TS_DSV41_ENGRAM_DEVICE=1` is
  therefore refused here before the checkpoint is opened rather than accepted
  and ignored, and so is any value other than `0` or `1`; `=0` names the host
  mappings this path already uses and is accepted. Because the tables are
  host-mapped, the [host-mapped Engram options](#host-mapped-engram-tables) —
  `TS_DSV41_ENGRAM_WARM`, `TS_DSV41_ENGRAM_THREADS`, `TS_DSV41_ENGRAM_RANDOM` —
  all apply on this backend.
- `TS_DSV4_VRAM_RESERVE_MB` is subtracted from the one device's free memory
  before layers are packed onto it. That device is the host here, so its
  default reserve (at least 2 GiB; about 2,318 MiB at the 256-token ubatch and
  65,536 context this path uses, see
  [Device memory held back for the graph](#device-memory-held-back-for-the-graph))
  is held back from system RAM, and the loader's refusal
  when the model does not fit is worded for VRAM ("not enough VRAM ... Re-run
  with `--n-cpu-moe N`"). The advice is still the right advice — see below.
- `TS_DSV41_COMPACT_RAW_GATHER` was measured on CUDA and defaults off; it
  changes the graph rather than the kernel, so it is reachable here, but
  nothing on this path has been measured with it enabled. Leave it off.
- `TS_DSV41_SPARSE_FA` does nothing here. It sets flash attention's `n_kv_max`
  bound, and ggml's CPU flash-attention kernel reads only the first three op
  parameters (`ggml-cpu/ops.cpp`), never that one. Setting it is silently
  inert rather than slow or wrong; there is no mask-compacted CPU kernel to
  opt into.

Read from the loader rather than measured: layer weights are allocated on the
selected compute device, which on this path is the host, so they are a private
anonymous allocation rather than an evictable GGUF mapping. That is the one
operational decision worth making before a long load — `--cpu-moe` (or
`--n-cpu-moe N`) moves the routed experts of those layers into the loader's
host context, which *is* served from the GGUF mapping, so the kernel can evict
those pages instead of the process being killed for them. It is also what the
loader tells you to do if the weights plus this context's caches do not fit,
even though it says "VRAM" while doing so. Neither the resident footprint nor
the load time of a CPU-only full-checkpoint load has been measured.

The vision companion follows the text model onto the same backend. It loaded
with a hardcoded `CUDA` until this path existed, which would have pulled a GPU
into an explicitly CPU-only run; a backend with no ggml registry name is now
refused by name instead of attempted.

`--backend cpu` is the pure C# executor (`DeepSeek4CpuExecutor`), which now
implements V4.1's graph as well as V4's: the compressors at ratios 1 and 2, the
shared compressed and indexer caches, candidate pruning, the Engram tables, the
delayed hyper-connection gates and the trained cache quantization. It is held to
`eng/dsv41-reference.py` at atol=rtol=2e-5 by
`InferenceWeb.Tests.Dsv41CpuExecutorTests`, across one-shot prefill, chunk sizes
1/3/5/8 and reset. That gate is fixture-scale — a five-layer, 256-hidden,
16-token F32 synthetic model — so it establishes architectural agreement with
the oracle rather than parity on the released 246 GiB Q2_K weights, and the
tests return silently unless `TS_DSV41_FIXTURE_DIR` names the fixture
directory. Unlike `--backend ggml_cpu` it takes no vision companion: that
encoder is a native ggml component, and `LoadVisionEncoder` throws here, so
image and video input are not available. The native loader's Engram and
attention knobs — `TS_DSV41_ENGRAM_WARM`, `_THREADS`, `_RANDOM`,
`TS_DSV41_SPARSE_FA`, `TS_DSV41_COMPACT_RAW_GATHER` — are inert on it.
Engram configuration comes directly from the GGUF;
`TS_DSV4_THREADS` defaults to `ProcessorCount` here rather than
min(cores, 32); and `TS_DSV4_CPU_TRACE_DIR` writes the same per-tensor files
that `eng/dsv41-reference.py --output` writes, so the two directories diff
tensor by tensor. Like `--backend ggml_cpu` it is a correctness and portability
path, not a serving one: no throughput, load time or resident footprint has been
measured for a full checkpoint on it.

`--backend cuda`, the direct-CUDA engine, also runs V4.1 with its own kernels and
no ggml. It has per-sequence slots, batched decode, retained conversation state and
local whole-layer placement through `--layer-split N`; routed-expert `--tp N`
remains a `ggml_cuda` mode. The source includes targeted numerical contracts:
[`Dsv4ExpertKernelTests`](../../InferenceWeb.Tests/Dsv4ExpertKernelTests.cs)
checks synthetic quantized expert projections against upstream dequantization,
while [`Dsv41CudaSlotTests`](../../InferenceWeb.Tests/Dsv41CudaSlotTests.cs)
checks slot isolation, batched decode, ring rewind and retained state against a
fresh run of the same direct engine, at 1e-5 of the largest logit. These are
kernel and state-consistency checks, not an independent full-model numerical
oracle or full-checkpoint quality validation. Model/device-gated tests require
their fixture and CUDA hardware; unavailable cases are skips. The CUDA backend notes,
`docs/validation/deepseek41-cuda-backend/README.md` (local validation evidence, not committed),
record what has been verified and what blocks the rest. `--backend mlx` remains refused.

## Forward graph and state

The native graph uses four residual streams and V4.1's delayed
hyper-connection mixing. Layers 1 and 14 add Engram features selected by
deterministic token n-gram hashes. Token normalization and bucket layouts come
from the GGUF metadata; sequence slots retain separate token histories.

Each attention block includes a 128-token raw sliding window. The first two
layers have no compressed attention; the next 18 use compression ratio 2 and
the final 20 use ratio 1. Compressed KV sources and indexer selections are
shared according to the official causal encoder-decoder topology. Query
projections, cache quantization, inverse RoPE, and the grouped output LoRA
follow the V4.1 graph. Index selection uses the lightning indexer and candidate
block filtering. The MoE uses the shared expert plus normalized selected
routed-expert outputs.

The implementation reuses native quantized matrix multiplication,
`mul_mat_id`, attention, hyper-connection kernels, per-sequence slots, and
graph caching. V4.1 activation quantization and candidate filtering have
dedicated operations. It does not call the pure C# or direct-CUDA V4 executor.

Useful source locations:

- [Architecture gate](../../TensorSharp.Models/Models/DeepSeek4/DeepSeek41Architecture.cs)
  and [managed driver](../../TensorSharp.Models/Models/DeepSeek4/DeepSeek4Model.cs).
- [Native loader and scheduler](../../TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp)
  and [V4.1 graph](../../TensorSharp.GGML.Native/ggml_ops_deepseek41.inc).
- [Engram metadata loader](../../TensorSharp.GGML.Native/dsv41_engram_gguf.h)
  and [hashing and configuration validation](../../TensorSharp.GGML.Native/dsv41_engram.h).
- [Chat renderer](../../TensorSharp.Runtime/ChatTemplate.DeepSeek41.cs)
  and [output parser](../../TensorSharp.Runtime/DeepSeek41OutputParser.cs).

## Chat, tools, and JSON

V4.1 uses explicit BOS and `<｜System｜>` framing, and spaced DSML tags such as
`<｜DSML｜ calls>` and `<｜DSML｜ invoke name="tool">`. V4's unspaced DSML format
is incompatible. The renderer handles system, user, developer, assistant, and
tool history; parallel tool results are reordered by their source call IDs.
String tool arguments preserve whitespace, and incomplete invocations are not
dispatched. Inside complete DSML invokes, the parser also accepts the plain
`<parameter name="...">` and `</parameter>` variants observed from Q2_K.
Unrecognised or malformed parameter markup is rejected rather than converted
to empty or partial arguments. The OpenAI chat endpoint retains incoming tool calls, reasoning,
and tool-result IDs when rendering the next turn.

The V4.1 OpenAI chat endpoint constrains declared tool calls with a
request-local DSML grammar. `tool_choice: "auto"` leaves ordinary answers
unconstrained and activates only after the model opens a calls block. With
thinking enabled, activation also waits for `</think>`, so quoted tool syntax
inside reasoning does not start a call. `required` requires a call to a
client-declared function; a named choice restricts that call to the named
function. `none` prevents tool-call output, and `parallel_tool_calls: false`
limits a calls block to one invocation. Internal skill rounds receive fresh
grammar state. Other model families keep their existing policy behavior.

The tool grammar enforces declared names, required parameters, optional
omissions, primitive types, primitive enums/constants, and recursively typed
objects/arrays. It emits parameters and object properties in schema order.
Nested open objects with no declared properties accept arbitrary JSON maps.
For objects with declared properties, generation emits only those properties,
even when the schema permits additional keys. Function parameter schemas with
no declared properties retain the no-argument function convention. Typed
`additionalProperties` schemas are unsupported and rejected explicitly.
Declared integer `enum`/`const` values must be JSON integer literals in the
signed 64-bit range; other encodings or values are rejected before generation.
Ordinary numeric arguments retain the existing Int64-or-double parser behavior;
the lossless argument guarantee below concerns strings, not arbitrary-precision
JSON numbers.
Unsupported assertions, including type unions, schema combinators, patterns,
numeric bounds and string/array length bounds, return HTTP 400 before
generation. This supported subset applies to tool parameters; JSON response
schemas use the existing separate compiler.

Ordinary strings can use the trained raw `string="true"` representation.
Strings containing reserved DSML delimiters remain representable through
`string="false"` with a JSON string: JSON escapes such as `\u003c` preserve
the exact decoded value without closing the surrounding tool markup. The
same protection is retained when parsed calls are rendered into later tool
history, including strings and keys inside nested JSON. Ordinary history
formatting remains unchanged. Raw strings also reserve the `<param`, `</param`,
`<invoke` and `</invoke` tag families so these mistyped tool tags cannot absorb the
rest of the response as argument text. Literal strings containing those prefixes
use the same lossless JSON alternative; ordinary XML such as `<x>` and comparison
signs remain valid raw text. The strict parser is unchanged. Grammar tests establish syntax and argument
round trips; full-checkpoint tool selection and accuracy are measured
separately, with unconstrained baseline failures retained.

The reasoning renderer uses the reference default effort of 50. Ordinary chat
drops past reasoning; tool-enabled chat preserves it. Cached raw assistant
tokens cannot override this history policy. With thinking enabled, JSON grammar
enforcement starts after `</think>`; otherwise it starts at the first output
token. Prompt and parser tests establish format compatibility, not model-level
tool selection, reasoning quality, or JSON task accuracy.

The Chat Completions endpoint accepts `response_format` with thinking enabled for V4.1
because its protocol declares that delayed grammar trigger. This combination
requires JSON grammar enforcement. To request a
JSON final answer after a tool round trip, retain the tool history and catalog
and send `tool_choice: "none"`. Active tool generation and `response_format`
remain mutually exclusive. Validation checks the assistant content channel;
reasoning-only text never counts as a final answer.
These tool-policy and thinking/JSON guarantees apply to `/v1/chat/completions`.
The existing `/v1/responses` surface does not support the same V4.1 tool-history
round trips or reasoning-plus-JSON combination.

Because V4.1 renders tool declarations and parses DSML calls, it is also
eligible for skills, the code tools (`--code-exec`) and, on the server,
[sub-agent delegation](../multi_agent.md), which is on by default on the chat
paths. No delegation results are published for V4.1.

For V4.1, reaching `TS_THINKING_BUDGET` emits the trained `</think>` token and
continues the final answer within the original `max_tokens` limit. The default
budget is 75% when the requested output allowance is at least 512 tokens.
Smaller allowances have no automatic thinking budget; an explicit positive
`TS_THINKING_BUDGET` still applies. `0` disables that budget.
The closing token also consumes one output token. This transition has managed
test coverage; full-checkpoint thinking workflows are measured separately below.
While this V4.1 policy is active, the repetition guard also requests the same
normal closing-token transition if reasoning enters a detected loop. Repetition
in the final answer still stops generation. Disabling the request or scheduler
repetition guard disables this early transition; cancellation, EOS and the
original output limit retain precedence.

## Current limits and tensor-parallel work

- `ggml_cuda` is the serving backend for V4.1. `ggml_cpu` loads the same
  native graph on its scalar CPU implementations, as a correctness and
  portability path with no measured throughput; see
  [Running on the ggml CPU backend](#running-on-the-ggml-cpu-backend). `cpu`
  runs a pure-C# V4.1 executor checked against the PyTorch reference at
  2e-5, and `cuda` runs V4.1 through the direct-CUDA engine's own kernels,
  which has targeted kernel/slot contracts but no independent full-checkpoint numerical gate; both are correctness and portability paths
  rather than serving ones. `mlx` fails before the weights are read, rather
  than loading V4.1 weights into a graph that does not implement it.
- Execution uses one GPU by default. `--layer-split N` selects whole-layer
  placement; `--tp N` selects routed-MoE tensor parallelism with F32 activation/output gathers.
  Attention tensor parallelism and distributed groups are not implemented.
- Concurrent requests have isolated sequence slots. On the native executor
  with the CUDA fused backend (`--backend ggml_cuda`), their decode steps run
  as one token-batched graph (see
  [Token-batched decode](#token-batched-decode)). `--backend ggml_cpu`,
  `TS_DSV4_FUSED=0`, a loaded DSpark drafter or `TS_BATCHED_FUSED_DECODE=0`
  keeps them on per-slot forward calls.
- V4.1 DSpark speculative decoding is experimental. The loader accepts a
  `deepseek41-dspark` drafter (`--draft-model`) on
  `ggml_cuda` and `ggml_cpu` only, refuses it on every other executor, and
  rejects V4 drafters. Synthetic integration tests
  (`DeepSeek41DsparkIntegrationTests`) and initial trained text/image HTTP
  probes with two-GPU layer split and experimental routed-expert TP on
  `ggml_cuda` passed. Broad quality and throughput remain unqualified under
  heavy paging. Separate 24-token text/image pairs matched plain token IDs and
  `max_tokens` finishes in both modes with active DSpark and clean exit. While a
  drafter is loaded, token-batched decode and the retained cache are off.
  Without a drafter, `--spec` (including `--spec-type ngram`) serves standard
  decode.
- The K/V cache is F16 on every executor and `KV_CACHE_DTYPE=q8_0` / `q4_0`
  is **refused at load** (`NotSupportedException`, before the checkpoint is
  opened, from `DeepSeek41Architecture.ValidateLoad`); an explicit `f32` is
  announced on stderr and reported as `f16`. It used to be accepted silently:
  the native graph allocated F16 caches regardless and `KvCacheDtype`
  reported `q8_0`. The caches are not the per-layer K/V tensors the shared
  families hand to ggml flash attention (which reads q8_0/q4_0 K/V at its
  64/128-wide vector-kernel head sizes): they are the MLA latent rows (K
  doubles as V at the latent width, F16-only in the CUDA flash-attention
  kernels) in the sliding-window ring, the compressed and indexer rows, the
  rewind-checkpoint shadows and the DSpark draft rings, written by
  `TSG_DSV4_FUSED_ATTN_PREP` / `TSG_DSV4_FUSED_COMPRESS` and read by
  `TSG_DSV4_FUSED_KGATHER`, the compact and TP gathers and the checkpoint
  copies as F16 rows with no dequantize step, on the ggml CUDA, ggml CPU,
  direct-CUDA and pure-C# executors alike. Quantizing them would mean
  re-typing every one of those kernels; and the checkpoint's own trained
  cache quantization (FP8 E4M3 raw rows, MXFP4 indexer rows, NVFP4
  compressed rows) is already applied before each F16 store, so a q8_0
  block would not shrink the cache's information content. The launch
  examples above pass `KV_CACHE_DTYPE=f16`, which is the only value that
  changes nothing.
- Image/video input requires the separately prepared vision companion.
  The encoder and image/text graph have CPU/CUDA fixture coverage and
  complete-checkpoint media checks under layer placement and routed TP.
  Real-image BF16 feature comparisons exceed
  the small-fixture elementwise tolerance; see the validation report. There is no
  validated audio inference path.
- A compatible llama.cpp V4.1 inference runtime is needed for a same-weight
  comparison. The linked GGUF repository's patch adds conversion support only.
  An unavailable reference does not establish quality or performance parity.
- The full-checkpoint numerical smoke produces the expected tokens but fails
  the strict F32-input oracle comparison (relative L2 0.146216, maximum absolute
  error 2.708920). Quantized activation arithmetic differs from that reference;
  the retained stage analysis
  (`docs/validation/deepseek41/smoke18-reference/README.md`, local validation
  evidence, not committed) does not fully attribute the final discrepancy. Greedy agreement is not
  strict numerical parity.

Extending tensor parallelism to attention requires rank-local graphs, weight
shards, and cache state, with a reduction after attention output before the
next nonlinear residual operation. Existing GLM support in
[ggml_ops_glm_dsa.cpp](../../TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp) supplies
reusable block-aligned weight slicing and rank-local graph execution with
device collectives. V4.1's eight output groups should remain intact when
sharding query heads and grouped output projections; the single KV head and
shared indexer state can initially be replicated.

The implemented expert sharding uses unequal block-aligned partitions. The
2304-wide dimension contains nine 256-element K-quant blocks: two ranks can
use widths 1280 and 1024, and four ranks can use 768, 512, 512, and 512.
Gate/up tensors split along that intermediate dimension for every expert;
down tensors split along their input dimension, followed by host-staged
reduction. Shared-expert and CPU-offload outputs are counted once. Current
ggml CUDA does not expose the old split-buffer interface; this path builds
and executes the rank-local expert graphs explicitly.

See the [validation protocol](../deepseek41_validation.md) for short and long
prompts, JSON, tool round trips, agent workflows, concurrency, placement, and
existing-model regression checks. Unsupported scenarios remain explicitly
unverified.

The latest managed stage passed **3,651/3,651 tests locally and on the requested
VM**, with no skips, including 84 focused audio/image/API checks on both hosts.
Coverage includes tool-history delimiter round trips, valid partial Unicode
tokens, malformed UTF-8 rejection and image/audio request validation before
streaming. The preceding image-validation stage passed 3,635/3,635 tests on
both hosts, including 91 local focused checks.
Each Runtime DLL is unchanged from its verified 3,597-test raw tag-family
serialization stage. The initial matched placement baseline used the earlier
3,566-stage host; those measurements and the intervening 3,582-stage checks
remain preserved. The inference harness separately passed 33 unit tests locally
and on the VM; its current scope is listed in the
[validation report](../deepseek41_validation.md).
The first image-validation VM attempt
exposed a test admission race
(`docs/validation/deepseek41/retained-cache-admission/README.md`, local validation
evidence, not committed);
the synchronized test class and full lane pass on both hosts with the original
assertions preserved.
Exact commands, exclusions, counters, hashes and the retained intermittent test
failure (`docs/validation/deepseek41/managed-correctness/README.md`, local
validation evidence, not committed) are separate from full-checkpoint quality and
performance results.

The final layer and routed-TP profiles use native `6b3b5ab3…` and managed stage 3,651.
Layer split passed **138/138 inference cases**; routed TP passed **129/130**.
The layer plan also includes eight concurrent long-context cases:

| Scenario group | Layer split | Routed TP |
|---|---:|---:|
| Short, JSON, schema, history and default-parallel tool workflows | 30/30 | 29/30 |
| Required, named, none, serial and parallel tool policies | 30/30 | 30/30 |
| Thinking workflows | 4/4 | 4/4 |
| Non-streaming workflows | 4/4 | 4/4 |
| Images and sampled video frames | 25/25 | 25/25 |
| Chinese and Unicode JSON | 10/10 | 10/10 |
| Separately configured serial tool workflows | 10/10 | 10/10 |
| Sustained decode | 15/15 | 15/15 |
| Long retrieval at approximately 8k and 32k input tokens | 2/2 | 2/2 |
| Four concurrent requests at each long-context size | 8/8 | Not in this profile |

On the eight-A40 VM, the layer profile's median sustained decode was
**34.83 tokens/s** for one request and **8.46 tokens/s per request** at concurrency
four, with exactly 512 generated tokens per measured request. Time to first
token was **19.985 s** for 7,706 input tokens and **80.240 s** for 30,585.
The concurrent-long phase recorded 153,187 native prefill tokens including
warmup and zero KV-pool preemptions.
The decode results (`layer8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-steady.json`), single-request long results
(`layer8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-long.json`) and concurrent-long accounting
(`layer8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-long-parallel-runs.json`), all in `docs/validation/deepseek41/full-checkpoint/`
(local validation evidence, not committed), retain the measured scope and timing evidence.

The previously failing named-thinking and thinking-agent cases now pass in both
final profiles. In routed TP, one default-parallel agent request still issued
`calculate_total` with zero placeholder arguments alongside `read_invoice`,
before receiving the invoice result. The strict checker rejected it; this
group changed from the first TP profile's 30/30 to 29/30. Separate serial-policy
success does not remove that failure. A local prompt/grammar/parser diagnosis
(`docs/validation/deepseek41/parallel-tool-dependency/README.md`, local validation
evidence, not committed) preserves the model-emitted calls and found no defect forcing the extra call or
zero values. The final four-layer CPU-offload profile separately passed 28/30
default-parallel quality cases, retaining two premature dependent-call failures.
The layer profile's 30/30 result does not remove either placement's failures.
Native code and launch settings also
changed, so these results do not isolate the grammar change or establish a
blanket quality gain. The twelve HTTP input-rejection checks above are separate
from the inference plans. Exact reports and remaining comparisons are in
the final placement records, `docs/validation/deepseek41/final-placements/README.md`
(local validation evidence, not committed).

Existing-model checks also retain regressions. The final 75-case comparison
(`docs/validation/deepseek41/existing-model-regressions/final3651-native6b3/README.md`,
local validation evidence, not committed) passed 39/75 cases and introduced no failures relative to its paired references
in that run; separate Unicode JSON coverage passed 15/15. The subsequent
repeated JSON comparison
(`docs/validation/deepseek41/json-performance/completed-r2/README.md`, local
validation evidence, not committed) exposed an additional failure for an identical request and recorded slower
Qwen3.5 first-token latency despite faster short-answer decode. Those results
remain separate from the earlier run's zero-introduced-failure observation.
They do not establish a blanket absence of regressions.

Qwen3.5's shorter alternating control
(`docs/validation/deepseek41/json-performance/qwen35-alternating/README.md`)
also showed slower final latency. A later 72-request control
(`docs/validation/deepseek41/json-performance/qwen35-solo72/README.md`; both
are local validation evidence, not committed) held the native library fixed, passed every answer and did not reproduce the
slowdown. No production fix was made from these diagnostics; the differing
results and their limits remain in the validation report.

## Routed CPU expert reads (Linux experiment)

`TS_DSV4_HOST_EXPERT_READ=1` enables bounded parallel reads for routed experts
offloaded by the GPU executor. Unset or `0` retains the existing path; other
values are rejected. This requires contiguous GGUF host mappings on Linux.
After routing, it reads the selected experts' original gate/up/down bytes into
the OS page cache. The original ggml operators still use the mmap weights;
quantization, routing and reduction order are unchanged. It does not predict
future routes or overlap computation across layers.

The TensorSharp CPU backend prepares registered expert mappings immediately
before the original `MUL_MAT_ID` operations. It adds no routing or read nodes to
the graph. If the selected IDs are produced on the CPU, their graph prefix is
executed first; gate/up/down using the same selected-ID tensor share preparation.
Reads run outside the CPU compute team, and all arithmetic and unrelated custom
operations still execute on the unchanged upstream CPU backend.

Descriptors and I/O workers live with the model. Worker count respects process
CPU affinity/cgroup availability and configured threads, with a maximum of 16.
Staging scales with the actual host-memory allowance up to 64 MiB, with at most
4 MiB per read. When shared `hostPools` are attached, staging reserves RAM credit
before allocation and retains it until model destruction frees the buffers.
Mapped/OS-cached weight pages and other runtime allocations are outside this
staging charge.

Resident ranges skip reads. During preparation, multi-token inputs check residency
and single-token inputs may reuse a per-expert observation for eight layer calls.
After 32 single-token CPU graphs with no reads or major faults, the wrapper can
delegate the entire graph directly. A process major-fault change resets this
heuristic and invalidates the per-expert hints; multi-token graphs also resume
preparation. With the reader enabled, `TS_DSV4_HOST_EXPERT_HOT_BYPASS=0` disables
this heuristic (`1` is its default; other values are rejected). The threshold is
experimental, not a calibrated storage cost model. These are advisory hints, not
page locks or guaranteed residency: the original mmap can still fault normally
if the OS reclaims a page. Read errors reject the
forward and require reloading the model; partially read staging never serves as
weights. The default remains disabled pending broader storage/workload coverage.
See the [unified-memory validation record](../design/unified-memory.zh-CN.md)
for cold and warm measurements and their limits.
