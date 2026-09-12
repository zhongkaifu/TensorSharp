# DeepSeek V4.1 Flash (`deepseek41`)

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

## Prepare the Q2_K checkpoint

The supported artifact under validation is the seven-part Q2_K release from
[vcruz305/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/tree/8e0c4de3cb6519bfc11ed69dc87184b457a57bb5),
revision `8e0c4de3cb6519bfc11ed69dc87184b457a57bb5`. Keep all seven shards in
one directory and give TensorSharp the first shard. The release contains mixed
tensor types, including Q2_K and Q3_K; the filename does not imply every tensor
uses Q2_K. The seven files total 264,514,761,248 bytes (246.35 GiB). Their
[complete-file SHA-256 verification record](../validation/deepseek41/checkpoint-sha256.json)
lists every filename, expected size, and matching digest.

The same repository's eleven-part Q4_K_M release (415 GiB) is tested too, and
the [quantization report](../validation/deepseek41-quants/README.md) records
what changes with it: the two Engram tables grow to 51.5 GiB each, so they stay
host mappings instead of going GPU-resident, and the routed experts need CPU
offload on eight 46 GB cards. Everything below applies to either release, the
Engram sidecar included.

V4.1 also needs a small tokenizer-derived Engram sidecar. The published GGUF
does not contain all of the causal encoder-decoder and Engram configuration.
Some Engram keys use the older `deepseek4` prefix, and the tokenizer padding
metadata differs from the Engram hash padding. The preparer reads the official
configuration and tokenizer rather than guessing these values.

From the repository root, on the machine storing the model:

```bash
python3 -m venv /workspace/dsv41-tools
/workspace/dsv41-tools/bin/python -m pip install \
  numpy==2.0.2 tokenizers==0.22.2 huggingface_hub

/workspace/dsv41-tools/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="vcruz305/DeepSeek-V4.1-Flash-GGUF",
    revision="8e0c4de3cb6519bfc11ed69dc87184b457a57bb5",
    allow_patterns=["DeepSeek-V4.1-Flash-Q2_K-*.gguf"],
    local_dir="/workspace/models/deepseek41-q2",
)
PY

/workspace/dsv41-tools/bin/python eng/dsv41-prepare.py \
  /workspace/models/deepseek41-q2 \
  --repo deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277
```

The preparer downloads only the official `config.json` and `tokenizer.json`,
then writes `deepseek41.engram.bin` and a provenance file,
`deepseek41.config.json`, beside the GGUF shards. It checks vocabulary size,
compressed vocabulary size, and the official Engram layout, and records source
and sidecar SHA-256 hashes. The native loader checks the GGUF tensor dimensions
against that layout. Python is required for preparation, not inference.
Use `--source-dir` to prepare from already-downloaded official files.

The pinned official files used during implementation have these SHA-256 hashes:

| File | SHA-256 |
|---|---|
| `config.json` | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| `tokenizer.json` | `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |

## Prepare the optional vision companion

The supplied GGUF conversion omits the vision tower, aligner, learned image
delimiters and per-layer visual routing biases. The official release places
the approximately 970 MB vision/aligner weights in an isolated shard, so these
can be prepared without downloading the original text weights:

```bash
/workspace/dsv41-tools/bin/python -m pip install gguf
/workspace/dsv41-tools/bin/python eng/dsv41-prepare-vision.py \
  /workspace/models/deepseek41-q2 \
  --repository deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277
```

This creates `deepseek41.vision.gguf` and `deepseek41.vision.json` beside the text
shards and Engram sidecar.
The preparer downloads the isolated vision shard and small byte ranges for
the delimiter/router tensors and preserves BF16/F32 storage. It verifies the
complete isolated vision shard against its LFS SHA-256 and records individual
range hashes for delimiter/router weights; it does not download or verify the
entire original text shards. The [companion provenance record](../validation/deepseek41/vision-companion.json)
contains all 306 tensors, source revisions, ranges, and the output digest.
The native loader checks the parent tokenizer fingerprint and model
dimensions before attachment. To enable images, add
`--mmproj /workspace/models/deepseek41-q2/deepseek41.vision.gguf` to the server
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
eight [image rejection checks](../validation/deepseek41/full-checkpoint/tp8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-image-input-rejections.json)
and four [Responses audio rejection checks](../validation/deepseek41/full-checkpoint/tp8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-audio-input-rejections.json).
Streaming and non-streaming requests returned JSON 400 without SSE for remote
image URLs, malformed image base64, and valid-shaped or malformed audio parts.
The earlier CPU-offload host's eight image checks remain preserved separately.

## Run the implemented path

Install the .NET 10 SDK, CMake, a C++ compiler, and the CUDA toolkit with `nvcc`
on `PATH`. Build from the repository root. These commands target the requested
A40 VM (CUDA architecture 8.6); adjust both architecture values for other GPUs:

```bash
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON \
  TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES=86 \
  bash TensorSharp.GGML.Native/build-linux.sh
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release \
  -p:CudaArch=compute_86 -p:TensorSharpSkipGgmlNative=true

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_CONTEXT=65536 \
  TS_CPU_MOE_THREADS=32 TS_DSV41_TP=0 TS_DSV4_UBATCH=256 \
  TS_DSV41_ENGRAM_WARM=1 TS_DSV41_SPARSE_FA=1 \
  TS_DSV41_COMPACT_RAW_GATHER=0 KV_CACHE_DTYPE=f16 \
  TS_SCHED_MAX_RUNNING_SEQS=4 TS_SCHED_MAX_BATCHED_TOKENS=4096 \
  TS_SCHED_PREFILL_CHUNK=256 TS_SCHED_SOLO_PREFILL_CHUNK=8192 \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /workspace/models/deepseek41-q2/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
  --backend ggml_cuda --tp 8 --port 5000
```

The host build copies the native library beside the server DLL. This launch
uses the conservative benchmark matrix's microbatch and scheduler settings;
the optimized profiles in the validation report use different settings. Choose
`TS_CPU_MOE_THREADS` for the available CPU quota and record it for each run.
Set it in the launch environment, including for GPU-only placements: native
CPU graph work and host reduction can still affect latency. The current CLI
also accepts `--cpu-moe-threads N`; use the same value if supplying both, since
the positive environment value takes precedence in the native loader.

To select the final measured eight-A40 layer profile instead, use this optional
launch after preparing the vision companion. It sets the optimized flags
explicitly; the conservative example above and the defaults remain unchanged:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_CONTEXT=65536 \
  TS_DSV4_NGPU=8 TS_DSV4_UBATCH=1024 KV_CACHE_DTYPE=f16 \
  TS_CPU_MOE_THREADS=32 TS_DSV41_TP=0 \
  TS_DSV41_SPARSE_FA=1 TS_DSV41_COMPACT_RAW_GATHER=1 \
  TS_DSV41_ENGRAM_WARM=1 TS_DSV41_ENGRAM_THREADS=16 TS_DSV4_PERF=1 \
  TS_SCHED_MAX_RUNNING_SEQS=4 TS_SCHED_MAX_BATCHED_TOKENS=4096 \
  TS_SCHED_PREFILL_CHUNK=1024 TS_SCHED_SOLO_PREFILL_CHUNK=1024 \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /workspace/models/deepseek41-q2/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
  --mmproj /workspace/models/deepseek41-q2/deepseek41.vision.gguf \
  --backend ggml_cuda --tp 8 --n-cpu-moe 0 --cpu-moe-threads 32 \
  --host 127.0.0.1 --port 5000 --max-tokens 2048
```

The [measured launch record](../validation/deepseek41/full-checkpoint/layer8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-launch.json)
contains the original VM paths, binary hashes and environment. This profile
passed 138/138 inference cases; its throughput and limits are recorded below.
Sparse flash attention and compact gathering remain opt-in, with their
documented floating-point differences. Startup and page warming are excluded
from the inference measurements.

Without an explicit thread setting, a GPU-only V4.1 load uses the caller's
`TS_DSV4_THREADS` value, defaulting to at most 32; CPU expert offload instead
uses the detected available CPU parallelism. Earlier native builds did not
receive the CLI thread override. In particular, the first measured TP run's
`--cpu-moe-threads 48` did **not** configure 48 native threads: its native pool
is inferred to have used the default 32 from the loader source, absent native
environment override, and the VM override probe. That original host did not
expose a pool-width getter, so 32 was not measured directly there. Keep that
baseline distinct from later explicit-thread experiments.

For this architecture, `--tp 8` requests **eight GPUs using layer split**. The
startup diagnostic states the placement mode. TensorSharp distributes whole
layers according to available VRAM by default.
`TS_DSV4_NGPU` overrides the GPU count. Set `CUDA_VISIBLE_DEVICES` to the exact
devices intended for the run. An explicit `TS_DSV4_NGPU=0` selects visible
devices automatically and defers rank-count validation to the native loader.

`TS_DSV41_TP=8` additionally enables experimental **routed-MoE tensor
parallelism** on those eight GPUs. This setting accepts `0` (disabled) or a
rank count from `2` through `8`, which must equal the GPU count selected by
`--tp` or `TS_DSV4_NGPU`. With automatic GPU selection, the native loader
checks the count after enumerating visible devices. An invalid value or count
mismatch is an error.

In this mode, routed-expert gate/up/down matrices are partitioned along the
FFN intermediate dimension and executed concurrently across all selected
GPUs. Partial outputs are reduced through host-staged F32 buffers. Attention,
shared experts, and caches retain their layer placement. This is a partial
tensor-parallel implementation; it does not shard attention or enable
distributed tensor-parallel groups. Host transfers can limit throughput, so
this option does not establish a speedup over layer split. The first full Q2_K
TP quality/performance run has completed and was slower than layer split; see
the [measured placement profiles](../deepseek41_validation.md#full-checkpoint-routed-moe-tp).

Independent numerical fixtures passed on 2/4/8 GPUs, including quantized
expert shards and complete-model oracle checks. Those small fixtures do not
establish that the full Q2_K checkpoint fits on two or four A40s. The VM example
uses eight; smaller placements require enough CPU expert offload to fit.
Its Engram warming consumes approximately 60 GiB of host page cache before
readiness. Record cold-load and warming time separately from warm throughput.

If the weights and context do not fit, add `--n-cpu-moe N` to keep the routed
experts of the first N layers on the host, or `--cpu-moe` for all routed
experts. Attention, routing, and the shared expert remain on the GPU. Engram
tables always remain memory-mapped on the host; only selected embedding rows
are read and transferred for each input batch. CPU MoE offload and layer split
are implemented, but their throughput must be measured for the chosen hardware
and context. When combined with `TS_DSV41_TP`, CPU-offloaded leading layers
retain whole CPU experts; the remaining layers use the routed-expert shards.

Native `6b3b5ab3…` explicitly assigns shared gate/up/down projections to the
layer device. This corrects earlier scheduler placement that could send shared
gate/up work to CPU after a CPU-offloaded or TP routed branch. The
[placement and numerical checks](../validation/deepseek41/shared-expert-placement/README.md)
passed 597/597 on two GPUs. Earlier full-checkpoint placement benchmarks retain
their original binaries and results. Final CPU-offload, routed-TP and layer-split
profiles using the corrected placement have completed. See the
[placement records](../validation/deepseek41/final-placements/README.md).

For a warm CPU-offload benchmark, read the offloaded expert pages separately
after model loading and before starting the timed requests:

```bash
/workspace/dsv41-tools/bin/python eng/dsv41-warm-experts.py \
  /workspace/models/deepseek41-q2/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
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
together are 60.2 GiB. vLLM and SGLang store the same rows as FP8 values plus
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

| `TS_DSV41_ENGRAM_WARM` | behaviour |
|---|---|
| unset (default) | warm in the background once the model is serving |
| `1` | warm synchronously during load, as before; startup takes ~110-210 s longer |
| `0` | never warm |

Sparse-read mapping advice (`MADV_RANDOM`) is applied only after warming
finishes, whichever form it took: the advice turns off the readahead the warm
pass depends on.

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
paired scratch-file results are in the
[Engram investigation](../validation/deepseek41/cli-gpu-execution/README.md).

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
lookups still use host memory, and layer split executes successive layers on
successive GPUs, so low per-device utilization alone does not establish a
CPU fallback. `TS_DSV4_PERF=2` reports input preparation and graph-compute times;
`TS_DSV4_PERF=3` additionally logs actual scheduler backend transitions. These
are diagnostic modes whose logging overhead affects throughput. See the
[CLI execution investigation](../validation/deepseek41/cli-gpu-execution/README.md).

`TS_DSV41_SPARSE_FA=1` opts into CUDA mask-compacted flash attention for
single-token batches or at least 16,384 cached keys. It attends to at most
128 raw-window keys plus 512 selected compressed keys. Shorter prefill
uses dense flash attention because its shared KV tiles were faster on the
tested A40. The measured complete-checkpoint profiles explicitly enable this
option; its default remains disabled. This option reduces attention work; it does not eliminate
cross-GPU copies of the shared compressed cache during prefill.

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
is supplied. `TS_DSV4_UBATCH` controls the forward microbatch, defaulting to 256
for V4.1. A larger advertised model window does not establish that a particular
GPU configuration can allocate or efficiently serve it.

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

`TS_DSV4_LOAD_DROP_CACHE=1` releases each chunk's page cache once it is on the
device. It does not make the load faster (5,374 s of read thread-time with it
against 5,539 s without, inside the run-to-run spread) but it ends the load with
~39 GiB of page cache instead of ~330 GiB, which leaves room for the host experts
the next phase pins. It is off by default because each call costs real time on a
FUSE mount.

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
reported prompt then two `continue` turns, `TS_KV_DEBUG=1`). The divergence lands
exactly where the policy puts it - in both turns the cache holds token 128821
(`<think>`) where the render holds 128822 (`</think>`), one token past
`<｜Assistant｜>`:

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
246 GiB Q2_K checkpoint, on general-purpose cores. Those reference kernels run
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
  --model /models/deepseek41-q2/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
  --backend ggml_cpu --port 5000
```

The preparation steps are unchanged: the tokenizer-derived
`deepseek41.engram.bin` sidecar is required on the CPU backend exactly as it is
on CUDA, and the load is refused without it before any weight is read.

`TS_DSV4_THREADS` sets the compute thread count, defaulting to at most 32. That
cap was chosen for GPU runs, where those threads only do auxiliary host work; on
a CPU-only run they are the whole engine, so set it to the cores the run may
actually use. `TS_DSV4_UBATCH` (default 256) and `MAX_CONTEXT` (default 65,536)
behave as they do on CUDA, and both cost host memory rather than VRAM here.

The options that name GPUs behave as follows:

- `TS_DSV41_TP` shards routed-expert dimensions across GPUs. Combined with
  `ggml_cpu` it is refused before the checkpoint is opened, not ignored.
- `TS_DSV4_NGPU` selects how many GPUs to enumerate. There are none to
  enumerate here, so the loader never reads it; it is neither an error nor a
  way to get more than the one CPU device.
- `--tp N` finds no second device to split layers across, so the multi-GPU
  gate degrades it to a single device with a warning. That warning is written
  for GPU hosts and says "Running on ONE GPU"; on this backend read it as one
  CPU device.
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
  2048 MiB default holds back 2 GiB of system RAM, and the loader's refusal
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
attention knobs — `TS_DSV41_ENGRAM_WARM`, `_THREADS`, `_RANDOM`, `_SIDECAR`,
`TS_DSV41_SPARSE_FA`, `TS_DSV41_COMPACT_RAW_GATHER` — are inert on it, though
the prepared `deepseek41.engram.bin` sidecar is still mandatory;
`TS_DSV4_THREADS` defaults to `ProcessorCount` here rather than
min(cores, 32); and `TS_DSV4_CPU_TRACE_DIR` writes the same per-tensor files
that `eng/dsv41-reference.py --output` writes, so the two directories diff
tensor by tensor. Like `--backend ggml_cpu` it is a correctness and portability
path, not a serving one: no throughput, load time or resident footprint has been
measured for a full checkpoint on it.

`--backend cuda`, the direct-CUDA engine, also runs V4.1 with its own kernels and
no ggml. It is not yet held to a numerical gate — see
[the CUDA backend notes](../validation/deepseek41-cuda-backend/README.md) for what
has been verified and what blocks the rest. `--backend mlx` remains refused.

## Forward graph and state

The native graph uses four residual streams and V4.1's delayed
hyper-connection mixing. Layers 1 and 14 add Engram features selected by
deterministic token n-gram hashes. Token normalization and bucket layouts come
from the prepared sidecar; sequence slots retain separate token histories.

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
- [Engram hashing and sidecar reader](../../TensorSharp.GGML.Native/dsv41_engram.h)
  and [preparer](../../eng/dsv41-prepare.py).
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
requires JSON grammar enforcement; `TS_JSON_GRAMMAR=0` is rejected. To request a
JSON final answer after a tool round trip, retain the tool history and catalog
and send `tool_choice: "none"`. Active tool generation and `response_format`
remain mutually exclusive. Validation checks the assistant content channel;
reasoning-only text never counts as a final answer.
These tool-policy and thinking/JSON guarantees apply to `/v1/chat/completions`.
The existing `/v1/responses` surface does not support the same V4.1 tool-history
round trips or reasoning-plus-JSON combination.

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
  which has no numerical gate yet; both are correctness and portability paths
  rather than serving ones. `mlx` fails before the weights are read, rather
  than loading V4.1 weights into a graph that does not implement it.
- Multi-GPU execution defaults to whole-layer placement. `TS_DSV41_TP` enables
  experimental routed-MoE tensor parallelism with host-staged reduction.
  Attention tensor parallelism and distributed groups are not implemented.
- Concurrent requests have isolated sequence slots. V4.1 currently falls back
  to per-slot forward calls instead of the V4 fused token-batched graph, so
  concurrency does not imply batched GPU throughput.
- V4.1 DSpark speculative decoding is not implemented; V4 draft models are
  rejected.
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
  the [retained stage analysis](../validation/deepseek41/smoke18-reference/README.md)
  does not fully attribute the final discrepancy. Greedy agreement is not
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
exposed a [test admission race](../validation/deepseek41/retained-cache-admission/README.md);
the synchronized test class and full lane pass on both hosts with the original
assertions preserved.
[Exact commands, exclusions, counters, hashes and the retained intermittent test failure](../validation/deepseek41/managed-correctness/README.md)
are separate from full-checkpoint quality and performance results.

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
[Decode results](../validation/deepseek41/full-checkpoint/layer8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-steady.json),
[single-request long results](../validation/deepseek41/full-checkpoint/layer8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-long.json)
and [concurrent-long accounting](../validation/deepseek41/full-checkpoint/layer8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-long-parallel-runs.json)
retain the measured scope and timing evidence.

The previously failing named-thinking and thinking-agent cases now pass in both
final profiles. In routed TP, one default-parallel agent request still issued
`calculate_total` with zero placeholder arguments alongside `read_invoice`,
before receiving the invoice result. The strict checker rejected it; this
group changed from the first TP profile's 30/30 to 29/30. Separate serial-policy
success does not remove that failure. A [local prompt/grammar/parser diagnosis](../validation/deepseek41/parallel-tool-dependency/README.md)
preserves the model-emitted calls and found no defect forcing the extra call or
zero values. The final four-layer CPU-offload profile separately passed 28/30
default-parallel quality cases, retaining two premature dependent-call failures.
The layer profile's 30/30 result does not remove either placement's failures.
Native code and launch settings also
changed, so these results do not isolate the grammar change or establish a
blanket quality gain. The twelve HTTP input-rejection checks above are separate
from the inference plans. Exact reports and remaining comparisons are in
the [final placement records](../validation/deepseek41/final-placements/README.md).

Existing-model checks also retain regressions. The final
[75-case comparison](../validation/deepseek41/existing-model-regressions/final3651-native6b3/README.md)
passed 39/75 cases and introduced no failures relative to its paired references
in that run; separate Unicode JSON coverage passed 15/15. The subsequent
[repeated JSON comparison](../validation/deepseek41/json-performance/completed-r2/README.md)
exposed an additional Qwen3 failure for an identical request and recorded slower
Qwen3.5 first-token latency despite faster short-answer decode. Those results
remain separate from the earlier run's zero-introduced-failure observation.
They do not establish a blanket absence of regressions.

A [matched chunk control](../validation/deepseek41/existing-model-regressions/qwen3-json-chunks/README.md)
reproduced the Qwen3 response change in both builds; the original concurrent
chunk partitions were not recorded. Qwen3.5's shorter
[alternating control](../validation/deepseek41/json-performance/qwen35-alternating/README.md)
also showed slower final latency. A later
[72-request control](../validation/deepseek41/json-performance/qwen35-solo72/README.md)
held the native library fixed, passed every answer and did not reproduce the
slowdown. No production fix was made from these diagnostics; the differing
results and their limits remain in the validation report.
