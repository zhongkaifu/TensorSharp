# Embedding models

[English](embeddings.md) | [中文](embeddings_zh-cn.md)

TensorSharp serves GGUF BERT, XLM-RoBERTa, Nomic BERT, and EmbeddingGemma 2 encoders through
`TensorSharp.Server` and `TensorSharp.Server.Host`. Packages and archives reflect
their own release tags; older releases may not include this support. It addresses
[discussion #183](https://github.com/zhongkaifu/TensorSharp/discussions/183),
including semantic search over source code stored in SQLite.

## Current EmbeddingGemma 2 status

EmbeddingGemma 2 is implemented for text and, with its optional projector,
ordered image/audio/video content. It exposes 768-dimensional vectors and
renormalized 512/256/128-dimensional Matryoshka outputs within an 8192-token
window. The six execution paths are `cpu`, `mlx`, `cuda`, `ggml_cpu`,
`ggml_metal` and `ggml_cuda`; Vulkan embeddings are unavailable.
Use [`config/embeddinggemma2.json`](../config/embeddinggemma2.json) for a
checksum-pinned native CPU text service or add `--mmproj` for media.
This support is newer than `v2026.10.10`; use a build containing the changes.

The [implementation and validation record](#embeddinggemma-2-implementation-and-validation)
below separates independent full-model numerical checks, protocol/lifecycle
checks and paired timings. Current CPU text and GPU multimodal latency parity
with llama.cpp remain unachieved. The latest controlled CPU media timings meet
the original margin in all seven cases in both orders, while original
cross-engine numerical reports remain failed where llama.cpp differs from the
independently checked vectors. Synthetic inputs and STS-B regression do not
establish broad retrieval quality. Historical timings retain their original
precision policies and do not qualify later executor changes.

## Embeddings for personal knowledge

The [personal knowledge store](knowledge-system.md) has BM25 retrieval by default
and can use a separate local sentence encoder for semantic retrieval. Set
`TS_KNOWLEDGE_EMBEDDING_MODEL` to a supported GGUF, with
`TS_KNOWLEDGE_EMBEDDING_BACKEND` and the query/document prefixes required by that
model, before starting TensorAgent or the configured server. This does not require
starting an HTTP `--embeddings` service; the knowledge runtime owns its encoder
separately from the chat model. Reindex existing documents to add or refresh their
vectors. Model and prefix changes retain lexical search but never mix incompatible
vector spaces. Knowledge retrieval added after `v2026.10.10` requires a newer build.

## Models and downloads

| Model | File | Dimensions | Context including special tokens | Pooling |
|---|---|---:|---:|---|
| Snowflake Arctic Embed L v2.0 | `snowflake-arctic-embed-l-v2.0-q8_0.gguf` | 1024 | 8192 | CLS |
| all-MiniLM-L6-v2 | `all-MiniLM-L6-v2-Q8_0.gguf` | 384 | 512 in this GGUF | Mean |
| EmbeddingGemma 2 | `embeddinggemma-2-Q8_0.gguf` or `embeddinggemma-2-BF16.gguf` | 768 (MRL: 512, 256, 128) | 8192 | Mean + learned projection |

Download the exact revisions used for validation:

```bash
mkdir -p models/embeddings
curl -fL --retry 3 -o models/embeddings/snowflake-arctic-embed-l-v2.0-q8_0.gguf \
  https://huggingface.co/fisher046/snowflake-arctic-embed-l-v2.0-Q8_0-GGUF/resolve/2b05c46c74499a1a8e5075cabc6f58490aa6f2c1/snowflake-arctic-embed-l-v2.0-q8_0.gguf
curl -fL --retry 3 -o models/embeddings/all-MiniLM-L6-v2-Q8_0.gguf \
  https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/544f204f2eaa2d71361ffc74d6df7170285b286a/all-MiniLM-L6-v2-Q8_0.gguf
```

Both are Apache-2.0 models. The first download is about 635 MB; MiniLM is about
25 MB. Checksums and GGUF metadata are in the validation manifest
`docs/validation/embeddings-2026-09/models.json` (local validation evidence, not committed).
Supported encoder architecture keys are `bert`, `nomic-bert`, and
`gemma-embedding2`, with their tokenizer and pooling metadata. Rerankers, sparse
embeddings, and multi-vector retrieval are separate features.

For a pinned download and startup recipe in one file, use
[`config/embedding-snowflake.json`](../config/embedding-snowflake.json) or
[`config/embedding-minilm.json`](../config/embedding-minilm.json). The host config
loader downloads the specified model revision and verifies its SHA-256.

## Host an embedding service

For native GGML execution, build the native library as described in
[Development](../DEVELOPMENT.md), then build and start the host. On Apple silicon:

```bash
bash TensorSharp.GGML.Native/build-macos.sh
dotnet build TensorSharp.Server.Host -c Release
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model models/embeddings/snowflake-arctic-embed-l-v2.0-q8_0.gguf \
  --embeddings --backend ggml_metal --embedding-threads 8 \
  --host 127.0.0.1 --port 5000 --no-webui
```

Use `cpu` for 100% pure C# execution without native inference libraries,
`ggml_cpu` for native CPU execution, or `ggml_cuda` with a CUDA-enabled
native build. Only the backends actually benchmarked on available hardware
have measured performance claims; see the validation report
`docs/validation/embeddings-2026-09/README.md` (local validation evidence, not committed).
Direct `cuda` and MLX embedding execution currently support EmbeddingGemma 2;
Vulkan embedding execution is not implemented. `--embedding-context-size N` reduces the per-input limit; zero
uses model metadata.

EmbeddingGemma 2 supports `cpu`, `mlx`, `cuda`, `ggml_cpu`, `ggml_metal`, and
`ggml_cuda`. `cpu` is pure C#; `mlx` and `cuda` execute their own tensor kernels;
the `ggml_*` choices execute the native GGML graph. Its default input window is
8192 tokens, following the published model
contract; the GGUF's 262144-token rotary range does not enlarge that window.
The text model has 512 hidden channels and a learned projection to 768 output
channels. Task prefixes and BOS/EOS participate in the mean pooling.

Download the pinned text model and optional media projector from
[Unsloth's GGUF repository](https://huggingface.co/unsloth/embeddinggemma-2-GGUF):

```bash
curl -fL --retry 3 -o models/embeddinggemma-2-Q8_0.gguf \
  https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/031f0d4b35536f69ab3509d4893c923264fcf253/embeddinggemma-2-Q8_0.gguf
curl -fL --retry 3 -o models/embeddinggemma-2-mmproj-Q8_0.gguf \
  https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/031f0d4b35536f69ab3509d4893c923264fcf253/mmproj-Q8_0.gguf
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model models/embeddinggemma-2-Q8_0.gguf --embeddings --backend ggml_metal \
  --mmproj models/embeddinggemma-2-mmproj-Q8_0.gguf --host 127.0.0.1 --port 5000 --no-webui
```

Omit `--mmproj` for text-only loading. The checksum-verified
[`config/embeddinggemma2.json`](../config/embeddinggemma2.json) provides a native
CPU text service recipe. EmbeddingGemma 2 uses F32 hidden states and F32 CPU
attention accumulation; BF16 model weights are supported. Metal flash
attention stores RMS-normalized K/V in F16 and requests F32 accumulation;
the remaining hidden states retain F32. Native CUDA uses F16 encoder linear
weights for stable, efficient tensor-core multiplication; its raw input,
per-layer input gates, and output projection retain F32. Running the complete model in F16
is inappropriate for its dynamic range.

Use `task: search result | query: ` for search queries and `title: none | text: `
for untitled documents, or `title: {title} | text: {content}` for titled documents.
Code search uses `task: code retrieval | query: `, question answering uses
`task: question answering | query: `, and symmetric similarity uses
`task: sentence similarity | query: ` on both texts. The HTTP API preserves
supplied text; callers choose their task prefix. The knowledge provider chooses
the search/document defaults when prefix environment variables are omitted;
an explicitly empty variable disables the corresponding prefix.

EmbeddingGemma 2 can keep the leading 128, 256, or 512 dimensions; the HTTP
`dimensions` parameter renormalizes the truncated vector. Queries and documents
must use the same dimensions. Follow the
[published model card](https://huggingface.co/google/embeddinggemma-2) for task
prompts and dimension tradeoffs.

The library defaults to the pure C# backend. Select the native CPU backend
explicitly when using the C# API:

| Host `--backend` | Library `EmbeddingModelOptions.Backend` | Execution |
|---|---|---|
| `cpu` | `CPU` (default) | 100% pure C# CPU |
| `ggml_cpu` | `GGML_CPU` | Native GGML CPU |
| `ggml_metal` | `METAL` or `GGML_METAL` | Native GGML Metal |
| `ggml_cuda` | `GGML_CUDA` | Native GGML CUDA |
| `cuda` | `CUDA` | Direct CUDA (EmbeddingGemma 2) |
| `mlx` | `MLX` | MLX (EmbeddingGemma 2) |

To build and run the pure C# CPU host without building native inference libraries:

```bash
dotnet build TensorSharp.Server.Host -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model models/embeddings/all-MiniLM-L6-v2-Q8_0.gguf \
  --embeddings --backend cpu --embedding-threads 8 \
  --host 127.0.0.1 --port 5000 --no-webui
```

`--embedding-threads N` configures CPU execution for both `cpu` and `ggml_cpu`.
On the managed backend, `0` selects four threads; a positive value is the total
number of computation threads, including the calling thread.

The process keeps one encoder resident. Run chat and embedding services on
different ports when an application needs both. With `--embeddings`, POST requests
to the generation routes (`/v1/chat/completions`, `/v1/responses`,
`/v1/systemone`, `/v1/videos/generations`, `/api/generate`, `/api/chat`,
`/api/chat/ollama`, `/api/models/load`, and `/api/image-generate`,
`/api/image-edit` and `/api/video-generate` with their `/stream` forms) return
HTTP 400 `This server hosts an embedding model. Use /v1/embeddings or /api/embed.`
An explicit `--backend` that this machine does not have stops startup with exit
code 2 and `error: model load refused: Backend 'X' is not supported on this
machine.` Set `--host 0.0.0.0` to listen
on network interfaces, and use the deployment's existing TLS and authentication
gateway for public access. `/v1/models`, `/api/tags`, and `/api/show` identify
the hosted model and its embedding capability. Requests name the model's GGUF
basename, as in the examples below.

## OpenAI-compatible API

```bash
curl http://127.0.0.1:5000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"snowflake-arctic-embed-l-v2.0-q8_0","input":["query: find a function that reads files","def read_file(path): return open(path).read()"],"encoding_format":"float"}'
```

The response is an OpenAI `list` of indexed `embedding` objects and a `usage`
object containing `prompt_tokens` and `total_tokens`. Each vector has unit
L2 norm. Input accepts a string, an array of strings, one integer token array,
or an array of integer token arrays. Token arrays must already include the
model's special tokens; IDs belong to this model's tokenizer.

With an EmbeddingGemma 2 projector loaded, both endpoints also accept one
`{"content": [...]}` object or an array of such objects. Content parts preserve
their order and each object produces one vector for the complete sequence:

```json
{
  "model": "embeddinggemma-2-Q8_0",
  "input": [{"content": [
    {"type": "text", "text": "Waterproof running shoes. "},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]}],
  "dimensions": 256
}
```

Media URLs are base64 data URIs. Audio uses
`{"type":"input_audio","input_audio":{"data":"...","format":"wav"}}`
(also `mp3`, `ogg`, or `flac`); video uses
`{"type":"input_video","input_video":{"url":"data:video/mp4;base64,..."}}`
(also WebM). Media decoding/preprocessing and all soft tokens count toward the
same 8192-token window. Oversized media sequences are rejected; the Ollama text
truncation option does not cut media. Body limits also apply to encoded media.
Images use the reference aspect-preserving bicubic preprocessing with a target
budget of 280 soft tokens; video uses 140 per sampled frame, 1 fps, at most 32
frames. Actual token counts depend on image geometry. Audio is decoded to mono
16 kHz and passed through the Gemma 4 audio encoder. Platform media codecs
must be available for video and formats outside the managed audio decoders.

For local files, use the C# API; media towers can be loaded selectively:

```csharp
using var model = EmbeddingGemma2MultimodalEncoder.Load(textGguf, projectorGguf,
    new EmbeddingModelOptions { Backend = "GGML_METAL" },
    new EmbeddingGemma2MultimodalOptions { LoadAudio = false });
var result = await model.EmbedAsync(new[] {
    EmbeddingGemma2InputPart.Text("Waterproof running shoes. "),
    EmbeddingGemma2InputPart.Image("shoe.png"),
});
float[] vector = result.Embeddings[0];
```

The wrapper inserts BOS/EOS and media delimiters once, scales text embedding
rows by sqrt(512), and passes media rows without that extra scale. Use explicit
media parts; literal media delimiters such as `<|image>` and `<|audio>`, and
unmatched `<|image|>`, `<|audio|>`, or `<|video|>` text placeholders are rejected
by this API.

### EmbeddingGemma 2 implementation and validation

The TensorSharp-owned native graph uses 24 bidirectional transformer layers:
five local-attention layers followed by one global layer. Local attention
includes positions up to 512 tokens on either side. Local/global head widths
are 256/512 and KV head counts are 2/1. Q and K have learned RMS normalization;
V has unweighted RMS normalization and attention uses scale 1. Initial input
embeddings produce all per-layer inputs through a single projection, followed
by GELU-gated per-layer residuals. Each layer also has a learned output scalar.
The final RMS-normalized token states are mean-pooled, projected to 768
channels, and L2-normalized.

QKV and FFN gate/up matrices are joined once while loading when their storage
types match. Complete sequences stay in one graph; mixed batches share the
linear operations while each attention segment has independent masks and
positions. Reused shapes retain graph allocations and CPU work plans. Metal
fusion dependencies participate in allocation planning. These changes live in
TensorSharp; upstream ggml sources remain unchanged.

On native GGML CPU, BF16 matrices and the qualified Q8 model's learned linears
are expanded to F32 once at load, while the vocabulary embedding table retains
its original format. This avoids repeated BF16 conversion and Q8 activation
rounding at the cost of additional resident CPU weight storage. The small
output projection is F32 so pooling before projection preserves linearity
without quantizing the pooled activation. Metal retains the model's original
linear weight storage with the hidden-state and attention precision described above.

The pure C# CPU executor keeps the vocabulary compressed and expands encoder
linear matrices to F32, with packed F32 GEMM on AVX2/AVX-512 CPUs. An experimental
packed Q8 path is available with `TENSORSHARP_EMBEDDING_Q8_AVX=1`; it retains F32
activation scales but has not met the full-model performance and accuracy gates,
so it is disabled by default. Raw input and output projections always retain
F32 arithmetic. The executor fuses QKV and
gate/up projections and uses its own persistent worker pool, cached rotary
tables, and SIMD attention.

An additional CPU experiment uses a separately built TensorSharp-owned library
for Float32 attention (`TENSORSHARP_EMBEDDING_NATIVE_ATTENTION=1`) and Q8-weight,
Q16-activation VNNI projections (both `TENSORSHARP_EMBEDDING_Q16_AVX512=1` and
`TENSORSHARP_EMBEDDING_NATIVE_Q16=1`). The graph and worker pool remain managed;
this native arithmetic is distinct from `GGML_CPU`. PLE, raw input gates and
output projections retain Float32. The combined route passed 24 arithmetic
cases and 9 independent NumPy model goldens with zero fallback, but the actual
CPU HTTP latency bound passed only 1/7 cases, so it stays opt-in. See
`TensorSharp.Cpu.Native/README.md` for explicit library packaging, feature guards,
actual math counters and validation limits. Ordinary managed builds require no
native compiler or CPU library.

MLX and direct CUDA have separate tensor executors with
shared batch projections, sequence-isolated attention, and device mean
pooling. Direct CUDA additionally has a TensorSharp-owned streaming attention
kernel for bidirectional GQA and symmetric windows. These executors retain
F32 hidden states. MLX keeps Q8 linear weights compressed for short inputs,
caches F32 projections for larger prefills, and caches rotary tables and common
attention masks. Its Metal matrix multiplication can use the runtime's TF32
hardware precision without reducing the activation range. Long inputs use
query tiles and local key halos to avoid quadratic attention intermediates.
MLX can narrow statically bounded, RMS-normalized attention operands and
QKV/gate-up projections for 256–768-row prefills to F16, then restore F32
outputs. The projection bound includes F16 weight rounding and every partial
dot product; unsafe models retain F32. Residual states, raw media inputs,
per-layer input projections, attention output projections, and FFN down
projections remain F32. Short ragged batches pack independent padded sequences
into one attention call, excluding padding when restoring the original rows.
Direct CUDA expands linear weights to F32,
which increases resident weight memory compared with compressed GGML. Its current
learned projections and device mean pooling use per-call cuBLAS pedantic F32
operands, accumulation, and output. This avoids the TF32 rounding that failed
sensitive short-input batch/single checks, without changing global cuBLAS math mode.
The explicit `CudaEmbeddingOps.TryLinearTf32` API remains available for diagnostics;
audio also uses a scoped pedantic F32 route described below.

The audio frontend is verified against the published Gemma 4 feature
extractor: semicausal 160-sample left padding, periodic Hann-320, FFT-512,
HTK 128-bin magnitude mel filters, and `log(mel + 0.001)`. Analysis/soft rows
introduced only by waveform padding are excluded from embedding pooling.
Malformed WAV chunks and non-finite PCM are rejected before inference.
Sparse triangular mel filters and cached FFT stage twiddles avoid redundant
work while preserving the tested feature bytes. Platform audio decoders must
return finite PCM with a positive sample rate and consistent channel lengths;
downmixing uses a wide accumulator. Non-16-kHz input uses the time-aligned
Hann-windowed sinc resampler described below, qualified against the selected
torchaudio policy at eight source rates. The canonical independent audio checks
use 16-kHz PCM; decoder/resampler policies across HF libraries can differ.

Managed CPU, MLX, and direct CUDA reject unsupported partial rotary embeddings
or rotary scaling instead of silently ignoring that metadata. Their key/value
head widths must match. Odd sliding-window widths use `floor(width / 2)`,
consistent with llama.cpp. Select a GGML executor for its supported rotary variants.
Direct execution preserves self-only attention when that radius is zero.
Short ragged MLX attention retains F32 operands to preserve batch/single
agreement; narrow-window MLX variants use independent sequence graphs when
needed. Narrow-window direct CUDA variants also use independent graphs.
The published 1024-wide model keeps batch packing and now uses the same pedantic
F32 learned projections and pooling as other direct CUDA variants. Earlier MLX
and direct CUDA latency measurements predate these precision fixes and do not
qualify their current performance. GGML CUDA and GGML Metal execution are
unaffected by these direct-executor policies.

Implementation references were inspected at these revisions:

| Reference | Revision | Relevant source |
|---|---|---|
| llama.cpp | `4413a192410fad1f054e9216114087a3e01527a0` | `src/models/gemma-embedding2.cpp`, `tools/server/server-context.cpp`, Gemma 4 media encoders |
| vLLM | `5555a2d8f7116b2b4877f66516f4128176cf722a` | `vllm/model_executor/models/embedding_gemma2.py`, `tests/models/multimodal/pooling/test_embedding_gemma2.py` |
| SGLang | `ea13359da99e869534d28ead3989c6f9293c01a0` | `python/sglang/srt/models/embedding_gemma2.py` and its multimodal processor |

The local validation tools are
[`eng/validation/embeddinggemma2-http.py`](../eng/validation/embeddinggemma2-http.py),
[`EmbeddingGemma2Probe`](../eng/validation/EmbeddingGemma2Probe/Program.cs),
[`generate-embeddinggemma2-media.py`](../eng/validation/generate-embeddinggemma2-media.py),
and the independent float32
[`NumPy oracle`](../eng/embeddinggemma2-reference.py). Unit/model tests are in
`InferenceWeb.Tests/EmbeddingGemma2Tests.cs`; the real-model lane requires
`TENSORSHARP_EMBEDDINGGEMMA2_MODEL` and the requested GGML backend. Missing
models/devices are explicitly skipped, not accepted as successful validation.
Generated evidence stays in ignored `docs/validation/embeddinggemma2/` and
`docs/validation/embeddinggemma2-reference/`. The GGML dependency used for these
runs is unchanged revision `3612c0af93f2cfc8f048295300a61187723b12ae`.

The final local Q8 validation on an Apple M5 Pro with 48 GiB used sequential
HTTP engine runs, at least 20 measured samples, and identical model/input bytes. All
16 native Metal scenarios were faster than llama.cpp (latency ratios
0.748–0.945). The final MLX run met a 5% on-par margin in all 16 scenarios:
13 were faster, the mixed 739-token batch was 3.6% slower, and 8192 tokens
were 33.6% faster. Earlier failed performance reports remain in the evidence
directories. These are steady-state HTTP measurements; model loading is
excluded. Independent NumPy checks include eight exact token sequences and
the full 8192-token context. Both backends also passed the first 200 human-scored
STS-B development pairs at 768, 512, 256, and 128 dimensions; this is a subset
regression rather than a complete retrieval benchmark.

Seven synthetic text/image/audio/video/mixed cases completed with clean
process exits, and both embedding protocols passed media parsing, invalid-input,
temporary-file cleanup, and graceful server shutdown checks. Controlled image
comparisons use verified identical PNG pixels. Video comparisons match source
clips, sampling/token geometry and the explicit `Video:` prefix that llama.cpp
inserts; native decoder RGB bytes can differ by one level across platforms. Image/video cosine
similarity exceeded 0.99993. Default preprocessing differs: TensorSharp follows
the HF 280-token floor/stretch policy, while llama.cpp defaults to a 70–1120-token
ceil/letterbox policy and adds a video prefix. Controlled fixtures do not establish
default-pipeline equivalence or multimodal performance parity.

The validated Q8 vision geometry selects scoped F32 Metal projections for
patch embedding, all sixteen transformer blocks, and the final projection.
The native entry points preserve the ordinary graph path for other models.
An owned F32 attention kernel covers sequences up to 128 rows; larger vision
sequences retain upstream attention. Fresh exact-pixel raw-row checks pass at
96×96, 192×144, and 768×768, with maximum errors 0.0000177, 0.0000200, and
0.00579 respectively. Independent projection and attention checks include
values above the F16 range, offset views, and output canaries. The matching
desktop library exports all 299 symbols retained by the mobile manifest;
this symbol check does not qualify a physical iOS run.

Audio uses the published HF `log(mel + 0.001)` frontend. The pinned HF mask
passes a left-window argument of 12 and admits distances strictly below 12;
the maximum admitted past distance is 11. Earlier audio comparisons used an
incorrect inclusive distance of 12 in both the implementation and NumPy
reference. Those reports are retained but superseded as audio quality evidence.
The corrected complete audio tower passes the original raw-row error limit of
0.01 on CPU (maximum error 0.000103), MLX (0.00684), and the precise Metal F32
projection path with owned attention/depthwise kernels (0.0000632). Scoped pedantic CUDA projections also pass this
limit (0.000126). GGML CPU passes (0.0000668), and scoped pedantic GGML CUDA
SSCP/conformer graphs pass (0.00893). Earlier ordinary direct CUDA (0.0388),
ordinary GGML CUDA (0.01174), and unqualified GGML CUDA graph (0.02227)
results remain failed reports. The corrected default-HF complete pooled audio embedding
passes on Metal and direct CUDA (cosine above 0.9999997, maximum component
error below 0.0001). Scoped GGML CUDA graphs pass above 0.999998 with maximum
component error below 0.000269. Their recorded media HTTP checks include both
protocols, malformed input handling, temporary-file cleanup, and clean shutdown.
The MLX encoder retains original Q8 projector matrices with F32 activations
and output. The EmbeddingGemma 2 wrapper requests per-instance validated F32
precision: Metal uses complete SSCP/conformer graphs and full-precision MPS
projections; GGML CUDA uses scoped complete graphs with private pedantic F32
products, while direct CUDA uses per-call pedantic F32 projections.
The graph descriptor's `RequireFloat32Precision` policy scopes learned projection
and SSCP precision; it does not promise exact arithmetic for every ordinary
ggml operation. Without that policy, GGML CUDA audio graphs still compute their
F32 matrix products (projections with F32 weights, relative attention, the depthwise
convolution and the SSCP convolutions) on TensorSharp's pedantic cuBLAS handle:
ggml-cuda creates its own cuBLAS handles with TF32 tensor-op math, so an F32
`ggml_mul_mat` of more than three rows could round its operands to TF32 despite
its F32 precision request. The conformer fixture's 25- and 96-row blocks were
5.8e-4 and 7.1e-4 from the double reference that way, and are 4.3e-7 and 7.6e-7 now.
On an RTX 5090, the 12-block conformer at EmbeddingGemma 2's width is also faster
this way: 9.2 instead of 10.3 ms for 188 frames, 22.7 instead of 28.7 ms for 750.
Q8_0 projections keep ggml's integer kernels;
`TS_GEMMA4_AUDIO_GGML_CUDA_PEDANTIC=0` keeps ggml's TF32 products. The scoped GGML CUDA graph policy passes fresh no-override raw tower, complete
HF embedding, and both-protocol HTTP checks. It selects a private pedantic
cuBLAS handle and does not change shared backend math modes. This
request does not change global backend math modes or the existing two-argument
Gemma chat audio constructor. Explicit process-start values of `0` for
`TS_GEMMA4_AUDIO_SSCP_GRAPH`, `TS_GEMMA4_AUDIO_BLOCK_GRAPH`,
`TS_GEMMA4_AUDIO_METAL_MPS_MM`, `TS_GEMMA4_AUDIO_CUDA_PRECISE_SSCP`, or
`TS_GEMMA4_AUDIO_CUDA_PRECISE_LINEAR`, or `TS_GEMMA4_AUDIO_GGML_CUDA_PEDANTIC` provide diagnostic opt-outs. Optional
owned Metal tile projection remains an opt-in diagnostic. The complete validated
12-block Metal audio conformer selects owned F32 relative attention and causal
depthwise kernels by default. Their measured request latency improves about 7.6%
in both orders of a matched on/off comparison, with unchanged quality gates.
`TS_GEMMA4_AUDIO_METAL_FUSED_OPS=0` selects the earlier precise graph for diagnostics;
other geometries retain the explicit experiment. Slower Metal and MLX
SIMD matrix experiments were rejected and retained only as validation evidence. The AudioProbe
`--require-embedding-default true` rejects all audio precision/graph
overrides and verifies the scoped default directly.
The one-row projection diagnostic `TS_GEMMA4_AUDIO_PRECISE_ROWS=1` has bounded
graph/context allocation and independent 96/128-row composition checks.

For explicit llama.cpp comparisons, `Gemma4AudioLogPolicy.LlamaClampFloor`
or the validation environment `TS_EMBEDDING_AUDIO_LOG_POLICY=llama_clamp`
selects `log(max(mel, 0.001))`. The default remains
`HuggingFaceAddFloor`; controlled clamp results do not establish default
frontend equivalence. A required fixture executes the pinned HF scalar mask
definition directly to guard its boundary semantics. Relative attention and
causal depthwise convolution have independent scalar/double checks on native
CPU, Metal, CUDA, and direct MLX/CUDA, including values above the F16 range.
Corrected controlled seven-case media pairs still show performance gaps.
The latest controlled Metal audio/vision run passes all seven numerical
comparisons, while meeting the 5% latency margin in only the text case.
Its media latency ratios against llama.cpp range from 1.139 to 1.948;
audio takes 26.6 ms versus 23.4 ms. The fresh default-HF run separately passes
seven cases through both APIs, 22 malformed requests, temporary-file cleanup,
clean shutdown, and the independent complete pooled audio oracle
(cosine 0.999999812, maximum component error 0.0000847).
Both controlled benchmark servers exit successfully, while the driver correctly
fails its overall performance gate. Earlier Metal and MLX experiments retain their
own failed performance reports. These comparisons deliberately use llama's clamped mel policy;
default-HF numerical evidence is separate. Neither text results nor component
tests establish all-six-backend multimodal quality or performance parity.

An earlier A40 direct-CUDA text run passes eight exact NumPy sequences and the
8192-token independent oracle (cosine above 0.9999986 and maximum component
error below 0.000175). Only six of sixteen timed scenarios met the 5% latency
margin against the original fast llama CUDA baseline. That reference also
fails the unchanged single-versus-batch cosine gate (0.999726 against 0.9999)
and has a larger 8192-token error against the independent oracle. The explicit
`--measure-reference-after-quality-failure` diagnostic retains these failed
strict checks while collecting all paired timings; it does not lower candidate
or reference numerical thresholds and exits unsuccessfully.

An opt-in direct CUDA materialized F32 attention plan uses per-call pedantic
cuBLAS products and an encoder-owned score buffer bounded to 512 MiB. Its bounded
local-window extension caches at most four F32 masks and 64 MiB, admits symmetric
distances through 512, and retains streaming attention below 256 tokens and for
local windows longer than 2048 tokens. Twenty-two independent actual CUDA
operator checks pass, including window boundaries, packed/equal batches,
unrounded double references, and mask-cache readback/disposal lifetimes. The
stronger eight-sequence and 8192-token NumPy gates also pass, with minimum cosine
0.99999799 and maximum component error below 0.000239. The latest complete
sixteen-case pair meets the 5% latency margin in ten cases; 8192 tokens takes
466.4 ms versus 672.8 ms. Short input, 1023/1024/1025/2048-token boundaries, and
the mixed-window batch remain slower, so this experimental route remains
disabled by default. Candidate batch consistency passes above 0.99999975;
the unchanged reference consistency and 8192-token comparison failures keep
the overall benchmark unsuccessful. Earlier failed strategies remain recorded.

The fresh default direct CPU controlled media pair meets the 5% latency margin
in six of seven cases, with latency ratios from 0.831 to 0.980 for those cases.
Audio remains slower at 270.5 ms versus 108.9 ms (ratio 2.483). All seven token
counts match and both servers exit successfully, but only three cross-engine
numerical comparisons pass. The complete independent clamped-audio model passes
the CPU candidate at cosine 0.999999999997 and maximum error 0.000000311;
the original fast llama CPU reference fails at 0.999523970 and 0.00373045.
The overall report remains unsuccessful. Main-server RSS after media is
7186.7 MiB (7.02 GiB) for TensorSharp and 2939.8 MiB for llama, including retained scratch
and process state. This controlled result uses the explicit clamp frontend and
video prefix; separate default-HF/API evidence defines the default behavior.

The qualified EmbeddingGemma 2 GGML CPU geometry expands learned projections to
F32. This fixes a complete audio embedding failure that was hidden by a passing
raw audio tower: the earlier default cosine was 0.99983029 with maximum component
error 0.002502, while the explicit F32 diagnostic and fixed default pass above
0.99999994 with maximum error below 0.000041. Fresh no-override checks pass the
eight text goldens, the 8192-token golden, canonical raw and pooled HF audio, and
seven media cases through both protocols with clean shutdown. Expanded weights
cost additional memory. The fresh eight-thread default F32 text pair meets the
5% latency margin in none of seven cases; ratios against the original fast
llama CPU baseline range from 1.316 to 15.738. The controlled media pair matches
all seven token counts, passes three cross-engine numerical comparisons and
one latency comparison, and takes 259.6 ms versus 110.4 ms for audio. Both
servers exit successfully in each pair, while both overall reports fail.
The independent complete clamped-audio comparison passes TensorSharp
(cosine 0.999999946, maximum error 0.0000392), but fails that fast llama CPU
reference (0.999523970, 0.00373045), without relaxing the original gates.

Main-server resident memory after the text run is 1214.5 MiB for TensorSharp
and 2202.0 MiB for llama; after the media run it is 4124.3 and 2928.3 MiB.
These whole-process observations include scratch and retained state, exclude
child processes and GPU memory, and preserve llama's original context/slot
configuration. They are not isolated model-weight memory measurements.
`TSG_EMBEDDING_F32_LINEAR=0` selects the earlier
approximation for diagnostics; other architectures and GPU backends retain
their own policies. The [audio validation guide](../eng/validation/EmbeddingGemma2AudioProbe/README.md)
records the failed default, explicit diagnostic, and fixed-default procedure.

Reproduce the numerical checks
with [`embeddinggemma2-audio-embedding-reference.py`](../eng/validation/embeddinggemma2-audio-embedding-reference.py)
and [`embeddinggemma2-compare-audio.py`](../eng/validation/embeddinggemma2-compare-audio.py),
the source-derived mask fixture with [`generate-gemma4-audio-mask.py`](../eng/validation/generate-gemma4-audio-mask.py),
the sequential media comparison with [`embeddinggemma2-media-pair.py`](../eng/validation/embeddinggemma2-media-pair.py),
and the bounds audit with [`embeddinggemma2-attention-bounds.py`](../eng/validation/embeddinggemma2-attention-bounds.py).

### Validation follow-up: 2026-10-14

Fresh local Q8 HTTP comparisons on the M5 Pro pass all sixteen text scenarios,
including the full 8192-token input, with unchanged ggml
`3612c0af93f2cfc8f048295300a61187723b12ae`. The rebuilt local llama.cpp reference
is `62df1815bc053066761f4eac58e288f23778e164`; the further implementation audit
also inspected vLLM `efad400092bac444608ffdac937431239696a06f` and SGLang
`ed40c7fc66e74095d215c690c2ff59564a415bc0`.
The final immutable Metal builds pass both engine orders with latency ratios
0.734–0.955, and all strict consistency, token-accounting,
and cross-engine gates pass. Binary hashes remain unchanged across each pair.
Separate independent NumPy comparisons pass eight
text/token inputs and 8192 tokens (maximum component errors 0.000271 and
0.000199). Timings exclude startup and include HTTP/tokenization/serialization.

The complete 1500-pair STS-B development split passes the 0.001 maximum
Spearman-drop gate at all four output dimensions. At 768 dimensions TensorSharp
scores 0.886722506 and llama.cpp 0.886685768. This is the development split,
not the test split or MTEB. The runner now reads the original TSV's literal
quotation marks correctly and requires all 1500 complete scored rows before
launching either engine; earlier 200-pair measurements did not expose that parser bug.

Fresh default-HF Metal media checks pass seven cases through both APIs,
22 malformed requests, cleanup, and successful process shutdown. The complete
pooled audio oracle passes at cosine 0.9999998124 and maximum component error
0.0000847. The optimized frontend is bit-identical in eight waveform/policy
cases and 1.40–4.37 times faster in a separate standalone measurement; that
component speedup does not establish media HTTP performance parity.

On the remote A40, native GGML CUDA's first sixteen-case text comparison has
latency ratios 0.422–0.640; 8192 tokens takes 334.7 ms versus llama.cpp's
547.8 ms. The final frozen reverse-order run also wins all sixteen latency
cases (ratios 0.372–0.719), with unchanged runtime hashes; its 8192-token case
takes 340.9 ms versus 539.7 ms. TensorSharp passes its strict HTTP checks and the independent eight
and 8192-token NumPy gates. The latter has cosine 0.99993484 and maximum
component error 0.001639 against the unchanged 0.9999/0.002 bounds. llama.cpp
fails both its strict batch/duplicate consistency gates and those independent
checks, so the paired report remains unsuccessful and its timings are diagnostic.
The complete remote STS-B development split passes all four dimension gates;
768-dimensional Spearman is 0.886670442 versus llama.cpp's 0.886554762.
The VM exposes 96 logical CPUs but has a 7.65-core cgroup quota; both engines
use eight CPU threads.
Final direct CUDA pedantic-F32 checks pass the stricter 0.99999/0.0005 NumPy
gates, with maximum component errors 0.000000913 for eight cases and
0.00000143 for 8192 tokens. Thirty-eight actual CUDA operator/model/window
checks pass with zero skips. These direct-CUDA numerical results do not
establish its current latency parity.

Remote testing found that the committed PTX targeted SM120 and could not load
on the A40 without nvcc. The five portable modules are now regenerated from
TensorSharp sources with CUDA 12.8 and `compute_61`. Committed-baseline refreshes
force the portable target, recompile despite stale architecture stamps, and reject
newer explicit targets. No-nvcc builds select the portable modules and replace
newer stale output/publish files. Ordinary GPU-specific builds still select
their detected architecture. (That compute_61 / CUDA 12.8 baseline was later
retired: the committed modules now target compute_75 and are refreshed with
CUDA 13.0; see [Development](../DEVELOPMENT.md).) The optional
normalized CUDA projection plan also now uploads Half weights through an owned
host mirror and preserves surrounding storage when writing partial views.

Native CPU performance remains below the fast llama.cpp baseline in most
cases even with optional BLAS, so that route remains opt-in. With the 32-row
threshold, the fresh short case improves from the earlier BLAS experiment's
76.3 ms to 35.6 ms; only one of seven cases meets the latency margin. CPU
NumPy goldens still pass. Multimodal latency parity remains unmet, and the
explicit TF32 attention experiment stays disabled by default: on large query/key
exponents its error (6.3e-5 on an RTX 5090, up to 8.1e-5 by exact TF32 emulation)
exceeds the 3e-5 pedantic F32 bound. That case's test gates on TF32's own
rounding bound (1.2e-4, which BF16 operands fail) and reports the excess.
These failures are retained in the reports.

An optional Linux native CPU projection route uses an unmodified system LP64
pthread OpenBLAS through TensorSharp-owned graph callbacks. Build the ordinary
native library, install that BLAS provider, and start the embedding process with:

```bash
OPENBLAS_NUM_THREADS=1 TSG_EMBEDDING_LINUX_BLAS=1 \
  TSG_EMBEDDING_BLAS_LIBRARY=/absolute/path/to/libopenblas.so.0 \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /path/to/embeddinggemma-2-Q8_0.gguf --embeddings \
  --backend ggml_cpu --embedding-threads 8 --host 127.0.0.1 --port 5000 --no-webui
```

The optional text route retains F32 matrices, activations and products and partitions projections
with at least 32 token rows over at most eight existing ggml workers. Smaller
projections retain the ordinary precise path because the measured BLAS overhead
regressed short requests. At 128 or more rows it partitions token rows; very
wide projections (at least 8192 output channels, including PLE) retain channel
partitioning below 1024 rows after exact-shape measurements favored it.
It never changes library thread
settings globally. Unavailable/incompatible BLAS keeps the ordinary path;
the `EMBEDDING_LINUX_BLAS` disposal record identifies the actual provider,
successful GEMM calls and rejections. A changed library thread policy after
graph construction takes a precise scalar fallback for each output partition.
The optional graph regression is `GgmlOpsEmbeddingCpuMatmulTest --linux-blas`
with the same environment; it requires an available provider and checks
actual graph execution plus fallback after closing the provider lease.

The CPU/media continuation adds optional tiled F32 native attention with
`TSG_EMBEDDING_CPU_ATTENTION=1` alongside the BLAS settings above. It computes
QK and probability/value products over the existing CPU workers, retains
strided grouped heads and independent sequences, and applies both the explicit
mask and the symmetric local window. Scratch is shared across serial layers
(16 MiB at 8192 keys and eight workers). A provider closure or policy rejection
uses a precise disjoint scalar fallback. Scratch growth commits its new stride
only after successful allocation, preserving the old stride if allocation throws.
Growth and graph reuse are tested; allocation failure injection is not covered.
Supported Linux x86 CPUs use AVX512F/F16C/FMA softmax with F32 mask/max/exp
operations and F64 denominator reduction; other CPUs use the scalar path.
`TSG_EMBEDDING_CPU_ATTENTION_SIMD=0` selects a same-binary scalar ablation.
On the eight-thread remote host, an unprofiled 8192-token ABBA comparison
reduces median latency by 17.83% and 18.13% in its two orders (W2/R5). All four
runs pass the original independent 0.99999 cosine / 0.0005 component gates.
This improves the optional route; six of seven CPU text HTTP cases still miss
the 5% latency margin against the strongest measured Q8 llama.cpp configuration.
The native attention test includes an independent dense probability sweep,
F16/F32 masks, local-window boundaries, grouped heads, batch-broadcast masks,
fully masked rows, workspace growth and persistent graph-plan reuse. Linux
qualification must run `GgmlOpsEmbeddingCpuAttentionTest --linux-blas --require-simd`;
ARM scalar execution does not qualify the AVX512 lane.

Two diagnostic selectors provide attribution without changing upstream ggml:
`TSG_EMBEDDING_CPU_PROFILE=1` executes persistent category subgraphs, which
changes fusion/scheduling and must not be used for paired latency claims.
`eng/validation/embeddinggemma2-cpu-phase-summary.py` validates its accounting.
`TSG_EMBEDDING_CPU_ATTENTION_PROFILE=1` reports QK, softmax and PV intervals
summed across workers; these intervals overlap in wall time. Ordinary benchmark
runs leave both selectors unset and bind their runtime hashes before and after.

The continuation also fixes CPU vision precision: fused blocks retain F32 K/V
and use a TensorSharp-owned F32 tanh-GELU product instead of the upstream CPU's
F16 lookup. CPU blocks borrow read-only weights only for synchronous execution,
avoiding repeated copies without retaining a pointer cache. Independent raw
vision rows at 96×96, 192×144 and 768×768 now pass the unchanged 0.9999/0.01
gates; the earlier 96×96 maximum error of 0.01292 is retained as a failure.
Observed local small-image tower medians are 44.59 before and 30.81 ms after.
These sequential raw-tower runs use unequal warmup/repeat counts and exclude
decoding, the text tower and HTTP; they do not qualify end-to-end parity.

The validated EmbeddingGemma 2 constructor also selects a typed precise CUDA
vision-block entry point. Owned F32 projections and attention execute through a
private pedantic backend, with synchronization before releasing graph leaves
and the wrapper. Required failures propagate instead of silently falling back;
ordinary Gemma chat constructors retain their existing CUDA policy. Actual A40
raw-tower checks at the same three sizes pass the unchanged 0.9999/0.01 gates,
with maximum errors 0.00318, 0.00352 and 0.00160. The previous 768×768 error of
0.05052 remains recorded as a failure. Whole-tower checks include the ordinary
patch and final projections; the block test alone does not qualify the tower.
For the validated 768-wide, twelve-head CUDA tower, immutable block matrices
use the existing device-copy cache when its budget permits. This retains
576 MiB; budget refusal streams the same F32 matrices, and budget zero keeps
its existing unlimited meaning. Actual A40 tests cover warm reuse, stable
resident bytes, one-byte-budget fallback, disposal and same-process reload,
with zero skips. A small-image raw-tower ABBA comparison improves from
85.39–96.56 ms to 18.07–18.50 ms with identical outputs. These component
measurements exclude decoding, text encoding and HTTP.
An owned F32 head-64 attention kernel shares K/V tiles across queries on the
validated CUDA route. Eight actual-device operator cases and the complete
raw-tower checks pass; its 768-pixel tower improves by about 14–15% in both
orders relative to the cached-weight route. Ordinary chat constructors retain
their execution policy. Allocation/upload failure injection is not covered.

Qualified EmbeddingGemma 2 GGML CPU audio now defaults to the complete SSCP and
conformer graphs. The ordinary two-argument Gemma chat constructor retains its
existing policy. Same-binary eight-thread remote raw-tower tests in both
configuration orders reduce median latency by 21.0–21.1% (W3/R20, at least one second).
Observed local medians are 87.72 before and 56.63 ms after with unequal repeat
counts; a separate managed CPU change records 108.18 and 99.02 ms. These
component timings exclude the frontend, text encoder and HTTP.
On builds without the upstream AVX tiled GEMM, the optional Linux BLAS provider
lets the validated CPU audio graph use owned F32 projections for the classic tower's 1024-to-1024,
1024-to-4096 and 4096-to-1024 matrices. The graph owns its provider through
synchronous completion; other shapes, convolution and attention retain their
existing route. The same provider qualification and precise fallback apply.
Nine actual Linux operator graphs and eight model/fixture/lifetime tests pass
with zero skips. The native raw-tower oracle, complete pooled embeddings,
44.1/48-kHz source inputs and both HTTP protocols also pass their original gates.
The prior Linux build records a 7.7–8.6% raw audio-tower reduction in both
orders against its default F32 route (W3/R20, at least one second). This
component measurement excludes the frontend, text encoder and HTTP. With
upstream tiled GEMM enabled, a same-binary ABBA screen instead measures
48.40–50.00 ms upstream versus 78.03–78.19 ms through BLAS. The validated
audio graph therefore prefers upstream when its compiled LLAMAFILE and AVX
capabilities are present; OFF builds retain the optional provider. These
measurements apply to the test VM. Local ARM policy/fallback checks do not
qualify the Linux provider.
Complete canonical audio embeddings and independently resampled 44.1/48 kHz
inputs pass their unchanged numerical gates. Non-16 kHz decoding now uses a
time-aligned bandlimited Hann-windowed sinc resampler (width 64, rolloff 0.99),
qualified against 21 executed primary torchaudio cases across eight rates.
Native 16 kHz PCM is unchanged. This specifies the selected resampling policy;
HF codecs, librosa/soxr and torchcodec do not all produce identical waveforms.

New Linux x86 native builds enable unchanged upstream ggml's tiled F32 GEMM
(`GGML_LLAMAFILE=ON`). Explicit settings and existing CMake caches are respected;
reconfigure an existing OFF build with `-DGGML_LLAMAFILE=ON` to use it. The
validated EmbeddingGemma 2 text geometry selects that path through 512 token
rows, with its wide per-layer projection crossing to optional BLAS above 256
rows. Larger projections retain the existing BLAS policy. This selection also
requires an AVX-enabled upstream build. Apple, Windows and other architectures
keep their existing build defaults. No upstream source is patched.

`--embedding-threads` configures the text encoder. The per-op GGML media CPU
pool is separate; set `TS_GGML_CPU_THREADS=8` at process start for an explicit
eight-worker media comparison. Validation records this thread policy separately
from arithmetic selectors. On Linux its automatic pool count floors the CFS
quota, which gives seven workers on the 7.65-core test VM; on Apple it defaults
to physical cores. No inference call changes the process environment or a
shared library's thread settings.

Fresh continuation evidence is in ignored
`docs/validation/embeddinggemma2-cpu-media-20261014/`. The latest local v7 native-only follow-up
passes the default-HF HTTP checks: all seven modality cases through both APIs, 22 malformed
requests, MRL/base64, cleanup and clean shutdown. It passes nine targeted CPU
tests, with one additional Linux-only test skipped on Apple and four actual Metal tests, plus
independent raw and complete pooled audio checks. The previous frozen v6 broad
CPU suite executes 308 passing tests and skips 59 unavailable device/model
scenarios; its separate actual Metal suite executes six tests with zero skips.
Six independent v6 CPU/Metal raw vision checks also pass unchanged numerical
gates. These repeated lanes are separate coverage, not additive test counts.
The complete 1500-pair CPU STS-B development regression passes at 768, 512,
256 and 128 dimensions with matching token counts, clean server exits and
stable binary hashes. At 768 dimensions it scores 0.88668819 versus llama.cpp's
0.88661665; the maximum allowed drop remains 0.001.
The final frozen controlled eight-worker CPU pair meets the 5% latency margin
in six of seven cases in both engine orders; audio is 60.4% and 16.8% slower.
Four original cross-engine quality comparisons fail, so both combined reports
remain failed. The CUDA pair meets the margin for text in both orders and audio
in the forward order only; images, video and mixed inputs remain slower. Independent complete
image, audio, video, mixed and all-modality references pass for TensorSharp;
the original llama outputs fail several of those numerical gates. The composite
reference uses four fresh video-frame vision forwards and three complete F32
text forwards, with exact Linux decoded pixels. The stricter independent
macOS FFmpeg decoder-byte proof remains unqualified because its codec rounds some
channels one level differently. These controlled checks do not qualify broad
multimodal retrieval quality or default HF/llama pipeline equivalence.
A separate complete independent default-HF reference now passes all seven
cases on CPU and CUDA through both APIs at the unchanged 0.9999/0.002 limits.
Six fresh vision forwards and seven complete F32 text forwards include the
learned projection, independently checked decoded pixels and resize geometry,
and exact frozen input/runtime/source identities. Maximum embedding error is
0.00004186 on CPU and 0.00038354 on CUDA. Independent HF audio learned rows are
reused with exact hashes; they are not counted as fresh audio model forwards.
These opaque synthetic cases provide numerical regression coverage, with
observed tokenizer IDs and a bounded PIL float resize reference. They do not
qualify a retrieval leaderboard or HF/llama default pipeline equivalence.
The controlled Metal pair
passes all seven numerical comparisons, but six media latency comparisons
still miss the 5% margin. Accurate scalar-Q8 and two owned float SIMD matrix
experiments were measured at the actual projection shapes and rejected as
slower than F32 MPS; a previously removed mixed-half route remains rejected.
CPU and multimodal latency parity must not be inferred from component wins.

The qualified Linux x64 CPU path distributes complete Q/K/V normalization rows
across seven or eight workers for sequences of at least eight tokens. It keeps
the original F32 arithmetic; other worker counts and geometries retain upstream
normalization. The eight-input and 8192-token outputs are byte-identical to the
previous build at both worker counts with the explicit optional CPU policy.
Separate checks without BLAS or attention overrides also pass the original
strict numerical limits at both worker counts. Fresh complete 1500-pair STS-B checks pass
all four Matryoshka dimensions against llama.cpp, and all 28 complete independent
default-HF protocol comparisons pass their original numerical limits.

In matched eight-worker HTTP ABBA runs with the explicit optional BLAS and tiled
attention policy, five of seven text shapes improve in both orders. Medium,
long and batch8-medium inputs improve by 4.2–10.5%, 7.7–9.2% and about 6.5%,
respectively. Reverse-order batch8-short and batch32-short regress by 1.7% and
4.8%; all cases stay within the original 5% latency margin. This compares the
previous and new TensorSharp builds, not llama.cpp, and does not establish full
CPU latency parity. The service timings above also use the explicit optional
CPU compute policy; default-HF describes preprocessing, not compute overrides.

The same exact-arithmetic Q/K/V normalization path is also validated on macOS
ARM for seven and eight native CPU workers, starting at eight token rows.
All eighteen independent text cases, including 8192 tokens, and twenty-eight
default-HF multimodal API vectors are bit-identical to the prior build.
Both-order HTTP comparisons pass the original 5% latency margin for all seven
text shapes at both worker counts. Batch8-medium improves by about 4.7% in all
four orders; smaller gains vary by shape. These measurements compare two
TensorSharp builds and do not establish latency parity with llama.cpp.

A separate current Mac CPU reference passes all seven default-HF cases through
both APIs at the original 0.9999 cosine and 0.002 absolute-error limits. Six
vision forwards and seven complete text forwards execute fresh; preserved HF
audio rows are reused. Maximum embedding error is 0.00004449. A separately
implemented Apple decoder application matches all four production video frames
byte for byte, and all six image/video resize inputs pass the original pixel
limit. This validates the tested Mac codec path through independent learned
math; the decoder applications share the Apple codec. The earlier FFmpeg
equality failure remains recorded, and this makes no latency or broad retrieval
claim.

The validated single-device GGML CUDA vision path can execute all sixteen
blocks in one graph when its immutable matrix cache fits. Partial cache budgets
retain the per-block path. Four forced raw vision checks and all 28 complete
independent default-HF protocol comparisons pass their original limits.
An actual CUDA trace reduces the selected 768-pixel activation round trips
from 216 MiB to 13.5 MiB, excluding patch projection and weights.

Against the previous TensorSharp build, raw vision latency improves by 35–39%
at 96×96, 20–22% at 528×528 and 17–18% at 768×768 in both orders.
Four vision-heavy HTTP cases also improve in both orders with identical
embedding values. The first seven-case HTTP comparison fails its full latency
gate because text and audio regress in one order. A separate repeat with five
warmups and at least thirty samples passes all fourteen quality and latency
checks across both orders; the earlier failed result remains recorded.
These results do not establish latency parity with llama.cpp.

Current controlled media comparisons against unchanged llama.cpp use the same
Q8 GGUF model/projector, matched pixels, audio clamp policy and video prefix.
The default CPU compute path meets the original latency margin in all seven
cases in both orders. Independent full-vector checks of all seven controlled
cases pass all fourteen TensorSharp comparisons at `.9999/.002`; llama.cpp
passes six. Optional CPU and CUDA also pass all fourteen TensorSharp quality
checks each; llama.cpp passes six and four, respectively.
The original cross-engine reports remain failed where their embeddings differ.
The optional CPU policy has an audio latency failure in one order. CUDA meets
the latency margin only for text and video in both orders. Six of seven broader
CPU text latency shapes miss the margin in every measured order and policy.
Full CPU text and GPU multimodal latency parity remain unachieved.

Generated reports, raw vectors, commands, binary hashes, and logs are in ignored
`docs/validation/embeddinggemma2-20261014/` and the continuation directory above;
reusable validation tools
and parser fixtures remain in `eng/validation/`. Missing model/device lanes
are not passing validation.

`encoding_format` accepts `float` (default) and `base64` (little-endian float32).
`dimensions` selects a positive prefix of the full vector and normalizes it
again. Snowflake is trained for Matryoshka reduction to 256 dimensions; choosing
arbitrary dimensions on a model without that training can lower retrieval
quality. Oversized OpenAI inputs are rejected, as are empty inputs, mixed types,
unknown model names, invalid token IDs, and unsupported formats. A request is
limited to 2048 inputs and 262144 input tokens in total. Embedding and model-show
JSON request bodies are limited to 16 MiB, including chunked requests; overflow
returns HTTP 413.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="local")
result = client.embeddings.create(
    model="snowflake-arctic-embed-l-v2.0-q8_0",
    input=["query: read a file", "def read_file(path): return open(path).read()"],
    dimensions=256,
)
vectors = [item.embedding for item in result.data]
similarity = sum(a * b for a, b in zip(*vectors))
```

## Ollama-compatible API

```bash
curl http://127.0.0.1:5000/api/embed \
  -H 'Content-Type: application/json' \
  -d '{"model":"snowflake-arctic-embed-l-v2.0-q8_0","input":["query: read a file","def read_file(path): return open(path).read()"],"truncate":false,"dimensions":256}'
```

`/api/embed` returns `model`, `embeddings`, `total_duration`, `load_duration`
(nanoseconds), and `prompt_eval_count`. `truncate` defaults to `true`; set it
to `false` to reject overflow. Truncation preserves the final separator token.
The service keeps its startup model resident; per-request `keep_alive` and
generation `options` do not reconfigure or unload it.

## Concurrent requests

The embedding service starts the first request immediately. After each encoder
call finishes, it combines whole requests already waiting in FIFO order, up to
64 input sequences and 4096 tokens per encoder call. There is no batching timer
or artificial delay. A request exceeding either grouping limit runs alone,
including a single sequence longer than 4096 tokens; the model still applies its
own microbatching. These grouping limits do not change the API admission limits
of 2048 inputs, 262144 tokens, and 16 MiB of JSON per request.

Each caller keeps its own row order, token usage, dimensions, and encoding.
Canceled waiting requests are skipped. Canceling one caller completes that caller
promptly while other callers continue; a combined encoder call is canceled when
all participating callers cancel. Disposal fails waiting callers and waits for
active computation to finish after requesting cancellation where supported.

## Retrieval quality

For Snowflake, prepend `query: ` to queries and embed documents without that
prefix. The server does not add it because it cannot infer whether an input is
a query or a document. Store each normalized vector beside its source file,
chunk text, and model/version identifier. Compare query and document vectors
with a dot product (equivalent to cosine similarity for unit vectors). Rebuild
an index when changing the model, tokenizer, dimensions, or chunking policy.

GGUF context capacity is not a quality guarantee for long documents. MiniLM's
upstream sentence-transformers configuration uses a shorter default length than
this GGUF's 512-position capacity. Split code at useful function/class boundaries
and evaluate retrieval with queries representative of your application.

### Tokenization conformance

XLM-R uses the GGUF's compiled SentencePiece character map and globally optimal
unigram segmentation. BERT uses WordPiece with Unicode NFD and the declared
case/accent settings. Tests cover 30 independent HuggingFace/llama.cpp cases per
model plus 13 Snowflake HTTP reference cases.

The [tokenizer oracle fixture](../InferenceWeb.Tests/Fixtures/EmbeddingTokenizer/huggingface-tokenization.json)
(the 13 Snowflake reference cases are in `snowflake-tokenization.json` beside it)
retains upstream differences: this Snowflake GGUF omits literal `<mask>` and
removes extra whitespace according to its metadata; MiniLM treats vertical-tab
and form-feed as whitespace. For decomposed Korean syllables and Indic spacing
vowels, TensorSharp preserves the complete HuggingFace Unicode normalization;
llama.cpp's simplified implementation drops some components.

## Library API and implementation

```csharp
using TensorSharp.Models.Embeddings;

using var model = EmbeddingModel.Load("models/embeddings/all-MiniLM-L6-v2-Q8_0.gguf",
    new EmbeddingModelOptions { Backend = "CPU", Threads = 8 });
EmbeddingBatchResult result = await model.EmbedAsync(new[] { "read a file", "open a document" });
float[] vector = result.Embeddings[0];
```

`Backend = "CPU"` uses the managed BERT/XLM-R encoder with the same tokenizer,
pooling, normalization, and API contracts as the native backends. It does not load
a native inference library. Use `Backend = "GGML_CPU"` to select native CPU kernels.

### Managed CPU execution

Each model owns a persistent worker pool with the configured number of computation
threads, including the caller (`Threads = 0` selects four). The same pool runs token
and position gathers, projections, activation
quantization, attention, normalization, and GELU. Workers spin briefly between
jobs, then park until work arrives; disposing the model stops and joins them.
Embedding math does not initialize the shared CPU pool or dispatch parallel
loops through the .NET thread pool.

Quantized weights stay compact. On ARM processors with signed dot-product
instructions, Q8_0 projections use a C# SIMD kernel that computes four token rows
and four output columns per tile. Load-time packing retains the original int8
weights, caches the half-precision scales as float32, and releases the original
projection byte arrays. Activations are quantized per Q8 block with the original
scale and round-to-even rules; scaled block sums use fused multiply-add (FMA).
A one-row remainder has its own kernel, avoiding duplicate work in a four-row tile.
Other instruction sets and weight formats use
the managed quantized kernels. Q/K/V projections are fused when their storage
types match; normalization and bias vectors are decoded once at load.

On x86, the managed path supports AVX through the runtime-selected
`Vector<float>` operations in
[`TensorComputePrimitives`](../TensorSharp.Core/TensorComputePrimitives.cs),
and explicit AVX2 / AVX-512 quantized dot products in
[`ManagedQuantizedOps`](../TensorSharp.Models/ManagedQuantizedOps.cs).
For example, Q8_0 projections select `VecDotQ8_0Q8_0Avx512` only when both
`Avx512F.IsSupported` and `Avx512BW.IsSupported` are true, then try
`VecDotQ8_0Q8_0Avx2` when `Avx2.IsSupported`; fallback code handles other hardware.
Attention also checks FMA support before using fused vector arithmetic.
The four-query × sixteen-column long-attention kernel is ARM-specific;
other processors use the portable vector-width implementation. Current embedding
timing evidence covers Apple Silicon; x86 performance has not been measured.

Projections share a compact batch of actual tokens. FP32 attention remains
isolated per sequence and computes four query rows together. Keys are transposed
once per layer so tokens of the same channel are contiguous. Each SIMD
score lane accumulates a different adjacent key; ARM uses FMA with a selected
query coefficient, and other processors use the available portable vector width.
A padded key stride permits the final vector load while only real token scores
are stored. Values are packed by channel tile, token, and SIMD lane (four channels
per tile on ARM). Four query rows share each tile through FMA, reusing probability
loads and accumulating separate output channels directly in vector lanes. Head
sizes that do not align to the tile width use transposed values and the four-query
dot-product fallback. The short and fallback paths hold four temporary score rows
per worker, with score scratch growing linearly with sequence length. Reusable activation arrays
reduce allocation between requests. CLS and last-token pooling compute only the
required query rows in the final attention and feed-forward stages; mean pooling
uses all real tokens.

When a batch contains a sequence of at least 1024 tokens and the heads support
packed values, the managed encoder uses 64-query × 128-key tiles. On ARM, inner
kernels compute four queries against sixteen keys or value channels, reusing each
query/probability vector across four SIMD column vectors. Other processors retain
the portable kernels. Online FP32
softmax carries each query's running maximum, exponential sum, and weighted-value
accumulator across key tiles, then normalizes the result. Subtracting the maximum,
exponentiating, and summing each probability tile share one pass. Each head's adjacent
query tiles share compact K/V blocks; worker scratch is bounded by the tile and
head sizes. Attention jobs are ordered by head. Unaligned heads and the selected
final rows for CLS/last pooling keep the fallback; shorter batches use query-group
order.

Tiling preserves the mathematical attention formula and FP32 precision, while
changing the reduction order. [Scalar encoder fixtures](../InferenceWeb.Tests/EmbeddingModelTests.cs)
check long mixed batches and all pooling modes within `2e-6` absolute error;
this is a numerical agreement check, not a claim of bitwise identity.

The native-free host check (`docs/validation/embeddings-2026-09/managed-native-free.json`,
local validation evidence, not committed; `benchmarks/EmbeddingBench/native_free_smoke.py`
reruns it) exercises both downloaded models after removing the host's custom native assets
and inspects its loaded libraries after inference.

### Native GGML execution

The native backends use TensorSharp's own GGML integration, with resident quantized
weights and reusable whole-encoder graphs. Quantized Q/K/V projections are fused
once at load while preserving the GGUF weight bytes. Attention is bidirectional and
GPU batches pack heterogeneous sequences with a block-diagonal attention mask;
equal-length sequences use an independent batch axis. Native CPU batches pack
projection tokens and pad attention sequences independently for numerical stability.
Outputs are restored to input order. CLS pooling prunes unused final-layer feed-forward
rows. Mean pooling excludes padding. Native CPU attention uses float32 K/V and aligned
shapes to stabilize numerical results across different batch sizes. Short native
CPU attention reads F32 K/V views directly; attention lengths of at least 1024
pack each head's K/V contiguously once while keeping F32 precision. The work plan
is cached with graph allocations. Calls serialize access to a model's graph
state; cancellation is checked before and between native batches and after an
in-flight device computation finishes.

Native CPU projection weights use optimized buffer layouts when the backend supports
them, preserving their GGUF quantization. Metal graph optimization runs before
graph-memory allocation so fused operations and tensor lifetimes agree.

Design references reviewed for this implementation:

- llama.cpp: BERT graph construction, GGUF tensor layout, exact UGM/WordPiece
  tokenization, pooling, and no-KV embedding execution.
- vLLM: pooling-model separation, per-input pooling metadata, normalization,
  and validated embedding responses.
- SGLang: embedding request validation, batch scheduling, and float32 base64
  response encoding.

See [the reproducible HTTP benchmark](../benchmarks/EmbeddingBench/README.md)
and the validation results in `docs/validation/embeddings-2026-09/README.md`
(local validation evidence, not committed). These checks
compare identical GGUFs against llama.cpp, with an additional independent NumPy
forward check ([`eng/embedding-reference.py`](../eng/embedding-reference.py); its
results are in `docs/validation/embeddings-2026-09/numpy-oracle/`, local validation
evidence, not committed).
The latter covers both models on pure C# CPU, native GGML CPU, and Metal in both
engine orders. These fixtures are not an MTEB evaluation.

### Benchmark concurrent API clients

[`concurrency_bench.py`](../benchmarks/EmbeddingBench/concurrency_bench.py) measures
separate single-input requests from concurrent clients. It reuses the commands
and inputs in a recorded benchmark setup, runs the engines sequentially, and
keeps one persistent HTTP connection per client. By default eight clients each
send four sequential requests per round, after ten seconds of runtime prewarm.

```bash
python3 benchmarks/EmbeddingBench/concurrency_bench.py \
  --base-results docs/validation/embeddings-2026-09/minilm-managed/results.json \
  --clients 8 --requests 32 --warmup 3 --rounds 10 \
  --output /tmp/minilm-managed-concurrency --require-performance
```

`--base-results` takes the `results.json` that `embedding_bench.py --output DIR`
writes; the path shown is the recorded run (local validation evidence, not
committed). Use a new output directory for a repeat; add `--tensorsharp-first` to reverse
engine order. The output records per-request latency through JSON parsing,
whole-round latency, requests/second, p95, vectors, token accounting, and binary
hashes. Vector validation runs after each measured round. The performance gate
compares median whole-round time with llama.cpp, allowing at most 5% overhead.
Exact parsed vectors are stored once in the report’s top-level `vectors` table;
resolve a request with `report["vectors"][request["vector_ref"]]`. This complements
the single-request and batched-input benchmark.

Sources: [Snowflake model card](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0),
[MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[OpenAI embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create),
[Ollama embed API](https://docs.ollama.com/api/embed).
