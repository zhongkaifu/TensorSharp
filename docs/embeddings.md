# Embedding models

[English](embeddings.md) | [中文](embeddings_zh-cn.md)

TensorSharp serves GGUF BERT and XLM-RoBERTa sentence encoders through
`TensorSharp.Server` and `TensorSharp.Server.Host`. This is source-build support;
older release archives may not include it. It addresses
[discussion #183](https://github.com/zhongkaifu/TensorSharp/discussions/183),
including semantic search over source code stored in SQLite.

## Models and downloads

| Model | File | Dimensions | Context including special tokens | Pooling |
|---|---|---:|---:|---|
| Snowflake Arctic Embed L v2.0 | `snowflake-arctic-embed-l-v2.0-q8_0.gguf` | 1024 | 8192 | CLS |
| all-MiniLM-L6-v2 | `all-MiniLM-L6-v2-Q8_0.gguf` | 384 | 512 in this GGUF | Mean |

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
Model architecture support is specifically GGUF `bert` with supported tokenizer
and pooling metadata. Decoder embedding architectures, rerankers, sparse
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
The direct `cuda`, MLX, and Vulkan embedding paths are not
implemented. `--embedding-context-size N` reduces the per-input limit; zero
uses model metadata.

The library defaults to the pure C# backend. Select the native CPU backend
explicitly when using the C# API:

| Host `--backend` | Library `EmbeddingModelOptions.Backend` | Execution |
|---|---|---|
| `cpu` | `CPU` (default) | 100% pure C# CPU |
| `ggml_cpu` | `GGML_CPU` | Native GGML CPU |
| `ggml_metal` | `METAL` or `GGML_METAL` | Native GGML Metal |
| `ggml_cuda` | `CUDA` or `GGML_CUDA` | Native GGML CUDA |

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

curl http://127.0.0.1:5000/api/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"snowflake-arctic-embed-l-v2.0-q8_0","prompt":"query: read a file"}'
```

`/api/embed` returns `model`, `embeddings`, `total_duration`, `load_duration`
(nanoseconds), and `prompt_eval_count`. `truncate` defaults to `true`; set it
to `false` to reject overflow. Truncation preserves the final separator token.
The legacy `/api/embeddings` accepts one `prompt` and returns one `embedding`.
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
