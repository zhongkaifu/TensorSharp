# Embedding HTTP benchmark

`embedding_bench.py` launches TensorSharp and llama.cpp **sequentially** and
compares the same GGUF over `/v1/embeddings`. It uses Python's standard library.
The llama command uses 32 parallel slots so it can batch all 32 inputs, with
8192 context positions per slot. Comparing a batched TensorSharp request with
llama forced to one slot would overstate the speedup. Both commands are JSON
argument arrays, so paths containing spaces work and
no shell interprets their contents. Build both binaries first and keep the
machine idle during measurement.

```bash
python3 benchmarks/EmbeddingBench/embedding_bench.py \
  --model snowflake-arctic-embed-l-v2.0-q8_0 \
  --model-file /path/to/snowflake-arctic-embed-l-v2.0-q8_0.gguf \
  --tensorsharp-command '["dotnet","TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll","--model","/path/to/snowflake-arctic-embed-l-v2.0-q8_0.gguf","--embeddings","--backend","ggml_metal","--embedding-threads","8","--host","127.0.0.1","--port","18380","--no-webui"]' \
  --llama-command '["/path/to/llama-server","--model","/path/to/snowflake-arctic-embed-l-v2.0-q8_0.gguf","--embeddings","--alias","snowflake-arctic-embed-l-v2.0-q8_0","--host","127.0.0.1","--port","18381","--ctx-size","262144","--batch-size","8192","--ubatch-size","8192","--parallel","32","--threads","8","--n-gpu-layers","99","--flash-attn","on"]' \
  --prewarm-seconds 10 --warmup 5 --repeats 20 --minimum-measure-seconds 1 \
  --keep-alive --require-performance --output /tmp/embedding-results
```

For MiniLM, change the file and model name to `all-MiniLM-L6-v2-Q8_0`. Keep
llama's batch capacity large enough for the whole input batch; individual
sequences remain under the model's 512-position limit. For native CPU, use TensorSharp
`ggml_cpu` and llama `--n-gpu-layers 0` with the same thread count. For the 100%
pure C# encoder, use TensorSharp `cpu` against the same llama CPU command. Compare the
backends actually being used, and record native build options (including BLAS).

## What is checked

- Same-model vector cosine agreement (default minimum 0.999), dimensions,
  finite numbers, unit norms, and token accounting.
- Batch versus individual inference, reversed order, duplicates, variable
  lengths, Unicode normalization, and simple code/text retrieval rankings.
- TensorSharp's OpenAI float/base64 and reduced-dimension responses, both
  Ollama endpoints, model discovery, malformed requests, and concurrent calls.
- Seven latency cases: short/medium/long individual inputs, batches of 8/32,
  and a mixed-length batch. Each includes warmup and all recorded samples.

The output directory contains raw vectors, exact inputs/commands, model SHA256,
sample latencies, medians, p95, tokens/second, texts/second, comparisons, and server
logs. `binary_artifacts` records SHA-256 hashes of the host assemblies and nearby
GGML/llama binaries outside the measured work. Requests include tokenization and JSON serialization. No result caching is
used. Both engines receive a ten-second runtime prewarm before per-shape warmups;
`--prewarm-seconds` controls this and the output records its duration/call count.
This matters for tiny models that can complete the correctness corpus before
.NET's tiered compilation promotes hot methods. In the recorded diagnostic runs,
two seconds was insufficient: MiniLM's first measured case still improved from
about 2.2 ms to 1.0–1.1 ms during sampling. The longer warmup applies equally to
both engines. These are steady-state latency
measurements, excluding model startup and kernel compilation.
`--require-performance` fails if any TensorSharp median exceeds llama's
by more than 5%; change `--max-slowdown` to tighten the gate.

Repeat with `--tensorsharp-first` and a different output directory to check
engine-order effects. The small retrieval set is a regression smoke test, not
a retrieval leaderboard or MTEB score. Model-specific tokenizer differences
must be inspected against an independent tokenizer, not hidden by relaxing
the vector gate. Model-level CPU/Metal tests live in
`InferenceWeb.Tests/Embedding*Tests.cs`; the independent tokenizer oracle
fixtures are
`InferenceWeb.Tests/Fixtures/EmbeddingTokenizer/huggingface-tokenization.json`
and `snowflake-tokenization.json` beside it.

Use `--keep-alive` to measure SDK-style persistent HTTP connections. The default
opens a connection per request, including connection setup and teardown. Both
engines use the same mode, and `connection_mode` records it. Compare results
within a mode, especially for sub-millisecond inference where HTTP overhead is
a substantial part of latency. To measure with routine logging disabled, prefix
the TensorSharp command array with `"env","TENSORSHARP_LOG_FILE=0",
"TENSORSHARP_LOG_LEVEL=Warning"` and add `"--log-disable"` to llama.cpp's command.
These settings are recorded in the output commands.
For small GPU workloads, use `--minimum-measure-seconds 1` to measure each case
for at least a second as well as the requested repeat count. This reduces the
effect of short scheduling/clock-frequency bursts on a millisecond-scale case;
`sample_count` records the actual number of samples. Keep both engine-order runs
and inspect their distributions rather than selecting the faster run.

For the recorded CPU baseline, llama.cpp's batch/individual cosine can fall below
0.9999 because its kernel choices differ by batch shape. Reproduce that baseline
with `--reference-min-batch-cosine 0.999`; this changes only llama.cpp's consistency
gate. TensorSharp still requires 0.9999, and both engines' cross-vector and
independent NumPy checks remain in force. Keep the default for Metal.

See [embedding usage](../../docs/embeddings.md) and the recorded validation in
`docs/validation/embeddings-2026-09/README.md` (local validation evidence, not committed).

## Additional long-context cases

Use `--scenario-file cases.json --cases case_name` to measure additional shapes.
The JSON object maps each distinct case name to an array of input strings. For
example, the recorded full-context input
`docs/validation/embeddings-2026-09/full-context-inputs.json` (local validation
evidence, not committed) contains one Snowflake input that produces exactly 8192 tokens. The runner still
performs the complete short correctness corpus and symmetric runtime prewarm.
Additional cases also retain their measured output vectors and check unit norms,
cross-engine cosine, and identical token accounting. For expensive full-context
requests, choose explicit warmup and sample counts and report them separately
from the seven standard cases.
