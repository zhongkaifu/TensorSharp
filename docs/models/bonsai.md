# Bonsai Q1_0 models

[← back to model index](README.md)

TensorSharp has been validated against two local, text-only Bonsai GGUFs. They
share a name and quantization, but not a transformer architecture: the 8B file
is a dense Qwen 3 decoder, while the 27B file is a dense Qwen 3.5 hybrid with
GatedDeltaNet recurrent layers.

| Property | Bonsai 8B | Bonsai 27B |
|---|---|---|
| File | `Bonsai-8B-Q1_0.gguf` | `Bonsai-27B-Q1_0.gguf` |
| GGUF architecture | `qwen3` | `qwen35` |
| Source class | [`Qwen3Model`](../../TensorSharp.Models/Models/Qwen3/Qwen3Model.cs) | [`Qwen35Model`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.cs) |
| Declared parameters | 8.2B | 27B |
| Layers | 36 full-attention | 48 GatedDeltaNet + 16 full-attention |
| Context in GGUF | 65,536 (YaRN from 16,384) | 262,144 |
| Vocabulary | 151,669 (`gpt2` / `qwen2`) | 248,320 (`gpt2` / `qwen35`) |
| Thinking | The embedded template always starts the assistant with an empty `<think></think>` block; it has no `enable_thinking` switch | Supported by the embedded template |
| Tool calls | Supported by the embedded template and TensorSharp's ChatML parser | Supported by the embedded template and TensorSharp's Qwen 3.5 parser |
| Modalities | Text only (no projector accompanies the validated artifact) | Text only (no projector accompanies the validated artifact) |

## Exact local artifacts

Their GGUF metadata contains neither a publisher/source repository URL nor a
license. On 2026-09-16, the publisher's Hugging Face file records were verified
to match both exact hashes below: [Bonsai 8B](https://huggingface.co/prism-ml/Bonsai-8B-gguf/blob/48516770dd04643643e9f9019a2a349cf26c5dbd/Bonsai-8B-Q1_0.gguf)
and [Bonsai 27B](https://huggingface.co/prism-ml/Bonsai-27B-gguf/blob/f10afb355f104535e3e3e98cf7ab7795c72bd292/Bonsai-27B-Q1_0.gguf).
Both publisher cards declare Apache-2.0. This provenance comes from the publisher
records; it does not change the embedded GGUF metadata or qualify a new runtime.

| File | Exact bytes | SHA-256 | Tensor mix |
|---|---:|---|---|
| `Bonsai-8B-Q1_0.gguf` | 1,158,654,496 | `284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54` | 254 Q1_0 + 145 F32 tensors |
| `Bonsai-27B-Q1_0.gguf` | 3,803,452,480 | `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0` | 498 Q1_0 + 353 F32 tensors |

Download the pinned files and verify them before use:

```bash
hf download prism-ml/Bonsai-8B-gguf Bonsai-8B-Q1_0.gguf --revision 48516770dd04643643e9f9019a2a349cf26c5dbd --local-dir ./bonsai
hf download prism-ml/Bonsai-27B-gguf Bonsai-27B-Q1_0.gguf --revision f10afb355f104535e3e3e98cf7ab7795c72bd292 --local-dir ./bonsai
cd bonsai
shasum -a 256 Bonsai-8B-Q1_0.gguf Bonsai-27B-Q1_0.gguf
```

`Q1_0` is GGML tensor type 41: each 128-value block stores one F16 scale and
128 one-bit signs (18 bytes, or 1.125 bits/weight). TensorSharp recognizes that
layout in the GGUF reader, native GGML bindings, and managed fallback rather
than confusing it with a 256-value K-quant block.

## Architecture

### Bonsai 8B: dense Qwen 3

The file declares hidden width 4,096, 32 query heads, 8 KV heads, head width
128, FFN width 12,288, RMSNorm epsilon `1e-6`, RoPE base 1,000,000, and YaRN
factor 4 from an original 16,384-token context.

```text
tokens -> Q1_0 embedding
  -> 36 x [ RMSNorm
            -> Q/K/V -> per-head QK RMSNorm -> YaRN RoPE
            -> causal GQA (32 Q heads / 8 KV heads)
            -> output projection + residual
            -> RMSNorm -> SwiGLU(gate, up) -> down + residual ]
  -> RMSNorm -> Q1_0 LM head -> logits
```

The GGUF stores Q, K, V, gate, and up separately. At load time TensorSharp
forms QKV and gate/up backing buffers for the optimized graph; the original
logical projections and the managed fallback remain available.

### Bonsai 27B: dense Qwen 3.5 hybrid

The file declares hidden width 5,120, FFN width 17,408, and 64 layers. Every
fourth layer is full attention (human-numbered layers 4, 8, ... 64), giving 16
attention and 48 recurrent layers.

```text
tokens -> Q1_0 embedding
  -> repeat 64 layers:
       layers 1-3 of each group:
         RMSNorm -> GatedDeltaNet recurrent update -> projection + residual
       layer 4 of each group:
         RMSNorm -> gated causal GQA -> projection + residual
       all layers:
         RMSNorm -> dense SwiGLU(gate, up) -> down + residual
  -> RMSNorm -> Q1_0 LM head -> logits
```

Full attention uses 24 query heads and 4 KV heads of width 256. Its Q
projection also produces a sigmoid output gate. Text positions use MRoPE
sections `[11, 11, 10, 0]` with RoPE base 10,000,000. Each GatedDeltaNet layer
has 16 K groups, 48 V heads, 128-wide K/V state, inner width 6,144, and a
four-tap causal convolution. Only the 16 full-attention layers grow a KV cache;
the other 48 carry fixed-size convolution and recurrent state.

The full Qwen 3.5 forward equations and state lifecycle are documented in the
[Qwen 3.5 / 3.6 card](qwen35.md).

## TensorSharp inference paths

### 8B

- Multi-token input uses [`ggml_ops_qwen3_prefill.cpp`](../../TensorSharp.GGML.Native/ggml_ops_qwen3_prefill.cpp): embedding, all 36 blocks, direct KV writes, final norm, and last-token LM head are one native graph. Long prompts use 512-token chunks on the eligible Metal fast path; interior chunks stop after committing the last layer's KV, while the final chunk returns logits. `startPos` makes continuation and chunk boundaries equivalent to a one-shot prompt.
- Single-token Metal decode uses the persistent whole-model graph in [`ggml_ops_qwen3_decode.cpp`](../../TensorSharp.GGML.Native/ggml_ops_qwen3_decode.cpp). It keeps the quantized embedding, all layers, KV update, final norm, and LM head on device and replays by padded attention bucket.
- The native prefill accepts F32, F16, Q8_0, and Q4_0 KV caches. Unsupported geometry, mixed projection storage, tensor parallelism, or a disabled fast path cleanly returns to the existing model/layer implementation.
- [`Qwen3Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Qwen3/Qwen3Model.BatchedForward.cs) supplies the paged multi-sequence interface. A block-quantized KV setting deliberately declines its F32 paged-buffer path so the scheduler can retain the fast per-sequence route instead of silently expanding the cache.

### 27B

- GGML CUDA, Vulkan, and Metal prefill use `TSGgml_Qwen35ModelVerify`, one whole-model graph containing both layer types, recurrent-state updates, attention KV writes, dense FFNs, and the last-token head. Bonsai's Metal geometry defaults to 512-token chunks, with recurrent state kept device-resident across chunks.
- Decode uses the persistent `TSGgml_Qwen35ModelDecode` whole-model graph. Full-attention KV plus GatedDeltaNet convolution/delta state advance together, avoiding hundreds of managed/native dispatches per token.
- Server continuous batching is on by default for Qwen 3.5 models and owns an isolated KV/recurrent-state holder per sequence. Use `--no-continuous-batching` only for an explicit fallback comparison.

## CLI and Server

From the repository root, place a prompt in `prompt.txt`, then choose either
artifact. F16 KV is the highest-quality cache; Q8_0 and Q4_0 reduce memory.
Setting `MAX_CONTEXT` is especially useful for the 27B file, whose advertised
262k window is much larger than most local memory budgets.

```bash
# Bonsai 8B on Apple Silicon
MAX_CONTEXT=16384 KV_CACHE_DTYPE=f16 \
dotnet run --project TensorSharp.Cli -c Release -- \
  --model /path/to/Bonsai-8B-Q1_0.gguf --backend ggml_metal \
  --input prompt.txt --max-tokens 256 \
  --temperature 0.5 --top-k 20 --top-p 0.85

# Bonsai 27B; add --think to expose its reasoning stream
MAX_CONTEXT=32768 KV_CACHE_DTYPE=f16 \
dotnet run --project TensorSharp.Cli -c Release -- \
  --model /path/to/Bonsai-27B-Q1_0.gguf --backend ggml_metal \
  --input prompt.txt --max-tokens 256 --think \
  --temperature 1.0 --top-k 20 --top-p 0.95
```

The same model path works in the OpenAI-compatible server:

```bash
MAX_CONTEXT=16384 KV_CACHE_DTYPE=f16 \
dotnet run --project TensorSharp.Server.Host -c Release -- \
  --model /path/to/Bonsai-8B-Q1_0.gguf --backend ggml_metal \
  --host 127.0.0.1 --port 5000

curl http://127.0.0.1:5000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Bonsai-8B-Q1_0.gguf","messages":[{"role":"user","content":"Explain KV caching briefly."}],"max_tokens":128}'
```

Replace the 8B path with the 27B file and raise `MAX_CONTEXT` as memory permits.
Use `ggml_cuda`, `ggml_vulkan`, or `ggml_cpu` on the corresponding platform;
the specialized persistent 8B decode measurements below are for Metal.

## TensorAgent sideload

Both built-in TensorAgent entries are intentionally `SideloadOnly`; an empty
publisher URL can never fall through to a network download.

1. Make the exact GGUF available to the device's Files picker.
2. Open **Models**, find **Bonsai 8B** or **Bonsai 27B**, and tap **Import**.
3. Select the matching filename. TensorAgent copies to a staging file, verifies
   both byte count and SHA-256, and only then atomically publishes and selects it.
   A wrong or interrupted file cannot replace an already verified import.

The catalog IDs are `bonsai-8b-q1-0` and `bonsai-27b-q1-0`; both entries require
a 12 GB device tier. Their catalog budgets are 16,384 and 32,768 tokens and ask
for Q8_0 KV. The visible Settings choice wins over the catalog request; its
current default is Q4_0, and F16/Q8_0/Q4_0 are all available.

## Correctness and performance

The 8B native path was compared against a current llama.cpp build on the exact
hash-pinned file. For the nine-token input
`785,3974,13876,38835,34208,916,279,15678,5562`, TensorSharp's one-shot F16
prefill produced logits 13.423851 (token 13), 12.851368 (token 1), and 12.267226
(token 1189), versus llama.cpp 13.4242, 12.8526, and 12.2681. Stepped decode
kept the same top token sequence. Raw F16, Q8_0, and Q4_0 KV-cache smokes all
produced the same first two greedy tokens, `13, 576`; model-backed Metal
batched-versus-fallback parity also passed.

Paired M5 Pro Metal runs used the same GGUF, F16 KV, a 512-token physical
chunk, and no sampler/argmax inside timed decode:

| Test | TensorSharp | llama.cpp | Ratio |
|---|---:|---:|---:|
| pp128 | 1,436.59 tok/s | 1,372.94 tok/s | 1.046x |
| pp512 | 1,508.56 tok/s | 1,517.48 tok/s | 0.994x |
| pp2048 | 1,343.23 tok/s | 1,334.25 tok/s | 1.007x |
| tg128 | 142.48 tok/s | 142.39 tok/s average | 1.001x |

The 27B run was bracketed by fresh llama.cpp measurements to account for the
large thermal drift of a sustained 27B workload. The table uses the midpoint
of the immediately-before and immediately-after llama.cpp best runs:

| Test | TensorSharp | bracketed llama.cpp | Ratio |
|---|---:|---:|---:|
| pp128 | 387.71 tok/s | 383.03 tok/s | 1.012x |
| pp512 | 410.12 tok/s | 397.30 tok/s | 1.032x |
| pp2048 | 398.16 tok/s | 381.66 tok/s | 1.043x |
| tg128 | 41.48 tok/s | 36.15 tok/s | 1.147x |

The raw llama.cpp brackets were 382.35/383.71, 408.11/386.48,
395.40/367.92, and 36.10/36.20 tok/s respectively. Reporting both endpoints
keeps the comparison auditable rather than hiding the machine's drift.

Fast non-model tests cover the Q1_0 row ABI, managed/native dequant agreement,
architecture routing, ChatML behavior, catalog pins, and staged import:

```bash
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release \
  --filter FullyQualifiedName~BonsaiCompatibilityTests
dotnet test TensorAgent/tests/TensorAgent.Tests/TensorAgent.Tests.csproj -c Release \
  --filter FullyQualifiedName~BonsaiCatalogTests
```

## Useful overrides

Defaults are the measured path. These switches are primarily for memory policy,
diagnosis, or A/B comparison:

| Setting | Effect |
|---|---|
| `MAX_CONTEXT=N` | Hard-cap and pre-size the usable context instead of treating the GGUF length as a ceiling. |
| `KV_CACHE_DTYPE=f16\|q8_0\|q4_0` | Select cache precision. F16 maximizes fidelity; block quantization saves memory. |
| `TS_PREFILL_CHUNK=N` | Override the prompt chunk width (measured Bonsai Metal default: 512). |
| `TS_QWEN3_MODEL_PREFILL=0` | Disable the 8B whole-model native prefill graph. |
| `TS_QWEN3_FUSED_LOGITS_DECODE=0` | Disable the 8B specialized embedding-to-logits Metal graph; the generic Qwen 3 path remains. |
| `TS_QWEN3_MODEL_DECODE=0` | Also disable the generic model-wide Qwen 3 decode, forcing the per-layer fallback. |
| `TS_QWEN3_FD_PERSIST=0` | Rebuild rather than replay the specialized 8B decode graph (diagnostic; slower). |
| `TS_QWEN35_PREFILL_VERIFY=0` | Disable the 27B whole-model prefill/verify graph. |
| `TS_QWEN35_FULL_DECODE=0` | Disable the 27B persistent whole-model decode graph. |
| `TS_QWEN35_BATCHED=0` | Disable Qwen 3.5 paged/continuous batching for an A/B. |
| `TS_GGML_ASYNC_COMPUTE=0` | Disable GGML async submission globally for diagnosis. |
| `TS_GGML_PHASE_TIMING=1` | Print native graph build/bind/allocate/upload/compute/download timing. |

Projection-splitting knobs in the native source are engineering diagnostics,
not recommended tuning: the fused QKV and fused gate/up storage was faster in
the Bonsai 8B comparisons and remains the default.
