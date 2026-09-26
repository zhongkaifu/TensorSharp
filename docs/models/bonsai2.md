# Bonsai2 27B

[← back to model index](README.md) | [中文](bonsai2_zh-cn.md)

Bonsai2 uses the dense Qwen 3.5 hybrid architecture with PRISM's signed
Hadamard rotations and custom low-bit GGUF encodings. It is distinct from
the earlier [Bonsai Q1_0 models](bonsai.md). Merely interpreting its weights
as ordinary ternary numbers produces the wrong network: projection inputs
and embedding outputs must also receive the declared transforms.

## Local artifacts and architecture

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7,206,168,928 | `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` |
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5,946,648,928 | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |

The companion files are `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` and
`Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`. When an image is given, the CLI looks
for them beside the model (BF16 first); the server loads a projector only
through `--mmproj`. Pass `--mmproj` to select one explicitly when comparing
image quality or memory use.

Both language-model files declare `general.architecture=qwen35`, 64 layers,
hidden width 5,120, FFN width 17,408, vocabulary 248,320, and a 262,144-token
context. There are 48 recurrent GatedDeltaNet layers and 16 full-attention
layers. Full attention uses 24 query heads and four KV heads of width 256;
GatedDeltaNet uses 16 key groups and 48 value heads of width 128. See the
[Qwen 3.5 card](qwen35.md) for the underlying attention, recurrent state,
RoPE, vision encoder, and chat/tool protocol.

```text
token ID -> low-bit embedding row -> inverse signed Hadamard
  -> 64 hybrid Qwen 3.5 layers:
       RMSNorm -> attention or GatedDeltaNet -> residual
       RMSNorm -> dense gated FFN -> residual
       (declared low-bit projections rotate their input activations)
  -> RMSNorm -> signed Hadamard -> low-bit output head -> logits
```

## Storage and transforms

The GGUF reader recognizes PQ2_0 as tensor type 142 and PTQ1_0 as 143.
TensorSharp-owned native code expands their block packing losslessly to
upstream GGML Q2_0 blocks. This preserves represented weight values without
re-quantization; it increases quantized payload size by approximately 6% for
PQ2_0 and 29% for PTQ1_0. Loaded memory therefore does not equal file size.
The original GGUF is never rewritten, and TensorSharp does not patch ggml
to introduce publisher-specific tensor types.

`prism.hadamard.*` metadata specifies the transform version, normalized
Sylvester-Walsh-Hadamard transform, input axis, block size, explicit ±1 signs,
and the weight names receiving forward or inverse transforms. TensorSharp
validates the complete contract before loading weights and rejects unknown
transform layouts. These artifacts use blocks of 1,024 elements.

For a projection input `x`, each block computes `H(Dx)/sqrt(block_size)`,
where `D` contains the stored signs and `H` is the unnormalized Hadamard
matrix. Embedding rows use the inverse order,
`D(Hx)/sqrt(block_size)`. Grouped GDN metadata additionally permutes the
SSM output input into the publisher's head order before its transform.
Fused QKV and fused gate/up weights retain the transforms of their original
projections.

Implementation entry points:

- [BonsaiHadamardMetadata.cs](../../TensorSharp.Runtime/BonsaiHadamardMetadata.cs)
  validates metadata and implements the managed inverse embedding transform.
- [ModelBase.Bonsai.cs](../../TensorSharp.Models/ModelBase.Bonsai.cs) handles
  transcoding and transform registration during model loading.
- [QuantizedWeight.Bonsai.cs](../../TensorSharp.Models/Weights/QuantizedWeight.Bonsai.cs)
  owns transform registration for weight cache keys and unregisters them on
  disposal.
- [bonsai_quant.cpp](../../TensorSharp.GGML.Native/bonsai_quant.cpp) implements
  exact block decoding/transcoding without modifying upstream GGML.
- [ggml_ops_bonsai.cpp](../../TensorSharp.GGML.Native/ggml_ops_bonsai.cpp)
  integrates the signed transform into TensorSharp's native graph paths.

## Running the model

The initial integration requires a single-device GGML backend. Pure managed
CPU, direct CUDA, MLX, and tensor-parallel configurations are rejected instead
of silently omitting the rotations. Device-specific end-to-end validation is
separate from that loader eligibility; see the validation procedure below.
Bonsai2 has no TensorAgent catalog entry; the app's two Bonsai entries are the
[Q1_0 files](bonsai.md#tensoragent-sideload).

Use an explicit context cap rather than allocating for the entire advertised
262k window:

```sh
MAX_CONTEXT=4096 KV_CACHE_DTYPE=f16 \
dotnet run --project TensorSharp.Cli -c Release -- \
  --model /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --backend ggml_metal --input prompt.txt --max-tokens 128 \
  --temperature 0

MAX_CONTEXT=4096 KV_CACHE_DTYPE=f16 \
dotnet run --project TensorSharp.Server.Host -c Release -- \
  --model /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --mmproj /path/to/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --backend ggml_metal --host 127.0.0.1 --port 5000
```

The existing Qwen 3.5 implementation provides whole-model prefill/decode,
per-sequence KV and recurrent states, and continuous batching. The PRISM
transforms are inserted into those paths so the model can use their existing
fusion and state management. Rotations add work, and the repacked weights use
more memory than the publisher's custom formats; performance must be measured
on the actual workload.

## Reproducible validation

The [Bonsai2 validation procedure](../../eng/validation/bonsai2.md) covers exact
raw-token references, single/parallel streaming requests, arithmetic,
structured JSON, tool calls, both projectors, and parser tests. Generated
evidence belongs in ignored `docs/validation/bonsai2/` or `artifacts/`.

The publisher baseline is the separate `PrismML-Eng/llama.cpp` checkout at
`bdc23b56b4458b9f1655aec5287f3ab56ee8daaa`. An upstream llama.cpp build that
cannot load PQ2_0/PTQ1_0 is not a valid performance comparison. Record the
unchanged upstream ggml revision used by TensorSharp independently.

The available validation host is an Apple M5 Pro with 48 GiB unified memory.
Its Python environment has no installed Torch, vLLM, or SGLang, and it has no
CUDA device. The inspected vLLM revision
`88aa0d287dd3abac89741bbea349350a4d49194e` and SGLang revision
`1d59ce7c9063edec20fd3f5a49a504b8c12acd23` contain Qwen 3.5/GatedDeltaNet
implementations but no PQ2_0/PTQ1_0 format support. Their fused projection,
chunked recurrent prefill, and isolated sequence-state patterns inform the
implementation; these local checkouts do not supply a measured Bonsai2
throughput baseline. No parity claim against their parallel throughput is
established by tests on this host.

### Measured integration coverage (2026-09-22)

Using unchanged upstream ggml `179b60f27b1019d42da01ac532cabdb8f73ba8b7`,
both formats reproduced the publisher baseline's four greedy prompts of
32 tokens exactly on Metal. Four concurrent sequences also matched their
serial outputs exactly, with 31 fused decode steps and no fallback steps.
Each format passed 21 HTTP solo/concurrent comparisons and arithmetic,
JSON, tool-call, and image-color checks. The image checks used PQ2 + Q8_0
projector and PTQ1 + BF16 projector. These are functional smokes, not a
comprehensive quality evaluation.

The additional CPU model check covered PQ2 only (two prompts, eight tokens
each). The longer Metal prompt covered 1,623 input tokens and two output
tokens ending at EOS. Neither qualifies the advertised 262k context.
CUDA, Vulkan, iOS, and tensor-parallel model execution were not validated.

| Format | Metric (tokens/s) | TensorSharp | Publisher llama.cpp |
|---|---|---:|---:|
| PQ2_0 | Prefill 512 | 364.05 | 384.44 |
| PQ2_0 | Decode 64 | 25.50 | 26.96 |
| PQ2_0 | HTTP, concurrency 4 | 28.40 | 32.57 |
| PTQ1_0 | Prefill 512 | 364.49 | 357.67 |
| PTQ1_0 | Decode 64 | 25.53 | 26.31 |
| PTQ1_0 | HTTP, concurrency 4 | 28.17 | 16.15 |

Model-only rates are means of three warmed runs at context depth zero;
HTTP rates are medians of three runs with 64 generated tokens per request.
Both engines used F16 KV and a 512-token physical prefill batch; the servers
had a 2,048-token context budget per request. Engines ran separately, without
thermal telemetry or interleaved repetitions. Small differences are therefore
not evidence of a general performance advantage.

The full performance target remains unmet: model-only decode is 5.4% slower
for PQ2 and 3.0% slower for PTQ1, and PQ2 concurrency-four HTTP throughput is
12.8% lower. vLLM/SGLang throughput is unmeasured. The ignored local report
`docs/validation/bonsai2/REPORT.md` retains commands, all samples, exact
artifact identities, and test limitations. The broader native CPU suite also
has one DeepSeek41 tolerance failure reproduced from unchanged TensorSharp
HEAD; it is not counted as a passing test.

### CUDA status after that run

A CUDA-only defect in the signed-Hadamard path was fixed afterwards. The
rotation matmul asks the backend to run its own fast Walsh-Hadamard transform
over the input; ggml-cuda's kernel does not check the input type and read an
F16 row as F32. In the inverse direction that gave a maximum absolute error of
16.0 against the dense oracle on an A5000, while the CPU backend was correct.
[ggml_ops_bonsai.cpp](../../TensorSharp.GGML.Native/ggml_ops_bonsai.cpp) now
widens the transform input to F32 before that node. Model activations were
already F32, so the model's own paths gain no node.

[`bonsai2-reference-comparison.sh`](../../eng/validation/bonsai2-reference-comparison.sh)
is the reference-engine harness for an A5000 CUDA host. It records the
upstream llama.cpp refusals for PQ2_0/PTQ1_0, builds the PrismML fork at the
pinned commit in a separate tree while asserting that TensorSharp's ggml
checkout stays unchanged, measures model-only and HTTP (concurrency 1 and 4)
throughput on the same GGUF, runs a projector smoke, and writes its output under
ignored `artifacts/`. No results from it are committed, so CUDA model execution
remains unvalidated as stated above.
