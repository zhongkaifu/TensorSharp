# Hunyuan Dense (`hunyuan-dense`)

[← back to model index](README.md) | [中文](hunyuan-dense_zh-cn.md)

TensorSharp runs Tencent's dense Hunyuan decoders — the checkpoints whose GGUF
declares `general.architecture = hunyuan-dense`, such as the Hy-MT2 translation
releases. Before this architecture was registered, those official Q4 files
failed at load rather than falling back to a similar family: the loader fails
closed on an unknown architecture instead of guessing a graph.

It is a single-device, text-only path on the generic per-op executor: there is
no fused whole-model graph, no tensor parallelism and no layer split. The
server serves it through the continuous-batching engine with a batched paged
forward (see [Serving](#serving-continuous-batching)).

| Property | Value |
|---|---|
| GGUF architecture | `hunyuan-dense` |
| Source class | [`HunyuanDenseModel`](../../TensorSharp.Models/Models/HunyuanDense/HunyuanDenseModel.cs) |
| Architecture plug-in | [`HunyuanDenseArchitecture`](../../TensorSharp.Models/Models/HunyuanDense/HunyuanDenseArchitecture.cs) |
| Reference template | `tencent/Hy-MT2-1.8B` `chat_template.jinja` |
| Modalities | Text only |
| Thinking | No |
| Tool calling | No — the protocol renders neither tool declarations nor `role: "tool"` results |
| Batched / paged forward | **Default** — [`HunyuanDenseModel.BatchedForward.cs`](../../TensorSharp.Models/Models/HunyuanDense/HunyuanDenseModel.BatchedForward.cs) (`IBatchedPagedModel`); `TS_HUNYUAN_BATCHED=0` selects the K/V-snapshot swap path |
| Speculative decoding | Not supported |
| Multi-GPU | Single device. `MultiGpuLimitation` says so on stderr rather than leaving extra GPUs idle in silence |
| Backends | Every backend, through the generic per-op path: `cpu`, `ggml_cpu`, `ggml_metal`, `ggml_cuda`, `ggml_vulkan`, `cuda`, `mlx` |

## Run it

```bash
dotnet run --project TensorSharp.Cli -- \
    --model models/Hy-MT2-1.8B-Q4_K_M.gguf \
    --backend ggml_metal \
    --input "Translate to French: the harbour was quiet before dawn."
```

The server takes the same model flag:

```bash
dotnet run --project TensorSharp.Server.Host -- \
    --model models/Hy-MT2-1.8B-Q4_K_M.gguf \
    --backend ggml_cuda --port 5000
```

## Architecture

The text graph matches llama.cpp's `hunyuan-vl` text tower:

```text
tokens -> embedding
  -> N x [ RMSNorm
            -> fused QKV -> NeoX RoPE -> per-head Q/K RMSNorm
            -> causal GQA
            -> output projection + residual
            -> RMSNorm -> SwiGLU(gate, up) -> down + residual ]
  -> final RMSNorm -> LM head
```

**QK-norm runs after RoPE**, which is the opposite order from Qwen 3.5. Getting
that order wrong produces fluent but wrong output rather than a load failure, so
the loader treats the norms as mandatory: a layer missing `attn_q_norm.weight`
or `attn_k_norm.weight` throws at load. Key and value head widths must be equal;
an unequal pair is refused with `NotSupportedException` rather than silently
reshaped.

### RoPE base and NTK alpha

When the GGUF carries `hunyuan-dense.rope.scaling.alpha`, the loader applies
llama.cpp's Hunyuan formula before any position is computed:

```text
base = rope_theta * alpha^(dim / (dim - 2))
```

The Hy-MT2 Q4 files ship `scaling.type = none` and no alpha, so their base is
used as published. Startup prints the effective base, the scale and the RoPE
dimension count, so which branch ran is visible in the log rather than inferred.

### Weight fusion and the KV cache

At load the model fuses each layer's Q/K/V into one `attn_qkv.weight` and each
layer's gate/up into one `ffn_gate_up.weight`, in both the quantized and F32
paths, and only when the three tensors share a GGML type and input width. A
layer that does not qualify keeps its separate projections; the per-layer flags
are precomputed once rather than re-tested per token.

Per-layer K and V caches are allocated at the model-aligned KV dtype and start
at the initial allocation length, doubling on demand up to the configured
maximum context. Each growth is printed.

### Tokenizer

`hunyuan-dense` uses the same three-pass Unicode pre-tokenizer as the DeepSeek
V3/V4 and JoyAI vocabularies: digits in runs of up to three, then CJK runs, then
the general pattern. Folding those passes into one alternation moves boundaries
at mixed CJK/Latin text, which is why they stay separate.

## Serving (continuous batching)

The server's continuous-batching engine needs a model to offer a batched paged
forward or a K/V-state snapshot. Hunyuan Dense first shipped with neither, so
`InferenceEngineHost.TryGetEngine` returned null and **every** chat request was
an HTTP 500 (`Continuous-batching engine is unavailable for this model`), with
the startup prefix-cache warm-up failing the same way. It now offers both:

- **Batched paged forward (default).** One `ForwardBatch` packs every
  scheduled token of every running request, scatters K/V into per-layer paged
  buffers through the slot mapping and runs per-request causal attention through
  the native paged-attention kernel (`TS_PAGED_ATTN_KERNEL`, as for Mistral 3).
  The per-layer order is the one above: NeoX RoPE, then per-head Q/K RMSNorm.
  A block-quantized KV cache (`q8_0`/`q4_0`) declines this path, because the
  paged buffers are F32.
- **K/V-state snapshot.** Every layer is full causal attention over a linear
  cache, so `TryExtractKVBlock` / `TryInjectKVBlock` restore exactly what a
  fresh prefill writes. With `TS_HUNYUAN_BATCHED=0` (or a block-quantized KV
  cache) concurrent requests take turns on the single cache by swapping
  snapshots; they are served correctly but serially.

Prefix reuse works on both paths. A block the batched path wrote is adopted in
the model's paged storage; a block the snapshot path captured is restored into
the linear cache (see `KvBlock.HoldsModelPagedKv` / `HoldsSnapshotBytes`).

## Chat template

Hy-MT2 framing always opens with BOS, and the assistant marker is appended only
for the generation prompt — it is not glued onto the user turn. That differs
from llama.cpp's `LLM_CHAT_TEMPLATE_HUNYUAN_DENSE` (Hunyuan-4B-Instruct), which
omits BOS and does glue the marker on.

| Turn | Rendered |
|---|---|
| User only | `<｜hy_begin▁of▁sentence｜><｜hy_User｜>Hello<｜hy_Assistant｜>` |
| System + user | `<｜hy_begin▁of▁sentence｜>SYSTEM<｜hy_place▁holder▁no▁3｜><｜hy_User｜>Hello<｜hy_Assistant｜>` |
| With assistant history | each past answer closes with `<｜hy_place▁holder▁no▁2｜>` |
| Without a generation prompt | the render ends with `<｜hy_place▁holder▁no▁8｜>` |

The renderer is preferred over the GGUF's embedded Jinja for this architecture,
and it emits no tool declarations. Agent Skills therefore fall back to inlined
instructions here, as they do for every family without a tool parser.

## Current limits

- Text only. No projector is wired up, so `--image`, `--video` and `--audio` do
  not apply.
- No thinking channel and no tool-call parser.
- Single device: no tensor parallelism, no layer split, no fused whole-model
  graph, and no speculative decoding.
- Hy-MT2-1.8B is a translation model and follows its own output habits: it
  answers `What is 17 + 25? Reply with only the integer.` with `42.`, and on the
  release translation fixtures it copies the input back for the
  `structured_json` and `||`-delimited word-list prompts instead of translating
  them. That is the model, not the serving path. With
  `validate-release-translation.py --concurrency 1,4 --repeats 3` on
  `ggml_cuda`, both serving paths showed the same pattern:
  `zh_en`, `en_zh`, `fr_en` and `long_translation` passed every request
  (3 at concurrency 1 and 12 at concurrency 4 each), and `delimiters` failed
  every request. `structured_json` failed all 3 at concurrency 1, but 3 of 12
  (batched) and 4 of 12 (snapshot) translated at concurrency 4. The model sits
  on a near-tie there (`"Hello"` against `"你好"`), and batching changes the
  kernel shapes enough to flip it. llama.cpp (`llama-server`, CUDA) on the same
  GGUF also copies the `delimiters` prompt back. It translates `structured_json`
  at concurrency 1, with top-2 log-probabilities of -0.63 (`你好`) and -0.79
  (`Hello`) at the token that decides it.
