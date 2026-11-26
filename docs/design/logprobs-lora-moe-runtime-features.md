# Log Probs, Dynamic LoRA, Runtime MoE, and MoE Selection — Design

**Status:** Partially implemented.
- **Feature 1 (Log probs):** not started.
- **Feature 2 (Dynamic LoRA):** **implemented** — GGUF (llama.cpp) and safetensors adapters (§2.2.1), per-request adapters under continuous batching ("adapter epochs", §2.2.5), the deny-by-default plan with exact fused composites and a per-forward audit (§2.2.4), `lora` field on completions/responses, `/v1/loras` admin routes, `/v1/models` listing with `applicable`. See `TensorSharp.Models/Lora/` and `TensorSharp.Server/Endpoints/LoraEndpoints.cs`. Limitations: applied only by nemotron_h (its simple dense projections), mistral3 and hunyuan-dense on one device with the cpu, ggml_cpu, ggml_cuda or cuda backend; every other family, backend and multi-device mode, and every adapter module a model cannot apply, is refused with 400 before any output. TP-sharded factors and the families whose projections run in native graphs are follow-ups.
- **Feature 3 (Dynamic MoE):** foundation in progress (module format + registry); forward-path expert swap (P3–P6) not yet wired.
- **Feature 4 (MoE selection by index):** not started.

**Date:** 2026-10-04 (design), 2026-10-19 (LoRA implemented)
**Scope:** Four orthogonal features touching the API surface, model lifecycle, expert routing, and sampling pipeline.

---

## What Exists Today

| Capability | Status |
|---|---|
| Log probs in completions/responses API | **Absent.** `TokenSampler.Sample()` returns one `int` (token id). No probability distribution survives sampling. The API adapters (`OpenAIChatAdapter`, `OpenAIResponsesAdapter`, `OllamaAdapter`) emit no `logprobs` field. `SamplingConfig` has no `logprobs_count` or equivalent. |
| LoRA adapters for text models | **Absent.** The only LoRA implementation is `QwenImage21LoraSet` — image-generation-specific, loaded at startup, never swapped. The `--lora` CLI flag routes exclusively to Qwen-Image-2.1's DiT transformer. The `ModelLifecycleService` comment says: *"LoRA plug-ins apply to Qwen-Image-2.1's transformer only. Other models load without them."* |
| Dynamic MoE at runtime | **Absent.** MoE expert weights are part of the GGUF checkpoint, loaded once at startup via `ModelBase.LoadWeights()`, sharded for TP, and immutable afterward. There is no mechanism to add, remove, or replace experts while the model is serving. |
| MoE selection by embedding index | **Absent.** Every MoE model uses a learned router (a linear projection `ffn_gate_inp.weight` + softmax top-K) per layer. There is no external index, no domain-embedding lookup, and no mechanism to bypass the router for a hand-picked expert set. |

---

## 1. Log Probs in Completions and Responses APIs

### 1.1 Objective

Return the log-probability of sampled tokens (and optionally the top-N alternatives) on both `POST /v1/chat/completions` and `POST /v1/responses`, matching the [OpenAI logprobs contract](https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs) for completions and extending the Responses API with the same data.

### 1.2 Design

#### 1.2.1 Where probabilities are computed

The bottleneck is `TokenSampler.Sample(float[] logits, ...)`. Today it applies temperature, top-K, top-P, min-P, penalties, grammar masks, and forced first-token constraints, then samples one token and returns its id. All per-token probability information is discarded.

We need **two paths** from the sampler, controlled by `SamplingConfig`:

```csharp
public class SamplingConfig
{
    // NEW
    public int LogprobsCount { get; set; } = 0;  // 0 = disabled; N = top-N alternatives

    /// <summary>
    /// The sampled token plus its logprob, and optionally the top-N alternatives
    /// with their logprobs. Non-null only when LogprobsCount > 0.
    /// </summary>
    public LogprobsResult? LastLogprobs { get; internal set; }
}

public sealed class LogprobsResult
{
    public int TokenId { get; init; }
    public float Logprob { get; init; }
    public string? TextOffset { get; init; }  // for token->text mapping
    public TopLogprob[]? TopLogprobs { get; init; }  // top-N alternatives

    // OPENAI: returns logprob = null for the sampled token when top_logprobs is requested
    // and logprobs=false, but we always capture the sampled token's logprob.
    // RESPONSES EXTENSION: always includes both.
}

public readonly struct TopLogprob
{
    public int TokenId { get; init; }
    public float Logprob { get; init; }
    public string? Token { get; init; }
}
```

**Implementation in `TokenSampler.Sample()`:**

When `_config.LogprobsCount > 0`, after computing the final probability vector (post-temperature, post-penalties, post-grammar mask) but BEFORE the random draw:

1. Normalize to a proper probability distribution via softmax.
2. Record the sampled token's logprob.
3. If `LogprobsCount > 1`, do a partial sort (heap over top-N) and record each alternative's token id and logprob.
4. Store the result in `_config.LastLogprobs`.

The greedy path (`IsPlainGreedyArgmax == true`, which avoids materializing host logits) must force host logits when logprobs are requested. That path is a decode-only optimization; a client asking for logprobs is accepting the overhead.

#### 1.2.2 API parameter parsing

In `SamplingConfigParser.ParseOpenAI()`:

```
logprobs: bool?           → if true, LogprobsCount = 1 (sampled token only)
top_logprobs: int? (0-20) → if > 0, LogprobsCount = max(logprobs ? 1, top_logprobs)
                              (top_logprobs implies logprobs=true, per OpenAI spec)
```

In the Responses API (`OpenAIResponsesAdapter`), add the same fields under `text`:

```json
{
  "model": "...",
  "input": "...",
  "text": {
    "format": {"type": "json_schema", ...},
    "logprobs": true,            // NEW — sampled token logprob
    "top_logprobs": 5             // NEW — top-N alternatives
  }
}
```

#### 1.2.3 Response serialization

**Completions** (`OpenAIResponseFactory`): each `choice` carries an optional `logprobs` block matching OpenAI's shape:

```json
{
  "choices": [{
    "index": 0,
    "logprobs": {
      "content": [
        {
          "token": " the",
          "logprob": -0.12,
          "bytes": [32, 116, 104, 101],
          "top_logprobs": [
            {"token": " the", "logprob": -0.12, "bytes": [32, 116, 104, 101]},
            {"token": " a",   "logprob": -1.45, "bytes": [32, 97]},
            ...
          ]
        },
        ...  // one entry per generated token
      ]
    },
    "message": { "role": "assistant", "content": "..." },
    "finish_reason": "stop"
  }]
}
```

Streaming chunks carry `logprobs` in each delta chunk (the same shape). The final chunk (`finish_reason ≠ null`) may carry logprobs for the last token only, matching OpenAI's behavior.

**Responses** (`OpenAIResponsesFactory`): this is an extension over what OpenAI provides. The `output_text` content part gains `logprobs`:

```json
{
  "output": [{
    "type": "message",
    "id": "msg_...",
    "status": "completed",
    "content": [{
      "type": "output_text",
      "text": "... ",
      "logprobs": {                         // TENSORSHARP EXTENSION
        "token": " there",
        "logprob": -0.34,
        "top_logprobs": [...]
      }
    }]
  }]
}
```

Because the Responses API emits one `output_text.delta` event per token in streaming mode, `logprobs` attaches to each delta event.

#### 1.2.4 Token-to-text mapping

`TopLogprob.Token` is the detokenized string of each alternative token id. `LogprobsResult.TextOffset` tracks token boundaries in the accumulated output, needed when `logprobs` must reproduce the exact text the client received. The `ITokenizer` interface already has `Decode(int[])`; each alternative is decoded independently.

#### 1.2.5 Phasing

| Phase | Scope | Risk |
|---|---|---|
| **P1 — Core sampler** | `LogprobsCount`, `LogprobsResult`, top-N partial sort in `TokenSampler.Sample()` | Low: additive, no existing path changed |
| **P2 — Completions API** | Parse `logprobs`/`top_logprobs`, serialize in `OpenAIResponseFactory`, streaming chunks | Low: new fields, opt-in |
| **P3 — Responses API** | Parse `text.logprobs`/`text.top_logprobs`, serialize in `OpenAIResponsesFactory` | Low: extension field, opt-in |
| **P4 — Ollama API** | Add `logprobs` to Ollama protocol if desired | Medium: non-OpenAI contract, separate design needed |

---

## 2. Dynamic LoRA Adapter Control

### 2.1 Objective

Let API callers select which LoRA adapter(s) to apply per request, and load/unload adapters at runtime without unloading the base model.

### 2.2 Architecture

The existing `QwenImage21LoraSet` is the only LoRA implementation and is **image-specific**. A general-purpose LoRA system for text models is a significant new capability. The design must:

1. Define a universal LoRA weight format (GGUF-based, since every model already loads from GGUF).
2. Define a runtime registry that owns adapter weight buffers, decoupled from the base model.
3. Define the merge point: a LoRA'd linear projection is `W·x + (α/r)·B·A·x`, applied inside the model's forward path.
4. Make adapter selection per-request, so two concurrent requests can use different adapters.

#### 2.2.1 LoRA weight format

> **As implemented (2026-10-28).** The original sketch here (a `lora.{target}` prefix, a
> per-tensor `alpha` tensor, `__metadata__.base_model`) matched neither PEFT nor llama.cpp
> and was never built. What is implemented is llama.cpp's convention, in two containers
> (`TextLoraLoader`).

Every target is named by the **base GGUF's tensor name** with a `.lora_a` / `.lora_b`
suffix, in PEFT's orientation:

| Tensor | Row-major shape | GGUF `ne` | Description |
|---|---|---|---|
| `<tensor>.lora_a` | `[rank, in]` | `[in, rank]` | Down projection (A) |
| `<tensor>.lora_b` | `[out, rank]` | `[rank, out]` | Up projection (B) |

e.g. `blk.0.attn_q.weight.lora_a`, `blk.12.ffn_down.weight.lora_b`, `output.weight.lora_a`.
Projections TensorSharp fuses at load (`attn_q/k/v` into `attn_qkv`, `ffn_gate/up` into
`ffn_gate_up`) keep their GGUF names in an adapter; the model resolves them (§2.2.4). The fused
names are accepted too, but whether a layer is fused depends on the quantization, so the GGUF
names are the stable ones.

**Containers.**

1. **GGUF adapter** (`.gguf`, or any file starting with the GGUF magic), exactly what
   llama.cpp's `convert_lora_to_gguf.py` writes: `general.type = adapter`,
   `adapter.type = lora`, `adapter.lora.alpha` (0 means scale 1, as in llama.cpp; it must be
   one FLOAT32 value, as llama.cpp's `gguf_get_val_f32` asserts, so a BOOL, STRING, integer or
   array is refused rather than converted) and `general.architecture`, which the loaded
   model checks against its own (registry aliases count, so a `nemotron_h` adapter fits a
   `nemotron_h_moe` model) and against the base
   GGUF's own label, which is llama.cpp's rule: a Mistral 3 file labelled `llama`, served as
   mistral3, takes the `llama` adapter a converter writes for it, while a `llama` adapter on a
   base not labelled so is refused. The per-weight shape checks apply either way. Factors may
   be F32, F16, BF16 or Q8_0; they go through the managed dequantizer
   (`GgufFile.ReadTensorDataToFloat32` is a raw copy that is right only for F32). The file is
   opened alone, never its `-0000N-of-` siblings, so `--loras-dir` confinement holds. A GGUF
   adapter reads no `adapter_config.json`.
2. **safetensors** (`.safetensors`): F32/F16/BF16/F64 factors; `r`, `lora_alpha` and
   `use_rslora` from a PEFT `adapter_config.json` (next to the file, or `config` on
   `POST /v1/loras`), else the `__metadata__` keys `rank` / `alpha`; without an alpha the
   scale is 1.

In both containers an alpha of 0 means none, so the scale is 1, with or without `use_rslora`
(standing in the rank for the alpha would give `sqrt(rank)` under rsLoRA), as llama.cpp reads
`adapter.lora.alpha`; the adapter is listed with `alpha = rank`. A config's `lora_alpha` of 0
leaves the `__metadata__` alpha to apply.

**Scale.** Each module's scale is `alpha / rank` (`alpha / sqrt(rank)` with `use_rslora`), or 1
without an alpha, times the `scale` given at load, folded into `B` once. Modules may have
different ranks (each uses its own); a config `r` is PEFT's single rank, so a module of another
rank is refused as a file that does not match its config.

**Refused, never skipped.** Anything whose effect cannot be reproduced exactly fails the
load with the reason:

- PEFT / Hugging Face module names (`base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`).
  Mapping them onto GGUF tensors needs the architecture's own conversion applied to A and B
  (the llama q/k row permutation for Mistral, head reorders, split projections), which is
  what llama.cpp's converter does; a rename-only mapper would recreate the silent-wrong
  output this design exists to prevent. The message says what to run:
  `python convert_lora_to_gguf.py <peft_adapter_dir> --base <hf_base_model_dir> --outtype f16`,
  whose output loads directly.
- PEFT settings `use_dora`, `fan_in_fan_out`, an `alpha_pattern` or `rank_pattern` that is not
  empty (it is keyed by PEFT module names, so applying it exactly needs the converter's
  mapping; a malformed one, a number say, is refused too, never read as empty),
  `modules_to_save`, trained token embeddings (`trainable_token_indices`, its list or dict
  form, unless empty), trained biases (`bias` other than `"none"`, `lora_bias`),
  `layer_replication`, activated LoRA (`alora_invocation_tokens`, any value but null, or the
  older `invocation_string` of IBM's granite aLoRA adapters unless empty, which llama.cpp's
  converter also writes as `adapter.alora.*`), and PEFT's other LoRA variants, any value but
  null or false (an empty object is the variant with its defaults): `kasa_config` (KaSA
  applies `lora_B(lora_A(x) * lora_diag)`), `arrow_config` (Arrow routes each token to its
  top-k LoRA experts) and `use_bdlora` (BD-LoRA stores a block-diagonal factor,
  `[out, in / nblocks]`). `velora_config` and `monteclora_config` load: VeLoRA changes only
  the backward pass and MonteCLoRA perturbs `lora_A` only while training, so PEFT's forward
  when not training is `lora_B(lora_A(x)) * scaling` for both, and both merge as plain LoRA.
  The boolean flags read (`use_dora`, `use_rslora`, `fan_in_fan_out`, `lora_bias`) must be
  `true` or `false` (or null): PEFT tests them by Python truth, so `1` or `"true"` would turn
  DoRA or rsLoRA on there, and is refused rather than read as off.
- An `init_lora_weights` that rewrites the base weights when PEFT creates the adapter:
  `"pissa"` and `"pissa_niter_<N>"`, `"olora"` (in any case, as PEFT's layer matches it),
  `"corda"`, `"astra"`, `"loftq"` and `"lora_ga"`. These set `W` to `W - s * B0 * A0` (LoftQ
  to a quantized residual), so the trained adapter applies on top of that, not on top of `W`;
  text LoRA adds it to `W`, which for PiSSA adds the top-r SVD component twice. PEFT writes
  a plain LoRA adapter (`init_lora_weights` true, `r` and `lora_alpha` doubled) only when
  `save_pretrained` is given `path_initial_model_for_weight_conversion`, and the message says
  to convert it that way; LoftQ's residual has no such conversion. Every other value
  (`true`, `false`, `"gaussian"`, `"eva"`, `"nora"`, `"orthogonal"`, `"mica"`) only
  initializes the factors and loads.
- Any tensor that is not a factor (a fully trained module, a DoRA magnitude), a factor
  without its partner, a GGUF whose `general.type` is not `adapter` (a model GGUF), an
  `adapter.type` other than `lora` (control vectors), and activated LoRA (`adapter.alora.*`).
- An alpha that is negative or not a finite number, from `adapter.lora.alpha` (including one
  that is not a FLOAT32), `lora_alpha` (including one that is not a JSON number) or the
  `__metadata__` `alpha`. Replacing a negative one by the rank would apply the adapter with
  the wrong sign, and an infinite one cannot be listed (`GET /v1/loras` and `GET /v1/models`
  write alpha as a JSON number).
- A module whose scale (`alpha / rank` times the load's `scale`) or folded factors are not
  finite: a NaN or infinity in the file, or a value that overflows once scaled.

**Memory.** Factors are held as F32: `4 * rank * (in + out)` bytes per module, once per
`(model, adapter)` (§2.2.5).

`eng/text-lora-fixture.py` writes a seeded test adapter for a base GGUF in both containers,
plus the merged-weights GGUF (`W + scale * B @ A`) as an oracle; see §2.2.6.

#### 2.2.2 Runtime adapter registry

```csharp
/// <summary>
/// Owns loaded LoRA adapters independently of the base model.
/// Thread-safe for concurrent reads (inference) and rare writes (load/unload).
/// </summary>
public sealed class LoraRegistry : IDisposable
{
    private readonly ConcurrentDictionary<string, LoraAdapter> _adapters = new();
    private readonly ReaderWriterLockSlim _lock = new();

    /// <summary>Load an adapter from a GGUF file. Fails if the architecture
    /// does not match the base model.</summary>
    public LoraAdapter Load(string name, string ggufPath, ModelBase baseModel);

    /// <summary>Unload and free an adapter's memory.</summary>
    public void Unload(string name);

    /// <summary>Get the currently active adapter (or null for no adapter).
    /// The active adapter is the one applied to all requests that do not
    /// specify one.</summary>
    public LoraAdapter? Active { get; set; }

    /// <summary>Get a snapshot of loaded adapter metadata. Cheap; no lock.</summary>
    public IReadOnlyDictionary<string, LoraMetadata> List();

    public void Dispose();
}

public sealed class LoraAdapter
{
    public string Name { get; }
    public LoraMetadata Metadata { get; }

    /// <summary>Per-target deltas keyed by base-model weight name.</summary>
    internal IReadOnlyDictionary<string, LoraWeightDelta> Deltas { get; }
}

public readonly struct LoraWeightDelta
{
    public IntPtr A;  // [rank, in] — pinned F32 buffer
    public IntPtr B;  // [out, rank] — pinned F32 buffer
    public int Rank;
}
```

#### 2.2.3 API parameter

Both completions and responses APIs accept a new `lora` parameter:

**Completions:**
```json
{
  "model": "qwen3.5-9b",
  "messages": [...],
  "lora": "coding-assistant"  // null, omitted or "none" = the base model, for this request only
}
```

**Responses:**
```json
{
  "model": "qwen3.5-9b",
  "input": "...",
  "lora": "coding-assistant"
}
```

The `/v1/models` endpoint exposes available adapters, and whether the loaded model can apply
each (`GET /v1/loras` reports the same, plus `in_flight`):

```json
{
  "object": "list",
  "data": [{
    "id": "ministral-3b",
    "object": "model",
    "loras": [
      {"id": "coding-assistant", "rank": 64, "alpha": 128, "modules": 182, "applicable": true},
      {"id": "embed-tuned", "rank": 8, "alpha": 16, "modules": 1, "applicable": false,
       "reason": "LoRA adapter 'embed-tuned' cannot be applied to the loaded model, ... Unsupported modules: token_embd.weight (not a decoder projection ...)."}
    ]
  }]
}
```

A request that names an adapter it cannot run with gets a 400 before any output, streamed or
not, in the OpenAI error shape:

```json
{"error": {"message": "LoRA adapter 'embed-tuned' cannot be applied ...", "type": "invalid_request_error",
           "param": "lora", "code": "lora_not_applicable", "unsupported_modules": ["token_embd.weight"]}}
```

`code` is `lora_not_found` for a name that is not loaded, `model_not_loaded` when there is no
model, and `lora_not_applicable` otherwise; `unsupported_modules` (layers collapsed, e.g.
`blk.{0-27}.attn_output.weight`) comes with `lora_not_applicable`. `POST /v1/loras` refuses
such an adapter the same way (400, `code: "lora_not_applicable"`, `unsupported_modules`) and
registers nothing.

#### 2.2.4 Forward-path integration

The base model's forward path must apply the active LoRA at the right points. There are two integration strategies:

**Strategy A — Modify each linear op call site.** Each `LinearOp`, `TpColumnParallelLinear`, `TpRowParallelLinear`, and fused matmul must check for a LoRA delta on that weight and apply it. This is invasive but precise — only the projections named in the adapter are affected.

**Strategy B — Hook into the weight lookup.** `ModelBase._weights` and `_quantWeights` could return LoRA-augmented versions. This doesn't work for in-place fused ops (GGML graphs, CUDA graphs) that directly read the weight pointer.

**Recommended: Strategy A with a fast-path check.** Each linear op already has a `string weightName` or equivalent. Add a `LoraWeightDelta? delta = _loraRegistry?.GetDelta(weightName); if (delta.HasValue) { ... }` pattern. The check is one dictionary lookup and null test — negligible overhead when no LoRA is active.

For **GGML backends**, the LoRA product must be applied as a separate `ggml_mul_mat` + `ggml_add` node in the graph. This means the fused kernel graphs must become LoRA-aware: when a LoRA delta exists for a weight, the graph builder inserts the `(α/r)·B·A·x` nodes after the main projection. This is similar to how `QwenImage21LoraSet` injects its deltas into the native graph.

For **direct CUDA**, the same pattern applies but with device kernels:
1. `A·x` → rank-dimensional intermediate (CUBLAS SGEMM)
2. `B·(A·x)` → output-dimensional result
3. Scale by `alpha/rank`
4. Add to the main projection result

For the **CUDA graph path** (e.g., `RunCudaDecodeLayerLoop`), LoRA deltas are not capturable because the adapter can change between requests. Two options:
- **Fall back to per-op execution** when a LoRA is active (simplest, correct, ~2-3× slower decode for the LoRA request only).
- **Use device-side function pointers** in the graph so the LoRA delta is a dynamic parameter, similar to how `CudaDecodeDynParams` feeds per-token values. This keeps CUDA graph speed but is complex.

**Recommendation:** start with the per-op fallback and measure. A LoRA request is a specialized operation; a 2× decode slowdown for those specific requests is acceptable.

> **As implemented (2026-10-28).** Strategy A, made deny-by-default, so a `lora` request never
> gets base or partly adapted output with HTTP 200 (PR #282 review item 2). The delta is
> applied in one place, `ModelBase.LinearForward`; a family is allowed an adapter only when
> every forward path it runs reaches that hook for every weight the adapter touches.

**The plan.** `TextLoraBinding.Plan` decides, once per `(model, adapter)`, where an adapter
applies, and the request path reads it (`ModelService.TryResolveLoraRequest`) before any output:

1. *Model-level refusal* (`ModelBase.DescribeTextLoraModelRefusal`): more than one device or
   node (managed or native tensor parallelism, `--layer-split`, distributed), a backend other
   than cpu, ggml_cpu, ggml_cuda and cuda (Vulkan, Metal and MLX are unvalidated), a family
   that has not opted in (`SupportsTextLora`), or a GGUF adapter converted for another
   architecture.
2. *Per target*: it must be a decoder projection (`blk.N.*` or `output.weight`; never
   `token_embd`, which a tied LM head would pick up, nor vision, audio or draft weights), not a
   routed expert, a loaded weight of the delta's shape, either directly or as rows of a
   projection fused at load, and one the family runs through the hook on every path
   (`TextLoraWeightRefusal`, an allow-list per family).
3. One target that fails refuses the whole adapter. The message groups the modules by layer
   (`blk.{0-27}.attn_output.weight (...)`); the 400 also carries them as `unsupported_modules`.

| Family | Adapted weights (single device; cpu, ggml_cpu, ggml_cuda, cuda) |
|---|---|
| mistral3 | `attn_q/k/v` (fused into `attn_qkv` or separate), `attn_output`, `ffn_gate/up` (fused), `ffn_down`, `output` |
| hunyuan-dense | the same, plus `ffn_gate` and `ffn_up` when they stay separate |
| nemotron_h / nemotron_h_moe | Mamba2 `ssm_in` and `ssm_out`, `attn_q/k/v` (fused into `attn_qkv`), `attn_output`, dense `ffn_up` and `ffn_down`, `ffn_up_shexp` and `ffn_down_shexp`, `ffn_latent_in` and `ffn_latent_out` (GGUF `ffn_latent_down` / `ffn_latent_up`), the router `ffn_gate_inp`, `output`. Routed experts are never adapted |
| every other family | refused: qwen35 (`LinearForwardCached` and native graphs on every backend), deepseek4/41, glm, minimax-m3 (native executors), gemma4, gpt-oss, muse-glimmer, muse-spark, qwen4exp, diffusion-gemma (fused or native paths) |

**Fused projections, exactly.** `ModelBase.OnFusedProjectionCreated` (called by the shared QKV
fusion and by `FuseGateUpWeights`, quantized and F32) records where each source's rows live.
All targets that land on one weight become one factor pair: the members' A are stacked along
the rank (`AT [in, sum r]`) and each member's B is placed at its own rows of a block matrix
(`BT [sum r, out_fused]`, zeros elsewhere), plus any delta named on the fused weight itself.
`x @ AT @ BT` is then exactly the concatenation of the members' deltas: no re-quantization,
no SVD, adapters that touch only some members (q and v, say) work, and the hook stays one
lookup per weight. The cost is a `B` as wide as the fused weight (at gate/up, twice the size of
one member's). A sidecar NVFP4 scale is applied to the base output before the delta, which
adds to the true weight.

**Fast paths.** Every fused graph that bypasses the hook checks one predicate,
`ModelBase.IsLoraActiveForStep`, true only while a step runs with an adapter selected:
`TryFusedDenseSwiGLUFFNInto` and `TryFusedDenseFFNProject` decline then. Loading or binding an
adapter never costs base requests a fused path. A fast path over one weight asks, behind that
predicate, whether the step's adapter adapts that weight (`ModelBase.HasLoraDelta`), so the
weights an adapter leaves alone keep it. Nemotron's decode paths work this way:
`TryLinearAddInto` (the `FusedMatMulQuantAdd` residual add of `attn_output`, `ffn_down`,
`ssm_out`, `ffn_down_shexp` and `ffn_latent_out` on GGML decode tokens) and the MoE decode
residual decline for an adapted weight, so `LinearForward` and an add run instead, and
`LinearForwardInto` (which feeds `ssm_in` into the native Mamba2 decode and `ffn_latent_out`
into the MoE decode) applies the delta itself through the name-keyed hook
`ModelBase.TryApplyLoraDelta`, which `LinearForward` calls too. (These Nemotron paths belonged to the legacy
`NemotronModel`, removed in 2026-11; on the shared decoder runtime Nemotron-H applies each delta inside the step's graph,
`AppliesTextLoraInGraphs`; see [the model card](../models/nemotron.md#text-lora-adapters).)

**Audit.** Defense in depth against a bypass the allow-list missed: while an adapter is
selected, the hook stamps each weight it applies with the forward's serial, and the outermost
`Forward`, `ForwardRefill` and the opted-in families' `ForwardBatch` check that every planned
weight was stamped (`output.weight` only when the forward produced logits), throwing
`TextLoraNotAppliedException` otherwise, so the step's requests fail instead of returning
partly adapted text. Nemotron's `SpecForward` audits itself the same way. Without a selected
adapter the cost is one null check per forward and per linear. Speculation is off for adapter
steps (§2.2.5), so the draft and verify paths never run under an adapter.

#### 2.2.5 Concurrency

> **As implemented (2026-10-28).** The original sketch here (one process-wide active
> adapter, reference-counted unload) was replaced, because the continuous-batching
> engine puts several requests into one forward and the prefix cache shares state between
> them. A global adapter leaked into concurrent plain requests, was swapped or cleared by
> other requests mid-generation, and leaked into the prefix cache.

**The adapter belongs to the request.** `ModelService.TryResolveLora` resolves the `lora`
field once, before any output: `null`, `""` or `"none"` is the base model; an unknown name,
a block-diffusion model, or an adapter the loaded model cannot apply in full is a 400 that
names the modules. The resolved `TextLoraAdapter` travels on the request
(`ChatTurnContext.Adapter`, every skill-loop round and sub-agent included) to its
`SequenceState.Adapter`. Nothing a request does changes what another request runs with.

**Adapter epochs.** No forward ever sees two adapters:

1. `ContinuousBatchScheduler` keeps the running set adapter-homogeneous. A request whose
   adapter differs from the running set's waits until that set drains, and so does every
   request queued behind it (strict first-come-first-served, so neither side starves).
   Requests that share an adapter instance still batch together. Base-only traffic never
   enters this branch.
2. `BatchExecutor.ExecuteStep` checks that every scheduled sequence carries
   `SchedulerOutput.Adapter`, binds it if needed, selects it on the model
   (`IAdapterAwareModel.SelectStepAdapter`) under `GpuComputeLock`, and clears it in a
   `finally` inside the same lock region. Media encoders that yield the lock between their
   blocks therefore always see the base model, and `ModelBase.YieldGpuComputeLock` refuses
   to yield while an adapter is selected.
3. Speculative decoding is off for adapter steps (`ExecutionStepFeatures.AdapterActive`
   and the executor's arming gate): no drafter or verify trunk applies the adapter.
4. Fused fast paths that bypass `LinearForward` check one predicate,
   `ModelBase.IsLoraActiveForStep`, which is true only while a step runs with an adapter
   selected. Merely loading or binding an adapter never costs base requests a fast path.

**Cache isolation.** A sequence with an adapter gets the explicit empty
`CacheBreakpoints` policy, `SharedPrefixTokens = 0` and no public checkpoint boundaries,
so it neither reads nor publishes radix pages, retained holders, end states, live-cache
continuations or shared-prefix checkpoints. Radix keys are tokens only, and a public node
is readable from any scope, so salting the cache scope would not isolate adapters.

**Binding once.** Each `(model, adapter)` pair has one `TextLoraBinding`: the plan (which
targets apply, decided once from the loaded weights, logged once) and the factor tensors,
materialized once when the adapter is loaded (`POST /v1/loras` binds eagerly under the GPU
lock when a model is loaded) or on the first step that needs it. An adapter is applied whole
or refused; it is never applied in part.

**Reload and unload.** Loading a name again registers a new instance (a new `Generation`,
so a new epoch) and retires the old one; `DELETE` retires it too. Requests that already
hold a retired adapter finish with it. The engine releases the model's binding between
steps once no request in flight references it (`InferenceEngine.ReleaseRetiredAdapters`);
`ModelBase.Dispose` releases every binding. `GET /v1/loras` reports `in_flight` (requests
running with the adapter) instead of the old `active` flag.

**Costs.** Mixed base and LoRA traffic serializes at every adapter switch: one long base
generation delays a waiting LoRA request and everything queued behind it. LoRA turns get no
prefix reuse, so every turn of a long LoRA conversation re-prefills. Both are the price of
correctness on every serving path; per-row adapter segmentation (SGMV/BGMV-style nodes in
the fused graphs) and adapter-salted radix keys are the follow-ups that would lift them.

#### 2.2.6 Phasing

| Phase | Scope | Risk |
|---|---|---|
| **P1 — LoraRegistry + GGUF loader** | Standalone GGUF adapter format, `LoraRegistry` class, load/unload lifecycle | Medium: new code, no existing paths changed |
| **P2 — API parameter** | `lora` parameter parsing in both adapters, `/v1/models` listing | Low: additive |
| **P3 — Forward-path integration (CPU/GGML)** | Per-linear-op LoRA apply, GGML graph injection for CPU/Metal/CUDA | High: touches every model's forward path |
| **P4 — Forward-path integration (direct CUDA)** | cuBLAS LoRA products, per-op fallback when CUDA graphs are active | High: CUDA graph compatibility |
| **P5 — Concurrency hardening** | Reference counting, graceful unload under load | Medium |

**Validation and follow-ups (2026-10-28).** Covered by model-free and synthetic tests:
`TextLoraAdapterFormatTests` (both containers, PEFT refusal, DoRA/rsLoRA/patterns, trained
biases, aLoRA, flags that are not booleans, the GGUF alpha's type, per-module rank),
`TextLoraPlanTests` (model-level refusals including DeepSeek V4.1, Qwen3.5, Gemma4, tensor
parallelism and layer split, Nemotron under `--tp` and as a layer-split node included; exact
fused composites; every Nemotron projection accepted on cpu, ggml_cpu, ggml_cuda and cuda,
routed experts refused; the audit against Nemotron decode gates broken on purpose; fused FFN and
Nemotron decode paths kept for base traffic while an adapter is bound, and for the weights a
selected adapter leaves alone) and `TextLoraModelParityTests` (tiny mistral3 and hunyuan-dense
GGUFs, and Nemotron-H Mamba2 + attention + dense or MoE hybrids, with an adapter on every
adapted projection equal their merged-weights twin through prefill, decode and the engine's
batched forward, on cpu and ggml_cpu, and on ggml_cuda in the CUDA lane; the Nemotron rows run
the GGML decode paths above and, on ggml_cuda, the native Mamba2 decode). The Nemotron ggml_cuda
rows, the latent MoE fixture included, run in the shared CUDA lane with no opt-in: the native
caches whose CUDA free aborted the test host at thread or process exit are now released on model
unload and `TSGgml_Shutdown`, and the sub-4 KiB Q8_0 weights (2,176 bytes) of the latent
fixtures in `TextLoraPlanTests` and `TextLoraModelParityTests` no longer read released host
memory: the per-expert routed slices `ffn_*_exps.{e}` in both, plus the latent projections in
`TextLoraPlanTests`, whose projections default to Q8_0. The parity fixture stores its
projections as F16, so its latent projections are exactly 4 KiB and come from the device cache.
The direct `cuda` backend (the managed forward on device tensors) has its own `[CudaTheory]`
rows, run 2026-10-29 on an RTX 3080 Laptop GPU (driver 621.01) against unchanged ggml
`facdc44a`: in `TextLoraModelParityTests`, mistral3 and hunyuan-dense with every matrix F32
match the merged twin to 2.0e-6 and with layer 1 F16 to 5.4e-4, the CUDA F16 kernel's own
rounding (with no adapter, that base and an F32 copy of its weights differ by 3.7e-4);
Nemotron dense, moe and moe-latent with F32 projections to 1.8e-6, and with F16 projections
to 1.3e-3, 1.1e-2 and 5.9e-3 (base against F32 copy: 9.2e-4, 1.2e-2, 1.0e-2); the engine's
batched forward picks tokens within 0.01 (mistral3, hunyuan-dense) or 0.05 (Nemotron dense
and moe-latent) of the twin's top logit; and in `TextLoraPlanTests` every
Nemotron projection of the Q8_0 hybrids is applied, under the audit, through prefill and
decode.
Not yet validated on a real checkpoint against llama.cpp `--lora` or
`llama-export-lora` (the fixture script and its env-gated test, `TextLoraFixtureValidationTests`
with `TS_TEXT_LORA_FIXTURE_DIR`, are the tools for that run). Follow-ups: Qwen3.5 (key the delta
in `LinearForwardCached` and decline its ~12 native/fused paths), device-resident factors through
`TryGgmlF32LinearResident` (today `Ops.Addmm` re-uploads them on GGML backends), and
tensor-parallel factor sharding. The Nemotron vision tower's `v.post_ln` is unrelated to LoRA and
tracked in the Nemotron design doc.

---

## 3. Dynamic MoE at Runtime

### 3.1 Objective

Three distinct capabilities:

1. **Read MoE modules from disk at startup** (existing behavior; MoE weights are in the GGUF).
2. **Add MoE modules to a running model without reloading it** (e.g., load a coding-specific expert set after startup).
3. **Support runtime-created MoE modules** (a learning process creates new expert weights and registers them; the learning process itself is out of scope).

### 3.2 What "MoE module" means

In the codebase, an MoE layer's experts are:

```
blk.L.ffn_gate_exps.weight   (3D: [nExp, nFf, hidden] — gate projection per expert)
blk.L.ffn_up_exps.weight     (3D: [nExp, nFf, hidden] — up projection per expert)
blk.L.ffn_down_exps.weight   (3D: [nExp, hidden, nFf] — down projection per expert)
```

Plus the router: `blk.L.ffn_gate_inp.weight [nExp, hidden]`.

A "MoE module" is a set of additional experts for one or more layers, plus optionally a new router.

### 3.3 Architecture

#### 3.3.1 MoE weight format on disk

A standalone MoE module is a GGUF file containing:

```
__metadata__.base_architecture: "qwen35" or "gemma4" etc.
__metadata__.base_model_name: "Qwen3.5-35B-A3B"  (for validation)
__metadata__.module_type: "moe" | "moe_router" | "moe_experts"
__metadata__.target_layers: "10-19"               (layers this module applies to)
__metadata__.expert_type: "coding" | "math" | "general" | "custom"
__metadata__.num_experts: 4

experts.blk.10.ffn_gate.weight   [4, nFf, hidden]
experts.blk.10.ffn_up.weight     [4, nFf, hidden]
experts.blk.10.ffn_down.weight   [4, hidden, nFf]
... (one triplet per target layer)
router.blk.10.weight             [4, hidden]  (optional; if absent, reuse base router)
```

#### 3.3.2 Runtime MoE registry

```csharp
/// <summary>
/// Owns dynamically loaded MoE expert modules. Decoupled from the base model.
/// Thread-safe for concurrent reads and rare writes.
/// </summary>
public sealed class MoeRegistry : IDisposable
{
    /// <summary>Load a MoE module file. Validates architecture and layer
    /// dimensions against the base model. Returns a unique module id.</summary>
    public string Load(string ggufPath, ModelBase baseModel);

    /// <summary>Unload a module by id. Waits for in-flight requests to drain.</summary>
    public void Unload(string moduleId);

    /// <summary>Active MoE selection: which module's experts are currently
    /// swapped into which layers. Null = base model's original experts.</summary>
    /// <remarks>
    /// This is per-layer: layer 0-9 could use the base experts, layer 10-19
    /// could use a coding module, layer 20-39 could use another. The selection
    /// is set atomically.
    /// </remarks>
    public MoeSelection? ActiveSelection { get; set; }

    /// <summary>List loaded modules with metadata.</summary>
    public IReadOnlyList<MoeModuleInfo> List();

    /// <summary>
    /// Register a module whose weights were created in-process (the learning
    /// use case). The caller owns the memory and must keep it alive.
    /// </summary>
    public string RegisterInProcess(MoeModule module);

    public void Dispose();
}

public sealed class MoeSelection
{
    /// <summary>Per-layer override: layer → module id. Layers not in the map
    /// use the base model's original experts.</summary>
    public IReadOnlyDictionary<int, string> LayerOverrides { get; init; }

    /// <summary>Optional per-layer router override. If set, the MoE router
    /// for that layer uses the module's router weights instead of the base.</summary>
    public IReadOnlyDictionary<int, string> RouterOverrides { get; init; }
}

public sealed class MoeModule
{
    public string Id { get; init; }
    public string Architecture { get; init; }
    public int[] TargetLayers { get; init; }

    /// <summary>Expert weights per layer.</summary>
    internal IReadOnlyDictionary<int, ExpertSet> Experts { get; init; }
}

public sealed class ExpertSet
{
    public int NumExperts { get; init; }
    public IntPtr GateWeights;   // [nExp, nFf, hidden]
    public IntPtr UpWeights;      // [nExp, nFf, hidden]
    public IntPtr DownWeights;    // [nExp, hidden, nFf]
    public IntPtr? RouterWeights; // [nExp, hidden] — optional
    public DType WeightDtype;
}
```

#### 3.3.3 Forward-path integration

The MoE forward path in each model must become **expert-index agnostic**: instead of reading from the fixed `_quantWeights["blk.L.ffn_gate_exps.weight"]`, it reads from the active `ExpertSet` for that layer.

**For the host path** (`MoEForward` in Gemma4, GptOss, Qwen35, etc.):

```csharp
private Tensor MoEForward(Tensor hiddenState, int layer, string prefix, int seqLen)
{
    ExpertSet experts = _moeRegistry?.GetActiveExperts(layer)
        ?? _baseExperts[layer];  // the original checkpoint's experts

    var logits = experts.RouterWeights != null
        ? ComputeExpertScoresDevice(hiddenState, experts.RouterWeights, seqLen)
        : ComputeExpertScoresDevice(hiddenState, prefix, seqLen);  // base router

    // Route identically; the experts are just a different set
    var (routingWeights, selectedExperts) = MoERoute(expertScores: logits, nUsed, experts.NumExperts);

    // Run selected experts
    for each active expert e:
        int localIdx = e;  // within the module
        gateOut = matmul(experts.GateWeights[localIdx], input)
        upOut = matmul(experts.UpWeights[localIdx], input)
        activated = SiLU(gateOut) * upOut
        downOut = matmul(experts.DownWeights[localIdx], activated)
        output += weight * downOut
}
```

This is the same loop; only the weight pointers change. The per-expert matmul dispatch already works with arbitrary expert counts in the fused `ggml_mul_mat_id` path.

**For GGML fused graphs** (`GgmlBasicOps.MoEFFNPrefillSwiGLU`): the `ggml_mul_mat_id` node already accepts an expert-id → weight-matrix mapping. The `StackedExpertWeights` structure built at model load must become mutable — or the graph must be rebuilt when the active MoE module changes.

This is the hardest part. The fused graphs are captured once and replayed. Changing the expert weights invalidates the graph. Options:

1. **Re-capture on MoE swap.** Accept a one-time ~100 ms pause when activating a new MoE module. Acceptable for a manual load-once-use-many pattern. Not acceptable for per-request MoE switching.

2. **Build all loaded MoE modules into a single `ggml_mul_mat_id` node** with non-overlapping expert IDs. The graph stays the same; the router's selected expert IDs just pick from a different range. This is the preferred approach: at MoE load time, assign a `globalExpertOffset` to the module, and the router outputs `globalExpertOffset + localExpertIndex`. The base model's experts stay at offset 0.

3. **Per-request graphs.** Expensive but general.

**Recommendation:** Option 2 (offset-based expert ID space). This is how the model already dispatches within a single MoE layer; extending the expert ID space is a router modification and a wider `ggml_mul_mat_id` table. The graph is captured once with the union of all loaded expert sets, and the router (including replacement routers) outputs IDs in the correct range.

For **direct CUDA** (`CudaFusedOps.TryMoEExpertFFNDecode`): the `_cudaMoEGateUpPtrTable[layer]` and `_cudaMoEDownPtrTable[layer]` are flat device pointer arrays indexed by expert ID. Making these mutable and resizable is the path. A module load would:
1. Allocate larger pointer tables (old + new experts).
2. Copy old pointers.
3. Append new expert's device pointers (uploaded at load time).
4. Atomically swap the table pointers (`Interlocked.Exchange`).
5. Free old tables after a drain period.

#### 3.3.4 API parameter

```json
{
  "model": "qwen3.5-35b",
  "messages": [...],
  "moe": "coding-experts-v2"     // module id from MoeRegistry
}
```

The `/v1/models` endpoint lists loaded MoE modules:

```json
{
  "object": "list",
  "data": [{
    "id": "qwen3.5-35b",
    "object": "model",
    "moe_modules": [
      {"id": "coding-experts-v2", "layers": [10,11,12,13,14,15,16,17,18,19], "experts": 4, "has_router": true}
    ]
  }]
}
```

#### 3.3.5 The learning use case

The `RegisterInProcess(MoeModule module)` path is specifically for a learning process that:

1. Computes new expert weights (e.g., fine-tuned on a coding corpus).
2. Registers them in-process via `MoeRegistry.RegisterInProcess()`.
3. Optionally computes a new router.
4. The inference engine immediately starts using them (the offset-based expert ID space supports this without graph rebuild).

The learning process is responsible for weight validity. The registry only checks shape compatibility. The caller owns the weight memory — the registry does not copy it. This is the "zero-copy" path for in-process learning.

#### 3.3.6 Phasing

| Phase | Scope | Risk |
|---|---|---|
| **P1 — MoE module GGUF format** | Standalone file format, validation against base model | Low: new code |
| **P2 — MoeRegistry** | Load/unload lifecycle, in-process registration, `ActiveSelection` | Medium: concurrency model |
| **P3 — Host-path MoE dispatch** | Per-model `MoEForward` generalization to use `ExpertSet` from registry | High: touches every MoE model |
| **P4 — GGML fused-graph expert ID space** | Offset-based `ggml_mul_mat_id`, graph rebuild on module load | High: native graph changes |
| **P5 — Direct CUDA pointer table swap** | Resizable device pointer tables, atomic swap | High: CUDA memory management |
| **P6 — Router replacement** | Module-supplied router weights, per-layer router override | Medium |

---

## 4. MoE Selection via Embedding Index

### 4.1 Objective

Instead of (or in addition to) the learned per-token router, select which experts are active based on a pre-calculated embedding index. A query (the user's prompt or the current hidden state) is embedded, and the most similar expert domains are selected.

### 4.2 Use cases

1. **Domain-specific expert selection.** A coding prompt should route to coding experts; a medical prompt to medical experts. The router doesn't need to learn this — an external index already maps domains to expert sets.

2. **Expert discovery.** New MoE modules are registered with a description; the index finds the right ones for a query without hardcoded routing.

3. **Hybrid routing.** The learned router handles fine-grained per-token routing; the embedding index pre-selects which expert pool the router draws from (a two-level hierarchy).

### 4.3 Architecture

#### 4.3.1 Expert-domain embeddings

Each MoE module (and the base model's built-in experts, optionally) has a **domain embedding**: a dense vector that describes what it's good at. This can come from:

- **A description string** ("Python coding, algorithms, data structures") encoded by a sentence embedding model (the same infrastructure that already serves Snowflake Arctic Embed from the same process — see FEATURES.md "Sentence embeddings").
- **A representative prompt corpus** whose average hidden-state embedding (from the base model's intermediate layer) serves as the domain vector.
- **Manual assignment** by the operator.

The embedding is a `float[]` of configurable dimension (256–1024). It is stored in the MoE module's GGUF metadata:

```
__metadata__.domain_embedding: [0.0123, -0.0456, ...]  (JSON array of floats)
__metadata__.domain_description: "Python, JavaScript, Rust coding tasks"
__metadata__.embedding_model: "snowflake-arctic-embed-l-v2.0"  // which embedder produced this
```

#### 4.3.2 Embedding index

```csharp
/// <summary>
/// A flat L2 or cosine-similarity index over expert-domain embeddings.
/// Thread-safe for reads; rebuilt on MoE module load/unload.
/// </summary>
public sealed class MoeEmbeddingIndex
{
    private ExpertIndexEntry[] _entries;
    private float[][] _embeddings;  // normalized for cosine search

    /// <summary>Rebuild the index from the current set of loaded modules.</summary>
    public void Rebuild(MoeRegistry registry, IEmbeddingModel embedder);

    /// <summary>
    /// Select the top-K expert modules for a query. Returns module ids
    /// ordered by similarity (highest first).
    /// </summary>
    public IReadOnlyList<ScoredModule> Query(
        string queryText,   // user's prompt
        int topK = 3,
        float minSimilarity = 0.5f);

    /// <summary>
    /// Select the top-K expert modules for a query embedding (when the
    /// caller already has an embedding, e.g. from a previous step).
    /// </summary>
    public IReadOnlyList<ScoredModule> Query(
        float[] queryEmbedding,
        int topK = 3,
        float minSimilarity = 0.5f);
}

public readonly struct ScoredModule
{
    public string ModuleId;
    public float Score;  // cosine similarity [0, 1]
}

internal readonly struct ExpertIndexEntry
{
    public string ModuleId;
    public string Description;
    public float[] Embedding;  // normalized
}
```

#### 4.3.3 Selection strategies

The design supports multiple strategies that can be composed:

**A. Prompt-based pre-selection (simplest).** Embed the user's prompt, find the top-K modules, and route ALL tokens in that request through those experts' pool. This is the MVP.

```
Request → embed prompt → top-K modules → select those modules in MoeRegistry.ActiveSelection
         → run inference (all tokens use selected expert pool)
```

**B. Per-token dynamic selection (advanced).** At each layer's MoE gate, instead of `softmax(router(x))`, compute `cosine_sim(x_embedding, domain_embeddings)` for each module and select the top-K. This requires a per-token embedding projection (a small learned matrix `W_embed: [hidden] → [embed_dim]`).

This is more complex because it needs a trained projection and changes the routing dynamics. It is a research direction rather than an immediate feature.

**C. Two-level routing (hybrid).** The index selects a pool of K modules; the learned router selects experts from within that pool. This keeps the existing routing infrastructure but restricts the expert space.

```
For each layer L:
    pool = embedding_index.Query(hidden_state_embedding, K_pool)  // module-level
    expert_ids = learned_router(x, experts_in(pool))               // expert-level
```

#### 4.3.4 API parameter

```json
{
  "model": "qwen3.5-35b",
  "messages": [...],
  "moe": "auto",                  // "auto" = use embedding index
  "moe_top_k": 3,                 // number of modules to select
  "moe_min_similarity": 0.5,      // cosine similarity threshold
  "moe_strategy": "prompt" | "per_token" | "hybrid"
}
```

When `"moe": "auto"`, the server embeds the last user message using the configured embedding model, queries the index, and activates the top modules before inference starts.

When `"moe": "module-id"`, the server uses the explicitly named module (bypassing the index), as described in Section 3.

### 4.4 Alternative: Prompt-keyword matching (zero-cost MVP)

Before building the full embedding index, a much simpler alternative: match the user's prompt against each module's `domain_description` with TF-IDF or substring matching. This requires no embedding model and works for exact-match domains:

```csharp
public IReadOnlyList<ScoredModule> QueryByKeywords(string prompt, int topK = 3)
{
    var words = new HashSet<string>(prompt.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries));
    return _entries
        .Select(e => new ScoredModule
        {
            ModuleId = e.ModuleId,
            Score = (float)words.Count(w => e.Description.Contains(w)) / words.Count,
        })
        .Where(s => s.Score > 0)
        .OrderByDescending(s => s.Score)
        .Take(topK)
        .ToList();
}
```

This is a legitimate MVP: description fields like "Python coding, algorithms, data structures" match "write a Python function to sort a list" with reasonable precision. It requires no additional dependencies and can be implemented in an afternoon.

**Recommendation:** Ship keyword matching as Phase 0/1, then add the embedding index as Phase 2. The `MoeEmbeddingIndex.Query()` interface is the same for both; the implementation is swappable.

### 4.5 Phasing

| Phase | Scope | Risk |
|---|---|---|
| **P0 — Keyword matching** | Simple token-overlap scorer against `domain_description` metadata | Trivial |
| **P1 — Embedding index data model** | `domain_embedding` in GGUF metadata, `MoeEmbeddingIndex` class with rebuild interface | Low: new data, no model change |
| **P2 — Sentence embedding integration** | Use the existing embedding pipeline to produce domain embeddings for modules and queries | Low: embedding infra already exists |
| **P3 — Prompt-based pre-selection** | Wire `MoeEmbeddingIndex.Query()` → `MoeRegistry.ActiveSelection` before inference | Medium: threading, graph rebuild timing |
| **P4 — Per-token dynamic selection** | Trainable `W_embed` projection, in-router similarity computation | High: requires training, changes routing dynamics |
| **P5 — Two-level hybrid routing** | Pool selection + standard routing within pool | High: changes every MoE model's routing |

---

## Cross-Cutting Concerns

### Thread safety

All three registries (`LoraRegistry`, `MoeRegistry`, `MoeEmbeddingIndex`) follow the same pattern:
- **Read path:** lock-free. Cached pointers/snapshots that inference workers read without synchronization.
- **Write path:** exclusive lock (reader-writer lock or RCU). Writers block briefly; readers never block.
- **Drain before free:** when unloading, the write path increments a version counter and waits for in-flight requests on the old version to complete before freeing memory.

### Memory pressure

- **LoRA adapters:** `2 * rank * (in + out) * dtype_size` per linear. A rank-64 adapter on a 40-layer Qwen 3.5 model (~96 linear projections, each ~4096×4096, F32): `96 * 2 * 64 * 8192 * 4 ≈ 384 MB`. Realistic for a server with GPU memory.
- **MoE modules:** Same as the base model's expert size per layer, multiplied by the number of new experts. A coding module adding 4 experts to 10 layers of Qwen 3.5-35B (nFf=2560, hidden=4096): `10 * 4 * 3 * 2560 * 4096 * type_size` ≈ 1.2 GB (Q8_0) to 5 GB (F32). Significant but manageable for a GPU with 24+ GB.
- **Embedding index:** `N_modules * embed_dim * 4` bytes. Negligible (~4 KB per module at embed_dim=256).

### Model warm-up / graph rebuild

When a MoE module is loaded or an `ActiveSelection` changes:

- **GGML fused graphs** must rebuild if the expert ID space changes (the `ggml_mul_mat_id` table grows). This is a one-time ~100 ms pause triggered by the first request after the change.
- **CUDA graphs** must rebuild for the same reason. Same cost profile.
- **LoRA activation** may force per-op fallback (CUDA graph paths can't capture dynamic LoRA deltas). The first LoRA request pays the graph rebuild (or per-op execution) cost.

### Configuration

All features are opt-in. Existing behavior (no logprobs, no LoRA, no MoE swap, learned router only) is unchanged when no new parameters are specified.

New CLI/env flags for the server:

| Flag | Purpose |
|---|---|
| `--loras-dir <dir>` | **Implemented** (`TextLoraHostOptions`, `LoraAdapterDirectory`). The text-LoRA adapter directory. Every top-level `*.safetensors` / `*.gguf` file is loaded at startup, after the model, under its file stem. `POST /v1/loras` loads only files inside the directory, named relative to it (an absolute path only when inside), confined by `SkillPathGuard` (`..`, rooted/UNC/drive paths, sibling-prefix directories, and symlink or junction escapes are refused, as is a sibling `adapter_config.json` that escapes). Every resolution failure gets one fixed 400 message, load errors are sanitized, and at most 32 adapters are held (409). Without the directory, `POST /v1/loras` answers 403. |
| `--moe-modules-dir <path>` | Directory scanned for `.gguf` MoE module files at startup |
| `--moe-index-embedder <model>` | Embedding model for MoE selection (default: reuse the loaded embedding model) |
| `--moe-auto-selection` | Enable `"moe": "auto"` on the API (disabled by default until proven) |

Env vars:

| Variable | Purpose |
|---|---|
| `TS_LORAS_DIR` | Same as `--loras-dir`, which wins when both are set; an empty value means unset |
| `TS_MOE_MODULES_DIR` | Same as `--moe-modules-dir` |
| `TS_MOE_INDEX_EMBEDDER` | Same as `--moe-index-embedder` |
| `TS_MOE_AUTO_SELECTION=1` | Enable auto MoE selection |

### Test strategy

| Feature | Tests |
|---|---|
| Log probs | Unit: `TokenSampler` logprob extraction for greedy/temperature/top-K. Integration: `ChatLayeringTests` with `logprobs=true` on completions and responses APIs. Verify shape matches OpenAI reference. |
| Dynamic LoRA | Unit: `LoraRegistry` load/unload lifecycle, shape validation. Integration: load a small LoRA on a tiny synthetic model, verify different adapters produce different output. |
| Dynamic MoE | Unit: `MoeRegistry` load/unload, offset-based expert ID assignment. Integration: load an MoE module on a synthetic 2-layer model, verify expert weights are swapped at runtime. |
| MoE embedding index | Unit: keyword matching on module descriptions. Integration: embedding-based selection on a set of 3 modules with distinct domains, verify the right module is selected for domain-specific prompts. |

---

## Summary

**Implementation status (2026-10-19):**
- Feature 2 (Dynamic LoRA) is implemented end to end (adapter loader/registry, `ModelBase` hook, `lora` API field, `/v1/loras` admin routes, `/v1/models` listing). See `TensorSharp.Models/Lora/` and `TensorSharp.Server/Endpoints/LoraEndpoints.cs`.
- Feature 3 (Dynamic MoE) foundation is implemented: `MoeModule`/`ExpertSet` types, safetensors `MoeModuleLoader`, and `MoeRegistry` (load/unload/register/list/selection). The forward-path expert swap (P3–P6) is not yet wired — `TensorSharp.Models/Moe/`.
- Features 1 and 4 are not started.

| # | Feature | Feasibility | Effort | Key Risk |
|---|---|---|---|---|
| 1 | Log probs | **High.** Additive to `TokenSampler` and API serialization. No architecture change. | Small (2–3 days) | None. |
| 2 | Dynamic LoRA | **Medium-High.** Registry and format are straightforward. Forward-path integration touches every model's linear ops and fused graphs. | Large (2–4 weeks) | Forward-path integration is invasive; CUDA graph compatibility requires per-op fallback. |
| 3 | Dynamic MoE | **Medium.** Registry and format are straightforward. Expert ID space extension for fused graphs is the hard part. Router replacement adds complexity. | Large (3–5 weeks) | GGML/CUDA fused graph rebuild on MoE swap; mutable device pointer tables. |
| 4 | MoE embedding index | **High** (keyword MVP), **Medium** (full embedding). Reuses existing embedding infra. Two-level routing is research-grade. | Small (keyword, 1–2 days) to Large (full + two-level, 3–4 weeks) | Two-level routing changes routing dynamics and needs training; per-token embedding projection is a research project. |

### Recommended build order

1. **Log probs** (Feature 1) — highest feasibility, lowest risk, self-contained. Ship first.
2. **LoRA registry + API** (Feature 2, Phases 1–2) — establishes the dynamic-adapter pattern without touching forward paths yet.
3. **MoE keyword matching** (Feature 4, Phase 0) — quick win, demonstrates the MoE selection concept.
4. **MoE registry + API** (Feature 3, Phases 1–2) — establishes the runtime MoE loading pattern.
5. **LoRA forward-path integration** (Feature 2, Phases 3–5) — the hard part.
6. **MoE forward-path integration** (Feature 3, Phases 3–6) — the hard part.
7. **Embedding index** (Feature 4, Phases 1–3) — full semantic selection.
8. **Advanced routing** (Feature 4, Phases 4–5) — research-grade; defer until real need.