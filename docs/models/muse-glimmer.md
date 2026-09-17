# Muse-Glimmer

[← back to model index](README.md)

| Property | Value |
|---|---|
| GGUF architecture key | `muse-glimmer` |
| Source class | [`MuseGlimmerModel`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerModel.cs) (legacy per-seq) |
| Speculative drafter | [`MuseGlimmerModel.DFlash.cs`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerModel.DFlash.cs) + [`DFlashConfig`](../../TensorSharp.Models/Speculative/DFlashConfig.cs) |
| Vision encoder | [`MuseGlimmerVisionEncoder`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerVisionEncoder.cs) |
| Image processor | [`MuseGlimmerImageProcessor`](../../TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerImageProcessor.cs) |
| Example models | Muse-Glimmer-30B |
| Modalities | Text, image |
| Thinking mode | Yes (the chat template emits an `assistant to=self` reasoning channel) |
| Tool calling | Yes (ATEM XML markup in the chat template) |
| Batched / paged forward | No (legacy per-seq) |
| Fused whole-model kernel | GGML CUDA / Vulkan / Metal / CPU (persistent decode graph on all four) |
| Tensor parallelism | Yes — GGML CUDA / Vulkan, `--tp 2` max (the 30B has 2 KV heads) |

## Quick start

```bash
# text (pick the backend for your machine: ggml_cuda, ggml_metal, ggml_cpu, mlx)
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/Muse-Glimmer-30B-UD-IQ2_XXS.gguf \
  --input prompt.txt --backend ggml_metal --max-tokens 256

# image understanding (needs the mmproj)
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/Muse-Glimmer-30B-UD-IQ2_XXS.gguf \
  --mmproj models/mmproj-Muse-Glimmer-30B-Q8_0.gguf \
  --image photo.png --input question.txt --backend ggml_cuda --max-tokens 300

# DFlash speculative decoding (lossless; output is identical to plain greedy)
dotnet run --project TensorSharp.Cli -c Release -- \
  --model models/Muse-Glimmer-30B-UD-IQ2_XXS.gguf \
  --draft-model models/dflash-kquant.gguf \
  --spec-draft 15 --input prompt.txt --backend ggml_cuda
```

`--draft-model` can also be supplied as `TS_MUSE_GLIMMER_DFLASH`.

## 1. Text architecture

52 dense layers, `n_embd` 6656, `n_ff` 19968, 32 query heads / 2 KV heads,
`head_dim` 128, vocab 202048.

* **Interleaved sliding-window attention.** `muse-glimmer.attention.sliding_window_pattern`
  is a scalar period P (4 for the 30B). Layer `l` is a sliding-window layer when
  `l % P < P - 1`, so every P-th layer (l = 3, 7, 11, ... 51) is full causal —
  39 SWA + 13 full for the 30B. This mirrors `llama_hparams::set_swa_pattern(P)`
  with `dense_first = false`. The window predicate is llama.cpp's STANDARD SWA
  rule: a key at `p0` is visible to a query at `p1` iff `p1 - p0 < n_swa` (2048).
* **RoPE only on the sliding-window layers**; the full layers are NoPE. The RoPE
  flavour is ggml's NORM (interleaved adjacent pairs, `mode 0`), not NeoX — the
  converter un-permutes transformers' rotate_half layout at conversion time.
* **Per-head QK RMSNorm.** The Q norm weight is synthesized at conversion to
  carry the model's `qk_scale_factor`; the K norm weight is ones.
* **Attention output gate.** `attn = attn * sigmoid(W_gate @ attn_norm(x))`,
  applied before `o_proj`. The gate is projected from the same post-norm tensor
  that feeds Q/K/V.
* **Four RMSNorms per layer.** `attn_norm` and `ffn_norm` use the model's
  `f_norm_rms_eps` (1e-5); `post_attention_norm` and `post_ffw_norm` use a
  **hardcoded 1e-8** (see `llama.cpp/src/models/muse-glimmer.cpp`). Getting this
  epsilon wrong is a silent numeric divergence.
* **Unweighted RMSNorm on the input embeddings** (no learned scale) — this
  replaces the `sqrt(hidden_size)` scaling other Gemma-like models use.
* **Dense SwiGLU FFN** (no MoE).
* **Output path:** `logits = lm_head(h)`, then `logits *= logit_scale` (0.19612),
  then tanh softcapping at `final_logit_softcapping` (20.0). The scale is applied
  **before** the softcap.

## 2. Vision tower

50-layer ViT, `n_embd` 1536, `n_ff` 8960, 16 heads (`head_dim` 96), patch 14,
LayerNorm with bias, plain 2-linear MLP with exact **erf**-GELU (not the tanh
approximation), learned 32x32 position embedding.

* **Preprocessing** is a pure stretch — no padding, no tiling. The merged-token
  grid is chosen by llama.cpp's `muse_glimmer_grid_size` (aspect-closest of the
  four floor/ceil candidates, ties broken toward more tokens, capped at 4096
  merged tokens), then the image is resized to `grid * 28` px with a
  Pillow-compatible **Lanczos-3** filter and normalized with the mmproj's
  `image_mean` / `image_std` (0.5 / 0.5).
* **Sparse window attention.** Patches are permuted into 32x32 windows (the
  window side is `sqrt(position_embd_rows)`), edge windows clipped. Layers where
  `(il + 1) % 4 == 0` or `il == n_layer - 1` attend globally; the rest attend
  only within their window. Because the permutation makes the mask
  block-diagonal over contiguous row ranges, the encoder runs attention once per
  window instead of materializing an `n_tok x n_tok` mask.
* **2D RoPE** (`theta` 10000): the first half of `head_dim` is roped with the
  1-indexed patch column, the second half with the 1-indexed row, both in the
  interleaved-pair (NORM) flavour.
* **2x2 pixel shuffle with CHANNEL-OUTER packing**: `out[o][c * 4 + s]`, not
  `s`-outer. This is the classic trap and is silently wrong the other way.
* **Adapter**: 6144 -> 4096 -> 4096 -> 6656, erf-GELU between, no biases.

The tower's 2D matmul weights are kept in their GGUF quantization on GGML
backends and fed straight to `AddmmQuant`. Dequantizing this tower to F32 would
cost ~7.4 GB and evict the language model's device-resident weights;
`TS_MUSE_GLIMMER_VENC_F32=1` restores the F32 path for A/B testing.

### Prompt plumbing

The chat template renders an image content part as a single `<|patch|>`.
`ChatTemplate.InjectMultimodalTokens` emits it, and the host (CLI or
`ModelMultimodalInjector`) expands each one into
`<|image_start|>` + N filler rows + `<|image_end|>`, where N is the encoder's
merged-token count. The filler rows are overwritten by the projected embeddings
*before* the input RMSNorm — llama.cpp feeds image embeddings straight into
`build_inp_embd` and norms the merged stream.

## 3. DFlash speculative decoding

DFlash is a **block** drafter: a separate 5-layer GGUF
(`general.architecture = dflash`) that proposes the whole speculative window in
one forward. It borrows the target's `token_embd` and `output` (lm_head) and
keeps its own SWA KV ring. Three passes:

1. **Encode** — the target's per-layer *input* residuals at
   `dflash.target_layers` (`[2, 14, 26, 38, 50]`) are concatenated into one
   33280-wide row per position, projected by `fc.weight` and RMS-normed by
   `enc.output_norm`.
2. **Inject** — that row feeds `attn_k` / `attn_v` of every draft layer; keys get
   a per-head RMSNorm and **NeoX** RoPE (the drafter's rope flavour differs from
   the target's) at the target position, and land in the drafter's ring.
3. **Draft** — `[anchor, MASK x (block_size - 1)]` runs the 5 blocks
   non-causally over `[ring window | the block's own keys]` and the target's
   lm_head turns the result into `block_size` rows of logits. Row 0 is the
   anchor's own prediction and is discarded.

The drafter's logits get **neither** the target's `logit_scale` nor its softcap
(llama.cpp's dflash graph ends at the lm_head matmul). argmax is invariant to
both, but the acceptance confidence is not, so the per-position confidences
handed to the executor are the softmax of the raw drafter logits.

Verification is greedy against the target, so the emitted token stream is the
plain-greedy stream up to floating-point near-ties: the verify batch's GEMM has
a different shape from the 1-row decode GEMM, the logits differ in the last
bits, and a near-tie can flip. Both TensorSharp and llama.cpp show this on the
same corpus; it is the tie flip, not a verification bug.

Both the drafter and the trunk verify run as fused native graphs
([`ggml_ops_dflash.cpp`](../../TensorSharp.GGML.Native/ggml_ops_dflash.cpp)),
persistent on CUDA / Vulkan / Metal (CUDA additionally graph-captures them).
Two graphs:

* `TSGgml_DFlashInject` — `fc` -> RMSNorm -> per draft layer
  {k/v projection, per-head k norm, NeoX RoPE, `ggml_set_rows` into the ring}.
  No Q, no attention, no FFN, no LM head — llama.cpp's `build_dflash`
  early-returns at the same point.
* `TSGgml_DFlashDraftBlock` — `[anchor, MASK x (b-1)]` through the 5 draft
  blocks, then the *target's* LM head (borrowed, never duplicated), a softmax,
  and an **on-device top-1** (`ggml_argmax` + a `get_rows` gather of the winning
  probability) — two 16-element downloads instead of the whole `[202048, 16]`
  probability block.

Design points that make the persistent drafter correct:

* **Attention reads the WHOLE ring**, `kv_len = ring_rows + b`, fixed across
  steps. Attention is permutation-invariant over the KV axis, so the ring's
  circular order does not matter; a host-built mask keyed on a per-slot
  position map expresses exactly which slots are live.
* **The mask cutoff is the ANCHOR's position, not the query's.** After a
  partially-rejected draft the ring still holds keys the drafter wrote for
  positions past the anchor; those rows are stale and no query in the block may
  see them.
* **On Metal, the queue is drained before the draft ids are read.** Metal's
  `graph_compute` returns with the GPU still running and a shared-buffer
  `tensor_get` is a raw memcpy, so reading the argmax ids without a
  synchronize returned stale drafts — output stayed correct (verification
  rejects them) but acceptance silently collapsed.

### The adaptive cost governor

Speculation is only ever a speed optimization, so
[`SpeculativeExecution`](../../TensorSharp.Runtime/Speculative/SpeculativeExecution.cs)
measures ms per *emitted* token with drafting and without, and **parks** the
drafter when drafting loses. Current design (all measured-in-anger):

* the estimator is a **ratio of sums** (`sum(ticks) / sum(tokens)`), not a mean
  of per-step rates — a fully rejected step must not carry the weight of a step
  that emitted 16 tokens;
* the **first sample of each side is discarded** every probe round (absorbs
  cold graph builds on either side);
* the **worst speculative sample of the round is trimmed** (absorbs a mid-probe
  rebuild);
* `SpecWinMargin` is 1.15, and a park **backs off** (16 steps for the first
  losing verdict, doubling to 64) — the verdict right after a prefill is the
  least trustworthy and also the cheapest to get wrong;
* `Reset()` clears the verdict so a park never leaks into the next request.

The confidence floor defaults to `confMin = 0.35` (llama.cpp's `p_min` defaults
to 0 on this path). Neither setting wins everywhere; making the floor adaptive
(a small bandit over `{0, 0.15, 0.35, 0.6}`) is the natural next step.

`SpecPrefillChunkSize` (env `TS_DFLASH_PREFILL_CHUNK`, default 1024) sets the
width of the **trunk** forward used while a DFlash prefill catches the drafter
up. It is clamped to what the two rings can absorb in one forward (the
drafter's `RingRows` = 2080 and the trunk's SWA ring `rows - n_swa`).
Shrinking it multiplies the number of full trunk forwards a long prompt pays;
the historical hardcoded 128 was the single biggest DFlash prefill cost.

## 4. Parity with llama.cpp

`InferenceWeb.Tests/MuseGlimmerParityTests.cs` checks the implementation against
golden outputs captured from a `llama-server` running the same GGUF
(`.parity/gen_ref.py`, `.parity/gen_ref_long.py`):

```bash
llama-server -m Muse-Glimmer-30B-UD-IQ2_XXS.gguf -ngl 99 -c 8192 --port 8899
python .parity/gen_ref.py      http://127.0.0.1:8899 .parity/ref_text.json
python .parity/gen_ref_long.py http://127.0.0.1:8899 .parity/ref_text_long.json

TS_TEST_MODEL_DIR=<model dir> TS_TEST_GGML_BACKEND=metal \
TS_MUSE_GLIMMER_BACKEND=GgmlMetal \
dotnet test InferenceWeb.Tests --filter MuseGlimmerParityTests
```

Two harness details that cost an afternoon each if rediscovered:

* `TS_MUSE_GLIMMER_BACKEND` takes the **enum name** (`GgmlMetal`, `GgmlCpu`,
  `Mlx`, `GgmlCuda`), not the CLI spelling — an unparsable value silently falls
  back to `GgmlCuda`. `TS_TEST_GGML_BACKEND` (`metal`/`cpu`/`cuda`) must agree,
  because a module initializer pins the process-global GGML backend before the
  first test.
* When capturing goldens, greedy is `"temperature": 0.0` and **nothing else**.
  Passing `"samplers": []` to llama-server skips the temperature sampler and the
  final dist pick samples the RAW distribution — coherent-looking,
  **non-deterministic** goldens (two identical requests return different
  tokens).

Results on the Apple M5 Pro host (2026-08-14, IQ2_XXS, goldens from
llama.cpp b10385):

| Backend | Tokenizer | 5 greedy continuations (28 tok) | Long context (5062-token prompt) | DFlash lossless |
|---|---|---|---|---|
| Mlx | identical | 5/5 token-identical | token-identical | 5/5 |
| GgmlMetal | identical | 5/5 token-identical | near-tie flip (see below) | 5/5 |
| GgmlCpu | identical | 3/5 token-identical, 2 near-tie flips at index 24/12 | token-identical | 3/5 (flips the same 2 prompts) |

**The near-tie flips are not correctness bugs.** At the long-context divergence
point llama.cpp's own top-2 logprobs are ' rising' −1.5323 vs ' The' −1.5414 —
a 0.009-nat margin. Metal picks ' The' and produces the same content
("The population trend was rising") in different words; the per-op and fused
Metal paths agree with each other, and the flip reproduces with kernels from
before the current optimization pass. IQ2_XXS at long context simply leaves
near-ties that different backend kernel stacks resolve differently — the same
behaviour the two engines show against each other. CUDA-host runs historically
matched the long golden token-for-token.

The vision geometry checks (`ComputeTargetSize` / `ComputeTokenCount` against
`muse_glimmer_grid_size`; 1024x1024 -> 1036x1036 -> 1369 tokens, 336x336 -> 144
tokens) and the image-description near-verbatim match are unchanged from the
CUDA-host validation.

## 5. Performance — Apple Silicon (2026-08-14)

Measured on an Apple **M5 Pro** (6 P-cores + 12 E-cores, 48 GB unified memory,
Metal 4 with the tensor API active), macOS 26.6.
`Muse-Glimmer-30B-UD-IQ2_XXS.gguf` (10.0 GB), greedy, engines alternating.
llama.cpp `a4a4c51f3` (b10385, 2026-08-12) built with Metal, `-fa 1`; the
vendored ggml (`8846b79`) is byte-compatible with it. TensorSharp numbers are
`--benchmark` (prefill tok/s; decode tok/s **including** greedy sampling on the
host), llama.cpp numbers are `llama-bench` (tg excludes sampling), so the decode
comparison slightly favours llama.cpp.

### ggml_metal vs llama.cpp Metal

Engines ran back-to-back in one session (the SoC throttles over a long session,
so cross-session absolute numbers move a few percent; within-session ratios are
stable — llama.cpp's own tg64 repeated at 22.22 and 22.29 across sessions).

| Measure | llama.cpp | TensorSharp | TS / llama.cpp |
|---|---:|---:|---:|
| prefill 512 | 427.2 | 413.6 | 0.97x |
| prefill 2048 | 407.7 | 392.1 | 0.96x |
| prefill 16384 (whole prompt, 0→16K) | bracket: 407.7 (pp2048\@d0) … 286.6 (pp8192\@d16K) | 320.9 | ≈0.93x vs the bracket midpoint |
| decode, ~512 ctx | 22.29 (tg64\@d0) | 21.2 | 0.95x |
| decode, ~2048 ctx | ≈21.9 (interp d0…d4096) | 20.7 | 0.95x |
| decode, 16384 ctx | 18.81 (tg64\@d16384) | 18.0 | 0.96x |

TensorSharp's decode column *includes* host greedy sampling (llama-bench's tg
excludes sampling entirely); the model-only figure is ~0.5% higher. Where the
optimization pass started, decode ratios were 0.94x/0.94x/0.94x with a shape
that worsened with context (the per-token graph rebuild grows with nothing, but
the O(context) mask refill does); the persist/replay port is the single biggest
contributor — same binary, same session: decode at 2K context is 19.5 tok/s
with `TS_MUSE_GLIMMER_PERSIST=0` and 21.3 with the replay path (+9%).

### ggml_cpu vs llama.cpp CPU

llama.cpp run with `--device none -ngl 0` (with a Metal build, `-ngl 0` alone
still op-offloads batch≥32 matmuls to the GPU). llama.cpp defaults to P-cores
only (`-t 6` on this machine); TensorSharp's ggml CPU backend now defaults to
**all physical cores** (18 here) because this workload scales with the E-cores
— llama.cpp itself moves from 3.69 to 7.87 tg when handed `-t 18`, while its
prompt throughput *drops* with E-cores and ours rises.

| Measure | llama.cpp `-t 6` (its default) | llama.cpp `-t 18` | TensorSharp ggml_cpu (default) |
|---|---:|---:|---:|
| prefill 256 | 25.1 | 23.6 | 8.9–9.2 |
| decode (short ctx) | 3.69 | 7.87 | 6.8–8.2 |

Decode is the headline: **parity with llama.cpp at matched threads (0.86–1.04x
across probes) and ~2x llama.cpp's own default configuration** — from a
starting point where the per-op path could not finish a 256/16 benchmark in 20
minutes (≈940 synchronous graph submissions per token, each spawning a
disposable 4-thread pool).

**Prefill is a known-open gap (≈0.4x at matched threads).** The signature is
precise: TensorSharp's per-prefill-token cost ≈ its per-decode-token cost at
every thread count (1/6/12/18), i.e. the batch dimension amortizes nothing,
where llama.cpp gets 3–7x per-token amortization from cache-blocked weight
reuse. It is not the fused graph (the per-op path measures the same), not the
SWA ring, and not thread count (all ruled out by direct A/B on the same
binary). Tracked as the next CPU work item.

The pure-managed `--backend cpu` is the never-touch-native correctness
reference (`NativeDequant.PreferManaged`), not a serving backend: it runs this
model at 2.7 prefill / 0.2 decode tok/s (IQ2_XXS has no direct integer-dot
plan in `ManagedQuantizedOps`, so every dot re-expands weight rows to F32).
Use `ggml_cpu` for real CPU inference.

### mlx backend

The MLX backend is correct and now stays on device for every prefill chunk
(see change 8), but its hand-written IQ-quant matmul kernels do not use the
M5 tensor API and are the wall on both axes. On the same machine, same GGUF:

| Measure | ggml_metal | mlx |
|---|---:|---:|
| prefill 512 | 413.6 | 29.0 |
| prefill 4096 (multi-chunk, banded) | ~392 | 27.7 |
| decode, short ctx (real generation) | 21.2 | 14.2 |
| decode, ~5K ctx (real generation) | ~20 | 11.7 |

The MLX decode figure is the **pipelined greedy** path a real generation uses
(device argmax, chained steps, zero per-token logits readback — 0 host copies
over a 96-token generation). The `--benchmark` decode mode measures the per-op
MLX path instead, which falls to a host attention loop at depth and is not
representative. On Apple Silicon, `--backend ggml_metal` is the recommended
backend for this model; `--backend mlx` is the MLX-ecosystem integration path
(its hand-written IQ-quant matmul kernels, not its architecture, are the gap).

### DFlash on this hardware

Speculation does not pay at short context on an M5 Pro — for either engine.
Same 54-token prompt, 256 greedy tokens:

| Engine | plain decode | DFlash decode |
|---|---:|---:|
| llama.cpp (`--spec-type draft-dflash`) | ~22.3 | 8.2 (0.37x) |
| TensorSharp (`--draft-model`) | 20.7 | 13.9 (0.67x, acceptance 70.9%) |

The M5's GPU makes a verify batch and the drafter's own forward expensive
relative to a plain token (the CUDA hosts this feature was built on have the
opposite ratio, and there DFlash is a 1.3–5x win). TensorSharp degrades much
less than llama.cpp does here, and the adaptive governor keeps the drafted
path within ~5% of the best it can reach once a drafter is attached — but if
latency matters on Apple Silicon today, run plain decode.

### What the 2026-08-14 pass changed

Every optimization below was verified numerics-neutral (byte-identical greedy
continuations across the A/B envs on the same binary) before it was kept.

1. **The persistent decode graph now covers Metal and CPU**
   (`ggml_ops_muse_glimmer.cpp`, was CUDA/Vulkan-only). Metal has no CUDA-graph
   analog — every submit re-encodes the nodes — but the replay path still
   removes the per-token graph metadata rebuild (~2,000 nodes), ~790 tensor
   re-binds, the gallocr lifetime re-plan, 104 small norm re-uploads and the
   O(context) full-class mask regeneration (the replay extends that mask by
   2 bytes per token instead). Same-binary attribution at 2K context on Metal:
   decode 19.5 -> 21.3 tok/s (+9%).
2. **Metal graphs get the backend's `graph_optimize` reorder** (alias-aware
   reorder that widens the encoder's concurrent sets), but on **persist builds
   only**: `ggml_backend_sched` does this for llama.cpp automatically, a direct
   `graph_compute` call does not. Running it per transient build measured as a
   net loss (the reorder costs more than one submit recovers), so prefill
   chunks skip it.
3. **In-graph embedding is now the default on Metal and CPU.** On unified
   memory the quantized `token_embd` binds zero-copy from the GGUF mmap, so the
   discrete-GPU trade-off (pinning a second ~1.1 GB tensor) does not exist.
   Removes a host row-dequant, a tensor alloc and an RMSNorm dispatch per
   decoded token, and a 54 MB hidden-state upload per 2048-row prefill chunk.
4. **Mask fills are span fills, parallelized.** `fill_mg_mask` /
   `fill_mg_ring_mask` write three block-fills per row instead of a per-element
   loop, fanned across up to 8 threads above 2M entries. A 2048-row chunk at
   64K context is a 256 MB mask on the backends without a device-side fill
   kernel (Metal, CPU); this was tens of milliseconds per chunk, single-threaded.
5. **The shared ggml CPU backend got real threading.** A bare
   `ggml_backend_cpu_init()` runs `GGML_DEFAULT_N_THREADS` (4) and spawns a
   **disposable thread pool per graph_compute** — the per-op path paid ~940
   pool spawn/join cycles per decoded token, on 4 of 18 cores. The backend now
   pins a persistent `ggml_threadpool` sized to ALL physical cores
   (`hw.physicalcpu`; llama.cpp defaults to P-cores only, which leaves 2x on
   the table for this workload's decode). `TS_GGML_CPU_THREADS` overrides.
6. **The fused whole-model kernel now runs on GgmlCpu** (it was the only fused
   kernel in the repo that excluded CPU; GPT-OSS and Gemma 4 already include
   it). One graph per token instead of ~940 synchronous per-op submissions.
   The historical "1024-row fused graph faulted the CPU backend during warmup"
   did not reproduce — the 2048-row warmup and the whole parity suite pass.
   `TS_MUSE_GLIMMER_FUSED_CPU=0` restores the per-op CPU path.
7. **GgmlCpu keeps uniform KV caches (no SWA ring).** A ring is read whole
   (slots are not in position order), and ggml-cpu's flash-attention evaluates
   every KV column, masked or not — so 39 of 52 layers would pay the full
   4352 ring rows of attention at ANY depth, where the uniform moving span
   costs `pad256(window + chunk)` only once the context is actually that long.
   The GPU backends keep the ring: their fixed graph shape is what preserves
   the persistent graph (and the CUDA capture), and their flash kernels skip
   fully-masked blocks.
8. **MLX prefill stays on device for every chunk.** The fast-SDPA path used to
   accept only the first prefill chunk; chunks ≥ 2 fell back to a chain that
   downloaded the whole KV cache to the host, expanded GQA 16x in F32, and
   round-tripped a `[32, seqLen, kvLen]` score tensor for host-side masking —
   per layer, per chunk. `MlxFusedOps.TryPrefillAttentionBanded` now builds the
   causal(+SWA) band mask **on device** (two aranges + compares + where, passed
   to `mlx_fast_scaled_dot_product_attention` as an array mask), and
   sliding-window layers narrow the cache read to
   `[qStart − window + 1, total)` so long prefills do not score out-of-window
   keys. First chunks keep the plain `"causal"` string mask, which skips the
   mask array entirely.
9. **Two latent Metal races fixed** (correctness, found by inspection, both
   pre-existing): Metal's `graph_compute` is asynchronous and a shared-buffer
   `tensor_get` is a raw memcpy, so (a) the DFlash capture rows and (b) the
   drafter's argmax ids could be read mid-flight. (a) corrupts the drafter's
   encoder features; (b) silently verifies stale drafts — output stays correct,
   acceptance collapses. Both paths now synchronize before the read.

### Engineering notes that still govern the design

* **Why fused:** the per-op forward submits ~600–940 GGML ops per token, each
  with host-visible overhead; every model in this repo that reaches
  llama.cpp-class decode does it as ONE whole-model graph per forward
  (`TSGgml_MuseGlimmerModelForward` — all 52 layers, final norm, LM head,
  logit scale and softcap). `TS_MUSE_GLIMMER_FUSED=0` forces the per-op path.
* **Persistent, capturable graphs.** A decode graph is built once with stable
  tensor addresses (raw `ggml_init` + `ggml_backend_alloc_ctx_tensors`, not a
  gallocr, whose lifetime packing moves addresses). Topology is held
  byte-identical between steps by writing KV with `ggml_set_rows` (the write
  row is an I64 *input*) and reading a window padded to a 256-row stride with
  an F16 mask input — a rebuild happens once every 256 tokens rather than every
  token. On CUDA the replay is additionally graph-captured. The pool is keyed
  by `(model, KV holder, n_tokens)` so 1-row decode and k-row verify shapes
  never evict each other.
* **One mask per attention class, not one per layer.** The mask depends only on
  `(window, n_tokens, start_pos)`; every sliding-window layer shares one tensor
  and every full layer the other (52 masks -> 2). llama.cpp has always done
  this (`build_attn_inp_kv_iswa`). On CUDA the prefill masks are filled by a
  device kernel; Metal/CPU fill on the host (parallelized, see above).
* **Prefill is chunked** at `TS_MUSE_GLIMMER_PREFILL_CHUNK` (default 2048),
  exactly as llama.cpp splits at `n_ubatch`; one 16K-row graph would need tens
  of GB of activations. Multimodal prompts are never chunked (vision rows are
  injected at absolute offsets). Prefill goes through the shared reuse-gallocr
  (lifetime-packed intermediates); decode does not (stable addresses win).
* **The SWA ring** (GPU backends): 39 of 52 layers never look back past 2048,
  so they get a `pad(n_swa + chunk + 1, 256)` = 4352-row ring indexed by
  `position % rows` instead of full-context caches — 29% of a uniform cache at
  64K. The `+1` is load-bearing (sized without it, a 4651-token prompt diverged
  from llama.cpp on the first decode step). The kernel reads the whole ring
  (slots are not in position order); the mask carries liveness. The ring only
  arms when the fused kernel is available; if the fused forward declines while
  the ring is armed, the per-op path throws rather than returning quietly wrong
  logits. `TS_MUSE_GLIMMER_SWA_RING=0` restores uniform sizing.
* **A rewind on a wrapped ring is bounded by its slack.** Truncating the cache
  only moves the head; rows the wrap overwrote do not come back, and the next
  query still attends a whole window behind the new head. So once a sequence
  is longer than the ring, a rewind is accepted only while
  `cached - target <= rows - n_swa - 1` (2303 tokens at the default chunk,
  which covers the engine's 16-token live-cache rewind). A deeper one is refused
  — `CanTruncateKVCache`/`TryTruncateKVCache` say no and the turn re-prefills,
  `TruncateKVCache` throws. An unwrapped ring, a uniform cache and a drop to 0
  rewind to any depth, as before (`KvBlockTransferRingTests`).
* **The padded KV window is materialized on CUDA only where ggml's own
  flash-attention VEC kernel would be selected** — that kernel misreads a
  truncated-prefix K/V view (all 16 query heads sharing a KV head return the
  same wrong vector). `kv_window_needs_cuda_flash_attn_copy` mirrors
  `ggml_cuda_get_best_fattn_kernel`: with `gqa_ratio >= 2` the copy is skipped
  on Turing/Ampere outright and on Ada+ for windows ≥ 8192 rows (the MMA kernel
  honours strides). Taken unconditionally the copy was 26 `ggml_cont` nodes per
  decode step and up to 3.3 GB/token of traffic at 128K. `TS_KV_FATTN_COPY`
  (`0`/`force`) pins either behaviour; re-check the mirrored heuristic on any
  `ExternalProjects/ggml` bump. Metal needs no copy — its flash kernels address
  K/V purely via strides.
* **KV caches are zero-filled at allocation** on the GPU backends: the fused
  kernel reads the *padded* window, whose extra rows are masked with `-inf`,
  and `-inf + NaN` is still NaN, so those rows must be finite.
* **SwiGLU is one `GGML_OP_GLU` node** (`swapped=false` applies SiLU to the
  first half — the kernel is authoritative, `ggml.h`'s comment is not, and the
  parity suite fails loudly if the halves are swapped). The K/V write path uses
  `ggml_set_rows` on a `0,2,1,3` permute directly (`ggml_is_contiguous_rows`
  is the actual precondition; the `ggml_cont` copies it replaced were pure
  overhead).
* **Shrinking the SWA ring by lowering the prefill chunk makes everything
  worse** (measured on CUDA at 64K: chunk 2048/1024/512 -> prefill 476/443/422,
  decode 16.1/15.7/15.1 tok/s). Decode is not KV-bandwidth bound at IQ2_XXS —
  the matvec is ALU-bound — so the smaller ring buys nothing and the smaller
  chunk costs GEMM efficiency.

## 6. Tensor parallelism

`--tp 2` splits the model across two GPUs on the GGML CUDA / Vulkan backends.
Two is the ceiling for the 30B: it has **2 KV heads**, and no model in this repo
replicates KV heads when `num_kv_heads < tp`.

| Weight | Split |
|---|---|
| `attn_q` / `attn_k` / `attn_v` / `attn_gate` | column-parallel (by head) |
| `ffn_gate_up` | column-parallel, **per segment** — each half of the fused `[gate\|up]` is split independently |
| `attn_output` / `ffn_down` | row-parallel → AllReduce |
| all four per-layer norms, `attn_q_norm`, `attn_k_norm` | replicated (the QK norms are per-head `[headDim]` vectors; the Q norm also carries the folded `qk_scale_factor`) |
| `output_norm`, `token_embd`, `output` | replicated; the tail runs on rank 0 |

The attention output gate is column-parallel alongside Q and is applied **inside**
the per-rank region, before the row-parallel `o_proj`. Both AllReduce points land
on the raw matmul output, **before** the 1e-8 post-norms — RMSNorm is non-linear,
so reducing after it produces coherent-looking but wrong output.

`TSGgml_MuseGlimmerModelForward` takes a `tp_degree` / `tp_plan_out` pair: in TP
mode it builds each rank's graph and returns a `TpRankPlan` instead of executing,
so the driver runs all ranks with the collectives at the segment boundaries.

Measured on 2× RTX PRO 4000 Blackwell 24 GB (PCIe), prefill 512 / decode 64,
best of 3 — the only TP host measured to date:

| Model | | prefill tok/s | decode tok/s |
|---|---|---|---|
| 30B-UD-IQ2_XXS (10.2 GB) | `--tp 1` | 1171 | 40.2 |
| 30B-UD-IQ2_XXS | `--tp 2` | **1569** (1.34×) | **63.2** (1.57×) |
| 30B-Q8_0 (28.2 GB) | `--tp 2` | 1691 | 34.3 |

`--tp 2` is byte-identical across repeat runs and tracks the `--tp 1` greedy
continuation for 468 of 500 characters before a benign paraphrase divergence
(row-parallel partials sum in a different order). The Q8_0 has no single-GPU row
on that machine — 28.2 GB does not fit one 24 GB card.

DFlash speculative decoding and pooled KV block snapshots follow the single-GPU
path only; multi-turn reuse under `--tp` comes from live-cache continuation.

## 7. Environment variables

| Variable | Effect |
|---|---|
| `TS_MUSE_GLIMMER_FUSED` | `0` = disable the fused whole-model kernel everywhere (per-op A/B) |
| `TS_MUSE_GLIMMER_FUSED_CPU` | `0` = per-op path on GgmlCpu only (the pre-2026-08-14 default) |
| `TS_MUSE_GLIMMER_PERSIST` | `0` = disable the persistent/replayed decode graph, rebuild every call |
| `TS_MUSE_GLIMMER_INGRAPH_EMBED` | `1` = force the in-graph embedding stage on any backend, `0` = force it off (default: on for tied LM head, Metal and CPU) |
| `TS_MUSE_GLIMMER_DFLASH` | DFlash drafter GGUF path (same as `--draft-model`) |
| `TS_MUSE_GLIMMER_VENC_F32` | `1` = dequantize the vision tower to F32 (A/B; ~7.4 GB) |
| `TS_MUSE_GLIMMER_VENC_FUSED` | `0` = disable the CUDA fused vision-block/flash-attention path |
| `TS_MUSE_GLIMMER_GELU_TANH` | `1` = tanh GELU approximation in the tower instead of exact erf |
| `TS_MUSE_GLIMMER_VENC_TRACE` | `1` = per-stage checksums of the vision residual stream |
| `TS_MUSE_GLIMMER_LAYER_TRACE` | `1` = residual checksum entering every layer (fused and per-op emit the same format, so diffing localizes a divergence to a layer) |
| `TS_MUSE_GLIMMER_LAYER_TRACE_POS` / `_N` / `_DIR` | first traced position / how many forwards / raw F32 dump dir |
| `TS_MLX_MUSE_GLIMMER_EVAL_EVERY_N_LAYERS` | MLX per-op lazy-graph flush interval (default 4, `0` disables) |
| `TS_MLX_PIPELINED_DECODE` | `0` = disable the MLX pipelined greedy decode fast path |
| `TS_PREFILL_CHUNK` | Prompt chunk size for `ForwardRefill` (default 2048) |
| `TS_MUSE_GLIMMER_PREFILL_CHUNK` | Tokens per prefill forward (default 2048, `0` disables chunking) |
| `TS_MUSE_GLIMMER_SWA_RING` | `0` = size every layer for the full context instead of ringing the SWA layers (GPU backends; GgmlCpu is always uniform) |
| `TS_MUSE_GLIMMER_SWA_ROWS` | Override the SWA ring size in rows (diagnostics) |
| `TS_DFLASH_FUSED` | `0` = disable the fused DFlash drafter (per-op A/B) |
| `TS_DFLASH_PERSIST` | `0` = rebuild the DFlash graphs every step instead of replaying |
| `TS_DFLASH_PREFILL_CHUNK` | Tokens per **trunk** forward while a DFlash prefill catches the drafter up (default 1024) |
| `TS_KV_FATTN_COPY` | `0` = never materialize a padded KV window (reproduces the ggml-cuda flash-attention **vec** fault); `force` = always materialize it |
| `TS_GGML_CPU_THREADS` | Thread count for the shared ggml CPU backend (default: all physical cores) |
