# Design: closing the Gemma-4 long-prompt prefill gap vs llama.cpp (CUDA-graph prefill)

Status: **PARITY/BEAT ACHIEVED (2026-06-30)** with three changes — the CUDA-graph rewrite (below)
was NOT needed. E4B end-to-end vs llama went 0.78–0.91x → 8k 0.98x, **16k 1.06x, 32k 1.09x**,
64k 0.98x, **128k 1.01x** (beats at 16k/32k/128k; 0.98x parity at 8k/64k). What shipped:
(1) in-kernel PLE, (2) KV pre-grow, (3) on-GPU causal mask (below).

1. **In-kernel PLE gather** (the big win). Phase-0's "priority-1 memcpy" turned out to be the
   per-layer-embedding (PLE) device→host→device round-trip (~88 MB×2/chunk at 2k tail, ~352 MB×2
   at 8k), NOT KV residency (residency is fine — a red herring). The verify graph now reproduces
   the FULL PLE on-device — `(sqrt(ple_dim)·get_rows(per_layer_token_embd, ids) +
   rmsnorm((hidden@per_layer_model_proj)/sqrt(hidden), per_layer_proj_norm)) / sqrt(2)` — from the
   resident quantized table; only the token-id list is uploaded. `TS_G4_PLE_IN_KERNEL` (default on),
   text-only, get_rows-safe quant table. Byte-identical (PrefillBench MATCH). This cut the 64k
   busy 88%→94.7% and the kernel-floor 1.35x→1.08x.
2. **KV pre-grow** (`IModelArchitecture.PrepareForPrefill`): pre-size the grow-on-demand global KV
   cache to the prompt length at the first prefill chunk (free at start_pos 0, no committed K/V to
   copy) — kills the incremental doubling grows that re-copied+round-tripped the cache (~+5–7% at
   16k/64k).

3. **On-GPU causal mask** (Option A below, implemented): `ggml_ops_mask.cu` fills the verify's
   `[kvLen,N]` F16 causal(+windowed) masks straight into their device buffers (fill on stream 0,
   one sync before graph compute) instead of the O(N·kvLen) host fill + H2D upload. Bit-identical,
   `TS_G4_GPU_MASK` (default on), CUDA text-only. Isolated +4% @8k, +8.7% @64k, +4.4% @128k
   (mask ∝ kvLen, so it helps most at long context — pushed 128k 0.97x→1.01x).

Remaining 8k/64k at 0.98x (~2%) is fixed per-request overhead (HTTP, first-token, cold graph) +
inter-kernel idle. **Option B (CUDA-graph persistent verify) NOT pursued** — see §6 (huge effort,
tiny headroom; 64k kernel-floor barely clears llama even ideal). The rest of this doc is the
original investigation + the (now-moot) CUDA-graph plan, kept for reference.

---

Original notes from a profile-driven investigation (2026-06-30). This documents *why* our
GGML-CUDA prefill trailed llama.cpp on long prompts and a phased, de-risked plan to close it. It
deliberately front-loads a measurement gate (Phase 0) so we do not commit to the large rewrite
before confirming it will pay off.

---

## 1. Objective & success criteria

**Goal:** make Gemma-4 E4B (and the 12B/26B siblings) long-prompt prefill on `ggml_cuda`
match or beat llama.cpp.

**Baseline (Gemma-4 E4B Q8_0, 16 GB, sm_86), prompt-tokens / TTFT:**

| length | TensorSharp | llama.cpp | ratio |
|-------:|------------:|----------:|------:|
| 2k     | 2083        | 2365      | 1.14x |
| 8k     | 2174        | 2793      | 1.28x |
| 64k    | 1653 (→1670 after tail-chunk fix) | 2118 | 1.28x |

**Success:** ≥ parity at 8k–128k (stretch: beat), **byte-identical greedy output** vs the
current path on the dense/E-series/MoE + multimodal + MTP prefill test matrix.

---

## 2. What we already ruled out (evidence)

All measured on the numbers above; details in the `kv-pool-longprompt-hang-fix` memory note.

- **Not a slow path.** The tail is already the fused whole-model verify
  (`TSGgml_Gemma4ModelVerify`) on the fast **MMA tensor-core** flash kernel for head_dim 512
  (`flash_attn_ext_f16<512,512,16,4>`); GQA opt applies; KV padded to `%256`. `GGML_PREC_F32`
  is a **no-op** for the MMA path (the kernel doesn't read precision).
- **Not chunk sizing.** Solo-tail sweet spot is 2048 (shipped); 1024/512/8192 are all worse;
  making the fresh chunk 2048 too gives nothing at 64k.
- **Not the matmul path.** `GGML_CUDA_FORCE_CUBLAS=1` → no change (MMQ Q8_0 is fine).
- **Not KV-cache growth.** `MAX_CONTEXT` pre-alloc (no doubling grows) → +1%.
- **Not per-chunk host overhead in steady state.** Instrumented per-phase timing on a late
  2048-token chunk (start_pos≈60k): `compute≈1225 ms`, host `build+maskfill+upload≈80 ms` (6%).
- **nsys kernel mix** (steady state): MMQ matmuls ~55%, flash ~18%, small kernels
  (rms_norm / rope / quantize_mmq_q8_1 / cpy / concat / scale) ~25% — **the same ggml kernels
  llama runs.**

**Cleanest signal:** even a *single* 8k fresh chunk (one graph, no tail / swaPrev / multi-chunk
overhead) is 1.28x slower. So the gap is **per-forward execution efficiency**, not multi-chunk
or host bookkeeping.

**Working hypothesis:** the residual is GPU-idle from CPU kernel-submission latency between the
~1000 kernels of a forward. llama eliminates it with **CUDA-graph capture + replay** across its
fixed-shape 2048-token ubatches ("graphs reused = 13" in its logs). We never capture a CUDA
graph. This hypothesis is **strong but not yet proven** for the *compute-bound late* chunks —
see Phase 0.

---

## 3. Why we get no CUDA-graph replay today (grounded in the vendored ggml)

From `ExternalProjects/ggml/src/ggml-cuda/ggml-cuda.cu`:

- `ggml_cuda_graph_get_key(cgraph)` returns **`cgraph->nodes[0]`** (a tensor pointer).
- Replay requires the key's cached graph to reach `warmup_complete`: **≥ 2 consecutive computes
  with the same key and no "property change".**
- `ggml_cuda_graph_update_required` reports a change if node count, or any node's op / shape
  (`ne`) / strides (`nb`) / **data pointer** differ from the last capture.

`TSGgml_Gemma4ModelVerify` today (`ggml_ops_transformer.cpp` ~6144):

1. **Builds a fresh `ggml_context` + `ggml_new_graph_custom` every chunk.** → `nodes[0]` is a new
   pointer each call → new (empty) cache entry → **warmup never completes.** *(Primary blocker.)*
2. **`kvLen` grows every chunk** (`flash_attn_kv_length(totalSeqLen,…)`, padded only to `%256`).
   → mask `[kvLen,N]` and the K/V read view `ne` change → property change even if (1) were fixed.
3. **KV write is `ggml_cpy` into a view at a per-chunk offset** (`writePart` → `ggml_view_3d` at
   `dstSlot`/`cacheBase` → `ggml_cpy`). → the write-node **data pointer changes every chunk** →
   property change. *(Second structural blocker; this is the one llama avoids by writing the KV
   cache with `ggml_set_rows(cache, values, idx)` where `idx` is an input tensor.)*
4. **KV cache doubling grows** re-allocate the cache tensors → base pointers change → resets warmup.

**Net:** four independent things force `properties_changed = true` on every chunk, so ggml runs
the graph in "direct eval" mode (per-kernel launches) forever.

---

## 4. Requirements to enable replay

To let ggml's built-in CUDA-graph path engage for prefill, a chunk-to-chunk sequence must be
structurally identical:

R1. **Persistent graph + context** reused across chunks (stable `nodes[0]`, stable tensor
    pointers); rebuild only when the *bucket* changes.
R2. **Fixed-bucket KV padding**: round the attended `kvLen` up to a bucket (e.g. next 8192, or a
    small ladder 8k/16k/…/131k) so mask/KV-read shapes are constant within a bucket. Extra
    padded keys are `-inf`-masked (already how we pad to `%256`, just coarser).
R3. **Pre-allocated KV cache** to the run's max (no doubling grows → stable base pointers). We
    already have `MAX_CONTEXT`; make solo long-prefill pre-grow automatically.
R4. **Offset-free KV writes**: replace the per-chunk view-offset `ggml_cpy` with
    `ggml_set_rows(cache, k_write, pos_idx)` (write positions as an *input* tensor), so the write
    node topology/pointers are constant. Circular-SWA layers write `pos % window` indices; global
    layers write `start_pos + i`. This is the biggest single kernel change.
R5. **In-place input updates**: `hidden`, `pos`, and the causal `mask` keep the same tensor
    pointer/shape and are refreshed via `ggml_backend_tensor_set` each chunk (content changes are
    fine — the update-check compares pointers, not contents; replay re-reads the buffers).

R1–R3 + R5 are mechanical. **R4 is the hard, correctness-critical part.**

---

## 5. Options (increasing risk / reward)

### Option A — host-overlap only (no CUDA graph). Low risk, ~5–8%.
Keep the per-chunk rebuild but hide the ~80 ms host phase behind GPU compute:
- **On-GPU causal mask**: a tiny kernel that fills the `[kvLen,N]` F16 causal(+window) mask
  directly in device memory, removing the host `get_causal_mask` fill (~42 ms) and the H2D
  upload (~33 ms). The pattern is a per-row contiguous `[lo,hi]` band — trivial to generate on
  device from `(start_pos, N, window)`.
- **Double-buffer** chunk N+1's host prep (descriptor marshalling, pos/mask) while chunk N
  computes.

Ceiling ≈ the 6% steady-state host overhead + early-chunk host-bound share. **Will not reach
parity by itself**, but it is safe, independently shippable, and reduces the surface Phase 0 has
to explain.

### Option B — persistent-graph CUDA-graph prefill. High risk, targets the full 1.28x.
Implement R1–R5. New internal entry, e.g. `TSGgml_Gemma4ModelVerifyPersistent`, that:
- Owns a persistent `ggml_context`/`cgraph`/gallocr per **(bucket, model-fingerprint)**, cached
  like the decode graph (`g_g4dc`) is today.
- Uses bucketed `kvLen` (R2), pre-grown KV (R3), `set_rows` KV writes (R4), in-place input
  refresh (R5).
- Lets ggml's `USE_CUDA_GRAPH` machinery capture on the 2nd chunk of a bucket and replay after.
- Falls back to the current per-chunk path when: multimodal `is_except` spans present, MoE verify,
  first/last partial chunk, or any capture-incompatibility (`ggml_cuda_graph_check_compability`
  returns false, e.g. split buffers).

This mirrors what already works for decode (persistent captured graph) — the novelty is doing it
for a **multi-token** graph with a **moving KV write** and **bucketed masks**.

---

## 6. Recommended phased plan

**Phase 0 — DONE (2026-06-30). Measured, hypothesis partly revised.**
Method: env-gated per-phase timers in the verify kernel + nsys on `PrefillBench`
(`TS_PREFILL_ENGINE_ONLY=1` / `TS_PREFILL_LEGACY_ONLY=1`, `MAX_CONTEXT=65536` to remove cache
grows, `--delay/--duration` to capture only a late steady-state window), then interval-union of
CUPTI kernel+memcpy activity vs wall span.

Steady-state late window (start_pos≈20–40k, N=2048), **both** engine and legacy paths agree:

| metric | value |
|---|---|
| GPU busy (kernel∪memcpy) | **~88%** |
| pure idle (no GPU op) | **~12%** |
| memcpy (serialized, ~0 overlap with kernels) | **~11%** (≈1.1 s / 10 s window) |
| kernel-only floor speedup if all non-kernel removed | **~1.30–1.35x** |

`1653 × 1.30 ≈ 2149` ⇒ recovering *almost all* overhead would *just* beat llama's 2118. **The
margin is thin** — a partial recovery (~70%) lands ~1980, i.e. still short. Proceed only with
that expectation set.

**Two recoverable components, in priority order:**

1. **Per-chunk PLE (and hidden) device→host→device round-trip (~11% memcpy) — ROOT-CAUSED
   (2026-06-30), the concrete bounded win; lower risk than CUDA graphs.**

   Correction to an earlier mis-read: this is **NOT** a KV-cache residency failure. Env-gated
   residency counters (`TS_DBG_KV_RESIDENCY`) confirm the KV cache is fully resident on chunks 2+
   (48 non-shared K/V tensors `resident`, 0 `upload`, 0 `stream`). The 88 MB copies are something
   else.

   The real source (CUPTI: copies are on a **separate copy stream**, device↔**pinned**-host, **no
   graphNodeId**, and serialized — 0 kernel overlap, i.e. they *do* block the GPU; they sit next to
   `get_rows`/`convert_unary`/`scale_f32`): the **per-layer-embedding (PLE) input**. E4B has
   `embedding_length_per_layer_input = 256`, so `ple = N(2048)·num_layers(42)·ple_dim(256)·4B =
   88.1 MB` — exactly the copy size. `ComputePLE` gathers the PLE **on-device** (`GetRowsQuant`
   into a fresh device tensor), but `NativeGemma4ModelVerify` passes it as a host `float*` via
   `GetFloatPtr(perLayerInputs)` → `TensorComputePrimitives.GetFloatPointer` **forces a device→host
   sync (the 88 MB D2H)**, and the kernel then **re-uploads it H2D** via
   `ggml_backend_tensor_set(ple_input, ple_data, …)`. Net: the already-on-device PLE is bounced to
   host and straight back, **176 MB/chunk of pure waste**. The `hidden` input round-trips the same
   way (`GetFloatPtr` D2H + `ggml_backend_tensor_set(current,…)` H2D), but it's only ~21 MB
   (`embedding_length 2560·N·4B`). Present on **both** engine and legacy paths (both call the verify
   with `GetFloatPtr`), which is why block-capture was a red herring.

   **Fix (well-scoped, ~5–6% for PLE + a bit for hidden):** keep the PLE (and hidden) on-device and
   hand the verify the device buffer directly instead of a host pointer — either (a) a *persistent*
   device PLE/hidden buffer that `ComputePLE`/the caller writes into on-device each chunk and the
   verify binds resident (skip the `ggml_backend_tensor_set` upload), or (b) plumb the ggml device
   buffer of `perLayerInputs`/`hidden` across the C#↔native boundary and `ggml_backend_tensor_alloc`
   the graph input onto it (zero-copy). (a) is simpler and mirrors the KV residency pattern. The
   same round-trip exists on the **decode** path (`NativeGemma4ModelDecode` also takes
   `GetFloatPtr(perLayerInputs)`), so the fix helps decode too. Validate byte-identical + re-nsys
   (the 88 MB D2H+H2D pair must vanish).
2. **Inter-kernel idle (~12%) — needs CUDA graphs (Option B).** Confirmed: even the single-graph
   8k chunk is 1.28x slower and busy is only ~88%, i.e. ~12% of wall the GPU has no op running —
   CPU kernel-submission gaps across the ~1000 ops/forward. This is the part the persistent-graph
   + bucketed-KV + CUDA-graph-capture work (Option B / Phase 2) targets.

**Not attributable to:** chunk size, cuBLAS vs MMQ, precision, cache-grow doubling (all ≤ few %,
already tested), or block capture (engine≈legacy: 1650 vs 1664 t/s).

(Original Phase-0 cross-check via `GGML_CUDA_DISABLE_GRAPHS` was not runnable — this vendored ggml
has no such env; graphs only auto-disable on MoE `MUL_MAT_ID` / split buffers.)

**Phase 1 — ship Option A (safe, ~5–8%)** regardless of Phase 0 outcome. On-GPU mask + host
overlap. Gate behind `TS_G4_PREFILL_GPU_MASK` / `TS_G4_PREFILL_PIPELINE`.

**Phase 2 — Option B, only if Phase 0 is green.** Land incrementally, each step behind an env
flag and validated byte-identical before the next:
1. R3 auto pre-grow KV for solo long prefill (no behavior change, removes grow-resets).
2. R2 bucketed `kvLen` on the existing per-chunk path (correctness-only; slightly more masked
   compute — measure the cost; must be < the replay win).
3. R4 `set_rows` KV writes (dense first; the riskiest — validate SWA circular wrap + global
   append byte-identical in isolation).
4. R1+R5 persistent graph/ctx + in-place inputs; confirm ggml logs `CUDA graph … reused` and
   GPU-busy rises.

Ship Phase 2 default-on only after the full matrix passes; keep `TS_G4_VERIFY_PERSISTENT=0`
kill-switch.

---

## 7. Correctness validation (blocking gate for every Phase-2 step)

- **Byte-identical greedy** vs current path on: E4B dense, 12B dense, 26B-A4B MoE, E-series
  KV-donor (`swaFreshShared`/`swaPrev`), multimodal bidi spans, MTP verify. Lengths spanning a
  bucket boundary and a mid-bucket chunk, and a >window SWA-wrap tail.
- Existing suites: `Gemma4BatchedForwardTests`, `Gemma4MtpTests`, `EngineParallelInferenceTests`
  (real-model, opt-in), `PrefillChunkingTests`.
- SWA circular-cache wrap is the highest-risk correctness area for R4 — add a targeted test that
  prefills past the window in bucketed chunks and diffs the KV cache + logits against the per-op
  reference.

## 8. Risks & rollback

- **R4 SWA-wrap miscompute** (highest). Mitigation: land dense-global `set_rows` first; keep SWA
  on the existing write path until independently validated.
- **Bucket padding adds masked compute** — a large bucket at low `start_pos` wastes attention.
  Mitigation: bucket ladder (start small), not a single 131k bucket; measure R2 in isolation.
- **VRAM**: R3 pre-grow + persistent per-bucket graphs pin more memory. 128k already sits at
  ~14/16 GB; cap buckets and evict other-N cached graphs (as the decode path already does).
- **Capture incompatibility**: multimodal / MoE / partial chunks fall back to the per-chunk path
  (feature-gated), so those never regress.
- Every phase is env-gated and independently revertible; Phase 1 is orthogonal to Phase 2.

## 9. Effort estimate

- Phase 0: 0.5–1 day (decisive; do this first).
- Phase 1: 1–2 days (on-GPU mask + pipeline).
- Phase 2: ~1–2 weeks incl. the `set_rows` rewrite, per-bucket graph cache, and the full
  byte-identical validation matrix — with a real chance Phase 0 redirects it.

## 10. Key code touchpoints

- `TensorSharp.GGML.Native/ggml_ops_transformer.cpp` — `TSGgml_Gemma4ModelVerify` (build/mask/
  KV-write/compute), `get_causal_mask`, `flash_attn_kv_length`.
- `TensorSharp.Models/Models/Gemma4/Gemma4Model.cs` — `CanUseWholeModelPrefillVerify`,
  `NativeGemma4ModelVerify` marshalling, `EnsureCacheCapacity` (R3), decode-graph cache pattern
  to mirror (`Gemma4ResetDecodeCache`).
- `ExternalProjects/ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_cuda_graph_get_key`,
  `ggml_cuda_graph_update_required`, `ggml_cuda_graph_check_compability` (the constraints R1–R5
  must satisfy).
- Bench/verify: `benchmarks/PrefillBench` (add single-chunk-at-fixed-start_pos mode for Phase 0),
  `benchmarks/engine_comparison` (end-to-end vs llama).
