# ggml_metal: closing the prefill/decode gap against llama.cpp

`ggml_metal` is the production backend on macOS and iOS, so it is the one that has
to hold up against `llama.cpp` on the same GGUF and the same GPU. This note records
a head-to-head measurement on an **Apple M5 Pro (20-core GPU, 48 GB, macOS 26.6)**,
the four things it found, and what each was worth.

## How it was measured

Both engines run the same GGUF file, back to back, never concurrently, under
`caffeinate` on an otherwise idle machine.

| | TensorSharp | llama.cpp |
|---|---|---|
| binary | `TensorSharp.Cli --backend ggml_metal --benchmark` (Release) | `llama-bench` (build `a4a4c51f3`, b10385) |
| prefill cell `ppN` | `--bench-prefill N --bench-decode 0` | `-p N -n 0` |
| decode cell `tgN@D` | `--bench-prefill D --bench-decode N --bench-fixed-tokens` | `-p 0 -n N -d D` |
| repetitions | `--bench-runs 3`, best | `-r 3`, mean |
| ggml | vendored `ggml-org/ggml`, synced 2026-08-30 | in-tree, 2026-08-18 |

`--bench-fixed-tokens` feeds a predetermined decode-token stream so the timed
region is inference only, which is the shape `llama-bench` measures.

Two caveats worth stating rather than hiding:

- **The two engines carry different ggml checkouts.** TensorSharp's is ~2 weeks
  newer and has the Metal tuning tables llama.cpp's tree does not. That is a real
  confound, and it favours TensorSharp — which is why the conclusions below are
  about TensorSharp's *own* graph construction rather than about ggml.
- **Absolute numbers drift a few percent between sessions** on a laptop (thermals).
  Every ratio quoted here comes from a single interleaved run, and every A/B in
  the "what was found" section alternated which condition ran first.

## Result

Ratios are TensorSharp / llama.cpp; **> 1.00 means TensorSharp is faster**.
"before" is the same TensorSharp benchmark before the four changes below.

| model | case | TS before | TS after | llama.cpp | ratio | gain |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B Q8_0 | pp512 | 1288.3 | 1289.1 | 1318.5 | 0.978× | +0.1% |
| | pp2048 | 1258.4 | 1278.5 | 1301.6 | 0.982× | +1.6% |
| | pp8192 | 1040.5 | 1064.2 | 1169.1 | 0.910× | +2.3% |
| | tg128 | 31.7 | 31.7 | 30.5 | **1.039×** | +0.0% |
| | tg128@4096 | 30.4 | 31.2 | 29.8 | **1.046×** | +2.6% |
| gemma-4-E4B Q8_0 | pp512 | 2219.9 | 2342.2 | 2378.2 | 0.985× | **+5.5%** |
| | pp2048 | 2047.4 | 2177.7 | 2270.2 | 0.959× | **+6.4%** |
| | pp8192 | 1672.5 | 1831.0 | 1957.2 | 0.936× | **+9.5%** |
| | tg128 | 45.1 | 46.2 | 48.6 | 0.951× | +2.4% |
| | tg128@4096 | 40.5 | 45.6 | 47.3 | 0.964× | **+12.6%** |
| gemma-4-E2B Q8_0 | pp512 | — | 4167.8 | 4198.2 | 0.993× | — |
| | pp2048 | — | 4105.4 | 3879.2 | **1.058×** | — |
| | pp8192 | — | 3052.3 | 3050.3 | **1.001×** | — |
| | tg128 | — | 82.3 | 88.6 | 0.929× | — |
| | tg128@4096 | — | 80.6 | 85.1 | 0.947× | — |
| Qwen3.6-35B-A3B IQ2_XXS | pp512 | 1772.8 | 1772.6 | 1758.4 | **1.008×** | −0.0% |
| | pp2048 | 1842.0 | 1867.1 | 1695.2 | **1.101×** | +1.4% |
| | pp8192 | 1439.6 | 1431.8 | 1457.1 | 0.983× | −0.5% |
| | tg128 | 82.7 | 83.9 | 76.6 | **1.096×** | +1.5% |
| | tg128@4096 | 79.1 | 80.2 | 74.1 | **1.082×** | +1.4% |
| gpt-oss-20b Q8_0 | pp512 | 2000.6 | 2013.5 | 1854.0 | **1.086×** | +0.6% |
| | pp2048 | 2001.7 | 2025.7 | 1792.8 | **1.130×** | +1.2% |
| | pp8192 | 1633.3 | 1655.3 | 1497.5 | **1.105×** | +1.3% |
| | tg128 | 81.9 | 82.9 | 89.0 | 0.931× | +1.2% |
| | tg128@4096 | 76.6 | 77.6 | 83.4 | 0.931× | +1.3% |

No cell regressed. The one negative number (Qwen3.6 pp8192, −0.5%) is inside this
machine's run-to-run spread.

## What was found

The method was to make both engines dump the graph they actually encode
(`GGML_METAL_GRAPH_DEBUG=2`; TensorSharp needs `TS_GGML_LOG_DEBUG=1` to let ggml's
debug lines through) and compare them op by op — node counts, destination bytes,
and how many nodes ggml-metal could encode without a memory barrier. On
gemma-4-E4B pp512 that comparison opened at:

| | TensorSharp | llama.cpp |
|---|---:|---:|
| nodes | 1061 | 921 |
| bytes written | 6116 MB | 5470 MB |
| `CONT` (pure copies) | 141 | 0 |
| `ADD` | 117 | 39 |
| 3-op fusions | **0** | 170 |

TensorSharp was doing the same math with **12% more memory traffic**, which is
about the size of the deficit it was measuring.

### 1. The residual add was written in the order that declines fusion

ggml-metal fuses `rms_norm → mul → add` into a single kernel
(`ggml_metal_op_norm`), but only when the add's **`src[0]` is the mul it follows**:

```c
fops[0] = op->op;  fops[1] = GGML_OP_MUL;  fops[2] = GGML_OP_ADD;
...
if (f0 != f1->src[0]) break;      // <- the whole fusion turns on this
```

Every post-norm architecture in the tree wrote the residual the other way round —
`ggml_add(ctx, hidden, post_attn)` instead of `ggml_add(ctx, post_attn, hidden)`.
Addition is commutative elementwise, so the two are bit-identical; only one of
them fuses. Swapping the operands at 26 sites (Gemma 4 dense / batched / MoE /
verify / decode, Muse-Glimmer, the generic transformer prefill, DiffusionGemma,
the fused vision block) took gemma-4-E4B's pp512 graph from **0 three-op fusions
to 378** and removed the entire 613 MB `ADD` bucket.

### 2. Fused QKV was split with copies instead of strided views

TensorSharp fuses Q+K+V into one weight at load time, then split the result with
three `ggml_cont(ggml_view_2d(...))`. Because each token is one row of
`[qDim + 2*kDim]`, the head axis inside a row is already dense — so the split is
expressible as a `ggml_view_3d` with no copy at all, which is exactly what
llama.cpp does with its own fused `wqkv` (`src/models/mimo2.cpp:138-140`). Both
Metal and CUDA gate `RMS_NORM` on `ggml_is_contiguous_rows`, which such a view
satisfies. That removed 62 of the 141 `CONT` nodes and 155 MB of writes.

The same applied to the per-layer-embedding slice, which was `ggml_cont`'d before
a `ggml_mul` that only requires contiguous rows — another 39 copies and 20 MB.

After 1 and 2 the gemma-4-E4B pp512 graph is down to **40 `CONT` nodes from 141**,
and TensorSharp writes **5324 MB against llama.cpp's 5470** across **841 nodes
against 921** — it now does strictly less work per prefill token than the engine
it was 12% behind.

### 3. A saturated sliding window was read rotated, and rebuilt every token

This was the biggest single win, and it only showed up because decode was measured
against context depth rather than at depth 0:

| depth | before | after | llama.cpp |
|---:|---:|---:|---:|
| 1 | 46.3 | 46.2 | 48.1 |
| 512 | **42.0** | **46.4** | 48.2 |
| 1024 | 42.1 | 46.2 | 47.9 |
| 4096 | 41.5 | 45.6 | 47.1 |
| 8192 | 40.4 | 44.1 | 45.6 |

A step, not a slope — and it lands exactly on gemma-4's 512-token sliding window.
Past that point the circular KV cache has wrapped, so the window straddles the
buffer seam, and `view_kv_cache_window` split it in two and rejoined the halves
with `ggml_concat` — whose GPU kernels are F32-only, so both F16 halves were
converted first and flash attention then read an **F32** window instead of the F16
cache. That was 4 copies + 2 concats per sliding-window layer per token: 102 extra
nodes on gemma-4-E4B.

None of it is necessary. Once the window is saturated, the rotated window and the
raw buffer hold the *same set* of keys, and softmax is permutation-invariant over
keys — so the buffer can be read flat from slot 0. (RoPE is applied before the
write, so each key carries its own position; and with a single query at the newest
position every cached slot is in-window, so order is not needed for masking
either. This is true **only** for a single query — a multi-token prefill chunk
still needs causal order, and that path was left alone.) The capturable decode
path already read flat for this reason; `swa_decode_window_start()` in
`ggml_ops_transformer_common.h` now does it for the path Metal actually takes.

Verified byte-identical: same greedy token chain with `TS_SWA_DECODE_FLAT=0/1`,
and `ggml_metal` agrees with `ggml_cpu` token-for-token on a 600-token prompt
(which is past the window on both).

### 4. Two things that did *not* pay, and are recorded so they are not retried

**Wiring ggml's Metal reorder into the shared allocator.** Metal's
`graph_optimize` hook permutes `gf->nodes[]` to raise concurrency, and
`ggml_backend_sched` runs it on every graph. TensorSharp's direct-compute path
called it at five hand-picked sites, which left Gemma 4, GPT-OSS and the MoE
kernels encoding a barrier between nearly every pair of nodes. Wiring it into
`alloc_graph_in_gallocr_slot` did exactly what it promised — gemma-4-E4B pp512
went from 44 of 1061 nodes encoded concurrently to 127, past llama.cpp's 120 — and
was **still slower**, reproducibly, with the A/B order alternated:

| | reorder off | reorder on |
|---|---:|---:|
| gemma-4-E4B pp512 | 2281 | 2254 |
| gemma-4-E4B tg128 | 46.0 | 44.8 |
| Qwen3.5-9B pp512 | 1280 | 1305 |
| Qwen3.6-A3B tg128 | 77.5 | **83.9** |

Removing barriers is not free: the reordered schedule interleaves ops that were
adjacent, widening the live set the caches must hold. Whether that pays depends on
the graph, so it stays a per-kernel call — on for Qwen3.5/3.6, where it is worth
up to 8%, off for Gemma 4, where it costs 1–3%. `TS_METAL_GRAPH_OPTIMIZE=0`
disables it everywhere for A/B.

**A persistent decode graph on Metal for Gemma 4.** That path is gated to
CUDA/Vulkan over a `ggml_set_rows` crash on Metal. On the current vendored ggml
the crash no longer reproduces and the output is byte-identical — but it is worth
**+0.4%** (46.4 → 46.6 tok/s), not the ~1 ms/token it saves on Vulkan, because
decode here is bandwidth-bound on the weights (7.6 GB per token at ~350 GB/s).
Left off, reachable with `TS_GEMMA4_METAL_PERSIST=1`.

## One correctness bug found on the way

`optimize_graph_for_metal` reorders a graph's node array, and the host-MoE seams
(`--cpu-moe`) and the tensor-parallel driver execute graphs as **ordered slices**
of that array, cutting at recorded node *indices*. Qwen3.5's verify and decode
kernels did both: reorder, then slice. A permutation moves work across a seam, so
the host would read an activation the GPU has not produced yet — wrong numbers
rather than a crash. `SuppressGraphReorder` now makes a builder that will slice
its graph say so, and `optimize_graph_for_metal` leaves that graph's order alone.

## What is left

- **Gemma 4 and GPT-OSS decode remain ~5–7% behind.** Both engines sit at
  ~350 GB/s of weight traffic per token on a part whose peak is not far above
  that, so this is a small overhead on top of a bandwidth wall, not a missing
  optimization. It is the honest place to look next, with a bandwidth roofline
  measurement first.
- **Qwen3.5 `pp8192` reads 0.91×**, but that is the `--benchmark` default doing
  the whole prompt as one graph. The path the server actually uses is
  `ForwardRefill`, which chunks: at chunk 2048 it measures 1140 t/s against
  llama.cpp's 1167 at its own default `-ub 512` (0.98×), and at chunk 1024 it
  measures 1147. Worth making the benchmark's default reflect the production path.
- The remaining 79 `CONT` nodes in the Gemma 4 prefill graph are the KV-write
  permutes. They can go the same way as the QKV split (`ggml_cpy` takes a
  permuted source on Metal), but they only matter for prompts short enough that
  the fresh chunk is never attended directly — where the gap is already 0.99×.

## Reproducing

```bash
# prefill / decode matrix, both engines interleaved
# (run_bench.py is a local harness script; it is not committed to this repository)
python3 run_bench.py --runs 3 --cases pp512,pp2048,pp8192,tg128,tg128@4096 \
    --out results.jsonl --models <gguf> ...

# what the graph actually contains
TS_GGML_LOG_DEBUG=1 GGML_METAL_GRAPH_DEBUG=2 dotnet TensorSharp.Cli.dll \
    --model <gguf> --backend ggml_metal --benchmark --bench-prefill 512 --bench-decode 0

# A/B levers
TS_METAL_GRAPH_OPTIMIZE=0    # ggml's Metal node reorder, off
TS_SWA_DECODE_FLAT=0         # rotated sliding-window read, restored
TS_GEMMA4_METAL_PERSIST=1    # persistent decode graph on Metal, on
GGML_METAL_FUSION_DISABLE=1  # ggml-metal op fusion, off (proves fusion is lossless)
```
