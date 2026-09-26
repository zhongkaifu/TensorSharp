# DeepSeek V4 Flash (`deepseek4`)

[← back to model index](README.md) | [中文](deepseek4_zh-cn.md)

DeepSeek V4 Flash is a 284B-parameter MoE (256 routed experts, top-6 + 1 shared)
with a novel long-context attention stack: a tiny 128-token raw sliding window
per layer, plus per-layer *compressed* attention over 4:1 (CSA) or 128:1 (HCA)
block-compressed keys, with a lightning indexer selecting the top-512 compressed
rows on CSA layers. Residuals flow through 4-stream hyper-connections with
Sinkhorn-normalized mixing. Advertised context: 1M tokens (YaRN ×16).

## How TensorSharp runs it

DeepSeek V4 has three whole-model executors, reached through `--backend`:

- **`--backend cuda`**: a **direct-CUDA whole-model engine**
  (`TensorSharp.Backends.Cuda/Dsv4/Dsv4CudaEngine.cs`), independent of ggml.
  Quantized weights stream from the GGUF shards straight into per-device
  arenas and are layer-split across every visible GPU, so a model larger than
  one GPU's VRAM is hosted across several.
- **GPU backends** (`--backend ggml_cuda` / `ggml_vulkan`): the native ggml
  executor described below. ggml ships DeepSeek-V4's four architecture-specific
  ops (the three hyper-connection ops and the lightning indexer) for CPU and
  CUDA only. On any other backend `hc_pre` / `hc_post` are built instead out of
  batched `mul_mat` — they are exactly a contraction over the stream axis — so
  the whole layer stays on the accelerator; `hc_comb` (a 20-iteration Sinkhorn
  over a 4x4 matrix) and the lightning indexer still take the scheduler's CPU
  fallback, which is what is left of the Vulkan/CUDA gap. The decomposition is
  worth +34% prefill and +14% decode on Vulkan and is chosen automatically by a
  load-time `ggml_backend_supports_op` probe (`TS_DSV4_HC_NATIVE=0/1` to A/B).
- **`--backend ggml_cpu`**: the same native ggml executor on a single CPU
  compute device, with the architecture-specific ops running their scalar CPU
  kernels. Until V4.1 needed this path, the loader was asked for "any GPU" here
  and a CUDA box silently ran the GPUs instead — including for the Linux
  default, which is `ggml_cpu` when `--backend` is not passed. It now runs where
  it says, and says so once on stderr (`[dsv4] --backend ggml_cpu: DeepSeek V4
  will run on ONE CPU device, not on any GPU this host has...`) because the same
  launch line used to mean the GPUs. `--backend ggml_cuda` is how you ask for
  them now. No CPU throughput has been measured for V4 on this backend and none
  is quoted here; every number in this card is from a GPU executor.
- **`--backend cpu`**: a **100% pure C# whole-model executor**
  (`TensorSharp.Models/Models/DeepSeek4/DeepSeek4CpuExecutor.cs`) — no native
  dependencies at all. It serves the quantized weights straight from the
  memory-mapped GGUF shards and runs every op (hyper-connections with Sinkhorn
  mixing, the CSA/HCA block compressors, the lightning indexer, shared-K
  attention with sinks and inverse RoPE, sqrt-softplus MoE routing with hash
  layers) in managed SIMD code.

Both TensorSharp-native executors are built on the **shared tensor stack**
rather than private re-implementations: every buffer is an `IAllocator`-backed
`Tensor` (`CudaAllocator` pool on GPU, `CpuAllocator` on CPU), the linear
layers go through the one quantized-matmul router each backend already has
(`CudaQuantizedOps.AddmmResidentToFloat32` /
`ManagedQuantizedOps.AddmmQuantizedToFloat32`), and generic math uses `Ops`
(`Ops.RMSNorm`, `Ops.SiLUMulClamp`, …). Only genuinely DSV4-specific compute —
hyper-connections, the block compressors, the lightning indexer + top-k, the
sink attention, and the grouped MoE kernels — lives in the DeepSeek V4 files.

The native whole-model executor
(`TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp`):

- Loads the (split) GGUF directly and **layer-splits the weights across every
  visible CUDA GPU** — a model larger than one GPU's VRAM is hosted across all
  of them (the 128 GiB IQ4_XS build needs 2×80GB). This happens **by default,
  with no flag**: `--tp` is not what puts DSV4 on several cards, and it does
  not shard inside a layer either. On this family `--tp N` only caps the layer
  split at N devices, exactly like `TS_DSV4_NGPU`, which wins when both are
  set.
- Owns all DSV4 KV state on-device: raw SWA ring, CSA/HCA compressed-K caches,
  lightning-indexer cache, and the compressor state rings.
- Executes prefill/decode ubatches as single ggml graphs via
  `ggml_backend_sched`, with flash attention (512-dim shared K=V head +
  attention sinks), fused hyper-connection ops, and a shape-signature graph
  cache so steady-state decode replays a captured CUDA graph.

The C# side (`TensorSharp.Models/Models/DeepSeek4/DeepSeek4Model.cs`) handles
GGUF metadata, the `joyai-llm` BPE pre-tokenizer, the DeepSeek V4 chat template
(`<｜User｜>…<｜Assistant｜></think>`, `--think` opens `<think>`), and sampling.
The template renders tool declarations and `DeepSeek4OutputParser` parses the
DSML tool calls back, so V4 is eligible for the tool-based features: skills,
the code tools (`--code-exec`) and, on the server, [sub-agent
delegation](../multi_agent.md), which is on by default on its chat paths. No
delegation results are published for this model.

### Fitting the model: VRAM-aware split, and MoE CPU offload when you ask for it

Both GPU engines — the native ggml executor and the direct-CUDA one — size the
layer split against the VRAM each device *actually has free right now*, not
against an equal share of the bytes. Two things follow from that.

**The split is budget-proportional.** Every device is filled to the same
fraction of its own free VRAM, minus a run-time reserve for the scheduler's
compute buffers and minus the KV caches and compressor state rings the split
can compute exactly. The native executor estimates that reserve per load: 2 GiB
plus the lightning-indexer top-k transient and one ubatch of activations, so it
grows with `MAX_CONTEXT` and `TS_DSV4_UBATCH`. The direct-CUDA engine holds back
a flat 2048 MiB per device. `TS_DSV4_VRAM_RESERVE_MB` overrides either.
A device with a display attached, or one already hosting another process, gets
proportionally fewer layers instead of the OOM an equal-bytes split would hand
it.

**Routed experts spill to system RAM only when you ask.** 91% of a V4 Flash
checkpoint is routed-expert weights (137 of 151 GiB at UD-Q8_K_XL), so they are
the only knob with enough range — but offload is OFF by default here exactly as
it is for every other architecture. Choosing it silently is the wrong trade on a
host that *does* have the VRAM: it moves tens of GiB of experts to the CPU and
costs most of the decode throughput for no reason.

When the model does not fit, the loader says so and names the number that would
work, instead of loading into an out-of-memory abort:

```
[dsv4] not enough VRAM: 150.7 GiB of weights plus this context's KV caches
       against 152.9 GiB free across 7 device(s). Re-run with --n-cpu-moe 9
       (moves the routed experts of the first 9 layer(s), 28.7 GiB, to system
       RAM) or --cpu-moe to offload every layer.
```

Note what that example shows: the weights alone can look like they fit and still
not, because each device also holds its KV caches and the compute reserve
(never less than 2 GiB per device, so at least 14 GiB across 7).

`--n-cpu-moe N` / `--cpu-moe` then keep the routed experts (`ffn_gate_exps` /
`ffn_up_exps` / `ffn_down_exps`) of the first N layers in host RAM and run their
`mul_mat_id` chain on the ggml CPU backend; the router, the norms, the attention
stack and the always-active shared expert stay on the accelerator, so only
`[n_embd, n_tokens]` activations cross the bus in each direction per offloaded
layer. Asking for more layers than needed trades decode speed for VRAM.

`--backend cuda` runs the same seam on its own kernels: the offloaded layers'
experts are multiplied by `ManagedQuantizedOps` and the weighted-sum epilogue
stays the device `moe_scatter_add`, so a layer is interchangeably resident or
offloaded. Its host FFN batches every selected expert's projection into one
parallel dispatch per stage — one per (expert, projection) spent more time in
the scheduler than in the arithmetic (226 → 105 ms per token).

Measured on 3×RTX A6000 48 GB (UD-Q8_K_XL, 151 GiB of weights against 139 GiB
of VRAM, 2-socket Xeon Gold 6342 under a 23.8-CPU cgroup quota; another process
held ~15 GiB during the `ggml_cuda` runs):

| Metric | `ggml_cuda` (8 offloaded) | `cuda` (9 offloaded) | `ggml_vulkan` (7 offloaded) |
|---|---|---|---|
| Decode | 14.7 tok/s | 8.9 tok/s | 6.5 tok/s |
| Prefill (1024–2048) | 206 tok/s | — | 138 tok/s |
| Prefill (8.7K) | 195 tok/s | 43.6 tok/s | 73 tok/s |
| Host cost per offloaded layer | ~5.4 ms/token | ~8.3 ms/token | ~5.4 ms/token |

Vulkan's remaining decode gap is not the expert offload — it is the per-layer
CPU round trip for `hc_comb` and the lightning indexer, the two ops the
decomposition deliberately leaves alone (decomposing a 20-iteration Sinkhorn
costs more tiny dispatches than the boundary it removes, and expanding the
indexer would materialize an `[n_kv, n_head, n_tokens]` score tensor that is
2.2 GB at a 1024-token prefill chunk).

### DSpark and MoE CPU offload together

Both engines run the drafter alongside an offloaded MoE, and the automatic
split accounts for the drafter's own VRAM — loading one shifts exactly one more
layer's experts to the host. Whether speculation still pays depends on how fast
the host MoE is, because a verify batch of B tokens pulls B rows' worth of
experts through it:

| Engine | no DSpark | + DSpark | acceptance |
|---|---|---|---|
| `ggml_cuda` | 14.4 tok/s | **16.7 tok/s (1.16x)** | — |
| `cuda` | 8.9 tok/s | 6.2 tok/s (0.70x) | 51% |

So pair `--draft-model` with `--backend ggml_cuda` when experts are offloaded;
on the direct engine the managed host matmul is slow enough that the extra
verify rows cost more than the accepted tokens save.

Each offloaded layer reads ~80 MB of MXFP4 expert blocks per token, so the
loader offloading the *minimum* is what keeps this usable. **Size the host pool
to the CPUs the process may actually use, not to `nproc`** — see
`TS_CPU_MOE_THREADS` below.

## Usage

```bash
# point --model at the FIRST shard of the split GGUF
TensorSharp.Cli --model DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf \
    --backend ggml_cuda --chat
```

```bash
# direct-CUDA engine (no ggml): weights stream into per-GPU arenas, layer-split
# across every visible device
TensorSharp.Cli --model DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf \
    --backend cuda --chat
```

Multi-turn chat reuses the KV cache across turns (pure append); prompts that
rewind history re-prefill automatically (compressed caches cannot truncate).

```bash
# pure C# CPU inference (no native libraries; needs enough RAM for the caches,
# weights are served from the memory-mapped shards)
TensorSharp.Cli --model DeepSeek-V4-Flash-UD-IQ4_XS-00001-of-00004.gguf \
    --backend cpu --chat
```

## DSpark speculative decoding

DeepSeek ships a **DSpark** support module alongside the model (`mtp.*` in the
checkpoint): three DSV4 blocks that read the trunk's hidden states at layers
40-42 and propose a whole BLOCK of future tokens per step, plus a Markov head
that conditions each block position on the token before it and a confidence
head that predicts each position's acceptance probability. The trunk then
verifies the block in ONE batched forward and keeps the longest prefix its own
sampler would have drawn, so speculation is a speed path, not a quality change.

It is loaded as a separate drafter GGUF with `--draft-model` and engages for
single-sequence (solo) generation on **both GPU engines** —
`--backend cuda` (direct-CUDA) and `--backend ggml_cuda` (the native ggml
executor). `ggml_vulkan`, `ggml_cpu` and `cpu` have no speculative path for this
architecture and log a warning if a drafter is configured. Without a drafter
there is nothing to speculate with: `--spec` alone, `--spec-type ngram`
included, serves standard decode on this family.

Every single-sequence CLI generation path uses it: one-shot `--input`,
`--multi-turn-jsonl`, and the `--interactive` chat REPL (which streams the
accepted block token by token and reuses the cached prefix across turns). The
per-turn line reports what speculation did, e.g.
`spec=window5/accepted330of502(66 %)`.

```bash
TensorSharp.Cli --model DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf \
    --backend ggml_cuda --draft-model DeepSeek-V4-Flash-0731-DSpark.gguf \
    --input prompt.txt --max-tokens 200 --temperature 0

# Interactive chat, 4 GPUs
TensorSharp.Cli --model DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf \
    --backend ggml_cuda --draft-model DSpark-drafter-Q2K-Q8-0731.gguf \
    --interactive --think --tp 4 --max-tokens 20000
```

On the CLI, verification draws each row with whatever sampler the run
configured — argmax under `--temperature 0`, the chat sampler otherwise — so
speculation composes with `/temp`, `/top-k`, … in the REPL. One caveat under a
penalized sampler: DSpark proposes a whole block in one pass, so the
repetition/presence/frequency penalties verification applies are not applied to
the proposal, and acceptance falls as the penalized history grows. Speculation
stays off entirely for a turn carrying an image or audio attachment, whose
embeddings only the plain prefill can inject.

### On the server (TensorSharp.Server.Host)

The same drafter serves the HTTP API. Pass it with `--draft-model` — naming
the drafter enables speculation by itself (an explicit `--no-spec` vetoes it):

```bash
TensorSharp.Server.Host --model DeepSeek-V4-Flash-...-00001-of-00005.gguf \
    --backend ggml_cuda --tp 4 \
    --draft-model DSpark-drafter-Q2K-Q8-0731.gguf
```

As on the CLI, the engine draws every verify row with the **request's own
sampler**, so speculation composes with any sampling settings and the output is
whatever that sampler would have produced anyway. Penalties only cost
acceptance, and measurably little: the same prompt at `repeat_penalty` 1.1 vs
1.0 came out at 31.3 vs 32.1 tok/s (66% vs 57% acceptance — the difference is
which text was generated, not the penalty).

Two engine-level limits are worth knowing. Speculation is armed per request at
a fresh full prefill, and it serves **solo sequences only**: as soon as a second
request is in flight the planner logs
`PerSequenceFused; rejected: SpecPerSequence: multi-sequence step` and DSV4's
per-sequence slots serve the batch at normal decode speed. Concurrency is safe,
it just isn't speculative.

Measured on 4×A40 (`--tp 4`, 300-token OpenAI chat completion):

| Config | tok/s |
|---|---|
| No drafter | 25.1 |
| `--draft-model …` | **31.3 – 32.1 (1.25–1.28x)** |

`--spec-pmin` defaults to the value matching the loaded drafter — 0.35 for a
block drafter, 0.15 for a per-token draft head — so it needs no tuning. Setting
it explicitly still wins; the startup line reports which gate is in force
(`pMin=0.35, draft=block(5)`).

The two engines implement the same algorithm differently. The **ggml** engine
builds the drafter as three extra layers of the model graph: the trunk graph
captures the target features, projects them and commits the drafter's key ring
in the same pass, and the draft itself is one cached graph whose Markov chain
(`get_rows` -> `mul_mat` -> `argmax`, per block position) runs entirely
on-device. The **direct-CUDA** engine drives the same stages with its own
kernels and feeds the drafter its target features through the host. Nothing in
the C# speculative core is backend-specific.

### Getting a drafter

The drafter is NOT in the target GGUF: every GGUF conversion of DeepSeek V4 drops
the `mtp.*` tensors (`DeepSeek-V4-Flash-0731-UD-Q8_K_XL` has `blk.0`-`blk.42` and
nothing else). Either download a pre-built drafter GGUF (three are listed in
[Model Downloads](../../MODEL_DOWNLOADS.md#dspark-drafters); publishers spell the
tensors and metadata differently and the loader accepts each spelling), or build
one from the upstream **safetensors** checkpoint.

Any DeepSeek V4 release whose `config.json` has `dspark_block_size` carries the
module: `deepseek-ai/DeepSeek-V4-Flash-0731`, `deepseek-ai/DeepSeek-V4-Flash-DSpark`,
`deepseek-ai/DeepSeek-V4-Pro-DSpark`. Only the shards holding `mtp.*` are needed
(the last three, ~11 GB of the ~340 GB repo):

```bash
REPO=deepseek-ai/DeepSeek-V4-Flash-0731
B=https://huggingface.co/$REPO/resolve/main
mkdir -p dspark-src && cd dspark-src
curl -sLO $B/config.json
curl -sLO $B/model.safetensors.index.json

# the shards holding mtp.* (model-000{46,47,48}-of-00048 for the 0731 release)
python - <<'PY' | while read f; do curl -sLO $B/$f; done
import json
wm = json.load(open("model.safetensors.index.json"))["weight_map"]
print("\n".join(sorted({v for k, v in wm.items() if k.startswith("mtp.")})))
PY
```

Then convert (the tokenizer and the other 45 shards are never opened):

```bash
python eng/dsv4-dspark-to-gguf.py --checkpoint dspark-src \
    --out DeepSeek-V4-Flash-0731-DSpark.gguf --expert-type q2_k
```

The routed experts are stored as FP4 with per-32 E8M0 scales in the checkpoint,
which is GGUF's MXFP4 layout up to nibble order, so `mxfp4` repacks them
losslessly (~11 GB); `--expert-type q2_k` roughly halves that.

**Drafter size is a real trade-off.** Its weights are re-read on every
speculative step, so a bigger drafter has to earn its bandwidth back in
acceptance. On 4xA40 the effect was clear on the direct-CUDA engine (5.6 GB
2-bit: 34.0 tok/s / 69% accepted, vs 10.9 GB MXFP4: 30.9 / 65%) and inside
run-to-run noise on ggml (31.1 / 31.5 / 31.8 tok/s for the 5.6 GB, 7 GB and
10.9 GB builds on a 120-token sample, acceptance rising 63% -> 66% -> 68% with
size). Start with a small one; move up only if acceptance is your bottleneck.

| Flag | Default | Meaning |
|---|---|---|
| `--draft-model <path>` | none | DSpark drafter GGUF (env `TS_DSV4_DSPARK`) |
| `--spec-draft <N>` | block size (5) | Cap on tokens drafted per step |
| `--spec-pmin <p>` | `0.35` | Minimum CUMULATIVE acceptance probability (the product of the confidence head's per-position estimates) for a drafted position to be kept; `0` never gates |

`--spec-pmin` is the knob that matters: an extra verify row costs
roughly a quarter of a decode step on this sparse-MoE trunk (each row pulls in
its own set of experts), so drafting past a ~0.35 prefix-acceptance estimate is
expected-negative. Lower values draft more and roll back more; higher values
fall back to plain decode more often.

The drafter needs its three target feature layers on the device that owns the
output head; the layer split reserves room for it automatically, and loading
fails with a clear message if the split cannot satisfy that (reduce the GPU
count or drop `--draft-model`). The split balances the LARGEST per-device load,
counting each device's fixed residents — the embedding table on the first
device, the output head and the whole drafter on the last — where they actually
land. Spreading those over every device instead (what a plain byte-proportional
split does) left the first device ~1.7 GB of surplus weights with a drafter
loaded, which was enough for a long prompt's prefill compute buffer to no
longer fit in VRAM.

Measured on 4xA40 46 GB (DeepSeek-V4-Flash-0731 UD-Q8_K_XL, greedy, 200-token
generation, 5.6 GB drafter):

| Metric | `cuda` baseline | `cuda` + DSpark | `ggml_cuda` baseline | `ggml_cuda` + DSpark |
|---|---|---|---|---|
| Decode | 26.0 tok/s | **34.0 tok/s (1.31x)** | 26.4 tok/s | **37.1 tok/s (1.41x)** |
| Prefill (15K prompt) | 962 tok/s | 955 tok/s | 952 tok/s | 954 tok/s |
| Acceptance | — | 69% | — | 69% |

Multi-turn decode benefits most (2.2x on the third turn of the sample
conversation, 93% acceptance): a turn that continues an established context is
exactly where the drafter is confident.

Same box, `--interactive --think --tp 4` with the 7 GB Q2K-Q8 0731 drafter, a
5-turn chat: short answer, long explanation, follow-up summary, then a 10K-token
document with two questions about it.

| Turn | Baseline | + DSpark | Acceptance |
|---|---|---|---|
| 1 short (53 tokens) | 25.6 tok/s | **44.4 tok/s (1.73x)** | 87% |
| 2 long generation (512) | 26.4 tok/s | **39.6 tok/s (1.50x)** | 66% |
| 3 follow-up (470) | 26.4 tok/s | **45.3 tok/s (1.72x)** | 76% |
| 4 10K-token document (214) | 25.3 tok/s | **51.0 tok/s (2.02x)** | 85% |
| 5 second question on it (156) | 25.4 tok/s | **49.3 tok/s (1.94x)** | 82% |

Prefill stays at parity (831 vs 835 tok/s on the 10K prompt). Acceptance — and
so the speedup — is highest where the next tokens are most predictable: a
question answered out of a document in context beats free-form prose.

Greedy output was byte-identical to the non-speculative baseline on both the
200-token generation and the 15K-context run. Speculation only re-orders the
arithmetic of the accepted tokens, so a long enough run can still diverge from
sequential decode wherever a batched verify row lands on a near-tie — the same
batched-vs-sequential drift that separates prefill from decode.

Why 1.3x and not more: each extra token in the verify batch pulls its own set of
routed experts through VRAM (the trunk is 6-of-256 sparse), so a verify row costs
~a quarter of a full decode step no matter how cheap the draft was. The
confidence gate is what keeps that trade positive.

| Env | Default | Meaning |
|---|---|---|
| `MAX_CONTEXT` | 65536 | Context window (caches scale with it; metadata allows 1M) |
| `TS_DSV4_UBATCH` | 512 on `cpu` / 1024 otherwise | Prefill micro-batch |
| `TS_DSV4_NGPU` | all | Number of GPUs to layer-split across (GPU backends) |
| `TS_DSV4_VRAM_RESERVE_MB` | estimated per load (at least 2048); 2048 on `cuda` | GPU backends: overrides the VRAM held back per device for the scheduler's compute buffers. Unset, the ggml executor uses 2 GiB plus the lightning-indexer top-k transient and one ubatch of activations, so it grows with `MAX_CONTEXT` and `TS_DSV4_UBATCH`; `--backend cuda` uses a flat 2048. Lower it to offload fewer expert layers; raise it if a long prompt fails to allocate its graph |
| `TS_N_CPU_MOE` / `TS_CPU_MOE` | 0 (off) | Leading layers whose routed experts stay in system RAM (same as `--n-cpu-moe` / `--cpu-moe`). Off by default; a model that does not fit is refused with the number that would work |
| `TS_CPU_MOE_THREADS` | all usable CPUs (when offloading) | Worker threads for the host expert matmul on the ggml executors. With offload on, the pool takes `hardware_concurrency` clamped by the affinity mask and the cgroup CPU quota — not the halved default the other MoE architectures use, because a DSV4 offloaded layer reads far more expert bytes per token than theirs do and keeps scaling past that point. `--cpu-moe-threads N` overrides it, and an inherited `TS_CPU_MOE_THREADS` has the final say. Size it to the quota, not to `nproc`: 96 threads on a 23.8-CPU quota measured **25x** slower than 23. On a hosted server, leave the other threads room: the shared MoE pool on gemma-4-26B-A4B (not DSV4) ran 8.2 tok/s at 71 threads against 20.7 at 64 on a 95-CPU quota, so pass `--cpu-moe-threads` below the quota there |
| `TS_DSV4_LOAD_THREADS` | 16 | `--backend cuda`: reader threads for the stream-to-VRAM loader |
| `TS_DSV4_LOAD_STATS` | 0 | `--backend cuda`: 1 = per-stage loader timings |
| `TS_DSV4_STAGED_EXPERTS` | 1 | `--backend cuda`: 0 = per-token expert kernels (A/B) |
| `TS_CUDA_BF16_MATVEC` | 1 | 0 = single-row BF16 projections via cuBLAS instead of the dedicated matvec (`TS_DSV4_BF16_MATVEC` also accepted) |
| `TS_DSV4_FA` | 1 | Flash attention (auto-probed, GPU backends) |
| `TS_DSV4_PERF` | 0 | 1 = tok/s log + DSpark draft phase timings, 2 = per-ubatch stage timing |
| `TS_DSV4_DSPARK` | — | DSpark drafter GGUF path (same as `--draft-model`) |
| `TS_DSV4_DSPARK_CAPTURE` | 1 | 0 = skip the drafter's target-feature capture (A/B knob; drafts then go stale and are rejected) |
| `TS_DSV4_THREADS` | all cores | CPU executor worker threads |
| `TS_DSV4_MMAP` | 1 | CPU executor: 0 = copy all weights into RAM at load (parallel reads; use when the model sits on a network filesystem) |
| `TS_DSV4_BUFFER_SHARDS` | — | CPU executor: comma-separated 1-based shard indexes to copy into RAM (mmap the rest) |
| `TS_DSV4_MLOCK` | 1 | CPU executor: best-effort mlock of mapped shards (needs a memlock rlimit) |
| `TS_DSV4_SPINPOOL` | 0 | CPU executor: 1 = persistent spinning worker pool (helps on dedicated boxes, hurts under shared CPU quotas) |

## Performance (2×A100 80GB, IQ4_XS)

| Metric | TensorSharp | llama.cpp (same box) |
|---|---|---|
| Prefill (3.3K prompt) | ~500 tok/s | 574 (pp512) / 634 (pp4096) |
| Decode @3.3K ctx | ~33 tok/s | 40.3 (tg128) |

Long-context recall (needle tests at 4.6K and 15K tokens) retrieves planted
facts exactly — the compressed-attention paths are exercised beyond the raw
128-token window.
