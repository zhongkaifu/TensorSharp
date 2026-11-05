# NInfer Qwen3.8 native adapter

The route serves the NInfer v3 artifact `qwen3_8_27b_nvfp4.ninfer`
([neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)):
FP8 E4M3 row-scaled embedding, head, attention/GDN input and output projections and MLP
layers 56-63; NVFP4 MLP layers 0-55; a q8_g32/BF16 DFlash2 drafter. The v2 container (and
the Ignis `nvfp4full-v2` artifact) is no longer read.

## FP8 projections (TensorSharp-owned, `native/kernels`)

* `tensorsharp_small_t_mma.cuh`: small-T projections on BF16 tensor cores for decode,
  verification and the drafter. Weights widen exactly to BF16 operands (E4M3 and INT8
  are BF16 values); FP8 row scales apply once after the dot product, q8_g32 group scales
  per 32 values in F32. Activations are never quantized. A CTA owns 16-64 weight rows and
  KW warps split K; each warp streams its own K slice through a cp.async ring. Above 8
  rows the activations are cast to BF16 once per projection and shared by up to four
  16-row weight tiles. Gate|up rows are interleaved by 8 at load so SwiGLU fuses.
  Structure follows NInfer's sliced-K A16 kernels (Neroued/ninfer 1882c6ad, Apache-2.0);
  the code is independent.
* `ts_q35_gemm_w8a8_tma*` (`tensorsharp_q35_kernels.cu`): prefill W8A8 GEMM with the same
  warp-specialized TMA structure as the W4A4 GEMM, on per-row E4M3 activation codes
  (`ts_q35_fp8_quant_act`, amax / 448, NInfer's A8 numerics). The MMA is issued as
  `kind::mxf8f6f4.block_scale` with unit UE8M0 scales: on the RTX 5090 the plain FP8 form
  accumulates F32 at half rate (NInfer's finding), the block-scaled form at full rate with
  the same dot product. Scales apply at the store.
* `ts_q35_embed_fp8`: token rows as BF16(code x row scale).
* `ts_q35_fp8_gemv_{b,c}_*`: one activation row (non-speculative decode) on the CUDA cores.
  A warp owns 4 (b) or 2 (c) weight rows and each lane streams 16 codes per row per phase
  with `ld.global.nc.L1::no_allocate`, so the BF16-rounded F32 activations stay
  L1-resident; four F32 accumulator chains, one warp reduction, then the row scale.
  Schedule b serves the 16384-row GDN input projection, c everything else (NInfer's RTX 5090
  shape choices).

Prefill 8K / 32K TTFT on the RTX 5090: 0.90 / 4.48 s with half-rate FP8 MMA, 0.78 / 3.99 s
with the block-scaled form.

## One-row NVFP4 and norm kernels (`tensorsharp_q35_ninfer.cu`)

* `ts_ni_gemv_nvfp4_k*`: one activation row of an NVFP4 projection (TS planes), 8 warps x 2
  rows, 32 values per lane per phase (16 code bytes and 2 scale bytes). The generic
  `compute_120` build has no hardware E2M1 conversion (CUDA emulates it), so each code pair
  is placed straight into F16 bits, which is the value times 2^-14 exactly. The 2^14 is
  folded into the A16 coefficient (scale x inverse divisor), so every weight equals
  code x coefficient. The `_swiglu` form writes BF16(silu(gate) x up) for the
  8-row-interleaved gate|up matrix.
* `ts_ni_add_norm_5120`: `ts_ni_add_norm` with the row in registers; the same per-thread
  values, sum order and expressions, so the output is bit-identical.

RTX 5090, one row: gate|up [34816, 5120] 103 -> 56 us (0.97 -> 1.79 TB/s), down
[5120, 17408] 56 -> 29 us, FP8 projections 1.45-1.60 -> 1.60-1.70 TB/s, add+norm 6.8 ->
2.2 us. Non-speculative decode: 53 -> 73 tok/s. `TS_Q35_T1_GEMV=off` selects the previous
kernels (8-row FP8 tile and the legacy A16 NVFP4 kernel) for A/B runs.

## 8-bit prompt attention (`tensorsharp_q35_attn8.cuh`, both routes)

Prompt chunks with at least 1,024 visible keys (one sequence) run their attention in 8 bits, as
Ignis and NInfer do. The design follows the FP8 path of mkyt/qwen3-27b-5090 (MIT) as Ignis adapts
it (`kernel/src/hq_prompt_attention.cu`, gpillon/ignis ad81d52e, Apache-2.0); the code is
TensorSharp's own and reads TensorSharp's cache formats (q8_0 on the GGUF route, BF16 or FP8 on
the NInfer route).

- **Band, once per chunk and layer:** the visible keys become INT8 K with one F32 scale per key,
  plus E4M3 V^T with one power-of-two scale per 32-key tile. Converting per CTA tile instead
  would cost more than the attention itself.
- **Attention:** 128 GQA-packed rows per CTA, 8 warps. Q is quantized per row in registers.
  - Q K^T runs on the INT8 m16n8k32 MMA (exact int32).
  - The online softmax runs in base 2 with P = 2^(s - m + 8) in E4M3; the row sum uses the
    rounded P.
  - P V runs on the block-scaled E4M3 MMA with V's tile scale as the B scale, full rate on
    sm_120.
  - P never leaves registers: each V^T tile stores its keys in the order of the S accumulators.

RTX 5090, Qwen3.8-27B TTFT (2 repeats each):

| Route | 32K | 63K | 127K |
|---|---:|---:|---:|
| GGUF | 3.99 -> 3.65 s | 9.58 -> 8.21 s | |
| NInfer | 3.94 -> 3.43 s | 10.0 -> 7.87 s | 29.6 -> 20.2 s (FP8 KV) |

Every 8K-127K retrieval and record-order check passed. Against the 16-bit kernel the output
differs by about 2% relative RMS per layer (`TS_Q35_ATTN8_CHECK=1` prints it on the GGUF
route). `TS_Q35_ATTN8=0` restores the 16-bit kernels.

The band is scratch. It grows with the prompt in 16K-key steps and claims no growth reserve. A
short slot or block allocation releases it, and so does the reserve check before every
forward; the next prompt chunk rebuilds it. A chunk whose band does not fit runs the 16-bit
kernels.

## Decode and verify attention on the q8_0 (GGUF) route

Decode and verify steps run the prefill kernel's two-stage `cp.async` pipeline over the fixed key
splits of the decode plan (`ts_q35_attn_q8_split_pipe_f32`). The synchronous split kernel read a
long q8_0 history at about half the device bandwidth. Decode after 120K went from 143 to 161 tok/s
(`TS_Q35_DECODE_ATTN_PIPE=0` restores the synchronous kernel).

Sequences that resume one cached prefix share its cache blocks. Splits that lie within the blocks
every sequence shares run once, with all sequences' rows packed as one sequence; each sequence's
own splits follow. The result is bit-identical: `TS_Q35_SHARED_PREFIX_CHECK=1` recomputes each
shared launch without sharing and prints the largest difference. On the RTX 5090, four chats on
one cached 60K-token document went from 330 to 380 tok/s (`TS_Q35_SHARED_PREFIX_ATTN=0` turns
it off).

## Optional FP8 KV cache

`TS_Q35_NINFER_KV=fp8` stores KV rows as 256 E4M3 codes plus two F32 group-128 scales (272
bytes, 53% of BF16). The prep kernel quantizes K after norm and RoPE and V as projected;
decode/verify attention stages FP8 tiles and widens them in shared memory; prefill
attention reads a per-layer BF16 copy of the visible history (`ts_ninfer_kv_fp8_to_bf16`).
Prefix-cache namespaces and host checkpoints carry the KV format. Measured: no speed change
at 32K-64K (the decode split kernel is not KV-bandwidth bound), answers unchanged on the
long-context checks; the default stays BF16.

## Adapter

`tensorsharp_q35_ninfer.cu` is TensorSharp-owned code. It adapts the NInfer target
representation and numerical boundaries to the existing Q35 CUDA engine. It
neither includes nor patches ggml. Build with the same CUDA 13+ toolkit and
`sm_120a` architecture as the native FP4 engine; fast math must remain disabled.
The build project includes this separate module. An explicit
`TS_Q35_NINFER_PTX` path overrides module discovery in `cuda_kernels/`.

The adapter uses F32 storage for activations and rounds at the declared BF16
boundaries. Persistent recurrent state and the independent BF16 AB projection's
accumulation stay F32. Weight-only FP4 projections decode original code/scale
planes and retain the weight divisor; activation-quantized projections use the
original input calibration divisor. The activation pack implements E2M1
round-to-nearest-even using exact binary midpoint comparisons because the
consumer Blackwell target does not support `cvt.e2m1x2.f32`. It does not first
round to F16. The BF16 GDN a/b projection retains BF16 products and F32 accumulation.

Reference contracts were inspected in the public gpillon/ignis revision
`e600a7d9e0d5921d767a6cb464373583b3cdcaf4`: `kernel/src/gdn_layer.cu`,
`kernel/src/gqa_layer.cu`, `kernel/src/layer_internal.h`, and the corresponding
`kernel/vendor/src/ops/` NVFP4 codec, specialized projection plans, GDN recurrence,
RMS norm, and residual epilogues. This is an independently written adapter, not
an assertion of identical summation order or whole-model equivalence. In
particular, its first GDN prefill implementation uses a sequential recurrent
transition rather than the reference's chunked tensor-core prefill algorithm.
It must pass independent numerical and model-level quality tests before a speed
or quality-preservation claim is made.

## Qualified automatic optimizations

The complete Qwen3.8-27B NInfer recipe on compute capability 12.0 automatically
selects the packaged legacy-exact A16 vector module, lossless GPU weight-layout
conversion, and eligible decode CUDA graphs. These choices preserve the existing
projection arithmetic or encoded weight bytes. They do not turn on speculative
decoding, host embeddings, FlashInfer, or a new GDN algorithm. The separate
Verify8 policies below have narrower hardware and runtime requirements. Other
devices retain the existing implementations unless explicitly
configured. Automatic device qualification currently comes from RTX 5090 testing;
it is not a claim of model or performance coverage on every CC 12.0 board.

The normal `cuda_kernels/` asset resolver finds
`tensorsharp_q35_ninfer_a16_legacy_exact_vector.ptx` and
`tensorsharp_ninfer_repack.ptx`. The normal module loader prefers matching sibling
fatbins with valid PTX-hash sidecars; fatbin-only packages are also supported.
An absent or incompatible automatic module logs a fallback before inference.
An explicitly configured missing or incompatible module remains an error.
Execution failures are never retried on the fallback path.

| Control | Behavior |
|---|---|
| `TS_Q35_NINFER_A16_AUTO=0` | Disable automatic vector selection. |
| `TS_Q35_NINFER_REPACK=0` | Disable automatic GPU layout conversion. |
| `TS_Q35_NINFER_DECODE_GRAPHS=0` | Disable automatic graph capture/replay. |
| `TS_Q35_NINFER_VERIFY8_GEMM=0` | Keep the existing T8 verification matrix schedule. |
| `TS_Q35_NINFER_A16_PTX` | Explicit original-schedule A16 image; suppresses automatic vector selection. |
| `TS_Q35_NINFER_A16_LEGACY_EXACT_PTX` | Explicit legacy-exact image; suppresses automatic discovery. |
| `TS_Q35_NINFER_REPACK_PTX` | Explicit lossless layout converter; takes precedence over its automatic control. |

The two automatic module controls accept only `0` or `1`; unset means automatic.
Explicit `1` does not extend automatic qualification to another device or recipe.
The two explicit A16 paths remain mutually exclusive. Graph `1` retains its
existing explicit opt-in semantics on other devices. Graph capture can still fall
back to ordinary kernel launches for unsupported shapes or capture failures;
actual capture/replay counters distinguish use from a requested setting.

Engine diagnostics expose `NinferA16Selection`, `NinferA16ModulePath`,
`NinferA16ImageSha256`, `NinferLoadRepackSelection`,
`NinferLoadRepackModulePath`, and `NinferDecodeGraphsSelection`.
The numerical cache namespace includes the actual successfully loaded A16 image
and arithmetic policy. The layout converter reports its actual image SHA and
converted matrix/byte counts and is unloaded after model loading. The automatic
default composition was exercised on RTX 5090: 174 full-vocabulary F32 head
arrays (43,207,680 words) matched the previous explicit implementation exactly;
57 normalized API responses and all 66 original checks were unchanged. That
composition measured 50.42 code tokens/s versus 50.39 with the same optimizations
explicitly selected. It removes manual configuration; it is not an additional
speedup over those explicit settings. This finite validation does not establish
universal model-quality equivalence. Upstream ggml was unchanged at
`facdc44a67da17df4433adcc468b424f48e9ded2`. Generated evidence remains ignored
under `docs/validation/`; unavailable model/device scenarios are not passes.

The v2-era synthetic component probe (`NinferCudaKernelProbe`) was removed with the v2
reader; the unit tests in `InferenceWeb.Tests` (`Ninfer*`) cover the v3 descriptors, and
`eng/validation/Qwen35T1GemvBench` times the one-row projections.

Generated logs, module images, and probe outputs belong in ignored
`docs/validation/` or `artifacts/`. A skipped or unavailable GPU run does not
constitute passing device validation.


## Original BF16 prompt attention image

The packaged `tensorsharp_ninfer_prefill_attention` image holds the original Ignis
Br64/Bc64 BF16 attention core behind a TensorSharp-owned paged-cache adapter. It
serves ordinary single-sequence prefill chunks (see the defaults table below).
It was first qualified as a T8 verification core, where it beat the legacy split
kernel 3.95-7.04x at 74-1,024 keys; the GQA-packed, cp.async-pipelined split
kernel has since replaced it for verification at every history (it reads each key
once for all six query heads instead of six times), so the T8 entry and its
probes were removed.

The normal build adds this optional image only for target120 or an explicit
fatbin list containing120, with a dedicated C++20/sm120a compilation and exact
`-O3 --ftz=false --fmad=true` flags. PTX, fatbin and the PTX-binding SHA sidecar
flow to `cuda_kernels` in output/publish layouts. The unchanged core never enters
the compute75 portable baseline; managed-only builds preserve prebuilt assets.

Build the separate image with `eng/validation/build-ninfer-prefill-attention.py`.
The source manifest preserves the unchanged original core, notices and licenses
at Ignis `e600a7d9e0d5921d767a6cb464373583b3cdcaf4`. No ggml modifications
are needed.

## Full T8 verification GEMM

`TS_Q35_NINFER_VERIFY8_GEMM` is automatic when unset on the complete
Qwen3.8 NInfer recipe, CC12.0/170-SM hardware, with verification already enabled
and room for eight rows. The main CUDA image must contain
`ts_q35_gemm_w4a4_8`. It changes only complete T8 verification of the five plain
NVFP4 projection shapes: N/K 16384/5120, 34816/5120, 5120/17408, 5120/6144 and
14336/5120. Ordinary prefill, other shapes, BF16 matrices and GGUF retain their
existing dispatch. It enables neither speculation nor verification.

The kernel reuses the existing TensorSharp implementation with an eight-token
tile and two shared-memory stages. It removes three unused token fragments,
reducing dynamic shared memory from 76,800 to 43,520 bytes and measured registers
from 76 to 40. Encoded weights, activation quantization, alpha, split-K summation
order and consumer precision boundaries are unchanged. Gate/up still reaches
SwiGLU in F32; output/down projections remain F32 until their existing consumer.

Exact `0` disables selection; `1` requires a supported profile and the new entry.
Other values are rejected on NInfer. An older automatic main image without the
entry falls back before inference; other loading, ABI and execution failures
remain errors. Selection is fixed before any GEMM or graph capture. The actual
main image SHA and policy enter the KV namespace. Diagnostics expose selection,
image and host submissions; host submissions include capture and do not count
GPU graph replays.

CUDA 13.2 validation on RTX 5090 exercised the actual production image across
all five shapes: 702 component kernel executions, two module lifetimes,
three output poisons, complete F32/consumer comparisons against both the actual
old image and a same-source control, CPU quantization and scoped FP64 bounds.
The dense GDN warm full chain improved 1.1237x in five paired timing rounds;
the other shapes' component runs establish numerical coverage, not speed wins.
The affected managed suite passed 146 tests with zero failures or skips.

One same-runtime OFF/ON/ON/OFF block completed 36 frozen API requests. Code arm
medians were 400.04 / 406.46 / 406.98 / 399.83 tokens/s: means of arm medians
399.94 to 406.72, about 1.70%. Prose and copy increased about 1.76% and 1.61%.
All nine corresponding response contents and token usages matched across arms;
both arms retained the same automatic governor and speculative work. These are
descriptive client decode rates for K7/pMin0, BF16 KV, context40960, ubatch2048
and cap4, not a universal speed or engine-ranking claim.

For those same outputs, complete HTTP request times were code 2.5823 to 2.5405s,
prose 4.2749 to 4.2007s and copy 1.3505 to 1.3349s, using means of the two arm
medians. TTFT was mixed: code 66.34 to 67.19ms, prose 65.00 to 63.28ms and copy
414.59 to 415.30ms. The observed gain is primarily in decoding.

The actual automatic build also completed all 45 text and 30 program cases.
Against the preceding accepted default, all 141 inputs, 309 checks, 96 arithmetic
values and 189 program-input statuses matched. All 123 serial text responses and
30 generated programs were identical; parallel response variations changed no
checks. Scores remain 38/45 and 26/30, including the existing wrong interval
answer and 21 unexecuted program inputs. Finite parity does not establish
universal model quality. This change does not establish NInfer 128K/250K or
native Windows coverage. Reusable probes are in
`eng/validation/NinferA4GdnCoreProbe`; generated evidence remains ignored.

## Native DFlash cost governor

When `TS_SPEC_ADAPTIVE` is unset, the HTTP scheduler skips repeated cost-governor
calibration only for the measured native Qwen3.8 DFlash profile: the embedded
full-vocabulary drafter, effective K7/pMin0, CC12.0/170-SM hardware, exactly
40,960 configured context tokens, enabled decode graphs, the qualified legacy
vector A16 asset and native greedy verification already enabled. FI, exact-Verify and shortlist experiments retain the adaptive default.
The request must use plain greedy sampling without media, grammar, repetition
penalties or a logits observer. This policy enables neither speculation nor
verification and changes no kernels, draft window or confidence threshold.

Explicit `TS_SPEC_ADAPTIVE=0` still disables calibration; every other explicitly
set value enables it, as before. A caller-supplied governor is authoritative.
Other profiles and standalone executions without sampler context retain the
existing default. Each arming log reports `adaptive` and `adaptiveSource`, and
switching request profiles replaces the shared governor without mutating a live
execution's selected policy.

One same-runtime ON/OFF/OFF/ON block measured code decode medians of
389.22 / 399.97 / 399.63 / 388.32 tokens/s, about 2.84% higher without calibration.
Prose varied little and copy did not improve. Code outputs differed in length,
and total completion time did not improve, so this is a finite decode-throughput
result, not a universal speed or latency claim.

Actual automatic-policy validation completed all 45 text / 30 program cases.
It selected fixed drafting for 168 qualified requests; six JSON requests retained
adaptive calibration. Relative to explicit `TS_SPEC_ADAPTIVE=0`, all 141 request
identities, 309 checks, 96 arithmetic values and 189 program-input statuses were
unchanged. The text/program scores remained 38/45 and 26/30; one wrong interval
answer and 21 unexecuted inputs remain failures. Code throughput was 400.09
versus 400.06 tokens/s. The automatic policy removes manual configuration; its
finite quality parity is to that explicit configuration, not all model behavior.
The affected managed suite passed 155 tests with zero failures; one existing
9B GGUF model test was unavailable and skipped. Native kernels and upstream
ggml were unchanged throughout this governor evaluation.


## Fused residual and readonly norm for T8 verification

`TS_Q35_NINFER_RESIDUAL_NORM` is automatic when unset on the qualified
Qwen3.8 NInfer recipe with CC12.0/170-SM, configured context exactly 40,960 or
65,536, enabled verification,
layer graphs, Verify8 GEMM, split-K and the target W8 head. All Down projections
must retain their original NVFP4/A4 format; FI and ExactVerify overrides are
excluded. This selector enables none of those prerequisites itself. Only
full-valid T8/B1 verification uses the fused path. Exact `0` opts out; `1`
requires eligibility and both native entries. Missing automatically discovered
entries retain the existing path before work begins. ABI, allocation and
execution errors remain errors.

The existing BN8 GEMM folds the F32 residual addition and BF16 rounding into
its final split epilogue. Both F32 projection multiply boundaries and the
split order are preserved; the original F32 projection buffer remains available.
The following readonly norm retains the old 256-thread reduction and effective
F32 norm weights, while prefetching weights and avoiding another residual write.
The first embedding norm still performs the original null-delta addition,
including its signed-zero behavior. BF16 exception projections still use their
existing residual path. Local pass state selects normalization only after an
actually fused projection; the final norm requires a completed fused Down.
Graph/cache identities bind the actual native module and arithmetic policy.

`NinferResidualNormProbe` validates both output/down shapes against the actual
old image, CPU quantization and bounded FP64 references, including contraction,
rounding, signed zero, guards and graph dependencies. (The whole-model v2 comparison,
`NinferResidualNormBodyProbe`, was removed with the v2 reader.) Component speedups are
separate from model throughput and quality; generated evidence stays in ignored
validation directories.

## Qualified automatic GateUp/SwiGLU fusion

`TS_Q35_NINFER_GATEUP_SWIGLU` selects the qualified exact GateUp/SwiGLU fusion.
Unset selects automatically only for the known CC12.0/170-SM recipe,
configured context exactly 40,960 or 65,536, enabled verification/layer graphs/
Verify8 GEMM and original NVFP4 GateUp matrices, without FI or ExactVerify. Dispatch is full-valid T8/B1 only.
`0` disables it; `1` requires the qualified profile and native entry. It enables
none of its prerequisites. Other shapes keep their previous path.

The BN8/stages2/split1 epilogue preserves the old F32 projection boundaries,
helper `expf` SiLU and final BF16-RNE/F32 output. R58 passed full final-output
bits and numerical checks, but its 0.961us saving **failed** the original 1us
absolute micro gate. That result was retained. A separately declared R59 model
study then measured one OFF/ON/ON/OFF block: code +0.622%, prose +0.681%, copy
+0.712%, with matching responses, typed usage and draft work. This is not a
general speed guarantee. Code HTTP total time fell 0.693%; its TTFT was 0.763ms
slower in this block. R60 exercised the unset automatic path and retained 38/45
text tasks and 26/30 programs. Against R55, all 309 checks, 96 arithmetic values
and 189 program statuses were unchanged, with no lost successful check or
executed case. All 123 serial responses and 30 program contents matched; ten
parallel response contents changed without score loss. Existing model mistakes
remain. Those R59/R60 results used configured context 40,960; they do not alone
cover the later 64K profile described below. Upstream ggml remained unchanged at
`facdc44a67da17df4433adcc468b424f48e9ded2`; generated evidence stays ignored.

## Qualified 64K shared configuration

Residual/norm and GateUp fusion admit exactly configured context 40,960
or 65,536 with their existing CC12.0/170-SM, full-valid T8/B1 qualifications.
Capacity is not an operand of these three kernels.
Other configured contexts do not inherit these three automatic qualifications.
The automatic governor's measured scope remains 40K; both shared presets
explicitly set `TS_SPEC_ADAPTIVE=0` for stable K7 drafting.

R61 tested the three optimizations together at configured 65,536 in one fixed
OFF/ON/ON/OFF block: code 406.49→442.42 token/s (+8.84%), HTTP total time −8.10%,
with identical response contents, typed usage and draft work. This is a combined
finite result, not a separate gain attributed to each kernel. Four repeats of
one long diagnostic used 64,416 input +144 output tokens, cache0 and passed all ten
needle/order checks each; this does not establish general long reasoning quality.

R62 ran the 64K JSON through the actual CLI with inherited TS_*/MAX_CONTEXT
cleared, full-head DFlash and positive T8 captures/replays. R63 used the same
JSON through Server.Host for all 45 text +30 program cases: 38/45 and 26/30 passed.
Against R60, all 309 checks, 96 arithmetic values and 189 program statuses matched;
123 serial responses and 30 program contents were identical. Sixteen parallel
contents changed without score loss. Existing model mistakes remain.
The literal 40K JSON was parser tested; earlier 40K engine qualification is not a
claim that this new 40K file itself received that GPU campaign. Complete 128K/250K
campaigns for the optimized profile are **NOT_RUN**; arbitrary contexts and
other hardware remain outside this finite qualification.

## Ignis-derived prefill and long-context defaults (2026-11-04)

A per-kernel nsys comparison of the same 32K prompt against Ignis (same
artifact, BF16 KV) showed the legacy prefill attention kernel at 74% of GPU
time, a sequential GDN scan at 23%, and F32 storage-boundary passes
(`ts_ni_round`, `ts_ni_swiglu`, `ts_ni_quant_nvfp4`) that Ignis does not run.
These now apply automatically on the qualified recipe (CC 12.0, Q24/KV4/HD256):

| Feature | What changed | Opt-out |
|---|---|---|
| Original BF16 prompt attention | The packaged Ignis Br64/Bc64 core serves ordinary prefill chunks of 2+ rows. | `TS_Q35_NINFER_PREFILL_ATTN=0` |
| Chunked GDN prefill | Ignis' WY/state-passing/output kernels (`native/ninfer/chunk`, built as `tensorsharp_ninfer_gdn_chunk`) replace the sequential scan for spans of whole 64-token chunks; tails, decode and verification keep the scan. | `TS_Q35_NINFER_GDN_CHUNK=0` |
| TMA prefill GEMM | Prefill projections of 65+ rows run a TMA-fed, warp-specialized W4A4 kernel (the structure of NInfer's `nvfp4_w4a4_tma`, on TensorSharp's planes, 128- or 256-token tiles) whose epilogue applies the BF16 round, SwiGLU, and the down projection's NVFP4 input quantization. Same accumulation order and arithmetic: answers were byte-identical with and without it. The GGUF route's prefill projections run the same tiles (plain, and an F32 SwiGLU epilogue for gate/up): 1.4-1.6x the cp.async GEMM, bit-identical. | `TS_Q35_NINFER_PREFILL_EPILOGUE=0` (NInfer route) |
| Prefill norm + quantize | The residual add + RMSNorm before an A4 projection also writes that projection's NVFP4 input (`ts_ni_add_norm_quant`: add_norm's reduction order, then the quantizer's arithmetic on the values it just produced); the projection skips its own quantize pass. Full GDN chunks are no longer rounded in place before the chunk pack (the pack's BF16 conversion is the same rounding); only the scanned tail is. Byte-identical answers; 32K TTFT 3.70 -> 3.63 s. A residual-add GEMM epilogue was measured 2.4x slower per GEMM and is not shipped. | `TS_Q35_NINFER_PREFILL_EPILOGUE=0` |
| Tensor-core context append | The DFlash2 drafter's context projections over 64+ rows run Ignis' A16 BF16-MMA kernel (`ts_nidf_mma_*`); only drafts change, never the verified output. | none (draft image) |
| Decode/verify attention | One GQA-packed split kernel with cp.async double-buffered tiles: a fixed one-wave split count (SMs / KV heads / lanes), each split's key range derived on the device, so one captured graph serves every history. Replaced both the unpipelined split kernel and the Ignis T8 core. | none |
| Context qualification | Residual/norm and GateUp/SwiGLU also admit configured contexts 131,072 and 262,144 (context is not an operand of those kernels). | their existing `=0` options |

Measured on RTX 5090 with `config/qwen3.8-27b-ninfer-64k.json` (DFlash2 K7, BF16 KV):
TTFT 32K 14.29 -> 3.63 s, 60K 44.45 -> 8.74 s (8K 0.92 -> 0.74 s from the TMA GEMM
alone); a 512-token answer after 7K / 33K / 62K context decodes at
192.8 / 127.5 / 82.4 -> 217.2 / 189.8 / 183.2 tok/s.
The unchanged text suite (309 checks, 3 repeats) and the 30-program suite scored
297/339 both before and after; the differing answers are greedy flips of legal
summation-order changes, in both directions.

Not adopted: Ignis' 8-bit prompt-attention band exists only for its hq-e8-2b KV
codec, whose own measurements cost KLD +0.018-0.022 at 32K-64K. Deeper cp.async
pipelines with K128 tiles and 256-column cp.async tiles were measured slower and
are not shipped.

## CLI and Server.Host startup

Use the complete `qwen3_8_27b_nvfp4.ninfer` artifact with `--backend cuda`,
not `ggml_cuda`. Its tokenizer, special-token metadata and chat template are
embedded; no external `--tokenizer` is needed. This route currently serves text.
Build matching assets, including the separate attention/MMA modules:

```sh
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release -p:CudaArch=compute_120
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release -p:CudaArch=compute_120
```

The toolkit must support sm120a; portable compute75 assets alone are insufficient.
For DFlash, **use the same JSON in both hosts** (only the entry arguments differ):

```sh
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --config config/qwen3.8-27b-ninfer-64k.json --chat
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/qwen3.8-27b-ninfer-64k.json --host 127.0.0.1 --port 5088
```

Choose `config/qwen3.8-27b-ninfer.json` in either command for 40,960 capacity.
The profiles explicitly request full-head K7/pMin0, greedy verification and
Gov0; loading an NInfer file by itself does not enable speculation. Their
`${appBaseDirectory}` draft path locates each application's own
`cuda_kernels/tensorsharp_q35_ninfer_draft.fatbin`; `_PTX` accepts direct fatbins.
Use `--input prompt.txt` instead of `--chat` for CLI file input; `--prompt` is not
its text-generation option. An existing model can override the download entry:
`--model /workspace/hf/ninfer/qwen3_8_27b_nvfp4.ninfer`.

See the [shared configuration guide](../../../config/README-ninfer.md) for model
location, environment precedence, profile validation and the comment-only
finalization of the tested 64K settings. A standalone target-only run still uses
`--model FILE.ninfer --backend cuda` without the DFlash preset. Startup logs
identify actual selected module SHAs; host submissions include capture and are
not individual GPU kernel replay counts.
