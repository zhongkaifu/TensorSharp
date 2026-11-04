# NInfer Qwen3.8 native adapter

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
round to F16. Original BF16 exception matrices retain BF16 products and F32 accumulation;
qualified T8 verification can use the separate original MMA module instead of
cuBLAS. The original W8 output module is separate.

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

The synthetic GPU component probe is
`eng/validation/NinferCudaKernelProbe`. It covers representation, exact BF16
boundaries, activation-pack bytes, BF16/A16 projections, GDN grouped-head
mapping, causal convolution, count-limited state movement, verification
nonmutation, and recurrent-state commit. It does not load a model or measure
end-to-end inference. Run it with an actual compiled module:

```
dotnet NinferCudaKernelProbe.dll MODULE_PATH OUTPUT_JSON
```

Generated logs, module images, and probe outputs belong in ignored
`docs/validation/` or `artifacts/`. A skipped or unavailable GPU run does not
constitute passing device validation.


## Original BF16 Verify8 attention

`TS_Q35_NINFER_VERIFY8_ATTN_PTX` explicitly selects the original Ignis
Br64/Bc64 BF16 attention core through a TensorSharp-owned paged-cache adapter.
It serves only full-valid, single-sequence T8 verification on the tested
170-SM device, with Q24/KV4/HD256 and at most 40,960 visible keys. Other shapes
retain the existing attention implementation. The device cache-row base is a
graph input, so replay does not reuse a captured CPU position. Numerical
cache namespaces include the actual native image and arithmetic policy.

For the complete Qwen3.8 recipe on CC12.0/170-SM hardware, automatic selection
tries the packaged image when verification is already enabled and the batch
capacity permits eight tokens. It does not enable verification or speculation.
`TS_Q35_NINFER_VERIFY8_ATTN=0` disables automatic selection; its accepted values
are exactly `0` and `1`. An explicit `_PTX` path takes precedence even with `0`
and must pass its existing path/geometry checks. Missing or incompatible
automatic assets retain existing attention before use; allocation, context and
execution failures are errors. Startup diagnostics identify the selection and
actual loaded image. Other models, devices and runtime shapes retain their path.

The normal build adds this optional image only for target120 or an explicit
fatbin list containing120, with a dedicated C++20/sm120a compilation and exact
`-O3 --ftz=false --fmad=true` flags. PTX, fatbin and the PTX-binding SHA sidecar
flow to `cuda_kernels` in output/publish layouts. The unchanged core never enters
the compute75 portable baseline; managed-only builds preserve prebuilt assets.

Build the separate image with `eng/validation/build-ninfer-prefill-attention.py`.
The source manifest preserves the unchanged original core, notices and licenses
at Ignis `e600a7d9e0d5921d767a6cb464373583b3cdcaf4`. No ggml modifications
are needed. `NinferSmallAttentionProbe` checks the adapter against that original
core and FP64 references, exercises poisoned workspaces and changing positions
inside captured graphs, and times the complete old/new attention call chains.
`NinferSmallAttentionAudit` independently replays retained evidence on CPU.
Actual RTX 5090 validation covered 624 component kernel executions, 24 dynamic
graph replays, and 1,680 timing kernel executions; four tested key lengths
(74, 202, 512, 1,024) won all five paired timing rounds, with median complete-call
speedups of 3.95–7.04×. These are attention-component results, not engine-wide
speedups.

This implementation uses the original valid BF16 arithmetic; summation order
and intermediate rounding differ from the legacy implementation. A changed
answer alone is not an engine-bug verdict. The full 45-text/30-program trial
completed with 38/45 and 25/30 passing, compared with 38/45 and 27/30 for the
previous DFlash path. One newly generated program merged adjacent integer
intervals incorrectly; another previously passing program was not executed
because of the unchanged execution policy. Preserve those failures separately
from verified kernel contracts. Finite quality results do not prove equivalent
whole-model output or sampling distributions.

One same-runtime OFF/ON/ON/OFF block completed 36 frozen measured requests:
code decode medians were 326.98 / 388.30 / 388.33 / 326.96 tokens/s, about 18.8%
higher with this path. Prose/copy decode rates improved 12.9% / 23.7%. Per-request
completion times were code 3.208 → 2.555 s, prose 4.863 → 4.920 s, copy 1.720 → 1.370 s;
outputs and lengths differed, so throughput gains are not identical to latency
gains. This single descriptive block does not establish universal speedup.

The normal CUDA 13.2 MSBuild target then reproduced the exact qualified native
image. An actual automatic-selection run completed all 45 text / 30 program cases
with neither Verify8 path nor auto option specified. All 141 request identities,
309 original checks, 96 arithmetic values and 189 program-case statuses matched
the explicit configuration; 125 normalized serial responses were identical. The
other 16 responses were parallel 2/4 cases, with unchanged checks and verdicts.
Code throughput was 375.90 versus 375.92 tokens/s, so discovery adds no extra
speedup over explicit selection. The retained 7 text / 5 program failures remain
failures. Final NInfer/SHA CPU tests passed 782 cases with zero skips; native
Windows and other unavailable devices are not covered by this run.

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
vector A16 asset, Verify8 attention and native greedy verification already
enabled. FI, exact-Verify and shortlist experiments retain the adaptive default.
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
rounding, signed zero, guards and graph dependencies. `NinferResidualNormBodyProbe`
compares independent old/new-off/new-on model processes through Verify, Commit,
Decode, clone, reset and DFlash proposal operations. Component speedups are
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

Residual/norm, BF16 MMA and GateUp fusion admit exactly configured context 40,960
or 65,536 with their existing CC12.0/170-SM, full-valid T8/B1 qualifications.
Capacity is not an operand of these three kernels. Verify8 attention separately
retains its cap of 40,960 actual visible keys and existing fallback above it.
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

## CLI and Server.Host startup

Use the complete `qwen3_8_27b_nvfp4full-v2.ninfer` artifact with `--backend cuda`,
not `ggml_cuda`. Its tokenizer, special-token metadata and chat template are
embedded; no external `--tokenizer` is needed. This route currently serves text.
Build matching assets, including the separate attention/MMA modules:

```sh
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release -p:CudaArch=sm_120a
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release -p:CudaArch=sm_120a
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
`--model /workspace/hf/ignis/qwen3_8_27b_nvfp4full-v2.ninfer`.

See the [shared configuration guide](../../../config/README-ninfer.md) for model
location, environment precedence, profile validation and the comment-only
finalization of the tested 64K settings. A standalone target-only run still uses
`--model FILE.ninfer --backend cuda` without the DFlash preset. Startup logs
identify actual selected module SHAs; host submissions include capture and are
not individual GPU kernel replay counts.
