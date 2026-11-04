# NInfer Qwen3.8 native adapter

`tensorsharp_q35_ninfer.cu` is TensorSharp-owned code. It adapts the NInfer target
representation and numerical boundaries to the existing Q35 CUDA engine. It
neither includes nor patches ggml. Build with the same CUDA 13+ toolkit and
`sm_120a` architecture as the native FP4 engine; fast math must remain disabled.
The build project includes this separate module. An explicit
`TS_Q35_NINFER_KERNELS` path overrides module discovery in `cuda_kernels/`.

The adapter uses F32 storage for activations and rounds at the declared BF16
boundaries. Persistent recurrent state and the independent BF16 AB projection's
accumulation stay F32. Weight-only FP4 projections decode original code/scale
planes and retain the weight divisor; activation-quantized projections use the
original input calibration divisor. The activation pack implements E2M1
round-to-nearest-even using exact binary midpoint comparisons because the
consumer Blackwell target does not support `cvt.e2m1x2.f32`. It does not first
round to F16. Original BF16 exception matrices use cuBLAS BF16 products and F32
accumulation. The original W8 output module is separate.

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
Verify8 policy below has narrower hardware and runtime requirements. Other
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
