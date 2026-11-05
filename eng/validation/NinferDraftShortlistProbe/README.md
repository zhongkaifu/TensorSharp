# Original NInfer shortlist head candidate

This optional standalone module uses the artifact's original
`text/draft_head[131072,5120]` Q4G64_F16S payload and I32 row-to-global-token map.
It does not quantize, repack or modify these weights. No engine hook or default
is changed by this module/probe. A smaller proposal head is not a model-quality
or end-to-end performance result; actual target verification and complete
quality/latency evaluation remain required.

The source pins Ignis e600a7d9e0d5921d767a6cb464373583b3cdcaf4. Fourteen original
files, including ten headers, are vendored unchanged with Git blob SHA1, SHA256,
lengths and the original license/notice files. The native adapter explicitly
instantiates the original global kernels. It does not copy their numerical
bodies into private kernels. Its separate remap entry preserves the original
`speculative_round.cuh442..448` behavior.

```sh
python3 eng/validation/build-ninfer-shortlist-module.py --verify-only
python3 eng/validation/build-ninfer-shortlist-module.py --output /absolute/artifacts/shortlist-fresh.fatbin
dotnet build eng/validation/NinferDraftShortlistProbe/NinferDraftShortlistProbe.csproj -c Release
dotnet eng/validation/NinferDraftShortlistProbe/bin/Release/net10.0/NinferDraftShortlistProbe.dll --self-test
dotnet eng/validation/NinferDraftShortlistProbe/bin/Release/net10.0/NinferDraftShortlistProbe.dll --self-test-all
dotnet eng/validation/NinferDraftShortlistProbe/bin/Release/net10.0/NinferDraftShortlistProbe.dll --execute --module /absolute/artifacts/shortlist-fresh.fatbin --draft-module /absolute/artifacts/frozen-draft.fatbin --out /absolute/artifacts/shortlist-components-fresh
```

`--draft-module` is the existing original DFlash component image, used to exercise
actual Top16 composition. Missing modules fail before any completed result; no
unavailable device/model scenario is counted as passing. The builder verifies
original bytes and licenses first, refuses an existing output image, and captures
actual compiler/version/command/module hashes. It uses stdc++17/O3/fmad=true/
ftz=false/sm120a with the unchanged original includes; no ggml sources or patches.

The internal `CudaNinferDraftShortlistModule` implements the independent candidate
interface with `Project(codes,scales,inputBF16,outputBF16,T,stream)`, `Remap`,
`ImageSha256` and idempotent disposal. It accepts T1..33 without splitting the
logical head. It enumerates exactly nine original projection entries plus the
remap bridge, materializes them, checks actual device parameter offsets/sizes,
and validates the current CUDA context, complete mathematical spans, alignment
and overlap before launch. Caller-owned spans/stream lifetimes remain caller
obligations; pointer arithmetic checks are not driver allocation-ownership proof.

The raw340MiB layout is335544320 code bytes followed by20971520 scale bytes.
Signed two's-complement nibbles encode adjacent logical K values, low nibble
first. Each64-value group has one original little-endian F16 scale. K5120 already
satisfies the row-split-K128 alignment. There is no NVFP4 divisor or extra plane.

The original K5120 dispatch is preserved exactly:

| T | Kernel | Numerical boundary |
| --- | --- | --- |
| 1 | r4/w1/direct GEMV | F32 scaled weights and ordered one-chain FMA, descending warp sum, final BF16. |
| 2..8 | exact-small-T8 | Integer Q4 is BF16-exact; each64-wide dot is accumulated before its F16 scale is applied in F32; eight K splits merge pairwise then serially; final BF16. |
| 9..33 | r64/c128 MMA, boundary variant | Each scaled weight is BF16-rounded before BF16 MMA/F32 accumulation; final BF16. |

Consequently K7 at one lane(T7) and two/four lanes(T14/T28) are different original
head recipes. They must not be silently chunked or compared as identical
arithmetic. The analytical cancellation witness is code byte0x67, F16scale0x3555,
input[0]=1,input[1]=-1: T<=8 produces BF16bits3eab (.333984375); T>=9 produces3ea8
(.328125). This is expected original dispatch behavior, not a waived error.

The independent component probe has48 cases:43 full131072x5120 projection
fixtures and5 remap-range fixtures. Projection token counts cover every original
T1..8 entry and9/14/21/28/33; a broad-envelope family additionally covers the
primary7/14/21/28 cases. Twelve active rows cross4/16/64/128 boundaries and the
last physical row; all other code rows are zero and complete outputs are checked.
The fixtures are synthetic, not the actual model's dense tensor content.

- Binary and dense-grid families prove every relevant MMA product lies on a binary lattice and the total absolute lattice units are<2^24, making every possible F32 partial sum exact. T1 has an independent original packed-word8 CPU FMA oracle; small-T has an independent exact64-dot, scale-FMA and split-reduction oracle.
- The dispatch witness requires exact bits on both sides of the T8/T9 boundary.
- Broad small-T remains bit-exact to the independent schedule. Broad large-T uses the original BF16-rounded-weight FP64 ideal and the fixed bound `(1+2^-8)*gamma_(5120+128)*sum(abs(products))+2^-8*abs(ideal)`, u=2^-24. Its Top16 composition is explicitly not run or credited.
- Every exact projection case executes GPU projection→Top16 over all131072 rows→I32 remap. Candidates are selected by shortlist row before mapping; duplicate global IDs are retained, and unary values remain exact F32 widening of selected BF16 logits.
- Primitive map tests include negative, minimum/maximum I32 and row==131072 inputs: out-of-range row IDs remain unchanged exactly, matching the original helper.

Two module lifetimes and three destination poison passes per case require96
unique lifecycle records and48 matching case identities before completion. Every
non-scratch output must also match its lifetime0 body SHA in lifetime1, including
bounded large-T heads. Full output arrays, exact expectations, applicable FP64
ideals/bounds, source allocation hashes, module/probe/backend hashes, guards,
input immutability and A/B/A comparisons are retained. Top16 workspace is
explicitly guards-only; inactive workspace is not credited as a finite/replayed
numeric result. Progress remains in `result.json`; authoritative completion is
created once in `completed-result.json` with its exact-byte SHA256 sidecar.
Use fresh ignored `artifacts/` or `docs/validation/` outputs and preserve failures.

CPU self-tests and header verification do not qualify native compilation or GPU
execution. This probe does not establish real-artifact projection agreement,
original-Ignis whole-drafter output agreement, proposal acceptance, target
verification/rollback consistency, cache/host handoff, API text quality,
concurrency/nondefault streams, long-context behavior or performance. Those
actual evaluations remain required before any adoption or quality claim.
