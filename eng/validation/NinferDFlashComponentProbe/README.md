# Original NInfer DFlash component probe

This reusable probe exercises `CudaNinferDraftModule` directly with actual BF16
buffers and original adjacent-K NVFP4 codes / m128x4 scale storage. It does not
attach a drafter or establish proposal quality, speculative correctness, state
rollback correctness, or end-to-end performance. The module and its unchanged
upstream headers are pinned to Ignis e600a7d9e0d5921d767a6cb464373583b3cdcaf4.

```sh
dotnet build eng/validation/NinferDFlashComponentProbe/NinferDFlashComponentProbe.csproj -c Release
dotnet eng/validation/NinferDFlashComponentProbe/bin/Release/net10.0/NinferDFlashComponentProbe.dll --self-test
dotnet eng/validation/NinferDFlashComponentProbe/bin/Release/net10.0/NinferDFlashComponentProbe.dll --self-test-all
dotnet eng/validation/NinferDFlashComponentProbe/bin/Release/net10.0/NinferDFlashComponentProbe.dll --execute --module /absolute/artifacts/draft.fatbin --out /absolute/artifacts/draft-components-fresh
```

CPU-only self-tests cannot qualify CUDA compilation or any GPU/model scenario.
The full CPU self-test constructs all 199 cases and checks oracle consistency.
Actual execution uses two module lifetimes and three destination poison passes
per case, validates complete output arrays, checks guards and immutable input
allocations, and preserves all full output captures. Failed runs retain their
original files and return a failing exit code. Output directories must be fresh
and inside ignored `artifacts/` or `docs/validation/`.

| Component | Cases | Independent contract |
| --- | ---: | --- |
| Raw A16 projections | 54 | All six original shapes, T1/2/4/8/16/17/20/21/32. Original F32 FMA chains and descending warp sum, followed by sole BF16 output rounding; bitwise CPU comparison and FP64 bound. Twelve active rows span swizzled m128 tiles; every inactive row must be exact positive zero. |
| Fused SwiGLU | 16 | Original raw gate-half/up-half rows, unrounded F32 projections into SiLU. Eight saturated cases are bitwise exact; eight bounded nonlinear cases use the fixed envelope below. |
| Dynamic convolution | 36 | Both sides, exact raw `[side][tap][channel]`, group16, two ordered FMA, lane-local zero padding, sole BF16 output rounding. H32/5120, block1/2/8, lanes1/2/8. |
| Residual addition | 4 | Exact BF16(F32(input)+F32(residual)), including in-place destination handling. |
| Plain RMSNorm | 6 | Width128/5120, rows1/7/32, FP64 ideal and plain stored weight; no +1. |
| DFlash RoPE | 12 | Q-only/K-only/both, theta1e7 FP64 frequencies, FP64 range reduction before F32 trigonometry; positions through4,000,001; position-zero identity exact. |
| Top16 | 12 | Vocab16/127/2049/248320, columns1/7/56, ties across warp/tile/tail boundaries choose smaller original row. IDs and F32 unary values exact. |
| Composed selectors | 9 | GPU predecessor→original raw-codebook rank256 scores→greedy walk, positions1/3/7 and lanes1/2/8. Binary-fraction fixtures make all products/reductions exactly F32 representable. |
| Ring append | 20 | Width1/7/8/64/2048, positions0/2047/2048/4095, two logical lanes mapped to physical slots2/0, unequal counts including zero. Every untouched ring byte must remain unchanged. |
| SWA attention | 30 | Twenty-seven exact zero-QK integer fixtures and three scoped bounded dense fixtures. Ring history uses2047 positions, current block is bidirectional across valid tokens, later query history cutoff, lane remapping, padded output zero, and poisoned stale/future values. |

The CPU projection oracle reads original raw bytes independently of the module's
helper routines. Original T1 uses four FMA chains, T2..32 one chain; T17..20 uses
eight values per lane, the other cases sixteen. Scale swizzle fixtures are written
with a five-axis traversal independently of the reference decoder's offset.
All test generation is deterministic and source-pinned; input byte lengths and
SHA256, module/probe/backend image hashes, expected arrays and applicable ideal /
bound arrays are retained alongside actual captures.

The following bounds are declared before device execution; none are fitted to
observed GPU values. Let u=2^-24, b=2^-8 and gamma_n=n*u/(1-n*u).

- Projection: `(1+b)*gamma_(K/32+16)*sum(abs(products)) + b*abs(FP64ideal)`; additionally requires exact CPU bits. Fixtures avoid overflow/underflow.
- Plain norm: `abs(FP64ideal)*(b+(width+32)*u)`. This includes BF16 output rounding and a conservative F32 reduction/normalization allowance.
- RoPE: `b*abs(ideal)+32*u*(abs(first)+abs(second))`, with the source's reduced F32 angle. Position-zero cases additionally require exact identity bits.
- SwiGLU saturation: `(b+u)*abs(FP64 gate*up)`, with exact CPU bits; gate>=32 makes F32 `1+exp(-gate)` exactly1. Nonlinear: `b*abs(idealSiLU)+32*u*abs(gate*up)` for the deliberately bounded finite gates.
- Exact SWA: `b*abs(ideal)+8*u`, additionally requiring exact bits. Q=K=0 makes p=1, integer partial sums<=72 remain exact BF16, and the CPU applies the original direct reciprocal or split division before BF16 rounding.
- Dense SWA: `1/64+abs(FP64ideal)/128` for admitted |Q|,|K|<1/8 and |V|<=1. This scoped envelope covers the original BF16 probability/partial/output boundaries; it is not a universal error theorem or a full-model quality gate.

SWA scratch and Top16 workspace are explicitly **guards-only** contracts. Their
internal payloads have no CPU exact/bound oracle; scratch repeat/finiteness is not
required, and inactive SWA statistics may legitimately retain poison or contain
negative infinity. Numeric counter fields are not separate passing checks for
these buffers; `scratchGuardsOnly` identifies the actual contract. Full output
arrays and all immutable inputs still receive their applicable checks.

Not covered here: actual artifact tensor content, target-tap copy / hidden packing
/ BF16-to-F32 bridge / block preparation helpers, reused full target W8 head,
complete five-layer draft execution, acceptance statistics, speculative verify /
rollback/cache handoff, concurrency, API text quality, long context, or runtime
performance. These require separate actual tests and cannot be inferred from
metadata binding or CPU construction success. Raw-projection fixtures use sparse
nonzero rows at full original matrix geometry; they do not qualify all dense
real-weight numerical cases.

`result.json` remains the replaceable progress record. After every existing gate
passes, the producer additionally creates `completed-result.json` exactly once
with `FileMode.CreateNew`, then creates `completed-result.json.sha256` with the
SHA256 of those exact serialized bytes. Both writes are flushed. A verified
completed file and sidecar are the authoritative final record; the progress file
alone cannot substitute for them. Existing completed files are never overwritten.
