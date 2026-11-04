# NInfer BF16 T8 MMA projections

This optional module accelerates the original NInfer Qwen3.8 BF16 exception
projections while preserving TensorSharp's F32 output and existing consumers.
It uses the pristine Ignis e600 MMA core with TensorSharp-owned instantiations
and an unrounded F32 store policy. Original sources, Apache-2.0 license and
Git object/SHA-256 manifest are retained unchanged in `upstream/`.

The automatic serving route is limited to the known NInfer recipe on CUDA
compute capability 12.0 with 170 SMs, declared context 40960, and verification
and decode graphs already enabled. Each eligible call must be a complete
single-lane T8 Verify pass. Only BF16 Input `14336x5120` and Output `5120x6144`
projections use this module. AB, ordinary prefill, decode, mixed/padded batches
and other shapes retain their existing paths. FlashInfer and ExactVerify
overrides are excluded from this route.

`TS_Q35_NINFER_BF16_MMA` controls selection:

| Value | Behavior |
| --- | --- |
| Unset | Automatically select for the qualified profile when the asset exists. |
| `0` | Disable this route. |
| `1` | Require the qualified profile and asset; unsupported or missing assets fail. |

Other values are rejected for NInfer. GGUF ignores this setting. Selection does
not enable verification, DFlash or speculative decoding. Automatic fallback is
limited to missing files/directories before module publication; an invalid ABI,
CUDA error or allocation/execution failure is not silently retried.

The dedicated MSBuild target `CompileNinferBf16Mma` uses C++17, `-O3`,
`--ftz=false`, `--fmad=true`, `-lineinfo` and `sm_120a`, with the shared host
compiler resolver. It emits the PTX, fatbin and PTX-binding hash sidecar, and
selects them plus the license/manifest for output and publish. It runs only for
a resolved 120/120a architecture or a fatbin architecture list containing 120.
The generic portable compute-75 kernel loop is unchanged. Managed-only
`TensorSharpSkipCudaPtx=true` builds preserve deployed assets and do not rebuild
this module; no-NVCC and design-time builds also skip its compilation.

The fixed schedule is `32x32x256`, warp `16x8`, three stages, `cg/cg`, PingPong,
TokenFast and Xor64, with `FullTokens=false`, tokens=8, 256 threads and 98,304
dynamic shared bytes. This is the observed Ignis binary schedule instantiated
from the original core, not the e600 source's different production alias.
The managed wrapper validates the actual four-argument ABI and opts in to the
shared-memory size before capture. Existing F32→BF16 input casts, Input rounding
and Output residual consumers remain unchanged.

Whole-file module hashes can differ through compiler-generated debug strings.
The normal-build admission compared executable PTX and all four CUDA SASS text
sections, constants, resources and relocations, then ran the actual normal
image's nine-matrix exact-old screen. Actual loaded image SHA and policy remain
part of graph/cache identities; debug differences never justify substituting
an old artifact's hash.

See [the validation guide](../../../../eng/validation/README-ninfer-bf16-mma.md)
for build commands, coverage and performance limits. The final automatic-path
suite scored 38/45 text tasks and 26/30 programs. Against R45, all 309 checks,
96 arithmetic fields (61 correct) and 189 program-case statuses were unchanged,
with no loss of an existing successful check or executed case. Existing model
mistakes remain; this is no-loss evidence within that fixed suite, not a general
quality or engine-ranking claim.
