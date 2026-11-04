# NInfer BF16 T8 MMA build and validation

The serving implementation automatically selects this optional module only for
the known NInfer Qwen3.8 recipe on compute capability 12.0/170 SMs, declared
configured context exactly 40960 or 65536, and verification plus decode graphs
already enabled. Runtime
dispatch requires a full single-lane T8 Verify pass and a BF16 descriptor for
Input `14336x5120` or Output `5120x6144`. AB, ordinary prefill, decode,
mixed/padded passes and other shapes retain their original paths. FlashInfer
and ExactVerify overrides are excluded.

`TS_Q35_NINFER_BF16_MMA` unset means qualified automatic selection, `0` disables
it, and `1` requires a supported profile and asset. Invalid values are rejected
for NInfer; GGUF ignores the setting. This flag does not enable DFlash,
speculation or verification. An absent file/directory can fall back before
automatic module publication; ABI, driver, allocation and execution failures
remain errors. There is no execution-time retry after a failed launch.

The startup log reports selection, actual module SHA, policy and scratch size.
`auto-sm120-170sm` means successful automatic selection. Host submissions include
capture/eager callbacks, not individual GPU graph replays; graph capture and
launch counters are reported separately.

Normal builds use the dedicated C++17/sm120a target, independently of the generic
portable compute-75 loop:

```sh
dotnet msbuild TensorSharp.Backends.Cuda/TensorSharp.Backends.Cuda.csproj \
  -t:SelectNinferBf16MmaContent -p:CudaArch=sm_120a
```

The fixed flags are `-O3 --ftz=false --fmad=true -lineinfo`; generic fast-math
settings do not alter them. The target reuses host compiler discovery, tracks
the owned source and pristine dependencies, and selects PTX/fatbin/PTX-hash
sidecar plus the original license/manifest for output and publish. Compilation
is skipped without NVCC, during design-time/managed-only builds, and without a
120/120a target or 120 in the fatbin list. Managed-only builds preserve existing
deployed assets. The paired sidecar binds the PTX hash, not the fatbin hash.

The separate component builder remains available for isolated qualification:

```sh
python3 eng/validation/build-ninfer-bf16-mma.py --verify-only
python3 -m unittest discover -s eng/validation/tests -p test_ninfer_bf16_mma_build.py
python3 eng/validation/build-ninfer-bf16-mma.py --out /absolute/fresh/build-directory
```

The original Ignis e600 MMA global kernel and dependencies are unchanged in
`TensorSharp.Backends.Cuda/native/ninfer/bf16_mma/upstream`, with Apache-2.0
license and Git object/SHA-256 manifest. The owned file explicitly instantiates
the observed Ignis binary schedule `32x32x256`, warp `16x8`, three stages,
minimum one block, `cg/cg`, PingPong, TokenFast, Xor64. This differs from the
e600 source's production alias. The serving wrapper reuses these original
instantiations and the existing input casts and consumer boundaries.

Each shape, `14336x5120` and `5120x6144`, has a raw F32-output instantiation
and an original BF16-output reference. Both use `FullTokens=false`, runtime
`tokens=8`, grid `outputRows/32`, block 256, and 98,304 dynamic shared bytes.
The caller must opt in to that dynamic shared-memory size before capture.
The serving wrapper performs that opt-in during initialization. No split-K or
post-reduction kernel is introduced.

The raw F32 policy only stores the original MMA accumulator; it does not
round before TensorSharp's existing projection consumer. Four kernel arguments
are input pointer, row-major weights pointer, by-value output policy, and I32
token count. Raw policy sizes/offsets are `8,8,8,4` / `0,8,16,24`;
original policy uses `8,8,16,4` / `0,8,16,32`. The builder records the actual
PTX entry names and parameter storage and refuses a differing ABI.

Same K16 MMA granularity alone does not prove equality with cuBLAS WMMA. R52's
nine-real-matrix screen and R53's full repeated component campaign passed exact
old raw F32, consumer and canonical residual bits, fixed FP64 bounds and guards.
The normal MSBuild image also passed all 811,008 real projection values and
their consumers. Future builds must retain the exact-old gate; stop and preserve
complete raw outputs on a mismatch before timing.

Normal compilation changed whole-file hashes only through compiler-generated
debug strings. Executable PTX, all four SASS text sections, constants, resource
metadata and relocations matched the qualified build. Original certificates
and hashes were retained; a separate normal-image manifest and actual-image
screen were used. Do not strip line information or relabel new bytes with an old
hash. The actual loaded SHA/policy binds graph and KV cache identities.

R54's one matched OFF/ON/ON/OFF block measured code decode throughput
436.124→438.705 token/s, **+0.59%**, with prose +0.68% and copy +0.53%.
Full responses, typed usage and draft work matched between arms. Component
speedups are not a 20% engine speedup, and this finite matched block is not a
general performance guarantee.

The final automatic-path evaluation scored **38/45 text tasks and 26/30
programs**. Against R45, all 309 checks, 96 arithmetic fields (61 correct) and
189 program-case statuses were unchanged, with zero loss of an existing
successful check or executed case. All 123 serial text responses and 30 generated
program contents were byte-identical; 16 parallel responses changed without a
score regression. Existing model mistakes remain. This establishes no loss
within the fixed evaluated suite, not perfect quality or a general engine rank.
CPU, GPU component, end-to-end performance and model quality results remain
separate; missing or unfinished cases never count as passes.

The subsequent 64K qualification reused this same native image and arithmetic.
R61 toggled BF16 MMA, residual/norm and GateUp fusion together: one matched block
measured code 406.49→442.42 token/s (+8.84%), with identical responses, typed
usage and draft work. That is a combined result, not the gain from BF16 MMA
alone. Four repeated 64,416-input/144-output/cache0 diagnostics passed all ten
needle/order checks; attention above 40,960 visible keys retained its fallback.
R63 used the same shared 64K JSON through Server.Host and completed 38/45 text
and 26/30 programs with all 309 checks, 96 arithmetic values and 189 program
statuses unchanged against R60. These finite tests do not extend qualification
to arbitrary configured contexts or complete 128K/250K campaigns.
