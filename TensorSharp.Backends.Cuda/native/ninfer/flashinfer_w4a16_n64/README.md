# Separate original N64/F32 W4A16 trial

Four Python files and their license under `upstream/` are byte-exact FlashInfer
`61bc8b6dd7049a44d614b2a0cd4a036cb3c46ad0` sources with pinned Git blob and SHA-256
identities. The original N64 class already has a generic output store; this
TensorSharp-owned build selects Float32 without changing an upstream file.

The fixed first tactic uses T1/T8, N34816/K5120, split1, four warps, K128, two
stages and PDL disabled. It stages only `min(M+1,16)` activation rows, including
one zero row, and 64 output columns per block. The grid is 544 blocks. Source
shared-memory reservations are 13,376 bytes at T1 and 17,184 bytes at T8.

Raw adjacent-K weights, compact M128x4 E4M3 scales, actual BF16 input and F32
post-MMA alpha/output have the same contract as the N128 trial. Global alpha
and accumulation association still differ from the legacy scalar schedule.
This supplies no serving hook, default, end-to-end gain or quality claim.

The separate component and timing projects link the frozen N128 test sources
unchanged, with a type alias selecting the N64 module. Thus the exact cases,
bounds, eight cases/two lifetimes/three poisons, full-array checks and certificate
gates are identical. The N128 builders, wrappers, tools and prior evidence are
not edited by this trial.

```sh
python3 eng/validation/build-ninfer-flashinfer-w4a16-n64.py --verify-only
/root/venvs/vllm/bin/python eng/validation/build-ninfer-flashinfer-w4a16-n64.py --out /fresh/artifacts/aot-n64
dotnet build eng/validation/NinferFlashInferA16N64Probe/NinferFlashInferA16N64Probe.csproj
dotnet eng/validation/NinferFlashInferA16N64Probe/bin/Debug/net10.0/NinferFlashInferA16N64Probe.dll --execute --manifest /aot-n64/module.json --out /fresh/artifacts/components
```

Use `NinferFlashInferA16N64TimingProbe` with the same seven explicit options as
the N128 timing tool. Matching actual component certificates are required. The
existing TensorSharp-owned parser/artifact helper is pinned by SHA-256 in the
N64 builder; it is used at build time only. All CUDA compile, device, quality and
speed checks must be reported separately from CPU source/contract checks.
