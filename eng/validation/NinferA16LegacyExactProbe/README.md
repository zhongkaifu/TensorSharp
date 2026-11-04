# Exact-legacy NInfer A16 scheduling candidate

This separate candidate changes storage loads and work scheduling, while keeping
the frozen `ts_ni_project_a16` arithmetic. It uses packed 8-byte code loads and
two output rows per warp, with shared activation staging for T=2/3. Each lane
still accumulates K in order 0,8,1,9,...,7,15 within each K16 group, advancing
groups by 32, followed by the same XOR butterfly. It does not merge accumulator
chains or round the final F32 output. The existing original-Ignis-schedule
candidate remains unchanged.

No engine automatically loads this module. Its separate wrapper is
`CudaNinferA16LegacyExactModule`. An eventual explicit serving hook can use
`TS_Q35_NINFER_A16_LEGACY_EXACT_PTX`; this tool does not enable one.

From the repository root on the CUDA machine:

```sh
python3 eng/validation/build-ninfer-a16-legacy-exact-module.py --output /absolute/artifacts/a16-legacy-exact.fatbin
dotnet build eng/validation/NinferA16LegacyExactProbe/NinferA16LegacyExactProbe.csproj -c Release
dotnet eng/validation/NinferA16LegacyExactProbe/bin/Release/net10.0/NinferA16LegacyExactProbe.dll --self-test
dotnet eng/validation/NinferA16LegacyExactProbe/bin/Release/net10.0/NinferA16LegacyExactProbe.dll --execute --module /absolute/artifacts/a16-legacy-exact.fatbin --legacy /absolute/artifacts/frozen-r5-helper.fatbin --out /absolute/artifacts/a16-legacy-exact-components
```

The probe retains the 90 fixed width/row/token/pattern/cancellation cases, runs
two module lifetimes and three different destination poison passes per case,
and captures both GPU paths. It requires bitwise equality to an independent CPU
oracle that reads the original adjacent-K codes and swizzled scale storage,
and bitwise equality to the frozen legacy GPU helper. The predeclared FP64
forward-error bound, input/weight immutability, padding, guards, and repeatability
checks also remain mandatory. The analytical cancellation result is 64 for all
three token counts; the original-Ignis-schedule witness is different.

Only use the actual frozen helper image, and preserve every build/capture SHA.
Managed build or CPU self-test results do not qualify native execution, model
quality, or performance. Passing these components still requires full-model
logit, text-quality, cache/state, and end-to-end performance validation before
serving adoption. Generated evidence belongs in ignored `artifacts/` or
`docs/validation/`; reusable tools remain here.

The separate `NinferA16LegacyExactTimingProbe` accepts the same five explicit
options as the older timing probe. It requires completed certificates matching
both module images, checks the legacy image used by the exact-component probe,
verifies exact outputs before timing, and retains the ABBA CUDA-event protocol
for warm repeated addresses and rotation over more than twice actual L2 size.
These are synthetic resident-weight timings at actual projection dimensions,
not model or end-to-end speed measurements.
