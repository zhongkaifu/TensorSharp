# FlashInfer AOT versus qualified exact A16 projection timing

This separate tool leaves all prior kernels and timing tools unchanged. It times
one full gate/up matrix, N34816/K5120, at T1 against the qualified vector exact
kernel and at T8 against the qualified batch exact kernel. The actual frozen
`NinferA16BatchExactTimingProbe.Make` is invoked by reflection from its referenced
assembly, preserving its raw bytes, TS planes, random sequence and F32 inputs.
Compact BF16 inputs are extracted losslessly. The generator assembly identity
and every source hash are recorded.

Matching actual GPU component certificates and raw-capture hashes for all three
images are mandatory before timing. The FlashInfer and batch certificates must
be authoritative write-once `completed-result.json` files with matching sidecars.
Each arm retains its own numerical gate: exact legacy CPU bits for the reference;
the predeclared normal-finite MMA FP64 envelope for FlashInfer. The two arms are
not expected to be bit-identical and no quality-equivalence claim is made.

Five ABBA blocks per cache-address mode use CUDA events around projections only.
All resident addresses and full outputs are checked before timing and each timed
interval's final output is checked afterward. Guards, input/weight immutability
and same-arm repeated bits are mandatory. Address rotation exceeds twice the
reported L2 capacity per arm; that does not establish physical cold-cache state.
The descriptive medians do not establish statistical significance, end-to-end
speed, model quality, or a fastest-engine ranking.

```sh
dotnet build eng/validation/NinferFlashInferA16TimingProbe/NinferFlashInferA16TimingProbe.csproj
dotnet eng/validation/NinferFlashInferA16TimingProbe/bin/Debug/net10.0/NinferFlashInferA16TimingProbe.dll --self-test
dotnet eng/validation/NinferFlashInferA16TimingProbe/bin/Debug/net10.0/NinferFlashInferA16TimingProbe.dll --execute --manifest /aot/module.json --component /fi-components/completed-result.json --vector-module /vector.fatbin --vector-component /vector-components/result.json --batch-module /batch.fatbin --batch-component /batch-components/completed-result.json --out /fresh/artifacts/comparison
```
