# CPU greedy argmax validation

The production sampler uses the unchanged scalar implementation by default.
Only exact `TS_TOKEN_SAMPLER_SIMD=1`, read once at construction, opts into portable
SIMD for arrays of at least128 floats when hardware acceleration is available. Ordered
comparisons and a first-equal search preserve leadingNaN, laterNaN, ties, signed
zeros, infinities and the original empty-array exception. No arithmetic or model
precision changes. Penalties, grammar and first-token policies keep their order.

Build this Runtime-only console in a separate CPU setup window, then run on an
otherwise idle CPU (no server/build/staging concurrently):

```sh
dotnet build eng/validation/TokenSamplerArgmaxBench -c Release
DOTNET_TieredCompilation=0 dotnet eng/validation/TokenSamplerArgmaxBench/bin/Release/net10.0/TokenSamplerArgmaxBench.dll \
  --sampler-source /ABS/REPO/TensorSharp.Runtime/TokenSampler.cs \
  --out /ABS/REPO/docs/validation/argmax/fresh.json --rounds 20 --iterations 100
```

Use `--rounds 20 --iterations 100` (spaces are required). The console rejects
fewer than20 rounds. Both arms invoke the real complete `Sample` method with
temperature0, in the same binary. Reflection only constructs the internal
explicit scalar/vector controls outside timing. Every measured ID matches an
independent scalar oracle and all input bits remain unchanged. Raw ABBA samples,
allocation counts, loaded Runtime SHA and actual SIMD/CPU ISA are recorded.
The console uses explicit scalar/vector controls regardless of the public
default. Its metadata records the inherited public control separately.

Run `TokenSamplerArgmaxVectorTests` on each relevant ISA; a passing ARM CPU
result does not establish x86 performance or GPU/E2E model acceptance. The
console's warm managed-array timings do not attribute any complete request gap
to argmax. Keep model-level quality and same-host server A/B separate.
Current model measurements do not establish an end-to-end speed gain, and the
paired multimodal no-loss gate was not met. Keep SIMD opt-in experimental;
finite CPU equivalence and microbenchmark gains do not waive model-level failures.
