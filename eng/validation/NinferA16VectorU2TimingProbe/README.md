# Direct qualified-vector versus U2 timing

This new tool times two `CudaNinferA16LegacyExactModule` images directly. The
reference must be the already-qualified vector image; the candidate must be the
separate U2 image. It never uses the generic scalar helper for a timed arm.
Existing frozen timing tools remain unchanged.

```sh
dotnet build eng/validation/NinferA16VectorU2TimingProbe/NinferA16VectorU2TimingProbe.csproj -c Release
dotnet eng/validation/NinferA16VectorU2TimingProbe/bin/Release/net10.0/NinferA16VectorU2TimingProbe.dll --self-test
dotnet eng/validation/NinferA16VectorU2TimingProbe/bin/Release/net10.0/NinferA16VectorU2TimingProbe.dll --execute --reference-module /absolute/artifacts/qualified-vector.fatbin --candidate-module /absolute/artifacts/fresh-u2.fatbin --reference-component /absolute/artifacts/vector-strict90/completed-certificate.json --candidate-component /absolute/artifacts/u2-strict90/completed-certificate.json --out /absolute/artifacts/vector-vs-u2-fresh
```

Both arms require completed strict90 certificates matching their actual image
SHA, 180 lifecycle records, three poison captures each, every existing exact /
FP64 / guard / source-immutability / repeat gate, and all original output hashes.
Both certificates must use the same frozen scalar helper for correctness
comparison. Their scalar helper is not timed. Distinct image hashes are required.
Keep build provenance tying the reference image to vector source SHA159f348e...
and the candidate image to U2 source SHA27edc4af...; the two images intentionally
share the same exported ABI, so function names alone cannot identify their source.

The frozen synthetic generator, all twelve shape/T profiles, resident-address
validation, five ABBA blocks, warm-repeat and over-two-L2-address rotation labels,
CUDA event intervals and descriptive statistics remain unchanged. Every full
output must match the same exact-legacy CPU oracle and FP64 bounds before and
after timing. Allocations, copies, preparation, warmup and validation are outside
events. No cache flush cost is hidden, and physical cold cache is not claimed.

No new acceptance threshold is invented. The tool retains min/max/median and
descriptive reference-over-candidate ratios, marks significance/adoption as
unestablished, and makes no model or end-to-end speed claim. The separate model
logit, quality and end-to-end gates remain necessary. Completed output uses
CreateNew `completed-result.json` plus its exact-byte SHA256 sidecar; progress
continues in replaceable `result.json`. Use fresh ignored evidence directories
and preserve failures.
