Release remains blocked. This candidate clears the retained checkpoint-shaped
CUDA tensor-parallel numerical failure, but its first per-strip benchmark
regresses materially. It is not performance-qualified or full-model-qualified.

The owned CUDA operation retains the unsplit stream-K K intervals and reduction
order while only reading its actual weight strip. Existing upstream tile and
quantization helpers are called unchanged. It uses TensorSharp-owned persistent
scratch, not a borrowed ggml CUDA pool. The independent full-weight oracle and
original absolute/relative tolerances are unchanged.

- The unchanged r3 binary reproduces Q2_K/Q3_K at 16 tokens on physical GPUs0/6:
  relative L2 `3.88634e-5`, exceeding `1e-5`, exit1.
- The candidate compiled with the production precision flags passes all16
  physical two-device checkpoint comparisons, both Q2_K/Q3_K and Q4_K/Q6_K.
  Maximum relative L2 is `1.6837e-7`; no weight bytes are duplicated or dropped.
- An earlier isolated fast-math build passes36 one-device projection/MoE cases
  across tokens9,16,17,33,65,129, one/six selected experts, and strip counts2/4/7/8.
  Gate/up values are bitwise equal to the untouched full-weight projection.
  This sweep is not attributed to the later precise-flags binary.
- The paired microbenchmark uses the precise-flags candidate on GPU6,60 warmed
  alternating pairs per case, with other campaigns active elsewhere on the VM.
  Across18 cases, per-strip MoE median latency increases22.8%–82.8%.
  This is diagnostic evidence of a regression, not an accepted quiet benchmark.
- Physical4/7/8-device comparisons, full-checkpoint tokens/logits, memory and
  quiet end-to-end throughput/latency qualification remain unfinished.
- DSpark strict CUDA prefix5 confidence and CPU prefix11 target-logit failures
  recorded in the sibling numerical-r3 report remain unresolved.

See record-r1.json for exact binaries, source hashes, dependency revision and
limitations; the logs retain all original assertions and failing/passing exits.
The upstream ggml checkout is clean at
`456172ec733a135778adcd32d00e576a58232e45`.

Commands (on the VM):

```sh
CUDA_VISIBLE_DEVICES=0,6 GGML_CUDA_DISABLE_FUSION=1 /workspace/ts-codex-20260916-r3/repo/TensorSharp.GGML.Native/build/GgmlOpsDsv41TpTest --cuda 2 --checkpoint-shape
CUDA_VISIBLE_DEVICES=0,6 GGML_CUDA_DISABLE_FUSION=1 /workspace/ts-tp-virtual-prototype-20260916/integrated-test --cuda 2 --checkpoint-shape
CUDA_VISIBLE_DEVICES=6 GGML_CUDA_DISABLE_FUSION=1 /workspace/ts-tp-virtual-prototype-20260916/strip-benchmark
```
