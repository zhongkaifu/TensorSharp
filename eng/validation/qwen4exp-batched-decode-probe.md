# Qwen4Exp numerical batch probe

Build native TensorSharp code against an unchanged `ExternalProjects/ggml`
checkout, then build `InferenceWeb.Tests` so its adjacent native copy matches.
The runner uses that existing test DLL directly and never builds, copies, or
patches anything. Use a fresh ignored evidence directory for every invocation.

```sh
python3 eng/validation/qwen4exp-batched-decode-probe.py \
  --model "$HOME/work/models/qwen38-flash-next-uncensored/Qwen3.8-Flash-Next-Uncensored-IQ2_XXS-00001-of-00002.gguf" \
  --backend metal --steps 16 \
  --output-dir artifacts/validation/qwen38-iq2-numerical-run1
```

`--capture-only` records provenance without launching a model. `provenance.json`
records repository and unchanged upstream GGML revisions, dirty status, source
hashes, explicit native build and test-adjacent copy hashes, managed assembly
hashes, available CMake build settings, device inventory, and relevant process
environment. Credential environment values are redacted. Model inventory records
every expected shard name, size, filesystem metadata, GGUF magic, and two small
sample windows. These samples are not full weight checksums or publisher pins;
the runner avoids hashing an entire very large checkpoint.

The default invocation records the exact argument array, exit status, elapsed
time, complete log, TRX, and managed probe report. A numerical pass requires all
of the following:

- Exactly the selected real-model test executed and passed; missing or skipped
  scenarios fail even when `dotnet test` exits zero.
- A fresh report with `completed=true`, no recorded failure, the requested model,
  effective backend, step count, and all widths 2, 3, and 4 with two repetitions.
- Complete per-step/per-row batched and round-robin traces, independent prefill
  controls, serial prefill controls, round-robin prefills, and solo continuations,
  satisfying the test's tight logit/KL bounds. Round-robin traces must follow
  step-then-row ordering, with cache binding included in the timed calls.
- The actual mapped native library reported by the test matches the explicit
  native build hash; the test-adjacent copy must also match before execution.
- Native, managed, model metadata/sample identity, and upstream revision/clean
  status remain unchanged during the test.

Failures remain failed evidence with logs and the test's partial trace when it
was able to produce one. An independent-prefill or serial-prefill-control failure
is separate from a batch comparison failure; do not attribute it to a batch that
has not executed. The probe checks teacher-forced logits and allows close-margin
greedy differences when they are numerically consistent with the logit bounds.
It does not establish exact autoregressive text parity or general language quality.

The independent serial arm completes all decode steps for one sequence before
starting the next. The round-robin arm first prefills all sequences, then performs
one token per sequence on every step, timing each `BindSequenceCache` plus
`Forward`. Its full logits are checked against the independent serial trajectory.
Old serial holders are released before allocating round-robin holders; those are
released before the batched arm. At most two groups of width-sized holders are
retained during these controls. Final batched solo continuations are compared to
saved independent serial continuation logits.

Both baseline times and ratios remain separate because their cache working sets,
live-state retention, and binding overhead differ. A batched slowdown against
independent serial does not establish a regression against interleaved scheduler
fallback. `decode_wall_speedup` keeps its original independent-serial/batched
meaning; `round_robin_decode_wall_speedup` is round-robin/batched. Timing includes
graph capture, allocation, state transfers, and dispatch. There is no excluded
decode warmup. The default statuses are `measured_not_gated`. Optional
`--min-decode-speedup 1.05` and `--min-round-robin-speedup 1.05` gate their respective
ratios separately, requiring every width/repetition measurement to satisfy the
requested bound; both can be supplied. Earlier reports without round-robin
coverage remain historical evidence and cannot satisfy this updated runner.

This short-context decode measurement makes no loading-time, prefill, HTTP
scheduler overhead, vision, or long-context throughput claim. Use the concurrent
HTTP runner and separate vision validation for those requested scenarios.

The runner records existing compiled binary identities. Source hashes do not
prove those binaries were built from the current source; rebuild after changes.

```sh
python3 -m unittest eng/validation/tests/test_qwen4exp_batched_decode_probe.py
```
