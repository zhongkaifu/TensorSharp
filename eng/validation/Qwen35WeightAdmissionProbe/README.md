# Qwen35 embedded draft admission

Use a real Qwen3.5/3.6 checkpoint with NextN weights on GGML CUDA:

```powershell
dotnet run --project eng/validation/Qwen35WeightAdmissionProbe -c Release -- --model C:\Works\models\mtp\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --output artifacts/admission-off --spec-mode off
```

`--spec-mode off|on|ngram|unset` resolves the startup policy before construction.
`off` must omit embedded draft tensors and their layer caches; `ngram` needs trunk
verification without learned draft weights. `on` must retain the learned head and
actually execute speculative generation after the fixed-history captures.
`unset` preserves the direct API's ability to attach a decoder after construction.
Changes to the process environment after construction do not reload weights.
With `--spec-mode ngram --exercise-ngram true`, the probe also runs a repetitive
prompt through the actual n-gram draft/verify path, rejects zero drafted tokens,
and saves the full output and counters for comparison across frozen builds.

Run the identical probe against frozen before/after binaries in separate fresh
processes. Both use the same native library, checkpoint, prompt tokens, teacher
tokens and context. The probe records 17 complete vocabulary vectors per
iteration (prefill plus 16 teacher-forced steps), rejects nonfinite logits,
records model/native/probe identities, loaded weight names and cleanup errors.
Validate every row's bytes and require matching histories and SHA-256 values;
do not substitute argmax agreement. In `on` mode also compare actual draft outputs
and execution counts against the frozen baseline.

Omitted checkpoint payload is not measured physical RAM or VRAM savings. Use the
same process/GPU sampler and application tasks for resource/performance claims.
The direct probe's durations include graph warmup and capture interference;
they are not throughput benchmarks or an independent model quality oracle.
Keep every generated file under ignored `artifacts/` or `docs/validation/`.
