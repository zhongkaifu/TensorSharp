# GLM unified session probe

This serial-request tool uses the public production route:
`UnifiedMemoryModelSession.Create` → `ConfigureScheduler` →
`InferenceEngine.SubmitRequest`. It does not call native evaluation directly or
substitute an independent memory-admission implementation.

Build the managed tool into a private output, explicitly skipping native builds.
Copy the separately qualified production `GgmlOps.dll` (or `libGgmlOps.so`) into
that output afterward. The project never copies a repository native build over
the caller's pinned deployment. Use a fresh process and an exclusive GPU window.

```powershell
dotnet build eng/validation/GlmUnifiedSessionProbe/GlmUnifiedSessionProbe.csproj -c Release -m:1 -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -p:OutDir=D:/TensorSharp-validation-cache/glm-session-probe/bin/
dotnet D:/TensorSharp-validation-cache/glm-session-probe/bin/GlmUnifiedSessionProbe.dll --self-test
```

`--self-test` exercises token parsing, exact task/EOS checks and the production
GLM output parser with synthetic text. It never opens a model or invokes a
native/device API. Building/testing has to be done by the invoking runner; a
source-only preparation is not a passing qualification.

From the repository directory, a matching plain-AR conditioning pilot is:

```powershell
dotnet D:/TensorSharp-validation-cache/glm-session-probe/bin/GlmUnifiedSessionProbe.dll --output artifacts/default-readiness-20261010/glm-unified-session-math-v1 --tokens-file artifacts/default-readiness-20261010/external-baseline-fairness-readonly-v2/glm-math-prompt.ids --model E:/models/GLM-5.3-Flash-GGUF/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf --model-identity artifacts/default-readiness-20261010/glm-e-identity-v1.json --context 1024 --max-new 128 --prefill 32 --expert-prefill-rows 32 --selected false --cpu-threads 8 --deadline-seconds 1200 --task math42 --native-sha256 REPLACE_WITH_QUALIFIED_NATIVE_SHA256
```

The raw IDs are historical input conditioning, not generated output ground
truth. The independent task is **17 + 25**. `math42` requires the decoded prompt
to contain that exact user instruction, natural checkpoint EOS, a clean finished
request and exact production-parsed content `42`. It does not trim whitespace,
strip punctuation, stop a repeating model or repair output. It preserves raw
tokens/text, reasoning and parsed content. `--task unknown` records other tasks
as unassessed; it does not turn them into passing quality results.

Operator ceilings are `--host-bytes`, `--device-bytes`, `--expert-bytes`;
omission selects the public session's automatic policy. `--selected true` opts
into its experimental GLM adapter. Async file prefill, growth, MTP/draft, image
loading, prefix reuse and additional requests remain disabled. Inherited
speculation or CPU placement overrides are rejected. Threads are applied through
`MoeCpuOffloadConfig.SetCpuThreads`, and relevant inherited/effective thread
environment is recorded; actual native thread-pool logging must also be kept by
the invoking process wrapper.

`--repetitions 2` runs two requests in the same model/session/engine using unique
request IDs and the identical original raw prompt. Prefix caching and retained
conversations stay disabled; every round must independently finish and pass
the semantic check. The first round is the first request of this process. Later
rounds retain expert weights and OS/vendor state, and are explicitly labelled
warm. The tool does not purge OS caches or assert that the first round is
physically cold. `Rounds` and separate `round-NN-report.json` files preserve all
raw outputs, request timing, admission ledgers and publication observations.
The top-level request fields describe the last started round. An incomplete or
failed round cannot be hidden by a later successful answer. The deadline bounds
the whole run, including construction and all repetitions.

`--cache-telemetry true` records native expert-cache usage, entry capacity and
resident slots, growth and pipeline counters at quiescent boundaries before
requests, after completion, and during disposal. These observations occur
outside measured request windows. The entry's `Layer` is the first-creator
label of a pooled quantization/layout group, not a separate cache per layer.
Native counters are process-wide and may reset during full teardown; preserve
snapshots rather than subtracting counters across a reset. Logical cache bytes
exclude complete vendor, driver and OS ownership.

For finer diagnostics, add `--cache-token-snapshots true`. This closes the
public compute gate and drains the model compute lock after each consumed
publication. One step already in flight may finish; the captured computed-token,
output and engine-step counts identify the actual boundary. This mode changes
timing and must not qualify quiet throughput. It does not supply an exact
prefill-only observation. Incremental `cache-snapshots.jsonl` remains available
if a subsequent native crash prevents the final report.

Per-entry `file_read_calls`/`file_read_bytes` are not publicly exported. Set
**inherited** `TS_HOST_MOE_EXPERT_CACHE_DIAGNOSTICS=1` before launching the
process and retain native stderr teardown markers. The pipeline source-byte
counter covers only its own path and cannot be called total expert reads.
Selected graph/seam counters require a native model handle unavailable through
the public session; the probe does not reflect or bypass that ownership boundary.

The report uses **`session.PrefillChunkTokens`** and verifies the configured
scheduler retains that effective cap. The requested cap is not assumed to have
been accepted by the architecture's planner. `InferenceCompletion` supplies
attributed forward time; request wall and engine/consumer first-token latency
are separate. Final output is forwarded here (`P+O`), whereas Maya stops before
forwarding its terminal sample (`P+O-1`). These timings alone do not establish
equal consumed work or fair performance ratios.

`--capture-final-logits true` copies the complete finite vocabulary vector under
the model compute lock after the terminal output token has been forwarded. This
validation-only reflection reads the model buffer, not the sampler's retained
logits. The resulting binary is little-endian F32. Compare vectors only when the
entire prompt and consumed output history agree; matching prompt text alone is
insufficient. Capture happens outside the request timing and is not an
independent arithmetic or semantic oracle.

For fixed teacher conditioning, `--capture-token-id ID` additionally requires
`--capture-final-logits true --max-new 1 --task unknown`. It uses the ordinary
sampler's first-token allowlist and verifies that exactly that non-EOS checkpoint
token was forwarded. Put any preceding teacher tokens in the raw prompt file.
This permits identical-history numerical comparisons even when ordinary greedy
paths would diverge. It cannot qualify answer quality, natural EOS, or decode
throughput, and does not change the production server or its sampling policy.

`report.json` is written in `finally`, including failures, native/managed/tool
identities, original model size/mtime, exact budget high-water marks and ledgers
at completion, engine disposal and session disposal. Large model content is
linked to the supplied historical full-hash manifest after exact size/mtime
checks, rather than rehashed during this run. `--model-identity none` leaves
model content identity unqualified and cannot produce overall `Passed=true`.
Observed RAM/commit/whole-device samples do not constitute complete physical
ownership or OS hard limits.

The finite cancellation deadline starts before synchronous model construction.
The public loader/native dispatch cannot safely be forcibly interrupted by this
tool. Engine disposal may wait up to its public 60-second join bound; if that
fails, the tool retains the session owner and reports incomplete cleanup rather
than freeing live model buffers. An external wrapper owning **this process** is
required for a hard process deadline or native crash recovery. The tool never
terminates or manages other processes. A native crash can only leave
`execution-started.json`; that checkpoint is not a successful cleanup receipt.
