# Qwen35 batched decode validation

Loads one real Qwen3.5-family GGUF and compares round-robin fused decode with
the native arena batch for 2, 3 and 4 independent requests. Each request has
a different prompt and length. Caller order rotates/reverses across steps.
The same teacher-forced token histories are used in both arms; a final serial
continuation checks that arena KV and recurrent state return correctly to holders.
By default, each row is prefilled and warmed with one serial token exactly once,
then checkpointed through the production API. Every serial, batched and sampled
arm clones that same retained checkpoint, reserves its complete decode headroom,
and starts from identical K/V and recurrent state with equal cache capacities.
No additional token is consumed after cloning. Checkpoint creation, copying and
reservation are outside the timer. Every batched step must succeed and increment the native
engagement counter. Missing models/devices or declines fail; they never count as
passing validation.

PowerShell (after building the current native CUDA library):

```powershell
$env:TS_TEST_QWEN35_MODEL = 'C:/Works/models/Qwen/Qwen3.8-27B-UD-IQ3_XXS.gguf'
$env:TS_TEST_GGML_BACKEND = 'cuda'
$env:TS_QWEN35_PROBE_STEPS = '32'
$env:TS_QWEN35_PROBE_PAIRS = '3'
$env:TS_QWEN35_PROBE_WIDTHS = '2,3,4'
$env:TS_QWEN35_PROBE_OUT = 'artifacts/qwen35-batched-decode/report.json'
$env:TS_VALIDATION_DEVICE = 'NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB'
# Record the actual unchanged dependency checkout used by the native build:
$env:TS_VALIDATION_GGML_REVISION = (git -C ExternalProjects/ggml rev-parse HEAD)
dotnet run --project eng/validation/Qwen35BatchedDecodeProbe -c Release -p:TensorSharpSkipGgmlNative=true
```

`MAX_CONTEXT` defaults to 2048 and `TS_KV_INITIAL_TOKENS` to 128 to keep the
microbenchmark bounded on a 16 GiB device. Reported capacities show what was
actually allocated. `KV_CACHE_DTYPE=q8_0` selects a separate quantized-cache run;
unset it for the production model-derived default (F16 for these weights).
Warmup pairs are excluded, and measured pair order alternates. Reports include
decode tokens/second, median wall time, engagement, numerical metrics and loaded
native/managed binary hashes. Save logs/reports under ignored `artifacts/` or
`docs/validation/`. Record `git status --porcelain` for the dependency separately.

Prompt lengths default to `32,73,137,273`, taking the first entries required by
each width. Override with `TS_QWEN35_PROBE_LENGTHS`, a comma-separated list with
at least as many positive token counts as the largest requested width. Every
used length must leave room for `steps + 2` tokens within the model's effective
`MaxContextLength`; invalid values fail with an explanatory error. The report
records the requested override and every run's actual prompt lengths.

For example, a three-request longer-context probe can use
`TS_QWEN35_PROBE_WIDTHS=3`, `TS_QWEN35_PROBE_LENGTHS=511,1025,2049`,
`MAX_CONTEXT=4096`, `TS_QWEN35_PROBE_STEPS=4`, and `TS_QWEN35_PROBE_PAIRS=1`.
Longer contexts require additional prefill time and cache memory. Clear the
override to return to the defaults; the xUnit regression explicitly retains its
original short-prompt fixture regardless of a benchmark override.

The numerical guard requires finite logits, raw cosine at least 0.995 and
temperature-one softmax `KL(reference || observed)` at most 0.02. It also rejects
any argmax change when the reference top-two logit margin exceeds the fixed value
0.25, independent of the observed error. These are explicit empirical budgets,
not a proof of quality or bitwise equality. The report retains maximum raw NRMSE,
maximum absolute error, maximum KL, and every argmax flip's step, row, token IDs,
reference logit/probability margins and KL. NRMSE remains a diagnostic rather
than an acceptance gate; tail-logit offsets can affect it substantially while
barely changing the output distribution. `TS_QWEN35_PROBE_TRACE=1` prints all
row metrics, including KL and margins.

`--collect-errors` is a diagnostic mode that retains every numerical gate
violation instead of stopping at the first one. It does not change any budget.
Each result includes `NumericalFailures` with the comparison/step/row, raw and
centered NRMSE, mean logit error, cosine, KL, maximum absolute error, token IDs,
margins, reference top probability, observed probability of that reference token,
total variation distance and violated gates. Summary statistics additionally
retain maximum centered NRMSE, maximum absolute mean error and maximum total
variation. A large top-two logit margin alone does not establish a near-one
top-token probability; the explicit probabilities resolve that distinction.
Warmup results remain excluded
from performance summaries but their numerical failures are preserved too.
Any failure leaves `NumericalPassed`, `ValidationPassed` and
`PerformanceQualified` false and exits nonzero after all requested numerical
comparisons. Raw timing rows in such a report are unqualified diagnostics.
Model/device errors, shape failures and native batch declines still stop the run.

`DistributionPassed` is a separate verdict requiring finite logits, KL at most
0.02, and no argmax flip above the fixed 0.25 reference margin. It omits the raw
cosine gate, which is sensitive to additive logit shifts even when softmax is
unchanged. `NumericalPassed` and the original `ValidationPassed` still fail when
raw cosine is below 0.995; a distribution pass never upgrades those verdicts.
Both verdicts and every violation are preserved in diagnostic reports.

Use `--greedy-reference --held-out --collect-errors` for a separate functional
continuation diagnostic. Set width 3, steps 128 and pairs 1 to evaluate the fixed
music/baking/astronomy fixture for 128 decode steps with one model load. This
requires matched checkpoints. The seed warmup's argmax supplies the first input;
the serial arm runs first and records each next argmax immediately. The batched
arm replays those exact reference inputs through its own recurrent state. Zero
argmax differences establishes greedy continuation equality by induction on
this fixture. `GreedyContinuations` records reference/observed token IDs and text,
including the common seed and final solo-continuation prediction; observed IDs
after any mismatch are predictions along the reference path, not an independently
branched greedy run. EOS does not terminate this fixed-length diagnostic.

Greedy diagnostics use a fixed serial-first order and set `PerformanceQualified`
false even if every quality check passes. They do not replace diverse task-level
quality tests or end-to-end host requests. `--held-out` selects the same fixed
three sources as the regression, plus a fourth lunar-phases question for width 4;
the exact sources are saved in the report. Omitting these switches preserves
the original teacher-forced benchmark and alternating order.

Use `--serial-control` to compare two serial runs at the largest requested width
and `max(32, requested steps)` decode steps. By default these arms clone the same
initial checkpoint; their complete metrics appear under `SerialControl`. Use
`--independent-prefill` to preserve the earlier behavior where every arm independently
prefills and warms each row. `InitialStateMode` records the mode in each result
and the report. Inspect the control before attributing observed errors to batching.
It requires corresponding context headroom. Thresholds must also be assessed on
held-out text and longer contexts; the xUnit fixture includes three fixed music,
baking and astronomy prompts in addition to the original short-prompt cases.

An earlier independent-prefill four-request, 32-step serial-only control failed
the cosine budget at step 20 despite the same top token (cosine 0.98431243,
NRMSE 0.313643, KL 0.00249914). This failure involved no batching. Matched-state
cloning isolates decode comparisons from that variation; it does not fix the
independent-prefill limitation or turn the failed original control into a pass.
Preserve its evidence and validate actual independent requests separately.

`--allow-fallback` can characterize an old binary's round-robin behavior. Its
report deliberately sets `ValidationPassed=false`. Compare only like-for-like
devices, models, settings, context lengths and load. Decode timing excludes
prefill, loading, checkpoint copies and continuation but includes graph capture,
initial device reseeding from host-authoritative clones, and managed logit copies.
It is neither steady-state-only throughput nor end-to-end agent latency. Retained
checkpoints also consume host memory during the experiment; both arms share the
same checkpoint set. Cloning is a validation setup, not a new production request
flow or a measurement of checkpoint-copy performance.

The exact IQ3_XXS regression can also run through xUnit:

The main regression runs 32 decode steps at each of widths 2, 3 and 4, covering
the previously discovered width-four KL-budget failure at step 28. Its separate
held-out three-request fixture runs eight steps. In the same F16 model load with
context 4096, it also runs 128 greedy-reference steps at lengths `500,1010,2030`
using those held-out sources, covering the discovered longer-context attention
window failure. This case requires all numerical gates and exact greedy token
agreement, 128 successful native calls and no fallback. The regression clears
and restores diagnostic snapshot and prompt-length environment overrides.
It additionally exercises the native sampled-token entrypoint,
requires its tokens to equal the host-logit batch's argmax, and compares its
final serial continuation. The three-request case checkpoints one live holder
midway, releases it, clones that exact state into its replacement while peers
remain live, then resumes decode without independent re-prefill. Checkpoint and
replacement work is outside the decode timer; retained snapshots and clones are
released through production APIs in finally blocks.

```powershell
$env:TS_TEST_MODEL_DIR = 'C:/Works/models/Qwen'
$env:TS_TEST_GGML_BACKEND = 'cuda'
dotnet test InferenceWeb.Tests -c Release -p:TensorSharpSkipGgmlNative=true --filter FullyQualifiedName~Qwen35QuantizedEmbeddingBatchedDecodeTests --logger 'console;verbosity=detailed'
```

For the exact reported prompt, run `eng/validation/probe_qwen35_reviewers.py`
against the live server. It retains all SSE frames and checks root spawn/wait
ordering and the comparison table's metric assignments. Inspect the host's
completed reviewer results and assigned inputs as well. The broader
`eng/validation/probe_multi_agent_host.py` additionally checks simple requests
and disconnect cleanup. Use the server's logs to verify the actual batched path;
correct answers alone do not establish batching.

Table cells may contain literal amounts or exact arithmetic equality chains.
Every equation segment is checked with bounded Decimal arithmetic supporting
only numbers, addition, subtraction, multiplication, division and parentheses;
the shared result must match the expected proposal and metric. The parser never
evaluates Python code. Its CPU regression tests run with
`python -m unittest discover -s eng/validation/tests -p test_qwen35_reviewers.py -v`.
Use `--verify-existing <report.json>` to reassess retained SSE evidence without
contacting the host; by default it writes a separate `.verified.json` sibling,
preserving the original report and verdict. Automated success still requires
manual review of reviewer assignments, collected results and native engagement.

For state diagnostics only, set `TS_QWEN35_PROBE_SNAPSHOT_DIR` to an ignored
artifact directory. After the first decode step, the probe binds logical row 2
and uses the production checkpoint/export APIs to write its complete Q5KC-v2
state for each comparison arm. Files share a generated comparison directory;
each has JSON metadata. This flushes arena ownership, so later decode behavior
and performance are diagnostic: the report sets `ValidationPassed=false` and
`PerformanceQualified=false`. Existing numerical thresholds remain enforced.

```powershell
python eng/validation/compare-qwen35-checkpoints.py <serial.q5kc> <batched.q5kc> --out artifacts/qwen35-batched-decode/state-comparison.json
```

The comparator requires NumPy, supports F16/F32/Q8_0/Q4_0 K/V rows, and reports
each layer's K/V prefix and newly written last row separately, plus convolution
history and delta state. Convolution rings are compared in logical time order.

For the diagnostic Q8 experiment, launch a fresh process with inherited
`TS_GGML_Q8_PARALLEL_VECTOR=1` and `TS_GGML_Q8_PARALLEL_SMALL_BATCH=1`.
Both remain off by default. The second switch covers N=2..8; each column uses
the same warp reduction as the first switch's N=1 path, without allocating or
retaining expanded weights. N>8 retains the original arithmetic. The report
records both switches and the full model SHA; native logs prove actual route
selection. Changing the environment within an already running process is not
supported because CUDA graph arithmetic must remain stable.

The local RTX 3080 Laptop run with Qwen3.5-0.8B Q8_0, F16 KV, context 8192,
initial cache 128 and prompt lengths 1024/2048/4096/6000 passed widths 2/3/4
for 32 steps and solo continuation. Matched-state logits had zero measured
error against serial and greedy continuations matched. This does not establish
language quality or resolve the separate independent-prefill diagnostic.

A separate quiet ABBA used the same native/managed files in four fresh
processes, vector mode on in both arms and only small-batch mode changed.
Each process excluded one warmup and measured three 64-step pairs per width;
context was 2048 and prompts 128/256/512/1024. With six samples per arm/width,
batch-2 median changed 3686.17 to 658.82 ms (5.60x), batch-4 3853.44 to 1228.56 ms
(3.14x). All numerical/distribution checks passed, no argmax differences or
fallbacks occurred, and all processes exited zero. This measures the described
decode microbenchmark including reseeding, capture and copies; it excludes
prefill and is not an end-to-end or independent-engine batch comparison.
Native SHA: `4bc7204e748dadb57fcc1da313d1c931da9f61fd9ab70623b9ad07ab6969f45c`.
Evidence is ignored under `artifacts/unified-memory-adaptive/q8-small-batch-model-v1/`.
