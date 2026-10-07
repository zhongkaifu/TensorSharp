# AgentTurnBench

Does a multi-token input reach the model as **one batched forward per chunk**, or
as one forward per token? This benchmark drives the continuous-batching engine
(`InferenceEngine` → `ContinuousBatchScheduler` → `BatchExecutor`) — the path the
server and TensorAgent use — with the conversation shapes an agent actually
produces, and reports per request: prompt tokens, tokens served from the cache,
engine steps, tokens per prefill step, TTFT, prefill and decode rates, and under
speculation the drafted / accepted / verify / plain counters. Every speculative
stream is compared against plain greedy token for token.

Timing fields distinguish model computation from token delivery:

- `PrefillComputeMs` and `DecodeComputeMs` use the engine completion's accumulated
  forward times, including speculative drafting, verification and accepted-prefix
  replay. They exclude scheduler waiting, media preparation, cache bookkeeping and
  client consumption. Shared batched forwards are apportioned equally among their
  sequences; concurrent rows sum the per-request compute times and counts.
- `PrefillComputeTokens` is completion prompt tokens minus prefix-cache reuse;
  `PrefillComputeTps` divides that count by prefill compute seconds.
  `DecodeComputeTokens` is the engine completion output count, **including terminal
  EOS**. `DecodeComputeTps` divides that count by decode compute seconds. This count
  differs from visible `OutTokens` and the API's `eval_count`, which exclude EOS.
  Failed requests have no compute fields; rates are absent when their denominator
  is zero. A concurrent row has no compute fields if any request failed.
- `TotalMs` is submission-to-completion wall time and `TtftMs` is time to the first
  delivered token. Legacy individual-row `PrefillTps` is `Fresh / (TtftMs / 1000)`;
  `DecodeTps` is `(OutTokens - 1) / ((TotalMs - TtftMs) / 1000)` when defined.
  A speculative first step can deliver several tokens together, so that decode
  formula removes only one token from the burst while excluding the entire first
  step's time. It must not be interpreted as measured decode compute throughput.
  Concurrent `DecodeTps` instead counts deliveries strictly after every request's
  first delivery over the remaining wave wall time.

Use the compute fields together with `TotalMs` for new performance comparisons,
and retain `TokenTimesMs` to assess delivery latency. Existing delivery fields and
the comparator's legacy thresholds remain unchanged for older result files.
The `image` scenario renders and prepares/encodes its image before submitting each
request. Consequently **both `TtftMs` and `TotalMs` exclude image preparation and
encoding for every candidate**, including the first candidate and later candidates
that may reuse an encoded-image cache. They measure engine execution after the
prepared prompt is ready. Use HTTP client wall time for end-to-end image latency;
do not attribute differences in these Agent image timings to encoder cache reuse.

| scenario | what it sends |
| --- | --- |
| `short` | a tiny prompt with a one-line system prompt; `--short-prompt <text>` overrides the default "Say the single word: apple." |
| `long` | an agent system prompt plus a long pasted source file (`--long`, default 4096 tokens) |
| `tool` | a turn, the same conversation extended by a `--tool`-token tool result (default 3000), then a short follow-up — rendered through the chat template with the model's raw output tokens spliced in, as the server does |
| `newchat` | two conversations sharing the system prompt (the shared-prefix checkpoint) |
| `spec` | plain greedy vs n-gram vs the checkpoint's own drafter, then the tool rounds under n-gram |
| `json` | grammar-constrained JSON, plain vs n-gram |
| `conc` | N concurrent requests on one engine, then a solo request after them |
| `image` | a turn carrying an image plus a code snippet to repeat, plain vs n-gram and the checkpoint's attached drafter; needs `--mmproj <gguf> --image <file>`, is skipped without them, and is not in the default scenario list |

Use `--layer-split N` for whole-layer placement or `--tp N` for tensor parallelism
on one node. The two modes cannot both exceed one; unsupported architectures
refuse the requested mode. Distributed placement is measured through the CLI/server.
`--spec-only` omits the extra tool rounds from the `spec` scenario, keeping its
plain, n-gram and attached-drafter comparisons.
`--image-new N` sets the image scenario's output budget (default 160 tokens).
`--image-file N` sets the code snippet's approximate token count (default 200).
`--spec-candidates ngram,auto` selects the algorithms compared with the plain
control in `spec` and `image`. By default both run when a head is attached;
explicit `auto` requires an attached head. Use `--spec-candidates auto --spec-only`
for a plain-versus-learned comparison without the ngram or extra tool rounds.

`--spec-engine ngram|auto` enables speculation on ordinary scenario engines.
The explicitly labeled plain controls in `spec`, `json`, `newchat` and `image`
always disable it, so they remain independent comparison controls. The `conc` rows measure
concurrency WITH speculation: the planner keeps a multi-sequence step plain and
the interesting cost is the transitions. `--conc-stagger <ms>` spaces the
submissions of a concurrent round so later requests arrive while earlier ones
already decode (and speculate); `--conc 1,2,4` includes a solo round as the
reference. Concurrent rows aggregate the actual per-request speculative counters;
zero counters do not establish speculative engagement.

`--conc-gate` holds the engine's compute gate closed until a whole concurrent round
is queued, so the first scheduler step sees every request. Without it the
submissions race the engine thread: one run may prefill the first request alone on
the solo fused path and the rest a step later, another may admit all of them
together. Different batches run different kernels, which agree only up to
floating-point near-ties, so greedy tokens of an ungated concurrent round can change
from run to run without a defect. Gated rows record `ArrivalOrderFixed: true`. It
cannot be combined with `--conc-stagger`. For a determinism investigation run the
engine with `TS_CB_DEBUG=1`: each step prints the requests it scheduled and a
fingerprint of their logits (top two tokens, margin, hash), so two runs can be
diffed step by step (see
[Output identity under concurrency](../../docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md#output-identity-under-concurrency)).

```
dotnet build benchmarks/AgentTurnBench -c Release
dotnet benchmarks/AgentTurnBench/bin/Release/net10.0/AgentTurnBench.dll \
    --model ~/work/models/gemma-4-E4B/gemma-4-E4B-it-Q8_0.gguf \
    --draft-model ~/work/models/phone-stage/e4b/mtp-gemma-4-E4B-it-Q8_0.gguf \
    --backend ggml_metal --chunk 1024 --scenarios short,long,tool,newchat,spec,json,conc
```

`--chunk` is the solo prefill chunk (1024 = the phone's setting), `--kv` the K/V
dtype, `--spec-file N` the size of the file the spec prompt asks the model to
repeat, `--spec-minimal-system` keeps the spec prompt under a 512-token sliding
window, `--out rows.json` writes the rows. `TS_SPEC_DRAFT` / `TS_SPEC_PMIN` set
the speculative window and gate; `TS_GMTP_PROFILE=1` prints the Gemma 4 verify's
phase timing. `TS_GMTP_NATIVE_PROFILE=1` separates native graph construction,
binding, allocation, uploads, compute (including the folded head), and downloads;
it adds a diagnostic synchronization and should be disabled for throughput comparisons.
The run fails when any multi-token input took more prefill steps
than `ceil(fresh / chunk) + 2`, or when a grammar-constrained answer the model
finished is not valid JSON.
Plain/speculative stream mismatches remain visible in the notes and require
investigation; they do not fail this batching check. A `PASS` line alone is not
evidence of identical speculative outputs.

Compare runs made with the same model, arguments, and environment:

```sh
python3 benchmarks/AgentTurnBench/compare.py before.json after.json --max-regression-percent 5
```

For steady-state measurements, pass `--warmup 1` to both benchmark runs. This
runs the complete selected workload before measuring, using normal .NET runtime
settings. Warm-up still checks outputs and batching; a failure fails the run.
With `--out`, each warm-up pass is saved separately as
`<output>.warmup1.json`, so startup measurements remain available. The default
`--warmup 0` continues to measure the first workload after kernel initialization.

The comparison requires identical output token IDs, request rows, prompt/cache
counts, and finish reasons. One exception: in a concurrent row with more than one
request whose arrival order was not fixed on both sides (no `ArrivalOrderFixed: true`,
which includes results written before the flag existed), a token difference is
printed as `TOKENS INFORMATIONAL` instead of failing, because different batch
compositions legitimately flip near-ties. Request lengths, finish reasons and the
other fields stay required. Pass `--require-concurrent-identity` to fail on those
differences too, or benchmark both runs with `--conc-gate`, whose rows are always
compared token for token. It reports prefill/decode throughput and TTFT changes,
and fails on output differences, benchmark errors, missing data, or a regression
above the threshold. Decode throughput is omitted when there are no tokens after
the first. For repeated measurements, add `--baseline-repeat before2.json` and
`--candidate-repeat after2.json` (repeat either option as needed); every run must
match outputs, and performance uses the median for each request.

Individual requests also export `TokenTimesMs`: each token's delivery time from
submission. A lower derived decode rate can result from an earlier first token
even when every token arrives earlier. The comparator prints an explicit
decode-rate exception only when every corresponding median token delivery and
median `TotalMs` is no later. Every repeat must contain a complete, valid timeline;
older results and concurrent aggregates retain the strict throughput check. Raw
metric changes remain visible, and prefill/TTFT checks still apply.

For a speculative mismatch, `--spec-diagnostic --spec-new 96 --out diagnostic.json`
replays the same `spec` prompt through the public model and speculative trunk.
It records the actual verify/rollback windows, full-distribution errors, and the
top-two logit margins until the first greedy mismatch. The two mismatching logit
rows are also saved as `.plain_logits.f32` and `.spec_logits.f32` beside the JSON.
This diagnostic disables timing-based draft parking to make proposal windows
repeatable; its timings and outputs do not replace the scheduler benchmark.

The first mismatch alone cannot tell a near-tie from a broken cache, because every
row after it compares different prefixes. `--spec-diagnostic-teacher-force` keeps
the speculative run on the plain token path past a mismatch, so all `--spec-new`
rows stay same-prefix comparisons; the JSON `summary` and the
`SPEC_DIAGNOSTIC_PHASE` / `SPEC_DIAGNOSTIC_FLIP` lines report the logit error by row
class (verify row 0, later verify rows, plain steps, and whether a rollback or a
verified-prefix commit preceded the row) and every argmax flip with both margins.
A bounded error with flips only at small margins is kernel arithmetic; an error
that jumps after a verify or a rollback (tens of logits) is a cache or state bug.
The other switches pick the prompt and drafter:

| switch | effect |
| --- | --- |
| `--spec-diagnostic-prompt <text>` | one user message, no system prompt (the server's `decode` shape) |
| `--spec-diagnostic-json` | the `json` scenario prompt, both runs drawn through the JSON-object grammar |
| `--spec-diagnostic-newchat` | the `newchat` scenario's chat B prompt, on the linear trunk |
| `--spec-diagnostic-speculator auto` | the checkpoint's own drafter instead of n-gram (pass `--draft-model` where it is a separate file) |
| `--spec-diagnostic-window N` | draft window (default 7) |
| `--spec-diagnostic-rowcheck` | after the prompt, the same next-token rows through a one-row spec forward, 2..window+1-row verifies, a kept-prefix re-forward, a decode after a committed verify and a plain two-token forward, each against the one-row decode; then exits. The prompt plus the window must stay under the model's sliding window (the checks rewind the cache, which cannot restore an evicted slot), so pair it with `--spec-diagnostic-prompt`; a longer prompt is refused |

```
dotnet benchmarks/AgentTurnBench/bin/Release/net10.0/AgentTurnBench.dll \
    --model gemma-4-12B-it-qat-UD-Q4_K_XL.gguf --backend ggml_cuda --chunk 1024 \
    --spec-diagnostic --spec-diagnostic-teacher-force --spec-new 192 --out 12b-spec.json
```

For a longer measurement window, use `--warmup 3 --measure-passes 20` with a
focused scenario list such as `--scenarios long,tool`. The model loads once;
each pass repeats the same scenario and cache-reset behavior. Every measured
pass is retained as `<out>.measureN.json`, with process CPU time, allocation
and GC collection deltas in `<out>.series.json`. The default one-pass output
format is unchanged. Summarize within each process before comparing independent
process runs: passes in the same process share runtime and device state and
must not be counted as independent process repeats.

For interleaved process runs, `abba.sh` rotates the arm order between paired
rounds. Supply the baseline twice to measure a baseline-versus-baseline noise
floor, then summarize the median pass in each process:

```bash
benchmarks/AgentTurnBench/abba.sh results/abba 6 \
  "--model /models/model.gguf --backend ggml_cuda --scenarios short,tool,conc --conc 1,4 --measure-passes 3" \
  base=/work/base candidate=/work/candidate control=/work/base
python3 benchmarks/AgentTurnBench/abba_summary.py results/abba \
  --baseline base --candidate candidate --control control
```

Use a fresh output directory and reserve the device for the whole run. The
runner writes `run-plan.json` before starting, records each process exit code in
`runs.txt`, and returns nonzero if any process fails. The summary rejects missing
processes, partial pass sets, failed passes, missing workload rows, and incomplete
runner logs. Older directories without a run plan cannot certify the requested
number of runs. Each arm runs from a private copy of its build output, with the
native library from the specified repository; `name=managed-repo:native-repo`
selects a different native build. `arm-identities.json` records repository paths
and assembly/native SHA-256 hashes. Original build outputs are preserved.
Run `compare.py` separately for token identity and workload shape;
the summary's performance verdict does not establish numerical correctness.

`python3 -m unittest discover -s benchmarks/AgentTurnBench -p 'test_abba.py'`
checks this failure reporting without loading a model.

## What it measured (2026-09-09, Apple M5 Pro, ggml_metal, chunk 1024)

The engine was already batching: a 3,019-token tool result reaches the model as
three forwards of ~1,006 tokens.

| model | long prompt / tool result (fresh tokens in prefill steps) | prefill tok/s | decode tok/s | plain → speculative, identical streams |
| --- | --- | ---: | ---: | --- |
| Gemma 4 E2B Q8_0 | 4,207 in 5; 3,031 in 3 | 3,200-3,600 | 81 | n-gram 81.5 → 80.9 on prose (24% acceptance): break-even |
| Gemma 4 E4B Q8_0 + draft head | 3,019 in 3 | 1,500-1,900 | 46 | draft head 46 → 92 (2.0x); n-gram 63 |
| Qwen 3.5-9B Q8_0 | 4,166 in 5; 3,020 in 3 | 1,000-1,200 | 31 | n-gram 31 → 86 (2.8x) quoting a file; JSON-mode valid both ways |
| gpt-oss-20b Q8_0 | 4,167 in 5; 3,069 in 3 | 1,460-1,760 | 81 | no speculative trunk (harmony model decodes plainly) |

The `tool` turn 2 rows are the agent's every tool round: the previous turn's
tokens come back from the cache (822 / 790 / 768 reused) and only the result is
forwarded, in chunks of the configured size.

The `newchat` scenario's chat B starts from a clone of the shared-prefix
checkpoint - a per-request fused holder, the shape of every new chat in
TensorAgent - and is where speculation has to arm over a holder rather than the
linear cache. With `TensorAgentTtftBench --verbose` the real host logs
`Speculative decoding armed for chat-… (trunk=fused holder over 5,0xx committed
tokens)` on every turn; its repetitive follow-ups ran at 83% acceptance (32
tokens in 0.3 s against 0.5-0.6 s plain).
