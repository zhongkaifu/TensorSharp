# AgentTurnBench

Does a multi-token input reach the model as **one batched forward per chunk**, or
as one forward per token? This benchmark drives the continuous-batching engine
(`InferenceEngine` → `ContinuousBatchScheduler` → `BatchExecutor`) — the path the
server and TensorAgent use — with the conversation shapes an agent actually
produces, and reports per request: prompt tokens, tokens served from the cache,
engine steps, tokens per prefill step, TTFT, prefill and decode rates, and under
speculation the drafted / accepted / verify / plain counters. Every speculative
stream is compared against plain greedy token for token.

| scenario | what it sends |
| --- | --- |
| `short` | a tiny prompt with a one-line system prompt |
| `long` | an agent system prompt plus a long pasted source file (`--long`, default 4096 tokens) |
| `tool` | a turn, the same conversation extended by a `--tool`-token tool result (default 3000), then a short follow-up — rendered through the chat template with the model's raw output tokens spliced in, as the server does |
| `newchat` | two conversations sharing the system prompt (the shared-prefix checkpoint) |
| `spec` | plain greedy vs n-gram vs the checkpoint's own drafter, then the tool rounds under n-gram |
| `json` | grammar-constrained JSON, plain vs n-gram |
| `conc` | N concurrent requests on one engine, then a solo request after them |

`--spec-engine ngram|auto` enables speculation on ordinary scenario engines.
The explicitly labeled plain controls in `spec`, `json`, `newchat` and `image`
always disable it, so they remain independent comparison controls. The `conc` rows measure
concurrency WITH speculation: the planner keeps a multi-sequence step plain and
the interesting cost is the transitions. `--conc-stagger <ms>` spaces the
submissions of a concurrent round so later requests arrive while earlier ones
already decode (and speculate); `--conc 1,2,4` includes a solo round as the
reference. Concurrent rows aggregate the actual per-request speculative counters;
zero counters do not establish speculative engagement.

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
counts, and finish reasons. It reports prefill/decode throughput and TTFT changes,
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
| `--spec-diagnostic-rowcheck` | after the prompt, the same next-token rows through a one-row spec forward, 2..window+1-row verifies, a kept-prefix re-forward, a decode after a committed verify and a plain two-token forward, each against the one-row decode; then exits |

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

## What it measured (2026-09-09, Apple M5 Pro, ggml_metal, chunk 1024)

The engine was already batching: a 3,019-token tool result reaches the model as
three forwards of ~1,006 tokens.

| model | long prompt / tool result (fresh tokens in prefill steps) | prefill tok/s | decode tok/s | plain → speculative, identical streams |
| --- | --- | ---: | ---: | --- |
| Gemma 4 E2B Q8_0 | 4,207 in 5; 3,031 in 3 | 3,200-3,600 | 81 | n-gram 81.5 → 80.9 on prose (24% acceptance): break-even |
| Gemma 4 E4B Q8_0 + draft head | 3,019 in 3 | 1,500-1,900 | 46 | draft head 46 → 92 (2.0x); n-gram 63 |
| Qwen 3.5-9B Q8_0 | 4,166 in 5; 3,020 in 3 | 1,000-1,200 | 31 | n-gram 31 → 86 (2.8x) quoting a file; JSON-mode valid both ways |
| gpt-oss-20b Q8_0 | 4,167 in 5; 3,069 in 3 | 1,460-1,760 | 81 | no speculative trunk (harmony model decodes plainly) |
| Bonsai-8B Q1_0 (qwen3) | 4,152 in 5; 3,029 in 3 | 1,100-1,400 | 105-128 | no speculative trunk; JSON-mode valid |

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
