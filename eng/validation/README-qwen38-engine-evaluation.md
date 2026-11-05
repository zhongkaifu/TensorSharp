# Reproducible Qwen3.8 engine evaluation

`eng/qwen38-engine-bench.py` measures OpenAI-compatible servers. It preserves
functional checks, raw failures, sampling settings, actual token counts and
source/binary/model manifests. The public inputs are controls, not a replay of
an unavailable private workload. Generated evidence belongs in ignored
`artifacts/` or `docs/validation/`; required fixtures live under
`InferenceWeb.Tests/Fixtures/Qwen38/`.

## TensorSharp backend and model route

The native evaluation uses `TS_BACKEND=cuda` (the launcher's default), producing
`--backend cuda`. With this dense NVFP4 GGUF, one device and `TS_Q35_NATIVE`
unset, the actual model is `Qwen35CudaModel`. Its startup log must contain
`Model: qwen35 on the direct-CUDA engine`. Check the recorded argv, environment
and startup log together: the backend argument alone does not establish which
Qwen model implementation was selected.

`TS_Q35_NATIVE=0` selects the separate managed `Qwen35Model` implementation on
the CUDA tensor backend. Label its state-residency diagnostics separately.
`--backend ggml_cuda` selects another backend and is not part of this native
performance matrix. The earlier configurations in
`README-qwen38-perf-configurations.md` document a separate ggml CUDA campaign.
Do not mix those runs or reuse their dependency revisions as current evidence.

## Run isolation and provenance

Build engines and prepare models before measured runs. Keep ggml upstream
unchanged. Pin every source revision and artifact SHA256, including drafters,
vision projectors, tokenizers and model configurations. The server launcher
records its exact command and relevant environment, including unset values.
Preserve the launcher manifest and logs alongside each JSONL result.

Only one model server may own the GPU. On a shared Linux benchmark host, hold
one lease around a whole profile:

```sh
flock -x --close /root/ts-eval-gpu.lock \
  bash eng/qwen38-engine-servers.sh bench tensorsharp dflash2 experiment -- \
  --quality-tasks 8 --gsm8k-limit 8 --long-order-quality \
  --keep-text 12000 --allow-failures
```

Configure the paths documented at the top of `qwen38-engine-servers.sh` first.
The example lock path assumes an authorized dedicated VM; use a shared absolute
lock path appropriate to the host. `--close` prevents a long-lived child such as
an MSBuild server from accidentally retaining the lock descriptor. Shut down
task-created compiler/build servers before measuring. Do not overlap CPU builds,
validation tests or model downloads with formal profiles.

The launcher refuses an occupied port and stops only its registered process
identity. Model readiness, compilation/JIT, warmups and loading are outside the
recorded requests. Keep cold-start cost separately when it matters. A byte-for-
byte model staged on tmpfs can reduce loading time; verify its full SHA and
record its actual launch path. It does not establish equal weights across
different model formats.

## Scenarios and quality

The default launcher suite includes factual answers, 4K/16K/32K retrieval,
structured JSON, tool calls, an agentic tool loop, multiple turns, images, long
generation, two/four concurrent requests, generated arithmetic and a pinned
GSM8K subset. The cross-engine scorecard scores retrieval at 8K/64K/128K/256K and
concurrency, KV-cache reuse and multi-turn on the long-context sessions below; it
ignores the 16K/32K retrieval, `prefix_reuse_16k` and the short `parallel_N` scenarios. Requests default to temperature zero, no thinking and no
repetition/presence/frequency penalties. Do not silently change these settings
or force tool calls for just one engine.

Realistic request sizes: the scenarios' own prompts are short (50-1,000 tokens).
`--system-tokens N --context-tokens M` shapes every scenario request (except
`long_prompt_*`, which keep their nominal lengths) like a deployed application's.
- **System message:** static per scenario: a per-invocation nonce, about N tokens of
  shared reference documentation, then the scenario's instructions. It is
  prefix-cacheable across the run's requests, as in production, but never across
  runs.
- **User message:** opens with the request's session nonce and about M tokens of
  retrieved material that no other request shares, followed by the request.

The 2026-11-05 campaign used `--system-tokens 2048 --context-tokens 2048` (about
4.4K prompt tokens per request). Per-request material belongs in the user
message: a hybrid recurrent model can only resume from a recurrent-state
checkpoint, which TensorSharp takes at the end of the leading system messages
and tools. With per-request text inside the system message, no prefix would be
reusable on such a model.

GSM8K contains a small MIT-licensed regression fixture, not an official full
benchmark. Eight question IDs repeated three times are eight independent
questions, not twenty-four. Generated arithmetic checks the expression result,
sorted list, sum and unique count independently in read-only diagnostics.
Functional checks do not prove broad language, reasoning or sampling quality.

Only scenarios completing every expected repeat and passing every original
quality check receive performance values in comparison tables. `--allow-failures`
continues the matrix; it does not turn a failure into a pass. Preserve failed
startup attempts with separate labels, even when a later retry succeeds.
Capacity rejection applies to that exact configuration, not automatically to
an entire engine. No formal cases means NOT_RUN, ERROR or an explicitly
documented capacity rejection, never successful validation.

For actual long input:

```sh
CTX=262144 LANES=1 \
SCENARIOS=long_prompt_128k,long_prompt_250k \
  bash eng/qwen38-engine-servers.sh bench tensorsharp plain long-native -- \
  --long-order-quality --keep-text 12000 --allow-failures
```

Use the same exclusive lease as above. Here 128K means 131072 and 250K means
256000 input tokens. A real tokenizer is mandatory, and reported prompt usage
must satisfy the predefined error tolerance. The five randomized records test
retrieval, document order and immediate neighbors. Do not truncate requests or
change RoPE to manufacture a completion. A smaller configured capacity may
still admit a genuine 128K request; label its capacity accurately. CPU offload
is a separate deployment configuration from GPU-resident inference.

Long-context sessions (`long_session_64k`, `long_session_128k`, `long_session_256k`) measure
what a deployed long-context application sees on one shared document. The document is the
tier minus 4K tokens, fitted with the tokenizer and placed in the system message. Eleven
ledger codes are hidden in it at 4-96% depth. One case runs, in order:
1. chat A, turn 1: the document is new to the server (cold TTFT);
2. chat A, turns 2-3: the conversation continues (multi-turn TTFT);
3. chats B and C: new conversations on the same document (KV-cache reuse TTFT);
4. waves of `--long-waves` (default 2,4) new chats on the document, sent at once. Each
   retrieves a code and writes about 150 words (`--long-wave-tokens`, default 384). The
   wave's throughput is all generated tokens over its wall time; per-request TTFTs are
   recorded too.

Every answer must contain its exact code. The document carries the repeat's nonce, so its
first request never hits another run's cache. Run each tier on a server whose context is
that tier (65,536 / 131,072 / 262,144 tokens), and record whether speculative decoding ran
(the scorecard reads `spec.txt` beside each tier's results).

### Equal lanes and failure reasons

A lane is a concurrent-request slot: TensorSharp `TS_SCHED_MAX_RUNNING_SEQS`, NInfer
`--max-concurrency`, Ignis `--spec-decode-lanes`, llama.cpp `-np`, vLLM `--max-num-seqs`,
SGLang `--max-running-requests` (with `--cuda-graph-max-bs-decode` equal to it). SGLang 0.5.23
also caps running requests at `--max-mamba-cache-size` / 5 on this hybrid model (5 state slots
per request with the radix cache on; its log says so), so the cache size is 5 x lanes: at
3 x lanes one lane served no request and two lanes served one. The lane count
changes memory per request and how a wave queues, so every engine runs every item at the same
lane count: the main suite and the plain-decode run at 1 lane, each long tier at 1, 2 and 4
lanes (`<engine>-<tier>-l<N>` directories). The scorecard takes a long tier's single-request
measurements (TTFT, decode after the document, new-chat reuse, turns 2-3) from its 1-lane run,
the wave of 2 concurrent chats from the 2-lane run and the wave of 4 from the 4-lane run; it
also reports every lane count's own values as `<metric>@l<N>`, which the poster shows row by
row (at 1 lane the waves queue).

The long tiers run with speculative decoding (DFlash2, 7 drafts) only. A configuration that
does not load, or whose KV pool cannot hold the tier's context, is not retried with other
settings: its directory gets `FAILED` and `reason.txt` (the campaign's reason, then the
server-log lines it came from), and the report shows the reason instead of a number. A tier that
did not fit at L lanes is recorded as not run at more lanes, and at longer tiers with as many. A result
that ran but is much slower than its peers carries the reason found in the server log (for
example a request waiting for a slot, or a new chat re-prefilling the document).

llama.cpp needs context checkpoints to reuse a prefix on this hybrid model: its recurrent
(GatedDeltaNet) state must be restored at the end of the shared prefix. With `-ctxcp 0` only an
exact continuation of a slot's own conversation is reusable; a new chat on a cached 64K
document re-prefilled all of it (16.5 s, although the slot matched 99.9% of the prompt). Serve
it with `-ctxcp 32 -cms 2048 --cache-ram 49152 -no-kvu`: a checkpoint every 2,048 prompt tokens
and a host prompt cache. The new chat then took 0.23-0.25 s; follow-up turns cost about 170 ms
more for the checkpoint. llama-server shares no prefix between slots (the prompt cache saves
a slot's state only when the slot is evicted or idle), so a chat that lands on a slot without
the document still prefills it; `-kvu` did not change that and made such waves slower.

`qwen38-quality-diagnostics.py` may identify a correct answer inside one outer
Markdown fence, but never rescales or replaces the original score.
`qwen38-code-quality.py` evaluates only two extracted functions with a bounded,
restricted AST interpreter; it never executes generated modules/imports/tests.
A truncated or otherwise unverified module stays UNVERIFIED even when its
function subset passes.

## Timing and controlled comparisons

TTFT is the client time to the first nonempty semantic output delta, including
tool names/arguments. Role-only metadata is excluded. Tool parsers may buffer
output. TTFT-derived prefill rate includes scheduling and other request work;
it is not GPU kernel throughput.

Client decode rate requires a tokenizer, multiple pure-text payloads and stable
tokenized prefix boundaries. Tokens already in the first payload are excluded.
SSE event counts are never token counts. Tool calls, reasoning and single-payload
responses have no client decode rate. Server-native timings remain separate.

Compare a candidate to the same checkpoint, KV format, drafter, context/lane
capacity, sampling, inputs and harness SHA. Reversing profile order, for example
A/B/B/A using the same binary and one feature switch, helps detect drift.
Three sequential repeats and conditional bootstrap intervals do not prove a
stable advantage. Different GGUF, safetensors and hybrid artifacts are different
deployment stacks; their speed differences cannot isolate engine implementation.

Use `qwen38-paired-performance.py` for same-request descriptive ratios. It
withholds a scenario if either original quality gate fails. For native changes,
also compare complete logits/state against the unchanged path and exercise the
actual device; an unavailable model or skipped CUDA test is not a pass.

## Reports

```sh
python3 eng/qwen38-eval-report.py 'docs/validation/RUN/raw/*.jsonl' \
  --context-json docs/validation/RUN/context.json \
  --output docs/validation/RUN/report.html --min-decode-tokens 128
python3 eng/qwen38-engine-figures.py docs/validation/RUN/raw \
  --context-json docs/validation/RUN/context.json --measurement-group vm1 \
  --output-dir docs/validation/RUN/figures-vm1
```

The sidecar context's `measurement_groups` maps each whole original JSONL
file's SHA256 to an explicit group ID, for example
`{"measurement_groups": {"<raw-input-SHA256>": "vm1"}}`. Use the raw source
hash (`summary.json`'s `sources[].sha256`), not a model or deployment-manifest
hash. Assign groups from recorded host/GPU UUID, CPU quota and memory provenance;
the tools do not infer a host from a profile label or a GPU model name.

Generate each group's figures separately with `--measurement-group vm1` or
`vm2` and distinct output directories. Mixed groups and multiple unmapped input
hashes are rejected before image rendering; `--data-only` and HTML may list
them without pooling latency or confidence intervals. Unmapped inputs remain
`UNSPECIFIED` and are isolated by source SHA, so they cannot overwrite another
unknown source's profile. One unmapped input retains the original CLI behavior.

Matplotlib is required only for rendering figures. The standalone HTML embeds
optional PNG figures from the context JSON, records its input hashes, and retains
failed/missing configurations. `qwen38-availability-attempts.py` indexes local
controller status, manifests, logs and actual raw-case counts without starting
an engine. EXECUTED means cases exist, not that quality passed.

Record actual coverage and limitations in the generated report: device and
dependency revisions, skipped cases, failed resource gates, precision/configuration
differences, private-workload gaps and instrumentation overhead. Isolated kernel
prototypes and theoretical occupancy are not production or end-to-end results.
