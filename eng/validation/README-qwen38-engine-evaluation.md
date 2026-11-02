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
GSM8K subset. Requests default to temperature zero, no thinking and no
repetition/presence/frequency penalties. Do not silently change these settings
or force tool calls for just one engine.

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
  --output-dir docs/validation/RUN/figures
```

Matplotlib is required only for rendering figures. The standalone HTML embeds
optional PNG figures from the context JSON, records its input hashes, and retains
failed/missing configurations. `qwen38-availability-attempts.py` indexes local
controller status, manifests, logs and actual raw-case counts without starting
an engine. EXECUTED means cases exist, not that quality passed.

Record actual coverage and limitations in the generated report: device and
dependency revisions, skipped cases, failed resource gates, precision/configuration
differences, private-workload gaps and instrumentation overhead. Isolated kernel
prototypes and theoretical occupancy are not production or end-to-end results.
