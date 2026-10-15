# Prefix and startup warmup HTTP validation

`prefix-warmup-http.py` sends bounded requests to an **already running Host**.
It never starts/kills the server, changes its prefix settings, or touches an
existing user session. Every new session is created by this invocation and
deleted at cleanup. Reusable transport and structured generation-timing
parsers are imported from the existing validation tools.

Three sequential new chats send exactly:

1. `请详细介绍最终幻想7`
2. `请详细介绍时间简史`
3. `请详细介绍最终幻想7`

All use greedy sampling and a fixed cap of8 or16 output tokens. Tools, skills,
discovery, multi-agent and tool-choice fields are **omitted**, retaining the
Host's real default catalog. The tool records full SSE bytes/frames, literal
visible frames including replacements, thinking, raw/content TTFT, terminal
usage and truncation. SSE chunk counts are never treated as tokenizer counts.

The first and third chats must have exactly matching visible frames/text,
prompt/completion usage, HTTP finish state and measured native finish reasons.
Thinking-only bounded output is explicitly inconclusive for visible parity;
two empty answers never establish passing visible output. This is not full
answer semantic quality, raw token-ID/logit identity or GPU-parallel acceptance.

`--server-log` must be the live Host's structured JSONL log. The existing timing
parser binds the generated HTTP request ID and session ID to actual model
completions. Prompt/cached/completion usage must agree with those measurements;
prefill/decode attribution remains separate from HTTP wall time. Tool rounds
remain recorded individually. A missing/mismatched log does not pass.

`--cache-expectation enabled` requires actual cached tokens in the repeated
new chat. `disabled` requires zero cached tokens in all three requests.
`observe` leaves the cache gate null: it can qualify transport/parity evidence,
but never counts cache reuse as passing. Think-on/off and prefix-disabled
controls must be separate Host launches with separately pinned provenance.
Positive cached usage alone does not identify a particular payload or prove
that startup warmup is optimal. Compare TTFT with a separately qualified
baseline using the same real prompt and resource configuration.

```powershell
python -X utf8 -B eng/validation/prefix-warmup-http.py `
  --base-url http://127.0.0.1:5000 `
  --server-log C:/Works/TensorSharp/artifacts/owned-host/server.jsonl `
  --provenance C:/Works/TensorSharp/artifacts/owned-host/launch-audit.json `
  --output C:/Works/TensorSharp/artifacts/prefix-warmup/candidate-think-off-v1 `
  --think off --max-tokens 16 --cache-expectation enabled
```

The root launcher owns server startup, model/build/upstream identity, memory
sampling and server cleanup. The supplied provenance is pinned before/after,
but this HTTP tool does not itself verify its physical-budget or binary claims.
Output must be a fresh ignored `artifacts/` or `docs/validation/` directory.

Optional `--cancellation` disconnects only its own partial128-token stream,
requires an actual server cancellation event and idle queue, then completes a
fresh recovery chat. A client disconnect alone is insufficient. This control
has no speedup or native-slot identity claim.

Optional `--follow-up --history-route /route/{sessionId}` requires a real
same-host production-history provider returning
`{sessionId, source:"production-wire", messages:[...]}`. The exact returned
assistant content/thinking must match the retained stream. Its message objects
are forwarded unchanged before appending `继续介绍。`; thinking/tool history is
never reconstructed or repaired. Actual cached tokens must exceed the prior
whole prompt to establish reuse beyond a public prefix.

The current standard session routes expose creation/deletion only. Without
an actual history provider, a requested followup reports inconclusive failure
and sends **no reconstructed transcript**. It does not pass private KV reuse.
Model reload, media invalidation and longer concurrent workloads remain separate
unexercised scenarios; omitted optional controls are explicitly not requested.

Separate `--client-visible-follow-up` uses the standard WebUI client contract
without a history GET route. It replays the actual emitted final content
(including whole-content replacements), literal thinking and original user
message as `role/content/thinking` DTOs. Production `ParseWebUi` consumes those
fields, and `ConversationTranscriptStore.Augment` can restore its recorded raw
IDs internally. This tool never manufactures or sends raw IDs. Tool frames,
unsupported metadata, incomplete/empty outputs and multiple measured engine
rounds are refused; the current WebUI parser does not parse tool-call history.

The most recent third FF7 chat is followed up first and must observe actual
cached tokens above **its own** prior whole prompt. The older first FF7 chat
then supplies a literal output/finish reference. Its cached/recomputed outcome
is recorded separately: limited conversation retention can legitimately evict
it, so public-only reuse or recomputation does not fail the recent-reuse gate.
Both followups still require matching bounded visible frames/text, finish and
usage linked to actual native timings. This establishes beyond-public reuse
only; it does not establish raw token/state-ID identity. This mode and strict
production-history `--follow-up` are mutually exclusive.

Controller-only tests use a loopback synthetic server, not TensorSharp models:

```powershell
python -X utf8 -B -m unittest discover -s eng/validation/tests -p test_prefix_warmup_http.py
```
The explicit `--think on --output-gate stream` arm checks literal emitted
thinking and content frames, finish and usage for the bounded repeated prompt.
Its report preserves the separate strict `repeat_parity` visible-quality result:
thinking-only output remains `inconclusive_no_visible_output`, even when bounded
stream observation passes. Raw/thinking TTFT and first visible-content TTFT are
separate; the latter remains null if no content arrived. Stream mode does not
qualify production-history continuation. The default `--output-gate visible`
keeps the existing nonempty visible-output requirement.
`--follow-up-max-tokens 64` or `128` gives the explicitly requested continuation
its own bounded generation cap. Initial new-chat requests keep their separate
`--max-tokens 8|16` workload. Visible continuation parity remains strict; a
thinking-only continuation is still inconclusive rather than a quality pass.
The option refuses invocation without `--follow-up` or
`--client-visible-follow-up`.

The optional `--long-new-chat-max-tokens 256` probe requires explicit ON stream
mode and `--long-new-chat-expected-cached-tokens` with an exact positive public
boundary. It adds one independent FF7 session after the original three8/16-token
requests, fully drains the response, and binds usage to actual measured server
completion. Its exact cached-token check does not borrow private conversation
credit. Followup and cancellation cannot be combined with this separate probe.
The report distinguishes first raw, first thinking and first visible-content
TTFT. If256 tokens produce only thinking, visible observation stays inconclusive;
there is no automatic escalation, semantic-quality or repeat-parity claim.

Structured logs follow normal UTC-day and numeric size rollover of the explicit
canonical `prefix-YYYYMMDD[-NNNN].jsonl` family. Within the bounded run, capture
requires contiguous numeric indices, at most32 files and128MiB across at most two
adjacent UTC dates. The report pins every original file and an exact ordered
combined snapshot; malformed/gapped/outside-family data fails qualification.
Generic synthetic single-file logs preserve their existing identity. Timing and
usage must still match each actual request/session; file rotation alone never
upgrades an original failed run or absent visible-output quality.
