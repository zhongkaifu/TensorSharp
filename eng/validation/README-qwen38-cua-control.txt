Public synthetic CUA-like 30-step control (NOT the private screenshot workflow)

qwen38-cua-control.py imports the frozen qwen38-engine-bench/3 Client.body,
Client.finish, stream_chat code, JSON-schema validator and stdlib Canvas image
generator. It does not edit the benchmark, its scoring or any production code.
The original agentic_loop uses generated history and a variable turn cap;
multi_turn has only three turns; image tests independent images. Those existing
cases cannot alone establish the requested fixed 30-step history comparison.

This tool prepares byte-fixed, public prescribed histories. Measured responses
are retained and checked but never fed into later requests. This keeps future
inputs identical across engine configurations despite formatting/tool-ID/failure
differences. It measures a controlled replay, not closed-loop agent completion.

Prepare locally, without network, GPU, Pillow or extra Python packages:

  python3 eng/validation/qwen38-cua-control.py prepare --out artifacts/qwen38-regression/cua-public-v1

Defaults: 30 steps, 1/2-lane plans, 65,536 literal padding characters, a fixed
34-tool table, 192 max output tokens, temperature=0, top_p=1, penalties=0,
seed=1234, thinking=false, tool_choice=auto. The 270 planned requests cover:

  append:   Retain and append every prescribed user/assistant/tool turn, including
            its old image. Earlier messages remain exact prefixes of later ones.
  truncate: Keep four complete earlier turns, dropping only whole oldest turns
            before appending. A tool result is never separated from its call.
  edit:     Retain history, but every five steps replace the POLICY_EDIT marker
            in the first user turn. The current call must return that new revision,
            detecting stale historical state rather than only parsing valid JSON.

Every current screenshot is generated with the frozen harness's explicit bitmap
glyphs: a four-digit number in a white box and one red/green/blue rectangle.
Resolution is 1344x768, matching the frozen screenshot's dimensions. The same lane/step image bytes are used by every scenario,
concurrency case and engine. Different lanes have distinct codes/scopes. Each
case/lane has a distinct fixed system marker before its immutable padding; the
tool does not set engine-specific cache-scope overrides or change isolation.
The bitmap is public synthetic content, not the private screenshot/template/history.
Literal prefix characters are not claimed as tokens. Actual usage includes the
engine's own template/tool and image expansion; no character-based prefill rate.
Append/edit can retain 30 screenshots. Use common CTX65536 server capacity for
every engine, even if a smaller scenario would fit less. The launch manifest's
context_tokens and lanes fields are checked against this declared capacity and
client concurrency. The tool never truncates a request to fit an engine.
Optional prepare --tokenizer PATH records a separate sum of text/tool encodings;
it excludes image expansion and template wrappers and is NOT actual prompt usage.

Prescribed replies can differ from the model's actual emitted prose even when
tool arguments agree. TensorSharp's ConversationTranscriptStore lineage proof
also checks prose, so default-lineage may reject ownership for these histories.
This is NOT representative of normal generated-history lineage. Record separate
deployments explicitly: --cache-profile default-lineage is a diagnostic group;
--cache-profile trusted-single-tenant-shared is only for a dedicated trusted
single-tenant benchmark using the existing shared deployment setting. External
engines declared --cache-profile engine-native-global belong with the shared
comparison group. No script changes the production default or launches/configures
the server. TS manifests must explicitly record TS_PREFIX_CACHE_ISOLATION (null
when unset); the selected profile must match lineage/default or shared exactly.
Actual-generated-history testing is a different experiment: cross-engine inputs
would diverge and it is not silently mixed into this fixed replay.

Keep the prepared directory unchanged and use it for every compared launch.
Its manifest pins fixture/client/generator SHA plus every image. An intentional
tool or frozen-client change requires preparing a fresh fixture. Use fresh server
processes per replay, or prepare one new shared seed for all matched repetitions:
the client cannot reset an engine's retained prefix cache or establish coldness.

During a later authorized GPU/network window, point at an ALREADY RUNNING server:

  python3 eng/validation/qwen38-cua-control.py run --fixtures artifacts/qwen38-regression/cua-public-v1 --engine tensorsharp --model qwen3.8-27b --url http://127.0.0.1:8000 --manifest artifacts/launch-main.json --label ts-main --mode plain --cache-profile default-lineage --context-capacity 65536 --lanes 1 --out artifacts/qwen38-regression/cua-ts-main-c1.jsonl

Repeat with --lanes 2 and a new output path. Run the same files against TS
candidate, Velox (--engine velox) and NInfer (--engine ninfer); model routing IDs
may differ. The launch manifest must record actual revisions, binaries, model /
vision companion, target/draft quantization, KV precision, parser and thinking-off
switches, engine context and concurrency. Unsupported vision/tools are FAIL;
the tool never substitutes text-only inputs or forces a tool for one engine.
NInfer/Velox launch thinking flags remain necessary where request options alone
do not configure their templates. No implicit discovery, warmup or server launch.

Each lane advances sequentially. A client barrier starts 1/2 requests together
for each step, and the next wave waits for both results. There is no client think
time. This is lockstep concurrency, not independent closed-loop arrival traffic.
Per-step raw TTFT/E2E/usage and actual output sizes remain in JSONL even on FAIL.
TTFT means first semantic content/reasoning/tool-name/argument delta; role/ID
metadata does not start it. Parser buffering remains an engine-specific limit.
Tool streams have no reliable raw decode rate, so no decode token/s is reported.

Each request retains the full actual body; untrimmed normalized content, reasoning
and calls; raw server usage; raw frozen observation; consumed SSE/HTTP-error bytes
as base64+SHA; expected arguments; typed checks; history edit/drop metadata; body
hashes. SSE capture stops where the frozen client stops at [DONE]/EOF; unread
trailing HTTP bytes are not claimed and no extra read alters client timing.

Identity fields have deliberately separate meanings:
  full_body_sha256:       Canonical JSON of every real body field, including model.
  canonical_input_sha256: Same JSON removing ONLY model routing ID. All generation,
                         messages, tools, tool IDs and engine-adapter fields remain.
  wire_request_body_sha256: Exact serialized bytes sent by frozen stream_chat.
  common_sha256:          Prepared portable fixture fields before engine adapters.
  frozen_normalized_request_sha256: Original helper, which also normalizes old tool
                         IDs; retained for diagnosis, never called byte identity.

TS and other engines use different documented request switches (think=false vs
chat_template_kwargs); those real differences remain visible in the canonical
input hash/body. A matching common hash alone does not mean actual requests were
byte-identical. Compare image/messages/tools/generation contract plus explicit
adapters; never claim broader identity than the recorded fields establish.

Quality requires HTTP success, completed stream, one choice, no reasoning,
exactly one correctly named tool, strictly typed complete arguments matching code,
color, revised policy, lane and step, and no freeform text. Correct calls ending
at the output limit fail. Missing actual server prompt/completion usage, bool /
invalid counts, missing/invalid TTFT or E2E leave measurements UNVERIFIED. No
tokenizer/event-count fallback qualifies performance. A case's latency aggregates
are withheld unless EVERY planned step/lane passes quality and measurement.
Case output includes actual completion totals alongside wall/TTFT/E2E, without
ranking, invented zero speeds, CI for correlated steps, or original-score changes.

CPU-only test command (uses in-memory fake SSE responses, never sockets):

  python3 -m unittest discover -s eng/validation/tests -p test_qwen38_cua_control.py -v

Optional independent old-media correctness probe (not in the default matrix):

  python3 eng/validation/qwen38-cua-control.py prepare --media-probe --steps 6 --out artifacts/qwen38-regression/cua-media-probe-v1

This prepares ONLY edit_media, at 6..10 steps (default10). At step5 it replaces
the image in the first retained user turn, leaves that early user text unchanged,
and requires a separate anchor_code OCR argument from the edited old image in
addition to the current image's code/color. A stale anchor fails even when the
current screenshot is correct. Old/new image SHAs and the unchanged-text edit
metadata are explicit. The three default scenarios and their budgets/tools are
unchanged. Replay with --fixtures pointing at this probe and normal deployment /
cache-profile flags; scenario defaults select edit_media from its own manifest.
Do not present the text-only edit scenario as evidence of old-media coverage.

Prepared means NOT_RUN. Model/vision/tool quality, retained-cache correctness,
performance and actual usage remain unverified until a real replay completes.
