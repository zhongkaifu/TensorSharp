# Multiple agents

TensorSharp can let a model delegate independent parts of a request to bounded
subagents, collect their results, and synthesize one answer. Delegation runs in
`TensorSharp.AgentHost/Agents/`, alongside the existing
[skill and code-execution loop](agent_skills.md), and uses the host's loaded
model and generation backend.

Automatic delegation is enabled by default on supported server chat paths.
The model decides whether to delegate and which task to assign. Enabling the
feature does not force a fixed number of agents or guarantee a faster or more
accurate answer. No latency, answer-quality or delegation-rate results are
published for it; the tools under [Validation and performance](#validation-and-performance)
write their reports to ignored `artifacts/`.

Delegation does not depend on skills or code execution. With `--no-skills` and
without `--code-exec`, every eligible request still carries the five
coordination tools and the coordination prompt, merged into the leading
system/developer message (or added as a new leading system message when the
request has none). Children in such a request then have no host tools
besides those five.

## Choosing work and models

The coordination prompt asks the model to analyze the user's request, identify
concrete subtasks and their dependencies, and delegate useful bounded work.
Each assignment includes the necessary context, expected output, role, and file
access. Independent tasks run concurrently within the host's capacity limit;
a task with prerequisites waits for their successful completion. Suitable examples
include analyzing separate documents, implementing distinct components, and
reviewing completed work. Short requests and tightly coupled work may stay
local when delegation would only add overhead.

Decomposition and role selection are model decisions, rather than a fixed
keyword-based splitter. The host enforces the resulting task graph, permissions,
and lifecycle; it cannot guarantee that a model has correctly understood an
assignment. The parent continues useful independent work, avoids repeating
assignments, verifies important claims, and integrates the results.

Each child uses the same model as the parent. `agent_type` chooses instructions
and tool access, not another set of model weights:

| Role | Intended work | Tool access |
|---|---|---|
| `explorer` | Focused investigation; default role | Advertised skill listing and reading; `read_file` when the parent offers it |
| `reviewer` | Independent checks and evidence | Advertised skill listing and reading; `read_file` when the parent offers it |
| `worker` | Bounded implementation work | Mutable tools in its private workspace when host and parent permit them; explicit `read-only` narrows access |

With the built-in OS runners, workers can use scoped file reads, writes, and
patches but cannot run child shell commands or skill scripts: those sandboxes do
not yet advertise the stronger workspace read-isolation guarantee required for
child execution. The parent runs builds and tests after reviewing and integrating
worker outputs. A custom backend can opt into child execution only when it
enforces both read and write boundaries.

There is no automatic selection among local or remote models in this version.
Sharing the loaded model avoids loading a separate copy of its weights for each
child. Each child has its own writable conversation state and can restore
compatible public-prefix checkpoints from the parent or another child.

## Context, tools, and lifecycle

Each child starts with governing system/developer instructions and a
self-contained task supplied by the parent. It does not receive the parent's
conversation transcript or attachments. Task text
must therefore include the facts, scope, ownership boundaries, and expected
evidence needed for the assignment. `input_files` selects explicit files to copy
from the parent's workspace; dependency outputs are staged separately. Follow-up
turns reuse the child's own conversation and workspace. The parent receives
bounded result reports and exported output paths rather than the child's full
tool transcript.

Children with the same role, governing instructions, and offered tools share a
stable system/tool prefix. Their unique agent IDs and assignments follow it in
the first task message; each child keeps its own conversation and cache scope.
When an identical public prefix is still being computed, the radix scheduler
defers a cold sibling until the producer can publish its checkpoint. Other
requests and active decoders can continue running. If capture fails or the
producer stops, the sibling can prefill normally.

For Qwen 3.5-family recurrent models, reuse requires a checkpoint at the exact
shared boundary, including both attention KV and recurrent state. Before parent
prefill, the host renders the possible child system/tool profiles and identifies
their common leading tokens. It captures an earlier public checkpoint there,
as well as the parent's complete public prefix. The first child can reuse this
common parent checkpoint and compute its different suffix. A sibling can then
reuse the child's longer checkpoint, even when the shorter ancestor was already
available at admission. No private parent transcript is included.

Tool declarations shared with read-only children are ordered first, followed by
parent-only or mutable tools. The model's normal template and each role's tool
permissions still apply. Different tools or instructions limit the identical
leading portion; matching text after a difference does not make the corresponding
KV state reusable.
Checkpoints remain subject to count and memory limits. Under the default count
budget of two, a shorter common checkpoint may be evicted once both branches
have longer public checkpoints. An older saved parent checkpoint without the
earlier state cannot be rewound; the first missing branch prefills safely and
becomes reusable afterward. Explicit cache opt-outs remain effective.

Each active child receives an independent mutable state copy: this saves
repeated prefill, but does not share physical KV pages between children or
guarantee lower peak VRAM. The existing prefix-cache controls also apply to
children. The approach follows the hybrid-cache distinction used in SGLang:
a matching radix path must also have an actual recurrent-state snapshot at its
branch boundary. TensorSharp implements its own checkpoint planning and
retention; it does not depend on SGLang or modify ggml.

The [parent-prefix validation probe](../eng/validation/Qwen35ParentPrefixProbe/README.md)
compares parent-plus-two-child workflows with and without the earlier checkpoint,
including its capture cost and exact generated-token comparisons. Both arms keep
ordinary radix caching and the default public checkpoint count of two enabled.

Agents belong to a request-scoped tree with parent identity and depth. Limits
apply across that tree, including children created by other children. Agent IDs
are paths below `/root`, the parent: a child named `review_api` is
`/root/review_api`, and its own child adds another segment. The
runtime manages their background tasks, status, cancellation, and completion;
the host supplies generation through its existing inference path. On the
server, each child uses the parent's model, token limit, thinking setting, and
sampling.

The model sees these native tools:

| Tool | Behavior |
|---|---|
| `spawn_agent(task_name, task, agent_type?, permissions?, input_files?, depends_on?)` | Registers a child and immediately returns its ID and state. The task name must be 1–48 letters, digits, underscores, or hyphens, unique under its parent. `agent_type` defaults to `explorer`. Jobs queue when capacity is full. |
| `wait_agent(agent_id?, timeout_ms?)` | Waits for the named direct child, or all direct children when the ID is omitted. Default timeout is 10,000 ms; maximum is 60,000 ms. Returns child status, reports, effective permissions, `workspace_id`, `depends_on`, and exported `files` (`path`, `bytes`), plus `timed_out`. A timeout does not mean the child completed or was cancelled. |
| `send_input(agent_id, message)` | Queues a message for a running child at its next generation boundary (at most four queued messages), or starts a follow-up turn on a completed child. |
| `list_agents()` | Reports direct children and their state. Waiting uses `wait_agent`, rather than repeated listing. |
| `close_agent(agent_id)` | Cancels a child and its descendants. Cancellation is not successful completion. |

The optional spawn fields use flat strings for local-model tool compatibility:

| Field | Meaning |
|---|---|
| `permissions` | Omit to use the role's host-permitted access. `read-only` narrows it; explicit `workspace-write` requires the `worker` role, host opt-in, and a parent that has mutable access, otherwise spawning fails. Writes stay inside the child's private workspace. |
| `input_files` | Newline-separated relative paths from the parent's workspace or named authorized attachments. Only selected bounded file snapshots are copied; directories, absolute paths, traversal, and symlink escapes are rejected. |
| `depends_on` | Comma- or newline-separated IDs of existing direct children of the same parent. The new task waits for every listed prerequisite to complete successfully. |

Dependencies must refer to already-created siblings, so self references, forward
references, cross-tree references, and cycles cannot enter the graph. A failed,
cancelled, blocked, or budget-exhausted prerequisite blocks dependent work before
its generation starts. Prerequisite reports are supplied as evidence, and their
exported files are staged under `dependencies/<task_name>/` in the dependent
workspace. A dependent task should still explain how to use and verify those
inputs. Create all independent tasks first, then dependent tasks, and collect the
results after other useful parent work.

Prerequisites initialize a child's first execution: their completed reports and
outputs are captured once. A follow-up reuses that child's existing history and
workspace; it does not rerun prerequisites or import newer outputs. Create a new
child to consume refreshed prerequisite results. While a dependent task remains
unfinished, `send_input` rejects attempts to change its prerequisite's assignment.

For example, these consecutive spawn calls allow two investigations to overlap
and schedule a review after both finish:

```json
[
  {"task_name":"api","task":"Inspect api.md and report compatibility risks with evidence.","agent_type":"explorer","permissions":"read-only","input_files":"api.md"},
  {"task_name":"storage","task":"Inspect storage.md and report migration risks with evidence.","agent_type":"explorer","permissions":"read-only","input_files":"storage.md"},
  {"task_name":"review","task":"Check both prerequisite reports for conflicting assumptions and prioritize the supported risks.","agent_type":"reviewer","depends_on":"/root/api,/root/storage"}
]
```

Children inherit only permitted host-owned tools. Client-owned function tools
are not delegated: their implementations belong to the caller, and an internal
child cannot ask that caller to service them. Read-only roles cannot execute
shell commands, run skill scripts, or modify files. They can analyze evidence
included in their task and use `read_file` when offered, within their private
workspace and the host's authorized skill resources. Their host allowlist is exactly
`skills_list`, `skills_read` and `read_file`, plus the five coordination tools.
Enabling worker tools does
not enable an execution surface that the operator has otherwise disabled.
Permissions only decrease down the tree: a `worker` receives mutable tools only
when its parent is the root or is itself a mutable worker, so a read-only child
cannot spawn a worker to regain them.

Each child has its own workspace, forked runner/tool context, and tool gate.
Host calls remain serialized within one agent's mutable context, while separate
agents can execute independent tools concurrently. Generation also overlaps
when the inference backend permits it. A client tool with the same name as a
coordination tool takes precedence over the built-in one.

Private workspaces use bounded snapshots of selected inputs rather than a full
repository checkout or Git worktree. Creating a workspace does not authorize
new tools, network access, or access outside the parent's existing sandbox.
Input and changed-output transfers are limited to 128 files and 64 MiB per
transfer. Output inspection also stops at 4,096 workspace files. Exceeding a
limit fails the operation rather than silently truncating it.
The model's role label and task text cannot grant permissions: the executor
checks effective tool access and confines mutable tools to the child workspace.
No child can regain permissions its parent lacks.

Changed output files are handed back under a unique parent-relative directory,
`agent-results-<agent_id>-<unique_id>/`. These outputs do not overwrite the parent's source
files. The parent reviews the exported paths, resolves conflicts, and applies
accepted changes to its own workspace. Deletions are not automatically applied;
workers must report intended deletions for the parent to review. Assign disjoint
final file ownership even when workers execute in separate directories. Agent IDs do not grant access to
another request's agents or workspace, and an agent can address only its own
direct children.

The concurrency limit controls runnable children across the entire tree.
Dependency waits and capacity queues do not consume execution slots. A child
waiting for its own descendants temporarily yields its slot and reacquires
capacity before continuing, so nested delegation also progresses with a limit
of one. A wait's timeout bounds the wait for results; a child may then queue for
a resume slot before the tool returns and generation continues. The root does
not need a resume slot. The parent must not treat an accepted spawn as proof that
execution has already begun.

Failures, timeouts, cancellation, and exhausted budgets are reported distinctly
from completed work. Terminal states are `completed`, `failed`, `cancelled`,
`limit_reached`, and `blocked`. In-flight work can be `queued`, `waiting`, or
`running`; cancellation can show `cancelling` while a stopped child winds down.
`blocked` means a required predecessor did not complete successfully; its report
identifies the prerequisite rather than claiming the dependent task ran.
A child generation that stops on a token, thinking-budget or repetition limit
is `limit_reached`, and so is a child that runs out of rounds or of the shared
generation budget. A child whose time limit expires is `cancelled`. Reports longer than `--agents-max-result-chars` end with
`[Report truncated by host result limit]`.

Required child results must be collected before the parent claims completion,
and the host enforces this. If the parent produces a final answer while a
direct child's report is still unread, the host holds that answer back, waits
for the reports, and adds them to the conversation. It then asks the parent to
integrate them and generates the answer again. It does the same when the
parent's round limit is reached. Only the parent's text is streamed to the
client. The final usage totals include the children's prompt, generated and
cached tokens; total time is the request's elapsed wall time, since child
generations overlap. Request cancellation also stops the request's descendants.
Agent state is not a durable cross-request session API.

In the Web UI, click the arrow or `wait_agent` row to expand or collapse its
details. The row also supports keyboard focus and activation.
While `wait_agent` is running, the expanded panel shows each subagent's task,
status, and available tool activity or result. Details update as the agents work
without changing whether the panel is expanded. The activity panel is temporary
and is removed when the current operation or response finishes.

The panel is built from the Web UI stream on `POST /api/chat`. While
`wait_agent` runs, whether the model called it or the host is collecting reports
before the final answer, its `tool_progress` frames carry an `agents` array. Each
entry has `agent_id`, `parent_id`, `task`, `agent_type`, `status`, `tool`,
`tool_status`, `detail`, `result`, `error`, `workspace_id`, `permissions`, and
`depends_on`. `skill_step` frames also carry
`agent_id`. The OpenAI and Ollama streams do not include these snapshots.
TensorAgent's own chat page does not render an agent panel, even while
delegation is on there; it only labels the sub-agent tool steps in words
("Starting sub-agent", "Waiting for sub-agents", "Messaging sub-agent",
"Stopping sub-agent", "Checking sub-agents").

## Host controls

The server startup flags below configure the entire request tree. The same
options are available through `ServerHostingOptions.MultiAgent` and
`MultiAgentOptions` in C#. `ServerHostingOptions.MultiAgent` is read again for
every request, and an embedding host can switch delegation on or off on a running
host with `ServerHostingOptions.RepointMultiAgent(bool enabled)`, which flips
`MultiAgentOptions.Enabled` and keeps every limit (TensorAgent's Sub-agents switch
uses it; the desktop server never calls it). Existing server JSON configuration expands to these
flags. A request can set the top-level boolean `"multi_agent": false` to use a
single agent. `true` or an omitted field follows the host policy and cannot
enable delegation when the host has disabled it. Requests cannot raise host
limits or enable worker tools.

| Flag | Default | Allowed range or meaning |
|---|---|---|
| `--no-multi-agent` | Absent | Disables coordination tools and automatic delegation. `TS_NO_MULTI_AGENT` set to a nonempty value other than `0` also disables it. |
| `--agents-max-concurrent` | `3` | `1`–`32` executing descendants; excludes the root and children waiting for dependencies or child results |
| `--agents-max-count` | `8` | `1`–`128` total children across the tree |
| `--agents-max-depth` | `2` | `1`–`8` child levels below the root |
| `--agents-max-rounds` | `8` | `1`–`64` tool-loop rounds per child turn; a capped loop permits one final answer generation |
| `--agents-max-generations` | `48` | `1`–`1024` child generations across the request |
| `--agents-timeout` | `180` | `1`–`3600` seconds per child run, reset for a follow-up turn |
| `--agents-max-result-chars` | `8000` | `256`–`64000` characters per result report |
| `--agents-allow-worker-tools` | Absent | Allows `worker` agents to use enabled mutable host tools in private workspaces; `permissions: "read-only"` still narrows access |

`MultiAgentOptions.MaxTaskCharacters` bounds task and message text, defaults to
16,000, and permits values from 256 through 64,000. It is a C# option, without a
dedicated startup flag. These limits bound orchestration; existing context,
generation, skill, and code-execution limits still apply. Timeouts signal
cancellation; generation callbacks must honor their cancellation token, and
host tool execution retains the existing runner's time limits. The child timeout
starts when its turn is registered, so time in the capacity queue or waiting for
prerequisites counts toward it.

The integrated server paths are OpenAI-compatible chat completions and
Responses (`/v1/chat/completions`, `/v1/responses`), Ollama chat
(`/api/chat/ollama`), and the Web UI/TensorAgent chat path (`/api/chat`).
Ollama `/api/generate` is not integrated. TensorAgent keeps the default limits
and has one switch, Settings > Sandbox > "Sub-agents" (`multiAgentEnabled` in its
settings, on by default): off stops the five coordination tools and the
coordination prompt from being declared, exactly like `--no-multi-agent`, from the
next message and without a restart (a turn already delegating finishes under its
old terms). It does not read `TS_NO_MULTI_AGENT`, which only the server's startup
options read. Structured-output
requests, `/v1/chat/completions` requests with `"tool_choice": "none"`, and model
families that cannot both render tool declarations and parse tool calls do not
offer coordination tools. That
excludes Mistral 3, Hunyuan Dense, DiffusionGemma, and any architecture without a
tool-call parser. There is no model-size gate: every family that can call tools
receives the same tools and coordination prompt.

Direct C# callers enable `SkillAgentLoopOptions.MultiAgent` and provide
`SubagentGeneratorFactory`. That factory must allocate independent generation
state for each child ID; passing callbacks that capture the root's mutable
session or KV state is not safe. The server integration supplies a separate
`ChatSession` and generation context per child.

Custom `ICodeRunner` implementations must opt into workspace delegation through
`DeclareWorkspaceTools(bool allowWrite)` and `ForkForWorkspace(...)`, enforcing
the declared scope during execution. The default interface implementation offers
no child tools and no runner. Custom skill script runners likewise opt into
`CanForkForWorkspace` and `ForkForWorkspace(...)`. Child shell and script
execution also requires a backend that explicitly
advertises workspace read isolation as well as write confinement. Existing
built-in OS sandbox profiles do not make that stronger guarantee, so their child
runners expose permitted file tools but withhold shell and script execution.
Worker opt-in does not bypass this restriction; a host providing a backend with
those guarantees must also keep its runner declarations and execution consistent.

`SkillsChatClient` local delivery supplies independent HTTP conversations
automatically. `SkillsChatClientOptions.MultiAgent` is enabled by default;
configure it to set local limits, or set `SkillsChatRequest.MultiAgent = false`
for a single-agent request (server delivery forwards that as `"multi_agent": false`).
Against a detected TensorSharp server, local delivery suppresses server-side
orchestration so only one host owns the tools. Token usage includes the children;
`SkillToolInvocation.AgentId` identifies each callback's owner. Callbacks may run
concurrently and must be thread-safe.
The returned `SkillsChatResponse.Messages` can be reused for the next turn;
it preserves the caller's preamble without accumulating injected host policies.

For example, the server can decide how to divide a substantial document review:

```json
{
  "model": "your-loaded-model",
  "messages": [{
    "role": "user",
    "content": "Review the supplied service specifications for migration risks. Investigate independent components as useful, verify conflicting findings, and produce one prioritized report with evidence."
  }],
  "multi_agent": true,
  "stream": true
}
```

Use `/v1/chat/completions` and supply the actual documents or authorized skills
alongside that request. Delegation does not itself expose any new documents.
The standalone CLI's direct decode loop is not integrated with this coordinator;
use the server or the C# agent host for subagents. No mobile-device performance
claim follows from the shared TensorAgent integration.

## Design sources

The user-provided [TensorSharp design discussion](https://chatgpt.com/share/6ab43386-9464-83e8-828a-a1e97583bee4)
proposed independent agent sessions, a runtime manager, tree-wide bounds,
lifecycle tools, and a read-only first stage integrated with existing skills.
The current implementation adds isolated workspaces and dependency scheduling.
Heterogeneous model selection and conversation-history forks remain outside
this coordinator's API.

Implementation and prompt design were compared with the unchanged public
[OpenAI Codex checkout](https://github.com/openai/codex/tree/e72da2b53805894878023d01949a25a082e0a5cb),
pinned at `e72da2b53805894878023d01949a25a082e0a5cb` on 2026-09-25.
TensorSharp adapts these ideas to its existing C# host and local-model tools:

| Codex reference at the pinned revision | Design applied here |
|---|---|
| [Delegation guidance](https://github.com/openai/codex/blob/e72da2b53805894878023d01949a25a082e0a5cb/codex-rs/core/src/tools/handlers/multi_agents_spec.rs#L711) | Bounded independent tasks, useful parent work during delegation, distinct ownership, and evidence-based synthesis |
| [Proactive mode instructions](https://github.com/openai/codex/blob/e72da2b53805894878023d01949a25a082e0a5cb/codex-rs/prompts/src/model_messages/multi_agent.rs) | Explicit prompting for the model's delegation decision |
| [Shared agent registry](https://github.com/openai/codex/blob/e72da2b53805894878023d01949a25a082e0a5cb/codex-rs/core/src/agent/registry.rs) | Tree identity, depth, and shared capacity accounting |
| [Child configuration](https://github.com/openai/codex/blob/e72da2b53805894878023d01949a25a082e0a5cb/codex-rs/core/src/agent/child_config.rs) | Inheriting effective runtime permissions and model settings |
| [Completion routing](https://github.com/openai/codex/blob/e72da2b53805894878023d01949a25a082e0a5cb/codex-rs/core/src/agent/control/completion.rs) | Separate child context and parent result delivery |
| [Wait handling](https://github.com/openai/codex/blob/e72da2b53805894878023d01949a25a082e0a5cb/codex-rs/core/src/tools/handlers/multi_agents_v2/wait.rs) | Bounded asynchronous waiting and explicit timeout state |

Codex delegates task decomposition and role selection to the model and enforces
runtime limits and inherited authority in host code. Its current V2
[workspace instructions](https://github.com/openai/codex/blob/e72da2b53805894878023d01949a25a082e0a5cb/codex-rs/prompts/src/multi_agent_instructions.rs#L13)
explicitly describe a shared filesystem and working directory. TensorSharp's
private selected-file workspaces, output handoff, and dependency-ready queue are
TensorSharp additions, not claims about Codex's workspace isolation or an
upstream automatic DAG planner.

This is not a claim of API compatibility with Codex. TensorSharp uses fresh
child contexts and the five tools listed above, rather than exposing all of
Codex's history-fork, messaging, and resume options. Codex's
[typed role overrides](https://github.com/openai/codex/blob/e72da2b53805894878023d01949a25a082e0a5cb/codex-rs/core/src/agent/role.rs#L36)
and parent-derived runtime policy inform the rule that delegation cannot expand
authority.

## Validation and performance

The [dependency scheduler benchmark](../eng/validation/MultiAgentSchedulerBench/README.md)
compares one and three execution slots on a ten-task graph, checking dependency
order, result handoff, completion counts and observed concurrency. Its zero-delay
arm measures orchestration overhead. The [workspace server probe](../eng/validation/probe_multi_agent_workspaces.py)
exercises model-created workers, private file edits, dependent review and parent
integration through the real Web UI API, with downloaded JSON artifact checks.
Both write generated evidence to ignored directories; neither substitutes for
representative model/device benchmarks.

The [Web UI activity regression](../eng/validation/validate-webui-agent-activity.py)
serves the shipped chat page with controlled SSE frames and synthetic agents.
With Python Playwright and Chromium installed, run
`python eng/validation/validate-webui-agent-activity.py --browser PATH_TO_CHROMIUM`.
It checks mouse and keyboard disclosure controls, live updates, literal text
rendering, and terminal cleanup. Evidence goes to `artifacts/webui-agent-activity/`;
this browser check does not require or validate a loaded model.

Evaluate single-agent and automatic multi-agent modes on the same tasks,
model, backend, hardware, sampling settings, and completion criteria. Include
both substantial independent work and small or dependent tasks, so unnecessary
delegation appears as overhead. Warm up each path, run repeated trials, and
record latency distribution, completed-task score, input/output token usage,
failed or truncated work, and observed concurrency.

Use separate evidence for two questions:

1. Deterministic generators and controlled tools establish lifecycle behavior,
   concurrency, isolation, cancellation, and overlap. A synthetic delay
   benchmark measures harness overlap and overhead; it cannot establish model
   quality or GPU speed.
2. Real-model end-to-end runs establish whether the model selects useful tasks,
   follows the tool protocol, finds evidence, and integrates results correctly.
   Grade against independent expected facts or executable acceptance checks.
   Compare repeated trials and report the exact model revision and device.

The reusable [MultiAgentBench harness](../eng/validation/MultiAgentBench/README.md)
supports both modes. For example:

```sh
dotnet run --project eng/validation/MultiAgentBench -c Release -- --iterations 8 --warmup 1 --work-ms 80 --out artifacts/multi-agent/scripted.json
dotnet run --project eng/validation/MultiAgentBench -c Release -- --endpoint http://localhost:8000/v1/chat/completions --model MODEL --tensorsharp --iterations 5 --warmup 1 --out artifacts/multi-agent/real-model.json
```

The scripted mode performs real skill reads with a controlled analysis delay.
The endpoint mode lets the actual model choose whether to delegate while the
benchmark owns orchestration and fixture tools; the endpoint supplies generation.
Follow the harness README when configuring a TensorSharp endpoint so that the
server does not independently orchestrate those calls. To evaluate the server's
own orchestration, compare API requests with `multi_agent: false` and
`multi_agent: true` separately. The harness's nine-fact recall and unsupported-fact
counts are narrow, reproducible quality proxies; they do not establish
superiority on general coding or reasoning tasks.

No results from these harnesses are committed to the repository. Their reports
go to ignored `artifacts/`, and no latency, completed-task score or
delegation-rate figure is published for any model or device.

Multiple agents can improve coverage and reduce wall time when useful work
overlaps. They also consume additional generation tokens and context memory.
On a single saturated inference device, scheduling more conversations may
increase latency. Improved quality and performance are workload-dependent
outcomes to measure, not guarantees of enabling the feature. Unavailable models
or devices, skipped scenarios, and synthetic-only measurements must be recorded
as such, never counted as successful real-model validation.

For Gemma 4, prefix reuse can leave parent and child KV caches at different
capacities. The native `Gemma4ModelDecodeBatchedEx2` path supports these mixed
capacities without enlarging the child caches. It requires an updated native
`GgmlOps` library as well as the managed server. The server logs a successful
fused batch once, and includes a reason when a batch declines. See the
[Gemma batching details](models/gemma4.md#token-batched-fused-decode-for-concurrent-requests-gemma4modeldecodebatchedex2).

Generated validation reports and logs belong in ignored `docs/validation/` or
`artifacts/`; reusable validation programs belong in `eng/`, with fixtures in
their test project. This implementation is TensorSharp-owned managed host
code. It does not modify ggml upstream sources or require a patched native
dependency, and makes no claim to a new native KV-sharing optimization.
