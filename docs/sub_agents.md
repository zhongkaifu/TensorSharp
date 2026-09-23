# Sub-agents

[English](sub_agents.md) | [中文](sub_agents_zh-cn.md)

A sub-agent is another copy of the model that the model itself starts for one task.
It has the same tools and works in the same working directory as the agent that
started it, but it has **its own context**: it sees its task, not the conversation.
It works in parallel with its parent and hands back one final answer.

TensorSharp implements this with the tool surface OpenAI's Codex uses, and it runs
every sub-agent as an ordinary sequence on the **same loaded model**. The
continuous-batching engine decodes the parent and all its sub-agents together, and
each sub-agent's prompt starts with its parent's exact instructions and tool block,
so the engine reuses that cached prefix instead of prefilling it again.

Sub-agents are **off by default**. Turn them on with `--sub-agents`.

## Quick start

```bash
TensorSharp.Server --model gemma-4-E4B-it-Q8_0.gguf --backend ggml_metal \
    --code-exec --sub-agents
```

Then ask for delegation explicitly — the tool descriptions tell the model to start
sub-agents only when the user, or a skill it is following, asks for them:

```bash
curl -s http://localhost:5000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "gemma-4-E4B-it-Q8_0",
  "messages": [{"role": "user", "content":
    "Use three sub-agents working in parallel, one per task: compute the sum of 1..1000, the number of primes below 10000, and the 50th Fibonacci number, each with a python3 program. Wait for all three and reply with the three numbers."}]
}'
```

Every chat surface gets the tools: the Web UI, `/v1/chat/completions`,
`/v1/responses` and Ollama's `/api/chat`. They are offered only on requests that
already have tools — Agent Skills or `--code-exec` — because a sub-agent with no
tools can do nothing its parent could not do itself.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--sub-agents` | off | Offer the five sub-agent tools. Env: `TS_SUB_AGENTS` (any value but `0`). |
| `--sub-agents-max-threads <n>` | `4` | Sub-agents open at once in one turn, 1–16. Starting one more first closes the longest-finished agent whose answer was already delivered; it fails only when every open agent is still working. Env: `TS_SUB_AGENTS_MAX_THREADS`. |
| `--sub-agents-max-depth <n>` | `1` | How deep sub-agents may nest, 1–4. At `1` the model may start sub-agents but a sub-agent may not start its own. Env: `TS_SUB_AGENTS_MAX_DEPTH`. |

`--sub-agents` also raises the engine's retained-state cap
(`TS_RETAINED_FUSED_CACHE_MAX`) from 4 to `2 × (max-threads + 1)` unless you set
it yourself, and says so at startup. See [KV cache](#kv-cache) for why.

The same switch exists on `SkillsChatClientOptions.SubAgents` for the C# client
that runs the tool loop locally against a remote endpoint.

## The tools

| Tool | Parameters | What it does |
|---|---|---|
| `spawn_agent` | `message` (required), `fork_context` | Starts a sub-agent on `message` and returns its id (`agent_1`, `agent_2`, …) at once. With `fork_context: true` the agent starts from a copy of the conversation instead of a fresh context. |
| `send_input` | `target`, `message` | A follow-up. A finished agent starts working again with its earlier context; a working one reads the message before its next step. |
| `wait_agent` | `targets` (ids, comma-separated, or `all`), `timeout_ms` | Waits until any awaited agent finishes and returns the final answer of every awaited agent that has finished and was not reported yet. Default 300 s, clamped to 10 s – 1 h. |
| `close_agent` | `target` | Stops an agent and every agent it started; returns its status before closing. |
| `list_agents` | — | The caller's agents: id, status, task. |

An agent can address only the agents **it** started.

### What the parent receives

A finished agent's answer arrives **exactly once**: as the result of
`wait_agent`, or — if the parent never waits — appended to the next tool result it
receives, inside `<subagent_notification>`. Each answer carries the host's own
record of what the agent did, next to the agent's claim:

```
agent_2 completed in 7.4 s (3 rounds; tools it ran: write_file, shell). Its final answer:
The sum of the integers from 1 to 1000 is 500500. File: sum.py
```

An agent that ran no tools is reported as such (`(1 round; it called no tools)`),
because a small model's answer is a claim, and the parent cannot read the agent's
transcript to check it.

### The end of a turn

Codex's guidance to its model is "wait for sub-agents before yielding". A small
local model forgets. So when the model answers while agents it started are still
working, or have answers it has not seen, the host waits for them, hands their
results over, and gives the model one more round to use them. The answer written
without them is not shown. If no round is left, still-working agents are stopped
and the answer says so. Sub-agents never outlive the turn: whatever is still
running when the request ends is stopped.

## How it runs

- **One runtime per turn.** `SubAgentRuntime` (`TensorSharp.AgentHost/Agents/`)
  owns every agent the turn starts — ids, tree, limits, status, delivery — and
  is disposed when the turn ends.
- **Each agent is a tool loop.** A sub-agent runs `SkillAgentLoop` with the same
  tool context as its parent. The host supplies only how a list of messages
  becomes a generation: on the server, each round goes through the same
  `ChatGenerationPipeline` as the parent's, as a separate sequence.
- **Parallel on one model.** The engine batches decode across sequences, so the
  agents' rounds run at the same time as each other and as the parent's.
- **Shared files, separate shells.** Agents share the parent's workspace — files
  one writes, the others see — but each has its own **lane**
  (`SessionWorkspace.ForAgent`): its own shell working directory, exported
  variables and read ledger, so one agent's `cd` cannot move another's next
  command. Code tools still take the workspace's execution lock one at a time.
- **Streaming.** While `wait_agent` runs, the Web UI shows what each agent is
  doing (`agent_1: round 2: shell`) under "Waiting for sub-agents…".

## KV cache

Two things decide how much a sub-agent costs.

**The shared prefix.** A fresh sub-agent's prompt is its parent's leading
system messages copied verbatim, the parent's tool list unchanged, and then its
task as the first user message. The engine checkpoints the public prefix — the
system messages plus the tool block — and every sub-agent resumes from that
checkpoint instead of prefilling the instructions again. That is why the
sub-agent tools stay declared even to an agent at the depth limit (the call is
refused instead): a tool list with one tool missing diverges from the parent's
in the first few hundred tokens.

**Forks.** A forked sub-agent (`fork_context: true`) runs in its parent's cache
scope, with the parent's conversation up to its spawn call as its prompt, so it
resumes from the parent's own cached state and prefills only the spawn call and
its task.

**Retained state.** The engine keeps a bounded number of finished
conversations' KV for reuse (`TS_RETAINED_FUSED_CACHE_MAX`, default 4, one
global LRU). Every sub-agent round publishes into that pool, so with the
default a parent waiting on four agents lost its own cached conversation during
their first round and re-prefilled it after `wait_agent`. `--sub-agents`
therefore raises the cap to two states per concurrent conversation.

## Observability

The server logs one line per event, all under the `SkillToolInvoked` event id:

```
agents.spawn id=agent_1 parent=root depth=1 fork=False evicted=- task=...
agents.round id=agent_1 fork=False promptTokens=3381 kvReused=3178 evalTokens=31 ttftMs=44 tokensPerSec=44.7 finishReason=eos
agents.finish id=agent_1 status=completed turns=1 rounds=3 toolCalls=2 ms=5440 hitRoundLimit=False answerChars=141
agents.deliver id=agent_1 via=wait_agent
agents.collect-before-answer round=3 ms=2210 delivered=True
agents.turn spawned=3 completed=3 failed=0 closed=0 stoppedAtTurnEnd=0
```

`eng/validation/sub-agents-e2e.py` drives a running server through a set of
scenarios and reads these lines back per request.

## Where TensorSharp follows Codex, and where it does not

| | Codex | TensorSharp | Why |
|---|---|---|---|
| Tools | `spawn_agent`, `send_input`, `wait_agent`, `close_agent`, `resume_agent`; v2 adds `list_agents` | the same minus `resume_agent`, plus `list_agents` | a turn-scoped runtime has nothing to resume |
| Ids | UUIDs (v1), model-chosen paths (v2) | `agent_1`, `agent_2`, … | small models copy short ids reliably |
| Spawn policy | only when the user or instructions ask | same, in the `spawn_agent` description | |
| Depth | 1; tools hidden at the limit | 1; tools declared, spawn refused with Codex's message | identical tool blocks keep the KV prefix shared |
| Open agents | 6 (v1), error at the limit; v2 evicts idle finished agents | 4, evicts finished agents whose answers were delivered, then errors | bounded GPU contention without relying on `close_agent` |
| Results | v1: `wait_agent` returns them *and* a notification repeats them; v2: mailbox only | exactly once, via `wait_agent` or a notification in the next tool result | an 8k context can afford the answer once |
| `wait_agent` timeout | default 30 s, 10 s – 1 h | default 300 s, same clamp | returns early anyway; local agents are slower |
| Forks | drop tool calls from the copied history | keep them | the copied prefix is only worth copying if it matches the parent's KV |
| End of turn | instructed to wait | host collects outstanding agents | small models forget |

## Limits and caveats

- Delegating pays when each sub-task is substantial and its result is short.
  When the parent must repeat its agents' output verbatim — four paragraphs
  written by four agents, pasted into the answer — the parent's final
  generation is as long as doing the work itself, and delegating is slower.
- Families that do not batch decode across sequences run agents one step at a
  time; they still get context isolation, not speed.
- The TensorAgent app does not enable sub-agents: it keeps one retained
  conversation to fit a phone's memory, which sub-agents would evict.
