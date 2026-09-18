# Agent Skills and agentic code work in TensorSharp

An Agent Skill is a folder of instructions that a model loads when — and only
when — a task needs it. The folder holds a `SKILL.md` written for the model
rather than for a person, plus whatever scripts, reference documents and assets
those instructions refer to. TensorSharp discovers such folders, advertises
their one-line descriptions to the model, and lets the model pull the rest in
mid-answer through two built-in tools that TensorSharp executes **itself**.

That last part is what makes the feature usable from an ordinary client. A
caller sends `"skills": ["pdf"]` on a normal chat request and gets back a
finished completion; the fetches the model made along the way happened inside
the server, next to the weights.

Everything described here lives in `TensorSharp.AgentHost/Skills/` and
`TensorSharp.AgentHost/CodeExec/` and is shared by the CLI,
`TensorSharp.Server` and the public C# API. This is one model in a bounded,
in-process tool loop. TensorSharp does not implement subagents, multi-agent
delegation, or an interactive per-command approval queue; the operator grants
the execution surfaces once at startup, and they remain unavailable otherwise.

## What a skill is

A skill is a directory whose name *is* the skill's name:

```
pdf/
├── SKILL.md              # required: frontmatter + instructions
├── scripts/
│   └── extract_tables.py
├── references/
│   ├── forms.md
│   └── api.md
└── assets/
    └── template.pdf
```

`SKILL.md` is a Markdown file that opens with a YAML frontmatter block:

```markdown
---
name: pdf
description: Extract text and tables from PDF files, fill in PDF forms, and merge or split documents. Use when the user provides a PDF or asks to produce one.
license: Apache-2.0
compatibility: Requires python3 with pypdf and pdfplumber installed.
metadata:
  version: "1.2.0"
  author: example
---

# PDF processing

## Extracting a table

Run `scripts/extract_tables.py <file.pdf>`; it prints one CSV per table...
```

Everything after the closing `---` is the **body** — the instructions the model
follows once the skill is in play.

### Frontmatter fields

TensorSharp follows the [Agent Skills
specification](https://agentskills.io/specification).

| Field | Required | Constraint |
|---|---|---|
| `name` | yes | 1–64 characters, lowercase `a-z`, `0-9` and single hyphens, no leading or trailing hyphen, no `--`. The specification requires it to match the directory name. |
| `description` | yes | 1–1024 characters. Whitespace is collapsed to a single line, so a folded or literal block scalar is fine to author. |
| `license` | no | Free text; an SPDX identifier by convention. |
| `compatibility` | no | Up to 500 characters describing what the environment must provide. Surfaced to the model as `requires:` in `skills_list`. |
| `metadata` | no | A mapping of string keys to string values. The specification reserves nothing inside it — `version` and `author` are conventions, not fields. |
| `allowed-tools` | no | A whitespace-separated string (a YAML list is also accepted). Marked experimental by the specification and **advisory** here: TensorSharp records and reports it, and enforces it only for the skill-owned tools it actually controls. |

Keys outside that set are not dropped: they are kept verbatim, because a client
that writes its own frontmatter key is entitled to read it back.

### Strict about two things, forgiving about everything else

`SkillManifestParser` refuses to load a skill in only a few cases, all of which
make the skill invisible to the model no matter what else it contains:

* no YAML frontmatter block delimited by `---`, or frontmatter that is not
  valid YAML (a duplicate key is a parse failure rather than a silent
  last-one-wins);
* a missing or empty `description`;
* a `name` that is unusable **and** no directory name to fall back on.

Everything else loads with a **warning** attached, reported by `--list-skills`
and by the management API and never fatal:

* `name` disagrees with its directory;
* `name` is not a legal skill name and was normalized into one (`My PDF Tool` →
  `my-pdf-tool`);
* `name` is missing and the directory name was used;
* `description` is over 1024 characters, or `compatibility` over 500;
* `metadata` is not a string-to-string mapping, or carries values that are not
  strings (those are dropped).

The rationale is one-sided on purpose: a skill three characters over the
description limit is still a skill the user wants to use, and rejecting it
serves nobody. Failing *loudly at load* beats failing *silently at inference*,
which is what a strict reader would produce for a skill with a wrong name.

### The directory

`scripts/`, `references/` and `assets/` are conventions, and TensorSharp
classifies bundled files by them — a file under `scripts/` (or an executable
file type anywhere) is a script, one under `references/` or `reference/` (or any
Markdown file elsewhere) is a reference, one under `assets/` (or any binary
resource) is an asset, and anything else is simply *other*.

They are conventions and not a schema. **Any file or directory layout loads.**
The published skills use per-language folders (`python/`, `typescript/`),
`examples/`, `core/`, and root-level `.md` files next to `SKILL.md`, and all of
it is indexed and readable. The classification exists so listings can be
grouped, not to constrain authors.

Per-skill ceilings (from `SkillRegistryOptions`): a `SKILL.md` is parsed up to
4 MB, one skill directory may hold up to 4096 files totalling 256 MB, and a
whole registry is capped at 512 skills so that a root pointed at the wrong
directory fails visibly instead of exhausting memory.

## Progressive disclosure

The point of a skill is that it costs nothing until it is used. TensorSharp
implements that as three tiers.

| Tier | What it is | When it reaches the model |
|---|---|---|
| **Metadata** | `name` + `description` | Always, for every reachable skill — selected or merely advertised |
| **Instructions** | the `SKILL.md` body | When the model **activates** the skill, by calling `skills_read(skill, "SKILL.md")` |
| **Bundled files** | `scripts/`, `references/`, `assets/`, anything else | Never inlined; the index of paths and sizes rides with the activation, contents come from `skills_read` |

Activation is the **model's** decision, not the user's. Selecting a skill scopes
what the conversation can reach and tells the model which skills to prefer; it
does not load anything. This is the specification's contract, and it is what
Codex and Claude Code both do — Codex renders exactly one `- name: description`
line per skill at turn start and has no code path that can put a body there.

TensorSharp used to inline every selected body. Measured on a four-skill request
(`doc-coauthoring`, `internal-comms`, `pdf`, `pptx`): the block cost **12,848
tokens**, 52% of it for skills the model never referenced, and the model called
`skills_read` for the largest one anyway — re-sending 20,675 bytes it already
had, visible as a KV-reuse drop to 79% and 5,570 re-prefilled tokens. The same
block now costs **1,050 tokens**.

### The injected block

`SkillPrompt.Plan` decides what goes in front of the model and
`SkillPrompt.Apply` puts it there. The block opens with `## Agent skills` and
has up to four parts:

1. one sentence defining what a skill is and that its instructions outrank the
   model's default approach for the task it covers;
2. **Skills selected for this conversation** — one `- name: description` line
   per skill, plus its approximate instruction size and the `skills_read` call
   that loads it;
3. **Other available skills** — the discovery catalog, one `name: description`
   line per skill, with an instruction to load one with `skills_read` if it fits
   the task better;
4. **How to use a skill** — the operating rules (read the whole `SKILL.md`
   before acting; treat relative paths as paths *inside the skill*, never host
   paths; prefer a shipped script over retyping its logic; do not load a
   reference the task does not call for; continue from the reported offset when
   a file comes back truncated).

A host that knows its model will not make the extra call can set
`SkillPromptOptions.InlineSelectedBodies` and get the old behaviour back; it is
off by default, and forced on for families that cannot carry tool declarations
at all, where there is no second chance to fetch anything.

### The budget

Sizes are approximated at 4 bytes per token (`SkillTextBudget`) rather than
tokenized. That approximation only ever decides what to *include* — nothing the
model sees is a token count TensorSharp presents as exact — and it avoids
tokenizing every registered skill on every request, at a point where a tokenizer
is not necessarily even available yet.

| Knob | Default |
|---|---|
| Metadata block (the normal path) | 2% of `contextTokens`, clamped to 1024–10000; 2000 when the context length is unknown — Codex's shape and very nearly its constants |
| Whole block when bodies ARE inlined | `contextTokens / 4`, clamped to 1024–48000; 16000 when the context length is unknown |
| One inlined `SKILL.md` body | three quarters of the block ceiling (minimum 512) |
| Description | trimmed to 1024 characters at a word boundary — the specification's own ceiling on the field, so a conformant skill is never cut |
| Catalog entries | at most 96, then "(N further skills are installed…, call skills_list)" |
| File index per skill | at most 40 paths, then "(N more; call skills_list)" |

### Deferred

A skill whose body is not in the prompt — which is now every selected skill on a
tool-capable family — is **deferred**: announced by name and description, with
its approximate size and the `skills_read(skill="…", path="SKILL.md")` call that
loads it. The model knows the skill exists and what it is for; it fetches the
instructions if it decides to use them.

The description is the whole routing signal once bodies are deferred, which is
why the trim went from 320 characters to the specification's 1024. At 320 the
cut landed on the "when to use it" clause every time — it deleted `pptx`'s
*"Trigger whenever the user mentions 'deck,' 'slides,' 'presentation'"* — so
deferring bodies while keeping the old trim would have made routing strictly
worse.

Where the model has no tools at all (see [Model-family
caveats](#model-family-caveats)) there is no second chance, so the rules change:
the discovery catalog is dropped — advertising skills that can never be loaded
is pure waste — the whole budget goes to inlining, and a single oversized skill
is inlined *anyway* and truncated at the cap with the truncation announced,
rather than silently doing nothing.

## The skill tools (three with `--skills-allow-exec`)

```
skills_list()                                  -> every reachable skill, its description and its files
skills_read(skill, path, offset)               -> one page of one file; "SKILL.md" is the instructions
skills_run(skill, path, args)                  -> run a bundled script; OFF unless the operator opts in
```

`--code-exec` adds four more, and they ride *alongside* these rather than
replacing them: `read_file`, which shows a file's real bytes with line numbers;
`write_file`, which creates a new file; `shell`, which runs a command
line in the current working directory; and `apply_patch`, which modifies a single
file or multiple files by anchored hunks, all or nothing. A Web/CLI conversation
keeps that directory for its session. An OpenAI or Ollama request keeps an isolated directory
for every internal tool round in that request and deletes it afterward, preserving
cross-request statelessness without forcing a repair round to regenerate a file
the preceding round just created. A skill's instructions and a command the model
writes are useful in the same turn.

`apply_patch` is the only tool models are instructed to use to modify existing
files, from a one-line replacement in one file to coordinated changes across
multiple files. It also supports creating, renaming, and deleting files.
`write_file` is reserved for creating new files. Read the relevant contents first,
then send the smallest anchored patch in a `*** Begin Patch` / `*** End Patch`
envelope; each existing file being modified has its own `*** Update File:` section.

`skills_read` returns at most 48 KB per call by default, with a header naming
the skill, the file and the byte range, and a footer that spells out the exact
follow-up call when there is more:

```
[Truncated. Continue with skills_read(skill="pdf", path="references/api.md", offset=49152).]
```

**The declarations are deliberately flat.** A tool parameter in TensorSharp
carries a type name, a description and an enum, and nothing else: nested
`properties`, `items` and the rest of JSON Schema are dropped when a tool is
parsed and cannot be re-emitted, and the Harmony renderer degrades an `array`
parameter to `any[]`. So every parameter here is a string or an integer. The
tool names use underscores rather than dots because several families splice a
tool's name into their markup unescaped.

The tools are also written to survive the ways models get them slightly wrong.
`skills_read` with no `path` is answered with `SKILL.md`; `path="pdf/api.md"`
is accepted as `api.md` inside skill `pdf`, because that is how the file is
spelled relative to the *skills directory* and models reach for it constantly;
`file`, `resource`, `script` and `arguments` are accepted as argument aliases;
and arguments arrive as `JsonElement`s, boxed primitives or strings depending on
which parser produced them, all of which are read. A `..` segment is
deliberately *not* rewritten — see [Security](#security-model).

A tool call that fails never throws. The model is handed a sentence it can act
on ("No skill called 'pdfs' is available. Available skills: pdf, xlsx.") and
keeps going, because a bad tool call should cost one round, not the answer.

If the caller's own tool list already contains a tool named `skills_read`, the
**client's** tool wins and the collision is reported to the host for logging.
The client has an implementation and an expectation; shadowing it to add a
feature it did not ask for would break a working integration.

## The in-process agentic loop

`SkillAgentLoop.RunAsync` drives: generate → answer any skill tool calls in
process → generate again, until the model stops asking.

With `--code-exec`, the same loop also answers four built-ins: bounded
`read_file`, new-file `write_file`, `shell`, and atomic `apply_patch` for
single-file or multi-file changes. These are operator-provided tools,
not caller tools. They are off by default; enabling them does not imply network
or package-install permission.

**Why it is in process.** An ordinary OpenAI client sends `skills: ["pdf"]` and
one user message. If TensorSharp returned `skills_read` as a tool call, that
client would have no implementation for it and the conversation would stall —
progressive disclosure would work only for clients that had been rewritten to
understand skills, which is exactly the clients that do not need it. Because
these particular tools are read-only and confined to a directory the operator
already chose to expose, TensorSharp can answer them itself. That is what makes
progressive disclosure work over a stateless HTTP API at all.

**Stateless does not mean stateless between the tool calls inside one request.**
When code execution is available, the OpenAI chat, OpenAI Responses and Ollama
chat adapters allocate a private request workspace before entering a tool-compatible
loop. A generated file, its failing test, an anchored `apply_patch` repair and the
successful rerun therefore operate on the same bytes. The workspace is released
after the buffered or streaming response ends (including rejection and cancellation);
another HTTP request never inherits it. With skill-script execution but no code
runner, scripts retain their per-call scratch directories.

**The caller's own tools are never executed.** A turn that calls one stops the
loop, and the call is returned to the caller as a normal `tool_calls` response
for it to service — with any skill results the same turn produced already in the
returned history. Only the client knows what its tools do.

Bounds:

| Bound | Default | Why |
|---|---|---|
| Rounds per turn | 8, or 24 with `--code-exec` (`--skills-max-rounds`, 1–64) | Eight covers the realistic reading case — read a skill, read two references, page through a long one — while bounding a model that loops on a file it keeps mis-naming. Once the same counter also gates writing a file with `shell`, running it, reading the traceback and fixing it, eight is not enough: a README → internal-comms doc → slide deck run spent three rounds reading skills, two producing the document and three on a deck it was still debugging. An operator's own number is used as given. Each round is a full generation. |
| Skill tool calls per round | 8 | A model emitting fifty reads in one turn is malfunctioning, and answering all of them would blow the context before the next generation. |

When the round budget runs out the model is told so *in the conversation*
("answer now using what you have already read, and say which part you could not
check") and one final generation runs. Returning a tool-call-only turn to the
user would show them nothing at all.

Each executed call is reported through `SkillAgentLoopOptions.OnInvocation`,
which is what the server's structured logs and `skill_step` SSE completion
metadata are built from. Code tools additionally emit transient
`tool_progress` phases (`writing`, `running`, `finished`), including produced
files when a run completes.

### Streaming

A request that selects or discovers skills streams token by token, on every
surface, exactly like one that does not.

The mechanism is that the loop parses the stream **itself**. It has to — it is
hunting for the `skills_read` calls it answers in process — so having parsed, it
forwards the *separated* pieces rather than the raw text: content as content,
reasoning as reasoning, tool markup not at all. Its updates carry `IsParsed`, and
an adapter that sees that flag skips its own `IOutputParser` and emits the pieces
directly. Every round streams, including the ones that end in a lookup, and the
markup a client cannot service never leaves the server.

That indirection is the point. The first implementation buffered each round and
replayed only the last, and it cost a skills request its whole stream: measured
on the Web UI, 26 SSE content frames starting 0.17 s in became 3 frames all
landing at 3.28 s. Stopping the forward at the first tool token instead is worse,
not better — it strands the adapter's parser inside a half-open tool-call span,
and the next round's text gets eaten as that call's arguments, so the answer
disappears rather than merely arriving late. Parsing once, in the loop, means
there is no span for the adapter to be caught inside.

Measured on `gemma-4-E4B-it-Q8_0` / `ggml_metal`, a two-round request — the model
reasons, reads `forms.md`, then answers:

| | Frames | First frame |
|---|---|---|
| Reasoning, round 1 | 264 | 1.75 s |
| `skill_step` (lookup done) | 1 | 7.76 s |
| Answer, round 2 | 263 | 11.11 s |

529 frames spread over the full 17.8 s. Buffered, the same request produced three
frames at the end.

The text is unchanged by streaming it: for the same deterministic request the
concatenated stream is **byte-identical** to what the non-streaming endpoint
returns, for single-round, multi-round and no-skills cases alike.

The Web UI additionally carries `skill_step` and `tool_progress` frames. It uses
them for one transient current-activity panel, which is replaced as work moves
on and cleared when the operation finishes; this is live status, not retained
execution history. Produced files remain visible as artifact download chips.

### What a round costs

The loop records each assistant turn with the **raw tokens** the model emitted,
and `SkillPrompt` clones messages completely rather than dropping
`RawOutputTokens` and `TextFilePaths` the way `StructuredOutputPrompt` does.
`KVCachePromptRenderer` splices those tokens back into the next render instead
of re-tokenizing the text, so the re-rendered prefix stays byte-identical from
round to round — and the KV cache follows it.

Measured on `gemma-4-E4B-it-Q8_0` / `ggml_metal`, a lookup that appends a 7.9 KB
skill body:

| | Round 1 | Round 2 |
|---|---|---|
| Prompt tokens | 1825 | 4197 |
| KV prefix reused | 0 (cold) | **1983 (47.2%)** |
| Time to first token | 1.0 s | 1.3 s |

47% is the ceiling, not a shortfall: the 2214 tokens that were *not* reused are
the skill file the model just asked for, which has never been in the cache. The
whole of round 1 — prompt and generated tokens alike — carries over.

Getting there needed an engine fix.
`BatchExecutor.ComputeLiveContinuationLcp` required the live KV cache to be an
**exact** prefix of the new prompt. A turn that ends on a control token the chat
template never re-renders fails that test by a single token: Gemma 4 answers a
tool call by emitting `<|tool_response>`, the engine forwards it into the cache,
and the template then renders the boundary as `<turn|>\n<|turn>tool` instead. The
whole conversation re-prefilled from token 0. It now rewinds a bounded number of
trailing tokens (`MaxLiveContinuationRewindTokens`, 16) through the same
`TruncateKVCache` path speculative decoding uses for a rejected draft, and
declines anything longer — on a sliding-window model the cache is circular, and a
long rewind means the prompt genuinely diverges, where a clean re-prefill is the
correct answer rather than a cheaper wrong one.

The same fix repaired an ordinary multi-turn case that had nothing to do with
skills: a turn ending on EOS used to report 0% reuse on the next turn while a
`max_tokens`-terminated turn of the same conversation reported ~95%. It now
reports 99.5%.

### When the model gets it wrong, the host recovers instead of reporting

Three days of this server's own logs (966 turns, 491 rounds) say where a coding
turn actually goes: **39.7% of rounds, 59.7% of output tokens and 48.3% of wall
clock were recovery, not work.** And the recovery was not hard — almost every
round of it was spent on something the host already knew.

So the rule these mechanisms are built on is the one the operator's own framing
states: *code and scripts are deterministic, the LLM is not.* Where the host can
do the right thing itself, it does it, instead of writing a sentence asking the
model to. Where it genuinely cannot, it says the one thing that names the next
action — and it says it in the **tool result**, because that is the only channel
a 4B-class model reliably reads. A capability described in a declaration a
thousand tokens ago is a capability it does not use.

| What went wrong | Rounds it cost | What the host does now |
|---|---|---|
| First run dies on a missing import | 17 incidents / 68 rounds / 116 min | Installs it and re-runs the command **inside the same call** (`ShellRunner.RunWithAutoInstall`), bounded by 5 distinct packages and by the call's own `timeout_ms` |
| The model guessed a library's API | 10 incidents / 60 rounds | Reads the real API out of the installed package and appends it — `did you mean: notes_slide` plus the names the class actually has (`ApiProbe`). Measured at 100 ms on the real failing script |
| A whole file re-typed to change one line | 38 re-emissions / ~52k output tokens | The current tool declarations and system prompt require `apply_patch` for every existing-file modification, including one-line changes in a single file; `write_file` is reserved for new files. The earlier logs recorded **zero** patch calls in ten opportunities despite a prefer-a-patch paragraph, so that historical observation is not evidence of compliance with the current prompt. |
| A byte-identical failing command re-sent nine times | one whole turn | Says so, with the count (`AppendRepeatWarning`). The nine results had been identical except a scratch filename, which reads as new information |
| Absolute host paths in every traceback | 13.9% of all result characters | Rewritten to paths relative to where the command ran (`OutputPaths`) — still usable, and one logged round was lost to a model splicing two session ids together |
| A patch that applied but broke the file | — | The parse is checked and reported (`SyntaxCheck`); matching all-or-nothing is not the same as being right |
| A hunk that did not match | 6 anchor drifts | The file's **real bytes** at the closest anchor, numbered; and if the context carries `nl`/`grep -n` line numbers, that is named as the cause |
| A hunk that fits in several places | silent | Applied to the first, as the reference does — and the result now says so, with the other line numbers |
| `python: command not found` | 5 incidents / 9 rounds | `python` means this host's Python, via a shim in the session's own PATH |
| Out of rounds mid-work | 6 of 12 capped turns | The turn says it ran out, appended to whatever the model had already written — it used to say nothing unless the reply was *entirely* markup |

Two of these were bugs where the host reported success for something it had not
done, which is the one kind of result a model cannot recover from because nothing
contradicts it: a refused install left `false | tail -5` running and answered
`exit 0`, and a redirection (`2>&1`) was read as a package name.

**None of it touches the prompt.** Every message above rides on a tool result or
on a tool's own top-level description, so the injected block stays a pure
function of the selection and the options and the KV prefix is untouched — see
the next section for why that matters.

#### Measured against Codex and Claude Code, item by item

Four of these ideas were checked against the two reference implementations before
being built, and **two were dropped because the references do the opposite.**

| Proposed | Verdict | The citation that settles it |
|---|---|---|
| Fix every "success reported for something not done" | **Aligned — and load-bearing** | Both references instruct the model *not* to verify: Codex's `prompt.md` says "Do not waste tokens by re-reading files after calling `apply_patch` on them. The tool call will fail if it didn't work", and Claude Code's `Read` tool says "Do NOT re-read a file you just edited to verify — Edit/Write would have errored". That instruction is only safe if the failure paths are exhaustive |
| Code-specific sampling | **Partly aligned** | Every coding turn removes TensorSharp's built-in `1.1` repetition penalty when it is still at that default, because code legitimately repeats structural tokens. TensorSharp does **not** impose a code temperature by default: `--code-exec-temperature` changes only a temperature still at the built-in default, while an explicit client value wins. This matches the useful part of the references, which do not add a task-specific temperature or repetition penalty. |
| Prefer Python first | **Not aligned — dropped** | The rule in both is the inverse. Codex's shell guidance is five bullets, two of which steer *away* from Python ("Avoid using Python scripts just to print large file chunks", "prefer `rg`"); Codex also contains one literal "Prefer Python stdlib for portability" — but scoped inside a single skill, never as a global rule. And the log evidence is confounded: 26 of 32 JavaScript failures are one library's option surface, not the language |
| Check dependencies before running | **Not aligned — rejected** | Claude Code hits `ModuleNotFoundError` in 0.016% of its ~30,400 shell calls, and provisions once per session with a cached setup script rather than checking per run. Installing-and-re-running inside the same call is strictly cheaper than a pre-flight: zero extra rounds against one guaranteed extra round |
| Tell an environment failure from a code bug | **Aligned — kept and generalised** | This is Claude Code's sandbox escape hatch almost verbatim: the harness names the violation in the result, the model classifies it, and the *same command* is retried with the environment changed — the code is never touched |

Two mechanisms were ported from Claude Code that none of the proposals mentioned,
and one of them is the largest single win here:

- **A benign exit code is not a failure.** `grep` exits 1 when it finds nothing,
  `diff` exits 1 when files differ, `test -f` exits 1 when the file is absent,
  `git diff --quiet` exits 1 when there *are* changes. Every one of those was
  reported to the model as a broken command. Claude Code's Bash tool exempts
  exactly this, as a closed list of names at exit 1 only — a *false failure*, which
  costs the same recovery round a false success does and which none of the five
  items mentions.
- **"This file now holds what you wrote — no need to read it back."** 98.4% of
  Claude Code's successful edits carry that clause. It is the other half of the
  do-not-verify contract, and it belongs on the result rather than only in the
  declaration.


## Prompt injection and the KV prefix cache

The block is merged into the **leading** `system`/`developer` message when there
is one, and only becomes its own leading system message when there is not.

That is not a style choice. Appending a *second* system message is silently
dropped by the Mistral 3 renderer, emits a duplicate system turn on GPT-OSS's
Harmony format (which lifts `messages[0]` into its developer block and
synthesizes its own system block), and is the shape a GGUF-embedded Jinja
template is most likely to reject — which falls back to the hardcoded renderer
and silently changes the entire prompt format. Merging into the first message is
the one injection point every chat template in the repository handles.

**Every byte of the block is a pure function of the sorted selection and the
options.** The prefix cache chains a SHA-256 over 256-token blocks starting at
block 0 and stops adopting at the first mismatch, and this block sits at the very
front of the prompt. A timestamp, an absolute path, a "3 skills registered"
counter, or a selection rendered in whatever order the caller's JSON happened to
list it would change block 0 on *every* turn and drop prefix reuse to zero for
the whole conversation — not merely for the part that changed. So skills are
sorted by id with an ordinal comparison, separators are fixed, and nothing
environment-derived is rendered. The same conversation with the same skills
re-hashes identically turn after turn.

## Security model

A skill is untrusted content. Someone uploads a ZIP, or the operator points the
server at a directory of skills pulled off GitHub, and the model is then invited
to name files inside it.

### Path containment

`SkillPathGuard` is the single security boundary. Every path the model names,
and every relative path a `SKILL.md` body links to, goes through it before it
reaches the filesystem. It closes three separate escapes, because closing only
the obvious one is what would make it worthless:

1. **Lexical** — `..` segments, absolute paths (`/etc/...`), `~`, UNC paths
   (`\\server\share`), Windows drive-qualified paths (`C:\`), NUL bytes, and
   segments ending in `.` or a space (Windows strips those when opening, so
   `run.py.` and `run.py` would name one file and compare as two strings).
   Rejected before touching the disk.
2. **Canonical** — after `Path.GetFullPath` collapses the path, the result must
   still sit under the skill root. This catches what survives normalization on
   one platform's rules but not another's.
3. **Symbolic** — a symlink inside the skill pointing out of it.
   `Path.GetFullPath` does not follow links, so `references/host -> /` would
   otherwise pass the first two checks and then read anything. Every existing
   component of the path is resolved to its final target and re-checked —
   component by component, not just the leaf, because a link on an intermediate
   directory (`references -> /etc`) redirects everything beneath it.

Comparison follows the host filesystem's own case rule: ordinal on Linux,
case-insensitive elsewhere. Comparing case-sensitively everywhere would reject
legitimate reads on macOS; comparing case-insensitively everywhere would let a
Linux path that merely *looks* like the root prefix pass containment.

This is also why a `..` in a model-supplied path is left in place rather than
rewritten. Quietly resolving `../other/SKILL.md` to something inside the current
skill would be safe, but would report "does not exist" — teaching the model the
file is missing rather than that it may not look there.

### One skill cannot read another

`skills_read(skill=X, path=…)` resolves `path` under **X's own root**. There is
no spelling of a path inside skill `A` that reaches a file in skill `B`; a
cross-skill read requires naming `B` explicitly, and `B` has to already be
reachable in that request.

Reachability is per request, not global. A request's tool context holds exactly
what it selected plus what the host chose to advertise for discovery — never the
whole registry. Handing every request the whole registry would let a prompt
injected into one skill's `SKILL.md` pull the contents of a skill the user never
enabled. `--skills-no-discovery` (or `"skills_discovery": false` on a request)
narrows it further, to exactly the skills that request named.

### Installing a ZIP

`ZipFile.ExtractToDirectory` is deliberately not used. An entry name in a ZIP is
attacker-controlled text, and the classic zip-slip payload ships an entry called
`../../../.ssh/authorized_keys`; the framework helper has grown guards against
the obvious form, but the file that lands still depends on how the name
normalizes on the host platform. Every entry is resolved through
`SkillPathGuard` instead — the same check that confines the model's reads — so
extraction and reading agree on exactly one definition of "inside the skill".

| Guard | Default |
|---|---|
| Per-entry decompressed size | 64 MB |
| Whole-archive decompressed size | 256 MB |
| Entry count | 4096 files |
| Compression ratio | 200× (rejected as a decompression bomb) |

Size is enforced on the **decompressed stream**, never on the entry's declared
`Length` — that number comes from the archive's own headers, which a crafted
upload simply lies about. An empty archive is rejected. Both upload shapes are
accepted (`pdf/SKILL.md`, which is what every archive tool produces when you
compress a folder, and a bare `SKILL.md` at the archive root); macOS's
`__MACOSX` sibling is skipped rather than mistaken for the skill; and an archive
holding *several* skill directories is refused rather than silently installing
the alphabetically first one.

### Running scripts

`skills_run` is off by default and stays off unless someone passes
`--skills-allow-exec` (or sets `TS_SKILLS_ALLOW_EXEC`). Under the default
`--skills-sandbox required`, an enabled script still runs **sandboxed or not at
all**; `preferred` and `off` are explicit operator choices that can weaken that
guarantee.

**Two layers.** In process, always: the path resolves through `SkillPathGuard` so
only files inside the skill can be named; the interpreter comes from an
allow-list rather than from a shebang (`.py` → `python3`, `.js`/`.mjs` → `node`,
`.sh` → `/bin/sh`, `.bash` → `bash`, all replaceable or removable); the
interpreter is exec'd directly, with no shell parsing the argument list, so `;`,
`|`, `>`, `$` and backticks in an argument are data rather than syntax; the
environment is reduced to `PATH`, `LANG`, `LC_ALL`, `TZ` and the Windows
equivalents, so a host credential in `AWS_SECRET_ACCESS_KEY` or `GITHUB_TOKEN`
never reaches the child; the working directory is outside the skill — with
`--code-exec` it is the Web/CLI chat workspace or the current HTTP request's
private workspace, the same directory `shell` commands run in, so one step's
output is the next step's input, and otherwise a per-call scratch deleted when
the call returns; stdin is closed; the process is
killed at a 60-second deadline; and stdout and stderr are each captured up to
32 KB.

In the OS, through `--skills-sandbox`:

| Mode | Behaviour |
|---|---|
| `required` | **Default.** Sandbox or refuse. A host with no sandbox declines the tool and tells the model so. |
| `preferred` | Sandbox where the platform provides one, run anyway where it does not. For your own machine. |
| `off` | In-process limits only. |

The point of `required` being the default is that *"isolation was unavailable"*
must never quietly become *"isolation was skipped"*.

### What each platform enforces

| | macOS | Linux | Windows |
|---|---|---|---|
| Mechanism | `sandbox-exec` (Seatbelt) | `bwrap` 0.12.0+ (bubblewrap) | job object |
| Writes confined to the working directory | yes | yes | **no** |
| Network denied | yes by default | yes by default | **no** |
| Home directory unreadable | yes | yes | **no** |
| Process tree bounded | **no** — ordinary process groups are stopped, but a deliberately detached child can outlive the request | yes | yes |
| Available by default | always | only if `bwrap` 0.12.0 or newer is installed | always |

The two network opt-ins are deliberately separate. `--skills-allow-network`
changes the first two columns for bundled `skills_run` scripts;
`--code-exec-allow-network` changes them for model-authored `shell` commands.
Neither flag grants network access to the other execution surface.

Windows is weaker and says so. The Windows sandbox bounds the process tree and
**declares the rest as gaps**: every `skills_run` result on Windows carries a
`Not confined on this host:` line naming what was not enforced, and the same
appears in the startup log and in `--list-skills`. `SkillSandboxCapabilities` is
the single source for all three, so the claim and the implementation cannot
drift.

The mechanism that would close those rows is an AppContainer, and it has been
measured rather than assumed. A CPython child launched with
`PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES` is confined exactly as the macOS
column describes: writes outside its work directory fail, reads of the user
profile fail, and a socket fails with `WSAEACCES` unless the `internetClient`
capability is granted. It is still not usable for this feature, because nothing
that acts as a *shell* survives inside one:

* Windows PowerShell cannot initialise its filesystem provider - it needs to open
  the drive root, which the container SID is not granted - so `Set-Location`
  fails and every relative path resolves against `C:\`.
* An msys `bash` (Git Bash, MSYS2) fails at load with `STATUS_DLL_INIT_FAILED`
  (`0xC0000142`): it cannot create the shared sections msys needs.
* Loopback is blocked outright, so the egress proxy that makes the install
  allow-list enforceable is unreachable from inside the container.

TensorSharp's current Windows job-object backend therefore does not confine a
model-written command's files or network. Stronger Windows designs exist — for
example a separate low-privilege identity plus firewall/WFP rules — but are not
implemented here. The reference implementations share the important operator
principle: do not imply isolation the selected backend does not provide. Here
the explicit opt-in is `--code-exec-unconfined`, accepted by **both** hosts and
required on Windows for `--code-exec` to do anything at all; for skill scripts
the equivalent is `--skills-sandbox preferred`. Both are stated at startup - a
host told to run scripts it cannot confine says so on its own line, rather than
only inside a tool result the model is free to summarise as success.

**The same two layers confine the `shell` tool.** A command line the model types
is wrapped by `ConfinedProcess` through the same `ISkillSandbox`, under the same
`required`-by-default rule, so the table above is what a model-written command
gets as well — and every result likewise names what was not confined on this
host. Network access is denied by default. The explicit
`--code-exec-allow-network` / `TS_CODE_EXEC_ALLOW_NETWORK` opt-in gives every
model-authored command unrestricted host IP-network access — including LAN/loopback
services and IP listening sockets — while write and home-read confinement remains
active on macOS and Linux. Linux additionally bounds descendants with a PID namespace.
On macOS, children inherit Seatbelt and ordinary process groups are stopped, but a
deliberately detached child can outlive the request; every result reports that gap.
macOS denies common
`/private/tmp/com.apple.launchd*` pathname sockets while permitting runtime-required
Mach lookup and the exact mDNSResponder pathname socket required for DNS, and Linux
hides common `/run` endpoints, but this is not a complete
local Unix-IPC boundary:
macOS retains shared-temporary-directory Unix IPC for compatibility, and Linux's
host network namespace may expose abstract sockets and pathname sockets outside
`/run`. Other host-readable files can therefore be exfiltrated. Installing permission
does not imply that broader access: the host still performs recognised installs.
The child environment remains credential-blank. Direct-network runs receive only
credential-free host HTTP/SOCKS proxy settings; an authenticated upstream therefore
needs a credential-free host-side forwarder. Configured custom-CA bundles no larger
than 16 MiB are read once, and only validated public certificates are copied into an
immutable, read-only per-session snapshot. The original host path and any adjacent
text or private material are never exposed to generated code.
`--code-exec-allow-install` makes the host READ a `pip install` instead of
running it — the tool and the package names, nothing else — validate the names,
and run the installer itself with an argument vector it built: wheels only, no
install scripts, and, where the sandbox can pin a single loopback port
(Seatbelt), egress through the proxy to the hosts in
`--code-exec-install-domains` and nowhere else. The install command is then
substituted out of the model's line with `true`, so `pip install x && python y.py`
installs and then runs `y.py` under the configured command-network policy
(offline by default); an install that fails stops the line where it stands and
is what comes back, so `y.py` never runs. Giving the install command its own
socket was the earlier design, and it had two holes that could not both be
closed: the line is written by the model, so `--index-url` pointed the installer
wherever it liked, and a socket belongs to the whole line, so anything sharing it
with the install shared its reach — and on a host whose sandbox cannot pin egress
to a proxy (bubblewrap is all-or-nothing about the network) that reach was the
internet. Screening arguments closed the first hole and not the second. Reading
the request closes both, at the price of understanding a SUBSET of what pip and
npm accept and saying so: an option that changes where a package comes from
(`--index-url`, `-i`, `--find-links`, a URL requirement) is refused by name, and
so is an installer the host cannot perform on your behalf (`uv`, `poetry`, `gem`,
`cargo`) — ignoring either would install something other than what was asked for
and report success. A `-r requirements.txt` is honoured by reading the file and
validating each line. It is also what brought `--code-exec-packages` back: the
host builds the install, so a name allow-list applies to a recognised request
however the model spelled it. Once unrestricted command networking is enabled, neither
that package list nor the install-domain list is a security boundary: generated
code can fetch or execute an artifact without using TensorSharp's host installer.

**"Writes confined" has one deliberate exception: the shared system temp.** On
macOS `/private/tmp` is readable and writable. A whole class of tools a skill
invokes keeps a fixed-path scratch or singleton-IPC node there, independent of
`TMPDIR` and `HOME` (both already redirected into the working directory):
LibreOffice's headless converter opens
`/private/tmp/OSL_PIPE_<uid>_SingleOfficeIPC_<hash>` and, denied it, exits without
building its profile — which is why the `xlsx` skill's `recalc.py` could never
recalculate a sheet under the sandbox. It is not much of a hole: `/private/tmp` is
world-writable already (mode 1777, shared by every process on the host), what
matters stays closed — the home directory is still unreadable and the network
remains under its separate, denied-by-default policy — and the session's own
files live under the scratch root, never there. The per-user Darwin temp
(`/var/folders/…/T`) is still denied; nothing
needed it once `/private/tmp` was open. On Linux nothing had to be opened:
bubblewrap 0.12.0 is the minimum accepted version: older releases have a published
[setup-time symlink traversal](https://github.com/containers/bubblewrap/security/advisories/GHSA-pxhw-h44j-8pfx) that can escape a sandbox built over attacker-controlled
paths. TensorSharp refuses those releases rather than silently weakening `required`.
Bubblewrap mounts a fresh `--tmpfs /tmp`, so a write there succeeds, is invisible
to the host, and dies with the process.

Verified on macOS against a deliberately hostile skill — a script that tries to
read `/etc/passwd` and `~/.ssh`, read a *different* skill, write to `/tmp`, write
back into its own skill directory, open a socket, and read the host's
environment:

```
read /etc/passwd:     ALLOWED     (system reads stay open; the interpreter needs them)
read ~/.ssh:          denied
read other skill:     denied
write /tmp:           ALLOWED     (the shared 1777 temp, opened on purpose — above)
write own skill dir:  denied
write cwd:            ALLOWED     (the working directory — where output belongs)
network:              denied
env secrets:          none visible
```

The two that would have been an escape did not happen: the planted file in the
skill directory did not exist afterwards, because the skill is mounted read-only,
and nothing the script could reach let it read the home directory or another
skill, or open a socket. `/tmp/ts-sandbox-escape.txt` *was* written — that row is
the allowance above doing what it says, on a directory anything on the machine
can write anyway, and it is the reason the claim is "writes confined to the
working directory" and not "writes confined, full stop".

### Launch backends

Everything above the launch — reading the model's arguments, extracting installs,
building the scrubbed environment, the attempt ledger, artifact capture and every
sentence of a result — is host-side C# shared by every platform. What actually
*runs* something sits behind one seam, `TensorSharp.AgentHost.CodeExec.IShellBackend`:

```csharp
public interface IShellBackend
{
    string Name { get; }              // "process", "in-process"
    ShellProgram? Shell { get; }      // the dialect the model is told about
    ISkillSandbox? Sandbox { get; }   // what this backend actually confines
    bool CanRun { get; }
    string? UnavailableReason { get; }
    bool TryStart(ShellLaunch launch, out IShellJob? job, out ConfinedResult failure);
    ConfinedResult Run(ShellLaunch launch);   // TryStart + WaitForExit(launch.Timeout)
}
```

A `ShellLaunch` is either a shell **command line** (the model's text, after the host
has performed and substituted its installs, plus the `ShellSession` whose working
directory and exports it restores and re-saves) or an **argument vector** (a skill
script, a syntax check, an API probe, a package install — no shell ever parses
these), together with the host's terms: the working directory, the environment,
what may be written and read, whether the network is open, the deadline and the
output cap. `Purpose` says which of the five callers made it.

`ProcessShellBackend` is the desktop implementation and the default everywhere:
the command becomes the wrapper script the session writes, the argv is launched as
it is, and both go through `ConfinedProcess` under the detected OS sandbox — the
behaviour described in the rest of this section, unchanged. `ShellRunner`,
`PackageInstaller`, `SyntaxCheck`, `ApiProbe` and `SkillScriptRunner` all launch
through it, so there is one launch sequence rather than the two there used to be.

**The in-process option.** A host that cannot start processes at all — the iOS app,
where there is no `posix_spawn`, no `sandbox-exec` and no child to put in one —
supplies its own backend through the `ShellRunner` constructor and
`SkillScriptRunnerOptions.Backend`, and runs the command inside its own process
over embedded interpreters. Three things make that honest rather than a bypass:

* **State is persisted through the session, not a wrapper.** `ShellSession.Load()`
  reads the saved working directory and parses `env.sh` (the `declare -x NAME="…"`
  and `export NAME='…'` shapes `export -p` emits); `ShellSession.Save(cwd, env)`
  writes both back through the same filter the wrapper pipes `export -p` through,
  atomically; `MarkEnvironmentReset()` writes the marker the result reads. So `cd`,
  `export`, "Working directory is now …" and the reset note behave identically
  whichever backend ran the command.
* **Confinement is what the backend's `Sandbox` says it is.** `InProcessSandbox`
  wraps nothing; it carries the capabilities the host asserts (writes confined to the
  workspace by path checks, sockets disabled in the interpreters, no home directory,
  nothing outlives the process). `CanRun`, the declaration's network promise and the
  result's `Not confined on this host:` line are all read from it, so a backend that
  confines writes and the network satisfies `required` without
  `--code-exec-unconfined` — and one that does not is refused, with the gap named.
* **The declaration describes the embedded runtime.** `CodeEnvironment.Configure`
  replaces the PATH probes — the "On this host:" tool list, interpreter resolution and
  Python version lookup — so the prompt and the coaching name what is actually there.
  `ShellProgram.InProcess("sh")` is the dialect such a backend presents.

On iOS `SkillSandboxFactory.Detect()` yields an `InProcessSandbox` claiming writes,
home reads and the process tree but **not** the network, so the app has to present
its own sandbox object (via the constructor) once its interpreters have sockets
disabled; `SpawnedProcess.TryStart` answers "this platform cannot start programs"
there rather than reaching the kernel.

## Model-family caveats

Skills are delivered differently depending on what a family's chat format can
actually carry. `SkillCapabilities.For(architecture)` works this out from the chat
protocol registry, so a new family with an unusual renderer gets it right for free.

| Family | Tool declarations rendered? | `role: "tool"` rendered? | What happens |
|---|---|---|---|
| Qwen 3.5 / 3.6, Gemma 4, GPT OSS, Nemotron-H, Muse-Glimmer, DeepSeek V4, GLM 5.x | yes | yes | Full progressive disclosure |
| **Mistral 3** | no | **no** | No tools are offered; selected skill bodies are written into the prompt up front, and any tool result the loop does produce is fed back as a `user` turn rather than a `tool` turn |
| **Any family nothing can parse** — `qwen4exp`, and every architecture with no registry entry at all | withheld | n/a | Selected skill bodies are written into the prompt up front and the catalog is dropped |

That last row is the one worth understanding, because it is the one that was wrong.
Offering a tool is two halves decided in two places: the protocol registry says whether
the renderer *writes* the declaration, and `OutputParserFactory` decides what *reads* the
reply. Nothing structural makes them agree. An architecture with no `CreateOutputParser`
— `qwen4exp` is registered and has none — falls back to `PassthroughOutputParser`, which
returns every byte as content and cannot produce a tool call at all.

Declaring `skills_read` to such a model is strictly worse than staying quiet: the model
emits the call, nothing answers it, and the raw tool markup reaches the user as though it
were the answer, while the disclosure loop never runs. So the capability is the **AND** of
the two halves, and a family that cannot complete the round trip gets its skill bodies
inlined instead — which works. `SkillCapabilityConsistencyTests` walks every registered
architecture and fails if the two halves ever drift apart again.

Mistral 3's renderer takes only the messages and the generation flag — the tool
list is discarded before the renderer sees it, and the request still succeeds.
Without this flag, skills would appear to work and silently do nothing. The prompt wording changes accordingly: with
no tools, telling the model to "call `skills_read`" is an instruction it cannot
follow, so it is told instead that everything available to it is already in
front of it.

Mistral 3's renderer additionally handles only `user` and `assistant` and drops
everything else on the floor, silently, with the request still succeeding. Left
alone, the loop there would ask the model to continue from an answer that is not
in its prompt, watch it call the same tool again, and burn the whole round
budget. Feeding results back as a user turn is not elegant; it is the difference
between working and appearing to work.

## Using skills

### CLI

```bash
# What is registered, where it came from, and any warnings.
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --skills-dir ~/skills --list-skills

# One-shot with a skill selected.
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-4-E4B-it-Q8_0.gguf \
    --backend ggml_metal --skills-dir ~/skills --skill pdf --input prompt.txt

# Two skills, no discovery: the model sees exactly these and nothing else.
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-4-E4B-it-Q8_0.gguf \
    --backend ggml_cuda --skills-dir ~/skills --skill pdf --skill xlsx \
    --skills-no-discovery --input prompt.txt

# Interactive, with script execution enabled (your own machine, your own skills).
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-4-E4B-it-Q8_0.gguf \
    --backend ggml_metal --skills-dir ~/skills --skills-allow-exec -i
```

Inside the REPL, `/skills` lists what is registered and which are active, and
`/skill <name>` toggles one on or off for the session — which resets the
conversation, exactly as `/system` does, because the skills block sits at the
front of the prompt.

### Operator surface

| Flag | Env var | Meaning |
|---|---|---|
| `--skills-dir <path>` (repeatable) | `TS_SKILLS_DIR` (path-separator list) | Directories to scan. Default: a `skills` directory next to the binary, created if missing. |
| `--skill <name>` (repeatable) | — | Select a skill for this run. |
| `--list-skills` | — | Print the registry and exit. |
| `--no-skills` | `TS_NO_SKILLS` (anything but `0`) | Turn the feature off entirely. |
| `--skills-no-discovery` | — | Do not advertise unselected skills to the model. |
| `--skills-allow-exec` | `TS_SKILLS_ALLOW_EXEC` (anything but `0`) | Allow `skills_run`. Off by default. |
| `--skills-sandbox <mode>` | `TS_SKILLS_SANDBOX` | `off`, `preferred`, or the default `required`. |
| `--skills-allow-network` | `TS_SKILLS_ALLOW_NETWORK` | Let bundled `skills_run` scripts reach the network. Off by default; separate from code execution. |
| `--skills-max-rounds <n>` | `TS_SKILLS_MAX_ROUNDS` | Skill lookups — and shell commands — per turn, 1–64. Default 8, or 24 with `--code-exec`, where one fix is a write, a run, a read of the traceback and a patch. |

`TensorSharp.Server` accepts exactly the same spellings, and a config-file key
*is* a CLI flag (`"skills-dir": ["/srv/skills"]`), so one config file drives
either host. A root that does not exist is a startup error naming the flag —
a mistyped path fails before a model loads, not on the first request. Roots are
scanned in precedence order with the install directory first, and a name
collision between two roots is reported as an error rather than resolved by
renaming: the name is what a user types and what a `SKILL.md` cross-references,
and it must not change when an unrelated root is added.

### HTTP — chat

Every chat surface takes the same two optional fields:

```json
"skills": ["pdf", "xlsx"],
"skills_discovery": true
```

`skills_discovery` defaults to `true`; `false` restricts the request to exactly
the skills it named.

```bash
# OpenAI-compatible
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-Q8_0.gguf",
    "messages": [{"role": "user", "content": "Pull the totals table out of the attached statement."}],
    "skills": ["pdf"],
    "max_tokens": 600
  }'

# Ollama-compatible
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-Q8_0.gguf",
    "messages": [{"role": "user", "content": "Build me a budget spreadsheet."}],
    "skills": ["xlsx"],
    "skills_discovery": false,
    "stream": false
  }'

# Web UI SSE
curl -N -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Summarize this PDF."}], "skills": ["pdf"], "maxTokens": 400}'
```

The response is an ordinary completion. Any `skills_read` the model performed
happened server-side; the client never sees a tool call it cannot service. The
Web UI stream additionally carries skill completion metadata; the UI renders
transient current activity rather than retaining a durable execution trace:

```
data: {"skill_step":"skills_read","skill":"pdf","detail":"references/forms.md","ok":true}
```

With `--code-exec`, `tool_progress` frames report `writing`, `running`, and
`finished`; the finished event may include generated files. The server retains
bounded artifacts for `GET /api/code/artifacts/{runId}` and
`GET /api/code/artifacts/{runId}/{*path}`, served as downloads with
`X-Content-Type-Options: nosniff`.

### HTTP — managing skills

```bash
# OpenAI-shaped listing
curl http://localhost:5000/v1/skills
curl http://localhost:5000/v1/skills/pdf          # adds "instructions": the SKILL.md body

# Web UI shape, with load errors included
curl http://localhost:5000/api/skills
curl http://localhost:5000/api/skills/pdf

# Install from a ZIP (of the skill folder, or of its contents)
curl -X POST http://localhost:5000/api/skills -F "file=@pdf.zip" -F "overwrite=true"

# Remove an installed skill
curl -X DELETE http://localhost:5000/api/skills/pdf
# {"removed":true}
```

A skill object:

```json
{
  "id": "pdf",
  "object": "skill",
  "name": "pdf",
  "description": "Extract text and tables from PDF files...",
  "license": "Apache-2.0",
  "compatibility": "Requires python3 with pypdf installed.",
  "files": [
    {"path": "scripts/extract_tables.py", "bytes": 4021, "kind": "script", "text": true},
    {"path": "references/forms.md", "bytes": 18233, "kind": "reference", "text": true}
  ],
  "bytes": 41288,
  "origin": "installed",
  "warnings": [],
  "modified": "2026-08-29T12:00:00Z"
}
```

`origin` is `discovered` for a skill found by scanning an operator-configured
root and `installed` for one uploaded at runtime; only the latter can be
deleted through the API. `/api/skills` wraps the list as
`{"enabled":…,"installable":…,"skills":[…],"errors":[{"path","message"}]}`, so a
management UI can show the directories that *looked* like skills and did not
load, with the reason.

Errors follow the server's per-prefix convention: `/v1/*` returns
`{"error":{"message":"…","type":"invalid_request_error"}}` and everything else
returns `{"error":"…"}`.

### The Web UI

`GET /api/models` grows a nullable block:

```json
"skills": { "enabled": true, "installable": true, "allowScripts": false, "count": 7 }
```

`null` means this build or this deployment has no skills, and the UI hides the
control entirely rather than showing an empty picker.

### The C# API

`SkillsChatClient` is how a .NET application gets skills, and it covers the two
situations that actually arise.

**Against TensorSharp.Server** — name the skills and the server does everything,
including the disclosure loop, next to the model. Nothing is uploaded per
request and the skill files never leave the server.

```csharp
using TensorSharp.Runtime;
using TensorSharp.AgentHost.Skills;

using var client = new SkillsChatClient(new SkillsChatClientOptions
{
    Endpoint = "http://localhost:5000",
    DefaultModel = "gemma-4-E4B-it-Q8_0.gguf",
    Delivery = SkillDelivery.Server,
});

SkillsChatResponse reply = await client.CompleteAsync(
    SkillsChatRequest.User("Extract the tables from report.pdf", "pdf"));

Console.WriteLine(reply.Content);
```

**Against any other OpenAI-compatible endpoint** — point the client at a local
`SkillRegistry` and it builds the prompt block, declares the skill tools and runs
the loop *in this process*, so an endpoint that has never heard of skills behaves
as though it had. The cost is one extra round trip per file the model reads.

```csharp
var registry = new SkillRegistry(new SkillRegistryOptions
{
    Roots = new[] { "/srv/skills" },
});

using var client = new SkillsChatClient(new SkillsChatClientOptions
{
    Endpoint = "https://api.example.com/v1",
    ApiKey = Environment.GetEnvironmentVariable("EXAMPLE_API_KEY"),
    DefaultModel = "some-hosted-model",
    Delivery = SkillDelivery.Local,
    Registry = registry,
    Discovery = true,
});

var reply = await client.CompleteAsync(new SkillsChatRequest
{
    Messages = { new ChatMessage { Role = "user", Content = "Fill in this AcroForm and tell me what you set." } },
    Skills = { "pdf" },
    MaxTokens = 800,
});

foreach (SkillToolInvocation call in reply.SkillInvocations)
    Console.WriteLine($"round {call.Round}: {call.Tool} {call.SkillId}/{call.ResourcePath} ok={call.Ok}");
```

`SkillDelivery.Auto` (the default) probes `/v1/skills` once and caches the
answer for the client's lifetime: server delivery if the endpoint answers, local
delivery if it does not and this client has a registry.

`SkillsChatResponse.ToolCalls` holds calls to the **caller's** own tools, which
the client never executes — service them and send the results back in a
follow-up request. `Rounds` is 1 when the model answered without reading
anything.

Two behaviours worth knowing before you build on it. The client is
**non-streaming** — `CompleteAsync` is the only completion method, because it
returns a whole `SkillsChatResponse` including the transcript and the round
count, which only exist once the loop has finished. This is a property of the C#
client, not of the server: the HTTP surfaces stream (see
[Streaming](#streaming)), and a caller that wants tokens as they arrive should
use `stream: true` against the endpoint directly. And under `SkillDelivery.Local` against a TensorSharp server, the client sends
`"skills": []` and `"skills_discovery": false` so the server's own machinery
stands down; without that both sides would inject a catalog and both would answer
the model's reads, and the transcript the client returns would be missing the
fetches the server made. The suppression is sent only when the one-time probe
showed the endpoint understands those fields, because some OpenAI-compatible
servers reject request fields they do not recognise.

## Writing a good skill

**The description is the whole trigger.** It is the only part of a skill that is
always in front of the model, and the model's decision to load the skill is made
from it alone. Write what the skill does *and when to use it*, in concrete terms
a user's phrasing would match: "Extract text and tables from PDF files, fill in
PDF forms, and merge or split documents. Use when the user provides a PDF or
asks to produce one" fires; "PDF utilities" does not. Name the file types, the
tools and the verbs. Keep it inside 1024 characters, which is both the
specification's ceiling and what TensorSharp renders, so nothing you write within
the limit is cut. Put the trigger conditions early anyway — the description is the
*only* thing the model sees before it decides whether to load your instructions.

**Keep `SKILL.md` under about 500 lines.** It is loaded whole the moment the
model activates the skill, and from then on it is paid for on every remaining turn
of that conversation. If it is growing past that, the material that grew is almost
always reference material.

**Move detail into `references/`.** A file the model opens only when the task
needs it costs nothing the rest of the time — that is the entire economics of
progressive disclosure. `SKILL.md` should say *what to do* and point at the
reference for the details: "for the form-field API, read `references/forms.md`".
The file index is already in front of the model, so the pointer is cheap.

**Ship scripts rather than pasting code.** A 200-line Python function in
`SKILL.md` is 200 lines of context on every turn and a fresh chance for the model
to retype it wrong. The same file under `scripts/` is one line in the index, is
read only when needed, and — where the operator has enabled `skills_run` — can be
executed rather than transcribed. The injected instructions already tell the
model to prefer what a skill ships over rewriting its logic.

**Write paths relative to the skill.** `scripts/extract.py`, not
`~/skills/pdf/scripts/extract.py`. Relative paths are what `skills_read` takes,
and an absolute one is rejected by the path guard.

**Name the directory what you named the skill.** `name` must match it; anything
else loads with a warning, and the name the user has to type is the directory's.

**Say what the environment needs in `compatibility`.** It is shown to the model
in `skills_list` as `requires:`, so a skill that needs `pdfplumber` can be
skipped intelligently rather than attempted and failed.

## Does it actually work?

Graded end-to-end, on `gemma-4-E4B-it-Q8_0` / `ggml_metal`, against tasks whose
answers exist **only** inside a skill — two synthetic skills with deliberately
unguessable rules (an invoice format with a mod-97 check digit, a deployment
procedure with a 7% canary and a 45-minute soak) and one real published skill
(`slack-gif-creator`). Three arms per task:

| Arm | What the model was given | Result |
|---|---|---|
| control | no skills at all | **0 / 5** |
| selected | the skill named in `skills: [...]` | **5 / 5** |
| discovery | a metadata-only catalogue; the model had to fetch | **5 / 5** |

The control failing every task is the point: it establishes that the questions
are genuinely unanswerable without the skill, so the other two arms measure the
feature rather than the model's prior knowledge. A negative control — an
unrelated question asked with skills loaded — passed in all three arms and did
**not** leak skill content into the answer, so the block does not distort
unrelated turns.

The deeper tiers were checked separately, because inlining a `SKILL.md` is the
easy half:

- **A bundled reference file.** A skill whose `SKILL.md` says "read
  `references/windows.md`, do not guess" and whose numbers live only in that
  file. The model followed the pointer and returned all three exactly (83 days,
  11 hours, 412 m/s).
- **A bundled script.** A skill whose fuel-budget figure can only be obtained by
  running `scripts/budget.py`. The model called `skills_run`, the sandbox
  executed it, and it reported **1238.21 kg** — the script's exact output.

That last one found a real bug worth recording. The `args` parameter is declared
as a string, because `ToolParameter` cannot express an array; the model passed an
array anyway, because that is what an argument list looks like. The read returned
nothing, the model gave up on the tool and did the arithmetic in its head, and
produced **1235.89** — a wrong number that looked entirely plausible. Accepting
both shapes fixed it. The lesson generalises: a tool declaration is a hint, not a
contract the model is bound by, so the reading end has to accept what models
actually emit.

## Compared with the reference implementations

TensorSharp targets the same specification as Claude Code and OpenAI's Codex, and
a skill written for either loads here unchanged. `SkillSpecConformanceTests` pins
that: the specification's limits (name ≤ 64, description ≤ 1024, compatibility ≤
500, the name grammar and its worked examples), the three disclosure tiers, the
"any file or directory is allowed" rule, and — where the spec is silent — the
values Codex uses, so the two stay comparable.

| | Codex | TensorSharp |
|---|---|---|
| Tool names | `skills.list` / `skills.read` | `skills_list` / `skills_read` |
| Token estimate for budgeting | 4 bytes/token | 4 bytes/token |
| Catalogue description cap | 1024 chars, `...` suffix | same |
| Long file delivery | paginated, `next_cursor` | paginated, byte offset |
| Tool schemas | generated JSON Schema, nesting allowed | flat scalars only |
| Script execution | the harness's shell tool, cleared per command by a human or a policy | a shell tool too, cleared once by the operator at startup; then sandboxed or refused |

Three differences are deliberate and asserted as such:

- **Underscores, not a dotted namespace.** Several chat templates splice a tool's
  name into their markup unescaped — Gemma 4 writes
  `<|tool>declaration:{name}{`, GLM writes `<tool_call>{name}<arg_key>`, Harmony
  writes `type {name} =`. A dot is not safe in that position across all eleven
  protocols. The capability is identical; only the spelling differs.
- **Flat tool parameters.** `ToolParameter` carries `{Type, Description, Enum}`
  and nothing else; `items` and nested `properties` are dropped when a tool is
  parsed and cannot be re-emitted. So every parameter is a scalar — and the
  reading end compensates, as the `args` bug above shows it must.
- **Nobody to ask, so the sandbox is mandatory.** TensorSharp has a general shell
  tool as well — `--code-exec` declares the file tools, `shell` and `apply_patch`, and a skill's
  own scripts run through the same confinement. What it does not have is the
  thing Codex and Claude Code lean on: a person at a terminal who is asked before
  a command runs, or who wrote a policy saying which ones not to ask about. A
  server answering an HTTP request has nobody to ask, and asking the *model*
  whether its own command is safe is not a control. So the trade is made once, by
  the operator at startup (`--code-exec`, `--skills-allow-exec`), and after that
  the confinement is not advisory: on a host that cannot confine a process the
  tool refuses rather than asking. `--code-exec-unconfined` is the deliberate
  way out accepted by both hosts and required on Windows, where it openly leaves
  filesystem and network access unconfined; it should not be enabled on a server
  reachable by untrusted users. Network access is a second operator decision:
  commands are offline by default, while `--code-exec-allow-network` gives every
  generated command unrestricted host IP-network access without removing macOS/Linux
  write or home-read confinement. Linux additionally bounds descendants with a PID
  namespace. On macOS, children inherit Seatbelt and ordinary process groups are
  stopped, but a deliberately detached child can outlive the request; every result
  reports that gap. macOS denies common
  `/private/tmp/com.apple.launchd*` pathname sockets while permitting runtime-required
  Mach lookup and the exact mDNSResponder pathname socket required for DNS, and retains
  shared-temporary-directory Unix IPC for compatibility; Linux hides common
  `/run` endpoints, but its host network namespace may expose abstract sockets and
  pathname sockets outside `/run`. Local Unix IPC is therefore not a complete
  isolation boundary. A package install remains separate and happens only
  because the host reads the request out of that command and performs it itself.

What has **not** been compared is live output against the OpenAI or Anthropic
APIs; no API key was available in the environment this was built in. The
comparison above is against the specification and against Codex's source, which
is checkable and does not drift. Running the same graded tasks against
`gpt-*` or `claude-*` through `SkillsChatClient` in `SkillDelivery.Local` mode
would take one key and no code changes.

## Where to get skills

Anthropic publishes a set of open-source skills at
**<https://github.com/anthropics/skills>** — document handling (`pdf`, `docx`,
`xlsx`, `pptx`), artifact building, canvas design, MCP builders, and a
`template/` starting point.

Point a root at the clone and every one of them is found; the scanner walks up
to three levels, which covers both `root/<skill>/SKILL.md` and that repository's
own `skills/<skill>/SKILL.md` layout:

```bash
git clone https://github.com/anthropics/skills.git ~/skills
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --skills-dir ~/skills --list-skills
```

All 19 published skills plus the template load with **zero errors**. Two carry
warnings, and both are real spec violations upstream rather than parser
pedantry:

* `claude-api`'s description is **1068 characters**, against the specification's
  1024-character limit;
* `template/` declares `name: template-skill`, which does not match its
  directory.

Both load and both work — which is precisely the behaviour the forgiving reader
exists for.
