# Actual agentic acceptance against an owned local server

`unified-agentic-acceptance.py` exercises the real WebUI tool loop. It does not
start a model, execute downloaded model code in the validation process, or
simulate tool results. Python 3 and `psutil` are required. Keep the server idle
and dedicated to this run; unrelated clients invalidate process attribution.

Prepare a fresh ignored fixture:

```powershell
python eng/validation/unified-agentic-acceptance.py prepare --output artifacts/unified-agentic-20261011/fixture-v1
```

Use the generated `plan.json` `server_flags` when starting the qualified server
with its normal checkpoint, projector and `--unified-memory` arguments. The
fixture pins its generated nonce and three input files as well as the known
skill and script. It requires real code execution and skill discovery:

```text
--code-exec --code-exec-timeout 30
--skills-dir <fixture>/skills --skills-allow-exec --skills-max-rounds 16
--agents-max-concurrent 2 --agents-max-count 4 --agents-max-depth 1
--agents-max-rounds 8 --agents-max-generations 12 --agents-timeout 600
--sampling-precedence request
```

On Windows, the current host needs the explicit `--code-exec-unconfined` and
`--skills-sandbox preferred` flags. That configuration does **not** confine
filesystem or network access. The fixture requests only Python standard-library
operations in its session work directory, with no packages, network access or
background commands. The child-agent test uses a read-only explorer, since the
unconfined Windows root shell does not qualify a confined child shell.

Run against an already-owned server. Substitute its actual PID and actual
logger/scratch paths; do not infer the paths from the working directory:

```powershell
python eng/validation/unified-agentic-acceptance.py run `
  --url http://127.0.0.1:52177 --server-pid <owned-pid> `
  --plan artifacts/unified-agentic-20261011/fixture-v1/plan.json `
  --server-log <actual-server-jsonl> --server-scratch <actual-code-scratch> `
  --agent-workspaces "$env:LOCALAPPDATA/TensorSharp/agent-workspaces" `
  --output artifacts/unified-agentic-20261011/run-v1 `
  --max-tokens 1024 --timeout 1800
```

The outer launcher remains responsible for exact server/native/managed hashes,
checkpoint identity, server lifetime and resource sampling. This runner verifies
that the supplied PID owns the listening port, preserves its executable and
creation identity, and requires `GET /api/models` to report unified Qwen4Exp.
Its three independent fresh sessions check:

* `skill`: `skills_list`, `skills_read` and `skills_run` execute the pinned skill.
  A downloaded receipt must carry the run nonce and the actual sum 16. A matching
  host script exit log and process-exit observation are separate requirements.
* `code`: the model writes `square(x) = x + 1`, runs it at 7, reads the source,
  patches it to `x * x`, and runs it again. Both source snapshots are parsed as
  complete Python ASTs, and actual downloaded receipts must report 8 and 49.
  Two successful shell tool invocations and process-exit log records are needed.
* `agents`: one read-only explorer must read the fixture's first FACT, finish,
  receive follow-up input, read its second FACT and finish again. The parent must
  list, close and wait for that same child. Host child tool events, progress
  records and workspace removal qualify lifecycle; parent claims do not.

`--cases skill`, `code` or `agents` can isolate a failure. Such a partial run does
not qualify the omitted scenarios. Each HTTP chat has an absolute deadline and
bounded evidence size. Every scenario requires the final WebUI `done` to be
neither truncated nor aborted, and the last correlated generation log must
record natural EOS. A 160-token introduction or promise to delegate is not a
passing task. All failed raw responses remain in the fresh output directory.

Each persistent session is deleted through the real API. Only that session's
scratch namespace and observed child workspace IDs are inspected. The runner
does not sweep or delete unrelated folders or kill arbitrary PIDs. The host's
child workspace location currently lives under LocalApplicationData; those
transient test-owned directories must be removed by the host. The downloaded
artifact copies and logs intentionally remain in ignored evidence directories.

Process sampling is every 100 ms; very short commands may not appear in its
samples. Actual host invocation/exit logs and nonce-bound receipts independently
establish their execution. Receipt PID checks distinguish later PID reuse.
Logical subagents share the serving model; they are not separate OS processes.
Neither these lifecycle checks nor process RSS constitute complete physical
accounting, an OS hard limit, a global cross-model reclamation test, or a throughput
comparison. Root shell network isolation and mutable child shell execution are
not covered by this fixture.

Run CPU gates independently of any model:

```text
python -m unittest discover -s eng/validation/tests -p test_unified_agentic_acceptance.py -v
```

The CPU suite uses a local HTTP server and the checked-in small skill script.
Its passing results validate the acceptance mechanism, not model behavior.
