# Qwen HTTP latency attribution with Nsight Systems

Use `qwen-nsys-profile.py` on an idle CUDA machine to capture one long-context
HTTP request. It starts a separate server through Nsight's interactive launcher,
waits for `/v1/models`, primes selected fixture steps without collecting, starts
collection for the requested step, exports SQLite, and terminates only its own
profiling session. Loading and priming are excluded. It refuses an already
serving port. The generated `.nsys-rep`, SQLite, logs and summaries stay under
ignored `docs/validation/` or `artifacts/`.

The wrapper targets Nsight Systems 2026 CLI. NVIDIA's
[installation guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/)
and [interactive CLI guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
describe the launcher and collection commands. A standalone CLI package can be
extracted with `dpkg-deb --extract` into a tools directory; this does not run
package installation scripts, change CUDA links or install a driver. Record its
version, source URL and package checksum in the evidence.

Prepare a fixture using `qwen38-agent-replay.py prepare`. Create a private server
spec JSON with the exact benchmark command, its environment overrides and an
optional working directory:

```json
{
  "argv": ["/workspace/dotnet/dotnet", "/workspace/tensorsharp/TensorSharp.Server.Host/bin/Release/net10.0/TensorSharp.Server.Host.dll", "--model", "/workspace/models/model.gguf", "--backend", "ggml_cuda", "--port", "5089"],
  "env": {"TS_PREFIX_CACHE_ISOLATION": "shared", "MAX_CONTEXT": "32768"},
  "cwd": "/workspace/tensorsharp"
}
```

Use the real server arguments, drafter, projector and cache settings from the
comparison. `shared` is only for a trusted single-tenant replay. Do not store
credentials in the spec: the report preserves it for reproducibility.

For a repeated 19K prompt, with step 0 priming the cache:

```bash
python3 eng/validation/qwen-nsys-profile.py \
  --nsys /path/to/nsight-systems/bin/nsys \
  --server-spec docs/validation/profile/server-spec.json \
  --fixture docs/validation/replay/fixture.jsonl \
  --url http://127.0.0.1:5089 \
  --scenario repeat19k --step 1 --prime-steps 0 \
  --max-tokens 1 --output docs/validation/profile/repeat19k
```

Repeat with `--scenario image_replace` for a changed last screenshot and
`--scenario incremental` for the approximately 2.5K-token extension. To inspect
cold work use `--step 0 --prime-steps ''`. Each invocation starts a new server.
Record observed prompt/image/cached token counts; fixture labels are targets,
not evidence that a tokenizer produced exactly that length.

The Linux wrapper records the actual server PID, its `/proc/PID/stat` start time,
the kernel boot ID, and a fresh invocation ID in the `.pid` JSON file before
`exec` replaces the wrapper with the server. It validates that identity before
reading process counters. The Nsight launcher exiting does **not** establish
that the server exited or released its CUDA context.

After collection, `nsys shutdown --kill=none` releases the profiling session;
the wrapper sends SIGTERM only to the independently identified server through a
Linux pidfd and waits for its real exit. After `--server-exit-timeout` (15 seconds
by default), it may send SIGKILL through a newly verified pidfd, then waits up to
`--server-kill-timeout` (5 seconds). It never signals a process group or falls
back to killing a bare numeric PID. Changed identities, unavailable pidfd support,
or unconfirmed exit fail the invocation and leave `complete: false`, even when
the capture/export succeeded. Inspect `server_identity`, `server_cleanup`,
`launcher_exit_code`, and any cleanup errors before running another GPU profile.
Startup and request failures also attempt this scoped cleanup. Use a foreground
server command; this ownership record covers that server PID, not independently
daemonized workers.

`--max-tokens 1` changes both prime and measured requests, records the changed
request hashes, and limits decode contamination. This is a prefill diagnostic,
not an answer-quality or throughput run. Tool parsers may emit no content until
a whole tool call exists, so a one-token image/tool request can have no TTFT;
its end-to-end interval and GPU trace remain diagnostic. Omit this override for
a full request, whose GPU totals also include decode.

`qwen-nsys-summary.py` can reprocess the exported database:

```bash
python3 eng/validation/qwen-nsys-summary.py \
  --sqlite docs/validation/profile/repeat19k.sqlite \
  --request docs/validation/profile/repeat19k.json \
  --output docs/validation/profile/repeat19k-rechecked.json
```

The summary separates GPU kernel categories, top kernel names, GPU memory-copy
bytes/times, the union of GPU-active intervals, CUDA host API durations and OS
runtime waits. Runtime and driver APIs overlap GPU execution and sometimes each
other; never sum them into a supposed total CPU time. Kernel grouping uses name
heuristics and retains names for review. The wrapper records `/proc` user/system
CPU counters over the request; these sum work across all server threads, including
background runtime activity. Minor/major fault counters are recorded before and after
with their differences, to test whether cold anonymous mappings dominate host work.
These are process-wide counts and do not attribute every fault to cache storage.
Host-to-host copies are not individually traced.

Wall time minus GPU-active time is an estimate of host/queue/idle time, not CPU
execution time or a precise TTFT breakdown. The capture includes the whole
request; client clocks are not assumed aligned to CUPTI timestamps. Compare
unprofiled timings separately. Graph-node tracing and optional
`--cpu-sampling process-tree` add overhead; sampling additionally needs Linux
profiling permissions. The default disables sampling/context-switch tracing.
No GPU events or missing model/device coverage must be reported as unavailable,
not as a zero-cost or passing GPU result.

If Nsight is unavailable, `TS_GGML_NODE_PROFILE=1`,
`TS_GGML_NODE_PROFILE_EVERY=1` and `TS_GGML_NODE_PROFILE_TAG=qwen35`
produce per-node diagnostics on participating graph paths. This synchronizes
after each node and disables some optimized paths. It is unsuitable for latency
or throughput qualification.

CPU-only tool tests (no model or GPU):

```bash
python3 -m unittest discover -s eng/validation/tests -p 'test_qwen_nsys*.py' -v
```

The lifecycle tests include real Linux `/proc` CPU processes: delayed exit after
SIGTERM, SIGKILL escalation, an unrelated sibling in the same process group,
stale identities, startup/exec failures, and a fake Nsight launcher that reports
exited while its server remains alive. They need neither Nsight nor a GPU. On
non-Linux hosts these cases are explicitly skipped; portable identity/race tests
still run. Run the complete gate on Linux before relying on lifecycle coverage.
