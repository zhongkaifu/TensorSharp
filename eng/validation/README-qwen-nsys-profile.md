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

Before starting the server, the wrapper queries `nsys --version` and
`nsys start --help`. Every collection option it uses must be advertised by that
installed CLI. `--discard-environment=true` is added only when explicitly offered;
CLI packages lacking it remain usable. Failed/timed-out capability queries or
missing mandatory options stop before model launch and retain `capture_status:
NOT_RUN` with `complete: false`. The supported option names and help SHA are
recorded in the report.

Every profiling subprocess, including version/help queries, launch, collection,
cleanup and summary, receives an explicit small process environment. Only PATH,
HOME, temporary-directory/locale names, LD_LIBRARY_PATH, CUDA device/cache names
and DOTNET_ROOT variants are inherited automatically; server-spec `env` overrides
are then applied. Other model settings such as TS_Q35_* must be declared in that
spec. Ambient authentication, proxy and Python-startup variables are not copied
into a captured target environment. `_exec` applies the same filter and retains
only named Nsight/CUDA interposition variables needed across exec, such as
LD_PRELOAD, CUDA_INJECTION64_PATH and NVTX_INJECTION64_PATH. This preserves the
profiler's own injection without inheriting arbitrary ambient variables.
[NVIDIA's guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
describes its launcher/interposition behavior. Actual CUDA-event coverage must
still be verified on the target; an HTTP-successful server alone does not prove
that interposition or CUDA collection worked.

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

The summary separates individually recorded kernel categories, top kernel names,
memory-copy bytes/times, whole-graph spans, CUDA host API durations and OS runtime
waits. `gpu_activity_union_ms` includes `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` intervals
alongside kernels, copies and memsets; overlapping/nested records count each
timestamp once. The `cuda_graph_aggregate` category is inclusive and must not be
added to kernel/copy totals. Whole-graph spans can include internal node gaps,
so this union measures observed workload coverage rather than exact kernel-busy
time or SM utilization. `gpu_detailed_activity_union_ms` retains the separate
kernel/copy/memset-only figure. Runtime and driver APIs overlap device execution
and sometimes each other; never sum them into a supposed total CPU time.
Kernel grouping uses name heuristics and retains names for review.

If graph spans exist but kernel `graphNodeId` values are absent/NULL/zero,
`cuda_graph_trace.aggregate_only` is true and per-kernel graph-body attribution
is explicitly unavailable. The report preserves graph-inclusive durations and
warns that sparse kernel categories cannot explain the graph body. Some positive
node IDs alone do not certify complete graph attribution. Host graph-launch APIs
without device graph/node records also produce a coverage warning. The wrapper records `/proc` user/system
CPU counters over the request; these sum work across all server threads, including
background runtime activity. Minor/major fault counters are recorded before and after
with their differences, to test whether cold anonymous mappings dominate host work.
These are process-wide counts and do not attribute every fault to cache storage.
Host-to-host copies are not individually traced.

Client wall time minus the observed workload-interval union is unclassified
uncovered time. It does not establish host idle, CPU execution or a TTFT
breakdown; aggregate graph spans themselves may contain internal gaps. The
capture includes the whole request, and client clocks/window boundaries are not
assumed aligned to CUPTI timestamps. Compare unprofiled timings separately.
Graph-node tracing and optional `--cpu-sampling process-tree` add overhead;
sampling additionally needs Linux profiling permissions. The default disables
sampling/context-switch tracing. No GPU events or missing model/device coverage
must remain unavailable, never a zero-cost or passing GPU result.

NVIDIA documents `--cuda-graph-trace` under launch/profile, while the interactive
start table does not list it. Graph mode records whole graphs and node mode
records their nodes; node tracing adds overhead. The wrapper keeps its existing
launch placement. Requested settings do not prove actual collection granularity:
inspect graph/node records and Diagnostics Summary. See the
[CLI launch options](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-launch-command-switch-options)
and [CUDA graph trace](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-graph-trace).
[NVIDIA's trace issues](https://docs.nvidia.com/nsight-systems/ReleaseNotes/#cuda-trace-issues)
describe hardware-trace restrictions in VMs/vGPU and possible software fallback;
this alone does not establish the cause of a particular aggregate-only capture.

For controlled follow-up diagnostics, capture the same fixture with TensorSharp
graphs disabled to expose individual body kernels. Separately, a graph-enabled
cold request (`--prime-steps ''`) moves graph construction inside collection and
can test whether node records then appear. Check the installed launch help
before choosing node/trace-method modifiers. These arms change execution or
cache history and are attribution experiments, not equivalent latency or quality
benchmarks; never invent an unsupported start option or treat a hoped-for node
capture as measured coverage.

If Nsight is unavailable, `TS_GGML_NODE_PROFILE=1`,
`TS_GGML_NODE_PROFILE_EVERY=1` and `TS_GGML_NODE_PROFILE_TAG=qwen35`
produce per-node diagnostics on participating graph paths. This synchronizes
after each node and disables some optimized paths. It is unsuitable for latency
or throughput qualification.

CPU-only tool tests (no model or GPU):

```bash
python3 -m unittest discover -s eng/validation/tests -p 'test_qwen_nsys*.py' -v
```

Portable tests cover CLI option detection (including older packages without the
optional discard flag), pre-launch refusal, explicit environment filtering and
retained Nsight injection names. SQLite tests also cover nested/overlapping graph, kernel and transfer spans,
aggregate-only and missing graph-body attribution, partial node visibility, and
invalid/unsupported graph interval schemas. The lifecycle tests include real Linux `/proc` CPU processes: delayed exit after
SIGTERM, SIGKILL escalation, an unrelated sibling in the same process group,
stale identities, startup/exec failures, and a fake Nsight launcher that reports
exited while its server remains alive. They need neither Nsight nor a GPU. On
non-Linux hosts these cases are explicitly skipped; portable identity/race tests
still run. Run the complete gate on Linux before relying on lifecycle coverage.
