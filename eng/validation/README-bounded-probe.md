# Owned probe execution and optional telemetry

`run-bounded-probe.py` owns one child process, preserves its exit code and stdout,
and kills that child at a deadline. It is not a shell or process-tree supervisor.
The original invocation remains supported:

```powershell
python eng/validation/run-bounded-probe.py --output artifacts/my-probe --timeout 600 -- dotnet path/to/Probe.dll --model path/to/model.gguf
```

The output directory must be fresh. `execution.json` and `process.log` preserve
completion, failure and timeout separately; exit zero never means model quality
passed. Child arguments follow `--` without shell evaluation.
An interrupted launch or telemetry failure still records the owned child's final
status. `cleanup_errors` and telemetry finish failures force a nonzero runner
exit; they do not replace or erase an earlier probe exit or timeout.

Optional `--telemetry` requires Python `psutil`, `nvidia-smi`, Windows or Linux,
the application's managed directory, and a reviewed lowercase SHA256 pin for the
native library. Set the model's resource/environment controls before launching;
the runner inherits them, and records only relevant backend/context variables.
For example, after choosing a fitting `MAX_CONTEXT` and placement:

```powershell
python eng/validation/run-bounded-probe.py `
  --telemetry `
  --managed-directory benchmarks/AgentTurnBench/bin/Release/net10.0 `
  --expected-native-sha256 <expected-sha256> `
  --output artifacts/owned-agent --timeout 600 -- `
  dotnet benchmarks/AgentTurnBench/bin/Release/net10.0/AgentTurnBench.dll `
  --model path/to/model.gguf --draft-model path/to/draft.gguf `
  --backend ggml_cuda --kv f16 --scenarios spec --spec-candidates auto --spec-only `
  --out artifacts/owned-agent/rows.json
```

The pin checks what the child actually loaded; it does not force a path. Live
module discovery reuses `multimodel-quality-bench.py` (`Get-Process.Modules` on
Windows, `/proc/PID/maps` on Linux). Missing observations cannot pass, and an
observed mismatched hash or multiple native modules rejects the owned child.
Polling stops after the **first verified live observation** to avoid recurrent
PowerShell work on CPU-offloaded models. The observed file is hashed again after
exit; subsequent module replacements or unloads are not continuously monitored.

Before/after hashes cover top-level DLL, deps and runtimeconfig files. They are
on-disk identity checks, not loaded managed-module enumeration or an immutable
snapshot. Models are not rehashed. Keep model/head/media identities separately.
`native-observation.json` captures the actual observed native path and time;
`memory.jsonl` and `execution.json` retain samples and their limits.

Process RSS/private memory and board VRAM are sampled once per second. Missing
metrics remain null, not zero; Linux has no equivalent Windows private counter.
Board usage includes other processes. These are neither allocator peaks nor
physical hard limits, and RSS/private/VRAM cannot be summed as independent owners.
Sampling and startup module-enumeration overhead are unquantified. Wrapper wall
time includes identity work and teardown; use scenario timing fields for model
execution, with their own cold/warm and media-encoding boundaries.

Process success, telemetry identity/completeness and model qualification are
separate. With telemetry enabled, incomplete/mismatched evidence causes a nonzero
runner exit even if the child returned zero. Then run scenario gates, for example
`validate-speculative-rows.py`, and separately require EOS rather than accepting
two identically truncated responses. Greedy parity does not prove semantic quality
or a stable MTP performance improvement.

The initial Gemma 26B owned-v2 evidence used repeated module polling. Its executed
source snapshot is retained with that ignored evidence; its timings must not be
attributed to the reduced-overhead one-observation policy added afterward.

For file-mapped weights, process read counters can miss page-fault I/O.
`observe-process-storage.py --pid <pid> --created <psutil-create-time> --output
artifacts/owned-agent/storage-observation.jsonl` independently records physical
disk counters every five seconds until that process exits or its observation
deadline expires. It checks PID creation identity, never stops the process, and
requires a fresh output file. Resolve disk names to actual volumes separately;
their naming is platform-dependent. Disk deltas include **all processes and OS
activity**, not exclusive model I/O or physical ownership. Late observation does
not recover earlier traffic. Empty/unsupported counters and errors remain in the
evidence instead of being inferred as zero traffic.

The Qwen/Strata runners also reap their directly owned child on interruption and
write `run.json` before launch and after cleanup. Process completion and
`gpu_telemetry_complete` are separate fields. `whole_process_seconds` ends at
execution completion or interruption; `harness_wall_seconds` also includes
cleanup and telemetry shutdown. Neither timer proves matched physical budgets.

For `llama-local-quality-bench.py`, `--timeout` is the total server lifetime,
including loading and every quality request. An independent watchdog stops the
server, and the separately owned quality client has the remaining deadline.
Both are reaped on timeout or interruption. The healthy loopback listener must
belong to that server PID. Teardown is reported separately and can extend past
the execution deadline; terminating a process does not validate native cleanup.

`llama-raw-completion.py --plan <plan.json> --request <request.json> --output
artifacts/<fresh-run> --timeout 300` runs one reviewed `/completion` request with
the exact supplied integer token IDs. It requires a single owned loopback server,
deployment pins and live executable/llama/CUDA module identity. The deadline
includes loading and HTTP; an independent watchdog and bounded cleanup reap the
server. Original request/response bytes, natural EOS IDs, strict semantic results,
memory samples and failures are retained separately. It never renders a chat
template, adds a schema, strips Markdown fences, or retries a failed model run.
The default five-minute deadline may be too short for some large checkpoints;
such a run remains a timeout, not a passing model-quality result.

`qwen38-prefix-logits.py --plan <reviewed-plan.json> --output <fresh-evidence-dir>
--timeout 300` is a separate full-distribution diagnostic. It runs two pinned
TensorSharp processes (cache 0 and 8192 MiB), each consuming the original prompt
once and one teacher-forced scalar token, then one owned llama server requesting
two predictions on that original prompt. The first llama token must equal the
supplied scalar; an enlarged prompt is never substituted. `--prepare-only` checks
the plan without starting model processes. Every completed TS capture must prove
its exact histories, full vocabulary, loaded identities and native cleanup. The
llama response must contain both complete, unique, finite, unclipped pre-sampling
distributions and the original CPU sampler. Its normalized log probabilities are
stored explicitly as F64 values and compared after subtracting a common token's
value; they are not raw logits. A two-token length stop completes this diagnostic
only, with no EOS, semantic, arithmetic-equivalence or throughput pass. The raw
quality runner's strict EOS and JSON gates remain unchanged. Capture HTTP bytes,
partial failures, pins, deadlines and owned-process cleanup remain in the report.
