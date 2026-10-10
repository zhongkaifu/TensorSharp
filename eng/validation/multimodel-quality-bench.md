# Serial model quality and performance coverage

`multimodel-quality-bench.py` starts one production server from an independently
prepared binary directory. Run only one GPU workload at a time, after builds and
downloads finish. It requires Windows or Linux, Python `psutil`, `nvidia-smi`, and .NET.
The server's native library is identified from the actual process module list
(Windows) or `/proc/PID/maps` (Linux) and
compared to the recorded deployment hash. ggml must remain unchanged.

On Linux use `--devices 0,1 --layer-split 2` or `--tp 2` for explicit placement,
and `--threads N` for the container's actual CPU allowance. `--n-cpu-moe N`
selects partial expert offload. Backend overrides use `--env TS_NAME=value` and
are retained in the report. These options configure the engine; they do not
constitute a whole-process RAM or VRAM budget.
All split GGUF siblings are required and identified. To avoid repeatedly reading
hundreds of GiB before every measurement, `--checkpoint-manifest` accepts the
completed report from `download-validated-model.py`: it uses the earlier full
SHA-256 verification and checks current paths/sizes. It explicitly does not
claim another integrity scan; use fresh full hashing if files may have changed.
Linux telemetry includes cgroup usage/limits/events and CPU throttling counters,
in addition to RSS, process I/O and per-card VRAM. Network filesystems and page
cache hits cannot be described as physical local SSD traffic.
The sampler resolves the server PID's controller memberships and mount roots,
including cgroup v1, v2, hybrid and nested mounts. The `memory.current`,
`memory.max` and `memory.peak` report keys normalize v1 byte counters while
retaining their original filenames and visible ancestor observations. v1
`memory.failcnt` is not a v2 OOM-event count. CPU throttle fields preserve their
native units (v1 `throttled_time` is ns, v2 `throttled_usec` is microseconds).
Unknown observations remain errors/missing values rather than zero usage.

```powershell
python eng/validation/multimodel-quality-bench.py `
  --server artifacts/my-validation/bin/TensorSharp.Server.Host.dll `
  --model C:/Works/models/gpt-oss-20b-Q8_0.gguf `
  --cases squares,tool_json,code,long_extract --tool-roundtrip `
  --output artifacts/my-validation/gpt-oss-20b
```

For a supported vision model, also supply `--mmproj`, `--image` pointing to the
existing shared `card-0.png` fixture from `qwen38-parallel-vision.py --prepare`,
and append `image_ocr` to `--cases`. That fixture's expected answer is `4821`.
Keep its hash and projector identity. The Qwen3-VL companion to Qwen-Image is
not currently a registered standalone chat architecture; an attempted load must
remain a failure, not a skipped pass.

The four text checks cover exact square values, a nested tool-selection JSON
object, a pure Python function, and extraction from 60 generated records. Code
is inspected with Python AST and never executed. The code oracle accepts the
requested single `return x*x` implementation, allowing whitespace differences;
it is not a general program-equivalence checker. JSON must have an object-valued
`arguments` field and unique keys. Markdown fences are not removed. Every check
requires a completed answer. Tool roundtrips separately use the existing OpenAI
fixture, including the model's own call ID and an exact receipt; they do not
execute an external weather service or certify a host agent workflow.
`--json-schema` adds a separate constrained request through the supported
OpenAI API. Its success never overrides an unconstrained JSON failure.

The sampler is explicitly greedy, with penalties disabled and seed 17. Prefix
reuse, speculation, skills discovery, and model-selected delegation are disabled.
`think:false` requests the architecture's non-thinking mode; it cannot guarantee
that a checkpoint actually supports switching off reasoning. Preserve reasoning
that consumes the generation limit as an incomplete result. A larger limit is
a separate experiment, not a repair of the earlier response.

Each case runs three times by default. Only the first case's first occurrence is
the process's first request. Report each case's first occurrence separately from
its subsequent two repetitions. Fresh hashing reads the model (manifest reuse
does not), and the server performs its own kernel warmup, so these are not
controlled cold-file measurements. The requested
context and actual runtime/cache capacities in the startup log may differ.
Neither `MAX_CONTEXT` nor `--expert-cache-mb` establishes a whole-process memory
budget. Models outside the explicit adaptive adapters are not thereby enrolled
in the unified memory planner.
For a locally generated vision companion, use `--hash-companion` with the main
checkpoint manifest. This reads and hashes the whole companion on each run,
without inventing a publisher hash; retain its generation provenance separately.
The compact expert cache currently applies to the Qwen4Exp integration. Setting
its quota when testing a different family does not prove that it was used; keep
that quota zero for such tests and verify actual placement in the startup log.

Ollama-compatible responses expose real accumulated prompt/decode forward times.
Prefill throughput excludes media encoding, rendering and waiting. Decode uses
the server API's emitted-token numerator and includes final EOS computation in
its duration; reasoning/framing tokens may be counted even when the parser hides
them. It is not visible-answer speed or a fixed-token kernel benchmark. Report
short-answer decode timings cautiously. Complete HTTP wall time is also retained.
The one-second sampler records process working set/private bytes and board VRAM;
sampled board usage includes the desktop, misses short peaks, and is not an
allocation ledger. An error or unsupported model never counts as tested quality.

The harness refuses occupied ports and preserves fresh evidence directories. It
terminates only its own server. A transport timeout stops the suite before any
other GPU request is launched. Shutdown by process termination is explicitly not
graceful native-owner cleanup validation. Keep separate native/budget tests for
that contract. No quality or throughput result establishes independent numerical
equivalence or matched-resource performance against another engine.

Validation of the checker:

```powershell
python -m unittest discover -s eng/validation/tests -p test_multimodel_quality_bench.py -v
```

`multimodel-quality-report.py --manifest matrix.json --output
docs/validation/my-matrix` produces an HTML/JSON report after recomputing each
raw response's quality and timing, checking prompts, sampler, image hash,
required case/repetition coverage, and actual native identity. Its manifest has
`runs` entries with `label`, absolute `report` path, `cases`, and `repetitions`,
plus optional explanatory `notes` and a `hardware` description. Main matrix entries must use the same binary
deployment; different output limits or supplemental reference runs should be
reported separately. `evidence_validated` does not mean `quality_all_passed`.

`llama-quality-control.py` replays the same strict tasks against an already
running, independently built llama.cpp server. Preserve that server's clean
revision, binary hashes, checkpoint verification, launch command, and telemetry
alongside its responses; this client does not start or identify the server.
For example:

```sh
python eng/validation/llama-quality-control.py --port 52176 \
  --cases squares,tool_json,code,long_extract,image_ocr \
  --image artifacts/vision-fixtures/card-0.png --reasoning-budget 0 \
  --output artifacts/llama-control
```

The optional `--reasoning-budget 0` sends llama.cpp's explicit
`reasoning_budget_tokens` override. Use it only with a server revision supporting
that field, and inspect the returned reasoning and generated token counts:
`enable_thinking=false` alone did not disable GLM reasoning in the tested
revision. Keep earlier runs with different reasoning settings as separate
experiments. Decode throughput uses `predicted_n - 1` forward steps and checks
against `predicted_per_second`, since the first sampled token comes from prefill.
This denominator differs from TensorSharp's emitted-token API; matching the
user prompt does not imply identical templates, generated tokens, or memory
allocations. Unsupported architectures and server load failures remain explicit
coverage gaps.

The related older semantic checker now rejects code fences and violations of
the requested integer/comma-separated format without normalizing them away.
The tool-roundtrip checker also rejects duplicate argument keys before sending
a tool result. Existing saved responses can be revalidated offline; this is not
an additional model execution.

For Linux page-cache controls, `checkpoint-page-cache.py --evict --output
artifacts/cache-before.json <all checkpoint shards>` first measures, then evicts
only the named files and measures their residency again. Stop model processes
using those files before running it. Require zero `resident_after` pages for
each arm if claiming a cleared client cache. This does not clear the storage
server, FUSE daemon or physical device cache; opening files can itself affect
FUSE caching. Keep load and first-request timings separate, since the loader
can populate model pages before the first request.

`deepseek-native-logit-capture.py` accepts `--library`, `--model`,
`--checkpoint-manifest` and `--output`. In a fresh Linux process it captures ten
complete FP32 vocabulary vectors for a fixed sequence of prefill, teacher-forced
decode and reset/refill calls, rejecting nonfinite outputs. Run both settings
against the same library and checkpoint, then require matching token histories,
reset flags, vocabulary sizes and every `.f32` SHA-256. This is numerical
invariance between configurations, not an independent language-quality oracle
or a throughput benchmark. Its test placement is two GPUs, context 4096,
16 threads and automatic CPU expert offload.
Optional `--host-staging-budget-mib 64` attaches the actual native budget ABI
during this correctness run and records peak/remaining host staging credit and
successful detach after model destruction. This is a staging limit, not a
whole-process RAM or GPU cap.

On Linux, `--evict-after-warm` additionally captures a fixed-history replay and
its next-token reference, replays that history again, and evicts only the paused
standalone owner's verified read-only checkpoint mappings between synchronous
forwards. It uses `MADV_DONTNEED` on those mappings before advising the named
files, requires zero resident pages in every shard, then requires the restored
next-token logits to match exactly. The resulting 21 vectors must also match
across configurations. Mapping/range validation failures or incomplete eviction
are failures of coverage, not passing pressure tests. Record the major-fault
counters and native read/bypass telemetry to establish whether preparation
actually resumed; this test does not measure throughput under concurrent load.

The Linux DeepSeek demand-reader experiment uses
`--env TS_DSV4_HOST_EXPERT_READ=0` / `=1` with one identical native binary.
Its residency hints are advisory; test cold first requests and repeated prompts
separately. Keep `TS_DSV41_ENGRAM_WARM=0` in both arms to exclude concurrent
background Engram warming. Record staging bytes and CPU-layer placement from
the server log. A matched prompt/answer does not replace full-logit invariance.

For backend changes, also compare node/split counts at the same token positions
and run both arm orders. Report CPU affinity and NUMA placement experiments
separately from the ordinary scheduler configuration. Do not pool different
native binaries or affinity settings into one throughput median. GPU contexts
may remain visible briefly after process exit; wait for teardown before starting
another timed arm, and fail if the device remains occupied.

Also record the actual I/O worker count and staging allocation after applying
OpenMP placement. A bound submitting thread can present a narrower affinity to
the resource planner than the process had at launch; the DeepSeek `spread` /
`cores` experiment changed its demand-reader pool from 16 workers / 64 MiB to
2 workers / 8 MiB. Such a run changes more than compute-thread placement and
must not be presented as an isolated affinity comparison or pooled with the
ordinary configuration.
