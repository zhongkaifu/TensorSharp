# Fixed-history reset regression probe

This probe tests whether identical token histories produce identical logits after
`ResetKVCache`, across prompt shapes, and after recreating a model. It does not
score generated prose and does not assert that a captured logit row is a
mathematically correct model oracle.

The included `qwen36-moe-msi.json` fixture contains exact 32/512-token prompts and
32 fixed continuation tokens, derived from the original MSI baseline matrix. The
tokens define inputs only. Its single-file GGUF SHA256, byte count, vocabulary
size and context are checked before model execution. There are no model weights
in this directory and the tool does not download any.

Split GGUF fixtures additionally provide an ordered `ModelFiles` array of
`FileName` (a sibling filename), `Bytes`, and full-payload `Sha256`. The probe
opens the same GGUF headers as the model loader, requires the fixture to cover
every discovered file in loader order, checks all lengths, then hashes every
file before model creation. `--model` must point to the first, metadata-bearing
shard. A metadata-only first shard cannot pass as a complete single-file model.
`ModelBytes` is the sum of every shard. For a fixture with `ModelFiles`,
`ModelSha256` is the uppercase SHA256 of UTF-8 text with this exact format:

```text
ResetLogitProbe model-files v1\n
<FileName>\t<Bytes in decimal>\t<uppercase payload SHA256>\n
...one line per file, in ModelFiles order...
```

Here `\n` and `\t` denote actual newline and tab characters. Paths and platform
line endings are not included. `metrics.json` retains the scheme, aggregate,
individual absolute paths, lengths, full hashes and hash timestamps. Fixtures
without `ModelFiles` keep their original single-file hash meaning. Locally
recorded hashes establish repeatable input identity; they do not establish the
checkpoint's publisher provenance. Each invocation rehashes the complete set;
exclude this preparation from inference timing and account for its page-cache
effect when comparing processes.

## Build and native identity

Build the matching native backend separately against unchanged upstream ggml.
Build this managed project with the same native-skip flags used by other probes:

```powershell
dotnet build eng/ResetLogitProbe/ResetLogitProbe.csproj -c Release -m:1 -p:UseSharedCompilation=false -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -p:TensorSharpSkipCpuNative=true -o artifacts/reset-probe-app
```

Put the intended `GgmlOps.dll` (or platform-native `libGgmlOps`) in that app
directory. Pass its SHA256 through `--native-sha256`. The probe also checks the
actual loaded module path and records the managed Models assembly hash. The
compiled probe assembly hash identifies the validation code that produced each
capture. Capture
stdout/stderr externally so native CUDA initialization and execution-path
diagnostics remain available. Do not run alongside another GPU job.

## Minimal discriminating sequence

Set `TENSORSHARP_MODEL_ROOT` to your directory containing the selected model.
Use the same settings for every process. For the original MSI host-MoE case:

```powershell
$env:TS_CPU_MOE='1'
$env:TS_CPU_MOE_THREADS='0'
$env:TS_GGML_CPU_THREADS='8'
$env:KV_CACHE_DTYPE='f16'
$env:TS_SPEC='0'
$env:TS_HOST_MOE_METAL_OVERLAP='0'
$env:TENSORSHARP_TP_DEGREE='1'
$env:TENSORSHARP_LAYER_SPLIT_DEGREE='1'
$taskModel=Join-Path $env:TENSORSHARP_MODEL_ROOT 'Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf'
$taskNativeHash=(Get-FileHash artifacts/reset-probe-app/GgmlOps.dll -Algorithm SHA256).Hash
$taskProbe='artifacts/reset-probe-app/ResetLogitProbe.dll'
$taskFixture='eng/ResetLogitProbe/qwen36-moe-msi.json'
```

First capture each case in a separate fresh process. These commands record
evidence, and deliberately do not produce a passing consistency assertion:

```powershell
dotnet $taskProbe --model $taskModel --fixture $taskFixture --native-sha256 $taskNativeHash --output artifacts/reset-fresh32-a --session fresh --sequence p32 --record-only
dotnet $taskProbe --model $taskModel --fixture $taskFixture --native-sha256 $taskNativeHash --output artifacts/reset-fresh512-a --session fresh --sequence p512 --record-only
```

Repeat those fresh-process cases against their first captures. This tests whether
fresh-process references are themselves repeatable before relying on them:

```powershell
dotnet $taskProbe --model $taskModel --fixture $taskFixture --native-sha256 $taskNativeHash --output artifacts/reset-fresh32-b --session fresh --sequence p32 --reference artifacts/reset-fresh32-a/metrics.json
dotnet $taskProbe --model $taskModel --fixture $taskFixture --native-sha256 $taskNativeHash --output artifacts/reset-fresh512-b --session fresh --sequence p512 --reference artifacts/reset-fresh512-a/metrics.json
```

Then compare same-shape and interleaved resets against those references:

```powershell
dotnet $taskProbe --model $taskModel --fixture $taskFixture --native-sha256 $taskNativeHash --output artifacts/reset-reused --session reused --sequence p32,p32,p512,p512,p32 --reference artifacts/reset-fresh32-a/metrics.json --reference artifacts/reset-fresh512-a/metrics.json
```

The default sequence is the same five requests. `--session fresh` creates and
disposes a model for every request in one process; native process-wide caches can
still survive. It is a separate control, not equivalent to fresh processes.
Record model reload and process restart outcomes separately.

## Evidence and strict result

Each request saves every full float32 logit row, not only the selected top token.
The binary layout is little-endian `[step][vocabulary]` with no header. Step 0 is
the prompt's prefill result; step 1 follows the first forced continuation token.
Each array is cloned immediately because `Forward` may reuse its host buffer.
The exact prompt and every forced token are included in `metrics.json`.
`Steps` counts exported rows: a 32-row fixture executes one prefill and 31
single-token forwards. Its final `ForcedTokens` entry labels the last predicted
position and is not fed into another forward. The eight-row Qwen3.8 fixture
therefore executes one prefill and seven single-token forwards per request.

`forwardSeconds` times each synchronous `Forward` call; `prefillSeconds`,
`decodeSeconds`, and `totalForwardSeconds` summarize those calls. Logit cloning,
binary export, comparison, model hashing/loading, tokenization and sampling are
outside these timings. These are fixed-history inference measurements, not HTTP
time-to-first-token or application end-to-end latency. Separate the first request
in a process from later warm requests and interleave baseline/fixed processes.

The metrics include logical cache lengths before reset/after reset/after forward,
full binary SHA256 and size, finite checks, top two logits/margins, and per-row
bit differences, relative L2, RMS, maximum absolute error and cosine. Comparisons
are against the first same-case request and, when supplied, validated external
reference binaries. Every reference request is read and checked, including all
duplicate cases; repeated histories must match bit for bit independently of the
JSON summary flags. A missing, corrupt or unequal duplicate is refused. All
output directories must be new, preserving previous evidence.
`unverifiedCases` lists requested cases that have no comparison yet. For example,
`--sequence p32,p32,p64` without an external reference cannot pass because `p64`
has only one capture. The default five-request sequences compare every case.

Default exit codes:

- `0`: completed, every requested case has a comparison, and every compared logit bit matches.
- `1`: completed but a bit differs, or a requested case remains unverified.
- `2`: incomplete execution or runtime validation error.

`--record-only` returns zero when capture completes; `passed` remains `null`.
It must not be counted as a passing test. No tolerance flag can turn a discrepancy
into a pass. If a backend has unavoidable nondeterminism, report it and establish
the cause instead of increasing a threshold. Exact equality here tests repeated
execution consistency; model accuracy still requires an independent oracle.

The package-free regression project exercises the reference validator using tiny
synthetic float files and checks coverage/result semantics. It loads no model or
native backend:

```powershell
dotnet run --project eng/ResetLogitProbe/tests/ResetLogitProbe.Validation.Tests.csproj -c Release
```

Five requests with this fixture export 158,924,800 bytes (about 152 MiB). Logs and
metrics are additional. Keep generated evidence under ignored `artifacts/` or
`docs/validation/`, and retain the original failing captures.

## Focused regression requirement

The original baseline exhibited two distinguishable failures under fixed inputs:
same-shape repeats matched the prefill row but changed at decode step 1; a short
prompt after the long shape already changed at prefill step 0. A fix must cover
both boundaries, not just preserve a greedy sentence or the first predicted ID.

When the responsible kernel/buffer lifetime is identified, add a small native
regression at that boundary with fixed inputs, poisoned recycled storage, both
prompt shapes and repeated decode. It should fail before the fix and pass after
it without model weights. Keep this real-model replay as integration evidence.
Do not introduce a synthetic test that merely rechecks the reset counter.

## Qwen3.8 Flash Next on the MSI

`qwen38-flash-next-msi.json` uses the three-shard UD-IQ1_M checkpoint, whose GGUF
architecture is `qwen4exp`. It starts with 32/64-token prompts, eight fixed output
positions and context 256. The 248,320-entry tokenizer metadata was compared
against the source input fixture's GGUF before reusing token IDs.
Only the padding ID differs; neither padding ID occurs in these unpadded inputs.
Continuation IDs remain fixed inputs, not expected generated answers. This short
fixture is a bounded reset check; it does not cover the longer 512-token case.

`create-sharded-fixture.py` creates this fixture from a bounded-header inventory,
completed full-payload hash evidence, and an existing fixed-token fixture. It
requires exact equality of `tokenizer.ggml.*` metadata, including every token
string, merge and token type. An unused padding-ID difference is recorded and
allowed because this probe does not pad inputs; every other difference is refused.
It reads only metadata and existing hash evidence, creates a new output file,
and never initializes a model or GPU. Example:

```powershell
python eng/ResetLogitProbe/create-sharded-fixture.py --inventory artifacts/moe-consistency/qwen38-model-inventory.json --payload-hashes artifacts/moe-consistency/qwen38-payload-hashes.json --input-fixture eng/ResetLogitProbe/qwen36-moe-msi.json --input-model $taskModel --output artifacts/qwen38-fixture.json --prompt-lengths 32,64 --steps 8 --context 256
```

For a bounded first run on the MSI's 32 GiB RAM / 16 GiB VRAM, use all host MoE
layers (`TS_CPU_MOE=1`), `TS_HOST_MOE_EXPERT_CACHE_MB=0`, `TS_GGUF_PREFAULT=0`,
`TS_Q4E_RETAINED_CACHE_MB=0`, f16 KV, and disabled speculation. Keep the generic
CPU and host expert thread settings fixed as above. The 28.8 GB PLE table stays
host-mapped and gathers 16 rows per token; it is excluded from CUDA preload.
Expert weights also remain mapped, while about 3.89 GB of non-expert payload
excluding PLE is potentially resident. These payload counts are not peak-memory
predictions: caches, scratch, dequantized tensors, route-dependent file pages and
driver allocations still require a monitored CUDA smoke run before larger tests.

## Windows native and HTTP functional checks

`local-moe-chat.py` uses the existing native IChatClient sample for a streaming
turn followed by a non-streaming follow-up. After that process exits, it starts
its own loopback HTTP server and checks JSON and raw SSE responses sequentially.
It requires prepared application directories and a selected `GgmlOps.dll` SHA256;
it does not build, copy binaries, download weights or hash model payloads.

The script checks both app-local native hashes and each running process's actual
loaded native module. It checks exact loaded model/architecture/backend, a single
expected discovery ID, positive bounded token usage, nonempty text, accepted
finish reasons, and the SSE `[DONE]` terminator. All HTTP bodies, native output and
server logs are retained. Native stdout contains the complete response together
with engine diagnostics; the existing adapter itself enforces nonempty assistant
text. Qwen4Exp's placement log records the actual host/device expert-layer counts.
The HTTP host receives `--cpu-moe` and must confirm all expert layers on the host.
The existing native example does not call `MoeCpuOffloadConfig.ConfigureFromEnvironment`;
its automatic placement is recorded separately (40 host / 8 CUDA layers in the
MSI functional run), despite the inherited `TS_CPU_MOE` setting.
For other architectures the same logs are retained without a placement assertion.

Native context defaults to 1088 because the existing sample requires more than
`1024 + max_tokens`; HTTP context defaults to 256. Both request f16 KV and disable
prefault/speculation/expert cache/state retention. Context and expert-placement
differences are recorded, and these results are
functional checks, not timing or mathematical-correctness comparisons. HTTP text
does not expose logits, so this script makes no logit-finiteness claim. It does
not run the HTTP .NET SDK example, cancellation tests, or vision.

Example after staging the final native and managed binaries (replace the hash
with the selected final native-build manifest's SHA256):

```powershell
$taskQwen38Model=Join-Path $env:TENSORSHARP_MODEL_ROOT 'Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf'
python eng/ResetLogitProbe/local-moe-chat.py --model $taskQwen38Model --native-dll 'artifacts/moe-chat-apps/native/TensorSharp.NativeChat.dll' --server-dll 'artifacts/moe-chat-apps/server/TensorSharp.Server.Host.dll' --native-sha256 '<SELECTED_FINAL_NATIVE_SHA256>' --model-hash-evidence artifacts/moe-consistency/qwen38-payload-hashes.json --out 'artifacts/moe-chat-final'
```

The optional prior hash evidence is rechecked by shard filename, length and
modification time; the report explicitly records that full payload hashes were
not recomputed by this functional run. Choose a new output directory for each
run. Only this script's own processes are stopped, including on failure.
