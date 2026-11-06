# Multi-node layer splitting and tensor parallelism

Both execution modes use the same peer configuration. `--layer-split N` assigns
whole layers to N local GPUs on each node; `--tp N` shards supported weights
across N local ranks on each node. Node 0 owns tokenization, sampling and output.
Other nodes run `TensorSharp.Cli` as workers. Every node must use the same build,
model bytes, backend, mode, local degree and execution settings.

## Launch a layer pipeline

Start workers and then the driver. Replace the addresses with endpoints reachable
between every pair of nodes. These are TensorSharp peer ports, not the VMs'
SSH login ports. With SSH forwarding, use the corresponding forwarded peer
addresses and keep the tunnels alive until both processes exit. IPv6 endpoints
use `[address]:port`.

```bash
# Set the same context budget on every node; increase after checking GPU headroom.
export MAX_CONTEXT=2048

# Node 1: two local GPUs, worker
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
  --model /models/model.gguf --backend ggml_cuda --layer-split 2 \
  --tp-node-id 1 --tp-peers 10.0.0.10:9500,10.0.0.11:9500

# Node 0: two local GPUs, driver
printf '%s\n' 'Explain why the sky is blue.' > prompt.txt
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
  --model /models/model.gguf --backend ggml_cuda --layer-split 2 \
  --tp-node-id 0 --tp-peers 10.0.0.10:9500,10.0.0.11:9500 \
  --input prompt.txt --chat --max-tokens 128
```

For HTTP serving, replace the driver executable with
`TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll` and omit `--input` and `--chat`.
Workers still use the CLI. The server accepts only node 0 and requires a startup
model; changing a running distributed cluster's model/backend is refused.

Use `--layer-split 1` for one GPU on each of several nodes. With neither placement
mode configured, the default is one local GPU. `--tp-node-id` and `--tp-peers`
without an explicit layer selection retain their tensor-parallel meaning.
`--tp N` and `--layer-split M` cannot both exceed one. Environment equivalents are
`TENSORSHARP_LAYER_SPLIT_DEGREE`, `TENSORSHARP_TP_DEGREE`,
`TENSORSHARP_TP_NODE_ID` and `TENSORSHARP_TP_PEERS`.

For native DeepSeek, Qwen and GLM, select physical CUDA devices with `CUDA_VISIBLE_DEVICES`
before launching each process. The managed dense layer executor also supports
`TENSORSHARP_LAYER_SPLIT_DEVICES` for multi-GPU device ordinals. Model files and
all shards must be available on every node; each stage loads its owned weights.

## Launch tensor parallelism

Use the same worker/driver commands and peer list, replacing `--layer-split 2`
with `--tp 2` on every node. Two nodes then provide four global TP ranks.
Restart both processes when changing modes. The local GPU count must match on
every node; the model's quantization layout must support the global rank count.
For example, Qwen checks complete output-row tiles, while distributed GLM checks
expert quantization-block boundaries before execution.

For Qwen3.8-Flash-Next, point `--model` at the first GGUF shard, for example
`Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`. For GLM-5.3-Flash use the
matching first shard, for example
`GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf`. These filenames identify formats,
not a guarantee that a particular device or context has enough memory. Keep all
sibling shards together on both nodes. DeepSeek V4.1 Flash 1017 uses the same
commands with its first shard; see the [DeepSeek settings](models/deepseek41.md#distributed-routed-moe-tensor-parallelism).

Use the same explicit `--kv-cache-dtype f16 --n-cpu-moe 0` on both nodes for a
matched GPU comparison. Sparse host-mapped tables such as Qwen PLE and DeepSeek
Engram remain host-backed with zero CPU expert offload. GLM distributed TP
requires GPU-resident routed experts and native hyper-connection kernels;
leave `TS_GLM_TP_SHARD` unset and disable tracing. Distributed GLM replicates
attention to preserve its full head geometry, shards routed-expert intermediate
rows across the global ranks and reduces their outputs before hyper-connections.
Attention weights and caches therefore require space on every rank.
Qwen distributed TP retains attention and recurrence on each rank and shards
the FFN output rows. It exchanges larger exact F32 activation gathers to retain
the local arithmetic, so network costs can dominate decode.

## Supported stage executors

| Family | Multi-node layer backend | State and limits |
| --- | --- | --- |
| DeepSeek V4.1 Flash, including 1017 | `ggml_cuda`, `cuda` (layer split only) | Native whole-layer graphs transfer residual streams, delayed gates, sparse selections, candidate masks and newly written shared cache rows. Exact prefill; approximate YOCO, vision and DSpark are unavailable in this mode. The direct-CUDA engine's stages move the same state in a layout-hashed payload, place layers from header bytes, need the driver on the last stage and keep every routed expert in VRAM ([card](models/deepseek41.md#backends)). |
| Qwen3.8 Flash Next (`qwen4exp`) | `ggml_cuda` | Native spans transfer the complete hyper-connection residual. GDN recurrence, convolution, PLE and QSA state stay with their owning layers. Text forwards only. |
| GLM-5.3-Flash (`glm5next` / `glm5-next`) | `ggml_cuda` | Native stages transfer the complete hyper-connection residual. KDA recurrence and pooled DSA/MLA caches stay with their owning layers. Text forwards only; older `glm-dsa` variants remain local. |
| Mistral 3 dense text tower | `ggml_cuda`, `ggml_vulkan` | Shared dense pipeline transfers F32 hidden activations. Each node loads only its layer weights/KV state; the first stage embeds and the last stage computes logits. |

The transport, ownership plan and dense-stage execution are independent of model
names. Architecture descriptors explicitly opt in after implementing their full
boundary state. Other families are refused before GPU allocation or network
connection; support for a local layer split alone does not imply cross-node
support. Backend availability is distinct from measured device/model coverage.

By default node 0, which tokenizes and samples, runs the **last** pipeline
stage and node 1 the first; with more nodes the order continues 1, 2, ..., N-1, 0.
Every worker receives each forward's tokens with its control message, so the
first stage can embed them, while the output projection runs where its logits
are consumed. A full-vocabulary F32 vector (about 1 MB for Qwen3.8 Flash Next)
therefore never crosses the network; only the stage boundary state does. Set
`TENSORSHARP_LAYER_SPLIT_DRIVER_STAGE=first` on every node for the previous
order, in which the last node returns logits to node 0. Nodes must agree; the
setting is part of the startup configuration check.

Placement balances estimated resident bytes while keeping each node's layers
contiguous and assigning at least one layer to every selected GPU. The dense
pipeline accounts for layer weights and embedding/output endpoints; DeepSeek
also accounts for KV state, cross-stage cache replicas and explicitly offloaded
experts. This avoids the large memory imbalance that equal layer counts can
produce in models with differently sized layers. Estimates do not guarantee a
fit on heterogeneous GPUs: runtime buffers and other device allocations still
require headroom. New architecture adapters must account for their own replicated
weights and boundary state.

Distributed requests currently use serialized forward/refill/reset/truncate
controls. Paged multi-sequence batching, KV snapshots and speculative paths that
bypass those controls are disabled for these stage executors. Layer splitting
increases aggregate model capacity. Each token still traverses the stages in
order, so more devices need not make single-request decode faster.

## Adding a model family

`ILayerSplitGroup` supplies ordered, lossless point-to-point transfers independently
of tensor sharding. `LayerSplitPlan` supplies contiguous ownership and weighted
partitioning. Dense adapters can use `ModelBase.ConfigureLayerPlacement` and
`ForwardLayerPipeline`, route weights and KV allocations through the selected
layer allocator, and leave embedding/output work at the endpoints. Mistral 3 is
the complete managed example. Native executors can use the same group through
a callback; DeepSeek demonstrates a boundary containing more than hidden
activations.

`NativeDistributedExecution` supplies a common managed bridge for native
all-reduce and opaque stage callbacks, startup configuration agreement and
network abort. The model retains its delegates through native handle cleanup.
Opaque transfers preserve every bit, including non-float state and partial
final words; they perform no arithmetic or lossy encoding.

An adapter must transfer every state value needed by later stages, retain cache
ownership across refill/reset/truncate, and agree on numerical settings before
loading stage weights. Enable the architecture descriptor only after testing
those operations against an unsplit control. In particular, a family with
cross-layer attention, recurrent state or delayed residuals cannot safely reuse
a hidden-activation-only boundary without accounting for that extra state.

## Transport and tuning

TP and layer splitting share persistent TCP connections with `TCP_NODELAY`,
keepalive, a versioned mesh handshake and mode/degree agreement. Tensor frames
transfer lossless 32-bit words directly into destination buffers; frames up to
64 KiB are written with their header in one write. Stage transfers perform no
floating-point reduction or compression.

Cross-node all-reduce defaults to a **symmetric** exchange: every node sends its
contribution to every peer and sums in node order, `((v0 + v1) + v2) + ...`. Each
node therefore waits one one-way latency instead of the round trip through
node 0 that the `star` algorithm needs, and every node computes bit-identical
results, equal to the star's. Payloads up to `TENSORSHARP_TP_SYMMETRIC_INLINE_BYTES`
are written before receiving; larger ones are written on a separate thread so
that two nodes writing large frames at the same time cannot deadlock. Wire
protocol 3 carries the algorithm and limit in the handshake, so a mismatched
node is refused at connection time.

Incorrect frame lengths/types, incomplete reads, timeouts and configuration
mismatches fail the operation. Transport failures close the mesh so peers cannot
continue consuming a partially read tensor. Every node must run the same wire
protocol version; restart the whole cluster when updating binaries.
Startup also checks the model's execution settings and the resolved warmup
schedule. Use the same `TS_PREFILL_WARMUP` and `TS_PREFILL_WARMUP_LEN` settings on
every node. Mistral selects its default KV precision from the entire checkpoint
before filtering stage weights, so mixed-precision layers keep the same cache
precision on every node.

| Variable | Default | Purpose |
| --- | --- | --- |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS` | `120` | Bound connection setup, inbound accepts and handshakes. |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS` | `300` | Bound socket reads and writes. Include the time a peer spends computing before its next transfer. |
| `TENSORSHARP_TP_OPERATION_TIMEOUT_SECONDS` | Receive timeout (`300` by default) | Deadline for one complete activation, broadcast-data or reduction frame, including heartbeats and partial transfers. Increase for long forwards. Idle worker controls and load barriers are exempt; their socket inactivity timeout still applies. |
| `TENSORSHARP_TP_SOCKET_BUFFER_BYTES` | OS default | Optional send/receive buffer budget, minimum 4096 bytes. Measure the route before overriding OS tuning. |
| `TENSORSHARP_TP_ALLREDUCE` | `symmetric` | Cross-node all-reduce: `symmetric` or `star`. Both give the same bits; every node must use the same value. |
| `TENSORSHARP_TP_SYMMETRIC_INLINE_BYTES` | `32768` | Largest symmetric payload written before receiving. Keep it below the socket buffering between the nodes; every node must use the same value. |
| `TENSORSHARP_LAYER_SPLIT_DRIVER_STAGE` | `last` | Pipeline stage node 0 runs: `last` keeps logits local, `first` restores the previous order. |

### Measured on two SSH-connected nodes (2026-10-28)

Two RunPod pods with three A40s each, reachable only through SSH (about 0.5 ms
TCP round trip), F16 KV, no CPU expert offload, 512-token random prefill and
128 decoded tokens, medians of runs 2-4. "Previous" is the earlier transport
(star all-reduce, logits returned from the last node) over an `ssh -N` chacha20
forward; "current" is the symmetric all-reduce, driver-last stage order and the
interactive aes128-gcm forward. Same-day runs on the same pods and weights:

| Model, mode (2 nodes x 2 GPUs) | Prefill tok/s previous -> current | Decode tok/s previous -> current |
| --- | --- | --- |
| Qwen3.8 Flash Next UD-Q4_K_XL, layer split | 838 -> 834 | 26.8 -> 39.5 |
| Qwen3.8 Flash Next, layer split, 2048-token prefill | 1157 -> 1222 | 22.9 -> 33.4 |
| Qwen3.8 Flash Next, tensor parallel (median of 3 -> 7 sets) | 68.2 -> 107.3 | 3.7 -> 8.2 |
| GLM-5.3-Flash UD-IQ3_XXS, layer split | 398 -> 410 | 24.6 -> 33.1 |
| GLM-5.3-Flash, tensor parallel | 43.1 -> 71.5 | 2.7 -> 6.0 |

Tensor-parallel results over this tunnel vary from set to set (Qwen decode
7.9-10.5 tok/s across the seven current sets, 3.7-3.8 across the three previous
ones), so that row reports the median of the per-set medians. Layer split with
three GPUs per node (Qwen, 874 / 38.6 tok/s) adds capacity, not single-stream
speed. Tensor parallelism stays far below layer split on a
route like this: every decoded token still needs one cross-node collective per
sharded projection (96 for Qwen, 84 for GLM). Distributed layer split and tensor
parallelism produced bit-identical full-vocabulary logits for both models on a
3,400-token teacher-forced probe. These figures qualify this route only.

DeepSeek V4.1 Flash 1017 (UD-IQ2_XXS, 300.5 GB) was measured on a second pair of
pods (two A40s each, 100 GB container limit, network model storage) with
`--n-cpu-moe 0`: layer split 163-179 prefill / 25.2-25.9 decode tok/s at 2,048
random tokens, TP 56-58 / 9.4-9.7 with exact prefill (139 / 9.2 with the default
YOCO replay). Cold host-mapped Engram reads, not the network, bound layer split
there; see [DeepSeek V4.1](models/deepseek41.md#current-limits-and-tensor-parallel-work).
Both placements give bit-identical logits, also on a 23,357-token probe.

Peer sockets have no built-in authentication or encryption. Use trusted private
networking or authenticated tunnels.

When only SSH reaches the peers, `eng/validation/multinode-ssh-tunnel.sh` opens the
forward from node 0. It runs the forward inside an interactive session
(`ssh -tt ... sleep infinity`) with `ObscureKeystrokeTiming=no`. OpenSSH sets
`TCP_NODELAY` on its own connection only for interactive sessions, at both
ends; a forward-only `ssh -N` session can hold small frames behind Nagle's
algorithm, and OpenSSH 9.5+ otherwise quantizes interactive traffic to 20 ms.
Every decode collective is a small frame, so this choice matters more than the
cipher for tensor parallelism; the cipher matters for large prefill transfers. A tunnel's latency and encryption cost are
part of the measured deployment; results on it do not qualify RDMA, GPUDirect or
a low-latency fabric. Compare small decode collectives and large prefill payloads
separately using `eng/validation/DistributedTransportProbe`.

## Reproducible validation

Keep ggml upstream unchanged. Record its commit and clean status, both nodes'
source/binary hashes, topology, RAM limits, model revision and full shard hashes,
route, execution settings and competing workloads. Verify weights before timing.
Use `eng/validation/capture-multinode-environment.py` for host evidence.

`eng/validation/DeepSeek41DistributedProbe` accepts either placement mode and
records complete finite logits for fixed-token full/chunked prefill, decode,
refill and resets. Compare local and distributed runs with
`eng/validation/compare-deepseek41-distributed.py`. The native diagnostic
`eng/validation/deepseek-layer-native-probe.py` separately isolates native stage
state from managed transport. Synthetic fixtures test numerical plumbing; they
do not establish checkpoint language quality.

The same managed probe accepts Qwen and GLM through the public model factory.
Generate deterministic mixed-attention fixtures with
[`QwenDistributedFixture`](../eng/validation/QwenDistributedFixture/README.md)
and [`GlmDistributedFixture`](../eng/validation/GlmDistributedFixture/README.md).
Their instructions cover unsplit controls, stage ownership and global TP ranks.
For real sparse-attention checkpoints, also compare a sufficiently long token
stream to activate sparse selection; a short prompt does not exercise that path.

Tools for running such a campaign live in `eng/validation/`:

- `run-multinode-case.py` drives one case (CLI benchmark, numeric probe or HTTP
  server) over SSH: workers first, then the driver, with runtime hashes, exit
  codes and parsed metrics in `summary.json`. `--collect-exclude '*.f32'` leaves
  large probe vectors on the nodes and records their hashes. `--kind suite`
  runs many benchmark rows (`--suite-workload`, `--suite-file`) in one process
  per node, so the model loads once. A cold row drops the model's host-mapped
  tables (DeepSeek V4.1's Engram) from every node's page cache inside the
  processes, and checks with `mincore` that they left (Linux only).
  `summarize-bench-suite.py` turns the driver's log and `bench-suite.json` into
  `suite-summary.json`; a cold row that is not verified on every node fails.
  `suites/dsv41-1017-perf.txt` is the DeepSeek V4.1 perf matrix as rows.
- `compare-distributed-quality.py` scores two probe runs by next-token
  perplexity, KL divergence and top-1 agreement, and flags a run that disagrees
  with itself across its prefill/chunk/refill phases. Use it where two placements
  are not expected to be bit-identical; compare bit-identical ones exactly.
- `run-http-quality-suite.sh` runs the answer, structured-output and tool-call
  HTTP checks; pass it as the orchestrator's `--server-command`.
- `stage-gguf-local.py` copies only the tensors a node reads (`--layers` for a
  layer-split stage, `--rows` for tensor-parallel expert slices) from the pinned
  Hugging Face files into same-size sparse local files, verifying every shard's
  SHA-256 while streaming. Use it when model storage is a slow network
  filesystem; `--plan` previews the selection.
- `warm-gguf-tensors.py` reads host-mapped tables (Qwen PLE, DeepSeek per-layer
  embeddings) into the page cache before timing.

For the full 1017 checkpoint, also run text/JSON/tool workflows and sustained
decode with matched prompts, contexts, precision and token budgets. Exclude
warmup, retain at least three measured repetitions and compare matched local
controls. Report failures, skipped configurations, host paging and short-output
limits. Generated logs/reports belong in ignored `artifacts/` or
`docs/validation/`; reusable tools belong in `eng/`.
