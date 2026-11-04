# Original BF16 attention at T8/B1

This isolated tool qualifies the module-only fixed T8/S4 seam. It does not enable
an engine route. Both native images remain unchanged: candidate `4ba17472…`,
legacy `b6754c46…`. The frozen R20 independent reference and device helpers are
linked without edits. The helper's nonsplit launcher is unused: the legacy arm
always runs the actual R12 128-thread split kernel and its combine.

```sh
dotnet build eng/validation/NinferSmallAttentionProbe/NinferSmallAttentionProbe.csproj -c Release
dotnet eng/validation/NinferSmallAttentionProbe/bin/Release/net10.0/NinferSmallAttentionProbe.dll --self-test
```

Explicit execution, by the GPU owner only:

```sh
dotnet /ABS/NinferSmallAttentionProbe.dll --execute \
  --module /ABS/tensorsharp_ninfer_prefill_attention.fatbin \
  --expect-sha256 4ba17472c196242b88adfe27c4446c136abae8c57225593e1eac16acfe6d4602 \
  --legacy-module /ABS/tensorsharp_q35_ninfer_attention.fatbin \
  --expect-legacy-sha256 b6754c46a8b0888eb6760a5027ae91a6f971e7b54e72adeb8ba98945ea37abc0 \
  --out /ABS/artifacts/FRESH-small-attention
```

Mandatory coverage is eight fixtures at keys74/128/202/512/1024/4096 plus
causal64/256-boundary sentinels, two module lifetimes and three poison passes.
Every BF16 core output and every F32 partial matches the unchanged original
contiguous64-page kernel bit for bit. Candidate and legacy gated outputs must
each meet the unchanged scoped FP64 envelope, including exact sentinel controls.
Their arithmetic can validly differ; they are not compared bit for bit.

The graph check captures all four candidate kernels once per lifetime. It reuses
the same device addresses while updating Q/gates, the external physical base
position and the pointer table across bases57/249/255/1017. Workspace is poisoned
before each replay. Captured-node inspection verifies external position argument4,
Pack's unused CPUbase0/fixed640 pages and all four launch geometries. Each replay
must match same-input uncaptured output/query/partials and the original reference.
There are 624 actual kernel executions; the eight captured nodes do not execute
during capture. Source inputs are reconstructable from their seven fixture SHAs;
only full outputs and any failed partials are retained in a bounded byte-checked
CAS. The final certificate is written once after resource disposal.

Only a complete passing certificate admits the timing mode:

```sh
dotnet /ABS/NinferSmallAttentionProbe.dll --timing \
  --module /ABS/tensorsharp_ninfer_prefill_attention.fatbin \
  --expect-sha256 4ba17472c196242b88adfe27c4446c136abae8c57225593e1eac16acfe6d4602 \
  --legacy-module /ABS/tensorsharp_q35_ninfer_attention.fatbin \
  --expect-legacy-sha256 b6754c46a8b0888eb6760a5027ae91a6f971e7b54e72adeb8ba98945ea37abc0 \
  --component-certificate /ABS/artifacts/FRESH-small-attention/completed-result.json \
  --expect-certificate-sha256 COMPONENT_SHA \
  --out /ABS/artifacts/FRESH-small-attention-timing
```

Timing replays both complete captured graphs: candidate Pack+Core+Reduce+Gate
versus legacy Split+Combine. It uses the same buffers/stream, five ABBA rounds,
six timed graph replays per interval and one symmetric warmup outside events.
All outputs are rechecked against each arm's qualified bytes after every interval.
Keys74/202/512/1024 are all mandatory. The predeclared admission is greater than
5% median improvement and at least four of five paired-round wins in every case.
The report separates numerical pass from performance win. A slower result is a
completed measurement, never permission to proceed to API benchmarking.

This is a component and full-chain GPU timing campaign. It does not establish
model quality, end-to-end throughput, cold-memory behavior or a speed ranking.
The synthetic error envelope is scoped to these represented BF16 inputs.

Graph inspection uses the public v1 CUDA kernel-node ABI described in NVIDIA's
[Driver Graph API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html),
and refuses other argument encoding. No graph updates or hidden external Pack
launches are used to make replay succeed.
