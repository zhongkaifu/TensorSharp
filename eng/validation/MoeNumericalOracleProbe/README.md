# MoE numerical diagnosis

This fresh-process probe reads selected **real GGUF expert slices**, checks the
independent C# weight decoder against native `ggml` dequantization, and compares
the actual fused MoE operation with scalar FP64 dot products, SiLU and weighted
aggregation. It does not use a second `ggml_mul_mat_id` as its mathematical oracle.
Upstream sources remain unchanged.

The default is layer 0, original experts 7 and 311, token counts 1/8/9/38, and
64 evenly spaced final output dimensions. Gate/up are computed completely for
each token and routed expert before sampled down-projection rows are computed.
Use `--output-rows 0` for all final dimensions. The original per-expert H/FF
geometry and raw bytes are preserved; the expert axis is compacted to the loaded
subset. This does not reproduce the original 512-expert kernel dispatch geometry.
The default does not load or map a complete model.

For a dispatch comparison with the original expert-axis geometry, add
`--preserve-expert-axis true`. This maps each complete gate/up/down tensor range
read-only and passes its original expert count and IDs to the native operator;
only selected expert bytes are read and dequantized by the CPU oracle. For the
local Flash checkpoint's layer 0 these ranges total 800 MiB; selected experts
7/311 require 3.125 MiB of explicit raw reads. This mode preserves E=512 and
does not compact it to E=2. CUDA may allocate/copy the original-size tensors,
particularly in the resident arm; the mapping itself is not a GPU-memory cap.
The default native weight-range limit becomes 1 GiB in this mode. No source
weight is rewritten and no full-stack F32 expansion is created.

The default uses two routed experts per token, which is a separate geometry
limitation from E. For the real Flash used-count of ten, supply ten source IDs
and `--used 10`, for example
`--experts 7,311,0,41,97,173,255,388,449,511 --used 10 --preserve-expert-axis true`.
This preserves H=2560, FF=640, E=512 and used=10 while retaining explicitly
synthetic activations and routes. External input/routing files are labelled as
supplied evidence; the tool cannot independently certify that a model captured
them.

By default, inputs and routes are deterministic synthetic values. The report
explicitly labels them; this is **not captured model routing**. To inspect a real
model activation, supply both `--input-file` (little-endian contiguous F32,
exactly `max(tokens) * H` values) and `--routing-file`:

```json
{"Used": 2, "SelectedExperts": [7, 311, 311, 7], "Weights": [0.375, 0.625, 0.5, 0.5]}
```

Routing arrays are token-major and must contain exactly `max(tokens) * Used`
entries. IDs refer to original GGUF experts and all must appear in `--experts`.
Separate gate/up/down SwiGLU tensors are required; bias/scale sidecars are
rejected. The native weight-range limit is checked before each allocation/map.

Build once with the usual native-build skip flags, then place the tested
`GgmlOps` library in this tool's output directory. Run each route in a separate
process and preserve a fresh output directory:

```powershell
$env:TS_HOST_MOE_EXPERT_CACHE_MB = '0'
$env:TS_HOST_MOE_PIN = '0'
$env:TS_HOST_MOE_DEVICE_MIN_BATCH = '0'
dotnet run --no-build -c Release --project eng/validation/MoeNumericalOracleProbe -- --model MODEL.gguf --output artifacts/moe-oracle/cpu --route cpu
$env:TS_HOST_MOE_DEVICE_MIN_BATCH = '1'
$env:TS_HOST_MOE_TIMING = '2'
dotnet run --no-build -c Release --project eng/validation/MoeNumericalOracleProbe -- --model MODEL.gguf --output artifacts/moe-oracle/stream --route gpu-stream
$env:TS_HOST_MOE_DEVICE_MIN_BATCH = '0'
dotnet run --no-build -c Release --project eng/validation/MoeNumericalOracleProbe -- --model MODEL.gguf --output artifacts/moe-oracle/resident --route gpu-resident
$env:TS_HOST_MOE_DEVICE_MIN_BATCH = '1'
dotnet run --no-build -c Release --project eng/validation/MoeNumericalOracleProbe -- --model MODEL.gguf --output artifacts/moe-oracle/stream-full-axis --route gpu-stream --tokens 38 --preserve-expert-axis true
```

These environment settings must be inherited from the launching process. The
probe rejects missing or conflicting settings: updating .NET's environment
inside an already running Windows process does not reliably update native CRT
`getenv`. Such a run can silently execute on CPU while displaying a requested
GPU-stream label. For streaming, `TS_HOST_MOE_TIMING=2` is mandatory; retain the
native `HOSTMOE-COPY` lines as evidence that the device upload path actually ran.
Its synchronization changes observed timings, so do not use them as performance
benchmarks. `TS_HOST_MOE_EXPERT_FILTER=0` is an optional full-stack-upload
diagnostic; preserve that setting explicitly when using it.

The CPU route initializes the CPU backend and disables device streaming. The
GPU stream route enables streaming from N=1 and bypasses selected-expert caching;
the resident route uses the ordinary CUDA operator. Inputs, selected IDs and
routing weights are identical across routes. Each run records source offsets,
raw-slice hashes, the loaded native-library hash, full finite-output checks,
FP64 sample errors and output dumps. Native owners are cleared before pinned
arrays or mapped source views are released. If native cleanup fails, the
diagnostic retains their roots until process exit instead of releasing borrowed
addresses early.

Exit zero means the diagnostic completed with valid finite outputs and matched
dequantization. It does **not** mean model quality or numerical parity passed.
CPU and CUDA activation quantization and reduction policies differ, so their
errors against full FP64 arithmetic are measured rather than hidden behind an
arbitrary universal tolerance. Same top token is not a correctness criterion.
Timings are observations only; this is not a controlled performance benchmark.

Relative L2 is accompanied by reference/actual/error L2 norms, absolute RMS
error and maximum reference magnitude. A zero reference has no relative error
denominator and is reported as null, not divided by an artificial epsilon.
Cancellation within projections or expert aggregation can make relative errors
large even when absolute errors are small; these metrics are diagnostics, not a
universal error allowance. The FP64 formula omits the native operator's
activation quantization, so deviation from it alone cannot locate a kernel bug.

Use `eng/validation/run-bounded-probe.py --timeout 120 --output FRESH_PROCESS_DIR
-- dotnet PATH_TO_PROBE.dll ...` for a hard external deadline. CPU oracle work
and synchronous native calls are not cooperatively cancellable in this tool.
The wrapper kills only its owned probe process on timeout and labels the run
incomplete; previously written case files remain partial evidence. In-process
cleanup failure retains source pins/maps until process exit. It is never safe
to unmap weights while a native call may still borrow them.

`compare.py --left DIR --right DIR --output NEW.json` compares full output
rows between routes. It first verifies the source layer, selected raw-weight
hashes, input hashes, original expert IDs and routing weights, then validates
each output dump against its recorded size/hash. It keeps dispatch geometry and
each arm's sampled FP64 error visible; neither route becomes the numerical
ground truth. Output JSON is created exclusively to preserve earlier evidence.

The original E512/used10 Flash layer-0 diagnostic found a graph-owned weight
lifetime bug at N=1/8: CUDA fused gate/up/GLU could overwrite a weight leaf that
the graph allocator had already recycled. Disabling expert filtering did not
fix it; disabling fusion isolated the cause. TensorSharp now retains uploaded
weight leafs throughout the graph, including deferred allocation fallbacks.
The unchanged upstream fusion overlap check assumes externally owned leafs.
The reusable native regression is `moe-stream-weight-alias-cuda`: its same
test executable fails with the old library and passes with the fix, using the
existing independent FP64 tolerance and repeated-output check. The true streamed
E512/used10 N=1/8/9/38 outputs then match resident CUDA bitwise; their sampled
FP64 relative L2 remains approximately 0.0069–0.0074. This does not resolve or
explain every full-model CPU/CUDA logit difference. Evidence belongs under
ignored `artifacts/unified-memory-adaptive/moe-numerical/`; the first `gpu-stream`
v1 run was actually a CPU route and must not be counted as GPU coverage.
