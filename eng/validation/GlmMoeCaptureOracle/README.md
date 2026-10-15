# GLM MoE capture oracle

This CPU diagnostic compares ordinary and compact GLM expert execution against
independent FP64 mathematics on exactly matched captured inputs. It does not load
GGML or run a model, change a production gate, or establish full-model quality.

Build the standalone project and run its analytic checks:

```powershell
dotnet run --project eng/validation/GlmMoeCaptureOracle -- --self-test true
```

For a private TEST_HOOKS capture with an explicit layer and byte ceiling:

```powershell
dotnet GlmMoeCaptureOracle.dll --captures E:/diagnosis/captures --models-assembly D:/frozen/TensorSharp.Models.dll --output E:/diagnosis/oracle.json
```

The capture index must record exact `type`, `type_name`, `ne`, `nb`, `nbytes`,
graph/model/evaluation identity, layer, original row range, clamp and raw payload.
Noncontiguous router IDs are read using their actual byte strides. Comparison
refuses unmatched full inputs, IDs, routing weights, original blocks, overlapping
rows, or compact selected-slot block identity. The decoder is TensorSharp's
managed implementation, independent of native projection kernels; the loaded
assembly, index and every payload are hashed in the report.

The FP64 reference evaluates gate/up dots, gate upper-only clamp, symmetric up
clamp, SiLU, down dots and the original route-order weighted fold. A separate
reference rounds explicit FP32 stage boundaries; it does not reproduce a CUDA
reduction tree, tensor cores or activation quantization. Neither ordinary GGML
results nor the closest rounding hypothesis are presumed to be ground truth.

Generate reproducible capture instrumentation on a fresh source copy:

```powershell
python eng/validation/GlmMoeCaptureOracle/prepare-private-capture.py --source-root D:/frozen-native/source --manifest D:/frozen-native/build-summary.json --native-reference D:/frozen-native/production/GgmlOps.dll --upstream C:/Works/TensorSharp/ExternalProjects/ggml --output E:/private-glm-capture
```

The generator requires a successful complete native source manifest, verifies
every original byte and production artifact hash, and requires the same clean
upstream revision. It refuses an existing output directory or an output inside
the original source/upstream tree. It copies all TensorSharp native inputs,
modifies only the private GLM and expert-cache implementations, adds a private
header, and records the full resulting source identity. Unknown source anchors
fail closed. It supports both narrow scalar and wide bulk layouts.

Compile that copy into a separate native output with `TSG_GGML_TEST_HOOKS` enabled.
Keep upstream source unchanged and record its revision, compiler commands, input
and output hashes. The standard private CMake build also needs unchanged ggml at
the expected `../ExternalProjects/ggml` location (a read-only source link is
sufficient). Do not run a publishing script or replace production libraries.

Run a bounded fixture with `TS_GLM_MOE_DIAG_ROOT` set to a fresh existing capture
directory, `TS_GLM_MOE_DIAG_LAYER` to its first routed MoE layer, and
`TS_GLM_MOE_DIAG_MIN_ROWS` to the intended width. The default payload ceiling is
64 MiB, single tensor ceiling 8 MiB, and read staging chunk 1 MiB. A larger total
ceiling requires an explicit `TS_GLM_MOE_DIAG_MAX_BYTES` and matching oracle
`--max-bytes`; it is recorded as diagnostic storage, not production credit.
The single tensor/work limits intentionally reject full large-model captures;
use bounded fixtures made from complete original quantization blocks.

The private main graph retains exact input/router/route/weight/projection/output
bytes. Its selected boundary uses the public scheduler callback API; compact
projections are read after their actual backend stream is fenced. Records include
the original row interval and resident remapped slot bytes. Explicit ordinary
input/selected-boundary markers separate evaluations even if graph addresses and
their local counters are reused.

Capture instrumentation retains additional graph outputs and may change fusion.
Capture payloads and decoded oracle weights are bounded diagnostic allocations,
outside the production shared-budget accounting scope. Store generated evidence
under ignored `artifacts/` or `docs/validation/`, never alongside this tool.
