# Qwen projected-image tensor export

This diagnostic exports the actual prepared Qwen3.8 Flash Next projected image
tensor before **any language-model forward**. It uses the public CUDA unified
session and the production renderer/image-preparation admission. It creates no
engine, scheduler request, teacher tokens, KV forward, or answer-quality grade.

Build only against a frozen, independently qualified SDK publish:

```powershell
dotnet build eng/validation/QwenVisionEmbeddingProbe/QwenVisionEmbeddingProbe.csproj `
  -p:ValidationRuntimeDirectory=D:/qualified-sdk-publish `
  -p:BaseIntermediateOutputPath=E:/validation-tmp/vision-probe-obj/ `
  -p:OutputPath=D:/validation-tmp/vision-probe-bin/ -m:1
```

There are deliberately no product `ProjectReference`s. A missing
`ValidationRuntimeDirectory` fails closed. Run `--self-test` with the native
library unavailable/masked first; this tests pure policy and requires zero
loaded GgmlOps modules. Building this tool is not a new product-build or device
qualification.

Supply seven unique absolute-path/value arguments:

```text
--model FIRST.gguf --mmproj mmproj-BF16.gguf --image card.png
--input input.json --output FRESH_DIRECTORY --merger legacy|reference
--native-sha256 EXPLICIT_PINNED_GGMLOPS_SHA256
```

The private tool directory must contain the exact qualified runtime dependency
closure and pinned native library. Use an external owned-process timeout for
loading/preparation; an unmanaged model load cannot be interrupted safely by an
in-process timer. Do not run another model or build during a measured run.

Input is strict JSON (case-insensitive property names, unknown fields rejected):

```json
{
  "Schema": "qwen-vision-embedding-input.v1",
  "Messages": [
    { "Role": "user", "Content": "请详细描述这幅图",
      "ImagePaths": ["sha256:ACTUAL_IMAGE_SHA256"] }
  ],
  "ExpectedPromptTokens": [0],
  "ImageSha256": "ACTUAL_IMAGE_SHA256"
}
```

Replace the illustrative token list with all **318 actual repaired HTTP input
IDs**, or provide the original three-message continuation history and all
**354 actual IDs**. Both histories must contain the original 300 expanded image
placeholders. The first image content binding is validated then replaced with
the explicitly supplied local image path; other messages cannot attach media.
The tool must independently reproduce every expanded ID. It intentionally
rejects the historical 317/353 IDs that omitted the final generation newline.

Controlled session settings: context 1024, N1, PP32 policy, cache 0, prefix
checkpoints 0, retained conversations 0, async expert prefill off, capacity
growth off, CPU expert threads 8, all routed experts host-resident. The merger
selection is explicit in each process; legacy selects tanh, reference selects
the diagnostic erf merger. Inherited TS/TENSORSHARP/GGML/LLAMA_ARG/CUDA model
execution overrides are rejected. CUDA installer discovery locations
`CUDA_PATH`, `CUDA_HOME` and versioned `CUDA_PATH_V<major>_<minor>` are allowed;
device visibility/order, module loading, launch blocking and CUDA cache-policy
overrides remain rejected. No streaming expert, prefix, MTP or TP execution is
activated by this tool.

`projected-000.f32` is little-endian row-major `[300, language-hidden]`. The
existing admitted tensor stays owned and bucket-locked throughout export. The
tool invokes `EnsureHostReadable` plus `SyncHostBuffer` on its real backing
storage, checks every float is finite, and writes/hashes a direct read-only
span. It makes **no additional media-sized managed/unmanaged array or tensor**.
FileStream/hash internals use fixed-size buffers. Shape, count, bytes, SHA256,
span boundaries, exact expanded IDs, absolute T/H/W positions, input/image
hashes, managed/native pins, model path/size/mtime bookends, and zero-ledger
cleanup are recorded in `report.json`.

Model checkpoint identity is metadata-only here; model shards are not rehashed
during this diagnostic. Existing qualified model hashes must be joined by the
caller before stronger checkpoint-identity claims. Preparation/export timing
includes diagnostics and is not an inference-performance benchmark. Comparing
this raw file with independently exported unchanged-upstream mtmd embeddings
can localize vision differences; equal token IDs alone cannot establish tensor
parity. An export success does not demonstrate image-continuation quality.
