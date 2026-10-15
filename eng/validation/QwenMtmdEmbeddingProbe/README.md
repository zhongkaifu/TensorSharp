# Qwen projected-embedding control

`main.cpp` is TensorSharp-owned diagnostic code that calls unchanged llama.cpp
public C APIs. It loads the language model's vocabulary/hyperparameters/tensor
metadata with `vocab_only=false`, `no_alloc=true`, `load_mode=NONE`, and
`lazy_mode=OFF`; it creates no `llama_context` and never decodes the language model. Only the vision projector
executes. It exports the actual finite `[300,2560]` F32 projection and absolute
T/H/W decoder positions for the original opaque `card-0.png` fixture. The native
tokenization must reproduce all repaired TensorSharp input IDs (318 or 354),
including the official generation boundary's final double newline.

This is a numerical isolation tool, **not a generation-quality pass or a
performance benchmark**. Flash attention deliberately matches the existing llama
image control; its projection is not automatically declared ground truth.

The Python driver compiles only this small owned translation unit, using the
baseline's existing `mtmd.lib`, `llama.lib`, `ggml.lib`, and `ggml-base.lib`. The
backend registry/loading APIs are exported by `ggml`, not `ggml-base`. It never builds or
changes an upstream source file. Build/prepare/run pin headers, import libraries,
the owned tool, native DLL closure, prompt, RGB bytes, and small PNG. The original
checkpoint is checked by path, size, and mtime rather than rehashing 74.5 GB. All
hashing is outside the encode interval. Run only in an exclusive GPU window.

Example Windows commands (fresh ignored destinations are required):

```powershell
$py = 'C:/Users/monke/AppData/Local/Programs/Python/Python310/python.exe'
& $py -X utf8 -B eng/validation/qwen-mtmd-embedding-probe.py build `
  --source C:/Works/llama.cpp `
  --baseline-build C:/Works/TensorSharp/artifacts/llama-flash-4ebdf-cuda `
  --vcvars 'C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat' `
  --output D:/TensorSharp-validation-cache/master-merge-20261013/mtmd-embedding-build-v3
if ($LASTEXITCODE) { throw 'Owned probe build failed' }

& $py -X utf8 -B eng/validation/qwen-mtmd-embedding-probe.py prepare `
  --build-record D:/TensorSharp-validation-cache/master-merge-20261013/mtmd-embedding-build-v3/build.json `
  --control-plan artifacts/master-merge-20261013/qwen-image-llama-control-run-v2/plan.json `
  --image artifacts/multimodal-local-preflight/vision-fixtures/card-0.png `
  --prompt artifacts/master-merge-20261013/vision-parity-tool-prepared-v1/turn-2-canonical-prompt.txt `
  --turn 2 --gpu `
  --output artifacts/master-merge-20261013/mtmd-embedding-plan-v3
if ($LASTEXITCODE) { throw 'Projection plan failed' }

& $py -X utf8 -B eng/validation/qwen-mtmd-embedding-probe.py run `
  --plan artifacts/master-merge-20261013/mtmd-embedding-plan-v3/plan.json `
  --output artifacts/master-merge-20261013/mtmd-embedding-run-v3
if ($LASTEXITCODE) { throw 'Projection capture failed; see preserved report/stderr' }
```

The model-independent Python tests use six-float synthetic files to check finite
payloads, exact byte geometry, IDs/positions, fail-closed capture policy, and pin
mutation. They initialize no native backend. Native `--self-test` also initializes
no backend and checks format primitives only. Neither counts as real-model or
device coverage. The compiled tool is currently Windows import-library driven;
the C++ calls themselves are public portable APIs, but other platforms require a
separate build/run qualification.

The baseline Windows prompt log contains CRLF in its text dump. The example
uses the separately prepared canonical LF copy: its receipt records the source
hash and exactly 15 CRLF-to-LF replacements. Do not assume a textual newline
normalization is harmless: the full expanded-ID equality guard remains required.

In unchanged llama.cpp revision `4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe`,
`src/llama-model.cpp::load_hparams` returns early for `vocab_only`, before setting
`n_embd`. That makes the projection width query zero. The current metadata-only
policy follows the same file's `load_tensors` path, which allocates zero-byte
dummy buffers for `ml.no_alloc` and returns before `load_all_data`. Explicit
`NONE` and lazy `OFF` prevent `src/llama-model-loader.cpp::init_mappings` from
mapping or prefetching the LM checkpoint. These implementation files are pinned
as reviewed evidence but are not compiled or modified. Native schema v2 records
the actual configured policy; it does not label this load vocabulary-only.

## TensorSharp side of the comparison

`eng/validation/QwenImageLogitProbe/StateCapture.cs::PreparedMedia` already reads
the request bucket's exact projected tensor, verifies F32, records shape/hash,
and returns the full absolute T/H/W positions. A future diagnostic-only export
can add an optional **fresh output directory** parameter and stream those same
existing tensor bytes into `projected-0.f32`; it needs no extra image-sized
allocation. Record the path, bytes, shape, and hash in the existing metadata.
Also write the positions to JSON. Keep the current pointer read after
`Storage.EnsureHostReadable()`, and do not export stale managed GDN state or
alter production tensor ownership. Keep the old method signature or add an
overload so older diagnostic binaries remain usable.

Do not apply that managed change during a frozen managed validation run. The
existing logit probe is explicitly pinned to historical 317-token input and must
not silently accept this repaired 318-token fixture. A new diagnostic version
must explicitly bind the new template/IDs before comparing or cross-feeding
projections.
