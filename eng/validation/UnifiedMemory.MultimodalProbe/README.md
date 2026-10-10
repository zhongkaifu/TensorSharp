# Multimodal native-budget probe

This explicit validation entry point runs real Qwen-Image 2.1 or MiniMax-H3
models under `GgmlCacheBudgetScope(includeGraphBuffers: true)`. It records
generation time, sampled process RSS and credit usage, raw output hashes,
native/managed binary hashes, and outstanding credit after model disposal and
explicit cache cleanup. Generated images, audio floats and reports belong in
ignored `artifacts/` or `docs/validation/`.

Build with the repository's unmodified pinned ggml dependency:

```sh
dotnet build eng/validation/UnifiedMemory.MultimodalProbe -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true \
  -o artifacts/multimodal-budget/bin
# Copy the matching GgmlOps native library into that output directory.
dotnet artifacts/multimodal-budget/bin/UnifiedMemory.MultimodalProbe.dll \
  --kind image --model /models/qwen_image_2.1_Q4_K_M.gguf \
  --output artifacts/multimodal-budget/image --budget-gib 8 \
  --width 512 --height 512 --steps 20 --repeat 2 --seed 42
```

Companion weights resolve beside the model, or through the model's documented
environment variables. Image editing takes `--images first.png|second.png` (quote
the value in a shell). Video takes `--kind video`, `--frames 22`, optional
`--image first.png` / `--end-image last.png`, and the H3 denoiser model. All
arguments are `--name value` pairs. `--budget-gib -1` omits the scope for a
numerical control; `0` requests zero covered allocation credit. `--cfg` defaults
to 1, `--steps` to 20. H3 needs its pinned tokenizer files as well as four models.

The quota is a **common constraint on explicitly routed native allocations**.
It does not include every model mapping, managed array, backend/vendor workspace
or driver allocation. It is neither a process RAM limit nor a hard board VRAM
limit; external resource sampling and OS limits must be recorded separately.
The 10 ms sampled peaks are lower bounds, not exact high-water marks. Load and
generation are measured separately from PNG/raw output export, but first-call
generation includes lazy companion loading. Runs in the same process may reuse
weights and backend caches.

`Status: passed` means generation returned finite output and cleanup returned
covered credit. It does not establish prompt adherence, lack of visual/audio
artifacts, equality with an independent implementation, or performance parity.
Compare raw float files for accounting-only changes, inspect images/video/audio
separately, and retain failures rather than treating missing outputs as passes.
