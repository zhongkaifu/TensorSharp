Qwen-Image-2.1 conditioning parity probe
======================================

This diagnostic saves the entire **pre-final-normalization** Qwen3-VL-8B hidden
tensor that TensorSharp's text encoder produces on one backend, so that two
backends, or two builds, can be compared. It requires the real model and an
available backend. It does not download weights or substitute missing
device/model scenarios with a passing result.

Build the current TensorSharp native library first, then:

On Windows, after building the probe, stage the native DLL from that build into
the probe output directory. A previously copied DLL is not refreshed by the
probe's project reference:

```powershell
dotnet build eng/QwenImage21CompanionProbe -c Release -p:TensorSharpSkipGgmlNative=true
Copy-Item TensorSharp.GGML.Native/build-windows/GgmlOps.dll eng/QwenImage21CompanionProbe/bin/Release/net10.0/GgmlOps.dll -Force
```

Verify the native hash when comparing runs. For other platforms, ensure the
current native library is on the platform loader's search path.

```sh
dotnet build eng/QwenImage21CompanionProbe -c Release
dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- \
  text ../models/qwen-image-2.1/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  docs/validation/qwen-image21/text-managed.f32 Cpu
dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- \
  text ../models/qwen-image-2.1/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  docs/validation/qwen-image21/text-fused.f32 GgmlMetal
dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- \
  compare docs/validation/qwen-image21/text-managed.f32 \
  docs/validation/qwen-image21/text-fused.f32
```

The text encoder takes the same path as in the model, and the probe has no switch
between paths. Every GGML backend (`GgmlCpu`, `GgmlMetal`, `GgmlCuda`) runs the
fused whole-trunk graph; `Cpu` runs the managed per-operation chain of the `cpu`
backend, with its projections on the 8-bit-activation integer GEMM. Comparing
`Cpu` with a GGML backend therefore compares the per-operation chain with the
fused trunk, and comparing two GGML backends compares the same graph on different
kernels. Either difference also includes each backend's quantization and
reduction order. To check a change to one path, run the same backend with two
builds and compare their outputs. A fused-trunk fallback to the per-operation
path is explicitly logged (`[te-fused]`); such a run measures the fallback and
cannot establish fused-trunk parity. The comparison reports relative L2 error and
maximum absolute error, rejects non-finite outputs, and exits unsuccessfully at
relative L2 error of 1% or above. This is a diagnostic tolerance, not a claim of
pixel-level end-to-end equivalence. Generated evidence belongs in ignored
`docs/validation/` or `artifacts/`. The measured real-model conditioning
limitations are recorded in
[`docs/perf/qwen-image21-cuda.md`](../../docs/perf/qwen-image21-cuda.md).

Real-weight CUDA transformer timing
----------------------------------

The `dit` command loads the actual quantized transformer into the diffusion
runtime's denoiser (`QwenImage21Denoiser`, the graph programs the model runs) and
consumes conditioning saved by `text`. It repeats velocity prediction with changing
timesteps and fixed latents (seed 42), excluding text encoding and VAE decoding from
the timings. The first prediction extracts the text prefix into the prefix KV cache
and the later ones read it; each iteration's JSON line names the graph it ran
(`mode`: `Extract`, `Cached`, or `Full` when the cache was declined). It does **not**
produce an image or measure semantic quality. Compare identical dimensions and
iteration counts; the final velocity depends on the count.

```powershell
dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- `
  text C:/Works/models/qwen-image-2.1/Qwen3VL-8B-Instruct-Q4_K_M.gguf `
  docs/validation/qwen-image21-cuda/conditioning.f32 GgmlCuda
dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- `
  dit C:/Works/models/qwen-image-2.1/qwen_image_2.1_Q4_K_M.gguf `
  docs/validation/qwen-image21-cuda/velocity.f32 GgmlCuda `
  docs/validation/qwen-image21-cuda/conditioning.f32 1024 1024 6
```

Report the first prediction separately from subsequent predictions: it includes
weight upload, program compilation, kernel warmup and the prefix extraction. CUDA
graph capture may also affect the second prediction. The retired native graph's
structural switches (`TS_QWEN21_GRAPH_REUSE=0` rebuilt the graph every call,
`TS_QWEN21_PAD_MASK=1` restored the padded image mask) were removed with it: the
probe no longer reads them, and the CLI and the server refuse to start while one is
set. For a baseline, compare the final F32 output of another build with `compare`. The migration commit
`f981cb6` still has the retired engines, which
[`eng/validation/QwenImage21DenoiserAB`](../validation/QwenImage21DenoiserAB/README.md)
ran as `legacy`.

Real-weight VAE checks
---------------------

`vae <weights.safetensors> <output.png> <Cpu|Metal|Cuda> [width] [height]`
decodes deterministic random normalized latents. It saves both a PNG and
interleaved F32 RGBA pixels in `<output.png>.rgba.f32`. This is a numerical and
timing fixture, not an image-quality example. `vae-encode` takes the same
arguments with an F32 output path and encodes a deterministic RGBA gradient and
checkerboard. Both commands reject non-finite results.

The probe selects the VAE path as the model does: the whole-VAE graph on `Cuda`
and `Metal`, and per convolution on the device on `Cpu` (ggml-cpu). There is no
switch between them, so compare the paths across backends: run the same command
at 64 by 64 pixels on `Cpu` and on `Cuda` or `Metal`, and compare the
`<output.png>.rgba.f32` files (or the `vae-encode` outputs) with `compare`. The
difference also includes each backend's rounding. To check a change to one path,
run the same backend with two builds and compare their outputs. A whole-VAE graph
that fails to build or returns non-finite values is logged (`[vae-fused]`) and
the run continues per convolution; such a run does not count as validation of
the fused graph. Use larger dimensions only after numerical checks pass;
whole-VAE scratch requirements grow with image area.

Real-weight vision and DeepStack checks
-------------------------------------

`vision <mmproj.gguf> <output-prefix> <Cpu|Metal|Cuda> [input.png|-] [width=512] [height=width] [iterations=1]`
loads only the actual Qwen3-VL vision projector. Supply the same reference PNG
used by an editing run, or `-` for a deterministic RGBA gradient/checkerboard.
Dimensions are the requested output geometry: the probe calls the model's
`QwenImage21Canvas.ReferenceDimensions` and `ImageIO.Resize`, then uses the conditioner's
alpha-over-white composition and CHW normalization to [-1,1]. The JSON records
the resulting reference geometry and normalized-input SHA-256.

The vision encoder takes the same path as in the model: the fused path on `Cuda`,
and the per-block path with its host erf approximation on `Cpu` and `Metal`. There
is no switch between them, so the example compares the two paths across backends;
the difference also includes each backend's rounding. To check a change to one
path, run the same backend with two builds and compare their outputs.

```powershell
dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- `
  vision C:/Works/models/qwen-image-2.1/mmproj-Qwen3VL-8B-Instruct-F16.gguf `
  docs/validation/qwen-image21-cuda/vision-cpu Cpu reference.png 512 512 2
dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- `
  vision C:/Works/models/qwen-image-2.1/mmproj-Qwen3VL-8B-Instruct-F16.gguf `
  docs/validation/qwen-image21-cuda/vision-cuda Cuda reference.png 512 512 2
foreach ($part in 'main', 'deepstack0', 'deepstack1', 'deepstack2') {
  dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- `
    compare "docs/validation/qwen-image21-cuda/vision-cpu.$part.f32" `
    "docs/validation/qwen-image21-cuda/vision-cuda.$part.f32"
}
```

Every iteration requires one main embedding and exactly three DeepStack
embeddings, each with `referenceWidth/32 * referenceHeight/32` rows and 4096
channels; all values must be finite. The last iteration is saved as
`<prefix>.main.f32` and `<prefix>.deepstack0.f32` through `.deepstack2.f32`.
`<prefix>.json` records shapes, per-iteration time including synchronous readback,
separate model-load/preprocessing time, the backend, and hashes of the probe,
managed model assembly and copied native library. The backend name alone is not
proof that the fused path ran; inspect the logs too. Compare all
four outputs and report actual errors; the generic `compare` command's 1%
diagnostic limit alone does not establish image-quality equivalence. Rebuild the
probe after production changes so its copied dependencies are current.

## Prefix KV cache and tensor-parallel parity (real DiT weights)

`dit-parity` loads the real diffusion transformer into the diffusion runtime's
denoiser and predicts one request layout (text, one reference image and the target)
at two denoising steps. Each step is predicted three ways: by the whole-sequence
graph (a denoiser whose prefix cache is always declined), through the prefix KV
cache (step 1 extracts the prefix, step 2 reads it), and with the blocks sharded
over a loopback tensor-parallel group of ranks on this one device (`GgmlCpu` or
`GgmlMetal`), with and without the cache. It fails when the extract step is not
bit-identical to the whole-sequence one, when the cached step is not (on `GgmlCpu`)
or exceeds 1% relative L2 (on a GPU backend), or when a sharded prediction exceeds
1%. A GPU backend picks its kernels, and how they split a product, by the rows the
product computes, and a cached step computes only the target's. On `GgmlCuda` (RTX
4000 Ada) at 1024 x 1024 with a 512 x 512 reference, the runtime's extract step was
bit-identical and its cached step 0.25% from the whole sequence. The retired engine's
two modes were 0.23% to 0.46% apart there at 512 x 512. The comparisons against 8-bit cache storage (`q8_0`, `q8_0_v`) were
removed with that storage. Conditioning and latents are seeded noise, so it checks
kernels on the real quantized weights, not image quality or multi-GPU speed.

Besides the JSON report, it writes every prediction's velocities as raw F32 into
`<report>.d/` (`full-step-1.f32`, `cache-step-2.f32`, `tp2-cache-step-1.f32`, …), so
two builds' predictions can be compared byte for byte, for example with `cmp`, or
with `compare` for a relative L2. On a Linux x64 `ggml_cpu` build with the
Qwen-Image-2.1-Turbo Q4_K_M weights, the runtime's predictions (whole sequence,
cache extract and cached, two-rank loopback whole, extract and cached) were
byte-identical to the retired native graph's, and its cache predictions to its
whole-sequence ones. That run still fails the 1% bound: at 256 x 256 with a 128 x 128
reference the two-rank predictions are 4.3% (whole, extract) and 2.7% (cached) relative
L2 from the unsharded ones, and the retired TP forward gave exactly the same bytes, so
this is how the sharded transformer rounds on these random inputs, not a change of the
migration. Metal has not been run. On macOS, stage the current native
library first, because the project reference does not refresh a previously copied
one:

```sh
dotnet build eng/QwenImage21CompanionProbe -c Release
cp TensorSharp.GGML.Native/build/libGgmlOps.dylib eng/QwenImage21CompanionProbe/bin/Release/net10.0/
dotnet eng/QwenImage21CompanionProbe/bin/Release/net10.0/QwenImage21CompanionProbe.dll dit-parity \
  ../models/qwen-image-2.1/qwen_image_2.1_Q4_K_M.gguf artifacts/qwen21/dit-parity.json GgmlMetal 512 512 256 2
```
