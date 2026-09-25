Qwen-Image-2.1 conditioning parity probe
======================================

This diagnostic compares the entire **pre-final-normalization** Qwen3-VL-8B
hidden tensor between TensorSharp's managed operator chain and its fused native
trunk. It requires the real model and an available backend. It does not download
weights or substitute missing device/model scenarios with a passing result.

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
TS_QWEN_TE_FUSED=0 dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- \
  text ../models/qwen-image-2.1/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  docs/validation/qwen-image21/text-managed.f32 GgmlMetal
TS_QWEN_TE_FUSED=1 dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- \
  text ../models/qwen-image-2.1/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  docs/validation/qwen-image21/text-fused.f32 GgmlMetal
dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- \
  compare docs/validation/qwen-image21/text-managed.f32 \
  docs/validation/qwen-image21/text-fused.f32
```

Use `GgmlCpu` or `GgmlCuda` to test a different available backend. A fused-kernel
fallback is explicitly logged; such a run measures the fallback and cannot
establish fused-kernel parity. The comparison reports relative L2 error and
maximum absolute error, rejects non-finite outputs, and exits unsuccessfully at
relative L2 error of 1% or above. This is a diagnostic tolerance, not a claim of
pixel-level end-to-end equivalence. Generated evidence belongs in ignored
`docs/validation/` or `artifacts/`.

To locate text-encoder divergence, set `TS_QWEN_TE_TRACE_DIR` to an ignored
directory and `TS_QWEN_TE_TRACE_LAYER` to a zero-based layer index, then run
`text` in separate processes with `TS_QWEN_TE_FUSED=0` and `1`. Matching
`unfused.L00.<stage>.f32` and `fused.L00.<stage>.f32` files contain planar F32
snapshots from normalization, projections, attention and MLP stages. Compare
their shapes through file lengths and their values using `compare`. Native
tracing retains intermediate tensors and may inhibit fusion, so also compare
each traced final output with its untraced output. `TS_QWEN21_TE_PRESCALE=1`
restores the older fused Qwen3-VL projection scaling for diagnosis. The measured
real-model conditioning limitations are recorded in
[`docs/perf/qwen-image21-cuda.md`](../../docs/perf/qwen-image21-cuda.md).

Real-weight CUDA transformer timing
----------------------------------

The `dit` command loads the actual quantized transformer and consumes conditioning
saved by `text`. It repeats velocity prediction with changing timesteps and fixed
latents (seed 42), excluding text encoding and VAE decoding from the timings. It
does **not** produce an image or measure semantic quality. Compare identical
dimensions and iteration counts; the final velocity depends on the count.

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
weight upload, allocator planning and kernel warmup. CUDA graph capture may also
affect the second prediction. `TS_QWEN21_GRAPH_REUSE=0` rebuilds the transformer
graph every call. `TS_QWEN21_PAD_MASK=1` restores the previous padded image mask
and per-segment F32 K/V preparation. Set both before launching a separate process
for a structural baseline, and compare its final F32 output with `compare`.

Real-weight VAE checks
---------------------

`vae <weights.safetensors> <output.png> <Cpu|Metal|Cuda> [width] [height]`
decodes deterministic random normalized latents. It saves both a PNG and
interleaved F32 RGBA pixels in `<output.png>.rgba.f32`. This is a numerical and
timing fixture, not an image-quality example. `vae-encode` takes the same
arguments with an F32 output path and encodes a deterministic RGBA gradient and
checkerboard. Both commands reject non-finite results.

For an independent scalar reference, set `TS_QWEN_VAE_GPU=0` in a fresh process
and use `Cpu` at 64 by 64 pixels. Compare CUDA results at that same size using
`compare`. Test `TS_QWEN21_VAE_FUSED=0` and the default fused path separately.
The probe selects the whole-VAE graph as the model does: by default on CUDA and
Metal, on CPU only with `TS_QWEN21_VAE_FUSED=1`, and never on Vulkan.
`TS_QWEN21_VAE_TRACE=1` identifies fused execution or fallback and reports
per-convolution tensor ranges in the fallback. A fallback does not count as
validation of the fused graph. Use larger dimensions only after numerical
checks pass; whole-VAE scratch requirements grow with image area.

Real-weight vision and DeepStack checks
-------------------------------------

`vision <mmproj.gguf> <output-prefix> <Cpu|Metal|Cuda> [input.png|-] [width=512] [height=width] [iterations=1]`
loads only the actual Qwen3-VL vision projector. Supply the same reference PNG
used by an editing run, or `-` for a deterministic RGBA gradient/checkerboard.
Dimensions are the requested output geometry: the probe calls the pipeline's
`ResolveReferenceDimensions` and `ImageIO.Resize`, then uses the conditioner's
alpha-over-white composition and CHW normalization to [-1,1]. The JSON records
the resulting reference geometry and normalized-input SHA-256.

```powershell
$env:TS_QWEN21_VISION_FUSED = '0'
dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- `
  vision C:/Works/models/qwen-image-2.1/mmproj-Qwen3VL-8B-Instruct-F16.gguf `
  docs/validation/qwen-image21-cuda/vision-old Cuda reference.png 512 512 2
$env:TS_QWEN21_VISION_FUSED = '1'
dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- `
  vision C:/Works/models/qwen-image-2.1/mmproj-Qwen3VL-8B-Instruct-F16.gguf `
  docs/validation/qwen-image21-cuda/vision-new Cuda reference.png 512 512 2
foreach ($part in 'main', 'deepstack0', 'deepstack1', 'deepstack2') {
  dotnet run --project eng/QwenImage21CompanionProbe -c Release --no-build -- `
    compare "docs/validation/qwen-image21-cuda/vision-old.$part.f32" `
    "docs/validation/qwen-image21-cuda/vision-new.$part.f32"
}
```

Every iteration requires one main embedding and exactly three DeepStack
embeddings, each with `referenceWidth/32 * referenceHeight/32` rows and 4096
channels; all values must be finite. The last iteration is saved as
`<prefix>.main.f32` and `<prefix>.deepstack0.f32` through `.deepstack2.f32`.
`<prefix>.json` records shapes, per-iteration time including synchronous readback,
separate model-load/preprocessing time, the requested switch, and hashes of the
probe, managed model assembly and copied native library. A requested switch is
not proof that an optimized kernel ran; inspect fallback logs too. Compare all
four outputs and report actual errors; the generic `compare` command's 1%
diagnostic limit alone does not establish image-quality equivalence. Rebuild the
probe after production changes so its copied dependencies are current.

## Prefix KV cache and tensor-parallel parity (real DiT weights)

`dit-parity` loads the real diffusion transformer and predicts one request layout
(text, one reference image and the target) at two denoising steps. Each step is
predicted three ways: by the whole-sequence graph, through the prefix KV cache
(default, `q8_0` and `q8_0_v` storage), and with the blocks sharded over a
loopback tensor-parallel group of ranks on this one device. It fails when the
default cache is not bit-identical, when 8-bit storage exceeds 2% relative L2,
or when the sharded prediction exceeds 1% relative L2. Conditioning and latents
are seeded noise, so it checks kernels on the real quantized weights, not image
quality or multi-GPU speed. On macOS, stage the current native library first,
because the project reference does not refresh a previously copied one:

```sh
dotnet build eng/QwenImage21CompanionProbe -c Release
cp TensorSharp.GGML.Native/build/libGgmlOps.dylib eng/QwenImage21CompanionProbe/bin/Release/net10.0/
dotnet eng/QwenImage21CompanionProbe/bin/Release/net10.0/QwenImage21CompanionProbe.dll dit-parity \
  ../models/qwen-image-2.1/qwen_image_2.1_Q4_K_M.gguf artifacts/qwen21/dit-parity.json GgmlMetal 512 512 256 2
```
