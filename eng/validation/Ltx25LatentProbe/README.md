# LTX final-latent decoder comparison

This diagnostic runs the registered native LTX model through final sampling once,
then saves normalized video latents in `[128,T,H,W]` order and joint audio latents
in `[tokens,128]` order. A probe-only callback stops before the production decoder;
the pipeline cleanup and model disposal run normally. Each selected VAE then
decodes the same saved video buffer. The production generation entry point has no
callback and retains its existing behavior.

Build from the repository root (the skip switches reuse existing native builds):

```sh
dotnet build eng/validation/Ltx25LatentProbe/Ltx25LatentProbe.csproj \
  --configuration Debug \
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
```

For MLX, select its installed library directory and set precision at process
launch. Generate and decode with both VAEs:

```sh
MLX_ENABLE_TF32=0 TENSORSHARP_MLX_LIBRARY_DIR="$PWD/TensorSharp.Backends.MLX/bin" \
dotnet eng/validation/Ltx25LatentProbe/bin/Debug/net10.0/Ltx25LatentProbe.dll \
  --operation generate --backend ggml_metal \
  --model ../models/ltx25/LTX-2.5-Distilled-Q4_K_M.gguf \
  --prompt-file artifacts/prompt.txt --image imgs/banner_1.png \
  --seed 3206401772581117598 --width 1024 --height 1024 --frames 145 --fps 24 \
  --decoders both --output artifacts/ltx-capture
```

For a diagnostic conditioning-strength comparison, add `--image-strength 0.8`
with `--image`. Omitting this flag uses the ordinary production shorthand policy:
a single first-image request has strength `0.7` in the half-size pass and `1` in
refinement. Every explicitly supplied value, including `--image-strength 1`, uses
the existing explicit frame-zero image condition at that strength in both stages.
Values must be finite and within `[0,1]`. Explicit `1` fixes the first latent frame to the encoded
image, while smaller values allow more denoising of that frame; `0` gives it the
full denoising mask. This changes the sampled latents and can reduce likeness, so
it is a diagnostic choice rather than a promised quality improvement. The
last-image guide, if supplied, retains strength `1`. The report records each
original source path, content hash, frame index and strength; the capture also
records the first-image strength. This option is only valid for generation with
`--image` and does not change an existing captured latent buffer during re-decode.
The report also records whether the strength was explicit and obtains effective
stage strengths from the same resolver as production, rather than duplicating
its policy in the probe.

To compare a weaker first-stage anchor while restoring the original image in
refinement, use `--first-stage-image-strength 0.7 --image-strength 1`. This
diagnostic overrides only the frame-zero encoded guide in the first half-size
pass. The full-size refinement pass keeps `--image-strength`, and other temporal
guides keep their own strengths. Both effective first-image strengths are saved
in the report and capture. Omitting the option preserves the ordinary pipeline
policy. The override requires a two-stage mode
with `--image` and cannot be used when re-decoding an existing capture.

For a bounded comparison with the official denoiser, use `--operation forward`
with the same model, image, prompt, seed and output geometry. This executes the
real image encoder, text encoder and conditioning, then captures the first
complete 48-block denoiser input and velocity. A diagnostic observer stops before
the sampler updates any latents. With a requested output of `1024x1024x145`, this
captures the real first pass at `512x512x145`; it does not run refinement or a VAE
decoder. `--decoders none` is implicit and other decoder choices are rejected.

`forward.json` records the checkpoint SHA-256, native model configuration, exact
Int64 seed, source-image hashes, sigmas, precision, executed depth and named
Float32 buffers. `inputs` entries contain absolute `path`, `sha256`, `shape` and
`dtype: float32-le`. Latents/contexts are token-major; position boundaries are
`[tokens,axes,2]`, with time/height/width axes for video and time for audio. Optional
timestep arrays preserve the actual call: null broadcasts the corresponding
scalar sigma. The frame-zero video timestep and keyframe masks are retained.
`outputs` and `output_sha256` identify the two token-major velocity arrays.
Sampler-only clean latents/denoise masks may be null because `Predict` receives
their derived timesteps. This isolates a real forward call; it does not establish
complete sampling or visual-quality parity. Production denoisers have no observer.
Use `--image-strength 1` explicitly to capture the full-strength `1/1` condition
independently of the shorthand production default.

After building the probe, run the CPU-only metadata guard:

```sh
python3 eng/validation/tests/validate_ltx25_probe_strength_metadata.py \
  --output artifacts/ltx-probe-strength-metadata
```

It checks omitted, explicit and diagnostic strengths against the report. A
deliberately invalid GGUF stops each case before model/device initialization.

VAE paths default to the model directory's `vae/` companions. Override them with
`--convolutional-vae` and `--diffusion-vae`. `--decoders none` saves only the latent
capture. Decode an existing capture without loading or running the denoiser:

```sh
MLX_ENABLE_TF32=0 TENSORSHARP_MLX_LIBRARY_DIR="$PWD/TensorSharp.Backends.MLX/bin" \
dotnet eng/validation/Ltx25LatentProbe/bin/Debug/net10.0/Ltx25LatentProbe.dll \
  --operation decode --backend mlx --capture artifacts/ltx-capture/capture.json \
  --decoders diffusion --output artifacts/ltx-diffusion-redecode
```

`--operation verify --capture FILE --output NEW_DIRECTORY` validates shape,
layout, byte counts, hashes and finite values without initializing a model or
device. Captures record the exact Int64 seed as a JSON string. Every output
directory must be empty. Float buffers are little-endian; their SHA-256 hashes
are checked before decoding and after each decoder to detect input mutation.

The default outputs are `capture.json`, two small latent buffers, `report.json`
and one silent MP4 per selected decoder. Audio latents are retained but audio is
not decoded. `--save-rgb 1` also writes decoded Float32 RGB files; a 145-frame
1024-square clip consumes about 1.8 GB per decoder, so these are off by default.
Reports include timings, model-binary and decoder-checkpoint hashes, latent/RGB/
MP4 hashes, geometry, precision environment and available MLX memory telemetry.
They do not establish perceptual quality or a controlled performance benchmark.
