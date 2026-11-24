# QwenImage21DenoiserAB

Times and compares Qwen-Image-2.1 on real weights: single denoiser steps (`QwenImage21Denoiser`, the diffusion runtime's
graph programs) and whole requests through the model's public API (`--e2e`).

```bash
dotnet build eng/validation/QwenImage21DenoiserAB -c Release
# libGgmlOps (TensorSharp.GGML.Native) must sit next to the probe, as for the test project.

# Denoiser steps on random conditioning and latents (seeded): time a step, write each step's velocities.
dotnet eng/validation/QwenImage21DenoiserAB/bin/Release/net10.0/QwenImage21DenoiserAB.dll \
    --dit qwen-image-2.1-turbo-Q4_K_M.gguf --backend ggml_cuda --size 512 --steps 4 --refs 1 \
    --engines runtime,runtime2 --out velocities/
# A later run (another backend, another build) against those velocities:
#   ... --engines runtime --against velocities/

# One request end to end (companions from TS_QWEN_IMAGE_VAE / _TE / _MMPROJ, variant from TS_QWEN_IMAGE_VARIANT):
dotnet eng/validation/QwenImage21DenoiserAB/bin/Release/net10.0/QwenImage21DenoiserAB.dll \
    --dit qwen-image-2.1-turbo-Q4_K_M.gguf --e2e --model runtime --backend ggml_metal \
    --prompt "A small orange cat beside a blue ceramic vase" --width 512 --height 512 --seed 42 --out cat.png \
    [--image source.png] [--mask mask.png] [--cfg 2.5 --negative "blurry"] [--previews 2] [--lora voxel.safetensors:1.0]
```

`--engines` takes `runtime`, `legacy` (the retired `QwenImage21DiT`) and `runtime2`/`legacy2` (a second, independent
instance on the same inputs: run-to-run determinism and the warm step time); `--e2e --model legacy` runs the retired
`QwenImageModel`. The legacy engines are deleted in the next commit; check this one out to compare a backend against them.
Outputs and logs belong in the ignored `artifacts/` or `docs/validation/`.
