# Explicit FlashInfer native-layout W4A16 prototype

The three Python sources in `upstream/` are byte-identical to FlashInfer
`61bc8b6dd7049a44d614b2a0cd4a036cb3c46ad0`; `upstream/manifest.json` pins their Git
blob IDs, SHA-256 digests, original URLs and Apache license. No upstream or ggml
files are patched.

The TensorSharp-owned build tool selects the existing staged kernel with static
T=1/T=8, N=34816, K=5120, split-K=1, four warps, K tile 128, two stages and PDL
disabled. It instantiates the unchanged generic output epilogue with `Float32`.
That original configuration remains the default (`--tactic s1-w4-st2`). Separate
explicit `s1-w8-st2` and `s1-w4-st3` trials change the warp count or staging depth,
and record their own policy identity, launch block and shared-memory reservation.
This bypasses the public Python runner's BF16/FP16 output restriction and does
not change its arithmetic implementation. The input is actual BF16, weights are
original adjacent-K FP4 bytes, scales are original compact M128x4 E4M3 bytes,
and the single global inverse divisor is applied after F32 MMA accumulation.

This arithmetic differs from TensorSharp's legacy per-weight global scaling and
scalar FMA association. Unrounded F32 output is necessary for gate/up followed
by the existing F32 SwiGLU. A component error bound cannot establish unchanged
whole-model quality. No serving hooks or default changes are provided here.

Verify the pinned sources without CUDA dependencies:

```sh
python3 eng/validation/build-ninfer-flashinfer-w4a16.py --verify-only
```

Build using an existing CuTe DSL/CUDA Python environment (no package install):

```sh
/root/venvs/vllm/bin/python eng/validation/build-ninfer-flashinfer-w4a16.py --out /fresh/output/directory
```

The builder exports standalone CUBINs and PTX, refuses an unknown device ABI,
records the compilation stack and source/image identities, and writes a new
`module.json` only when both images are built. Serving can use the CUDA driver
without Python, Torch, CuTe, TVM-FFI, or the CuTe host object. Compilation, actual
driver ABI checks, GPU numerical tests, model quality and performance must each
be recorded separately; unavailable checks are not passing validation.
