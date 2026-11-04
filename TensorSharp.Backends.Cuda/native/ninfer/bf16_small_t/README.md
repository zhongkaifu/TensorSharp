# Isolated BF16 small-T component

This component is not wired into model execution or a default policy. It is an
offline candidate for the two BF16 exception projections used by the qualified
Qwen NInfer model, only at eight physical input rows:

| Role | Output rows N | Input columns K | Grid | Block |
| --- | ---: | ---: | ---: | ---: |
| GQA input | 14336 | 5120 | 896 | 128 |
| Output projection | 5120 | 6144 | 320 | 128 |

Alpha/beta projections, other token counts, quantized projections, and ggml are
outside this component. No current serving path loads this module.

`upstream/` contains unchanged headers and the Apache-2.0 license from
[Ignis e600a7d9](https://github.com/gpillon/ignis/tree/e600a7d9e0d5921d767a6cb464373583b3cdcaf4).
The manifest pins both Git blob identities and SHA-256 hashes. The owned CUDA
file explicitly instantiates the original global kernel and original production
schedule. It does not copy or edit its accumulation, reduction, input decode,
PDL synchronization, or phase order.

The one owned output policy stores the kernel's computed `float` directly.
Inputs and row-major weights are BF16; outputs are compact token-major F32.
Weights must already have the desired TensorSharp row order. In particular,
the GQA weights loaded by TensorSharp already contain its Q/gate interleaving;
this component performs no row permutation. There are no extra output casts.
The caller retains the original consumer boundary: GQA rounds after projection;
the output projection adds its unrounded F32 result to the residual before the
existing BF16 round and normalization.

A second arm instantiates the original contiguous BF16 output policy. This is
a component reference for the original epilogue, not a replacement for the F32
projection contract. A rounded BF16 reference widened to F32 cannot serve as
the raw F32 output oracle.

Build from the repository root with a CUDA toolkit supporting `sm_120a`:

```sh
python3 eng/validation/build-ninfer-bf16-small-t.py --verify-only
python3 eng/validation/build-ninfer-bf16-small-t.py --out /absolute/fresh/ignored-output
```

The isolated build uses C++17, O3, FTZ false, FMAD true, line information, and
`sm_120a`; it produces PTX, fatbin, logs, and a write-once build manifest. The
paired fatbin sidecar stores the associated PTX SHA-256, following the existing
TensorSharp loader convention. It is not the fatbin's own digest.

Four actual mangled CUDA entry names are read from emitted PTX and written in
the build manifest. Each receives three by-value kernel parameters at byte
offsets 0, 8, and 16. The first two are eight-byte device pointers to BF16 input
and BF16 row-major weights. The candidate's third parameter is an aligned
eight-byte struct containing only the F32 output pointer. The original reference
uses a 16-byte struct containing BF16 output pointer, signed 32-bit N, and four
zero padding bytes. Dynamic shared memory is zero. Device-reported argument
offsets/sizes must also be checked by a GPU probe before launching.

Original `core/pdl.cuh` launches with the programmatic stream serialization
attribute (`cudaLaunchKernelEx`, driver attribute 6). Its entry dependency wait
is documented as a no-op without that attribute. Preserve the original PDL
path when qualifying the component; graph capture and actual node arguments
require device evidence.

Validation must keep three distinct claims separate:

* Compare the actual existing public `ProjectBf16` path, including its F32-to-BF16
  input cast, to the candidate on the same BF16 weights and source values.
* Compare the candidate's BF16-rounded results to the pristine BF16-output arm,
  with full output captures, aliases/guards, repeat and module-lifetime checks.
* Check raw F32 results against an independently computed FP64 dot product with
  a fixed forward-error bound, including cancellation. Legal F32 summation
  order changes do not imply bitwise equality to cuBLAS or model answer parity.

Timing must include the unchanged consumers and any required input cast. Output
projection residual buffers must be reset identically in both arms before each
replay; the reset is part of both measured chains. No isolated timing result
qualifies model quality, model speed, or automatic adoption. CPU source/ABI
checks alone do not claim that NVCC compilation, GPU execution, numerical
qualification, graph capture, model evaluation, or performance has passed.
