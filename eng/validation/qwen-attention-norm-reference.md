# Qwen attention normalization reference

`qwen-attention-norm-reference.py` reads GGUF headers and two small F32 Q/K
normalization tensors. It reads existing `q4e-attn-diagnostic` dumps; it never
loads or executes a model, scans bulk model weights, or calls CUDA.

```powershell
python eng/validation/qwen-attention-norm-reference.py `
  --model C:/Works/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf `
  --log <capture-process.log> --output artifacts/<fresh-reference-directory> `
  --path solo --call 0 --layer 39 --position 1501 --caller 0
```

The log must contain readable single-token Q/K prenorm tensors and a tight-packed
F16 K-cache prefix for that exact selection. Native FNV hashes and file lengths
are checked before interpretation. Geometry comes from the log and model header;
file offsets come from tensor descriptors. The report records selected weight
hashes, capture hashes, model shard size/mtime before and after, and tool hash.
These checks do not constitute a new full-checkpoint hash.

Each head is normalized independently, multiplied directly by its norm weight,
then rotated over the first `rope.dimension_count` channels with NEOX half-pairs.
Unrotated channels are preserved from weighted RMS normalization. This tool is
limited to text positions; it does not implement image multi-axis IMRoPE.

Two references are saved: FP64 mathematical evaluation and an explicitly rounded
sequential FP32 approximation using host math functions. Neither emulates CUDA
`block_reduce`, `rsqrtf`, `powf`, trigonometry, or FMA contraction bit for bit.
For example, the FP32 path retains the two power operations used by the CUDA
RoPE source, while the FP64 path uses the mathematical base exponent directly.

`max_absolute_error_to_unrounded` includes normal F16 quantization error.
`max_absolute_half_difference`, differing coordinates, and half-ULP distances
instead compare the actual F16 row against nearest-even F16 reference values.
Signed-zero differences remain visible. Nonfinite inputs/results fail analysis.
A completed analysis does not mean model quality passed or that a one-ULP CUDA
difference is a bug. Compare solo and arena inputs and results before attributing
the cause.
