# Bounded NInfer inventory inspection

This standard-library Python tool reads an artifact without loading a model or
initializing CUDA. It implements the published Ignis **v2** container rules in
TensorSharp-owned code; it does not copy or modify a kernel or ggml source.

```sh
python3 eng/validation/ninfer-inspect.py \
  /workspace/hf/ignis/qwen3_8_27b_nvfp4full-v2.ninfer \
  --known-sha256 abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af \
  --output artifacts/ninfer-inspection-r1.json
```

The output must be a fresh file under `artifacts/` or `docs/validation/` with no
symlinked parent. The artifact is never written. A separate full-file checksum
can establish artifact identity outside a benchmark measurement window;
`--known-sha256` records supplied evidence and **does not recompute or verify it**.

The v2 inspection checks the exact closed JSON schema, unique object names,
unsigned integer types and checked 64-bit geometry, ordering/nonoverlap,
alignment, file bounds and exact encoded sizes for direct BF16/FP32/I32/I64,
Q4/Q5/Q6G64 and W8G32 row-split, NVFP4 block-scale and FP8 row-scale tensors.
For each NVFP4 tensor it reads just its trailing four-byte FP32 divisor and
requires a positive finite value. It records header/directory hashes, each
object's geometry, bytes actually read, and unchanged file identity before and
after the inspection. Trailing unclaimed bytes are reported, as allowed by the
reference reader.

NInfer v3, VI v1 and NInfer v1 magic are recognized but return
`UNSUPPORTED_SCHEMA`. Known trellis and row-interleaved layouts are only bounded
opaque inventory and return `UNSUPPORTED_TENSOR_GEOMETRY`; their encoded size
is deliberately not guessed. Unknown formats/layouts are refused. Exit codes
are 0 for validated v2 structure and divisors, 2 for an unsupported format or
geometry, and 1 for invalid data or an inspection error.

An exit code of 0 is **not inference support, numerical equivalence, or model
quality validation**. Codes, per-block scales, resource contents and full
payload hashes are not read or validated. NInfer NVFP4 nibble order, scale
swizzle and global divisor differ from TensorSharp's current GGUF path; an
inventory must not be relabeled as a compatible GGUF weight descriptor.
W8G32 head/embedding and BF16 exception execution also need explicit support.

Schema reference: [Ignis artifact reader at e600a7d9](https://github.com/gpillon/ignis/blob/e600a7d9e0d5921d767a6cb464373583b3cdcaf4/crates/artifact/src/lib.rs).
Its geometry and strict parser are the specification for this independent
implementation. The source revision and captured source hash are in every
successful inspection result.

Run CPU tests without downloading a model:

```sh
python3 -m unittest discover -s eng/validation -p test_ninfer_inspect.py -v
```

Tests use tiny generated files plus one explicitly sparse fixture to verify
bounded reads. Such a fixture is never evidence of inspecting an actual model.
The pure `validate_directory(..., read_at=None)` helper supports separately
captured real metadata but leaves every NVFP4 divisor `NOT_READ_METADATA_ONLY`
and does not claim a full-file inspection.
