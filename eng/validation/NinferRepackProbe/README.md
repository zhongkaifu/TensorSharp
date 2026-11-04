# Lossless original NVFP4 GPU repack probe

```
python3 eng/validation/build-ninfer-repack-module.py --output artifacts/ninfer-repack.fatbin
dotnet NinferRepackProbe.dll MODULE OUTPUT_JSON [--artifact MODEL.ninfer] [--all-target-nvfp4]
```

This checks encoded-byte conversion only. It does not evaluate model answers,
projection arithmetic or startup performance. It fails on an unavailable GPU
or native module; those scenarios are not silently skipped.

Eleven synthetic matrices cover all three declared row mappings, m128/k64
boundary shapes, columns not divisible by the engine's 256-value GEMM stage,
the bounded maximum K=65536, and actual target projection geometries including
the largest 34816x5120 gate/up matrix. A scalar logical-code writer and a
five-axis original-scale writer generate the source independently. The
expected TS planes use source-to-destination row enumeration, opposite to the
CUDA gather. They must also match the existing real-reader CPU converter.

Every device destination byte is compared exactly to the CPU planes. An
independent inverse must recover every original code/scale byte. Two distinct
output poisons exercise A/A, both output guards must remain unchanged, and the
full original source including its four-byte divisor and guards must remain
unchanged. Same-base and shifted buffer overlap must be rejected before launch.
Kernel event intervals are diagnostic conversion-only observations; transfers,
file reads, validation and whole startup are not included.

With `--artifact`, the probe additionally uses four actual target matrices:
GDN input, gate/up, down, and an NVFP4 attention-input matrix. The artifact's
reader, typed target plan and CPU plane converter are used without changing the
file. `--all-target-nvfp4` explicitly expands this to all 247 original NVFP4
target projections. Without `--artifact`, actual-model conversion is NOT_RUN.
Actual artifact identity records contain file size, timestamps, directory and
prefix hashes; the probe does not recompute the entire model-file hash.

The standalone module accepts only complete m128/k64 matrices whose original
object, including the four-byte divisor trailer, fits 128 MiB. The caller must
validate original scale bytes (<0x7f), positive finite divisor and exact expected
divisor bits before a device launch. The kernel performs only integer byte
permutation and leaves calibration in the projection descriptor. Generated
module images, JSON and logs belong under ignored `artifacts/` or
`docs/validation/`.
