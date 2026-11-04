# Original NInfer W8 host-placement component check

This separate tool compares `CudaNinferHostEmbedding.GatherRows`, its pinned F32
asynchronous upload, and the existing `CudaNinferKernels.EmbedW8` kernel on the
same original W8 payload. It does not change serving defaults, run sampling,
time inference, or measure full-model memory use.

```sh
dotnet build eng/validation/NativeHostEmbeddingProbe/NativeHostEmbeddingProbe.csproj \
  -c Release -p:TensorSharpSkipCudaPtx=true
dotnet eng/validation/NativeHostEmbeddingProbe/bin/Release/net10.0/NativeHostEmbeddingProbe.dll --self-test
```

The managed-only build requires the already prepared backend output described
by the backend project. It does not compile or replace native assets. The
self-test performs no CUDA calls. It independently checks every finite F16
scale (63,488) crossed with every signed W8 code (256), plus all GPU fixtures,
against integer/exponent F16 decoding and adjacent-BF16 distance comparisons
with nearest-even ties and signed-zero preservation. The exhaustive scale sweep
is **CPU coverage**, not GPU coverage.

Run GPU checks only when the VM's other workloads are stopped. Use a new
runtime folder and a fresh ignored evidence directory for each source snapshot:

```sh
dotnet /ABSOLUTE/IMMUTABLE_RUNTIME/NativeHostEmbeddingProbe.dll \
  --execute --module /ABSOLUTE/IMMUTABLE_NATIVE/tensorsharp_q35_ninfer.fatbin \
  --out /ABSOLUTE/artifacts/host-embedding-component-r1
```

An explicit fatbin is required, avoiding PTX-to-sibling-fatbin selection. Module
identity uses a strict 64-hex-digit digest, insensitive only to hex letter case.
The file, loaded image, tool assembly, and backend assembly identities are
recorded. The original payload, token IDs, independent reference, and complete
guarded CPU/upload/GPU output vectors are retained. A failure remains a failed,
incomplete capture; do not replace it or adjust exact criteria to pass it.

The nine GPU fixtures use widths 128, 256, and the model's actual width 5120,
each with batches 1, 7, and 33. The original layout is contiguous signed-byte
codes `[vocabulary,width]` followed by little-endian F16 scales
`[vocabulary,width/32]`. The 33-row width-256 fixture crosses all 256 W8 codes
with all 32 selected finite scale patterns. Those patterns include both signed
zeros, both signs of minimum/maximum subnormals, minimum normal, ordinary scales,
BF16 midpoint witnesses, and maximum finite. Other fixtures vary each group and
row. IDs are nonsequential and include duplicates where the batch permits them.
Actual code/scale-pair and tie coverage is reported per fixture.

Each fixture runs in two host/module lifetimes. A/B/A NaN output poisons require
every active value to be overwritten and bitwise equal to the independent
reference, to the other placement, to the first pass, and to the first lifetime.
All outputs must be finite BF16 values in F32 storage. GPU weight/ID allocations,
pinned ID input, pinned output guards, and both device output guards must remain
intact. The borrowed weight source is disposed before gathering; repeated calls
must perform no further source reads. Upload completion is synchronized before
the single pinned buffer is reused or freed.

The real `248320 × 5120` W8 payload size, 1,350,860,800 bytes, is recorded as
arithmetic only. This synthetic test does not prove that serving frees that
amount, that a 250K request fits, or that complete-model logits/quality are
unchanged. Those require separate full-model before/after runs with the same
artifact, inputs, module identities, and sampling policy. No dequantized full
embedding, activation quantization, sampling change, upstream ggml modification,
or end-to-end performance conclusion is part of this component check.
