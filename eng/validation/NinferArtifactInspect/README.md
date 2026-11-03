# Typed NInfer v2 metadata inspection

This console references only `TensorSharp.Runtime`. It uses the read-only
`NinferArtifactReader` without a model factory, CUDA or weight conversion.

```sh
dotnet build eng/validation/NinferArtifactInspect -c Release
dotnet eng/validation/NinferArtifactInspect/bin/Release/net10.0/NinferArtifactInspect.dll \
  --artifact /workspace/hf/ignis/qwen3_8_27b_nvfp4full-v2.ninfer \
  --expect-directory-sha 222b7540e961a6455fff9f03cce451860e4d248785a4ef0a805f538c8eb82658 \
  --known-sha256 abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af \
  --output artifacts/ninfer-typed-inspection-r1.json
```

The output path is normalized once before validation and writing. It must be
fresh under `artifacts/` or `docs/validation/`, without symlinked parents. Existing files, directories and dangling links are refused.
The artifact is never written. `--help` reads no artifact and creates no output.
Exit 0 means metadata inspection completed, 1 means inspection failed, and 2
means argument/output refusal. Failed inspections retain `complete:false` and
their error in the fresh JSON output when it can be written safely.

Initialization validates the closed v2 schema, typed dimensions and formats,
checked geometry, alignment, ordering, file bounds and positive finite trailing
NVFP4 divisors. JSON includes original offsets/shapes/sizes/plane geometry and
each divisor's exact raw bits, directory/prefix hashes, resource inventory,
actual read counters and loaded managed assembly hashes. Initial artifact I/O
is only 16 prefix bytes, bounded JSON and four bytes per NVFP4 tensor. Native
partial reads can require additional calls; byte counts report actual I/O.

`--known-sha256` is supplied evidence, explicitly **not verified by this tool**.
An independent full-file checksum and stable file identity must be established
outside performance measurement. This console's path/length/time/link snapshot
does not replace an inode/whole-payload identity guard. Its managed assembly
checksums are separate from artifact I/O and never read all weight bytes.

Resources are not interpreted or automatically bound as tokenizer/config/chat
templates. To inspect a specific resource, repeat `--resource EXACT_NAME` (at
most 32 distinct names); `--max-resource-bytes N` bounds each read (default
16 MiB). Only the selected name, size and SHA256 are emitted, **never content**.
A resource name is an object lookup key, not a filesystem extraction path.
`--max-json-bytes` defaults to 16 MiB and `--max-objects` to 100,000.

NInfer v3, VI v1 and known unsupported geometries are refused. Codes and scale
values are not scanned during initialization. A successful metadata inventory
does not establish inference/import compatibility, original-checkpoint quality,
tokenizer correctness or performance. Factory/native serving remains
unimplemented. Real model validation requires a fresh build/source certificate;
earlier S4/S5 test results do not certify a changed Runtime assembly.
