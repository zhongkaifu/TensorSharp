# Actual NInfer target metadata binding probe

This is a separate tool; frozen `NinferArtifactInspect` remains unchanged. It
opens the actual artifact with `NinferArtifactReader`, then actually calls
`NinferQwen38TargetPlan.Create(reader)`. It checks the complete 906-tensor,
64-layer profile, original reader ownership, nine large BF16 exceptions, W8
endpoints, 247 separate unread activation-divisor descriptors and six resource
descriptors. Target-plan creation must add zero artifact reads.

```sh
dotnet build eng/validation/NinferQwen38TargetInspect/NinferQwen38TargetInspect.csproj \
  -c Release -o artifacts/ninfer-target-inspect-runtime \
  -p:CudaNvcc=/nonexistent -p:SkipGgmlNativeBuild=true -p:SkipGgmlNative=true \
  -p:SkipMlxNativeBuild=true -p:SkipMlxNative=true -p:UseSharedCompilation=false -nr:false
dotnet artifacts/ninfer-target-inspect-runtime/NinferQwen38TargetInspect.dll \
  --artifact /path/to/qwen3_8_27b_nvfp4full-v2.ninfer \
  --expect-directory-sha 222b7540e961a6455fff9f03cce451860e4d248785a4ef0a805f538c8eb82658 \
  --known-sha256 abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af \
  --output artifacts/ninfer-target-plan-actual.json
```

The output must be fresh with no symlink parents under `artifacts/` or
`docs/validation/`. `--known-sha256` is supplied provenance, never independently
verified by this tool. An external controller must bind the complete model SHA,
inode/stat and source/runtime identities before and after. The tool reports its
actual Models/Runtime/tool assembly hashes.

When reusing an already seeded native runtime, deployment controllers may use
the stricter `TensorSharpSkipCudaPtx`, `TensorSharpSkipGgmlNative` and
`TensorSharpSkipMlxNative` properties.
`TensorSharpSkipCudaPtx` requires prebuilt PTX in the output and deliberately
refuses an empty fresh local output; it is not a native-free bootstrap flag.

No weight conversion, calibration-content read, resource read, tokenizer
interpretation, exp/norm evaluation, CUDA call, model inference or performance
measurement occurs. Semantic flags describe the required future adapter: h/3
GDN kernels, raw A_log, distinct unit-offset/plain norms and source row orders.
Completion establishes metadata binding only. Unknown family, malformed target
profile or mismatched directory SHA refuses. Actual large-artifact success must
be recorded separately; local help/negative CLI and DTO tests do not prove it.
