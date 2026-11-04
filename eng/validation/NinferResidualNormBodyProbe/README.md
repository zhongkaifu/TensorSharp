# Exact residual/norm full-body integration check

This tool produces bounded same-history evidence in **three independent processes**: actual old runtime (`old`, main4242), new runtime with residual fusion off (`off`, mainEA), and the same new runtime on (`on`, mainEA). It uses public target/draft APIs and reflection for read-only diagnostics; old runtimes need no new friend declaration. It changes no sampler, kernel, score or ordinary-forward oracle.

Build and CPU-only contracts:

```sh
dotnet build eng/validation/NinferResidualNormBodyProbe/NinferResidualNormBodyProbe.csproj -c Release -p:TensorSharpSkipCudaPtx=true
dotnet eng/validation/NinferResidualNormBodyProbe/bin/Release/net10.0/NinferResidualNormBodyProbe.dll --self-test
```

Stage the probe DLL, deps and runtimeconfig **into a private copy of each pinned runtime**, retaining that runtime's actual TensorSharp DLLs/native assets. Do not overwrite the old runtime with newly built TensorSharp dependencies. Actual loaded managed DLL and native-image hashes are retained. Root alone runs GPU/model commands. No model/CUDA work happens without `--execute`.

The native manifest is a pinned JSON file with schema `ninfer-residual-body-native-pins/1` and exactly eight `files` entries `{ "role": ROLE, "path": CANONICAL_ABSOLUTE_ACTUAL_IMAGE, "bytes": LENGTH, "sha256": SHA }`. Roles are `main`, `helper`, `attention`, `verify8Attention`, `a16`, `draft`, `repack`, `w8`. Pin the actual loaded fatbin, rather than its PTX sibling or a historical filename. Main is strictly old4242d3e8891c6b8fec2fcd216069c3ce40698cd63416f96f6bb084614324d76b or newea0e322a36c48211750e0f569bd12c44f9d62caa6d3249c37f1547679bdccbea. A later source rebuild requires explicit new qualification/rebinding, not a silent expected-hash replacement.

All three processes keep the same artifact, body-input JSON and numerical environment/native roles, with explicit `TS_Q35_NINFER_VERIFY=1`, `TS_Q35_NINFER_DECODE_GRAPHS=1`, `TS_Q35_NINFER_VERIFY8_GEMM=1`. Keep original full-head native draft, qualified vector A16, lossless repack and Verify8 attention enabled. Set `TS_Q35_NINFER_RESIDUAL_NORM=0` for old/off and `1` for on. Original-prefill, FI, ExactVerify, shortlist and chunk overrides are refused. Context40960 and UB2048 are fixed. Existing attention/A4 arithmetic is retained in all arms.

```sh
# Repeat in separate pinned runtime directories for MODE=old,off,on.
dotnet /ABS/PRIVATE_RUNTIME/NinferResidualNormBodyProbe.dll --execute --mode MODE \
  --artifact /ABS/ORIGINAL.ninfer --expect-directory-sha DIRECTORY_SHA \
  --inputs /ABS/original-body-inputs.json --expect-inputs-sha INPUT_SHA \
  --native-pins /ABS/MODE-native-pins.json --expect-native-pins-sha MANIFEST_SHA \
  --out /ABS/artifacts/residual-body-MODE

# CPU-only independent raw-byte comparison, after all three producers complete.
dotnet /ABS/PROBE/NinferResidualNormBodyProbe.dll --compare \
  --old /ABS/artifacts/residual-body-old --off /ABS/artifacts/residual-body-off \
  --on /ABS/artifacts/residual-body-on --out /ABS/artifacts/residual-body-comparison
```

The complete protocol is bases127/255/1023 × keeps0/1/8 × two reset passes; each case performs Verify8→Commit→Decode1→seven full-head proposals. Base255 additionally covers Verify1→Commit1 (owned51200-byte pending)→Verify8→Commit1 in each pass. Base255/keep8 also clones and decodes both copies identically. One separate2048-token prefill between reset passes must preserve the existing candidate graph. Every target argmax is checked against every physical F32 logit, padded vocabulary winners are refused, all caller tail guards are checked. Selector confidence1 is only an admission marker, not a probability.

Exactly20 Verify8,2 Verify1,22 commits,24 decodes,20 proposal calls and326 full248320-value F32 head rows are mandatory, plus22 complete BF16 tap captures. Raw upper bound before CAS is332,103,680 bytes per producer, hard cap512MiB. Same-address process reset A/A and all clone outputs are exact. The on arm must use one candidate graph:20 replays/completed verifies,125 captured fused GEMMs,144 Plain submissions,1 initial-null and3 legacy BF16-post submissions. Counters are host submissions including capture, not per-layer GPU execution counts. Verify1 is excluded.

State snapshots read every Conv and recurrent State allocation and only live committed KV and logical draft-ring ranges. Each digest retains allocation role, relative byte offset and byte length. Pending/frontier metadata is retained; allocator addresses, future/unwritten KV, unused RingPos and namespace strings are excluded from numeric equality. Actual numerical namespaces/policies remain in the producer report. This is a complete byte/state integration check for the fixed campaign, not an independent model-quality or speed evaluation. No skipped case is a pass.

`completed-result.json` and its SHA sidecar are written once, after device/context disposal and input/native/DLL/CAS rechecks. The comparer admits complete fixed inventories, rehashes every retained CAS file, compares actual bytes and state metadata, and requires old→off equality before off→on. Preserve a failed progress result as failed evidence.
