# NInfer Qwen3.8 full-body diagnostics

This tool runs the actual `.ninfer` target through `Qwen35CudaExecutor` and the
TensorSharp native CUDA engine. Default execution requires `engine.HasDrafter`
to be false, including an embedded native drafter. The explicit
`--with-native-draft-taps` mode instead requires an attached native NInfer
drafter and exercises its taps/context maintenance during the same ordinary
target forwards. Neither mode calls Propose, Verify or Commit, changes the
model's numerical policy, runs a chat template or scores task quality.
Model/device absence is not a passing inference result.

Build with .NET 10:

```sh
dotnet build eng/validation/NinferQwen38BodyProbe/NinferQwen38BodyProbe.csproj -c Release
dotnet eng/validation/NinferQwen38BodyProbe/bin/Release/net10.0/NinferQwen38BodyProbe.dll --self-test
```

`--self-test` validates input, option-parser and drafter-admission contracts on CPU.
No mode performs GPU work unless
`--execute` is supplied. The native CUDA module and artifact must be configured
according to the model implementation; retain their hashes, source revision,
build log, unchanged ggml revision/status and actual GPU identity externally.

Prepare a fixed, shared token manifest from the actual artifact tokenizer. All
output directories must be fresh absolute paths under ignored `artifacts/` or
`docs/validation/` and have no symlinked parents:

```sh
dotnet NinferQwen38BodyProbe.dll --prepare-inputs \
  --artifact /models/qwen3_8_27b_nvfp4full-v2.ninfer \
  --expect-directory-sha ACTUAL_DIRECTORY_SHA256 \
  --out /absolute/artifacts/ninfer-body-inputs
```

The default lengths are 1, 7, 8, 9, 33, 128, 129 and 4096 tokens. These exercise
small-row dispatch boundaries and a larger prefill. `--lengths` explicitly
selects other exact token counts; `--context` must also fit the eight-token
continuation (maximum 262144). For example, use
`--lengths 131072,256000 --context 262144 --ubatch 4096` for long diagnostics.
This constructs repeated raw token streams, so it is a numerical/capacity input,
not a meaningful long-context quality test. It does not silently shorten tokens.

Run one GPU producer at a time:

```sh
dotnet NinferQwen38BodyProbe.dll --execute \
  --artifact /models/qwen3_8_27b_nvfp4full-v2.ninfer \
  --expect-directory-sha ACTUAL_DIRECTORY_SHA256 \
  --inputs /absolute/artifacts/ninfer-body-inputs/inputs.json \
  --out /absolute/artifacts/ninfer-body-ts \
  --context 8192 --ubatch 512
```

For a separate ordinary-forward comparison with native taps/context maintenance,
use the public executor's explicit native-draft configuration and add
`--with-native-draft-taps` to that command. For example, retain the same artifact,
input file, context, ubatch and numerical options as the target-only run, set
`TS_Q35_NINFER_VERIFY=1` and `TS_Q35_NINFER_DRAFT_PTX` to the actual pinned draft
module, and select a different fresh output directory. The mode flag does not
attach a drafter or silently enable environment options. A missing native
drafter or an external GGUF drafter fails admission before any fixture runs.
The default mode rejects an ambient native-draft configuration instead of
mislabeling it as a no-draft run.

`TS_Q35_NINFER_HOST_EMBEDDING=1` remains an explicit engine option. To check
placement and ordinary-tap preservation, compare all capture identities and
full-head vector SHA values from an actual host-embedding/native-taps run against
the frozen target-only producer using exactly the same fixed input manifest.
The fixture loops, two passes, clone check and numerical gates are identical in
both modes; no proposal or speculative-acceptance quality is evaluated. The
existing frozen 174-capture manifest still produces the same 174 positions.
CPU tests and an unexecuted plan do not establish this device comparison or a
measured memory saving.

Each case performs two reset/replay passes with the same chunking and fixed
teacher-forced continuation. It saves full head-vocabulary F32 logits at every
prefill chunk end and every continuation position. After prefill it copies the
slot and independently advances the original and clone by one token. Completion
requires bit-identical replay and clone results; any failure retains captures and
`complete=false`. A different chunking or a batched schedule may legitimately
select different arithmetic; this tool does not assert those paths equivalent.

`run.json` records token-manifest SHA, directory SHA, frontend resource hashes,
assembly hashes, TS_Q35 settings and each vector's length/SHA/valid-token argmax.
It records the requested mode, actual common/native/external drafter presence,
head policy, native draft payload/module hashes, numerical state namespace, and
actual host-embedding enablement/payload SHA/payload bytes/pinned staging bytes.
These are observations of the loaded engine; the requested mode alone is not
evidence of successful attachment or inference.
It also records the physical head argmax and whether a padded row would win;
this diagnostic does not silently change the engine's token masking policy.
The artifact itself is not rehashed in full. An external guard must establish
whole-artifact bytes and stable source/runtime/device identity before and after
the run. Per-call wall times include host work and synchronization and are
diagnostic observations, not qualified performance measurements.

For an independent Ignis oracle, use the **same artifact, original token manifest,
chunking and teacher-forced tokens**, capture its complete output vectors, and
export `ninfer-qwen38-body-logits/1` with its own engine/build provenance. Merely
renaming a TensorSharp run is not an independent oracle. The comparer verifies
the shared inputs and full vector files; it cannot establish the producer's
source identity by itself.

```sh
python3 eng/validation/ninfer-qwen38-body-compare.py \
  --reference /absolute/artifacts/ninfer-body-ignis \
  --candidate /absolute/artifacts/ninfer-body-ts \
  --output /absolute/artifacts/ninfer-body-comparison.json \
  --require-argmax --max-relative-l2 0
```

Thresholds are explicit. The example requests exact numerical L2 agreement; it
is not a suggested relaxed tolerance. Without a threshold the result is only
descriptive and `gate.passed` is null. Full padding rows remain in saved vectors,
while semantic argmax and error statistics use the valid tokenizer vocabulary.
All expected prefill/decode positions must be present. An incomplete run,
corrupt vector, nonfinite valid logit or different input/artifact/frontend
identity is refused. Synthetic comparer tests run with:

```sh
python3 -m unittest eng/validation/test_ninfer_qwen38_body_compare.py -v
```

This probe does not implement or qualify proposal generation, speculative acceptance, mixed
prefill/verify, attention KV precision parity across engines, multimodal input,
API streaming, tool calling, code execution, CUA, or general task quality. Those
require separate actual model evaluations. Passing this diagnostic is one
validation layer, not a quality-preserving end-to-end performance claim.
