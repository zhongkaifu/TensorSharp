# NInfer Qwen38 import format probe

This standalone, CPU-only validation tool exercises the **public** real-artifact
`NinferArtifactReader → NinferQwen38TargetPlan.Create → NinferQwen38Frontend.FromReader`
path. It does not construct reduced metadata, call private parsers, enable a model
factory, apply generation settings, render the chat template, or load CUDA.
Default invocation only prints a plan.

```sh
dotnet build eng/validation/NinferQwen38ImportProbe/NinferQwen38ImportProbe.csproj \
  -c Release -o artifacts/import-probe-managed \
  -p:CudaNvcc=/nonexistent -p:TensorSharpSkipGgmlNative=true \
  -p:TensorSharpSkipMlxNative=true -p:UseSharedCompilation=false -nr:false
dotnet artifacts/import-probe-managed/NinferQwen38ImportProbe.dll --self-test
dotnet artifacts/import-probe-managed/NinferQwen38ImportProbe.dll --self-test-output
python3 eng/validation/NinferQwen38ImportProbe/test_cli.py \
  --dll artifacts/import-probe-managed/NinferQwen38ImportProbe.dll
```

The managed build can reference CUDA assemblies transitively, but this executable
contains no CUDA API call and does not run inference. For an actual artifact:

```sh
dotnet RUNTIME/NinferQwen38ImportProbe.dll --execute \
  --artifact /workspace/hf/ignis/qwen3_8_27b_nvfp4full-v2.ninfer \
  --directory-sha 222b7540e961a6455fff9f03cce451860e4d248785a4ef0a805f538c8eb82658 \
  --known-full-sha abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af \
  --out /dev/shm/artifacts/ninfer-import-probe-FRESH
```

`--known-full-sha` is a supplied, **unverified** statement. The tool does not
rehash the 19GB file. The executor must separately bind a prior full-file hash,
current inode/stat, source/assemblies, and read-only pre/post identity. The tool
checks length/mtime/creation/link metadata only. Existing output paths and aliases
are refused; errors are saved only inside a newly owned output directory.

The output includes all six original resource hashes and per-stage reads. Its
generation metadata stays separate from tokenizer BOS behavior and is never
applied to sampling. `tokenizer-cases.input.json` preserves the embedded 22 UTF-8
test inputs. Output cases contain original UTF-8/base64, IDs with and without
`addSpecial`, decoded text and raw token bytes. No BOS/EOS is injected. The
`token-bytes.jsonl` file covers every valid ID **0..248076**, with exact raw bytes
and its SHA256. The aggregate framing is `uint32LE(id) || uint32LE(byteLength) ||
bytes`, in increasing ID order. No padded head IDs are invented.
`single-byte-token-ids.json` maps all 256 byte values to their exact single-byte
vocabulary IDs for an independent arbitrary-ID decoding oracle.

An external oracle must use the original HF `tokenizer.json`, compare exact
encode IDs with `add_special_tokens=false`, and decode with
`skip_special_tokens=false, clean_up_tokenization_spaces=false`. Compare every
token's raw bytes as well. NFC/added-token cases include combining marks around
`normalized=false` added tokens. Internal decode/NFC roundtrip checks are not an
HF-ID equivalence certificate; the HF oracle remains **NOT_RUN** here.

By default, all 40 complete 128-row tiles of `text/layers/0/gdn/output`
([5120,6144], 17,694,724 encoded bytes) are converted. An independent inverse
uses logical K positions and five-axis source scale order, without calling the
production converter's mapping/offset helpers. Every original code/scale byte
is compared and both complete planes receive source/inverse streaming hashes.
Original weight-divisor bits are checked separately and never combined with an
activation divisor. Unsupported signed/NaN E4M3FN scales fail without clipping.
The expected extra plane reads are **35,389,444 bytes / 161 calls**, including
the converter, independent source audit and final four divisor bytes.

Optional `--read-calibrations` explicitly reads 247 distinct input-scale scalar
descriptors (**988 bytes / 247 calls**), checks original raw FP32 bits are positive
finite, and keeps them separate from the 247 weight divisors. The immutable plan's
original `NotRead` declaration is not rewritten. Resource payload reads are
12,837,177 bytes / 6 calls; target plan creation adds no I/O after reader startup.

`complete=true` means the selected import-format checks completed. It is **not**
weight-compute equivalence, HF-tokenizer qualification, model quality, runtime
compatibility, or performance evidence. GateUp/GQA matrices, model inference,
draft/vision binding, actual-artifact success and GPU runs require separate
evidence; this first tool exposes no extra matrix selection or inference mode.
