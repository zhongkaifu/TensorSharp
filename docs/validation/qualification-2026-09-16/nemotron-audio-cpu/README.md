# Nemotron Omni audio implementation — CPU evidence

Status: implementation in progress; release remains unqualified. A new TensorSharp-owned Parakeet encoder and sound projector execute the official companion weights. The audio injector now encodes all attached clips, preserves marker order, and queues projected rows. This does not establish spoken-audio answer quality, GPU operation, or acceptable latency.

## Recorded lanes

| Lane | Result | Scope |
| --- | --- | --- |
| r5 portable | 14 passed, 0 skipped | Six independent official-reference clips on managed/GGML CPU, F32/BF16; frame lengths, attachment planning, malformed WAV rejection |
| r5 trained BF16 compute | **Failed**: 3,794/8,064 values | Actual official 24-layer audio encoder and projector, 21 mel frames / 20 valid frames, three output rows |
| r6 diagnostic | Same BF16 failure retained | Module traces isolate first propagated difference to a BF16 rounding boundary |
| r8 expanded portable | 19 passed, 0 skipped | Both 8/128-mel official configurations; actual audio-only two-clip WAV/mel/encoder expansion and sliced queueing with independent request retention |
| r7 trained F32 compute | 8,064/8,064 values passed | Same official BF16 weights, explicit F32 compute, official reference converted to F32; maximum absolute difference 2.1457672119140625e-6 |

F32 gate: `abs(actual-expected) <= 2e-5 + 2e-4*abs(expected)`. BF16 gate: `<= 0.001 + 0.01*abs(expected)`. Both were set before the first implementation result and remain unchanged. r7 is a separate precision configuration and does **not** clear the BF16 failure. The small fixtures' BF16 pass does not qualify trained BF16 compute.

At the first propagated trained BF16 mismatch (layer 0 pointwise convolution, flat index 4549), exact FP64 gives −1.2851561978459358. That rounds to TensorSharp's −1.28125, whereas the official BF16 kernel produces −1.2890625. Later differences amplify; no gate has been waived.

## Pinned inputs and executable identity

- Official source: [NVIDIA Nemotron Omni](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/tree/e5e9932441de940c9a62185c870ea5bcd4cd24e2), revision `e5e9932441de940c9a62185c870ea5bcd4cd24e2`.
- Independent encoder modules: [Transformers Parakeet](https://github.com/huggingface/transformers/blob/da7234ac435f6d7c75d8b88d1ac32f53fb1f19a6/src/transformers/models/parakeet/modeling_parakeet.py), source SHA256 `457f93a49db264cb4b7266ba81f45066232119adcc5fae2d96861e16e05eda07`. The harness executes these official classes and NVIDIA's projector with PyTorch 2.8.0+cu128 on CPU. It does not reimplement their forward functions.
- ggml remains unchanged at `456172ec733a135778adcd32d00e576a58232e45`. These audio changes contain no native modification. The derived app uses frozen r4 `libGgmlOps.so`, SHA256 `7eaf98656edcd043884f3eb776bed68abbee29f5cb446e237bfa16afa5598af7`.
- Per-run `manifest.json` identifies the exact Models DLL, test DLL and native library. The app closure derives from `/workspace/ts-codex-20260916-r4/repo/InferenceWeb.Tests/bin/Release/net10.0`; only Models and the isolated test DLL are replaced. Source hashes in `audio-source-sha256.json` describe the narrow audio sources at the r5–r7 build. Other shared-checkout changes are not qualified by this record.
- Model asset root: `/workspace/models/nemotron-omni/audio-e5e9932/`. All 689 BF16 encoder/projector tensors were extracted from the pinned official shard; 24 unused I64 batch-normalization training counters were omitted. Byte-range SHA256 `ae938e5da380d218243a6a0daeeb94f4a72c95d9c85ee154b7acee086d0a9a8e` is a locally measured extracted-content checksum, not a claim to verify the entire shard's LFS checksum.
- Original BF16-compute companion `mmproj-audio-bf16.gguf`: 1,248,398,240 bytes; SHA256 `a7b1c65c38255d119bfceea3d7d83ba9f6aa32d3d4651136bf8ce00ef4e81bf5`.
- Separate F32-compute companion `mmproj-audio-bf16-f32compute.gguf`: same weight bytes and size, metadata `nemotron.audio.compute_bf16=false`; SHA256 `61b7a35767ef764a8dbfd10d06df9cae18996f1d9f9fe47ce9465a7e978bc872`. Original assets were preserved.

The reference scripts preserve their exact VM paths for replay. `prepare.py` creates the original companion; `prepare_f32.py` creates the separate F32 configuration from the already downloaded raw bytes. Both require the recorded configuration/header files. Use an empty output directory when replaying asset preparation, and retain original evidence.

## Commands and limits

Managed build: `dotnet build TensorSharp.Models/TensorSharp.Models.csproj -c Release --no-restore -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -m:1 /nodeReuse:false` (0 errors).

For each isolated VM lane, run `python3 /workspace/ts-nemotron-audio-derived-rN/run.py`; its manifest records the exact vstest command. Trained comparison: `python3 /workspace/ts-nemotron-audio-derived-rN/probe.py`, with corresponding `Probe.cs` recording asset/ref paths, tolerance and all output comparisons. r7 uses `trained_reference_f32.py`; r5/r6 use the original BF16 reference. Tests ran on CPU, with no CUDA qualification or implicit skips.

The trained input is deterministic synthetic mel data, not speech. Its full output is compared, but no language model decode, frontend numerical gate, full request cancellation, simultaneous production concurrency, long audio or media-history quality is established. Timings in probe output come from a shared VM, include no controlled repeated comparison, and are **not performance passes**. Current CPU execution is slower than the independent reference. Full spoken-audio and performance qualification remain outstanding.
