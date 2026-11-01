# HostMoE Metal shared-branch overlap

`TS_HOST_MOE_METAL_OVERLAP` is a startup opt-in for the common segmented host-expert executor:

| Value | Execution |
| --- | --- |
| Unset or `0` | Existing sequential execution |
| `1` | Submit the independent Metal prefix, run the existing CPU experts, then drain |
| `2` | Run the same prefix synchronously before CPU experts, as a serial-split diagnostic |

Eligibility follows capabilities rather than model names, layer indices or expected expert counts. Execution requires an actual Metal backend, one configured device, one decode row, and no GPU-reference verification. All three host inputs are downloaded before prefix submission.

The pending host result must be a fresh contiguous F32 non-view leaf, pinned as both input and output. Boundary tensors and their storage roots remain outputs. The cached original-order prefix must start with a complete matrix multiplication and end immediately before `ADD(pending, last_MUL)`. Only the existing matrix multiplication, SiLU/sigmoid, multiplication and metadata/contiguous-copy grammar is accepted, with at most 64 nodes. Allocation and alias checks run after binding refresh. Unsupported graphs retain ordinary execution; no graph reordering or wider arithmetic grammar is introduced.

The CPU expert implementation, quantization and reduction order stay unchanged. A failed or unsupported fast CPU call drains outstanding Metal work before generic fallback. The drain also runs during exception unwinding, and sticky backend errors fail the pass before result upload or dependent graph submission.

Callers must serialize use of a shared native backend until the complete forward call returns. The ordinary one-model server holds `GpuComputeLock` across the complete inference step, including the CPU work and drain. The single-device check counts configured devices; it does not exclude another caller. Separate model instances and raw native callers sharing a backend need one external lock. This option does not add concurrent-model support to the existing Metal bridge.

`[HOSTMOE-METAL-OVERLAP]` reports eligibility, rejected capabilities, cached cuts, submitted nodes, failures, fallbacks and aggregate wall timings. These timings include host encoding and waits; they are not GPU-device durations. A run with no submissions does not establish an overlap speedup.

The model-free `host-moe-metal-overlap-plan-drain` CTest covers varied hidden/shared dimensions and routing counts, pending-result representations, graph/fusion boundaries, allocation aliases and drain lifetime/error handling. Full-model qualification must use identical weights, quantization, placement, context, CPU thread settings, prompts and generation settings. Validate actual vocabulary widths and finite full logit rows before comparing repeated interleaved latency measurements. Generic naming does not establish support or a performance benefit for every model.

## Reproduce correctness

Build `eng/ForcedLogitProbe` and put the matching `libGgmlOps.dylib` beside its DLL. Then run a dry plan using existing weights:

```sh
python3 eng/validation/moe-metal-overlap-quality.py \
  --model '<existing-GGUF-or-first-shard>' \
  --host-layers 4 --steps 64 --expect-path active \
  --output artifacts/moe-metal-quality-new
```

Add `--execute` to launch the two sequential model runs. The tool never builds or downloads weights and refuses an existing output directory. It holds expert and generic CPU threads at 12, context at 8192, F16 KV, prefill chunk 512, speculation off and the host-to-device prefill threshold at 128. Host-layer count is explicit. `--expect-path fallback` checks that overlap remains inactive; use zero host layers for a dense model. Built-in prompts are five plaintext numeric-equivalence probes. `--cases` accepts the same five named cases with model-specific chat formatting when needed.

The candidate must match every full-vocabulary float32 file byte for byte, with independently finite rows, identical argmax histories and actual asynchronous engagement for an active lane. The probe's relative-L2 tolerance alone is insufficient. Source, binary, model-file statistics, unchanged ggml revision and requested versus observed path are recorded. Generated vectors use `2 * 5 * steps * vocab_size * 4` bytes, potentially over 1 GB. Quality-run duration includes vector capture and comparison and is not inference performance evidence.

Run model-free oracle checks with:

```sh
python3 eng/validation/tests/test_moe_metal_overlap_quality.py
```

The scheduling idea was informed by [KTransformers' CPU-expert/shared-GPU/join sequence](https://github.com/kvcache-ai/ktransformers/blob/c9873312643b615a717832946fba0346e69ab035/archive/ktransformers/operators/experts.py#L742-L749) and [MLX's explicit command-buffer completion and lifetime handling](https://github.com/ml-explore/mlx/blob/bd63d3803008edda03ce74080955d9879e2ef1eb/mlx/backend/metal/device.cpp#L456-L587). This implementation uses TensorSharp-owned code and the unchanged pinned ggml backend; it does not import either framework's kernels.
