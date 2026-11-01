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

`[HOSTMOE-METAL-OVERLAP]` reports eligibility, rejected capabilities, cached cuts, submitted nodes, failures, fallbacks and aggregate wall timings. These timings include host encoding and waits; they are not GPU-device durations. A run with no submissions does not establish an overlap speedup.

The model-free `host-moe-metal-overlap-plan-drain` CTest covers varied hidden/shared dimensions and routing counts, pending-result representations, graph/fusion boundaries, allocation aliases and drain lifetime/error handling. Full-model qualification must use identical weights, quantization, placement, context, CPU thread settings, prompts and generation settings. Validate actual vocabulary widths and finite full logit rows before comparing repeated interleaved latency measurements. Generic naming does not establish support or a performance benefit for every model.
