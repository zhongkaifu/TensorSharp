# Qwen image chunk and merger diagnostic

Build with `-p:ValidationRuntimeDirectory=<frozen SDK publish>` to reference an
already qualified runtime. This does not compile Models or native sources. Run
`QwenImageLogitProbe --self-test` with CUDA masked before a GPU window.

The frozen input contains the original image request, expanded 317 input IDs and
image SHA. The original renderer and multimodal injector must reproduce all IDs.
This historical input precedes the Qwen4Exp generation-boundary repair: its
renderer removed the publisher's final two newlines. Use the pinned historical
runtime for this diagnostic. A repaired runtime must reject that old binding;
do not remove the identity gate or treat a 318-token prompt as the same input.
Natural continuation on the repaired HTTP host is a separate validation.
Run four isolated processes: `--chunk 32|317 --merger legacy|reference`. Both
processes explicitly set expert cache to zero, disable streamed CUDA expert
misses with `TS_HOST_MOE_DEVICE_MIN_BATCH=0`, and verify every expert layer remains
host routed. These controls isolate shape and activation, and are deliberately
unsuitable for a production performance comparison.

Each process saves the complete unmodified F32 distribution at P, P+1 and P+2.
Only after capture, its cloned sampling buffer is constrained to the two common
observed input tokens 271 and 101878. Native GDN/QSA/PLE state is synchronously
exported through an explicitly charged 8 MiB buffer. Live KV ranges, projected
embeddings and absolute MRoPE positions are hashed, excluding unused capacity.
No logits or model output are substituted into the model computation. This is
teacher forcing for diagnosis, not generated-answer quality evidence.

Use an external owned-process timeout of 600 seconds per arm; `--timeout 600`
also limits the request after load, but cannot interrupt an unmanaged model load.
Terminate only the exact owned process. Preserve failed arms and zero-ledger
cleanup evidence; never relax shape or cache-zero requirements after a failure.

`compare.py LEFT RIGHT --output OUTPUT` requires complete, disposed, identically
pinned arms and compares all logits plus state seams. A state hash difference is
localization evidence, not proof of incorrect arithmetic. First compare 32/317
within the same activation; then compare the two activations at the same shape.

Reference merger activation: Hugging Face Transformers v5.16.1
[`Qwen4ExpVisionPatchMerger`](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1557)
uses `nn.GELU()` (default exact erf). The transformer vision block MLP separately
uses the configured `gelu_pytorch_tanh`. `legacy` explicitly selects the prior
TensorSharp merger approximation using `TS_Q4E_VISION_MERGER_ERF=0`; `reference`
sets 1. Neither control changes the checkpoint, prompt or block MLP activation.
