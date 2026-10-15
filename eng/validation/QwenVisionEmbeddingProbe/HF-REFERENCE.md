# Independent CPU F32 vision-component reference

`../qwen-hf-vision-oracle.py` prepares and executes a strictly scoped reference
using the **installed** `Qwen3VLVisionModel`. The local environment has no
`Qwen4ExpVisionModel`; this is an architecture-equivalent vision-component
comparison, not official full Qwen4 model ground truth. It imports no TensorSharp
code, allocates no language-model weights, downloads nothing, changes no packages
or upstream sources, and executes no GPU code.

The installed distribution metadata says Transformers **4.56.1**, while its
`__init__.py` reports **5.12.1**. This discrepancy is retained in the plan. Actual
model/config/processor/vision-helper/activation source files are SHA-pinned;
neither version string alone qualifies the reference. PyTorch is **2.5.1+cpu**.
An import failure is a failed/unavailable reference, never passing model coverage;
the tool will not repair or install dependencies in the user's environment.

The explicit target configuration is 27 blocks, hidden1152, intermediate4304,
16 heads, patch16, temporal2, spatial-merge2, 2304 learned positions, output2560,
empty DeepStack, pre-shuffle LayerNorm with epsilon1e-6, tanh block GELU and erf
merger GELU. These components agree structurally with the primary
[Qwen4-Exp model](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)
and [vision configuration](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py),
but the tool deliberately does not substitute that claim for measured parity.

Every one of the actual projector's 334 GGUF tensors must match its explicit
name, shape, F32/BF16 type, and byte extent. They fill all 333 HF state parameters;
only the two F32 temporal convolution slices become the corresponding slices of
one Conv3d parameter. Unknown tensors, missing tensors, shape changes, active
DeepStack, or quantized fallback conversions are refused. The 907 MB mmproj is
the sole loaded checkpoint. Its path/size/mtime and bounded header are checked,
not a fresh full checkpoint hash.

The exact opaque640x480 `card-0.png` is already size-aligned and undergoes no
resize. The installed independent Qwen2-VL processor performs mean/std0.5
normalization and patch/temporal packing, which must produce1200×1536 values and
grid`[1,30,40]`. F32 eager noncausal attention is explicit. The reference exports
finite F32 patch embedding, position-added input, first/last block output and
the complete300×2560 projection. It also exports the HF vision rotary H/W
positions; these are distinct from the language decoder's T/H/W coordinates.
Export hooks and source checks make timings diagnostic only.

```powershell
python -X utf8 -B eng/validation/qwen-hf-vision-oracle.py prepare `
  --mmproj C:/Works/models/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf `
  --image artifacts/multimodal-local-preflight/vision-fixtures/card-0.png `
  --output artifacts/master-merge-20261013/qwen-hf-reference-plan-v1
if ($LASTEXITCODE) { throw 'HF reference plan failed' }

# Execute only after the current build/model-validation window ends.
python -X utf8 -B eng/validation/qwen-hf-vision-oracle.py run `
  --plan artifacts/master-merge-20261013/qwen-hf-reference-plan-v1/plan.json `
  --output D:/TensorSharp-validation-cache/master-merge-20261013/qwen-hf-reference-run-v1
if ($LASTEXITCODE) { throw 'HF reference did not qualify; inspect preserved report' }
```

The child is bounded to ten minutes and is killed/reaped on timeout. No BF16
execution, CUDA mode, generation-quality verdict, benchmark pass, or empirically
chosen numerical tolerance exists. Compare TensorSharp's explicit erf-merger and
legacy-merger outputs separately and retain numerical distances as observations.
Model-independent tests check mapping/schema refusal only and import none of the
heavy model libraries.
