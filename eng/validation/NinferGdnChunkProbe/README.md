# Original GDN chunk component probe

```
dotnet NinferGdnChunkProbe.dll CHUNK_MODULE OUTPUT_JSON [SEQUENTIAL_HELPER_MODULE]
```

Only an actual GPU execution validates this candidate. The source module is
built separately by `eng/validation/build-ninfer-gdn-chunk-module.py`; no model
or environment default selects it.

The probe uses deterministic bounded dense inputs at T=64/128/192 and structured
one-hot inputs at T=1024/4096. The latter exercises the original multi-job output
schedule without making the independent CPU recurrence unnecessarily costly.
All inputs crossing the original chunk kernels are represented BF16 values;
gates and persistent initial state are F32. A separate FP64 mathematical
recurrence uses the same represented inputs, grouped h/3 heads, decay and update
rule, and readout scale.

The acceptance criteria are fixed from
[the original test](https://github.com/gpillon/ignis/blob/e600a7d9e0d5921d767a6cb464373583b3cdcaf4/kernel/vendor/tests/ops/test_gated_delta_net.cpp):

- Output: relative L2 <= 0.0041 and maximum absolute error <=
  `0.000005 + 0.0055 * max(abs(reference))`.
- State: relative L2 <= 0.0027 and maximum absolute error <=
  `0.00001 + 0.0039 * max(abs(reference))`.

The combined criterion follows the original `ops/op_check.h`. These are
synthetic component gates, not a newly invented quality budget or permission
to degrade model answers. They do not establish whole-model equivalence.

State/output/workspace use guard regions. The second AA run resets the initial
state and output and poisons the scratch differently; output and state must
match the first run bit for bit. Packed inputs, decay and beta must remain
unchanged. The optional sequential adapter receives the same BF16 represented
normalized inputs and is checked independently against the FP64 reference.

CUDA event intervals include packing, all three original chunk stages and
unpacking for the candidate; the sequential interval includes its scan only.
They are diagnostic single-case intervals, without statistical or end-to-end
performance claims. Original revision, actual module image SHA, kernel symbols,
workspace bytes and failed criteria are recorded. Generated JSON and logs must
stay under ignored `docs/validation/` or `artifacts/`.
