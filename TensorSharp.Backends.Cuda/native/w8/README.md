# Experimental original-source W8G32 vocabulary module

This separately built module can be selected by the experimental, default-OFF
native-CUDA Qwen3.8 output-head companion. The companion keeps the GGUF body,
embedding, draft and tokenizer. The explicit `TS_Q35_NINFER_HEAD_SCOPE=draft`
recipe preserves the target Q6 head and uses the original NInfer W8/BF16 head only
for draft proposals. The current safety guard requires this explicit scope when
paths are configured; missing scope refuses, and all three settings unset is OFF.
Earlier binaries also allowed shared target/draft scope. The typed shared backend
component is now research-only; the guarded Factory route does not expose it. See the
[opt-in instructions and validation scope](../../../eng/validation/qwen38-ninfer-head.README.md).
Normal builds do not compile or select this optional module automatically.
The S12 shared-head short run failed its per-check no-loss gate; default remains
OFF. The separate S13 draft-only short comparison retained all original checks
and arithmetic-field outcomes, with baseline failures still present. Its 81 full
target-output pairs matched bitwise in a no-drafter diagnostic. The subsequent
actual 384-request S13 CUA failed its no-loss gate: three original
checks and four typed fields regressed, with improvements retained separately.
Earlier finite passes or observed A/A variability do not waive this failure.
**Default remains OFF; this candidate is not recommended.** The option guard is
not a quality repair. Subsequent S13 DFL long tests passed128K but250K returned
three device-OOM HTTP500 errors under M1/cap1; failed requests have no answer or
performance credit. These are configuration-capacity observations, not an engine-wide
context limit. The candidate remains rejected for quality and capacity; component
results do not establish whole-model quality or stable speed.


The final S14 parser/test build passed 66 focused CPU tests and two native smoke
probes (27 oracle calls each, explicit draft / OFF). It enforces configured paths plus
exact `draft` scope, with all 3 settings unset for OFF. This guard changes no native
math and does not repair/upgrade S13 CUA or 250K capacity results. Smoke is a separate
split-K-zero / greedy1 diagnostic, not a performance or comprehensive quality result.

`upstream/` is the byte-for-byte e600a7d9 Ignis six-header closure plus all six
license/NOTICE/provenance files. `official-source-manifest.json` has the original
SHA256 list and source URLs. Retain the entire subtree and applicable notices
when distributing. The owned .cu only explicitly instantiates original kernels
T=1..33 and defines the RNE F32→BF16 cast. No kernel clone, schedule override,
launch-bounds change, new quantizer or ggml modification is involved.

The fixed geometry is 248320 output rows by K5120: signed-int8 row-major codes,
F16 scales per32 K, codes1271398400B + scales79462400B =1350860800B. This is a
new W8/BF16 activation recipe, not lossless TS Q6_K repacking or an adoption of
Ignis's complete runtime. Activation support is finite BF16 normal/signedzero;
F16 scales must be finite. The caller/numerical validator checks payloads; the
launch component does not scan/synchronize device inputs for NaN/subnormals.

Build manually with eng/validation/build-qwen35-w8-head-module.py; normal backend
builds do not glob this directory, ship this module, or choose it automatically.
Its full SDK static_asserts check 64-bit Output/empty-class and PDL launch ABI.
The managed loader separately validates all 34 driver-enumerated symbols,
actual parameter offsets/sizes (and absence of extras), exact original schedule,
current owning context and >=1 launchable CTA. It records actual image SHA and
resources; original minBlocks is not an assertion of measured occupancy.
Unrecognized compiler mangling/ABI or unsupported T refuses rather than falling
back. Actual NVCC compilation and the R10 C# driver gate covered T=1..33, using
eight original ELF results and 25 source-supported token-prefix references.
The new engine bridge, real head payload and model quality/performance require
their own integration gates; the component result does not certify them.

CUDA Driver API 12.8 documents the enumeration/name/parameter APIs; this backend
requires CUDA13+. Original PDL is preserved using cuLaunchKernelEx attribute6.
The attribute union is64B, CUlaunchAttribute72B/valueoffset8, config56B with
stream32/attrs40/numAttrs48. Kernel params are x/codes/scales pointers,16-byte
Output(ptr,leadingDim,padding0), and two zero empty-class bytes; their device
layout is verified with cuFuncGetParamInfo before launch. No silent non-PDL mode.

The caller owns correctly sized/disjoint device allocations, a current context,
stream ordering and completion, including keeping inputs/workspace/module alive
and synchronizing before disposal. Capacity arguments are caller declarations,
not driver allocation-range proofs. LaunchF32 includes cast and matmul on the
same stream; its BF16 workspace is explicit. It adds no argmax or hidden host
copy. The companion adds an ordinary same-stream BF16-to-F32 consumer using an
existing TensorSharp kernel; ggml and original native kernel sources are unchanged.

Primary ABI references:
- https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__MODULE.html
- https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__EXEC.html
- https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/cuda_8h_source.html
