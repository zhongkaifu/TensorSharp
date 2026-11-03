Standalone Norm -> Q8_1 fusion experiment. Production and ggml are unchanged.

Generate (CPU only; no permission to claim native/GPU validation):
  python3 eng/validation/Qwen35Q81NormFusionBench/build.py --out ABS_FRESH_IGNORED_DIR
Compile privately, offline (does not initialize CUDA):
  python3 eng/validation/Qwen35Q81NormFusionBench/build.py --out ABS_FRESH_BUNDLE --compile --nvcc /usr/local/cuda-13.2/bin/nvcc
Managed build, skipping all native dependency builds:
  dotnet build eng/validation/Qwen35Q81NormFusionBench/Qwen35Q81NormFusionBench.csproj -c Release -p:CudaNvcc=/nonexistent/managed-only -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
CPU-only runner checks: dotnet PATH_TO_DLL --self-test
Default plan: dotnet PATH_TO_DLL --out ABS_FRESH_IGNORED_DIR

Exclusive GPU lease and Linux source-to-assembly/runtime provenance are supplied
by the outer operator. The runner itself starts no model server or background
process. Its default plan performs no model, CUDA or GPU call.
  dotnet PATH_TO_DLL --source-root ACTUAL_SOURCE_ROOT --bundle ABS_FRESH_BUNDLE --model PINNED_27B_GGUF --out ABS_FRESH_IGNORED_DIR --execute
Keep new optimization flags off. No drafter is loaded. The original matching
PTX/fatbin pair must resolve beside the CUDA assembly; the reference loads a
separate identified module and refuses any image except original BBD. The private
cubin contains only the new fusion kernel. Do not replace the baseline fatbin.

192 strict fixtures: six actual loaded projection matrices x N1..8 x four input
patterns, offset3/inactive rows/external guards, independent double-poison A/A.
Norm F32, residuals, Q81 codes/scales and complete downstream GEMV F32 words must
all match. Signed zero, subnormal, NaN/Inf and nonunit norm/alpha are retained.
The actual loaded first-GDN AttnNorm is used at width5120. Width6144/17408 use
explicit synthetic nonunit norm weights; these are component algebra fixtures,
not a claim that the model applies RMSNorm at those projection inputs.

Only after every strict fixture passes: six N8 finite shapes, five ABBA rounds,
32 complete norm+quant iterations per half. Every half checks all outputs and
guards against the strict reference, including downstream GEMV outside timer.
The GPU event interval includes both original launches versus one fused launch;
it includes no omitted quantization/preparation and does not time GEMV itself.
The timed fixture has no residual delta, so every iteration receives identical
input. Any failure retains raw buffers and prevents a quality/performance claim.

Fused row512 still has only N CTAs versus the original quantizer's many group
CTAs. It may be slower through low parallelism. No speedup is assumed. GPU bits,
event results, serving eligibility/fullmodel/DFlash/CUA and integration remain
NOT_RUN until the corresponding actual validations finish. Mixed n>8 is outside
this experiment. No lowering of target weights/KV/state/activation precision.
