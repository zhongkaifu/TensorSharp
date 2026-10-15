# Quantized projection resident benchmark

This isolated tool compares TensorSharp's original-format **Q5_K × F32**
operator with unchanged ggml's ordinary Q5_K matrix multiplication. K, output
rows M, and activation columns N are command-line arguments. The ordinary
backend may quantize activations; it is a **performance reference**, never the
numerical oracle. No model-specific production dispatch is introduced.

The driver allocates both operators' tensors, uploads deterministic original
176-byte Q5_K blocks and F32 inputs, and warms both graphs before timing.
Measured repeats alternate A/B then B/A. Each wall sample includes graph
submission, execution and a final backend synchronization, divided by the
iteration count. The public backend does not expose its original CUDA stream
to this tool, so `gpu_event_ms` is explicitly null. Events on an unrelated
stream are not used. Capture behavior is backend-controlled and is not asserted.

All output values must be finite and all original weight/input bytes must
remain unchanged. Every owned and ordinary F32 output is retained in little
endian column-major `[M,N]` order. Original Q5_K and F32 input bytes are also
saved. The controller records complete binary SHA-256 digests and compares
every owned output bit between qualified runs; differing source bytes or
geometry fail comparison.
Comparison also requires the known receipt and measurement schemas and exactly
all four binary digest entries. A missing owned-output digest cannot leave the
subsequently compared output unpinned.

An independent stored-weight FP64 dot check can cover every output (`full`),
evenly spaced rows/columns including endpoints (`sample`, default), or be
explicitly disabled (`none`). The generated half scales are exact powers of
two, and generated stored weights are exactly representable in F32. The oracle
does not call a production quantizer, dequantizer or dot kernel. Owned error
gates are declared before execution: max absolute error <= 5e-5 and relative
L2 <= 5e-6. The ordinary backend's errors are observations. Sampling is clearly
marked incomplete; a disabled oracle is null, never a passing numerical check.
The controller verifies the exact oracle count from the requested full or
sampled row/column geometry. In the frozen native driver,
`measurement.oracle_complete_outputs` conservatively means that explicit
`full` mode was selected. The separate controller field
`verified_oracle_coverage.oracle_verified_complete_outputs` reports actual
coverage: it is also true when a small sampled geometry covers every row and
column. The raw executable declaration is retained without rewriting it.
This benchmark supplements the complete adversarial SensitiveProjectionProbe
fixtures; it does not replace them or real-model teacher/semantic validation.

The default resident tensor payload cap is 1 GiB. It covers weights, input and
both outputs, not vendor/driver allocations or backend-private workspaces.
`backend_tensor_buffer_bytes` additionally records the actual tensor buffer
size. Host copies, diagnostic files, cache state and operating-system resources
are not physical ownership or hard-limit acceptance. There are no model,
end-to-end throughput, fairness, or shared-budget default-enablement claims.

## Build provenance

`prepare` requires a successfully qualified `rebuild-owned-native-cache.py`
incremental cache, with matching unchanged before/after inputs. It binds the
four **production FinalObjectPaths** to their exact provenance hashes, every
frozen native source, the unchanged upstream revision/files, and prebuilt ggml
libraries. It freezes this small benchmark into a fresh external directory.
No current mutable native source is compiled. Preparation does not build or
execute GPU code. It is valid to benchmark an older frozen generation after
current production sources change.

`build` runs the recorded CMake argument arrays from an appropriate compiler
environment. CMake only compiles the driver and the frozen owned Q8/quantized
and floating factory C++ units; it imports the four existing CUDA objects and
prebuilt upstream libraries. The controller checks the compile-command source
inventory, records the executable hash, and verifies input bookends. It never
fetches, compiles, or patches ggml. The paths and cache receipts are specific to
the platform/compiler that built them; Linux uses its own qualified cache.
Both `build` and `run` check the **actually executing Python controller's**
SHA-256 against the prepared frozen `bench.py` at entry, before work, and after
work, and record that controller's path and pins. A matching nearby snapshot
does not excuse a changed controller. Running a different controller version
requires fresh preparation and build receipts.

Example for the frozen Windows Native36 generation, from an x64 Visual Studio
developer shell with CUDA runtime DLLs on PATH:

```powershell
python -B eng/validation/QuantProjectionBench/bench.py prepare `
  --native-build E:/TensorSharp-validation-cache/master-merge-20261013/native-v36/build-v2 `
  --ggml-root C:/Works/TensorSharp/ExternalProjects/ggml `
  --output E:/TensorSharp-validation-cache/quant-projection/native36-v1
python -B eng/validation/QuantProjectionBench/bench.py build `
  --prepared E:/TensorSharp-validation-cache/quant-projection/native36-v1/prepared.json
```

Keep the preparation/build directories outside this source directory. Use a
fresh directory for every attempt; failed evidence must remain available.

## Execute explicitly selected cases

Execution is separate from preparation/build. Run after other model/GPU work
has stopped and while the system is quiet. The controller removes inherited
TensorSharp/ggml/CUDA/OMP/NVIDIA tuning variables and records the explicit
CUDA visibility and TF32 override. `--device` selects a device after CUDA
visibility remapping. A missing device or a build without CUDA returns 77;
the controller reports failure/unavailability, never a passing/skipped case.
Its timeout reaps the direct child and preserves the failure report. There is
no cross-platform descendant containment in this controller. In particular, a
timed-out CMake build does not prove compiler descendants were stopped; the
failure receipt explicitly records that limitation. Verify remaining compiler
processes have exited before reusing test resources.
The receipt validator requires exactly the owned and ordinary timing variants,
the requested device, warmup/repeat/iteration parameters, exact oracle counts,
and a positive finite sample for every repeat.

Two useful GLM dimensions are examples only: K=4096/M=8192 and K=8192/M=4096.
Each original-format matrix occupies 22 MiB. They introduce no production
model assumptions. Widths 1, 3 and 8 isolate narrow decode/verify behavior;
512, 774 and 809 exercise wide prefill. The default full oracle bound is
300 million products, sufficient for either example at N=8. Wide cases need
explicit sampling or a larger declared operation bound.

```powershell
python -B eng/validation/QuantProjectionBench/bench.py run `
  --prepared E:/TensorSharp-validation-cache/quant-projection/native36-v1/prepared.json `
  --executable E:/TensorSharp-validation-cache/quant-projection/native36-v1/build/QuantProjectionBench.exe `
  --output C:/Works/TensorSharp/artifacts/quant-projection/native36-k4096-m8192-n8-v1 `
  --k 4096 --rows 8192 --columns 8 --oracle full `
  --warmup 10 --repeats 20 --iterations 10 --timeout 600
```

Choose repeat counts so each measured sample is long enough to amortize host
timer resolution; retain every sample and report p50/p95. These are resident
warm measurements. They do not measure cold reads, full-model expert routing,
startup or request TTFT, and they do not prove hardware clock/power equivalence.
Run whole-tool ABBA sequences for cross-generation comparisons. Changing
generated tokens or expert routes cannot affect this operator-only workload.

```powershell
python -B eng/validation/QuantProjectionBench/bench.py compare `
  --left C:/Works/TensorSharp/artifacts/quant-projection/native36-k4096-m8192-n8-v1 `
  --right C:/Works/TensorSharp/artifacts/quant-projection/native37-k4096-m8192-n8-v1 `
  --output C:/Works/TensorSharp/artifacts/quant-projection/36-vs-37-n8.json
python -B -m unittest discover -s eng/validation/QuantProjectionBench -p test_bench.py
```

Generated evidence belongs in ignored `artifacts/`, `docs/validation/`, or an
external validation cache, never in this reusable source directory. Source-only
controller tests establish no native build/device/model coverage. The CMake
CPU-only option permits separate driver sanity work but does not qualify CUDA;
the receipt controller intentionally prepares the frozen CUDA-object build.
