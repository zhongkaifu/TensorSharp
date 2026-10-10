# CUDA activity diagnostics

`cuda-activity-trace.cpp` is a Linux diagnostic recorder built against the installed
CUPTI SDK (validated with CUDA 12.8 / CUPTI 2025.1.1). It records concurrent kernel,
runtime API and device-copy timestamps without changing the model graph or splitting
it into individual operations. It does not modify or patch ggml.

```sh
g++ -O2 -std=c++17 -fPIC -shared eng/validation/cuda-activity-trace.cpp \
  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
  -Wl,-rpath,/usr/local/cuda/lib64 -lcupti -pthread \
  -o artifacts/libTsCudaActivity.so

dotnet eng/validation/Qwen4ExpExpertCacheProbe/bin/Release/net10.0/Qwen4ExpExpertCacheProbe.dll \
  --model /path/to/model.gguf --placement device --backend ggml_cuda \
  --prefill-tokens 40 --decode-tokens 16 --max-context 4096 --warmup 0 --iterations 2 \
  --cuda-trace-library /absolute/path/to/artifacts/libTsCudaActivity.so \
  --output artifacts/cupti-run.json

python eng/validation/cuda-activity-report.py artifacts/cupti-run.cuda.jsonl \
  --iteration 1 --skip-steps 2 --output artifacts/cupti-summary.json
```

Add the probe's usual layer split / model identity options when needed. Use fresh
output paths; the recorder refuses to overwrite its trace. Model loading precedes
recording. Timestamp markers bracket prefill and every decode forward; the report
selects complete marked steps and keeps API and GPU time separate. It rejects
missing markers, incomplete activity, dropped buffers, and ranges without kernels.
Use one recorder session per probe process. The probe retains the loaded library
until process exit because CUPTI owns its callback addresses.

These are **instrumented diagnostics**, not throughput results. CUPTI can add
substantial CUDA graph launch overhead. Per-node profiling additionally breaks
fusion/capture and inserts synchronization, so its numerical output and speed must
not be substituted for an ordinary run. Establish invariance with identical input
histories and full logits, and use separate uninstrumented application runs for
performance claims. API time overlaps GPU work; adding their totals is invalid.

Record CUDA/CUPTI, native/managed binary and checkpoint identities with the probe
report. Keep trace JSONL, build logs and summaries in ignored `artifacts/` or
`docs/validation/`. No hardware counters, kernel replay, driver changes, or global
profiler settings are required by this tool. Unavailable CUPTI is a failed capture,
not a passing or silently skipped profile.
