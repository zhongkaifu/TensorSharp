# CUDA graph tracing admission

This small CUDA program validates the profiler before loading a model. It
captures two kernels, replays the graph eight times, and checks all 256 output
values. It is a tracing diagnostic, not an inference or throughput benchmark.

Build with an installed CUDA toolkit into an ignored artifact directory:

```sh
nvcc -std=c++17 -O3 -arch=sm_120a trace.cu -o /absolute/artifacts/graph-trace-probe
```

Use Nsight Systems with `--trace=cuda-sw --cuda-graph-trace=node` and inspect the
exported SQLite file for both kernel names, eight graph launches and sixteen
positive graph-node kernel records. Check the program's success JSON as well;
an empty trace is not passing validation. To test TensorSharp's profiler launch
boundary, run this executable through `qwen-nsys-profile.py _exec` with its
ordinary server-spec/identity arguments. Keep generated binaries, reports and
logs in ignored `artifacts/` or `docs/validation/`.
