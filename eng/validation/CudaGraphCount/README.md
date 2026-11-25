# CUDA graph capture counter

`cudagraph_count.c` is an `LD_PRELOAD` interposer for a Linux process whose GGML
library links the CUDA runtime dynamically (`libcudart.so.13`). It counts
`cudaStreamBeginCapture`, `cudaGraphInstantiate`, `cudaGraphExecUpdate` and
`cudaGraphLaunch` calls and prints them at exit as `[cudagraph] captures …`. It
shows whether, and how often, upstream ggml-cuda replays a workload's graphs: it
does not measure time. The process must exit normally to print, and the shim
resolves the runtime by `dlopen(…, RTLD_NOLOAD)`, because .NET loads the native
library and its dependencies into a local scope.

```sh
gcc -O2 -shared -fPIC -o artifacts/cudagraph_count.so eng/validation/CudaGraphCount/cudagraph_count.c -ldl
LD_PRELOAD=$PWD/artifacts/cudagraph_count.so dotnet eng/validation/QwenImage21DenoiserAB/bin/Release/net10.0/QwenImage21DenoiserAB.dll \
  --dit qwen-image-2.1-turbo-Q4_K_M.gguf --backend ggml_cuda --size 1024 --refs 1 --steps 8 --engines runtime
```

Compare two step counts. Captures that stay fixed while launches grow by the
same number every step mean that every step from the third on is replayed.
Keep the built library and logs in ignored `artifacts/` or `docs/validation/`.
