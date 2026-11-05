// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void ts_trace_probe_fill(float* values) {
    const int i = threadIdx.x;
    values[i] = static_cast<float>(i);
}

__global__ void ts_trace_probe_scale(float* values) {
    const int i = threadIdx.x;
    values[i] = 2.0f * values[i] + 1.0f;
}

int main() {
    float* device = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    CHECK(cudaMalloc(&device, 256 * sizeof(float)));
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    ts_trace_probe_fill<<<1, 256, 0, stream>>>(device);
    CHECK(cudaGetLastError());
    ts_trace_probe_scale<<<1, 256, 0, stream>>>(device);
    CHECK(cudaGetLastError());
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiateWithFlags(&executable, graph, 0));
    for (int i = 0; i < 8; ++i) CHECK(cudaGraphLaunch(executable, stream));
    CHECK(cudaStreamSynchronize(stream));
    float actual[256];
    CHECK(cudaMemcpy(actual, device, sizeof(actual), cudaMemcpyDeviceToHost));
    double checksum = 0;
    for (int i = 0; i < 256; ++i) {
        if (actual[i] != static_cast<float>(2 * i + 1)) return 2;
        checksum += actual[i];
    }
    CHECK(cudaGraphExecDestroy(executable));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(device));
    std::printf("{\"passed\":true,\"graphLaunches\":8,\"expectedKernelNodes\":16,\"checksum\":%.0f}\n", checksum);
    return 0;
}
