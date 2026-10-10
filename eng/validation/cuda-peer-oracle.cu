// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Independent CUDA Runtime API control for UnifiedMemory.CudaProbe failures.
// Build: nvcc -O2 cuda-peer-oracle.cu -o cuda-peer-oracle
// Runs only its own allocations on devices 0 and 1; no model or ggml dependency.
// --without-peer-access leaves peer access disabled, observing the Runtime's
// fallback for the same copy API; a pass then does not certify physical P2P.
// Exit 0: both direct and staged paths matched; 1: failure; 2: unavailable.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct Allocation {
    int device;
    void* data = nullptr;
    Allocation(int id, size_t bytes) : device(id) {
        check(cudaSetDevice(device));
        check(cudaMalloc(&data, bytes));
    }
    ~Allocation() {
        if (data) { cudaSetDevice(device); cudaFree(data); }
    }
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
};

int main(int argc, char** argv) {
    try {
        if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--without-peer-access") != 0))
            throw std::runtime_error("Usage: cuda-peer-oracle [--without-peer-access]");
        const bool request_peer_access = argc == 1;
        int count = 0;
        check(cudaGetDeviceCount(&count));
        if (count < 2) { std::puts("{\"status\":\"unavailable\",\"reason\":\"two GPUs required\"}"); return 2; }
        bool passed = true;
        for (int src = 0; src < 2; ++src) {
            const int dst = 1 - src;
            int accessible = 0;
            check(cudaDeviceCanAccessPeer(&accessible, dst, src));
            check(cudaSetDevice(dst));
            auto enabled = !request_peer_access ? cudaSuccess
                : accessible ? cudaDeviceEnablePeerAccess(src, 0) : cudaErrorPeerAccessUnsupported;
            if (enabled == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
            if (request_peer_access && (!accessible || (enabled != cudaSuccess && enabled != cudaErrorPeerAccessAlreadyEnabled))) {
                std::printf("{\"from\":%d,\"to\":%d,\"status\":\"unavailable\",\"cuda_error\":%d}\n",src,dst,int(enabled));
                return 2;
            }
            for (size_t bytes : {size_t(4096), size_t(1048576 + 17), size_t(16777216)}) {
                Allocation source(src, bytes), destination(dst, bytes);
                std::vector<unsigned char> expected(bytes), actual(bytes), staging(bytes);
                for (size_t i = 0; i < bytes; ++i) expected[i] = static_cast<unsigned char>((i * 31 + i / 251 + src * 17 + 1) % 251);
                check(cudaSetDevice(src));
                check(cudaMemcpy(source.data, expected.data(), bytes, cudaMemcpyHostToDevice));
                check(cudaDeviceSynchronize());
                for (const bool peer : {true, false}) {
                    check(cudaSetDevice(dst));
                    check(cudaMemset(destination.data, 0xa5, bytes));
                    check(cudaDeviceSynchronize());
                    if (peer) {
                        check(cudaMemcpyPeerAsync(destination.data, dst, source.data, src, bytes));
                        check(cudaDeviceSynchronize());
                    } else {
                        check(cudaSetDevice(src));
                        check(cudaMemcpy(staging.data(), source.data, bytes, cudaMemcpyDeviceToHost));
                        check(cudaSetDevice(dst));
                        check(cudaMemcpy(destination.data, staging.data(), bytes, cudaMemcpyHostToDevice));
                        check(cudaDeviceSynchronize());
                    }
                    check(cudaMemcpy(actual.data(), destination.data, bytes, cudaMemcpyDeviceToHost));
                    size_t mismatch = 0;
                    while (mismatch < bytes && expected[mismatch] == actual[mismatch]) ++mismatch;
                    const bool equal = mismatch == bytes;
                    passed &= equal;
                    std::printf("{\"from\":%d,\"to\":%d,\"bytes\":%zu,\"route\":\"%s\",\"peer_access_requested\":%s,\"equal\":%s,\"first_mismatch\":%lld}\n",
                        src,dst,bytes,peer?"peer-api":"host-staged",request_peer_access?"true":"false",equal?"true":"false",equal?-1LL:static_cast<long long>(mismatch));
                    std::fflush(stdout);
                }
            }
        }
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CUDA control failed: %s\n", error.what());
        return 1;
    }
}
