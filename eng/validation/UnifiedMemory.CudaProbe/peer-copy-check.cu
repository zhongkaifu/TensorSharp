// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Independent CUDA Runtime control for a failed managed peer-copy probe.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        std::exit(2);
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: peer-copy-check DEVICE_A DEVICE_B\n");
        return 2;
    }
    const int devices[] = {std::atoi(argv[1]), std::atoi(argv[2])};
    if (devices[0] < 0 || devices[1] < 0 || devices[0] == devices[1]) return 2;
    constexpr size_t count = 262144;
    constexpr size_t bytes = count * sizeof(unsigned);
    std::vector<unsigned> expected(count), actual(count);
    for (size_t i = 0; i < count; ++i) expected[i] = static_cast<unsigned>(i * 17 + 7);
    int failed = 0;
    for (int direction = 0; direction != 2; ++direction) {
        const int from = devices[direction], to = devices[1 - direction];
        unsigned *source = nullptr, *destination = nullptr;
        check(cudaSetDevice(from), "source device");
        check(cudaMalloc(&source, bytes), "source allocation");
        check(cudaMemcpy(source, expected.data(), bytes, cudaMemcpyHostToDevice), "source upload");
        check(cudaDeviceSynchronize(), "source synchronize");
        check(cudaSetDevice(to), "destination device");
        check(cudaMalloc(&destination, bytes), "destination allocation");
        int accessible = 0;
        check(cudaDeviceCanAccessPeer(&accessible, to, from), "peer capability");
        if (!accessible) {
            std::fprintf(stderr, "UNAVAILABLE directed GPU %d -> %d\n", from, to);
            return 2;
        }
        cudaError_t enable = cudaDeviceEnablePeerAccess(from, 0);
        if (enable != cudaErrorPeerAccessAlreadyEnabled) check(enable, "enable peer");
        for (int asynchronous = 0; asynchronous != 2; ++asynchronous) {
            check(cudaMemset(destination, 0, bytes), "clear destination");
            check(cudaDeviceSynchronize(), "clear synchronize");
            if (asynchronous)
                check(cudaMemcpyPeerAsync(destination, to, source, from, bytes), "peer async copy");
            else
                check(cudaMemcpyPeer(destination, to, source, from, bytes), "peer sync copy");
            check(cudaDeviceSynchronize(), "peer synchronize");
            check(cudaMemcpy(actual.data(), destination, bytes, cudaMemcpyDeviceToHost), "destination download");
            size_t mismatches = 0, first = count;
            for (size_t i = 0; i < count; ++i) {
                if (actual[i] != expected[i]) {
                    ++mismatches;
                    if (first == count) first = i;
                }
            }
            std::printf("%s GPU %d -> %d CUDA Runtime %s: %zu/%zu words mismatched",
                mismatches ? "FAIL" : "PASS", from, to, asynchronous ? "async" : "sync", mismatches, count);
            if (mismatches) {
                ++failed;
                std::printf("; first word %zu expected=%u actual=%u", first, expected[first], actual[first]);
            }
            std::printf("\n");
        }
        check(cudaFree(destination), "destination free");
        check(cudaSetDevice(from), "source cleanup device");
        check(cudaFree(source), "source free");
    }
    return failed ? 1 : 0;
}
