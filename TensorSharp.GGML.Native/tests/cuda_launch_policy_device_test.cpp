// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// An explicit GGML_CUDA_PDL survives the real CUDA backend's initialization:
// configure_cuda_launch_policy (ggml_ops_cuda_launch_policy.cpp) only sets a
// default when the variable is unset. Usage: <test> pdl0|pdl1. Exit 77 when no
// CUDA backend is available.
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
int TSGgml_IsBackendAvailable(int);
void TSGgml_Shutdown();
}

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "pdl0") && std::strcmp(argv[1], "pdl1"))) return 2;
    const char* value = argv[1] + 3;
#ifdef _WIN32
    if (_putenv_s("GGML_CUDA_PDL", value)) return 1;
#else
    if (setenv("GGML_CUDA_PDL", value, 1)) return 1;
#endif
    if (!TSGgml_IsBackendAvailable(3)) return 77;
    const char* actual = std::getenv("GGML_CUDA_PDL");
    if (!actual || std::strcmp(actual, value)) return 1;
    TSGgml_Shutdown();
    std::printf("PASS actual CUDA backend preserves explicit PDL=%s\n", actual);
    return 0;
}
