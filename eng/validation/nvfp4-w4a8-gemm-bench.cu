// NVFP4 W4A8 GEMM benchmark (--nvfp4-prompt-activations q8 on direct CUDA): the two committed kernels of
// tensorsharp_fp4_kernels.cu, ts_fp4_gemm_w4a8 (int8 tensor cores) and ts_fp4_gemm_w4a8_tma (F16 tensor cores, TMA-fed),
// on prompt-row shapes with random planar NVFP4 weights and q8_1 activations. Prints each kernel's time and TOPS
// (2 rows k n / time) and the TMA kernel's largest difference from the int8 kernel's result, relative to its largest
// value. Needs an sm_120a / sm_121a GPU. Results belong in ignored artifacts/.
//
//   nvcc -O3 -std=c++17 -arch=sm_120a -I TensorSharp.Backends.Cuda/native/kernels \
//        eng/validation/nvfp4-w4a8-gemm-bench.cu -lcuda -o nvfp4-w4a8-gemm-bench
//   ./nvfp4-w4a8-gemm-bench [rows k n] ...
#include "tensorsharp_fp4_kernels.cu"
#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// As CudaFp4Kernels.EncodeTma: a 2D map of bytes, rows of `columns` bytes.
static void Encode(TsFp4TmaMap* map, const void* address, unsigned long long columns, unsigned long long rows, unsigned boxColumns,
    unsigned boxRows, CUtensorMapSwizzle swizzle)
{
    cuuint64_t dims[2] = { columns, rows }, strides[1] = { columns };
    cuuint32_t box[2] = { boxColumns, boxRows }, elementStrides[2] = { 1, 1 };
    CUresult r = cuTensorMapEncodeTiled((CUtensorMap*)map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)address, dims, strides, box,
        elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { printf("cuTensorMapEncodeTiled: %d\n", (int)r); exit(1); }
}

template <typename F>
static float Time(F launch)
{
    for (int i = 0; i < 3; ++i) launch();
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < 20; ++i) launch();
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    return ms / 20;
}

int main(int argc, char** argv)
{
    struct Shape { int rows, k, n; };
    std::vector<Shape> shapes;
    for (int i = 1; i + 2 < argc; i += 3) shapes.push_back({ atoi(argv[i]), atoi(argv[i + 1]), atoi(argv[i + 2]) });
    if (shapes.empty())
        shapes = { {512, 5120, 17408}, {512, 17408, 5120}, {512, 5120, 5120}, {512, 2048, 8192}, {512, 8192, 2048}, {512, 2048, 2048},
                   {64, 5120, 17408}, {17, 5120, 17408} };
    int smem = 0;
    {
        int* d; CK(cudaMalloc(&d, sizeof(int)));
        ts_fp4_gemm_w4a8_tma_smem_bytes<<<1, 1>>>(d);
        CK(cudaMemcpy(&smem, d, sizeof(int), cudaMemcpyDeviceToHost));
        cudaFree(d);
    }
    CK(cudaFuncSetAttribute((const void*)ts_fp4_gemm_w4a8_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.f, 1.f);
    for (const Shape& sh : shapes)
    {
        const int rows = sh.rows, k = sh.k, n = sh.n;
        if (k % 256 != 0 || n % 16 != 0) { printf("%d x %d -> %d: k must be a multiple of 256 and n of 16\n", rows, k, n); continue; }
        // Planar weights: codes [n][k/2], then UE4M3 scales [n][k/16] (CudaFp4Kernels.ScalePlane).
        std::vector<uint8_t> planes((size_t)n * k / 2 + (size_t)n * k / 16);
        for (size_t i = 0; i < (size_t)n * k / 2; ++i) planes[i] = (uint8_t)(rng() & 0xFF);
        for (size_t i = (size_t)n * k / 2; i < planes.size(); ++i) planes[i] = (uint8_t)(0x28 + rng() % 24);
        std::vector<float> x((size_t)rows * k);
        for (float& v : x) v = normal(rng);
        uint8_t* dPlanes; float *dx, *dxd, *dxd16, *y8, *y16; int8_t* dxq; __half* dxh;
        CK(cudaMalloc(&dPlanes, planes.size()));
        CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMalloc(&dxq, x.size()));
        CK(cudaMalloc(&dxh, x.size() * 2));
        CK(cudaMalloc(&dxd, (size_t)rows * (k / 32) * 4));
        CK(cudaMalloc(&dxd16, (size_t)rows * (k / 32) * 4));
        CK(cudaMalloc(&y8, (size_t)rows * n * 4));
        CK(cudaMalloc(&y16, (size_t)rows * n * 4));
        CK(cudaMemcpy(dPlanes, planes.data(), planes.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        const unsigned qBlocks = (unsigned)(((long long)rows * k + 255) / 256);
        ts_fp4_quantize_q81_f32<<<qBlocks, 256>>>(dx, dxq, dxd, k, rows);
        ts_fp4_quantize_q81_f16<<<qBlocks, 256>>>(dx, dxh, dxd16, k, rows);
        CK(cudaDeviceSynchronize());
        const uint8_t* scales = dPlanes + (size_t)n * k / 2;
        TsFp4TmaMaps maps;
        Encode(&maps.xCodes, dxh, (unsigned long long)k * 2, rows, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B);
        Encode(&maps.xScales, dxd16, (unsigned long long)(k / 32) * 4, rows, 16, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
        Encode(&maps.wCodes, dPlanes, k / 2, n, 64, 128, CU_TENSOR_MAP_SWIZZLE_64B);
        Encode(&maps.wScales, scales, k / 16, n, 16, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
        const float msInt8 = Time([&] {
            ts_fp4_gemm_w4a8<<<dim3((n + 63) / 64, (rows + 63) / 64), 128>>>(dPlanes, scales, n, k, dxq, dxd, rows, 1.0f, y8);
        });
        const float msTma = Time([&] {
            ts_fp4_gemm_w4a8_tma<<<dim3((rows + 127) / 128, (n + 127) / 128), 384, smem>>>(maps, n, k, rows, 1.0f, y16);
        });
        std::vector<float> a((size_t)rows * n), b((size_t)rows * n);
        CK(cudaMemcpy(a.data(), y8, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), y16, b.size() * 4, cudaMemcpyDeviceToHost));
        double apart = 0, largest = 0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            apart = std::max(apart, (double)fabsf(a[i] - b[i]));
            largest = std::max(largest, (double)fabsf(a[i]));
        }
        const double ops = 2.0 * rows * (double)k * n;
        printf("%5d x %5d -> %5d: int8 %.3f ms %.1f TOPS | tma %.3f ms %.1f TOPS | max |tma - int8| / max %.2e\n", rows, k, n,
            msInt8, ops / (msInt8 * 1e-3) / 1e12, msTma, ops / (msTma * 1e-3) / 1e12, apart / largest);
        cudaFree(dPlanes); cudaFree(dx); cudaFree(dxq); cudaFree(dxh); cudaFree(dxd); cudaFree(dxd16); cudaFree(y8); cudaFree(y16);
    }
    return 0;
}
