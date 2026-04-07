#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"

extern "C" {
    struct TensorView2DDesc {
        void* data;
        int dim0;
        int dim1;
        int stride0;
        int stride1;
        std::int64_t raw_bytes;
    };

    struct ContiguousTensorDesc {
        void* data;
        std::int64_t element_count;
        int element_type;
    };

    const char* TSGgml_GetLastError();
    int TSGgml_IsBackendAvailable(int backendType);
    int TSGgml_GetRowsQuantF32(
        TensorView2DDesc result,
        void* src_data,
        int src_ggml_type,
        std::int64_t src_ne0,
        std::int64_t src_ne1,
        std::int64_t src_raw_bytes,
        ContiguousTensorDesc indices);
    void* TSGgml_AlignedAlloc(size_t size);
    void TSGgml_AlignedFree(void* ptr);
}

namespace
{
    constexpr int BackendTypeCpu = 2;
    constexpr int BackendTypeCuda = 3;
    constexpr int TensorSharpDTypeI32 = 3;

    struct AlignedBuffer
    {
        explicit AlignedBuffer(size_t size)
            : ptr(TSGgml_AlignedAlloc(size))
        {
        }

        ~AlignedBuffer()
        {
            if (ptr != nullptr)
                TSGgml_AlignedFree(ptr);
        }

        void* ptr = nullptr;
    };

    [[noreturn]] void fail(const std::string& message)
    {
        std::cerr << message << std::endl;
        std::exit(1);
    }

    float expected_value(int row, int col)
    {
        return static_cast<float>(row * 1000 + col);
    }

    void* make_unaligned_ptr(std::byte* storage)
    {
        std::byte* ptr = storage;
        while ((reinterpret_cast<std::uintptr_t>(ptr) % 32) == 0 || (reinterpret_cast<std::uintptr_t>(ptr) % alignof(float)) != 0)
        {
            ++ptr;
        }

        return ptr;
    }

    void fill_source(float* data, int rows, int cols)
    {
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col)
            {
                data[static_cast<size_t>(row) * cols + col] = expected_value(row, col);
            }
        }
    }

    void assert_rows(const float* result, const std::vector<std::int32_t>& indices, int cols, const std::string& label)
    {
        for (size_t row = 0; row < indices.size(); ++row)
        {
            for (int col = 0; col < cols; ++col)
            {
                float actual = result[row * static_cast<size_t>(cols) + col];
                float expected = expected_value(indices[row], col);
                if (std::fabs(actual - expected) > 1e-5f)
                {
                    fail(label + ": unexpected value at row " + std::to_string(row) +
                        ", col " + std::to_string(col) + ". Got " + std::to_string(actual) +
                        ", expected " + std::to_string(expected) + ".");
                }
            }
        }
    }

    void run_case(const char* label, void* result_ptr, void* source_ptr, int rows, int cols, const std::vector<std::int32_t>& indices)
    {
        std::memset(result_ptr, 0, indices.size() * static_cast<size_t>(cols) * sizeof(float));

        TensorView2DDesc result_desc {
            result_ptr,
            static_cast<int>(indices.size()),
            cols,
            cols,
            1,
            static_cast<std::int64_t>(indices.size() * static_cast<size_t>(cols) * sizeof(float))
        };

        ContiguousTensorDesc index_desc {
            const_cast<std::int32_t*>(indices.data()),
            static_cast<std::int64_t>(indices.size()),
            TensorSharpDTypeI32
        };

        int ok = TSGgml_GetRowsQuantF32(
            result_desc,
            source_ptr,
            GGML_TYPE_F32,
            cols,
            rows,
            static_cast<std::int64_t>(rows * static_cast<size_t>(cols) * sizeof(float)),
            index_desc);

        if (ok == 0)
        {
            const char* err = TSGgml_GetLastError();
            fail(std::string(label) + ": native call failed: " + (err != nullptr ? err : "<null>"));
        }

        assert_rows(static_cast<const float*>(result_ptr), indices, cols, label);
    }
}

int main(int argc, char** argv)
{
    int backendType = BackendTypeCpu;
    std::string backendName = "GGML CPU";

    if (argc >= 2)
    {
        std::string arg = argv[1];
        if (arg == "cpu" || arg == "ggml_cpu")
        {
            backendType = BackendTypeCpu;
            backendName = "GGML CPU";
        }
        else if (arg == "cuda" || arg == "ggml_cuda")
        {
            backendType = BackendTypeCuda;
            backendName = "GGML CUDA";
        }
        else
        {
            fail("Unknown backend argument. Use cpu or cuda.");
        }
    }

    if (TSGgml_IsBackendAvailable(backendType) == 0)
    {
        const char* err = TSGgml_GetLastError();
        fail("Failed to initialize " + backendName + " backend: " + (err != nullptr ? err : "<null>"));
    }

    constexpr int rows = 32;
    constexpr int cols = 64;
    const size_t source_bytes = rows * static_cast<size_t>(cols) * sizeof(float);
    const size_t result_bytes = 2 * static_cast<size_t>(cols) * sizeof(float);
    const std::vector<std::int32_t> indices = { 5, 1 };

    AlignedBuffer aligned_source(source_bytes);
    AlignedBuffer aligned_result(result_bytes);
    if (aligned_source.ptr == nullptr || aligned_result.ptr == nullptr)
        fail("Failed to allocate aligned buffers for regression test.");

    fill_source(static_cast<float*>(aligned_source.ptr), rows, cols);

    std::vector<std::byte> unaligned_source_storage(source_bytes + 64);
    std::vector<std::byte> unaligned_result_storage(result_bytes + 64);

    void* unaligned_source = make_unaligned_ptr(unaligned_source_storage.data());
    void* unaligned_result = make_unaligned_ptr(unaligned_result_storage.data());

    std::memcpy(unaligned_source, aligned_source.ptr, source_bytes);

    run_case("aligned buffers", aligned_result.ptr, aligned_source.ptr, rows, cols, indices);
    run_case("unaligned result buffer", unaligned_result, aligned_source.ptr, rows, cols, indices);
    run_case("unaligned source buffer", aligned_result.ptr, unaligned_source, rows, cols, indices);
    run_case("unaligned source and result buffers", unaligned_result, unaligned_source, rows, cols, indices);

    std::cout << "GgmlOpsGetRowsQuantTest passed (" << backendName << ")" << std::endl;
    return 0;
}
