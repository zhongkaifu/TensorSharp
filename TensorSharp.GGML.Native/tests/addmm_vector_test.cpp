#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

extern "C" {
    struct TensorView2DDesc {
        void* data;
        int dim0;
        int dim1;
        int stride0;
        int stride1;
        std::int64_t raw_bytes;
    };

    struct TensorView3DDesc {
        void* data;
        int dim0;
        int dim1;
        int dim2;
        int stride0;
        int stride1;
        int stride2;
        std::int64_t raw_bytes;
    };

    const char* TSGgml_GetLastError();
    int TSGgml_IsBackendAvailable(int backendType);
    int TSGgml_AddmmF32(
        TensorView2DDesc result,
        TensorView2DDesc src,
        TensorView2DDesc m1,
        TensorView2DDesc m2,
        float beta,
        float alpha);
    int TSGgml_AddmmBatchF32(
        TensorView3DDesc result,
        TensorView3DDesc src,
        TensorView3DDesc m1,
        TensorView3DDesc m2,
        float beta,
        float alpha);
    void* TSGgml_AlignedAlloc(size_t size);
    void TSGgml_AlignedFree(void* ptr);
}

namespace
{
    constexpr int BackendTypeCpu = 2;
    constexpr int BackendTypeCuda = 3;

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

    void assert_close(float actual, float expected, const std::string& label)
    {
        if (std::fabs(actual - expected) > 1e-5f)
        {
            fail(label + ": got " + std::to_string(actual) + ", expected " + std::to_string(expected));
        }
    }

    void fill_2d(float* data, int rows, int cols, float base)
    {
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col)
            {
                data[static_cast<size_t>(row) * cols + col] = base + row * 10.0f + col;
            }
        }
    }

    void fill_3d(float* data, int batches, int rows, int cols, float base)
    {
        for (int batch = 0; batch < batches; ++batch)
        {
            for (int row = 0; row < rows; ++row)
            {
                for (int col = 0; col < cols; ++col)
                {
                    const size_t index =
                        (static_cast<size_t>(batch) * rows + row) * cols + col;
                    data[index] = base + batch * 100.0f + row * 10.0f + col;
                }
            }
        }
    }

    void run_addmm_vector_case()
    {
        constexpr int rows = 3;
        constexpr int shared = 4;
        constexpr int cols = 1;

        const size_t result_bytes = rows * static_cast<size_t>(cols) * sizeof(float);
        const size_t m1_bytes = rows * static_cast<size_t>(shared) * sizeof(float);
        const size_t m2_bytes = shared * static_cast<size_t>(cols) * sizeof(float);

        AlignedBuffer result_buffer(result_bytes);
        AlignedBuffer m1_buffer(m1_bytes);
        AlignedBuffer m2_buffer(m2_bytes);
        if (result_buffer.ptr == nullptr || m1_buffer.ptr == nullptr || m2_buffer.ptr == nullptr)
            fail("Failed to allocate buffers for addmm vector regression test.");

        auto* result = static_cast<float*>(result_buffer.ptr);
        auto* m1 = static_cast<float*>(m1_buffer.ptr);
        auto* m2 = static_cast<float*>(m2_buffer.ptr);

        std::memset(result, 0, result_bytes);
        fill_2d(m1, rows, shared, 1.0f);
        fill_2d(m2, shared, cols, 0.5f);

        TensorView2DDesc result_desc {
            result,
            rows,
            cols,
            cols,
            1,
            static_cast<std::int64_t>(result_bytes)
        };

        TensorView2DDesc m1_desc {
            m1,
            rows,
            shared,
            shared,
            1,
            static_cast<std::int64_t>(m1_bytes)
        };

        TensorView2DDesc m2_desc {
            m2,
            shared,
            cols,
            cols,
            1,
            static_cast<std::int64_t>(m2_bytes)
        };

        TensorView2DDesc src_desc {};
        const int ok = TSGgml_AddmmF32(result_desc, src_desc, m1_desc, m2_desc, 0.0f, 1.0f);
        if (ok == 0)
        {
            const char* err = TSGgml_GetLastError();
            fail(std::string("2D addmm vector case failed: ") + (err != nullptr ? err : "<null>"));
        }

        for (int row = 0; row < rows; ++row)
        {
            float expected = 0.0f;
            for (int k = 0; k < shared; ++k)
            {
                expected += m1[row * shared + k] * m2[k * cols];
            }

            assert_close(result[row * cols], expected, "2D addmm vector case row " + std::to_string(row));
        }
    }

    void run_addmm_batch_vector_case()
    {
        constexpr int batches = 2;
        constexpr int rows = 3;
        constexpr int shared = 4;
        constexpr int cols = 1;

        const size_t result_bytes = batches * static_cast<size_t>(rows) * cols * sizeof(float);
        const size_t m1_bytes = batches * static_cast<size_t>(rows) * shared * sizeof(float);
        const size_t m2_bytes = batches * static_cast<size_t>(shared) * cols * sizeof(float);

        AlignedBuffer result_buffer(result_bytes);
        AlignedBuffer m1_buffer(m1_bytes);
        AlignedBuffer m2_buffer(m2_bytes);
        if (result_buffer.ptr == nullptr || m1_buffer.ptr == nullptr || m2_buffer.ptr == nullptr)
            fail("Failed to allocate buffers for addmmbatch vector regression test.");

        auto* result = static_cast<float*>(result_buffer.ptr);
        auto* m1 = static_cast<float*>(m1_buffer.ptr);
        auto* m2 = static_cast<float*>(m2_buffer.ptr);

        std::memset(result, 0, result_bytes);
        fill_3d(m1, batches, rows, shared, 1.0f);
        fill_3d(m2, batches, shared, cols, 0.5f);

        TensorView3DDesc result_desc {
            result,
            batches,
            rows,
            cols,
            rows * cols,
            cols,
            1,
            static_cast<std::int64_t>(result_bytes)
        };

        TensorView3DDesc m1_desc {
            m1,
            batches,
            rows,
            shared,
            rows * shared,
            shared,
            1,
            static_cast<std::int64_t>(m1_bytes)
        };

        TensorView3DDesc m2_desc {
            m2,
            batches,
            shared,
            cols,
            shared * cols,
            cols,
            1,
            static_cast<std::int64_t>(m2_bytes)
        };

        TensorView3DDesc src_desc {};
        const int ok = TSGgml_AddmmBatchF32(result_desc, src_desc, m1_desc, m2_desc, 0.0f, 1.0f);
        if (ok == 0)
        {
            const char* err = TSGgml_GetLastError();
            fail(std::string("3D addmmbatch vector case failed: ") + (err != nullptr ? err : "<null>"));
        }

        for (int batch = 0; batch < batches; ++batch)
        {
            for (int row = 0; row < rows; ++row)
            {
                float expected = 0.0f;
                for (int k = 0; k < shared; ++k)
                {
                    const size_t m1_index =
                        (static_cast<size_t>(batch) * rows + row) * shared + k;
                    const size_t m2_index =
                        (static_cast<size_t>(batch) * shared + k) * cols;
                    expected += m1[m1_index] * m2[m2_index];
                }

                const size_t result_index =
                    (static_cast<size_t>(batch) * rows + row) * cols;
                assert_close(
                    result[result_index],
                    expected,
                    "3D addmmbatch vector case batch " + std::to_string(batch) +
                        " row " + std::to_string(row));
            }
        }
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

    run_addmm_vector_case();
    run_addmm_batch_vector_case();

    std::cout << "GgmlOpsAddmmVectorTest passed (" << backendName << ")" << std::endl;
    return 0;
}
