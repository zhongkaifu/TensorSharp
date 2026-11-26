using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// F32 <c>Ops.Addmm</c> / <c>Ops.AddmmBatch</c> on cuBLAS. <c>cublasSgemm</c> and
    /// <c>cublasSgemmStridedBatched</c> take no compute type: their operands stay F32 because the
    /// shared handle is in <c>CUBLAS_DEFAULT_MATH</c> (see <see cref="CudaCublasHandle.Create"/>).
    /// ModelBase's F32 routers, LoRA deltas and every other <c>Ops.Addmm</c> on direct CUDA reach
    /// cuBLAS here. A <paramref name="computeType"/> other than <see cref="Unnamed"/> runs the same
    /// product through <c>cublasGemmEx</c> / <c>cublasGemmStridedBatchedEx</c> with that compute type
    /// (<see cref="CudaHalfOperandGemm"/>: the sites that accept half operands, by name).
    /// </summary>
    internal static class CudaBlas
    {
        /// <summary>No compute type named: <c>cublasSgemm</c>, F32 under the default-math handle.</summary>
        internal const int Unnamed = -1;

        public static bool TryAddmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2,
            int computeType = Unnamed)
        {
            if (!TryGetCudaStorage(result, out CudaStorage resultStorage) ||
                !TryGetCudaStorage(src, out CudaStorage srcStorage) ||
                !TryGetCudaStorage(m1, out CudaStorage m1Storage) ||
                !TryGetCudaStorage(m2, out CudaStorage m2Storage))
            {
                return false;
            }

            if (result.ElementType != DType.Float32 || src.ElementType != DType.Float32 ||
                m1.ElementType != DType.Float32 || m2.ElementType != DType.Float32)
            {
                return false;
            }

            if (m1.DimensionCount != 2 || m2.DimensionCount != 2 || result.DimensionCount != 2 || src.DimensionCount != 2)
                return false;

            int rows = checked((int)m1.Sizes[0]);
            int shared = checked((int)m1.Sizes[1]);
            int cols = checked((int)m2.Sizes[1]);
            if (m2.Sizes[0] != shared || result.Sizes[0] != rows || result.Sizes[1] != cols ||
                src.Sizes[0] != rows || src.Sizes[1] != cols)
            {
                return false;
            }

            if (!IsRowMajorMatrix(m1) || !IsRowMajorMatrix(result))
                return false;

            if (!TryGetRightOperand(m2, out int transa, out int lda))
                return false;

            if (beta != 0.0f)
            {
                if (ReferenceEquals(src.Storage, result.Storage) && src.StorageOffset == result.StorageOffset)
                {
                    resultStorage.EnsureDeviceCurrent();
                }
                else if (src.IsContiguous() && result.IsContiguous() && src.ElementCount() == result.ElementCount())
                {
                    resultStorage.CopyDeviceFrom(srcStorage);
                }
                else
                {
                    return false;
                }
            }

            resultStorage.EnsureDeviceCurrent();
            m1Storage.EnsureDeviceCurrent();
            m2Storage.EnsureDeviceCurrent();

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            allocator.Context.MakeCurrent();
            allocator.Blas.SetStream(allocator.Stream.Handle);

            IntPtr aPtr = m2Storage.DevicePtrAtElement(m2.StorageOffset);
            IntPtr bPtr = m1Storage.DevicePtrAtElement(m1.StorageOffset);
            IntPtr cPtr = resultStorage.DevicePtrAtElement(result.StorageOffset);
            int ldb = shared;
            int ldc = cols;

            if (computeType == Unnamed)
            {
                CublasApi.cublasSgemm(
                    allocator.Blas.Handle,
                    transa,
                    CublasApi.CUBLAS_OP_N,
                    cols,
                    rows,
                    shared,
                    ref alpha,
                    aPtr,
                    lda,
                    bPtr,
                    ldb,
                    ref beta,
                    cPtr,
                    ldc).ThrowOnCublasError();
            }
            else
            {
                CublasApi.cublasGemmEx(
                    allocator.Blas.Handle,
                    transa, CublasApi.CUBLAS_OP_N,
                    cols, rows, shared,
                    ref alpha,
                    aPtr, CublasApi.CUDA_R_32F, lda,
                    bPtr, CublasApi.CUDA_R_32F, ldb,
                    ref beta,
                    cPtr, CublasApi.CUDA_R_32F, ldc,
                    computeType, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
            }

            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryAddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2,
            int computeType = Unnamed)
        {
            if (!TryGetCudaStorage(result, out CudaStorage resultStorage) ||
                !TryGetCudaStorage(src, out CudaStorage srcStorage) ||
                !TryGetCudaStorage(m1, out CudaStorage m1Storage) ||
                !TryGetCudaStorage(m2, out CudaStorage m2Storage))
            {
                return false;
            }

            if (result.ElementType != DType.Float32 || src.ElementType != DType.Float32 ||
                m1.ElementType != DType.Float32 || m2.ElementType != DType.Float32)
            {
                return false;
            }

            if (m1.DimensionCount != 3 || m2.DimensionCount != 3 || result.DimensionCount != 3 || src.DimensionCount != 3)
                return false;

            int batch = checked((int)m1.Sizes[0]);
            int rows = checked((int)m1.Sizes[1]);
            int shared = checked((int)m1.Sizes[2]);
            int cols = checked((int)m2.Sizes[2]);

            if (m2.Sizes[0] != batch || m2.Sizes[1] != shared ||
                result.Sizes[0] != batch || result.Sizes[1] != rows || result.Sizes[2] != cols ||
                src.Sizes[0] != batch || src.Sizes[1] != rows || src.Sizes[2] != cols)
            {
                return false;
            }

            if (!IsRowMajorBatchMatrix(m1) || !IsRowMajorBatchMatrix(result))
                return false;

            if (!TryGetRightOperandBatch(m2, out int transa, out int lda))
                return false;

            if (beta != 0.0f)
            {
                if (ReferenceEquals(src.Storage, result.Storage) && src.StorageOffset == result.StorageOffset)
                {
                    resultStorage.EnsureDeviceCurrent();
                }
                else if (src.IsContiguous() && result.IsContiguous() && src.ElementCount() == result.ElementCount())
                {
                    resultStorage.CopyDeviceFrom(srcStorage);
                }
                else
                {
                    return false;
                }
            }

            resultStorage.EnsureDeviceCurrent();
            m1Storage.EnsureDeviceCurrent();
            m2Storage.EnsureDeviceCurrent();

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            allocator.Context.MakeCurrent();
            allocator.Blas.SetStream(allocator.Stream.Handle);

            int ldb = shared;
            int ldc = cols;
            IntPtr aPtr = m2Storage.DevicePtrAtElement(m2.StorageOffset);
            IntPtr bPtr = m1Storage.DevicePtrAtElement(m1.StorageOffset);
            IntPtr cPtr = resultStorage.DevicePtrAtElement(result.StorageOffset);

            if (computeType == Unnamed)
            {
                CublasApi.cublasSgemmStridedBatched(
                    allocator.Blas.Handle,
                    transa,
                    CublasApi.CUBLAS_OP_N,
                    cols,
                    rows,
                    shared,
                    ref alpha,
                    aPtr,
                    lda,
                    checked(m2.Strides[0]),
                    bPtr,
                    ldb,
                    checked(m1.Strides[0]),
                    ref beta,
                    cPtr,
                    ldc,
                    checked(result.Strides[0]),
                    batch).ThrowOnCublasError();
            }
            else
            {
                CublasApi.cublasGemmStridedBatchedEx(
                    allocator.Blas.Handle,
                    transa, CublasApi.CUBLAS_OP_N,
                    cols, rows, shared,
                    ref alpha,
                    aPtr, CublasApi.CUDA_R_32F, lda, checked(m2.Strides[0]),
                    bPtr, CublasApi.CUDA_R_32F, ldb, checked(m1.Strides[0]),
                    ref beta,
                    cPtr, CublasApi.CUDA_R_32F, ldc, checked(result.Strides[0]),
                    batch,
                    computeType, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
            }

            resultStorage.MarkDeviceModified();
            return true;
        }

        /// <summary>
        /// result = alpha m1 m2 over a batch (beta 0) where each operand's matrices have any row stride and batch stride: m1's
        /// and the result's rows unit-stride, m2's rows or columns. The heads of an attention laid out side by side in each row
        /// ([rows, heads, d] viewed as [heads, rows, d]) multiply where they lie, without copying them out or the result back.
        /// False (nothing written) for any other layout.
        /// </summary>
        internal static bool TryAddmmBatchStrided(Tensor result, float alpha, Tensor m1, Tensor m2, int computeType = Unnamed)
        {
            if (!TryGetCudaStorage(result, out CudaStorage resultStorage) || !TryGetCudaStorage(m1, out CudaStorage m1Storage)
                || !TryGetCudaStorage(m2, out CudaStorage m2Storage))
                return false;
            if (result.ElementType != DType.Float32 || m1.ElementType != DType.Float32 || m2.ElementType != DType.Float32
                || result.DimensionCount != 3 || m1.DimensionCount != 3 || m2.DimensionCount != 3)
                return false;
            long batch = m1.Sizes[0], rows = m1.Sizes[1], shared = m1.Sizes[2], cols = m2.Sizes[2];
            if (m2.Sizes[0] != batch || m2.Sizes[1] != shared || result.Sizes[0] != batch || result.Sizes[1] != rows
                || result.Sizes[2] != cols || batch <= 0 || rows <= 0 || shared <= 0 || cols <= 0)
                return false;
            // Column-major as cuBLAS sees row-major data: C^T [cols, rows] = op(A) [cols, shared] B [shared, rows], A = m2, B = m1.
            if (m1.Strides[2] != 1 || m1.Strides[1] < shared || result.Strides[2] != 1 || result.Strides[1] < cols
                || m1.Strides[0] < 0 || m2.Strides[0] < 0 || result.Strides[0] < 0)
                return false;
            int transa;
            long lda;
            if (m2.Strides[2] == 1 && m2.Strides[1] >= cols) { transa = CublasApi.CUBLAS_OP_N; lda = m2.Strides[1]; }
            else if (m2.Strides[1] == 1 && m2.Strides[2] >= shared) { transa = CublasApi.CUBLAS_OP_T; lda = m2.Strides[2]; }
            else return false;

            resultStorage.EnsureDeviceCurrent();
            m1Storage.EnsureDeviceCurrent();
            m2Storage.EnsureDeviceCurrent();
            CudaAllocator allocator = resultStorage.AllocatorImpl;
            allocator.Context.MakeCurrent();
            allocator.Blas.SetStream(allocator.Stream.Handle);
            IntPtr aPtr = m2Storage.DevicePtrAtElement(m2.StorageOffset);
            IntPtr bPtr = m1Storage.DevicePtrAtElement(m1.StorageOffset);
            IntPtr cPtr = resultStorage.DevicePtrAtElement(result.StorageOffset);
            float beta = 0f;
            if (computeType == Unnamed)
                CublasApi.cublasSgemmStridedBatched(allocator.Blas.Handle, transa, CublasApi.CUBLAS_OP_N,
                    checked((int)cols), checked((int)rows), checked((int)shared), ref alpha,
                    aPtr, checked((int)lda), m2.Strides[0], bPtr, checked((int)m1.Strides[1]), m1.Strides[0],
                    ref beta, cPtr, checked((int)result.Strides[1]), result.Strides[0], checked((int)batch)).ThrowOnCublasError();
            else
                CublasApi.cublasGemmStridedBatchedEx(allocator.Blas.Handle, transa, CublasApi.CUBLAS_OP_N,
                    checked((int)cols), checked((int)rows), checked((int)shared), ref alpha,
                    aPtr, CublasApi.CUDA_R_32F, checked((int)lda), m2.Strides[0],
                    bPtr, CublasApi.CUDA_R_32F, checked((int)m1.Strides[1]), m1.Strides[0],
                    ref beta, cPtr, CublasApi.CUDA_R_32F, checked((int)result.Strides[1]), result.Strides[0],
                    checked((int)batch), computeType, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
            resultStorage.MarkDeviceModified();
            return true;
        }

        private static bool TryGetCudaStorage(Tensor tensor, out CudaStorage storage)
        {
            storage = tensor?.Storage as CudaStorage;
            return storage != null;
        }

        private static bool IsRowMajorMatrix(Tensor tensor)
        {
            return tensor.Strides[1] == 1 && tensor.Strides[0] == tensor.Sizes[1];
        }

        private static bool IsRowMajorBatchMatrix(Tensor tensor)
        {
            return tensor.Strides[2] == 1 &&
                tensor.Strides[1] == tensor.Sizes[2] &&
                tensor.Strides[0] == tensor.Sizes[1] * tensor.Sizes[2];
        }

        private static bool TryGetRightOperand(Tensor tensor, out int transa, out int lda)
        {
            int logicalRows = checked((int)tensor.Sizes[0]);
            int logicalCols = checked((int)tensor.Sizes[1]);

            if (tensor.Strides[1] == 1 && tensor.Strides[0] == logicalCols)
            {
                transa = CublasApi.CUBLAS_OP_N;
                lda = logicalCols;
                return true;
            }

            if (tensor.Strides[0] == 1 && tensor.Strides[1] == logicalRows)
            {
                transa = CublasApi.CUBLAS_OP_T;
                lda = logicalRows;
                return true;
            }

            transa = 0;
            lda = 0;
            return false;
        }

        private static bool TryGetRightOperandBatch(Tensor tensor, out int transa, out int lda)
        {
            int logicalRows = checked((int)tensor.Sizes[1]);
            int logicalCols = checked((int)tensor.Sizes[2]);

            if (tensor.Strides[2] == 1 && tensor.Strides[1] == logicalCols)
            {
                transa = CublasApi.CUBLAS_OP_N;
                lda = logicalCols;
                return true;
            }

            if (tensor.Strides[1] == 1 && tensor.Strides[2] == logicalRows)
            {
                transa = CublasApi.CUBLAS_OP_T;
                lda = logicalRows;
                return true;
            }

            transa = 0;
            lda = 0;
            return false;
        }
    }
}
