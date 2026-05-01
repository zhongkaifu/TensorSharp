using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal static class CudaBlas
    {
        public static bool TryAddmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
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

            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryAddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
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
