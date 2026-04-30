using System;
using TensorSharp.Cuda.Interop;
using TensorSharp.Core;

namespace TensorSharp.Cuda
{
    internal enum CudaUnaryOp
    {
        Relu = 0,
        Sigmoid = 1,
        SiLU = 2,
        GELU = 3,
        Tanh = 4,
    }

    internal enum CudaBinaryActivationOp
    {
        SiLUMul = 0,
        GELUMul = 1,
        SigmoidMul = 2,
    }

    internal static class CudaKernelOps
    {
        public static bool TryFill(Tensor result, float value)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage storage, out IntPtr ptr, out int count))
                return false;

            CudaAllocator allocator = storage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            allocator.Context.MakeCurrent();
            kernels.LaunchFillF32(ptr, count, value, allocator.Stream.Handle);
            storage.MarkDeviceModified();
            return true;
        }

        public static bool TryCopy(Tensor result, Tensor src)
        {
            if (!TryGetContiguous(result, out CudaStorage resultStorage, out IntPtr resultPtr, out long resultCount) ||
                !TryGetContiguous(src, out CudaStorage srcStorage, out IntPtr srcPtr, out long srcCount))
            {
                return false;
            }

            if (result.ElementType != src.ElementType || resultCount != srcCount)
                return false;

            srcStorage.EnsureDeviceCurrent();
            srcStorage.SynchronizeDeviceWork();
            resultStorage.AllocatorImpl.Context.MakeCurrent();
            TensorSharp.Cuda.Interop.CudaDriverApi.cuMemcpyDtoD(
                resultPtr,
                srcPtr,
                new UIntPtr((ulong)(resultCount * result.ElementType.Size()))).ThrowOnError();
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryUnary(Tensor result, Tensor src, CudaUnaryOp op)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int count) ||
                !TryGetContiguousFloat(src, out CudaStorage srcStorage, out IntPtr srcPtr, out int srcCount) ||
                count != srcCount)
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            srcStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchUnaryF32(srcPtr, resultPtr, count, (int)op, allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryBinaryActivation(Tensor result, Tensor lhs, Tensor rhs, CudaBinaryActivationOp op)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int count) ||
                !TryGetContiguousFloat(lhs, out CudaStorage lhsStorage, out IntPtr lhsPtr, out int lhsCount) ||
                !TryGetContiguousFloat(rhs, out CudaStorage rhsStorage, out IntPtr rhsPtr, out int rhsCount) ||
                count != lhsCount || count != rhsCount)
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            lhsStorage.EnsureDeviceCurrent();
            rhsStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchBinaryActivationF32(lhsPtr, rhsPtr, resultPtr, count, (int)op, allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TrySiLUMulSplit(Tensor result, Tensor gateUp, int halfDim)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int count) ||
                !TryGetContiguousFloat(gateUp, out CudaStorage gateUpStorage, out IntPtr gateUpPtr, out _) ||
                result.DimensionCount != 2 ||
                gateUp.DimensionCount != 2 ||
                result.Sizes[0] != gateUp.Sizes[0] ||
                result.Sizes[1] != halfDim ||
                gateUp.Sizes[1] < halfDim * 2 ||
                count != result.Sizes[0] * halfDim)
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            gateUpStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchSiLUMulSplitF32(gateUpPtr, resultPtr, checked((int)result.Sizes[0]), halfDim, allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryRMSNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps)
        {
            if (!TryGetContiguousRows(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int rows, out int cols) ||
                !TryGetContiguousRows(src, out CudaStorage srcStorage, out IntPtr srcPtr, out int srcRows, out int srcCols) ||
                !TryGetContiguousFloat(alpha, out CudaStorage alphaStorage, out IntPtr alphaPtr, out int alphaCount) ||
                rows != srcRows ||
                cols != srcCols ||
                alphaCount != cols)
            {
                return false;
            }

            IntPtr betaPtr = IntPtr.Zero;
            CudaStorage betaStorage = null;
            if (beta != null)
            {
                if (!TryGetContiguousFloat(beta, out betaStorage, out betaPtr, out int betaCount) || betaCount != cols)
                    return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            srcStorage.EnsureDeviceCurrent();
            alphaStorage.EnsureDeviceCurrent();
            betaStorage?.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchRMSNormF32(srcPtr, alphaPtr, betaPtr, resultPtr, rows, cols, eps, allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TrySoftmax(Tensor result, Tensor src)
        {
            if (!TryGetContiguousRows(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int rows, out int cols) ||
                !TryGetContiguousRows(src, out CudaStorage srcStorage, out IntPtr srcPtr, out int srcRows, out int srcCols) ||
                rows != srcRows ||
                cols != srcCols)
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            srcStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchSoftmaxF32(srcPtr, resultPtr, rows, cols, allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryIndexSelect(Tensor result, Tensor src, Tensor indices, bool isAdd)
        {
            if (!TryGetContiguousRows(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int rows, out int cols) ||
                !TryGetContiguousRows(src, out CudaStorage srcStorage, out IntPtr srcPtr, out int sourceRows, out int sourceCols) ||
                !TryGetContiguous(indices, out CudaStorage indicesStorage, out IntPtr indicesPtr, out long indexCount) ||
                sourceCols != cols ||
                indexCount != rows ||
                (indices.ElementType != DType.Int32 && indices.ElementType != DType.Float32))
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            srcStorage.EnsureDeviceCurrent();
            indicesStorage.EnsureDeviceCurrent();
            if (isAdd)
                resultStorage.EnsureDeviceCurrent();

            allocator.Context.MakeCurrent();
            kernels.LaunchIndexSelectF32(
                srcPtr,
                indicesPtr,
                resultPtr,
                rows,
                cols,
                sourceRows,
                indices.ElementType == DType.Int32 ? 1 : 0,
                isAdd ? 1 : 0,
                allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryAddCausalMask(Tensor tensor, int seqLen, int startPos, float maskedValue)
        {
            if (!TryGetContiguousRows(tensor, out CudaStorage storage, out IntPtr ptr, out int rows, out int cols))
                return false;

            CudaAllocator allocator = storage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            storage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchAddCausalMaskF32(ptr, rows, cols, seqLen, startPos, maskedValue, allocator.Stream.Handle);
            storage.MarkDeviceModified();
            return true;
        }

        public static bool TryRoPE(Tensor result, Tensor src, int seqLen, int rowOffset)
        {
            if (!TryGetContiguousRows(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int rows, out int cols) ||
                !TryGetContiguousRows(src, out CudaStorage srcStorage, out IntPtr srcPtr, out int srcRows, out int srcCols) ||
                rows != srcRows ||
                cols != srcCols ||
                cols < 2)
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            srcStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchRoPEF32(srcPtr, resultPtr, rows, cols, seqLen, rowOffset, allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryRoPEEx(
            Tensor result,
            Tensor src,
            Tensor positions,
            int ropeDim,
            int mode,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow,
            bool addToResult)
        {
            if (positions == null)
                return false;

            if (!TryGetContiguousRows(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int rows, out int cols) ||
                !TryGetContiguousRows(src, out CudaStorage srcStorage, out IntPtr srcPtr, out int srcRows, out int srcCols) ||
                !TryGetContiguous(positions, out CudaStorage positionsStorage, out IntPtr positionsPtr, out long positionCount) ||
                rows != srcRows ||
                cols != srcCols ||
                positionCount != rows ||
                (positions.ElementType != DType.Int32 && positions.ElementType != DType.Float32))
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            srcStorage.EnsureDeviceCurrent();
            positionsStorage.EnsureDeviceCurrent();
            if (addToResult)
                resultStorage.EnsureDeviceCurrent();

            allocator.Context.MakeCurrent();
            kernels.LaunchRoPEExF32(
                srcPtr,
                positionsPtr,
                resultPtr,
                rows,
                cols,
                ropeDim,
                mode,
                positions.ElementType == DType.Int32 ? 1 : 0,
                originalContextLength,
                freqBase,
                freqScale,
                extFactor,
                attnFactor,
                betaFast,
                betaSlow,
                addToResult ? 1 : 0,
                allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        internal static bool TryGetContiguousFloat(Tensor tensor, out CudaStorage storage, out IntPtr ptr, out int count)
        {
            if (TryGetContiguous(tensor, out storage, out ptr, out long longCount) &&
                tensor.ElementType == DType.Float32 &&
                longCount <= int.MaxValue)
            {
                count = (int)longCount;
                return true;
            }

            storage = null;
            ptr = IntPtr.Zero;
            count = 0;
            return false;
        }

        internal static bool TryGetContiguousRows(Tensor tensor, out CudaStorage storage, out IntPtr ptr, out int rows, out int cols)
        {
            if (TryGetContiguousFloat(tensor, out storage, out ptr, out int count) &&
                tensor.DimensionCount > 0 &&
                tensor.Sizes[tensor.DimensionCount - 1] <= int.MaxValue)
            {
                cols = (int)tensor.Sizes[tensor.DimensionCount - 1];
                if (cols > 0 && count % cols == 0)
                {
                    rows = count / cols;
                    return true;
                }
            }

            storage = null;
            ptr = IntPtr.Zero;
            rows = 0;
            cols = 0;
            return false;
        }

        internal static bool TryGetContiguous(Tensor tensor, out CudaStorage storage, out IntPtr ptr, out long count)
        {
            storage = tensor?.Storage as CudaStorage;
            if (storage == null || !tensor.IsContiguous())
            {
                ptr = IntPtr.Zero;
                count = 0;
                return false;
            }

            count = tensor.ElementCount();
            ptr = storage.DevicePtrAtElement(tensor.StorageOffset);
            return true;
        }

        private static bool TryGetKernels(CudaAllocator allocator, out CudaKernels kernels)
        {
            kernels = allocator.Kernels;
            return kernels != null;
        }
    }
}
