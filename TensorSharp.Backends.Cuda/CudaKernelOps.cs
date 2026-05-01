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

    internal enum CudaBinaryOp
    {
        Add = 0,
        Sub = 1,
        Mul = 2,
        Div = 3,
    }

    internal enum CudaScalarOp
    {
        Add = 0,
        Sub = 1,
        Mul = 2,
        Div = 3,
        ReverseSub = 4,
        ReverseDiv = 5,
    }

    internal enum CudaTernaryOp
    {
        AddMul = 0,
        AddDiv = 1,
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
            if (!TryGetContiguous(result, out CudaStorage resultStorage, out _, out long resultCount) ||
                !TryGetContiguous(src, out CudaStorage srcStorage, out _, out long srcCount))
            {
                return false;
            }

            if (result.ElementType != src.ElementType || resultCount != srcCount)
                return false;

            srcStorage.EnsureDeviceCurrent();
            long byteCount = checked(resultCount * result.ElementType.Size());
            long resultByteOffset = checked(result.StorageOffset * result.ElementType.Size());
            long srcByteOffset = checked(src.StorageOffset * src.ElementType.Size());
            resultStorage.CopyDeviceFrom(srcStorage, resultByteOffset, srcByteOffset, byteCount);
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

        public static bool TryBinary(Tensor result, Tensor lhs, Tensor rhs, CudaBinaryOp op)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int count) ||
                !TryGetContiguousFloat(lhs, out CudaStorage lhsStorage, out IntPtr lhsPtr, out int lhsCount) ||
                !TryGetContiguousFloat(rhs, out CudaStorage rhsStorage, out IntPtr rhsPtr, out int rhsCount) ||
                count != lhsCount ||
                count != rhsCount ||
                !SameShape(result, lhs) ||
                !SameShape(result, rhs))
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            lhsStorage.EnsureDeviceCurrent();
            rhsStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchBinaryF32(lhsPtr, rhsPtr, resultPtr, count, (int)op, allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryScalar(Tensor result, Tensor lhs, float rhs, CudaScalarOp op)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int count) ||
                !TryGetContiguousFloat(lhs, out CudaStorage lhsStorage, out IntPtr lhsPtr, out int lhsCount) ||
                count != lhsCount ||
                !SameShape(result, lhs))
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            lhsStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchScalarF32(lhsPtr, resultPtr, count, rhs, (int)op, allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryTernary(Tensor result, Tensor x, Tensor y, Tensor z, CudaTernaryOp op)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int count) ||
                !TryGetContiguousFloat(x, out CudaStorage xStorage, out IntPtr xPtr, out int xCount) ||
                !TryGetContiguousFloat(y, out CudaStorage yStorage, out IntPtr yPtr, out int yCount) ||
                !TryGetContiguousFloat(z, out CudaStorage zStorage, out IntPtr zPtr, out int zCount) ||
                count != xCount ||
                count != yCount ||
                count != zCount ||
                !SameShape(result, x) ||
                !SameShape(result, y) ||
                !SameShape(result, z))
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            xStorage.EnsureDeviceCurrent();
            yStorage.EnsureDeviceCurrent();
            zStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchTernaryF32(xPtr, yPtr, zPtr, resultPtr, count, (int)op, allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryAddMulScalar(Tensor result, Tensor x, Tensor y, float z)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int count) ||
                !TryGetContiguousFloat(x, out CudaStorage xStorage, out IntPtr xPtr, out int xCount) ||
                !TryGetContiguousFloat(y, out CudaStorage yStorage, out IntPtr yPtr, out int yCount) ||
                count != xCount ||
                count != yCount ||
                !SameShape(result, x) ||
                !SameShape(result, y))
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            xStorage.EnsureDeviceCurrent();
            yStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchAddMulScalarF32(xPtr, yPtr, resultPtr, count, z, allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryMulMulAdd(Tensor result, Tensor x, Tensor y, Tensor z, Tensor w)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int count) ||
                !TryGetContiguousFloat(x, out CudaStorage xStorage, out IntPtr xPtr, out int xCount) ||
                !TryGetContiguousFloat(y, out CudaStorage yStorage, out IntPtr yPtr, out int yCount) ||
                !TryGetContiguousFloat(z, out CudaStorage zStorage, out IntPtr zPtr, out int zCount) ||
                !TryGetContiguousFloat(w, out CudaStorage wStorage, out IntPtr wPtr, out int wCount) ||
                count != xCount ||
                count != yCount ||
                count != zCount ||
                count != wCount ||
                !SameShape(result, x) ||
                !SameShape(result, y) ||
                !SameShape(result, z) ||
                !SameShape(result, w))
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            xStorage.EnsureDeviceCurrent();
            yStorage.EnsureDeviceCurrent();
            zStorage.EnsureDeviceCurrent();
            wStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchMulMulAddF32(xPtr, yPtr, zPtr, wPtr, resultPtr, count, allocator.Stream.Handle);
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

        public static bool TryGELUMulSplit(Tensor result, Tensor gateUp, int halfDim)
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
            kernels.LaunchGELUMulSplitF32(gateUpPtr, resultPtr, checked((int)result.Sizes[0]), halfDim, allocator.Stream.Handle);
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

        public static bool TryScaledDotProductAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor mask, float scale)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out _) ||
                !TryGetContiguousFloat(query, out CudaStorage queryStorage, out IntPtr queryPtr, out _) ||
                !TryGetContiguousFloat(key, out CudaStorage keyStorage, out IntPtr keyPtr, out _) ||
                !TryGetContiguousFloat(value, out CudaStorage valueStorage, out IntPtr valuePtr, out _) ||
                query.DimensionCount != 4 ||
                key.DimensionCount != 4 ||
                value.DimensionCount != 4 ||
                result.DimensionCount != 4 ||
                query.Sizes[0] != key.Sizes[0] ||
                query.Sizes[0] != value.Sizes[0] ||
                query.Sizes[2] != key.Sizes[2] ||
                query.Sizes[2] != value.Sizes[2] ||
                query.Sizes[3] != key.Sizes[3] ||
                result.Sizes[0] != query.Sizes[0] ||
                result.Sizes[1] != query.Sizes[1] ||
                result.Sizes[2] != query.Sizes[2] ||
                result.Sizes[3] != value.Sizes[3])
            {
                return false;
            }

            int batch = checked((int)query.Sizes[0]);
            int seqQ = checked((int)query.Sizes[1]);
            int heads = checked((int)query.Sizes[2]);
            int keyDim = checked((int)query.Sizes[3]);
            int seqK = checked((int)key.Sizes[1]);
            int valueDim = checked((int)value.Sizes[3]);
            if (seqK <= 0 || seqK > 8192)
                return false;

            IntPtr maskPtr = IntPtr.Zero;
            CudaStorage maskStorage = null;
            int hasMask = 0;
            if (mask != null)
            {
                if (!TryGetContiguousFloat(mask, out maskStorage, out maskPtr, out _) ||
                    mask.DimensionCount != 4 ||
                    mask.Sizes[0] != batch ||
                    mask.Sizes[1] != heads ||
                    mask.Sizes[2] != seqQ ||
                    mask.Sizes[3] != seqK)
                {
                    return false;
                }

                hasMask = 1;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            queryStorage.EnsureDeviceCurrent();
            keyStorage.EnsureDeviceCurrent();
            valueStorage.EnsureDeviceCurrent();
            maskStorage?.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchScaledDotProductAttentionF32(
                queryPtr,
                keyPtr,
                valuePtr,
                maskPtr,
                resultPtr,
                batch,
                seqQ,
                seqK,
                heads,
                keyDim,
                valueDim,
                scale,
                hasMask,
                allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryGqaPrefillAttention(
            Tensor result,
            Tensor query,
            Tensor key,
            Tensor value,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int seqLen,
            int kvLen,
            int maskStart,
            int windowSize,
            float scale)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out _) ||
                !TryGetContiguousFloat(query, out CudaStorage queryStorage, out IntPtr queryPtr, out _) ||
                !TryGetContiguousFloat(key, out CudaStorage keyStorage, out IntPtr keyPtr, out _) ||
                !TryGetContiguousFloat(value, out CudaStorage valueStorage, out IntPtr valuePtr, out _) ||
                query.DimensionCount != 3 ||
                key.DimensionCount != 3 ||
                value.DimensionCount != 3 ||
                result.DimensionCount != 2 ||
                numQHeads <= 0 ||
                numKVHeads <= 0 ||
                headDim <= 0 ||
                seqLen <= 0 ||
                kvLen <= 0 ||
                kvLen > 8192 ||
                numQHeads % numKVHeads != 0 ||
                query.Sizes[0] != numQHeads ||
                query.Sizes[1] != seqLen ||
                query.Sizes[2] != headDim ||
                key.Sizes[0] != numKVHeads ||
                key.Sizes[1] != kvLen ||
                key.Sizes[2] != headDim ||
                value.Sizes[0] != numKVHeads ||
                value.Sizes[1] != kvLen ||
                value.Sizes[2] != headDim ||
                result.Sizes[0] != seqLen ||
                result.Sizes[1] != numQHeads * headDim ||
                maskStart < 0 ||
                maskStart >= kvLen)
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            queryStorage.EnsureDeviceCurrent();
            keyStorage.EnsureDeviceCurrent();
            valueStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchGqaPrefillAttentionF32(
                queryPtr,
                keyPtr,
                valuePtr,
                resultPtr,
                numQHeads,
                numKVHeads,
                seqLen,
                kvLen,
                headDim,
                maskStart,
                windowSize,
                scale,
                allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryGqaDecodeAttention(
            Tensor result,
            Tensor query,
            Tensor keyCache,
            Tensor valueCache,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            bool circular,
            float scale)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !TryGetContiguousFloat(query, out CudaStorage queryStorage, out IntPtr queryPtr, out int queryCount) ||
                !TryGetContiguousFloat(keyCache, out CudaStorage keyStorage, out IntPtr keyPtr, out _) ||
                !TryGetContiguousFloat(valueCache, out CudaStorage valueStorage, out IntPtr valuePtr, out _) ||
                query.DimensionCount != 2 ||
                result.DimensionCount != 2 ||
                keyCache.DimensionCount != 3 ||
                valueCache.DimensionCount != 3 ||
                numQHeads <= 0 ||
                numKVHeads <= 0 ||
                headDim <= 0 ||
                attendStart < 0 ||
                attendLen <= 0 ||
                attendLen > 8192 ||
                cacheSize <= 0 ||
                numQHeads % numKVHeads != 0 ||
                query.Sizes[0] != 1 ||
                query.Sizes[1] != numQHeads * headDim ||
                result.Sizes[0] != 1 ||
                result.Sizes[1] != numQHeads * headDim ||
                keyCache.Sizes[0] != numKVHeads ||
                keyCache.Sizes[1] != cacheSize ||
                keyCache.Sizes[2] != headDim ||
                valueCache.Sizes[0] != numKVHeads ||
                valueCache.Sizes[1] != cacheSize ||
                valueCache.Sizes[2] != headDim ||
                queryCount != numQHeads * headDim ||
                resultCount != queryCount ||
                (!circular && attendStart + attendLen > cacheSize))
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            queryStorage.EnsureDeviceCurrent();
            keyStorage.EnsureDeviceCurrent();
            valueStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchGqaDecodeAttentionF32(
                queryPtr,
                keyPtr,
                valuePtr,
                resultPtr,
                numQHeads,
                numKVHeads,
                headDim,
                attendStart,
                attendLen,
                cacheSize,
                circular ? 1 : 0,
                scale,
                allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TrySliceColumns(Tensor result, Tensor src, int colOffset, int width)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !TryGetContiguousFloat(src, out CudaStorage srcStorage, out IntPtr srcPtr, out _) ||
                result.DimensionCount != 2 ||
                src.DimensionCount != 2 ||
                width <= 0 ||
                colOffset < 0 ||
                result.Sizes[0] != src.Sizes[0] ||
                result.Sizes[1] != width ||
                colOffset + width > src.Sizes[1] ||
                resultCount != result.Sizes[0] * width)
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            srcStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchSliceColumnsF32(
                srcPtr,
                resultPtr,
                checked((int)result.Sizes[0]),
                checked((int)src.Sizes[1]),
                colOffset,
                width,
                allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryFlatToHeadFirst(Tensor result, Tensor src, int numHeads, int seqLen, int headDim)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !TryGetContiguousFloat(src, out CudaStorage srcStorage, out IntPtr srcPtr, out int srcCount) ||
                result.DimensionCount != 3 ||
                src.DimensionCount != 2 ||
                numHeads <= 0 ||
                seqLen <= 0 ||
                headDim <= 0 ||
                src.Sizes[0] != seqLen ||
                src.Sizes[1] != numHeads * headDim ||
                result.Sizes[0] != numHeads ||
                result.Sizes[1] != seqLen ||
                result.Sizes[2] != headDim ||
                srcCount != seqLen * numHeads * headDim ||
                resultCount != srcCount)
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            srcStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchFlatToHeadFirstF32(srcPtr, resultPtr, seqLen, numHeads, headDim, allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TrySplitQkvToHeadFirst(Tensor result, Tensor qkv, int colOffset, int numHeads, int seqLen, int headDim)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !TryGetContiguousFloat(qkv, out CudaStorage qkvStorage, out IntPtr qkvPtr, out _) ||
                result.DimensionCount != 3 ||
                qkv.DimensionCount != 2 ||
                numHeads <= 0 ||
                seqLen <= 0 ||
                headDim <= 0 ||
                colOffset < 0 ||
                qkv.Sizes[0] != seqLen ||
                colOffset + numHeads * headDim > qkv.Sizes[1] ||
                result.Sizes[0] != numHeads ||
                result.Sizes[1] != seqLen ||
                result.Sizes[2] != headDim ||
                resultCount != seqLen * numHeads * headDim)
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            qkvStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchSplitQkvHeadFirstF32(
                qkvPtr,
                resultPtr,
                seqLen,
                checked((int)qkv.Sizes[1]),
                colOffset,
                numHeads,
                headDim,
                allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryCopyHeadFirstToCache(Tensor cache, Tensor src, int startPos, int seqLen, int cacheSize, bool circular)
        {
            if (!TryGetContiguousFloat(cache, out CudaStorage cacheStorage, out IntPtr cachePtr, out _) ||
                !TryGetContiguousFloat(src, out CudaStorage srcStorage, out IntPtr srcPtr, out int srcCount) ||
                cache.DimensionCount != 3 ||
                src.DimensionCount != 3 ||
                seqLen <= 0 ||
                cacheSize <= 0 ||
                startPos < 0 ||
                src.Sizes[0] != cache.Sizes[0] ||
                src.Sizes[1] != seqLen ||
                src.Sizes[2] != cache.Sizes[2] ||
                cache.Sizes[1] != cacheSize ||
                srcCount != src.Sizes[0] * seqLen * src.Sizes[2] ||
                (!circular && startPos + seqLen > cacheSize))
            {
                return false;
            }

            CudaAllocator allocator = cacheStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            srcStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchCopyHeadFirstToCacheF32(
                srcPtr,
                cachePtr,
                checked((int)src.Sizes[0]),
                seqLen,
                checked((int)src.Sizes[2]),
                startPos,
                cacheSize,
                circular ? 1 : 0,
                allocator.Stream.Handle);
            cacheStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryGatherCircularHeadFirst(Tensor result, Tensor cache, int startPos, int seqLen, int cacheSize)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !TryGetContiguousFloat(cache, out CudaStorage cacheStorage, out IntPtr cachePtr, out _) ||
                result.DimensionCount != 3 ||
                cache.DimensionCount != 3 ||
                seqLen <= 0 ||
                cacheSize <= 0 ||
                result.Sizes[0] != cache.Sizes[0] ||
                result.Sizes[1] != seqLen ||
                result.Sizes[2] != cache.Sizes[2] ||
                cache.Sizes[1] != cacheSize ||
                resultCount != result.Sizes[0] * seqLen * result.Sizes[2])
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            cacheStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchGatherCircularHeadFirstF32(
                cachePtr,
                resultPtr,
                checked((int)result.Sizes[0]),
                seqLen,
                checked((int)result.Sizes[2]),
                startPos,
                cacheSize,
                allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryConcatHeadFirst(Tensor result, Tensor a, Tensor b)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !TryGetContiguousFloat(a, out CudaStorage aStorage, out IntPtr aPtr, out _) ||
                !TryGetContiguousFloat(b, out CudaStorage bStorage, out IntPtr bPtr, out _) ||
                result.DimensionCount != 3 ||
                a.DimensionCount != 3 ||
                b.DimensionCount != 3 ||
                a.Sizes[0] != b.Sizes[0] ||
                a.Sizes[2] != b.Sizes[2] ||
                result.Sizes[0] != a.Sizes[0] ||
                result.Sizes[1] != a.Sizes[1] + b.Sizes[1] ||
                result.Sizes[2] != a.Sizes[2] ||
                resultCount != result.Sizes[0] * result.Sizes[1] * result.Sizes[2])
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            aStorage.EnsureDeviceCurrent();
            bStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchConcatHeadFirstF32(
                aPtr,
                bPtr,
                resultPtr,
                checked((int)result.Sizes[0]),
                checked((int)a.Sizes[1]),
                checked((int)b.Sizes[1]),
                checked((int)result.Sizes[2]),
                allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryNeoXRoPEHeadFirst(Tensor data, Tensor cosTable, Tensor sinTable, int numHeads, int seqLen, int headDim, int ropeHalf)
        {
            if (!TryGetContiguousFloat(data, out CudaStorage dataStorage, out IntPtr dataPtr, out int dataCount) ||
                !TryGetContiguousFloat(cosTable, out CudaStorage cosStorage, out IntPtr cosPtr, out int cosCount) ||
                !TryGetContiguousFloat(sinTable, out CudaStorage sinStorage, out IntPtr sinPtr, out int sinCount) ||
                data.DimensionCount != 3 ||
                cosTable.DimensionCount != 1 ||
                sinTable.DimensionCount != 1 ||
                numHeads <= 0 ||
                seqLen <= 0 ||
                headDim <= 0 ||
                ropeHalf <= 0 ||
                ropeHalf * 2 > headDim ||
                data.Sizes[0] != numHeads ||
                data.Sizes[1] != seqLen ||
                data.Sizes[2] != headDim ||
                dataCount != numHeads * seqLen * headDim ||
                cosCount != seqLen * ropeHalf ||
                sinCount != cosCount)
            {
                return false;
            }

            CudaAllocator allocator = dataStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            dataStorage.EnsureDeviceCurrent();
            cosStorage.EnsureDeviceCurrent();
            sinStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchNeoXRoPEHeadFirstF32(
                dataPtr,
                cosPtr,
                sinPtr,
                numHeads,
                seqLen,
                headDim,
                ropeHalf,
                allocator.Stream.Handle);
            dataStorage.MarkDeviceModified();
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

        private static bool SameShape(Tensor a, Tensor b)
        {
            if (a.DimensionCount != b.DimensionCount)
                return false;

            for (int i = 0; i < a.DimensionCount; i++)
            {
                if (a.Sizes[i] != b.Sizes[i])
                    return false;
            }

            return true;
        }
    }
}
