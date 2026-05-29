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
        private const int DecodeAttentionPartitionSize = 512;
        private const int DecodeAttentionPartitionThreshold = 2048;
        private const int DecodeAttentionSingleBlockMaxTokens = 8192;

        public static bool TryFill(Tensor result, float value)
        {
            if (!TryGetContiguous(result, out CudaStorage storage, out IntPtr ptr, out long longCount) ||
                longCount > int.MaxValue ||
                (result.ElementType != DType.Float32 && result.ElementType != DType.Float16))
            {
                return false;
            }

            CudaAllocator allocator = storage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            int count = (int)longCount;
            allocator.Context.MakeCurrent();
            if (result.ElementType == DType.Float32)
                kernels.LaunchFillF32(ptr, count, value, allocator.Stream.Handle);
            else
                kernels.LaunchFillF16(ptr, count, value, allocator.Stream.Handle);
            storage.MarkDeviceModified();
            return true;
        }

        public static bool TryCopy(Tensor result, Tensor src)
        {
            if (result == null || src == null ||
                result.Storage is not CudaStorage resultStorage ||
                src.Storage is not CudaStorage srcStorage ||
                result.ElementType != src.ElementType ||
                result.ElementCount() != src.ElementCount() ||
                !SameShape(result, src))
            {
                return false;
            }

            long elementCount = result.ElementCount();
            if (elementCount == 0)
                return true;

            if (result.IsContiguous() && src.IsContiguous())
            {
                srcStorage.EnsureDeviceCurrent();
                long byteCount = checked(result.ElementType.ByteLengthFor(elementCount));
                long resultByteOffset = ElementOffsetToBytes(result.StorageOffset, result.ElementType);
                long srcByteOffset = ElementOffsetToBytes(src.StorageOffset, src.ElementType);
                resultStorage.CopyDeviceFrom(srcStorage, resultByteOffset, srcByteOffset, byteCount);
                return true;
            }

            return TryCopyStridedBytes(result, src, resultStorage, srcStorage);
        }

        private static bool TryCopyStridedBytes(Tensor result, Tensor src, CudaStorage resultStorage, CudaStorage srcStorage)
        {
            if (ReferenceEquals(resultStorage, srcStorage) && !SameStorageView(result, src))
                return false;

            int dimCount = result.DimensionCount;
            ReadOnlySpan<long> resultSizes = result.Sizes;
            ReadOnlySpan<long> resultStrides = result.Strides;
            ReadOnlySpan<long> srcStrides = src.Strides;

            long innerElems = 1;
            int outerDims = dimCount;
            for (int d = dimCount - 1; d >= 0; d--)
            {
                long size = resultSizes[d];
                if (size == 1)
                {
                    outerDims = d;
                    continue;
                }

                if (resultStrides[d] != innerElems || srcStrides[d] != innerElems)
                    break;

                innerElems = checked(innerElems * size);
                outerDims = d;
            }

            if (outerDims == dimCount)
                return false;

            DType dtype = result.ElementType;
            if (dtype == DType.Q8_0 && (innerElems % 32) != 0)
                return false;

            long innerBytes = checked(dtype.ByteLengthFor(innerElems));
            long[] counter = outerDims > 0 ? new long[outerDims] : Array.Empty<long>();

            while (true)
            {
                long resultElemOffset = result.StorageOffset;
                long srcElemOffset = src.StorageOffset;
                for (int d = 0; d < outerDims; d++)
                {
                    resultElemOffset = checked(resultElemOffset + counter[d] * resultStrides[d]);
                    srcElemOffset = checked(srcElemOffset + counter[d] * srcStrides[d]);
                }

                long resultByteOffset = ElementOffsetToBytes(resultElemOffset, dtype);
                long srcByteOffset = ElementOffsetToBytes(srcElemOffset, dtype);
                resultStorage.CopyDeviceFrom(srcStorage, resultByteOffset, srcByteOffset, innerBytes);

                int dim = outerDims - 1;
                while (dim >= 0)
                {
                    counter[dim]++;
                    if (counter[dim] < resultSizes[dim])
                        break;
                    counter[dim] = 0;
                    dim--;
                }

                if (dim < 0)
                    break;
            }

            return true;
        }

        private static bool SameStorageView(Tensor result, Tensor src)
        {
            if (result.StorageOffset != src.StorageOffset ||
                result.DimensionCount != src.DimensionCount)
            {
                return false;
            }

            for (int d = 0; d < result.DimensionCount; d++)
            {
                if (result.Sizes[d] != src.Sizes[d] || result.Strides[d] != src.Strides[d])
                    return false;
            }

            return true;
        }

        private static long ElementOffsetToBytes(long elementOffset, DType dtype)
        {
            if (dtype == DType.Q8_0)
            {
                if ((elementOffset % 32) != 0)
                    throw new InvalidOperationException(
                        $"Q8_0 byte offset requires element offset ({elementOffset}) to align to 32-element blocks.");
                return checked((elementOffset / 32) * 34);
            }

            return checked(elementOffset * dtype.Size());
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

        public static bool TryAddBiasRows(Tensor tensor, Tensor bias)
        {
            if (!TryGetContiguousRows(tensor, out CudaStorage tensorStorage, out IntPtr tensorPtr, out int rows, out int cols) ||
                !TryGetContiguousFloat(bias, out CudaStorage biasStorage, out IntPtr biasPtr, out int biasCount) ||
                biasCount <= 0 ||
                biasCount > cols)
            {
                return false;
            }

            CudaAllocator allocator = tensorStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            tensorStorage.EnsureDeviceCurrent();
            biasStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchAddBiasRowsF32(tensorPtr, biasPtr, rows, cols, biasCount, allocator.Stream.Handle);
            tensorStorage.MarkDeviceModified();
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

        public static bool TrySwiGluOaiSplit(Tensor result, Tensor gateUp, int halfDim, float alpha, float limit)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int count) ||
                !TryGetContiguousFloat(gateUp, out CudaStorage gateUpStorage, out IntPtr gateUpPtr, out _) ||
                result.DimensionCount != 2 ||
                gateUp.DimensionCount != 2 ||
                halfDim <= 0 ||
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
            kernels.LaunchSwiGluOaiSplitF32(
                gateUpPtr,
                resultPtr,
                checked((int)result.Sizes[0]),
                halfDim,
                alpha,
                limit,
                allocator.Stream.Handle);
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

        public static bool TryAttentionSoftmaxWithSinks(
            Tensor scores,
            Tensor sinks,
            int numHeads,
            int seqLen,
            int kvLen,
            int maskStart,
            int windowSize,
            float scale)
        {
            if (!TryGetContiguousFloat(scores, out CudaStorage scoreStorage, out IntPtr scorePtr, out int scoreCount) ||
                scores.DimensionCount != 3 ||
                numHeads <= 0 ||
                seqLen <= 0 ||
                kvLen <= 0 ||
                kvLen > 8192 ||
                maskStart < 0 ||
                windowSize < 0 ||
                scores.Sizes[0] != numHeads ||
                scores.Sizes[1] != seqLen ||
                scores.Sizes[2] != kvLen ||
                scoreCount != numHeads * seqLen * kvLen)
            {
                return false;
            }

            IntPtr sinksPtr = IntPtr.Zero;
            CudaStorage sinksStorage = null;
            int hasSinks = 0;
            if (sinks != null)
            {
                if (!TryGetContiguousFloat(sinks, out sinksStorage, out sinksPtr, out int sinkCount) ||
                    sinkCount < numHeads)
                {
                    return false;
                }

                hasSinks = 1;
            }

            CudaAllocator allocator = scoreStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            scoreStorage.EnsureDeviceCurrent();
            sinksStorage?.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchAttentionSoftmaxSinksF32(
                scorePtr,
                sinksPtr,
                numHeads,
                seqLen,
                kvLen,
                maskStart,
                windowSize,
                scale,
                hasSinks,
                allocator.Stream.Handle);
            scoreStorage.MarkDeviceModified();
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

        public static bool TryGqaDecodeAttentionWithSinks(
            Tensor result,
            Tensor query,
            Tensor keyCache,
            Tensor valueCache,
            Tensor sinks,
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
                !TryGetContiguousFloatOrHalf(keyCache, out CudaStorage keyStorage, out IntPtr keyPtr, out _, out bool keyIsHalf) ||
                !TryGetContiguousFloatOrHalf(valueCache, out CudaStorage valueStorage, out IntPtr valuePtr, out _, out bool valueIsHalf) ||
                query.DimensionCount != 2 ||
                result.DimensionCount != 2 ||
                keyCache.DimensionCount != 3 ||
                valueCache.DimensionCount != 3 ||
                numQHeads <= 0 ||
                numKVHeads <= 0 ||
                headDim <= 0 ||
                attendStart < 0 ||
                attendLen <= 0 ||
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
                (!circular && attendStart + attendLen > cacheSize) ||
                keyIsHalf != valueIsHalf)
            {
                return false;
            }

            IntPtr sinksPtr = IntPtr.Zero;
            CudaStorage sinksStorage = null;
            int hasSinks = 0;
            if (sinks != null)
            {
                if (!TryGetContiguousFloat(sinks, out sinksStorage, out sinksPtr, out int sinkCount) ||
                    sinkCount < numQHeads)
                {
                    return false;
                }

                hasSinks = 1;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            queryStorage.EnsureDeviceCurrent();
            keyStorage.EnsureDeviceCurrent();
            valueStorage.EnsureDeviceCurrent();
            sinksStorage?.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            if (TryLaunchPartitionedGqaDecodeAttention(
                    kernels,
                    allocator,
                    queryPtr,
                    keyPtr,
                    valuePtr,
                    sinksPtr,
                    resultPtr,
                    numQHeads,
                    numKVHeads,
                    headDim,
                    attendStart,
                    attendLen,
                    cacheSize,
                    circular,
                    scale,
                    hasSinks,
                    keyIsHalf))
            {
                resultStorage.MarkDeviceModified();
                return true;
            }

            if (attendLen > DecodeAttentionSingleBlockMaxTokens)
                return false;

            if (keyIsHalf)
            {
                kernels.LaunchGqaDecodeAttentionSinksF16(
                    queryPtr,
                    keyPtr,
                    valuePtr,
                    sinksPtr,
                    resultPtr,
                    numQHeads,
                    numKVHeads,
                    headDim,
                    attendStart,
                    attendLen,
                    cacheSize,
                    circular ? 1 : 0,
                    scale,
                    hasSinks,
                    allocator.Stream.Handle);
            }
            else
            {
                kernels.LaunchGqaDecodeAttentionSinksF32(
                    queryPtr,
                    keyPtr,
                    valuePtr,
                    sinksPtr,
                    resultPtr,
                    numQHeads,
                    numKVHeads,
                    headDim,
                    attendStart,
                    attendLen,
                    cacheSize,
                    circular ? 1 : 0,
                    scale,
                    hasSinks,
                    allocator.Stream.Handle);
            }
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
                !TryGetContiguousFloatOrHalf(key, out CudaStorage keyStorage, out IntPtr keyPtr, out _, out bool keyIsHalf) ||
                !TryGetContiguousFloatOrHalf(value, out CudaStorage valueStorage, out IntPtr valuePtr, out _, out bool valueIsHalf) ||
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
                maskStart >= kvLen ||
                keyIsHalf != valueIsHalf)
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
            if (keyIsHalf)
            {
                kernels.LaunchGqaPrefillAttentionF16(
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
            }
            else
            {
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
            }
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryGqaPrefillAttentionWithSinks(
            Tensor result,
            Tensor query,
            Tensor keyCache,
            Tensor valueCache,
            Tensor sinks,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int seqLen,
            int kvLen,
            int cacheSize,
            int maskStart,
            int windowSize,
            float scale)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !TryGetContiguousFloat(query, out CudaStorage queryStorage, out IntPtr queryPtr, out int queryCount) ||
                !TryGetContiguousFloatOrHalf(keyCache, out CudaStorage keyStorage, out IntPtr keyPtr, out _, out bool keyIsHalf) ||
                !TryGetContiguousFloatOrHalf(valueCache, out CudaStorage valueStorage, out IntPtr valuePtr, out _, out bool valueIsHalf) ||
                query.DimensionCount != 3 ||
                keyCache.DimensionCount != 3 ||
                valueCache.DimensionCount != 3 ||
                result.DimensionCount != 2 ||
                numQHeads <= 0 ||
                numKVHeads <= 0 ||
                headDim <= 0 ||
                seqLen <= 0 ||
                kvLen <= 0 ||
                kvLen > 8192 ||
                cacheSize <= 0 ||
                maskStart < 0 ||
                maskStart >= kvLen ||
                windowSize < 0 ||
                numQHeads % numKVHeads != 0 ||
                query.Sizes[0] != numQHeads ||
                query.Sizes[1] != seqLen ||
                query.Sizes[2] != headDim ||
                keyCache.Sizes[0] != numKVHeads ||
                keyCache.Sizes[1] != cacheSize ||
                keyCache.Sizes[2] != headDim ||
                valueCache.Sizes[0] != numKVHeads ||
                valueCache.Sizes[1] != cacheSize ||
                valueCache.Sizes[2] != headDim ||
                kvLen > cacheSize ||
                result.Sizes[0] != seqLen ||
                result.Sizes[1] != numQHeads * headDim ||
                queryCount != numQHeads * seqLen * headDim ||
                resultCount != seqLen * numQHeads * headDim ||
                keyIsHalf != valueIsHalf)
            {
                return false;
            }

            IntPtr sinksPtr = IntPtr.Zero;
            CudaStorage sinksStorage = null;
            int hasSinks = 0;
            if (sinks != null)
            {
                if (!TryGetContiguousFloat(sinks, out sinksStorage, out sinksPtr, out int sinkCount) ||
                    sinkCount < numQHeads)
                {
                    return false;
                }

                hasSinks = 1;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            queryStorage.EnsureDeviceCurrent();
            keyStorage.EnsureDeviceCurrent();
            valueStorage.EnsureDeviceCurrent();
            sinksStorage?.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            if (keyIsHalf)
            {
                kernels.LaunchGqaPrefillAttentionSinksF16(
                    queryPtr,
                    keyPtr,
                    valuePtr,
                    sinksPtr,
                    resultPtr,
                    numQHeads,
                    numKVHeads,
                    seqLen,
                    kvLen,
                    cacheSize,
                    headDim,
                    maskStart,
                    windowSize,
                    scale,
                    hasSinks,
                    allocator.Stream.Handle);
            }
            else
            {
                kernels.LaunchGqaPrefillAttentionSinksF32(
                    queryPtr,
                    keyPtr,
                    valuePtr,
                    sinksPtr,
                    resultPtr,
                    numQHeads,
                    numKVHeads,
                    seqLen,
                    kvLen,
                    cacheSize,
                    headDim,
                    maskStart,
                    windowSize,
                    scale,
                    hasSinks,
                    allocator.Stream.Handle);
            }

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
                !TryGetContiguousFloatOrHalf(keyCache, out CudaStorage keyStorage, out IntPtr keyPtr, out _, out bool keyIsHalf) ||
                !TryGetContiguousFloatOrHalf(valueCache, out CudaStorage valueStorage, out IntPtr valuePtr, out _, out bool valueIsHalf) ||
                query.DimensionCount != 2 ||
                result.DimensionCount != 2 ||
                keyCache.DimensionCount != 3 ||
                valueCache.DimensionCount != 3 ||
                numQHeads <= 0 ||
                numKVHeads <= 0 ||
                headDim <= 0 ||
                attendStart < 0 ||
                attendLen <= 0 ||
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
                (!circular && attendStart + attendLen > cacheSize) ||
                keyIsHalf != valueIsHalf)
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
            if (TryLaunchPartitionedGqaDecodeAttention(
                    kernels,
                    allocator,
                    queryPtr,
                    keyPtr,
                    valuePtr,
                    IntPtr.Zero,
                    resultPtr,
                    numQHeads,
                    numKVHeads,
                    headDim,
                    attendStart,
                    attendLen,
                    cacheSize,
                    circular,
                    scale,
                    hasSinks: 0,
                    keyIsHalf))
            {
                resultStorage.MarkDeviceModified();
                return true;
            }

            if (attendLen > DecodeAttentionSingleBlockMaxTokens)
                return false;

            if (keyIsHalf)
            {
                kernels.LaunchGqaDecodeAttentionF16(
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
            }
            else
            {
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
            }
            resultStorage.MarkDeviceModified();
            return true;
        }

        private static bool TryLaunchPartitionedGqaDecodeAttention(
            CudaKernels kernels,
            CudaAllocator allocator,
            IntPtr queryPtr,
            IntPtr keyPtr,
            IntPtr valuePtr,
            IntPtr sinksPtr,
            IntPtr resultPtr,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendStart,
            int attendLen,
            int cacheSize,
            bool circular,
            float scale,
            int hasSinks,
            bool keyIsHalf)
        {
            if (attendLen <= DecodeAttentionPartitionThreshold)
                return false;

            int numPartitions = (attendLen + DecodeAttentionPartitionSize - 1) / DecodeAttentionPartitionSize;
            using var partial = new Tensor(allocator, DType.Float32, numQHeads, numPartitions, headDim + 2);
            if (!TryGetContiguousFloat(partial, out _, out IntPtr partialPtr, out _))
                return false;

            if (keyIsHalf)
            {
                kernels.LaunchGqaDecodeAttentionPartitionF16(
                    queryPtr,
                    keyPtr,
                    valuePtr,
                    sinksPtr,
                    partialPtr,
                    numQHeads,
                    numKVHeads,
                    headDim,
                    attendStart,
                    attendLen,
                    cacheSize,
                    circular ? 1 : 0,
                    scale,
                    hasSinks,
                    numPartitions,
                    DecodeAttentionPartitionSize,
                    allocator.Stream.Handle);
            }
            else
            {
                kernels.LaunchGqaDecodeAttentionPartitionF32(
                    queryPtr,
                    keyPtr,
                    valuePtr,
                    sinksPtr,
                    partialPtr,
                    numQHeads,
                    numKVHeads,
                    headDim,
                    attendStart,
                    attendLen,
                    cacheSize,
                    circular ? 1 : 0,
                    scale,
                    hasSinks,
                    numPartitions,
                    DecodeAttentionPartitionSize,
                    allocator.Stream.Handle);
            }

            kernels.LaunchGqaDecodeAttentionPartitionReduceF32(
                partialPtr,
                resultPtr,
                numQHeads,
                headDim,
                numPartitions,
                allocator.Stream.Handle);
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
            if (!TryGetContiguousFloatOrHalf(cache, out CudaStorage cacheStorage, out IntPtr cachePtr, out _, out bool cacheIsHalf) ||
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
            if (cacheIsHalf)
            {
                kernels.LaunchCopyHeadFirstToCacheF16(
                    srcPtr,
                    cachePtr,
                    checked((int)src.Sizes[0]),
                    seqLen,
                    checked((int)src.Sizes[2]),
                    startPos,
                    cacheSize,
                    circular ? 1 : 0,
                    allocator.Stream.Handle);
            }
            else
            {
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
            }
            cacheStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryGatherCircularHeadFirst(Tensor result, Tensor cache, int startPos, int seqLen, int cacheSize)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !TryGetContiguousFloatOrHalf(cache, out CudaStorage cacheStorage, out IntPtr cachePtr, out _, out bool cacheIsHalf) ||
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
            if (cacheIsHalf)
            {
                kernels.LaunchGatherCircularHeadFirstF16(
                    cachePtr,
                    resultPtr,
                    checked((int)result.Sizes[0]),
                    seqLen,
                    checked((int)result.Sizes[2]),
                    startPos,
                    cacheSize,
                    allocator.Stream.Handle);
            }
            else
            {
                kernels.LaunchGatherCircularHeadFirstF32(
                    cachePtr,
                    resultPtr,
                    checked((int)result.Sizes[0]),
                    seqLen,
                    checked((int)result.Sizes[2]),
                    startPos,
                    cacheSize,
                    allocator.Stream.Handle);
            }
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

        public static bool TryQwen35GatedDeltaNetPacked(
            Tensor result,
            Tensor packed,
            Tensor convState,
            Tensor ssmState,
            Tensor convWeight,
            Tensor dtBias,
            Tensor aLog,
            Tensor ssmNorm,
            int seqLen,
            int packedDim,
            int qkvDim,
            int qkDim,
            int vDim,
            int numKHeads,
            int numVHeads,
            int headKDim,
            int headVDim,
            int convKernel,
            int convWriteIdx,
            float eps)
        {
            int convDim = convKernel - 1;
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !TryGetContiguousFloat(packed, out CudaStorage packedStorage, out IntPtr packedPtr, out int packedCount) ||
                !TryGetContiguousFloat(convState, out CudaStorage convStateStorage, out IntPtr convStatePtr, out int convStateCount) ||
                !TryGetContiguousFloat(ssmState, out CudaStorage ssmStateStorage, out IntPtr ssmStatePtr, out int ssmStateCount) ||
                !TryGetContiguousFloat(convWeight, out CudaStorage convWeightStorage, out IntPtr convWeightPtr, out int convWeightCount) ||
                !TryGetContiguousFloat(dtBias, out CudaStorage dtBiasStorage, out IntPtr dtBiasPtr, out int dtBiasCount) ||
                !TryGetContiguousFloat(aLog, out CudaStorage aLogStorage, out IntPtr aLogPtr, out int aLogCount) ||
                !TryGetContiguousFloat(ssmNorm, out CudaStorage ssmNormStorage, out IntPtr ssmNormPtr, out int ssmNormCount) ||
                seqLen <= 0 ||
                packedDim <= 0 ||
                qkvDim <= 0 ||
                qkDim <= 0 ||
                vDim <= 0 ||
                numKHeads <= 0 ||
                numVHeads <= 0 ||
                headKDim <= 0 ||
                headVDim <= 0 ||
                convKernel <= 0 ||
                convDim <= 0 ||
                convWriteIdx < 0 ||
                convWriteIdx >= convDim ||
                eps <= 0.0f ||
                qkDim != checked(numKHeads * headKDim) ||
                vDim != checked(numVHeads * headVDim) ||
                qkvDim != checked(2 * qkDim + vDim) ||
                packedDim < checked(qkvDim + vDim + 2 * numVHeads) ||
                result.DimensionCount != 2 ||
                packed.DimensionCount != 2 ||
                result.Sizes[0] != seqLen ||
                result.Sizes[1] != vDim ||
                packed.Sizes[0] != seqLen ||
                packed.Sizes[1] != packedDim ||
                resultCount != checked(seqLen * vDim) ||
                packedCount != checked(seqLen * packedDim) ||
                convStateCount != checked(convDim * qkvDim) ||
                ssmStateCount != checked(numVHeads * headVDim * headKDim) ||
                convWeightCount != checked(qkvDim * convKernel) ||
                dtBiasCount < numVHeads ||
                aLogCount < numVHeads ||
                ssmNormCount < headVDim)
            {
                return false;
            }

            int updateTail = Math.Min(seqLen, convDim);
            if (updateTail > 0)
                _ = checked(updateTail * qkvDim);

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            packedStorage.EnsureDeviceCurrent();
            convStateStorage.EnsureDeviceCurrent();
            ssmStateStorage.EnsureDeviceCurrent();
            convWeightStorage.EnsureDeviceCurrent();
            dtBiasStorage.EnsureDeviceCurrent();
            aLogStorage.EnsureDeviceCurrent();
            ssmNormStorage.EnsureDeviceCurrent();

            allocator.Context.MakeCurrent();
            kernels.LaunchQwen35GatedDeltaNetPackedF32(
                packedPtr,
                convStatePtr,
                ssmStatePtr,
                convWeightPtr,
                dtBiasPtr,
                aLogPtr,
                ssmNormPtr,
                resultPtr,
                seqLen,
                packedDim,
                qkvDim,
                qkDim,
                vDim,
                numKHeads,
                numVHeads,
                headKDim,
                headVDim,
                convKernel,
                convWriteIdx,
                eps,
                allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            convStateStorage.MarkDeviceModified();
            ssmStateStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryQwen35GatedDeltaNetPackInputs(
            Tensor packed,
            Tensor qkv,
            Tensor z,
            Tensor beta,
            Tensor alpha,
            int seqLen,
            int qkvDim,
            int zDim,
            int numVHeads,
            int packedDim)
        {
            if (!TryGetContiguousFloat(packed, out CudaStorage packedStorage, out IntPtr packedPtr, out int packedCount) ||
                !TryGetContiguousFloat(qkv, out CudaStorage qkvStorage, out IntPtr qkvPtr, out int qkvCount) ||
                !TryGetContiguousFloat(z, out CudaStorage zStorage, out IntPtr zPtr, out int zCount) ||
                !TryGetContiguousFloat(beta, out CudaStorage betaStorage, out IntPtr betaPtr, out int betaCount) ||
                !TryGetContiguousFloat(alpha, out CudaStorage alphaStorage, out IntPtr alphaPtr, out int alphaCount) ||
                seqLen <= 0 ||
                qkvDim <= 0 ||
                zDim <= 0 ||
                numVHeads <= 0 ||
                packedDim != checked(qkvDim + zDim + 2 * numVHeads) ||
                packed.DimensionCount != 2 ||
                qkv.DimensionCount != 2 ||
                z.DimensionCount != 2 ||
                beta.DimensionCount != 2 ||
                alpha.DimensionCount != 2 ||
                packed.Sizes[0] != seqLen ||
                packed.Sizes[1] != packedDim ||
                qkv.Sizes[0] != seqLen ||
                qkv.Sizes[1] != qkvDim ||
                z.Sizes[0] != seqLen ||
                z.Sizes[1] != zDim ||
                beta.Sizes[0] != seqLen ||
                beta.Sizes[1] != numVHeads ||
                alpha.Sizes[0] != seqLen ||
                alpha.Sizes[1] != numVHeads ||
                packedCount != checked(seqLen * packedDim) ||
                qkvCount != checked(seqLen * qkvDim) ||
                zCount != checked(seqLen * zDim) ||
                betaCount != checked(seqLen * numVHeads) ||
                alphaCount != checked(seqLen * numVHeads))
            {
                return false;
            }

            CudaAllocator allocator = packedStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            qkvStorage.EnsureDeviceCurrent();
            zStorage.EnsureDeviceCurrent();
            betaStorage.EnsureDeviceCurrent();
            alphaStorage.EnsureDeviceCurrent();

            allocator.Context.MakeCurrent();
            kernels.LaunchQwen35GatedDeltaNetPackInputsF32(
                qkvPtr,
                zPtr,
                betaPtr,
                alphaPtr,
                packedPtr,
                seqLen,
                qkvDim,
                zDim,
                numVHeads,
                packedDim,
                allocator.Stream.Handle);
            packedStorage.MarkDeviceModified();
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

        internal static bool TryGetContiguousFloatOrHalf(Tensor tensor, out CudaStorage storage, out IntPtr ptr, out int count, out bool isHalf)
        {
            if (TryGetContiguous(tensor, out storage, out ptr, out long longCount) &&
                (tensor.ElementType == DType.Float32 || tensor.ElementType == DType.Float16) &&
                longCount <= int.MaxValue)
            {
                count = (int)longCount;
                isHalf = tensor.ElementType == DType.Float16;
                return true;
            }

            storage = null;
            ptr = IntPtr.Zero;
            count = 0;
            isHalf = false;
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
