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
        // The group4 kernels map one WARP to a key row (vs the generic partition
        // kernel's one THREAD per key), so they stay fully populated at much
        // smaller partitions. Smaller partitions matter because Gemma-class
        // models have very few KV heads (2): the grid is
        // (kv_heads x partitions) and at 512-token partitions a 2048-token
        // context ran on 8 CTAs of a 48-SM GPU.
        private const int Group4DecodeAttentionPartitionSize = 128;
        // Circular SWA ring (d=256): partition the physical ring so the window
        // stops serializing behind num_kv_heads CTAs. 64-token partitions put a
        // 512-token ring on 8 partitions x kv_heads CTAs.
        private const int Group4RingPartitionSize = 64;
        // Below this live window the single-block kernel's lower fixed overhead
        // (no partial scratch + reduce pass) wins over partitioning.
        private const int Group4RingPartitionMinTokens = 128;
        // Keep the ring partition grid bounded if a model ever runs a large
        // circular cache through this route.
        private const int Group4RingMaxPartitions = 64;
        // A/B-tunable because the crossover depends on head width, context and
        // GPU SM count. On the RTX 3080 report system, partitioning from 512
        // tokens keeps the small-KV-head Gemma/Qwen grids populated and improved
        // ~2K-context decode by 17% / 7% respectively.
        internal const int DecodeAttentionSingleBlockMaxTokens = 8192;
        // The grouped d=512 kernel keeps four score rows plus four query rows
        // in dynamic shared memory. At 2,048 tokens this is 40 KiB, below the
        // portable 48-KiB per-block limit; larger contexts use partitioning
        // even when the diagnostic threshold is raised.
        private const int GqaDecodeGroup4D512SingleBlockMaxTokens = 2048;
        private static readonly int DecodeAttentionPartitionThreshold =
            Math.Min(
                ReadPositiveEnvInt("TS_CUDA_DECODE_ATTN_PARTITION_THRESHOLD", 512),
                DecodeAttentionSingleBlockMaxTokens);

        private static int ReadPositiveEnvInt(string name, int fallback)
        {
            string value = Environment.GetEnvironmentVariable(name);
            return !string.IsNullOrEmpty(value) && int.TryParse(value, out int parsed) && parsed > 0
                ? parsed
                : fallback;
        }

        internal static int GetGqaDecodeAttentionRouteTier(
            bool keyIsHalf,
            bool circular,
            int numQHeads,
            int numKVHeads,
            int headDim,
            int attendLen,
            int cacheSize,
            int hasSinks)
        {
            int effectiveAttendLen = circular ? Math.Min(attendLen, cacheSize) : attendLen;
            bool useGroup4D512 =
                keyIsHalf &&
                CudaKernels.GqaDecodeGroup4Enabled &&
                !circular &&
                hasSinks == 0 &&
                numQHeads == numKVHeads * 4 &&
                headDim == 512;
            int partitionThreshold = useGroup4D512
                ? Math.Min(
                    DecodeAttentionPartitionThreshold,
                    GqaDecodeGroup4D512SingleBlockMaxTokens)
                : DecodeAttentionPartitionThreshold;
            return effectiveAttendLen <= partitionThreshold ? 0 : 1;
        }

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

            // Fold the innermost surviving loop dimension into the copy kernel's
            // "rows" axis: one pitched 2D device copy per remaining outer group,
            // instead of one cuMemcpyDtoDAsync per single row. For the dominant
            // 2D NewContiguous/Narrow case (outerDims == 1) that is a SINGLE
            // kernel launch covering every row — replacing what used to be
            // thousands of driver submissions per prefill forward pass.
            int rowDim = outerDims - 1;
            long rowCount = resultSizes[rowDim];
            bool pitchOk = dtype != DType.Q8_0
                || ((srcStrides[rowDim] % 32) == 0 && (resultStrides[rowDim] % 32) == 0);

            if (pitchOk)
            {
                long srcRowPitch = ElementOffsetToBytes(srcStrides[rowDim], dtype);
                long dstRowPitch = ElementOffsetToBytes(resultStrides[rowDim], dtype);
                int groupDims = rowDim; // host-looped dims are [0, rowDim)
                long[] gcounter = groupDims > 0 ? new long[groupDims] : Array.Empty<long>();

                while (true)
                {
                    long resultElemOffset = result.StorageOffset;
                    long srcElemOffset = src.StorageOffset;
                    for (int d = 0; d < groupDims; d++)
                    {
                        resultElemOffset = checked(resultElemOffset + gcounter[d] * resultStrides[d]);
                        srcElemOffset = checked(srcElemOffset + gcounter[d] * srcStrides[d]);
                    }

                    long resultByteOffset = ElementOffsetToBytes(resultElemOffset, dtype);
                    long srcByteOffset = ElementOffsetToBytes(srcElemOffset, dtype);

                    bool copied = resultStorage.TryCopyDeviceFrom2D(
                        srcStorage, resultByteOffset, srcByteOffset,
                        rowCount, innerBytes, dstRowPitch, srcRowPitch);
                    if (!copied)
                    {
                        // Cross-allocator storage pair: per-row driver copies.
                        for (long r = 0; r < rowCount; r++)
                            resultStorage.CopyDeviceFrom(
                                srcStorage,
                                resultByteOffset + r * dstRowPitch,
                                srcByteOffset + r * srcRowPitch,
                                innerBytes);
                    }

                    int dim = groupDims - 1;
                    while (dim >= 0)
                    {
                        gcounter[dim]++;
                        if (gcounter[dim] < resultSizes[dim])
                            break;
                        gcounter[dim] = 0;
                        dim--;
                    }

                    if (dim < 0)
                        break;
                }

                return true;
            }

            // Fallback: original one-row-per-driver-copy loop (Q8_0 with a
            // non-block-aligned row stride, so a byte pitch can't be formed).
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
            if (TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int count) &&
                TryGetContiguousFloat(lhs, out CudaStorage lhsStorage, out IntPtr lhsPtr, out int lhsCount) &&
                TryGetContiguousFloat(rhs, out CudaStorage rhsStorage, out IntPtr rhsPtr, out int rhsCount) &&
                count == lhsCount && count == rhsCount)
            {
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

            return TryBinaryActivationStrided(result, lhs, rhs, op);
        }

        // Device-side deinterleave of a fused Q+gate projection (src rows are
        // num_heads blocks of [q(head_dim) | gate(head_dim)], possibly a Narrow'd
        // slice of a wider fused-QKV row). Writes dense q/gate without the CPU
        // round trip the host deinterleave forces on every attention layer.
        public static bool TryDeinterleaveQGate(Tensor q, Tensor gate, Tensor src, int rows, int numHeads, int headDim)
        {
            long perRow = (long)numHeads * headDim;
            if (!TryGetContiguousFloat(q, out CudaStorage qStorage, out IntPtr qPtr, out int qCount) ||
                !TryGetContiguousFloat(gate, out CudaStorage gateStorage, out IntPtr gatePtr, out int gateCount) ||
                !TryGetRowStridedFloat(src, out CudaStorage srcStorage, out IntPtr srcPtr, out int srcRows, out int srcCols, out long srcStride) ||
                qCount != gateCount || qCount != checked(rows * perRow) ||
                srcRows != rows || srcCols != checked((int)(perRow * 2)))
            {
                return false;
            }

            CudaAllocator allocator = qStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            srcStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchDeinterleaveQGateF32(srcPtr, qPtr, gatePtr, rows, numHeads, headDim, srcStride, allocator.Stream.Handle);
            qStorage.MarkDeviceModified();
            gateStorage.MarkDeviceModified();
            return true;
        }

        // Row-strided variant: any operand may be a padded-row 2D view (e.g. the
        // gate/up halves Narrow'd out of a fused gate+up projection). Without this,
        // prefill-sized SwiGLU activations dropped to the CPU fallback, whose full
        // device->host->device round trip dominated direct-CUDA prefill time.
        private static bool TryBinaryActivationStrided(Tensor result, Tensor lhs, Tensor rhs, CudaBinaryActivationOp op)
        {
            if (!TryGetRowStridedFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int rows, out int cols, out long outStride) ||
                !TryGetRowStridedFloat(lhs, out CudaStorage lhsStorage, out IntPtr lhsPtr, out int lhsRows, out int lhsCols, out long lhsStride) ||
                !TryGetRowStridedFloat(rhs, out CudaStorage rhsStorage, out IntPtr rhsPtr, out int rhsRows, out int rhsCols, out long rhsStride) ||
                rows != lhsRows || rows != rhsRows || cols != lhsCols || cols != rhsCols)
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            lhsStorage.EnsureDeviceCurrent();
            rhsStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchBinaryActivationStridedF32(
                lhsPtr, rhsPtr, resultPtr, rows, cols, lhsStride, rhsStride, outStride, (int)op, allocator.Stream.Handle);
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
            float scale,
            int kvStride = -1)
        {
            // kvStride < 0: contiguous [numKVHeads, kvLen, headDim] (seq-heads case).
            // kvStride > 0: live cache [numKVHeads, kvStride, headDim] read in place;
            // the kernel attends the first kvLen logical positions (kvLen <= kvStride).
            bool liveCache = kvStride > 0;
            int kStride = liveCache ? kvStride : kvLen;
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
                kvLen > kStride ||
                numQHeads % numKVHeads != 0 ||
                query.Sizes[0] != numQHeads ||
                query.Sizes[1] != seqLen ||
                query.Sizes[2] != headDim ||
                key.Sizes[0] != numKVHeads ||
                key.Sizes[1] != kStride ||
                key.Sizes[2] != headDim ||
                value.Sizes[0] != numKVHeads ||
                value.Sizes[1] != kStride ||
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
            bool useGroup4D256 =
                CudaKernels.GqaPrefillGroup4Enabled &&
                CudaKernels.GqaPrefillWarpCooperativeEnabled &&
                numQHeads == numKVHeads * 4 &&
                headDim == 256 &&
                seqLen >= 32 &&
                windowSize > 0 &&
                windowSize <= 512 &&
                (long)maskStart + seqLen <= kvLen;
            bool useGroup4D512 =
                CudaKernels.GqaPrefillGroup4Enabled &&
                CudaKernels.GqaPrefillWarpCooperativeEnabled &&
                numQHeads == numKVHeads * 4 &&
                headDim == 512 &&
                seqLen >= 128 &&
                kvLen <= 2048 &&
                windowSize == 0 &&
                (long)maskStart + seqLen <= kvLen;
            bool useGroup4OnlineD512 =
                CudaKernels.GqaPrefillGroup4Enabled &&
                CudaKernels.GqaPrefillWarpCooperativeEnabled &&
                numQHeads == numKVHeads * 4 &&
                headDim == 512 &&
                kvLen > 2048 &&
                kvLen <= 8192 &&
                windowSize == 0 &&
                (long)maskStart + seqLen <= kvLen;
            // Flash-style tiled kernel: same routing eligibility as the two-pass
            // group4 kernels it replaces (both f16 and f32 K/V), but the Q tile
            // is staged once and K/V traffic is amortized across 16-32 score
            // rows per CTA.
            if (CudaKernels.GqaPrefillFlashEnabled &&
                (useGroup4D256 || useGroup4D512 || useGroup4OnlineD512))
            {
                // flash2 = flash1 with a larger K chunk (fewer sync rounds); same
                // f32 math, so it serves both cache dtypes.
                if (CudaKernels.GqaPrefillFlash2Enabled && kernels.Flash2Supports(headDim))
                {
                    kernels.LaunchGqaPrefillFlash2Group4(
                        queryPtr, keyPtr, valuePtr, IntPtr.Zero, resultPtr,
                        numQHeads, numKVHeads, seqLen, kvLen, headDim, maskStart, windowSize,
                        scale, kStride, hasSinks: 0, keyIsHalf, allocator.Stream.Handle);
                    resultStorage.MarkDeviceModified();
                    return true;
                }
                kernels.LaunchGqaPrefillFlashGroup4(
                    queryPtr,
                    keyPtr,
                    valuePtr,
                    IntPtr.Zero,
                    resultPtr,
                    numQHeads,
                    numKVHeads,
                    seqLen,
                    kvLen,
                    headDim,
                    maskStart,
                    windowSize,
                    scale,
                    kStride,
                    hasSinks: 0,
                    keyIsHalf,
                    allocator.Stream.Handle);
                resultStorage.MarkDeviceModified();
                return true;
            }
            if (useGroup4OnlineD512)
            {
                if (keyIsHalf)
                {
                    kernels.LaunchGqaPrefillAttentionGroup4OnlineD512F16(
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
                        kStride,
                        allocator.Stream.Handle);
                }
                else
                {
                    kernels.LaunchGqaPrefillAttentionGroup4OnlineD512F32(
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
                        kStride,
                        allocator.Stream.Handle);
                }
            }
            else if (useGroup4D512 && keyIsHalf)
            {
                kernels.LaunchGqaPrefillAttentionGroup4D512F16(
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
                    kStride,
                    allocator.Stream.Handle);
            }
            else if (useGroup4D512)
            {
                kernels.LaunchGqaPrefillAttentionGroup4D512F32(
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
                    kStride,
                    allocator.Stream.Handle);
            }
            else if (useGroup4D256 && keyIsHalf)
            {
                kernels.LaunchGqaPrefillAttentionGroup4D256F16(
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
                    kStride,
                    allocator.Stream.Handle);
            }
            else if (useGroup4D256)
            {
                kernels.LaunchGqaPrefillAttentionGroup4D256F32(
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
                    kStride,
                    allocator.Stream.Handle);
            }
            else if (keyIsHalf)
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
                    kStride,
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
                    kStride,
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
            // Flash-style tiled kernel for the 4-queries-per-KV-head layouts
            // (Qwen 3.5 full attention: 16q/4kv, d=256, no window). The
            // one-CTA-per-(q_head, position) sinks kernels walk the whole
            // visible K/V serially per query (~38 ms per 2k-token layer);
            // the flash tile amortizes K/V across 32 score rows.
            if (CudaKernels.GqaPrefillFlashEnabled &&
                CudaKernels.GqaPrefillGroup4Enabled &&
                numQHeads == numKVHeads * 4 &&
                (headDim == 256 || headDim == 512) &&
                seqLen >= 16 &&
                (long)maskStart + seqLen <= kvLen)
            {
                // flash2 = flash1 with a larger K chunk (fewer sync rounds).
                if (CudaKernels.GqaPrefillFlash2Enabled && kernels.Flash2Supports(headDim))
                {
                    kernels.LaunchGqaPrefillFlash2Group4(
                        queryPtr, keyPtr, valuePtr, sinksPtr, resultPtr,
                        numQHeads, numKVHeads, seqLen, kvLen, headDim, maskStart, windowSize,
                        scale, cacheSize, hasSinks, keyIsHalf, allocator.Stream.Handle);
                    resultStorage.MarkDeviceModified();
                    return true;
                }
                kernels.LaunchGqaPrefillFlashGroup4(
                    queryPtr,
                    keyPtr,
                    valuePtr,
                    sinksPtr,
                    resultPtr,
                    numQHeads,
                    numKVHeads,
                    seqLen,
                    kvLen,
                    headDim,
                    maskStart,
                    windowSize,
                    scale,
                    cacheSize,
                    hasSinks,
                    keyIsHalf,
                    allocator.Stream.Handle);
                resultStorage.MarkDeviceModified();
                return true;
            }
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
            IntPtr dyn = CudaDecodeDynParams.GetActiveDevicePtr(allocator);
            bool useGroup4D512 =
                keyIsHalf &&
                CudaKernels.GqaDecodeGroup4Enabled &&
                !circular &&
                numQHeads == numKVHeads * 4 &&
                headDim == 512;
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
                    keyIsHalf,
                    dyn))
            {
                resultStorage.MarkDeviceModified();
                return true;
            }

            if (attendLen > DecodeAttentionSingleBlockMaxTokens)
                return false;

            // Decode-graph capture: the single-block kernel reads attend_len from
            // the dyn block on replay, so the scores[] shared buffer must be sized
            // for the largest length the graph may see. Past the partition
            // threshold the plain path routes to the partitioned kernels, so the
            // captured entry expires there and gets re-captured on that route.
            int smemTokens = 0;
            if (dyn != IntPtr.Zero)
            {
                int captureTokenLimit = useGroup4D512
                    ? Math.Min(
                        DecodeAttentionPartitionThreshold,
                        GqaDecodeGroup4D512SingleBlockMaxTokens)
                    : DecodeAttentionPartitionThreshold;
                int availableCacheTokens = circular
                    ? cacheSize
                    : cacheSize - attendStart;
                smemTokens = Math.Min(captureTokenLimit, availableCacheTokens);
                // A circular cache no larger than this single-block allocation
                // clamps safely forever. A larger window must expire at the
                // partition threshold and recapture on the partitioned route.
                if (!circular || cacheSize > captureTokenLimit)
                    CudaDecodeDynParams.LimitCaptureAttendLen(smemTokens);
            }

            if (keyIsHalf &&
                CudaKernels.GqaDecodeGroup4Enabled &&
                circular &&
                (attendLen >= cacheSize || attendStart == 0) &&
                numQHeads == numKVHeads * 4 &&
                headDim == 256)
            {
                kernels.LaunchGqaDecodeAttentionGroup4D256F16(
                    queryPtr,
                    keyPtr,
                    valuePtr,
                    resultPtr,
                    numKVHeads,
                    attendLen,
                    cacheSize,
                    scale,
                    allocator.Stream.Handle,
                    dyn,
                    smemTokens);
            }
            else if (useGroup4D512)
            {
                kernels.LaunchGqaDecodeAttentionGroup4D512F16(
                    queryPtr,
                    keyPtr,
                    valuePtr,
                    resultPtr,
                    numKVHeads,
                    attendStart,
                    attendLen,
                    cacheSize,
                    scale,
                    allocator.Stream.Handle,
                    dyn,
                    smemTokens);
            }
            else if (keyIsHalf)
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
                    allocator.Stream.Handle,
                    dyn,
                    smemTokens);
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
                    allocator.Stream.Handle,
                    dyn,
                    smemTokens);
            }
            resultStorage.MarkDeviceModified();
            return true;
        }

        internal static bool TryLaunchPartitionedGqaDecodeAttention(
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
            bool keyIsHalf,
            IntPtr dyn = default)
        {
            // A circular SWA cache never attends more than its physical
            // capacity. Do not route it through the partition+reduce path just
            // because the model's logical sequence length has grown large.
            bool useGroup4D512 =
                keyIsHalf &&
                CudaKernels.GqaDecodeGroup4Enabled &&
                !circular &&
                hasSinks == 0 &&
                numQHeads == numKVHeads * 4 &&
                headDim == 512;
            // d=256 grouped decode attention, partitioned. Covers BOTH the
            // circular SWA ring (Gemma local layers; softmax is invariant to
            // the ring permutation) and the linear attend-from-0 case with
            // optional attention sinks (Qwen 3.5 full-attention layers) -- the
            // kernel walks physical positions [0, min(attend_len, cache_size))
            // either way. Partitioning matters because these models have very
            // few KV heads (2-4): the single-block kernels strand the GPU on
            // that many CTAs. Under decode-graph capture (dyn present) this
            // route is unconditional so ONE captured kernel sequence stays
            // valid for every live length; uncaptured short windows keep the
            // lower-overhead single-block kernels.
            bool useGroup4D256Ring =
                keyIsHalf &&
                CudaKernels.GqaDecodeGroup4Enabled &&
                numQHeads == numKVHeads * 4 &&
                headDim == 256 &&
                (circular
                    ? hasSinks == 0 && (attendLen >= cacheSize || attendStart == 0)
                    : attendStart == 0);
            if (useGroup4D256Ring)
            {
                int ringLen = Math.Min(attendLen, cacheSize);
                if (dyn == IntPtr.Zero && ringLen <= Group4RingPartitionMinTokens)
                    return false;
                int ringPartSize = Math.Max(
                    Group4RingPartitionSize,
                    (cacheSize + Group4RingMaxPartitions - 1) / Group4RingMaxPartitions);
                ringPartSize = (ringPartSize + 31) & ~31;
                int coveredLen = dyn != IntPtr.Zero ? cacheSize : ringLen;
                int ringParts = (coveredLen + ringPartSize - 1) / ringPartSize;
                using var ringPartial = new Tensor(
                    allocator, DType.Float32, numQHeads, ringParts, headDim + 2);
                if (!TryGetContiguousFloat(ringPartial, out _, out IntPtr ringPartialPtr, out _))
                    return false;
                kernels.LaunchGqaDecodeAttentionPartitionGroup4D256F16(
                    queryPtr,
                    keyPtr,
                    valuePtr,
                    sinksPtr,
                    ringPartialPtr,
                    numKVHeads,
                    attendLen,
                    cacheSize,
                    scale,
                    hasSinks,
                    ringParts,
                    ringPartSize,
                    allocator.Stream.Handle,
                    dyn);
                kernels.LaunchGqaDecodeAttentionPartitionReduceF32(
                    ringPartialPtr,
                    resultPtr,
                    numQHeads,
                    headDim,
                    ringParts,
                    allocator.Stream.Handle);
                return true;
            }
            if (GetGqaDecodeAttentionRouteTier(
                    keyIsHalf,
                    circular,
                    numQHeads,
                    numKVHeads,
                    headDim,
                    attendLen,
                    cacheSize,
                    hasSinks) == 0)
                return false;

            int partitionSize = useGroup4D512
                ? Group4DecodeAttentionPartitionSize
                : DecodeAttentionPartitionSize;
            if (useGroup4D512 && dyn != IntPtr.Zero)
            {
                // Keep the capacity-sized capture grid bounded: empty partitions
                // still write their partial rows and the reduce still reads them,
                // so a huge KV capacity must not explode the partition count.
                int minSize = (cacheSize + Group4RingMaxPartitions - 1) / Group4RingMaxPartitions;
                partitionSize = Math.Max(partitionSize, (minSize + 31) & ~31);
            }

            // Decode-graph capture: the partition grid and scratch are baked into
            // the graph, so size them for the cache CAPACITY and let the kernels
            // read the live attend_len from the dyn block. Partitions past the
            // live length write (max=-inf, sum=0) rows the reduce kernel skips,
            // which keeps the result bit-identical to the exact-partition launch.
            int numPartitions = dyn != IntPtr.Zero
                ? (cacheSize + partitionSize - 1) / partitionSize
                : (attendLen + partitionSize - 1) / partitionSize;
            using var partial = new Tensor(allocator, DType.Float32, numQHeads, numPartitions, headDim + 2);
            if (!TryGetContiguousFloat(partial, out _, out IntPtr partialPtr, out _))
                return false;

            if (useGroup4D512)
            {
                kernels.LaunchGqaDecodeAttentionPartitionGroup4D512F16(
                    queryPtr,
                    keyPtr,
                    valuePtr,
                    partialPtr,
                    numKVHeads,
                    attendStart,
                    attendLen,
                    cacheSize,
                    scale,
                    numPartitions,
                    partitionSize,
                    allocator.Stream.Handle,
                    dyn);
            }
            else if (keyIsHalf)
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
                    partitionSize,
                    allocator.Stream.Handle,
                    dyn);
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
                    partitionSize,
                    allocator.Stream.Handle,
                    dyn);
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
            // Decode-graph capture: the single-token append re-reads its write
            // position from the dyn block on every replay.
            IntPtr dyn = seqLen == 1
                ? CudaDecodeDynParams.GetActiveDevicePtr(allocator)
                : IntPtr.Zero;
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
                    allocator.Stream.Handle,
                    dyn);
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
                    allocator.Stream.Handle,
                    dyn);
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

        public static bool TryExpandKvHeads(Tensor result, Tensor cache, int groupSize, int seqLen)
        {
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !TryGetContiguousFloatOrHalf(cache, out CudaStorage cacheStorage, out IntPtr cachePtr, out _, out bool cacheIsHalf) ||
                result.DimensionCount != 3 ||
                cache.DimensionCount != 3 ||
                groupSize <= 0 ||
                seqLen <= 0 ||
                seqLen > cache.Sizes[1] ||
                result.Sizes[0] != cache.Sizes[0] * groupSize ||
                result.Sizes[1] != seqLen ||
                result.Sizes[2] != cache.Sizes[2] ||
                resultCount != result.Sizes[0] * seqLen * result.Sizes[2])
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!ReferenceEquals(allocator, cacheStorage.AllocatorImpl) ||
                !TryGetKernels(allocator, out CudaKernels kernels))
            {
                return false;
            }

            cacheStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchExpandKvHeads(
                cachePtr,
                resultPtr,
                checked((int)cache.Sizes[0]),
                seqLen,
                checked((int)cache.Sizes[1]),
                checked((int)cache.Sizes[2]),
                groupSize,
                cacheIsHalf,
                allocator.Stream.Handle);
            resultStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryRepeatInterleave(Tensor result, Tensor src, int repeats, int dim)
        {
            if (result == null || src == null ||
                result.Storage is not CudaStorage resultStorage ||
                src.Storage is not CudaStorage srcStorage ||
                result.ElementType != src.ElementType ||
                (src.ElementType != DType.Float32 && src.ElementType != DType.Float16) ||
                !result.IsContiguous() ||
                !src.IsContiguous() ||
                repeats <= 0 ||
                dim < 0 ||
                dim >= src.DimensionCount ||
                result.DimensionCount != src.DimensionCount)
            {
                return false;
            }

            long outer = 1;
            long inner = 1;
            for (int d = 0; d < src.DimensionCount; d++)
            {
                long expected = src.Sizes[d] * (d == dim ? repeats : 1);
                if (result.Sizes[d] != expected)
                    return false;
                if (d < dim)
                    outer = checked(outer * src.Sizes[d]);
                else if (d > dim)
                    inner = checked(inner * src.Sizes[d]);
            }

            long total = checked(outer * src.Sizes[dim] * repeats * inner);
            if (outer > int.MaxValue || inner > int.MaxValue ||
                src.Sizes[dim] > int.MaxValue || total > int.MaxValue)
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!ReferenceEquals(allocator, srcStorage.AllocatorImpl) ||
                !TryGetKernels(allocator, out CudaKernels kernels))
            {
                return false;
            }

            srcStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchRepeatInterleave(
                srcStorage.DevicePtrAtElement(src.StorageOffset),
                resultStorage.DevicePtrAtElement(result.StorageOffset),
                (int)outer,
                checked((int)src.Sizes[dim]),
                repeats,
                (int)inner,
                src.ElementType == DType.Float16,
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

        // NeoX RoPE for the FLAT [seqLen, numHeads*headDim] layout. The data tensor
        // need only be contiguous with numHeads*seqLen*headDim elements (q/k carry a
        // 2D [seqLen, numHeads*headDim] shape before ReshapeToHeads, or [1, *] for
        // decode), so this checks element count rather than a fixed rank.
        // residual = residual + rms_norm(input, alpha) (Gemma post-norm), fused into one
        // kernel. residual/input are [rows, cols] contiguous f32; alpha is [cols].
        public static bool TryRmsNormResidualAdd(Tensor residual, Tensor input, Tensor alpha, float eps)
        {
            if (!TryGetContiguousFloat(residual, out CudaStorage resStorage, out IntPtr resPtr, out int resCount) ||
                !TryGetContiguousFloat(input, out CudaStorage inStorage, out IntPtr inPtr, out int inCount) ||
                !TryGetContiguousFloat(alpha, out CudaStorage alphaStorage, out IntPtr alphaPtr, out int alphaCount) ||
                residual.DimensionCount != 2 ||
                input.DimensionCount != 2 ||
                residual.Sizes[0] != input.Sizes[0] ||
                residual.Sizes[1] != input.Sizes[1] ||
                alphaCount != residual.Sizes[1] ||
                resCount != inCount)
            {
                return false;
            }

            int rows = checked((int)residual.Sizes[0]);
            int cols = checked((int)residual.Sizes[1]);
            CudaAllocator allocator = resStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            inStorage.EnsureDeviceCurrent();
            alphaStorage.EnsureDeviceCurrent();
            resStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchRMSNormResidualAddF32(inPtr, alphaPtr, resPtr, rows, cols, eps, allocator.Stream.Handle);
            resStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryNeoXRoPEFlatInPlace(Tensor data, Tensor cosTable, Tensor sinTable, int numHeads, int seqLen, int headDim, int ropeHalf)
        {
            if (!TryGetContiguousFloat(data, out CudaStorage dataStorage, out IntPtr dataPtr, out int dataCount) ||
                !TryGetContiguousFloat(cosTable, out CudaStorage cosStorage, out IntPtr cosPtr, out int cosCount) ||
                !TryGetContiguousFloat(sinTable, out CudaStorage sinStorage, out IntPtr sinPtr, out int sinCount) ||
                numHeads <= 0 ||
                seqLen <= 0 ||
                headDim <= 0 ||
                ropeHalf <= 0 ||
                ropeHalf * 2 > headDim ||
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
            kernels.LaunchNeoXRoPEFlatF32(
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

        public static bool TryFillNeoXRopeTablesDynamic(
            Tensor localCos,
            Tensor localSin,
            Tensor localFrequencies,
            Tensor globalCos,
            Tensor globalSin,
            Tensor globalFrequencies)
        {
            if (!TryGetContiguousFloat(localCos, out CudaStorage localCosStorage, out IntPtr localCosPtr, out int localCosCount) ||
                !TryGetContiguousFloat(localSin, out CudaStorage localSinStorage, out IntPtr localSinPtr, out int localSinCount) ||
                !TryGetContiguousFloat(localFrequencies, out CudaStorage localFreqStorage, out IntPtr localFreqPtr, out int localFreqCount) ||
                !TryGetContiguousFloat(globalCos, out CudaStorage globalCosStorage, out IntPtr globalCosPtr, out int globalCosCount) ||
                !TryGetContiguousFloat(globalSin, out CudaStorage globalSinStorage, out IntPtr globalSinPtr, out int globalSinCount) ||
                !TryGetContiguousFloat(globalFrequencies, out CudaStorage globalFreqStorage, out IntPtr globalFreqPtr, out int globalFreqCount) ||
                localCosCount <= 0 ||
                localSinCount != localCosCount ||
                localFreqCount != localCosCount ||
                globalCosCount <= 0 ||
                globalSinCount != globalCosCount ||
                globalFreqCount != globalCosCount)
            {
                return false;
            }

            CudaAllocator allocator = localCosStorage.AllocatorImpl;
            if (!ReferenceEquals(localSinStorage.AllocatorImpl, allocator) ||
                !ReferenceEquals(localFreqStorage.AllocatorImpl, allocator) ||
                !ReferenceEquals(globalCosStorage.AllocatorImpl, allocator) ||
                !ReferenceEquals(globalSinStorage.AllocatorImpl, allocator) ||
                !ReferenceEquals(globalFreqStorage.AllocatorImpl, allocator))
            {
                return false;
            }
            IntPtr dyn = CudaDecodeDynParams.GetActiveDevicePtr(allocator);
            if (dyn == IntPtr.Zero)
                return false;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            localCosStorage.EnsureDeviceCurrent();
            localSinStorage.EnsureDeviceCurrent();
            localFreqStorage.EnsureDeviceCurrent();
            globalCosStorage.EnsureDeviceCurrent();
            globalSinStorage.EnsureDeviceCurrent();
            globalFreqStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchFillNeoXRopeTablesDynamicF32(
                localCosPtr,
                localSinPtr,
                localFreqPtr,
                localCosCount,
                globalCosPtr,
                globalSinPtr,
                globalFreqPtr,
                globalCosCount,
                dyn,
                allocator.Stream.Handle);
            localCosStorage.MarkDeviceModified();
            localSinStorage.MarkDeviceModified();
            globalCosStorage.MarkDeviceModified();
            globalSinStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryQKNormRopeNeox(
            Tensor data,
            Tensor alpha,
            Tensor positions,
            int rows,
            int cols,
            int ropeHalf,
            float eps,
            float ropeBase,
            float ropeFreqScale)
        {
            if (!TryGetContiguousFloat(data, out CudaStorage dataStorage, out IntPtr dataPtr, out int dataCount) ||
                !TryGetContiguousFloat(alpha, out CudaStorage alphaStorage, out IntPtr alphaPtr, out int alphaCount) ||
                !TryGetContiguous(positions, out CudaStorage posStorage, out IntPtr posPtr, out long posCount) ||
                rows <= 0 ||
                cols <= 0 ||
                ropeHalf <= 0 ||
                ropeHalf * 2 > cols ||
                dataCount != checked(rows * cols) ||
                alphaCount != cols ||
                posCount != rows ||
                positions.ElementType != DType.Int32)
            {
                return false;
            }

            CudaAllocator allocator = dataStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            dataStorage.EnsureDeviceCurrent();
            alphaStorage.EnsureDeviceCurrent();
            posStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchQKNormRopeNeoxF32(
                dataPtr,
                alphaPtr,
                posPtr,
                rows,
                cols,
                ropeHalf,
                eps,
                ropeBase,
                ropeFreqScale,
                allocator.Stream.Handle);
            dataStorage.MarkDeviceModified();
            return true;
        }

        /// <summary>
        /// Refreshes the two cached RoPE position tensors from the decode-graph
        /// dyn block (dyn[3]) on device. Launched INSIDE a decode-graph capture
        /// so every replay RoPEs with the current token's position.
        /// </summary>
        public static bool TryFillRopePositions(Tensor posQ, Tensor posK, IntPtr dynParams)
        {
            if (dynParams == IntPtr.Zero ||
                !TryGetContiguous(posQ, out CudaStorage qStorage, out IntPtr qPtr, out long qCount) ||
                !TryGetContiguous(posK, out CudaStorage kStorage, out IntPtr kPtr, out long kCount) ||
                posQ.ElementType != DType.Int32 ||
                posK.ElementType != DType.Int32 ||
                qCount <= 0 || kCount <= 0)
            {
                return false;
            }

            CudaAllocator allocator = qStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            qStorage.EnsureDeviceCurrent();
            kStorage.EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchFillRopePositionsI32(
                qPtr, checked((int)qCount), kPtr, checked((int)kCount), dynParams, allocator.Stream.Handle);
            qStorage.MarkDeviceModified();
            kStorage.MarkDeviceModified();
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
            // The packed kernel stages the whole per-head state in dynamic shared
            // memory; bail (host fallback) for geometries beyond the sm cap.
            if (headKDim > 0 && headVDim > 0 &&
                (2L * headKDim + headVDim + (long)headVDim * headKDim) * sizeof(float) > CudaKernels.GdnPackedMaxSharedBytes)
            {
                return false;
            }
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
            // Decode-graph capture: the single-token step re-reads the conv ring
            // write index from the dyn block on every replay (it advances mod
            // convDim each token).
            IntPtr dyn = seqLen == 1
                ? CudaDecodeDynParams.GetActiveDevicePtr(allocator)
                : IntPtr.Zero;
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
                allocator.Stream.Handle,
                dyn);
            resultStorage.MarkDeviceModified();
            convStateStorage.MarkDeviceModified();
            ssmStateStorage.MarkDeviceModified();
            return true;
        }

        public static bool TryQwen35GdnFused(
            Tensor result,
            Tensor qkv, Tensor z, Tensor beta, Tensor alpha,
            Tensor convState, Tensor ssmState, Tensor convWeight,
            Tensor dtBias, Tensor aLog, Tensor ssmNorm,
            int seqLen, int qkvDim, int zDim, int qkDim, int vDim,
            int numKHeads, int numVHeads, int headKDim, int headVDim,
            int convKernel, int convWriteIdx, float eps)
        {
            int convDim = convKernel - 1;
            if (!TryGetContiguousFloat(result, out CudaStorage resultStorage, out IntPtr resultPtr, out int resultCount) ||
                !TryGetContiguousFloat(qkv, out CudaStorage qkvStorage, out IntPtr qkvPtr, out int qkvCount) ||
                !TryGetContiguousFloat(z, out CudaStorage zStorage, out IntPtr zPtr, out int zCount) ||
                !TryGetContiguousFloat(beta, out CudaStorage betaStorage, out IntPtr betaPtr, out int betaCount) ||
                !TryGetContiguousFloat(alpha, out CudaStorage alphaStorage, out IntPtr alphaPtr, out int alphaCount) ||
                !TryGetContiguousFloat(convState, out CudaStorage convStateStorage, out IntPtr convStatePtr, out int convStateCount) ||
                !TryGetContiguousFloat(ssmState, out CudaStorage ssmStateStorage, out IntPtr ssmStatePtr, out int ssmStateCount) ||
                !TryGetContiguousFloat(convWeight, out CudaStorage convWeightStorage, out IntPtr convWeightPtr, out int convWeightCount) ||
                !TryGetContiguousFloat(dtBias, out CudaStorage dtBiasStorage, out IntPtr dtBiasPtr, out int dtBiasCount) ||
                !TryGetContiguousFloat(aLog, out CudaStorage aLogStorage, out IntPtr aLogPtr, out int aLogCount) ||
                !TryGetContiguousFloat(ssmNorm, out CudaStorage ssmNormStorage, out IntPtr ssmNormPtr, out int ssmNormCount) ||
                seqLen <= 0 || qkvDim <= 0 || zDim <= 0 || qkDim <= 0 || vDim <= 0 ||
                numKHeads <= 0 || numVHeads <= 0 || headKDim <= 0 || headVDim <= 0 ||
                convKernel <= 0 || convDim <= 0 || convWriteIdx < 0 || convWriteIdx >= convDim ||
                eps <= 0.0f ||
                qkDim != checked(numKHeads * headKDim) ||
                vDim != checked(numVHeads * headVDim) ||
                qkvDim != checked(2 * qkDim + vDim) ||
                result.DimensionCount != 2 || result.Sizes[0] != seqLen || result.Sizes[1] != vDim ||
                resultCount != checked(seqLen * vDim) ||
                qkvCount != checked(seqLen * qkvDim) ||
                zCount != checked(seqLen * zDim) ||
                betaCount != checked(seqLen * numVHeads) ||
                alphaCount != checked(seqLen * numVHeads) ||
                convStateCount != checked(convDim * qkvDim) ||
                ssmStateCount != checked(numVHeads * headVDim * headKDim) ||
                convWeightCount != checked(qkvDim * convKernel) ||
                dtBiasCount < numVHeads || aLogCount < numVHeads || ssmNormCount < headVDim)
            {
                return false;
            }

            CudaAllocator allocator = resultStorage.AllocatorImpl;
            if (!TryGetKernels(allocator, out CudaKernels kernels))
                return false;

            qkvStorage.EnsureDeviceCurrent(); zStorage.EnsureDeviceCurrent();
            betaStorage.EnsureDeviceCurrent(); alphaStorage.EnsureDeviceCurrent();
            convStateStorage.EnsureDeviceCurrent(); ssmStateStorage.EnsureDeviceCurrent();
            convWeightStorage.EnsureDeviceCurrent(); dtBiasStorage.EnsureDeviceCurrent();
            aLogStorage.EnsureDeviceCurrent(); ssmNormStorage.EnsureDeviceCurrent();

            allocator.Context.MakeCurrent();
            kernels.LaunchQwen35GdnFusedF32(
                qkvPtr, zPtr, betaPtr, alphaPtr,
                convStatePtr, ssmStatePtr, convWeightPtr,
                dtBiasPtr, aLogPtr, ssmNormPtr, resultPtr,
                seqLen, qkvDim, zDim, qkDim, vDim,
                numKHeads, numVHeads, headKDim, headVDim,
                convKernel, convWriteIdx, eps,
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

        // Accepts a float32 tensor that collapses to a 2D row-major view whose rows
        // may be padded: sizes [.., rows, cols] with unit stride on the last dim and
        // a fixed row pitch. Contiguous tensors qualify with rowStride == cols.
        internal static bool TryGetRowStridedFloat(
            Tensor tensor, out CudaStorage storage, out IntPtr ptr,
            out int rows, out int cols, out long rowStride)
        {
            storage = tensor?.Storage as CudaStorage;
            rows = 0;
            cols = 0;
            rowStride = 0;
            ptr = IntPtr.Zero;
            if (storage == null || tensor.ElementType != DType.Float32 || tensor.DimensionCount < 1)
                return false;

            int dims = tensor.DimensionCount;
            ReadOnlySpan<long> sizes = tensor.Sizes;
            ReadOnlySpan<long> strides = tensor.Strides;
            if (strides[dims - 1] != 1 || sizes[dims - 1] > int.MaxValue)
                return false;

            cols = (int)sizes[dims - 1];
            if (cols <= 0)
                return false;

            long rowCount = 1;
            long pitch = cols;
            for (int d = dims - 2; d >= 0; d--)
            {
                if (sizes[d] == 1)
                    continue;
                if (d == dims - 2)
                {
                    pitch = strides[d];
                    if (pitch < cols)
                        return false;
                }
                else if (strides[d] != checked(pitch * rowCount))
                {
                    return false; // outer dims must tile the row-strided plane densely
                }
                rowCount = checked(rowCount * sizes[d]);
            }

            if (rowCount > int.MaxValue || checked(rowCount * cols) > int.MaxValue)
                return false;

            rows = (int)rowCount;
            rowStride = pitch;
            ptr = storage.DevicePtrAtElement(tensor.StorageOffset);
            return true;
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
